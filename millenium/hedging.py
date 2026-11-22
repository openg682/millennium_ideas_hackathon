"""Capital preservation: zero-cost collar alerts and the VaR circuit breaker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .ingestion import option_mid
from .models import Allocation, AssetSignal, HedgeAlert


def historical_var(returns: pd.DataFrame, weights: dict[str, float], confidence: float = config.VAR_CONFIDENCE) -> float:
    """1-day historical-simulation VaR as a positive fraction of NAV."""
    cols = [c for c in weights if c in returns.columns]
    if not cols:
        return 0.0
    r = returns[cols].dropna().tail(config.VAR_LOOKBACK)
    pnl = r.to_numpy() @ np.array([weights[c] for c in cols])
    return float(max(0.0, -np.quantile(pnl, 1 - confidence)))


def collar_strikes(chain: dict, direction: int) -> dict | None:
    """Buy a ~5% OTM protective option and sell the opposite wing whose premium best offsets it."""
    spot = chain["spot"]
    if direction >= 0:   # long exposure: buy put, sell call
        buy_side, sell_side, buy_target = chain["puts"], chain["calls"], spot * config.COLLAR_PUT_MONEYNESS
        sell_ok = sell_side["strike"] > spot
    else:                # short exposure: buy call, sell put
        buy_side, sell_side, buy_target = chain["calls"], chain["puts"], spot * (2 - config.COLLAR_PUT_MONEYNESS)
        sell_ok = sell_side["strike"] < spot
    if buy_side.empty or not sell_ok.any():
        return None
    buy = buy_side.iloc[(buy_side["strike"] - buy_target).abs().argsort()[:1]].iloc[0]
    buy_premium = option_mid(buy)
    candidates = sell_side[sell_ok].copy()
    candidates["mid"] = candidates.apply(option_mid, axis=1)
    candidates = candidates.dropna(subset=["mid"])
    if candidates.empty or np.isnan(buy_premium):
        return None
    sell = candidates.iloc[(candidates["mid"] - buy_premium).abs().argsort()[:1]].iloc[0]
    return {
        "expiry": chain["expiry"],
        "buy": "put" if direction >= 0 else "call",
        "buy_strike": float(buy["strike"]), "buy_premium": round(buy_premium, 2),
        "sell": "call" if direction >= 0 else "put",
        "sell_strike": float(sell["strike"]), "sell_premium": round(float(sell["mid"]), 2),
        "net_premium": round(float(sell["mid"]) - buy_premium, 2),
    }


def cross_hedge_stats(spot: pd.Series, hedge: pd.Series, window: int = 36) -> dict | None:
    """Minimum-variance hedge of `spot` with `hedge` from monthly log changes.

    beta = hedge ratio (hedge notional per unit of spot notional); R² = share of spot variance the
    hedge removes; basis vol = annualized volatility of what is left after hedging.
    """
    df = pd.concat({"spot": spot, "hedge": hedge}, axis=1).dropna()
    r = np.log(df).diff().dropna()
    if len(r) < 24:
        return None
    cov = r.cov()
    beta = cov.loc["spot", "hedge"] / cov.loc["hedge", "hedge"]
    corr = r["spot"].corr(r["hedge"])
    resid = r["spot"] - beta * r["hedge"]
    roll_cov = r["spot"].rolling(window).cov(r["hedge"])
    return {
        "beta": float(beta), "corr": float(corr), "r2": float(corr ** 2),
        "spot_vol": float(r["spot"].std() * np.sqrt(12)), "basis_vol": float(resid.std() * np.sqrt(12)),
        "start": r.index[0], "end": r.index[-1], "n_months": len(r),
        "ratio": float(df["spot"].iloc[-1] / df["hedge"].iloc[-1]),
        "rolling_beta": roll_cov / r["hedge"].rolling(window).var(),
        "rolling_r2": r["spot"].rolling(window).corr(r["hedge"]) ** 2,
        "levels": df,
    }


class HedgeManager:
    def __init__(self, assets=config.ASSETS):
        self.assets = assets

    def collar_alerts(self, signals: dict[str, AssetSignal], vol_metrics: dict[str, dict],
                      chains: dict[str, dict]) -> list[HedgeAlert]:
        alerts = []
        for key, sig in signals.items():
            vm = vol_metrics[key]
            if sig.c_event < config.COLLAR_CONFIDENCE or vm["iv_spike"] is None or vm["iv_spike"] <= config.IV_SPIKE_THRESHOLD:
                continue
            asset = self.assets[key]
            strikes = collar_strikes(chains[asset.option_ticker], sig.direction) if asset.option_ticker in chains else None
            via = f" via {asset.option_ticker} options" if asset.option_ticker != asset.instrument else ""
            if strikes:
                legs = (f"buy {strikes['expiry']} {strikes['buy_strike']:g} {strikes['buy']} / "
                        f"sell {strikes['sell_strike']:g} {strikes['sell']} (net {strikes['net_premium']:+.2f})")
            else:
                legs = "no listed chain on Yahoo: price the collar on the exchange (e.g. ICE OJ options)"
            alerts.append(HedgeAlert(
                kind="ZERO_COST_COLLAR", asset=key,
                message=(f"{asset.name}: C_event {sig.c_event:.2f} ≥ {config.COLLAR_CONFIDENCE} and IV "
                         f"{vm['iv_spike']:+.0%} vs realized ({vm['iv_source']}). Collar {asset.instrument}{via}: {legs}"),
                details={"c_event": sig.c_event, **vm, **(strikes or {})},
            ))
        return alerts

    def circuit_breaker(self, allocations: dict[str, Allocation], returns: pd.DataFrame
                        ) -> tuple[float, float, bool, HedgeAlert | None]:
        weights = {a.instrument: a.signed_weight for a in allocations.values()}
        var_pre = historical_var(returns, weights)
        if var_pre <= config.VAR_LIMIT:
            return var_pre, var_pre, False, None
        for a in allocations.values():
            a.signed_weight *= 1 - config.CIRCUIT_BREAKER_CUT
        var_post = historical_var(returns, {a.instrument: a.signed_weight for a in allocations.values()})
        alert = HedgeAlert(
            kind="CIRCUIT_BREAKER", asset=None,
            message=(f"Portfolio 1-day {config.VAR_CONFIDENCE:.0%} VaR {var_pre:.2%} > limit {config.VAR_LIMIT:.2%}: "
                     f"all allocations cut {config.CIRCUIT_BREAKER_CUT:.0%} (VaR now {var_post:.2%})"),
            details={"var_pre": var_pre, "var_post": var_post, "limit": config.VAR_LIMIT},
        )
        return var_pre, var_post, True, alert
