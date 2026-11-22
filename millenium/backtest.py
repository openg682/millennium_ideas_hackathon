"""Benchmark (PM weights held static) vs AI resizing, replayed over observed weather.

No look-ahead: the weight held over the close(t) -> close(t+1) return is computed from weather
observed up to the previous calendar day and prices up to close(t). Live mode scores forecasts;
the backtest can only score observed weather, so it understates lead time. Collars (which need
historical option chains) are not simulated; the VaR circuit breaker is.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .hedging import historical_var
from .signals import RiskEngine
from .sizing import PositionSizer

MIN_VAR_HISTORY = 60


@dataclass
class BacktestResult:
    returns: pd.DataFrame       # daily portfolio returns by strategy
    equity: pd.DataFrame        # growth of 1.0
    ai_weights: pd.DataFrame    # AI signed weights by asset key
    c_event: pd.DataFrame       # C_event by asset key
    breaker: pd.Series          # circuit breaker tripped on that day
    metrics: pd.DataFrame       # one row per strategy
    held: pd.DataFrame          # AI weights actually held over each day's return
    asset_returns: pd.DataFrame # daily instrument returns by asset key
    direction: pd.DataFrame     # +1 / -1 when the AI had a directional (non-NEUTRAL) view, else 0


class Backtester:
    def __init__(self, assets=config.ASSETS, lookback_days: int = 3, cost_bps: float = 5.0):
        self.assets = assets
        self.lookback_days = lookback_days   # covers weekends between trading days
        self.cost = cost_bps / 1e4
        self.engine = RiskEngine(assets)
        self.sizer = PositionSizer(assets)

    def run(self, daily: dict[str, pd.DataFrame], prices: pd.DataFrame,
            base_weights: dict[str, float] | None = None) -> BacktestResult:
        base = base_weights or {k: a.base_weight for k, a in self.assets.items()}
        inst = {k: a.instrument for k, a in self.assets.items()}
        px = prices[list(inst.values())].dropna().rename(columns={v: k for k, v in inst.items()})
        rets = px.pct_change().dropna()
        rv14 = np.log(px).diff().rolling(config.VOL_WINDOW).std() * np.sqrt(252)
        first_weather = max(f.index.min() for f in daily.values()) + timedelta(days=31)

        weights, conf, tripped, views = {}, {}, {}, {}
        for t in rv14.dropna().index:
            as_of = t.date() - timedelta(days=1)
            if as_of < first_weather:
                continue
            signals = self.engine.evaluate(daily, {k: as_of for k in daily},
                                           lookback_days=self.lookback_days, horizon_days=0)
            allocs = self.sizer.size(signals, rv14.loc[t].to_dict(), base)
            w = {k: a.signed_weight for k, a in allocs.items()}
            hist = rets.loc[:t].tail(config.VAR_LOOKBACK)
            trip = len(hist) >= MIN_VAR_HISTORY and historical_var(hist, w) > config.VAR_LIMIT
            if trip:
                w = {k: v * (1 - config.CIRCUIT_BREAKER_CUT) for k, v in w.items()}
            weights[t], conf[t], tripped[t] = w, {k: s.c_event for k, s in signals.items()}, trip
            views[t] = {k: (s.direction if s.bias != "NEUTRAL" else 0) for k, s in signals.items()}

        W = pd.DataFrame.from_dict(weights, orient="index")
        held = W.shift(1).reindex(rets.index).dropna()
        r = rets.loc[held.index]
        turnover = held.diff().abs().sum(axis=1).fillna(0)
        ai = (held * r).sum(axis=1) - self.cost * turnover
        bench = (r * pd.Series(base)).sum(axis=1)
        # Same signal timing, levered to the benchmark's realized vol: separates timing from exposure level.
        matched = ai * (bench.std() / ai.std()) if ai.std() > 0 else ai

        returns = pd.DataFrame({"Benchmark (PM static)": bench, "AI resized": ai, "AI resized, vol-matched": matched})
        gross = {
            "Benchmark (PM static)": pd.Series(sum(abs(v) for v in base.values()), index=held.index),
            "AI resized": held.abs().sum(axis=1),
            "AI resized, vol-matched": held.abs().sum(axis=1) * (bench.std() / ai.std() if ai.std() > 0 else 1),
        }
        metrics = pd.DataFrame([performance(returns[c], gross[c]) | {"Strategy": c} for c in returns]).set_index("Strategy")
        metrics.loc["AI resized", "Circuit breaker days"] = int(pd.Series(tripped).loc[held.index].sum())
        return BacktestResult(
            returns=returns, equity=(1 + returns).cumprod(), ai_weights=W.loc[held.index],
            c_event=pd.DataFrame.from_dict(conf, orient="index").loc[held.index],
            breaker=pd.Series(tripped).loc[held.index], metrics=metrics, held=held, asset_returns=r,
            direction=pd.DataFrame.from_dict(views, orient="index").loc[held.index],
        )


def signal_hit_rates(bt: BacktestResult, horizon: int = 5) -> pd.DataFrame:
    """For days the AI took a directional view, how often the next `horizon` days moved its way."""
    fwd = (1 + bt.asset_returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    rows = []
    for k in bt.direction:
        active = (bt.direction[k] != 0) & (bt.c_event[k] >= config.MIN_SIGNAL) & fwd[k].notna()
        moves = fwd.loc[active, k] * bt.direction.loc[active, k]
        rows.append({"asset": k, "signal_days": int(active.sum()),
                     "hit_rate": float((moves > 0).mean()) if active.any() else np.nan,
                     "avg_move_in_signal_direction": float(moves.mean()) if active.any() else np.nan})
    return pd.DataFrame(rows).set_index("asset")


def performance(r: pd.Series, gross: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(252)
    return {
        "Total return": eq.iloc[-1] - 1,
        "Ann. return": eq.iloc[-1] ** (252 / len(r)) - 1,
        "Ann. vol": vol,
        "Sharpe": r.mean() * 252 / vol if vol > 0 else np.nan,
        "Max drawdown": (eq / eq.cummax() - 1).min(),
        "Avg gross": gross.mean(),
    }
