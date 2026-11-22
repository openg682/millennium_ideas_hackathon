"""Orchestration: ingest -> score -> size -> hedge, with optional stress scenarios."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import config
from .hedging import HedgeManager, cross_hedge_stats
from .ingestion import (BLSClient, EnsembleClient, FeedGrainsClient, MarketDataClient, MarketSnapshot,
                        OpenMeteoClient, WeatherSnapshot, WorldBankPinkSheetClient, add_derived_metrics,
                        merge_history, realized_vol)
from .models import PortfolioReport
from .probability import EnsembleAnalyzer, member_frames
from .signals import RiskEngine
from .sizing import PositionSizer

# Absolute overrides applied to forecast days +1 and +2, for demoing the alert path.
SCENARIOS: dict[str, dict] = {
    "Live": {"weather": {}, "iv_multiplier": 1.0},
    "Gulf hurricane landfall (Ida-like)": {
        "weather": {"donaldsonville": {"wind_speed_10m_max": 125, "wind_gusts_10m_max": 190, "precipitation_sum": 180}},
        "iv_multiplier": 1.8,
    },
    "Polar vortex freeze-off (Uri-like)": {
        "weather": {"haynesville": {"temperature_2m_min": -14}, "donaldsonville": {"temperature_2m_min": -7}},
        "iv_multiplier": 1.6,
    },
    "Florida hard freeze": {"weather": {"florida": {"temperature_2m_min": -4.5}}, "iv_multiplier": 1.7},
    "Atacama flash flood": {"weather": {"atacama": {"precipitation_sum": 38}}, "iv_multiplier": 1.7},
}
SCENARIO_DAYS = (1, 2)
KG_PER_ORANGE_BOX = 40.82   # a 90 lb field box
IV_PROXY_WINDOW = 10


class Pipeline:
    def __init__(self, weather_client: OpenMeteoClient | None = None,
                 market_client: MarketDataClient | None = None,
                 ensemble_client: EnsembleClient | None = None, assets=config.ASSETS):
        self.assets = assets
        self.weather_client = weather_client or OpenMeteoClient()
        self.market_client = market_client or MarketDataClient()
        self.ensemble_client = ensemble_client or EnsembleClient()
        self.engine = RiskEngine(assets)
        self.analyzer = EnsembleAnalyzer(assets)
        self.sizer = PositionSizer(assets)
        self.hedger = HedgeManager(assets)

    @property
    def facilities(self):
        return list(dict.fromkeys(config.FACILITIES[k] for a in self.assets.values() for k in a.facilities))

    def fetch(self) -> tuple[WeatherSnapshot, MarketSnapshot]:
        facilities = self.facilities
        anomaly_sites = {t.facility for a in self.assets.values() for t in a.triggers if t.metric.endswith("_anom_30d")}
        climatology = self.weather_client.fetch_climatology([f for f in facilities if f.key in anomaly_sites])
        weather = self.weather_client.fetch(facilities, climatology)
        weather.ensemble, weather.ensemble_models = self.ensemble_client.fetch(facilities)
        tickers = list(dict.fromkeys(t for a in self.assets.values() for t in (a.instrument, *a.context_tickers)))
        options = [a.option_ticker for a in self.assets.values() if a.option_ticker]
        return weather, self.market_client.fetch(tickers, options)

    def fetch_cross_hedges(self) -> dict[str, dict]:
        """Physical-vs-futures hedge statistics: sorghum vs CBOT corn, oranges vs ICE FCOJ.

        In each `series` dict the first entry is the primary estimate used by the calculator and advisor.
        """
        out = {}
        try:
            grains = FeedGrainsClient().fetch()
            corn = self.market_client.monthly_average("ZC=F", scale=0.01)          # cents/bu -> $/bu
            out["SORG"] = cross_hedge_bundle(
                title="Sorghum ↔ CBOT corn", physical="sorghum", hedge="ZC=F", hedge_series=corn,
                series={"Sorghum, Central Kansas cash (USDA ERS)": grains["sorghum_ks_cash"],
                        "Sorghum, US farm price (USDA ERS, 2000–)": grains["sorghum_farm"].loc["2000":]},
                unit_price_scale=1.0, qty_unit="bushels", qty_default=1_000_000, contract_size=5_000,
                contract_unit="bu")
        except Exception as exc:  # each hedge stands alone; the dashboard says which one is missing
            out["SORG"] = {"error": str(exc)}
        try:
            oranges = WorldBankPinkSheetClient().fetch(("Orange",))["Orange"]       # $/kg
            navel = BLSClient().fetch("APU0000711311", 2006)                         # $/lb, US retail
            oj = self.market_client.monthly_average("OJ=F", scale=0.01)              # cents/lb -> $/lb solids
            out["FCOJ"] = cross_hedge_bundle(
                title="Oranges ↔ ICE FCOJ", physical="oranges", hedge="OJ=F", hedge_series=oj,
                series={"Oranges, World Bank import price (2000–)": oranges.loc["2000":],
                        "Navel oranges, US retail (BLS, 2006–)": navel},
                unit_price_scale=KG_PER_ORANGE_BOX, qty_unit="boxes (90 lb)", qty_default=250_000,
                contract_size=15_000, contract_unit="lb orange solids")
        except Exception as exc:
            out["FCOJ"] = {"error": str(exc)}
        return out

    def fetch_weather_history(self, weather: WeatherSnapshot, start: date) -> dict:
        """Observed daily weather from `start` to today, for backtesting."""
        history = self.weather_client.fetch_history(self.facilities, start, min(weather.today.values()))
        return merge_history(history, weather)

    def run(self, scenario: str = "Live", base_weights: dict[str, float] | None = None) -> PortfolioReport:
        weather, market = self.fetch()
        return self.evaluate(weather, market, scenario, base_weights)

    def evaluate(self, weather: WeatherSnapshot, market: MarketSnapshot, scenario: str = "Live",
                 base_weights: dict[str, float] | None = None) -> PortfolioReport:
        spec = SCENARIOS[scenario]
        daily = apply_weather_scenario(weather.daily, weather.today, spec["weather"], weather.climatology)
        signals = self.engine.evaluate(daily, weather.today)
        worlds = [apply_weather_scenario(w, weather.today, spec["weather"], weather.climatology)
                  for w in member_frames(weather)]
        asset_probs, trigger_probs = self.analyzer.analyze(worlds, weather.today)

        vol_metrics = {k: self._vol_metrics(a, market, spec["iv_multiplier"]) for k, a in self.assets.items()}
        allocations = self.sizer.size(signals, {k: vm["rv14"] for k, vm in vol_metrics.items()}, base_weights)

        alerts = self.hedger.collar_alerts(signals, vol_metrics, market.option_chains)
        returns = market.prices.pct_change(fill_method=None)
        var_pre, var_post, tripped, breaker = self.hedger.circuit_breaker(allocations, returns)
        if breaker:
            alerts.append(breaker)

        return PortfolioReport(
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"), scenario=scenario,
            signals=signals, allocations=allocations, alerts=alerts,
            var_pre=var_pre, var_post=var_post, circuit_breaker=tripped, vol_metrics=vol_metrics,
            asset_probabilities=asset_probs, trigger_probabilities=trigger_probs,
            ensemble_models=weather.ensemble_models,
        )

    @staticmethod
    def _vol_metrics(asset, market: MarketSnapshot, iv_multiplier: float) -> dict:
        px = market.prices[asset.instrument]
        rv14, rv30 = realized_vol(px, config.VOL_WINDOW), realized_vol(px, 30)
        iv = market.implied_vol.get(asset.option_ticker) if asset.option_ticker else None
        source = "ATM options"
        if iv is None:
            iv, source = realized_vol(px, IV_PROXY_WINDOW), f"realized {IV_PROXY_WINDOW}d proxy"
        iv = iv * iv_multiplier if iv is not None and np.isfinite(iv) else None
        spike = iv / rv30 - 1 if iv and rv30 and np.isfinite(rv30) else None
        return {"rv14": rv14, "rv30": rv30, "iv": iv, "iv_source": source, "iv_spike": spike}


def apply_weather_scenario(daily: dict, today: dict, overrides: dict, climatology: dict) -> dict:
    if not overrides:
        return daily
    daily = dict(daily)
    for fac, values in overrides.items():
        frame = daily[fac].copy()
        days = [today[fac] + timedelta(days=d) for d in SCENARIO_DAYS]
        for metric, value in values.items():
            frame.loc[frame.index.isin(days), metric] = value
        daily[fac] = add_derived_metrics(frame, climatology.get(fac))
    return daily


def cross_hedge_bundle(title: str, physical: str, hedge: str, hedge_series, series: dict, unit_price_scale: float,
                       qty_unit: str, qty_default: int, contract_size: int, contract_unit: str) -> dict:
    stats = {label: cross_hedge_stats(s, hedge_series) for label, s in series.items()}
    stats = {k: v for k, v in stats.items() if v is not None}
    if not stats:
        raise ValueError(f"not enough overlapping history for {title}")
    primary = next(iter(series.values())).dropna()
    return {"title": title, "physical": physical, "hedge": hedge, "hedge_series": hedge_series, "series": series,
            "stats": stats, "unit_price": float(primary.iloc[-1]) * unit_price_scale,
            "unit_price_as_of": primary.index[-1], "qty_unit": qty_unit, "qty_default": qty_default,
            "contract_size": contract_size, "contract_unit": contract_unit}
