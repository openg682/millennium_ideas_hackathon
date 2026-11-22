"""Ensemble probabilities: run every forecast member through the same risk engine.

P(breach) for a trigger is the share of members whose weather crosses the trigger threshold at least
once in the window. Running members through the engine also gives a C_event distribution per asset,
which the advisor turns into a confidence level (how many members lead to the same action).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import config
from .config import ABOVE, Trigger
from .ingestion import WeatherSnapshot, add_derived_metrics, date_window
from .signals import RiskEngine, in_season


@dataclass
class TriggerProbability:
    asset: str
    trigger: str
    facility: str
    impact: str
    unit: str
    threshold: float
    direction: str
    p_breach: float                       # share of members breaching at least once in the window
    p_by_day: dict[date, float]           # share of members breaching on each day
    peak_p10: float                       # spread of each member's most adverse value in the window
    peak_p50: float
    peak_p90: float
    band: pd.DataFrame | None = None      # per-day p10 / p50 / p90 of the metric across members


@dataclass
class AssetProbability:
    asset: str
    n_members: int
    p_any: float                          # share of members breaching any of the asset's triggers
    c_event_p10: float
    c_event_p50: float
    c_event_p90: float
    member_c: list[float] = field(default_factory=list)
    member_direction: list[int] = field(default_factory=list)


def member_frames(weather: WeatherSnapshot) -> list[dict[str, pd.DataFrame]]:
    """One full daily dataset per ensemble member: shared observed past, member-specific forecast."""
    names = sorted({c for metrics in weather.ensemble.values() for f in metrics.values() for c in f.columns})
    worlds = []
    for name in names:
        world = {}
        for fac, frame in weather.daily.items():
            frame = frame.copy()
            for metric, members in weather.ensemble.get(fac, {}).items():
                if name not in members:
                    continue
                path = members[name].dropna()
                path = path[path.index >= weather.today[fac]]
                frame.loc[frame.index.isin(path.index), metric] = path.reindex(frame.index).dropna()
            world[fac] = add_derived_metrics(frame, weather.climatology.get(fac))
        worlds.append(world)
    return worlds


def breach_path(trigger: Trigger, frame: pd.DataFrame, today: date) -> pd.Series:
    start, end = date_window(today)
    values = frame.loc[(frame.index >= start) & (frame.index <= end), trigger.metric]
    hit = values > trigger.threshold if trigger.direction == ABOVE else values < trigger.threshold
    return hit & np.array([in_season(trigger, d) for d in values.index], dtype=bool)


class EnsembleAnalyzer:
    def __init__(self, assets=config.ASSETS):
        self.assets = assets
        self.engine = RiskEngine(assets)

    def analyze(self, worlds: list[dict[str, pd.DataFrame]], today: dict[str, date]
                ) -> tuple[dict[str, AssetProbability], list[TriggerProbability]]:
        if not worlds:
            return {}, []
        member_signals = [self.engine.evaluate(w, today) for w in worlds]
        assets, triggers = {}, []
        for key, asset in self.assets.items():
            any_breach = np.zeros(len(worlds), dtype=bool)
            for trig in asset.triggers:
                paths = pd.DataFrame({i: breach_path(trig, w[trig.facility], today[trig.facility])
                                      for i, w in enumerate(worlds)})
                hit = paths.any(axis=0).to_numpy()
                any_breach |= hit
                start, end = date_window(today[trig.facility])
                values = pd.DataFrame({
                    i: w[trig.facility].loc[(w[trig.facility].index >= start) & (w[trig.facility].index <= end), trig.metric]
                    for i, w in enumerate(worlds)
                })
                peaks = (values.max(axis=0) if trig.direction == ABOVE else values.min(axis=0)).to_numpy()
                band = values.quantile([0.1, 0.5, 0.9], axis=1).T.set_axis(["p10", "p50", "p90"], axis=1)
                triggers.append(TriggerProbability(
                    asset=key, trigger=trig.name, facility=trig.facility, impact=trig.impact, unit=trig.unit,
                    threshold=trig.threshold, direction=trig.direction, p_breach=float(hit.mean()),
                    p_by_day={d: float(v) for d, v in paths.mean(axis=1).items()},
                    peak_p10=float(pd.Series(peaks).quantile(0.1)), peak_p50=float(pd.Series(peaks).quantile(0.5)),
                    peak_p90=float(pd.Series(peaks).quantile(0.9)), band=band,
                ))
            cs = [s[key].c_event for s in member_signals]
            assets[key] = AssetProbability(
                asset=key, n_members=len(worlds), p_any=float(any_breach.mean()),
                c_event_p10=float(np.percentile(cs, 10)), c_event_p50=float(np.percentile(cs, 50)),
                c_event_p90=float(np.percentile(cs, 90)),
                member_c=cs, member_direction=[s[key].direction for s in member_signals],
            )
        return assets, triggers
