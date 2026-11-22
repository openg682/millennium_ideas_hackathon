"""Risk engine: weather -> per-trigger severity -> Event Confidence Score and directional bias."""
from __future__ import annotations

import math

import pandas as pd

from . import config
from .config import ABOVE, Asset, Trigger
from .ingestion import date_window
from .models import AssetSignal, TriggerReading


def severity(value: float, watch: float, threshold: float, extreme: float, direction: str) -> float:
    """Piecewise-linear ramp: watch -> 0.0, threshold -> 0.5, extreme -> 1.0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    s = 1 if direction == ABOVE else -1
    v, w, t, e = s * value, s * watch, s * threshold, s * extreme
    if v <= w:
        return 0.0
    if v <= t:
        return 0.5 * (v - w) / (t - w)
    return min(1.0, 0.5 + 0.5 * (v - t) / (e - t))


def in_season(trigger: Trigger, day) -> bool:
    return trigger.months is None or day.month in trigger.months


def score_trigger(trigger: Trigger, daily: pd.DataFrame, today,
                  lookback_days: int | None = None, horizon_days: int | None = None) -> TriggerReading:
    """Peak lead-time-discounted severity over [today - lookback, today + horizon]."""
    start, end = date_window(today, lookback_days, horizon_days)
    window = daily.loc[(daily.index >= start) & (daily.index <= end), trigger.metric]
    best = (0.0, float("nan"), today)
    for day, value in window.items():
        lead = max(0, (day - today).days)
        if not in_season(trigger, day):
            continue
        s = severity(value, trigger.watch, trigger.threshold, trigger.extreme, trigger.direction)
        s *= config.LEAD_DECAY ** lead
        if s > best[0]:
            best = (s, float(value), day)
    # When nothing is breached, report the most adverse raw value for context.
    if best[0] == 0.0 and not window.dropna().empty:
        adverse = window.idxmax() if trigger.direction == ABOVE else window.idxmin()
        best = (0.0, float(window[adverse]), adverse)
    return TriggerReading(
        name=trigger.name, facility=trigger.facility, metric=trigger.metric, unit=trigger.unit,
        threshold=trigger.threshold, peak_value=best[1], peak_date=best[2],
        severity=round(best[0], 4), bias=trigger.bias, in_season=in_season(trigger, today),
    )


def combine(readings: list[TriggerReading], weights: list[float]) -> float:
    """Noisy-OR: independent stressors compound, but C_event never exceeds 1."""
    survive = 1.0
    for r, w in zip(readings, weights):
        survive *= 1.0 - min(1.0, w * r.severity)
    return 1.0 - survive


class RiskEngine:
    def __init__(self, assets: dict[str, Asset] = config.ASSETS):
        self.assets = assets

    def evaluate(self, daily: dict[str, pd.DataFrame], today: dict, **window) -> dict[str, AssetSignal]:
        """`window` (lookback_days / horizon_days) overrides the live forecast window, e.g. for backtests."""
        return {key: self.evaluate_asset(asset, daily, today, **window) for key, asset in self.assets.items()}

    def evaluate_asset(self, asset: Asset, daily: dict[str, pd.DataFrame], today: dict, **window) -> AssetSignal:
        readings = [score_trigger(t, daily[t.facility], today[t.facility], **window) for t in asset.triggers]
        c_event = combine(readings, [t.weight for t in asset.triggers])
        net = sum(r.bias * r.severity * t.weight for r, t in zip(readings, asset.triggers))
        if c_event < config.MIN_SIGNAL or net == 0:
            bias, direction = "NEUTRAL", asset.default_stance
        else:
            bias, direction = ("LONG", 1) if net > 0 else ("SHORT", -1)
        return AssetSignal(asset=asset.key, c_event=round(c_event, 4), bias=bias,
                           direction=direction, readings=readings)
