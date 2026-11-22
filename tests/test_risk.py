from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from millenium import config
from millenium.hedging import historical_var
from millenium.ingestion import add_derived_metrics
from millenium.signals import RiskEngine, severity
from millenium.sizing import PositionSizer, conviction_multiplier, inverse_vol_scales

TODAY = date(2026, 1, 15)


def calm_frame(**overrides_on_day1) -> pd.DataFrame:
    days = [TODAY + timedelta(days=d) for d in range(-config.PAST_DAYS, config.FORECAST_DAYS)]
    frame = pd.DataFrame({
        "temperature_2m_min": 15.0, "temperature_2m_max": 25.0, "precipitation_sum": 1.0,
        "wind_speed_10m_max": 10.0, "wind_gusts_10m_max": 20.0, "et0_fao_evapotranspiration": 1.0,
    }, index=days)
    for metric, value in overrides_on_day1.items():
        frame.loc[TODAY + timedelta(days=1), metric] = value
    return add_derived_metrics(frame)


def calm_world(**facility_overrides):
    daily = {k: calm_frame(**facility_overrides.get(k, {})) for k in config.FACILITIES}
    return daily, {k: TODAY for k in config.FACILITIES}


@pytest.mark.parametrize("value,expected", [(40, 0.0), (45, 0.0), (55, 0.25), (65, 0.5), (119, 1.0), (200, 1.0)])
def test_severity_ramp_above(value, expected):
    assert severity(value, 45, 65, 119, config.ABOVE) == pytest.approx(expected)


def test_severity_ramp_below_for_freezes():
    assert severity(2, 2, 0, -4, config.BELOW) == 0.0
    assert severity(0, 2, 0, -4, config.BELOW) == pytest.approx(0.5)
    assert severity(-4, 2, 0, -4, config.BELOW) == pytest.approx(1.0)


def test_calm_weather_is_neutral():
    signals = RiskEngine().evaluate(*calm_world())
    assert all(s.c_event == 0 and s.bias == "NEUTRAL" for s in signals.values())


def test_hurricane_drives_ammonia_long_with_lead_decay():
    signals = RiskEngine().evaluate(*calm_world(donaldsonville={"wind_speed_10m_max": 130}))
    nh3 = signals["NH3"]
    assert nh3.bias == "LONG"
    assert nh3.c_event == pytest.approx(config.LEAD_DECAY)  # full severity, one day out


def test_freeze_off_alone_is_short_ammonia_producer():
    signals = RiskEngine().evaluate(*calm_world(haynesville={"temperature_2m_min": -15}))
    assert signals["NH3"].bias == "SHORT"


def test_sizing_formula():
    assert conviction_multiplier(0) == 0.5 and conviction_multiplier(1) == 1.75
    scales = inverse_vol_scales({"NH3": 0.2, "LI": 0.4, "FCOJ": 0.4})
    assert scales["NH3"] == pytest.approx(2 * scales["LI"])
    assert np.mean(list(scales.values())) == pytest.approx(1.0)

    signals = RiskEngine().evaluate(*calm_world())
    allocs = PositionSizer().size(signals, {k: 0.3 for k in config.ASSETS})
    assert allocs["NH3"].effective_size == pytest.approx(config.ASSETS["NH3"].base_weight * 1.0 * 0.5)


def test_historical_var_scales_with_weight():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"A": rng.normal(0, 0.02, 500)})
    v1, v2 = historical_var(rets, {"A": 1.0}), historical_var(rets, {"A": 0.5})
    assert v1 == pytest.approx(2 * v2)
    assert 0.025 < v1 < 0.045  # ~1.645 sigma


# --- advisor, ensemble probabilities, backtest -----------------------------------------------------
from millenium.advisor import action_for, confidence_level  # noqa: E402
from millenium.backtest import Backtester  # noqa: E402
from millenium.ingestion import WeatherSnapshot  # noqa: E402
from millenium.probability import EnsembleAnalyzer, member_frames  # noqa: E402


def test_action_and_confidence_levels():
    assert action_for(0.33, 0.20) == "REDUCE"
    assert action_for(0.33, 0.50) == "INCREASE"
    assert action_for(0.33, 0.34) == "HOLD"
    assert action_for(0.33, -0.10) == "FLIP TO SHORT"
    assert [confidence_level(x) for x in (0.95, 0.7, 0.3)] == ["High", "Medium", "Low"]


def test_ensemble_breach_probability_counts_members():
    daily, today = calm_world()
    forecast_days = [TODAY + timedelta(days=d) for d in range(config.FORECAST_DAYS)]
    # 3 of 4 members bring a hard freeze to Florida on day +2
    members = pd.DataFrame({f"m{i}": 15.0 for i in range(4)}, index=forecast_days)
    members.loc[TODAY + timedelta(days=2), ["m0", "m1", "m2"]] = -3.0
    snap = WeatherSnapshot(daily=daily, current={}, today=today,
                           ensemble={"florida": {"temperature_2m_min": members}})
    assets, triggers = EnsembleAnalyzer().analyze(member_frames(snap), today)
    freeze = next(t for t in triggers if t.trigger == "Florida hard freeze")
    assert freeze.p_breach == pytest.approx(0.75)
    assert freeze.p_by_day[TODAY + timedelta(days=2)] == pytest.approx(0.75)
    assert assets["FCOJ"].p_any == pytest.approx(0.75)
    assert assets["NH3"].p_any == 0.0


def test_backtest_has_no_look_ahead():
    """A freeze on day D can only change the weight held over returns after day D's close."""
    days = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({a.instrument: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
                           for a in config.ASSETS.values()}, index=days)
    cal = [d.date() for d in pd.date_range("2024-11-01", "2025-07-01")]
    base = pd.DataFrame({"temperature_2m_min": 15.0, "temperature_2m_max": 25.0, "precipitation_sum": 1.0,
                         "wind_speed_10m_max": 10.0, "wind_gusts_10m_max": 20.0, "et0_fao_evapotranspiration": 1.0},
                        index=cal)
    event_day = days[80].date()
    shocked = base.copy()
    shocked.loc[event_day, "temperature_2m_min"] = -6.0
    calm = {k: add_derived_metrics(base) for k in config.FACILITIES}
    hit = dict(calm, florida=add_derived_metrics(shocked))
    w_calm = Backtester().run(calm, prices).ai_weights["FCOJ"]
    w_hit = Backtester().run(hit, prices).ai_weights["FCOJ"]
    changed = w_hit.index[(w_hit - w_calm).abs() > 1e-12]
    assert len(changed) > 0 and changed.min() > pd.Timestamp(event_day)


# --- sorghum: seasonal triggers and the corn cross-hedge -------------------------------------------
from millenium.hedging import cross_hedge_stats  # noqa: E402
from millenium.signals import score_trigger  # noqa: E402


def test_out_of_season_trigger_scores_zero():
    heat = next(t for t in config.ASSETS["SORG"].triggers if t.name == "Kansas flowering heat")
    frame = calm_frame(temperature_2m_max=41.0)   # TODAY is mid-January: heat is out of season
    assert score_trigger(heat, frame, TODAY).severity == 0.0
    july = TODAY.replace(month=7)
    frame.index = [d.replace(year=2026) if d.month == 1 else d for d in frame.index]
    shifted = frame.set_axis([july + (d - TODAY) for d in frame.index])
    assert score_trigger(heat, shifted, july).severity > 0.5


def test_cross_hedge_recovers_beta():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2005-01-01", periods=240, freq="MS")
    corn = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.06, 240))), index=idx)
    noise = rng.normal(0, 0.02, 240)
    sorghum = pd.Series(np.exp(0.8 * np.log(corn).to_numpy() + np.cumsum(noise)), index=idx)
    stats = cross_hedge_stats(sorghum, corn)
    assert stats["beta"] == pytest.approx(0.8, abs=0.05)
    assert stats["r2"] > 0.8


# --- supply chain, flows and the end-of-day note ---------------------------------------------------
from millenium.eod import rule_based_note  # noqa: E402
from millenium.supplychain import (BREACH, CRITICAL, NORMAL, UNKNOWN, WATCH, activity_drop,  # noqa: E402
                                   ratio_level, stage_level)


def test_port_activity_drop_levels():
    calls = pd.Series([10.0] * 90 + [4.0] * 14, index=pd.date_range("2026-01-01", periods=104))
    ratio, recent, base = activity_drop(calls)
    assert ratio == pytest.approx(0.4) and recent == 4.0 and base == 10.0
    assert ratio_level(ratio, 0.7, 0.5) == BREACH
    assert ratio_level(0.65, 0.7, 0.5) == WATCH
    assert ratio_level(1.3, 1.25, 1.6, higher_is_worse=True) == WATCH


def test_river_stage_levels():
    low, high = (0, -3, -8), (25, 30, 38)
    assert stage_level(5, low, high) == NORMAL
    assert stage_level(-4, low, high) == BREACH
    assert stage_level(40, low, high) == CRITICAL
    assert stage_level(float("nan"), low, high) == UNKNOWN


def test_flow_marks_missing_feeds_unknown_but_keeps_weather():
    from millenium.models import AssetSignal, PortfolioReport
    from millenium.supplychain import evaluate_flow
    signals = RiskEngine().evaluate(*calm_world(florida={"temperature_2m_min": -5.0}))
    report = PortfolioReport(as_of="", scenario="Live", signals=signals, allocations={}, alerts=[], var_pre=0, var_post=0,
                             circuit_breaker=False, vol_metrics={})
    stages = {sr.stage.key: sr for sr in evaluate_flow("FCOJ", report, None)}
    assert stages["fl"].level >= BREACH                      # freeze trigger breached
    assert stages["plant"].level == NORMAL                   # no checks attached
    assert all(i.level == UNKNOWN for i in stages["port"].items if i.kind == "feed")


def test_rule_based_note_sorts_worked_and_failed():
    facts = {"assets": [
        {"asset": "Lithium", "instrument": "LIT", "return": 0.03, "pm_weight": 0.25, "ai_weight": 0.40, "ai_minus_pm": 4500},
        {"asset": "FCOJ", "instrument": "OJ=F", "return": 0.05, "pm_weight": 0.25, "ai_weight": 0.06, "ai_minus_pm": -9500},
        {"asset": "Sorghum", "instrument": "ZC=F", "return": 0.0, "pm_weight": 0.25, "ai_weight": 0.18, "ai_minus_pm": 0}],
        "signal_scorecard": [], "risk_controls": {"breaker_days_last_5": 0}, "data_issues": [
            {"name": "GDELT DOC 2.0", "message": "rate-limited"}], "watch_tomorrow": [], "supply_chain_flags": []}
    note = rule_based_note(facts)
    assert note["worked"][0].startswith("Lithium")
    assert note["failed"][0].startswith("FCOJ") and any("GDELT" in x for x in note["failed"])
