"""AI advisor: turns the engine's output into per-asset actions with evidence-based justifications.

Every sentence is built from a number the engine actually used, so a recommendation can always be
traced back to a trigger reading, a volatility figure or a hedging rule.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from . import config
from .models import PortfolioReport, Recommendation, TriggerReading
from .sizing import conviction_multiplier

HOLD_BAND = 0.02   # weight changes smaller than this are not worth trading


def action_for(pm_weight: float, ai_weight: float) -> str:
    if pm_weight * ai_weight < 0:
        return "FLIP TO LONG" if ai_weight > 0 else "FLIP TO SHORT"
    if abs(ai_weight - pm_weight) < HOLD_BAND:
        return "HOLD"
    return "INCREASE" if abs(ai_weight) > abs(pm_weight) else "REDUCE"


def confidence_level(share: float) -> str:
    return next(label for label, floor in config.CONFIDENCE_LEVELS if share >= floor)


def describe_reading(r: TriggerReading, today: date, p_breach: float | None = None) -> str:
    when = "observed" if r.peak_date <= today else f"forecast {r.peak_date:%a %d %b}"
    state = "breach" if r.breached else "building"
    odds = f"; {p_breach:.0%} of ensemble members breach" if p_breach is not None else ""
    return (f"{r.name}: {r.peak_value:.1f} {r.unit} {when} vs {r.threshold:g} {r.unit} trigger "
            f"({state}, severity {r.severity:.2f}{odds}).")


class Advisor:
    def __init__(self, assets=config.ASSETS):
        self.assets = assets

    def recommend(self, report: PortfolioReport, pm_weights: dict[str, float], today: dict,
                  cross_hedges: dict[str, dict] | None = None,
                  supply_evidence: dict[str, list[str]] | None = None) -> list[Recommendation]:
        collars = {a.asset: a for a in report.alerts if a.kind == "ZERO_COST_COLLAR"}
        p_trigger = {(t.asset, t.trigger): t.p_breach for t in report.trigger_probabilities}
        recs = []
        for key, asset in self.assets.items():
            sig, alloc, vm = report.signals[key], report.allocations[key], report.vol_metrics[key]
            fac_today = today[asset.facilities[0]]
            drivers = sorted((r for r in sig.readings if r.severity > 0), key=lambda r: -r.severity)

            if drivers and drivers[0].breached:
                headline = f"{drivers[0].name} breach → {sig.bias} conviction"
            elif drivers:
                headline = f"Watch: {drivers[0].name.lower()} building, below trigger"
            else:
                headline = "No disruption signal: sized at the 0.5× conviction floor"

            reasons = [describe_reading(r, fac_today, p_trigger.get((key, r.name))) for r in drivers[:2]]
            if not drivers:
                reasons.append(f"All {len(sig.readings)} triggers quiet across the {config.FORECAST_DAYS}-day forecast window.")
            reasons.append(f"C_event {sig.c_event:.2f} → conviction {conviction_multiplier(sig.c_event):.2f}× "
                           f"(0.5 + 1.25·C); bias {sig.bias}.")
            cut = 1 - config.CIRCUIT_BREAKER_CUT if report.circuit_breaker else 1.0
            if sig.bias != "NEUTRAL":
                calm = alloc.base_weight * alloc.vol_scale * conviction_multiplier(0.0) * sig.direction * cut
                capped = abs(alloc.signed_weight) < abs(pm_weights[key])
                reasons.append(f"The event lifts the AI size from {calm:+.3f} (calm weather) to {alloc.signed_weight:+.3f}"
                               + (f"; still under your {pm_weights[key]:.3f} because the {alloc.vol_scale:.2f}× vol scale caps it."
                                  if capped else "."))
            reasons.append(f"14d realized vol {alloc.realized_vol_14d:.0%} → {alloc.vol_scale:.2f}× inverse-vol scale vs the basket.")
            if vm["iv_spike"] is not None and vm["iv_spike"] > config.IV_SPIKE_THRESHOLD:
                reasons.append(f"Implied vol {vm['iv']:.0%} ({vm['iv_source']}) is {vm['iv_spike']:+.0%} over 30d realized.")
            bundle = (cross_hedges or {}).get(key, {})
            hedge_stats = next(iter(bundle.get("stats", {}).items()), None)
            if hedge_stats:
                label, hs = hedge_stats
                lead = (f"Held in {asset.instrument} as a cross-hedge" if asset.cross_hedge
                        else f"Physical {bundle['physical']} hedge with {bundle['hedge']}")
                reasons.append(f"{lead} ({label}): β {hs['beta']:.2f}, R² {hs['r2']:.0%}; {1 - hs['r2']:.0%} of "
                               f"{bundle['physical']} price variance is basis risk the future cannot remove.")
            elif asset.cross_hedge:
                reasons.append(asset.cross_hedge)
            for item in (supply_evidence or {}).get(key, [])[:2]:
                reasons.append(f"Supply chain: {item}.")
            if report.circuit_breaker:
                reasons.append(f"Circuit breaker: book VaR {report.var_pre:.2%} > {config.VAR_LIMIT:.0%} limit, size halved.")

            action = action_for(pm_weights[key], alloc.signed_weight)
            confidence, level, p_any = None, "n/a", None
            prob = report.asset_probabilities.get(key)
            if prob:
                # Re-size the book under each member's weather and count how often the action survives.
                member_actions = [
                    action_for(pm_weights[key], alloc.base_weight * alloc.vol_scale * conviction_multiplier(c) * d * cut)
                    for c, d in zip(prob.member_c, prob.member_direction)
                ]
                confidence = float(np.mean([a == action for a in member_actions]))
                level, p_any = confidence_level(confidence), prob.p_any
                reasons.append(f"Ensemble ({prob.n_members} members): {p_any:.0%} breach a trigger within "
                               f"{config.FORECAST_DAYS} days; C_event p10–p90 {prob.c_event_p10:.2f}–{prob.c_event_p90:.2f}; "
                               f"{confidence:.0%} of members support {action}.")

            collar = collars.get(key)
            hedge = collar.message.split("Collar ", 1)[-1] if collar else None
            recs.append(Recommendation(
                asset=key, action=action, pm_weight=pm_weights[key], ai_weight=alloc.signed_weight,
                c_event=sig.c_event, bias=sig.bias, headline=headline, reasons=reasons, hedge=hedge,
                confidence=confidence, confidence_level=level, p_disruption=p_any,
            ))
        return recs
