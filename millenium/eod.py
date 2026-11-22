"""End-of-day review: what worked, what failed and what to watch, built only from the day's numbers.

`eod_facts` collects the facts, `rule_based_note` turns them into a short note with no model call, and
`claude_note` optionally asks Claude to write the same note in prose (ANTHROPIC_API_KEY or an `ant` profile).
"""
from __future__ import annotations

import json

import numpy as np

from . import config
from .backtest import BacktestResult, signal_hit_rates
from .supplychain import LEVEL_NAME, WATCH, SupplyChainSnapshot

EOD_MODEL = "claude-opus-5-5"
MATERIAL_PNL = 500.0   # $ difference below which AI vs PM is a wash


def eod_facts(report, recs, bt: BacktestResult, supply: SupplyChainSnapshot | None, pm_weights: dict[str, float],
              nav: float, data_status: list[dict]) -> dict:
    day = bt.returns.index[-1]
    r, held = bt.asset_returns.loc[day], bt.held.loc[day]
    assets = []
    for k, a in config.ASSETS.items():
        pm_pnl, ai_pnl = pm_weights[k] * r[k] * nav, held[k] * r[k] * nav
        assets.append({"asset": a.name, "instrument": a.instrument, "return": round(float(r[k]), 4),
                       "pm_weight": round(pm_weights[k], 3), "ai_weight": round(float(held[k]), 3),
                       "pm_pnl": round(pm_pnl), "ai_pnl": round(ai_pnl), "ai_minus_pm": round(ai_pnl - pm_pnl)})
    week = bt.returns.tail(5)
    hits = signal_hit_rates(bt)
    watch = [{"risk": t.trigger, "asset": config.ASSETS[t.asset].name, "p_breach_8d": round(t.p_breach, 2)}
             for t in sorted(report.trigger_probabilities, key=lambda t: -t.p_breach) if t.p_breach >= 0.10][:4]
    supply_flags = [{"signal": s.label, "level": LEVEL_NAME[s.level], "detail": s.detail, "source": s.source}
                    for s in (supply.signals.values() if supply else []) if s.level >= WATCH]
    return {
        "date": str(day.date()), "scenario": report.scenario, "nav": nav,
        "assets": assets,
        "day_total": {"pm_pnl": round(sum(x["pm_pnl"] for x in assets)), "ai_pnl": round(sum(x["ai_pnl"] for x in assets))},
        "week_total": {"pm_return": round(float(week["Benchmark (PM static)"].sum()), 4),
                       "ai_return": round(float(week["AI resized"].sum()), 4), "days": len(week)},
        "signal_scorecard": [{"asset": config.ASSETS[k].name, "signal_days": int(row.signal_days),
                              "hit_rate_5d": None if np.isnan(row.hit_rate) else round(row.hit_rate, 2)}
                             for k, row in hits.iterrows()],
        "risk_controls": {"circuit_breaker_now": report.circuit_breaker,
                          "breaker_days_last_5": int(bt.breaker.tail(5).sum()),
                          "collar_alerts": [a.message for a in report.alerts if a.kind == "ZERO_COST_COLLAR"]},
        "recommendations": [{"asset": config.ASSETS[x.asset].name, "action": x.action, "to_weight": round(x.ai_weight, 3),
                             "confidence": x.confidence_level, "why": x.headline} for x in recs],
        "watch_tomorrow": watch, "supply_chain_flags": supply_flags,
        "data_issues": [d for d in data_status if not d["ok"]],
    }


def rule_based_note(f: dict) -> dict[str, list[str]]:
    worked, failed = [], []
    for a in sorted(f["assets"], key=lambda x: -abs(x["ai_minus_pm"])):
        line = (f"{a['asset']}: {a['instrument']} {a['return']:+.2%}; AI held {a['ai_weight']:+.3f} vs your "
                f"{a['pm_weight']:.3f} → {'+' if a['ai_minus_pm'] >= 0 else '−'}${abs(a['ai_minus_pm']):,.0f} vs your book")
        if a["ai_minus_pm"] >= MATERIAL_PNL:
            worked.append(line)
        elif a["ai_minus_pm"] <= -MATERIAL_PNL:
            failed.append(line)
    for s in f["signal_scorecard"]:
        if s["hit_rate_5d"] is None or s["signal_days"] < 10:
            continue
        if s["hit_rate_5d"] >= 0.55:
            worked.append(f"{s['asset']} weather signals: {s['hit_rate_5d']:.0%} right over 5 days ({s['signal_days']} signal days)")
        elif s["hit_rate_5d"] <= 0.45:
            failed.append(f"{s['asset']} weather signals: only {s['hit_rate_5d']:.0%} right over 5 days ({s['signal_days']} signal days)")
    rc = f["risk_controls"]
    if rc["breaker_days_last_5"]:
        worked.append(f"Circuit breaker cut risk on {rc['breaker_days_last_5']} of the last 5 days")
    failed += [f"Data: {d['name']} — {d['message']}" for d in f["data_issues"]]
    watch = [f"{w['risk']} ({w['asset']}): {w['p_breach_8d']:.0%} chance in 8 days" for w in f["watch_tomorrow"]]
    watch += [f"{s['signal']}: {s['detail']}" for s in f["supply_chain_flags"]]
    return {"worked": worked[:4] or ["Nothing material: AI and your book moved within $500 of each other."],
            "failed": failed[:4] or ["Nothing failed today."],
            "watch": watch[:4] or ["No supply-chain risks above 10% for the next 8 days."]}


SYSTEM_PROMPT = """You write the end-of-day note for a commodity portfolio manager who runs a weather-driven \
book in anhydrous ammonia (via CF), lithium (via LIT), FCOJ (OJ=F) and grain sorghum (via CBOT corn).
Use only the JSON facts you are given. Every number you mention must appear in the facts (rounding is fine); \
never invent prices, events, causes or forecasts. "AI" means the AI-resized weights, "your book" the PM's own weights.
Write four short markdown sections with these exact headings: **What worked**, **What failed**, \
**Data & model issues**, **Watch tomorrow**. Two to four bullets each; if a section has nothing, write one bullet \
saying so. Plain, direct language a busy PM can read in 30 seconds. No preamble, no sign-off."""


def claude_note(facts: dict) -> tuple[str | None, str | None]:
    """Returns (markdown, None) on success or (None, reason) when Claude is unavailable."""
    try:
        import anthropic
    except ImportError:
        return None, "The anthropic package is not installed."
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=EOD_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                      # re-run on Anthropic's recommended model if declined
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Today's facts (JSON):\n" + json.dumps(facts, sort_keys=True, default=str)}],
        )
    except anthropic.AuthenticationError:
        return None, "Anthropic rejected the credentials; check ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "Rate-limited by the Anthropic API; try again in a minute."
    except anthropic.APIStatusError as exc:
        return None, f"Anthropic API error {exc.status_code}."
    except anthropic.APIConnectionError:
        return None, "Could not reach the Anthropic API."
    except TypeError as exc:   # raised by the SDK when no credential source is configured at all
        if "authentication" not in str(exc).lower():
            raise
        return None, "No Anthropic credentials found. Set ANTHROPIC_API_KEY (or run `ant auth login`) to enable."
    if response.stop_reason == "refusal":
        return None, "Claude declined to write this note; the rule-based summary stands."
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text, None) if text else (None, "Claude returned an empty note.")
