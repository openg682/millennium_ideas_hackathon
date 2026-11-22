"""CLI: `python -m millenium [--scenario NAME] [--json]` runs the pipeline once."""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict

from .advisor import Advisor
from .pipeline import SCENARIOS, Pipeline

warnings.filterwarnings("ignore", module="yfinance")


def main() -> None:
    parser = argparse.ArgumentParser(description="Niche commodity disruption risk pipeline")
    parser.add_argument("--scenario", default="Live", choices=list(SCENARIOS))
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args()

    pipeline = Pipeline()
    weather, market = pipeline.fetch()
    report = pipeline.evaluate(weather, market, args.scenario)
    if args.json:
        print(json.dumps(asdict(report), default=str, indent=2))
        return

    print(f"As of {report.as_of}  |  scenario: {report.scenario}")
    print(f"{'Asset':<6}{'Instr':<7}{'C_event':>8}  {'Bias':<8}{'RV14':>7}{'VolScl':>8}{'EffSize':>9}{'Weight':>9}")
    for key, sig in report.signals.items():
        a = report.allocations[key]
        print(f"{key:<6}{a.instrument:<7}{sig.c_event:>8.2f}  {sig.bias:<8}{a.realized_vol_14d:>7.1%}"
              f"{a.vol_scale:>8.2f}{a.effective_size:>9.3f}{a.signed_weight:>+9.3f}")
        for r in sig.readings:
            flag = "BREACH" if r.breached else ""
            print(f"      · {r.name:<40} peak {r.peak_value:>8.1f} {r.unit:<5} on {r.peak_date}  sev {r.severity:.2f} {flag}")
    print(f"Portfolio VaR(95%, 1d): {report.var_pre:.2%} -> {report.var_post:.2%}  circuit breaker: {report.circuit_breaker}")
    for alert in report.alerts or []:
        print(f"[{alert.kind}] {alert.message}")
    if not report.alerts:
        print("No active hedging alerts.")
    print("\nAI recommendations (vs configured base weights):")
    base = {k: a.base_weight for k, a in report.allocations.items()}
    for rec in Advisor().recommend(report, base, weather.today):
        conf = f"{rec.confidence_level} ({rec.confidence:.0%})" if rec.confidence is not None else "n/a"
        p_txt = f"{rec.p_disruption:.0%}" if rec.p_disruption is not None else "n/a"
        print(f"  {rec.asset:<5} {rec.action:<14} {rec.pm_weight:.3f} -> {rec.ai_weight:+.3f}  "
              f"confidence {conf}  P(disruption 8d) {p_txt}  | {rec.headline}")


if __name__ == "__main__":
    main()
