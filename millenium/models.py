"""Result types passed between pipeline stages and rendered by the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TriggerReading:
    name: str
    facility: str
    metric: str
    unit: str
    threshold: float
    peak_value: float
    peak_date: date
    severity: float         # lead-time-discounted 0..1
    bias: int
    in_season: bool = True

    @property
    def breached(self) -> bool:
        return self.severity >= 0.5


@dataclass
class AssetSignal:
    asset: str
    c_event: float
    bias: str               # LONG / SHORT / NEUTRAL
    direction: int          # +1 / -1 used for sizing
    readings: list[TriggerReading] = field(default_factory=list)


@dataclass
class Allocation:
    asset: str
    instrument: str
    base_weight: float
    realized_vol_14d: float
    vol_scale: float
    effective_size: float   # Base * VolScale * (0.5 + 1.25 * C)
    signed_weight: float    # after direction and any circuit-breaker cut


@dataclass
class HedgeAlert:
    kind: str               # ZERO_COST_COLLAR / CIRCUIT_BREAKER
    asset: str | None
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PortfolioReport:
    as_of: str
    scenario: str
    signals: dict[str, AssetSignal]
    allocations: dict[str, Allocation]
    alerts: list[HedgeAlert]
    var_pre: float
    var_post: float
    circuit_breaker: bool
    vol_metrics: dict[str, dict]
    asset_probabilities: dict = field(default_factory=dict)    # asset -> probability.AssetProbability
    trigger_probabilities: list = field(default_factory=list)  # probability.TriggerProbability
    ensemble_models: tuple = ()


@dataclass
class Recommendation:
    asset: str
    action: str             # INCREASE / REDUCE / HOLD / FLIP TO LONG / FLIP TO SHORT
    pm_weight: float
    ai_weight: float
    c_event: float
    bias: str
    headline: str
    reasons: list[str] = field(default_factory=list)
    hedge: str | None = None
    confidence: float | None = None       # share of ensemble members that lead to the same action
    confidence_level: str = "n/a"
    p_disruption: float | None = None     # share of members breaching any trigger in the window

    @property
    def delta(self) -> float:
        return self.ai_weight - self.pm_weight
