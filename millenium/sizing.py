"""Dynamic position resizing: inverse-volatility scaling times event conviction."""
from __future__ import annotations

import numpy as np

from . import config
from .config import Asset
from .models import Allocation, AssetSignal


def conviction_multiplier(c_event: float) -> float:
    return 0.5 + 1.25 * c_event


def inverse_vol_scales(vols: dict[str, float]) -> dict[str, float]:
    """1/sigma normalized so the average scale is 1.0; missing vols get a neutral 1.0."""
    inv = {k: 1.0 / v for k, v in vols.items() if v and np.isfinite(v) and v > 0}
    if not inv:
        return {k: 1.0 for k in vols}
    mean_inv = float(np.mean(list(inv.values())))
    lo, hi = config.VOL_SCALE_BOUNDS
    return {k: float(np.clip(inv[k] / mean_inv, lo, hi)) if k in inv else 1.0 for k in vols}


class PositionSizer:
    def __init__(self, assets: dict[str, Asset] = config.ASSETS):
        self.assets = assets

    def size(self, signals: dict[str, AssetSignal], vols_14d: dict[str, float],
             base_weights: dict[str, float] | None = None) -> dict[str, Allocation]:
        """`base_weights` (e.g. the PM's book) override the configured strategic weights."""
        scales = inverse_vol_scales(vols_14d)
        out = {}
        for key, asset in self.assets.items():
            sig = signals[key]
            base = asset.base_weight if base_weights is None else base_weights[key]
            effective = base * scales[key] * conviction_multiplier(sig.c_event)
            out[key] = Allocation(
                asset=key, instrument=asset.instrument, base_weight=base,
                realized_vol_14d=vols_14d[key], vol_scale=scales[key],
                effective_size=effective, signed_weight=effective * sig.direction,
            )
        return out
