"""Portfolio allocation blending: HRP + Kelly + Views + dynamic corr proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot_cripto.risk.hrp import hrp_allocate


@dataclass(frozen=True)
class BlendAllocationResult:
    weights: dict[str, float]
    hrp_weights: dict[str, float]
    kelly_weights: dict[str, float]
    view_weights: dict[str, float]
    mean_abs_corr: float
    corr_shrink_applied: float
    method: str


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = float(sum(clipped.values()))
    if s <= 0:
        n = max(len(clipped), 1)
        return {k: 1.0 / n for k in clipped}
    return {k: v / s for k, v in clipped.items()}


def _kelly_like_weights(returns: pd.DataFrame) -> dict[str, float]:
    # Long-only Kelly proxy: edge / variance, clipped to positive.
    mu = returns.mean(axis=0).astype(float)
    var = returns.var(axis=0).astype(float).replace(0.0, np.nan)
    raw = (mu / var).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    payload = {k: max(0.0, float(v)) for k, v in raw.to_dict().items()}
    return _normalize(payload)


def _view_weights(assets: list[str], views: dict[str, float] | None) -> dict[str, float]:
    if not views:
        return {a: 1.0 / max(len(assets), 1) for a in assets}
    raw = {a: max(0.0, float(views.get(a, 0.0) + 1.0)) for a in assets}
    return _normalize(raw)


def _mean_abs_corr(returns: pd.DataFrame) -> float:
    corr = returns.corr().fillna(0.0)
    vals: list[float] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            vals.append(abs(float(corr.loc[a, b])))
    if not vals:
        return 0.0
    return float(np.mean(np.array(vals, dtype=float)))


def _apply_corr_shrink(
    weights: dict[str, float],
    mean_abs_corr: float,
    threshold: float,
    max_shrink: float,
) -> tuple[dict[str, float], float]:
    if mean_abs_corr <= threshold:
        return weights, 0.0
    over = min(1.0, (mean_abs_corr - threshold) / max(1e-9, (1.0 - threshold)))
    shrink = max(0.0, min(max_shrink, over * max_shrink))
    n = max(len(weights), 1)
    eq = {k: 1.0 / n for k in weights}
    out = {k: ((1.0 - shrink) * weights[k] + shrink * eq[k]) for k in weights}
    return _normalize(out), float(shrink)


def blend_allocations(
    returns: pd.DataFrame,
    views: dict[str, float] | None = None,
    w_hrp: float = 0.5,
    w_kelly: float = 0.3,
    w_views: float = 0.2,
    corr_threshold: float = 0.45,
    corr_max_shrink: float = 0.50,
) -> BlendAllocationResult:
    """Blend HRP, Kelly proxy, and discretionary/model views."""
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets for blended allocation")
    clean = returns.dropna(axis=0, how="any")
    if clean.shape[0] < 30:
        raise ValueError("Need at least 30 aligned return rows for blended allocation")

    weights_raw = np.array([w_hrp, w_kelly, w_views], dtype=float)
    weights_raw = np.clip(weights_raw, 0.0, None)
    if float(weights_raw.sum()) <= 0:
        weights_raw = np.array([1.0, 0.0, 0.0], dtype=float)
    mix = weights_raw / float(weights_raw.sum())

    hrp = hrp_allocate(clean).weights
    kelly = _kelly_like_weights(clean)
    vw = _view_weights(list(clean.columns), views)

    blend = {}
    for asset in clean.columns:
        blend[asset] = (
            mix[0] * float(hrp.get(asset, 0.0))
            + mix[1] * float(kelly.get(asset, 0.0))
            + mix[2] * float(vw.get(asset, 0.0))
        )
    blend = _normalize(blend)

    mac = _mean_abs_corr(clean)
    adjusted, shrink = _apply_corr_shrink(
        blend,
        mean_abs_corr=mac,
        threshold=corr_threshold,
        max_shrink=corr_max_shrink,
    )

    return BlendAllocationResult(
        weights=adjusted,
        hrp_weights=hrp,
        kelly_weights=kelly,
        view_weights=vw,
        mean_abs_corr=mac,
        corr_shrink_applied=shrink,
        method="blend_hrp_kelly_views_corr_proxy",
    )
