"""Hierarchical Risk Parity (HRP) allocator MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HRPAllocationResult:
    weights: dict[str, float]
    ordered_assets: list[str]
    method: str


def _inv_variance_weights(cov: pd.DataFrame) -> pd.Series:
    diag = np.diag(cov.values).astype(float)
    inv = np.where(diag > 0, 1.0 / diag, 0.0)
    s = float(inv.sum())
    if s <= 0:
        n = len(inv)
        return pd.Series(np.full(n, 1.0 / max(n, 1)), index=cov.index)
    return pd.Series(inv / s, index=cov.index)


def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    sub_cov = cov.loc[assets, assets]
    w = _inv_variance_weights(sub_cov).values.reshape(-1, 1)
    v = float((w.T @ sub_cov.values @ w)[0, 0])
    return max(v, 1e-12)


def _distance_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    d = ((1.0 - corr) / 2.0).clip(lower=0.0)
    return d


def _greedy_seriation(distance: pd.DataFrame) -> list[str]:
    """Greedy ordering approximation for quasi-diagonalization."""
    assets = list(distance.index)
    if len(assets) <= 2:
        return assets

    # Start with closest pair.
    best_pair: tuple[str, str] | None = None
    best_d = float("inf")
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            d = float(distance.loc[a, b])
            if d < best_d:
                best_d = d
                best_pair = (a, b)
    if best_pair is None:
        return assets

    left = [best_pair[0], best_pair[1]]
    used = set(left)
    remaining = [a for a in assets if a not in used]
    while remaining:
        candidate = None
        best_ext_d = float("inf")
        best_side = "right"
        for asset in remaining:
            d_left = float(distance.loc[asset, left[0]])
            d_right = float(distance.loc[left[-1], asset])
            if d_left < best_ext_d:
                best_ext_d = d_left
                candidate = asset
                best_side = "left"
            if d_right < best_ext_d:
                best_ext_d = d_right
                candidate = asset
                best_side = "right"
        if candidate is None:
            break
        if best_side == "left":
            left.insert(0, candidate)
        else:
            left.append(candidate)
        used.add(candidate)
        remaining = [a for a in assets if a not in used]
    return left


def _recursive_bisection(cov: pd.DataFrame, ordered: list[str]) -> pd.Series:
    w = pd.Series(1.0, index=ordered, dtype=float)
    clusters: list[list[str]] = [ordered]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        c1 = cluster[:split]
        c2 = cluster[split:]
        v1 = _cluster_variance(cov, c1)
        v2 = _cluster_variance(cov, c2)
        alpha = 1.0 - (v1 / (v1 + v2))
        w[c1] *= alpha
        w[c2] *= 1.0 - alpha
        clusters.append(c1)
        clusters.append(c2)
    s = float(w.sum())
    if s > 0:
        w /= s
    return w


def hrp_allocate(returns: pd.DataFrame) -> HRPAllocationResult:
    """Compute HRP-like portfolio weights from asset returns matrix."""
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets with return history")
    clean = returns.dropna(axis=0, how="any")
    if clean.shape[0] < 30:
        raise ValueError("Need at least 30 aligned return rows for HRP allocation")

    cov = clean.cov()
    corr = clean.corr().fillna(0.0).clip(lower=-1.0, upper=1.0)
    distance = _distance_from_corr(corr)
    ordered = _greedy_seriation(distance)
    weights = _recursive_bisection(cov, ordered)
    payload = {k: float(v) for k, v in weights.items()}
    return HRPAllocationResult(weights=payload, ordered_assets=ordered, method="hrp_greedy")
