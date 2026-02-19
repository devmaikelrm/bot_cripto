"""Triple-barrier labeling and purged time split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def get_ewm_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Estimate realized volatility with exponentially weighted std."""
    ret = close.astype(float).pct_change()
    return ret.ewm(span=span, adjust=False).std()


@dataclass(frozen=True)
class TripleBarrierConfig:
    pt_mult: float = 2.0
    sl_mult: float = 2.0
    horizon_bars: int = 20
    vol_span: int = 100
    min_target_vol: float = 1e-6


def _events_from_close(close: pd.Series, cfg: TripleBarrierConfig) -> pd.DataFrame:
    idx = close.index
    trgt = get_ewm_volatility(close, span=cfg.vol_span).clip(lower=cfg.min_target_vol)
    t1_pos = np.arange(len(idx)) + int(cfg.horizon_bars)
    valid = t1_pos < len(idx)
    events = pd.DataFrame(index=idx[valid])
    events["t1"] = idx[t1_pos[valid]]
    events["trgt"] = trgt.loc[events.index].astype(float)
    events["side"] = 1.0
    return events


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] = (2.0, 2.0),
) -> pd.DataFrame:
    """Apply triple barrier and return touched barriers and labels."""
    out = pd.DataFrame(index=events.index)
    out["t1"] = events["t1"]
    pt_mult, sl_mult = float(pt_sl[0]), float(pt_sl[1])
    trgt = events["trgt"].astype(float)
    side = events.get("side", pd.Series(1.0, index=events.index)).astype(float)

    pt = (pt_mult * trgt) if pt_mult > 0 else pd.Series(np.nan, index=events.index)
    sl = (-sl_mult * trgt) if sl_mult > 0 else pd.Series(np.nan, index=events.index)

    out["pt"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["sl"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["ret_at_touch"] = 0.0
    out["first_touch"] = "t1"
    out["label"] = 0

    close_f = close.astype(float)
    for loc, end_ts in events["t1"].items():
        path = close_f.loc[loc:end_ts]
        if path.empty:
            continue
        rel = (path / float(close_f.loc[loc]) - 1.0) * float(side.loc[loc])

        first_pt = rel[rel >= float(pt.loc[loc])].index.min() if pt_mult > 0 else pd.NaT
        first_sl = rel[rel <= float(sl.loc[loc])].index.min() if sl_mult > 0 else pd.NaT

        chosen = end_ts
        touch = "t1"
        label = 0
        if pd.notna(first_pt) and (pd.isna(first_sl) or first_pt <= first_sl):
            chosen = first_pt
            touch = "pt"
            label = 1
        elif pd.notna(first_sl):
            chosen = first_sl
            touch = "sl"
            label = -1

        out.at[loc, "pt"] = first_pt
        out.at[loc, "sl"] = first_sl
        out.at[loc, "ret_at_touch"] = float(rel.loc[chosen]) if chosen in rel.index else 0.0
        out.at[loc, "first_touch"] = touch
        out.at[loc, "label"] = int(label)
    return out


def build_triple_barrier_labels(
    frame: pd.DataFrame,
    price_col: str = "close",
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Generate triple-barrier labels aligned to the input frame index."""
    cfg = config or TripleBarrierConfig()
    if price_col not in frame.columns:
        raise ValueError(f"Missing price column: {price_col}")
    close = frame[price_col].astype(float)
    events = _events_from_close(close, cfg)
    touched = apply_triple_barrier(close, events, pt_sl=(cfg.pt_mult, cfg.sl_mult))

    out = frame.copy()
    out["tb_label"] = 0
    out["tb_ret"] = 0.0
    out["tb_first_touch"] = "t1"
    out["tb_t1"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out.loc[touched.index, "tb_label"] = touched["label"].astype(int)
    out.loc[touched.index, "tb_ret"] = touched["ret_at_touch"].astype(float)
    out.loc[touched.index, "tb_first_touch"] = touched["first_touch"].astype(str)
    out.loc[touched.index, "tb_t1"] = touched["t1"]
    return out


def purged_train_test_split(
    n_samples: int,
    test_start: int,
    test_end: int,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple purged temporal split by index positions."""
    if n_samples <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if test_start < 0 or test_end > n_samples or test_start >= test_end:
        raise ValueError("Invalid test range")

    test_idx = np.arange(test_start, test_end, dtype=int)
    left_end = max(0, test_start - purge_size)
    right_start = min(n_samples, test_end + embargo_size)

    left = np.arange(0, left_end, dtype=int)
    right = np.arange(right_start, n_samples, dtype=int)
    train_idx = np.concatenate([left, right])
    return train_idx, test_idx
