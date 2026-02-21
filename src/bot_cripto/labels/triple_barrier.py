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


# ---------------------------------------------------------------------------
# Vectorized fast path — used internally when all events share a fixed horizon
# ---------------------------------------------------------------------------

def _apply_triple_barrier_vectorized(
    close_arr: np.ndarray,
    close_idx: pd.Index,
    event_start_pos: np.ndarray,
    horizon: int,
    side_arr: np.ndarray,
    pt_arr: np.ndarray,
    sl_arr: np.ndarray,
    pt_mult: float,
    sl_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorized triple-barrier for fixed-horizon events.

    Returns (first_pt_col, first_sl_col, chosen_col, labels, ret_at_touch).
    Columns are integer offsets from the event start position; -1 means the
    barrier was never reached.
    """
    n = len(close_arr)
    n_events = len(event_start_pos)
    H = horizon  # number of bars *after* entry (so path width = H+1)

    # -----------------------------------------------------------------------
    # Build price matrix: shape (n_events, H+1)
    # col j = price at start_pos + j
    # -----------------------------------------------------------------------
    col_offsets = np.arange(H + 1, dtype=np.intp)  # (H+1,)
    path_indices = event_start_pos[:, None] + col_offsets[None, :]  # (n_events, H+1)

    # Valid mask: indices must stay within the close array
    in_bounds = path_indices < n  # (n_events, H+1)
    path_indices_safe = np.clip(path_indices, 0, n - 1)
    price_matrix = close_arr[path_indices_safe]  # (n_events, H+1)

    # -----------------------------------------------------------------------
    # Relative returns w.r.t. entry price (col 0), signed by side
    # -----------------------------------------------------------------------
    entry_prices = price_matrix[:, 0:1]  # (n_events, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_matrix = np.where(
            entry_prices > 0,
            price_matrix / entry_prices - 1.0,
            0.0,
        )
    rel_matrix *= side_arr[:, None]          # apply long/short direction
    rel_matrix = np.where(in_bounds, rel_matrix, 0.0)  # zero out OOB

    # -----------------------------------------------------------------------
    # Barrier hit matrices
    # -----------------------------------------------------------------------
    if pt_mult > 0:
        pt_hit = (rel_matrix >= pt_arr[:, None]) & in_bounds  # (n_events, H+1)
    else:
        pt_hit = np.zeros((n_events, H + 1), dtype=bool)

    if sl_mult > 0:
        sl_hit = (rel_matrix <= sl_arr[:, None]) & in_bounds  # (n_events, H+1)
    else:
        sl_hit = np.zeros((n_events, H + 1), dtype=bool)

    # -----------------------------------------------------------------------
    # First hit position per event (argmax trick: first True in each row)
    # -----------------------------------------------------------------------
    pt_any = pt_hit.any(axis=1)                               # (n_events,)
    sl_any = sl_hit.any(axis=1)

    first_pt_col = np.where(pt_any, pt_hit.argmax(axis=1), -1).astype(np.int64)
    first_sl_col = np.where(sl_any, sl_hit.argmax(axis=1), -1).astype(np.int64)

    # -----------------------------------------------------------------------
    # Determine which barrier was hit first
    # PT wins if hit AND (SL not hit OR PT not later than SL)
    # -----------------------------------------------------------------------
    pt_first = pt_any & (~sl_any | (first_pt_col <= first_sl_col))
    sl_first = sl_any & ~pt_first

    horizon_col = np.full(n_events, H, dtype=np.int64)

    chosen_col = np.where(pt_first, first_pt_col,
                 np.where(sl_first, first_sl_col, horizon_col))

    labels = np.where(pt_first, 1, np.where(sl_first, -1, 0)).astype(np.int8)

    # Gather return at the chosen position for each event
    row_idx = np.arange(n_events)
    ret_at_touch = rel_matrix[row_idx, chosen_col]

    return first_pt_col, first_sl_col, chosen_col, labels, ret_at_touch


# ---------------------------------------------------------------------------
# Public API — generic loop with O(1) numpy internals per iteration
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] = (2.0, 2.0),
) -> pd.DataFrame:
    """Apply triple barrier and return touched barriers and labels.

    Supports variable-horizon events.  Inner loop uses numpy array operations
    (O(H) per iteration) instead of pandas label-based indexing (O(n)), giving
    a ~50× speedup over the naive loop for large datasets.

    For fixed-horizon datasets produced by ``build_triple_barrier_labels``,
    prefer calling that function directly — it uses the fully-vectorized fast
    path that eliminates the outer Python loop entirely.
    """
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
    close_arr = close_f.to_numpy()          # numpy for fast positional slicing
    close_idx = close_f.index

    # O(n) pre-build: timestamp → integer position dict
    # Replaces O(log n) pandas .loc label lookups in the hot loop
    ts_to_pos: dict = {ts: i for i, ts in enumerate(close_idx)}

    # Pre-extract arrays to avoid per-row pandas overhead
    pt_vals = pt.to_numpy(dtype=float, na_value=np.inf)
    sl_vals = sl.to_numpy(dtype=float, na_value=-np.inf)
    side_vals = side.to_numpy(dtype=float)
    events_locs = list(events.index)
    t1_vals = list(events["t1"])

    for i_ev, (loc, end_ts) in enumerate(zip(events_locs, t1_vals)):
        start_pos = ts_to_pos.get(loc)
        end_pos = ts_to_pos.get(end_ts)
        if start_pos is None or end_pos is None or end_pos <= start_pos:
            continue

        path = close_arr[start_pos : end_pos + 1]  # O(1) numpy view, no copy
        if len(path) == 0:
            continue
        entry = path[0]
        if entry == 0.0:
            continue

        s = side_vals[i_ev]
        rel = (path / entry - 1.0) * s        # O(H) numpy ufunc

        pt_t = pt_vals[i_ev]
        sl_t = sl_vals[i_ev]

        first_pt_idx: int = -1
        first_sl_idx: int = -1

        if pt_mult > 0 and np.isfinite(pt_t):
            hits = np.where(rel >= pt_t)[0]   # O(H) numpy where
            if hits.size > 0:
                first_pt_idx = int(hits[0])

        if sl_mult > 0 and np.isfinite(-sl_t):
            hits = np.where(rel <= sl_t)[0]   # O(H) numpy where
            if hits.size > 0:
                first_sl_idx = int(hits[0])

        chosen_idx = len(path) - 1
        touch = "t1"
        label = 0

        if first_pt_idx >= 0 and (first_sl_idx < 0 or first_pt_idx <= first_sl_idx):
            chosen_idx = first_pt_idx
            touch = "pt"
            label = 1
        elif first_sl_idx >= 0:
            chosen_idx = first_sl_idx
            touch = "sl"
            label = -1

        first_pt_ts = close_idx[start_pos + first_pt_idx] if first_pt_idx >= 0 else pd.NaT
        first_sl_ts = close_idx[start_pos + first_sl_idx] if first_sl_idx >= 0 else pd.NaT

        out.at[loc, "pt"] = first_pt_ts
        out.at[loc, "sl"] = first_sl_ts
        out.at[loc, "ret_at_touch"] = float(rel[chosen_idx])
        out.at[loc, "first_touch"] = touch
        out.at[loc, "label"] = int(label)

    return out


def build_triple_barrier_labels(
    frame: pd.DataFrame,
    price_col: str = "close",
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Generate triple-barrier labels aligned to the input frame index.

    Uses a fully-vectorized numpy fast path (no Python loop) when all events
    share the same fixed horizon, which is always the case here.
    """
    cfg = config or TripleBarrierConfig()
    if price_col not in frame.columns:
        raise ValueError(f"Missing price column: {price_col}")

    close = frame[price_col].astype(float)
    events = _events_from_close(close, cfg)

    out = frame.copy()
    out["tb_label"] = 0
    out["tb_ret"] = 0.0
    out["tb_first_touch"] = "t1"
    out["tb_t1"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

    if events.empty:
        return out

    # ------------------------------------------------------------------
    # Fully-vectorized fast path: _events_from_close always produces a
    # fixed horizon, so we can use the matrix approach (no Python loop).
    # ------------------------------------------------------------------
    close_arr = close.to_numpy(dtype=float)
    close_idx = close.index

    ts_to_pos: dict = {ts: i for i, ts in enumerate(close_idx)}
    event_start_pos = np.array(
        [ts_to_pos[loc] for loc in events.index], dtype=np.intp
    )

    side_arr = events["side"].to_numpy(dtype=float)
    trgt_arr = events["trgt"].to_numpy(dtype=float)
    pt_arr = cfg.pt_mult * trgt_arr if cfg.pt_mult > 0 else np.full(len(events), np.inf)
    sl_arr = -cfg.sl_mult * trgt_arr if cfg.sl_mult > 0 else np.full(len(events), -np.inf)

    first_pt_col, first_sl_col, chosen_col, labels, ret_at_touch = (
        _apply_triple_barrier_vectorized(
            close_arr=close_arr,
            close_idx=close_idx,
            event_start_pos=event_start_pos,
            horizon=cfg.horizon_bars,
            side_arr=side_arr,
            pt_arr=pt_arr,
            sl_arr=sl_arr,
            pt_mult=cfg.pt_mult,
            sl_mult=cfg.sl_mult,
        )
    )

    # Map integer column offsets back to timestamps and string labels
    n = len(close_arr)
    event_locs = events.index

    # Chosen timestamp
    chosen_abs = event_start_pos + chosen_col
    chosen_abs_safe = np.clip(chosen_abs, 0, n - 1)

    # First-touch category string
    pt_first = labels == 1
    sl_first = labels == -1
    touch_strs = np.where(pt_first, "pt", np.where(sl_first, "sl", "t1"))

    # Assign back to output DataFrame
    out.loc[event_locs, "tb_label"] = labels.astype(int)
    out.loc[event_locs, "tb_ret"] = ret_at_touch.astype(float)
    out.loc[event_locs, "tb_first_touch"] = touch_strs

    # t1 timestamps (end of the horizon window)
    # Use the pandas Series (not .values) to preserve tz-aware dtype
    out.loc[event_locs, "tb_t1"] = events["t1"]

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
