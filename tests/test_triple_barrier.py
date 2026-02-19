from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.labels.triple_barrier import (
    TripleBarrierConfig,
    build_triple_barrier_labels,
    purged_train_test_split,
)


def test_build_triple_barrier_labels_has_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 102.0, len(idx)), index=idx)
    frame = pd.DataFrame({"close": close, "volume": 1.0}, index=idx)
    labeled = build_triple_barrier_labels(
        frame,
        price_col="close",
        config=TripleBarrierConfig(horizon_bars=10, vol_span=20),
    )
    assert "tb_label" in labeled.columns
    assert "tb_ret" in labeled.columns
    assert "tb_first_touch" in labeled.columns
    assert set(labeled["tb_label"].unique()).issubset({-1, 0, 1})


def test_purged_train_test_split_respects_gaps() -> None:
    train, test = purged_train_test_split(
        n_samples=100,
        test_start=40,
        test_end=60,
        purge_size=5,
        embargo_size=3,
    )
    assert len(test) == 20
    assert test[0] == 40 and test[-1] == 59
    assert np.max(train[train < 40]) <= 34
    assert np.min(train[train > 59]) >= 63
