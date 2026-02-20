"""Global capital allocator across market domains."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot_cripto.risk.allocation_blend import blend_allocations


@dataclass(frozen=True)
class CapitalAllocation:
    total_capital: float
    weights: dict[str, float]
    notionals: dict[str, float]
    method: str


class CapitalAllocator:
    """Compute per-symbol capital notionals from return matrix."""

    def allocate(
        self,
        returns: pd.DataFrame,
        total_capital: float,
        views: dict[str, float] | None = None,
    ) -> CapitalAllocation:
        if total_capital <= 0:
            raise ValueError("total_capital must be > 0")
        blend = blend_allocations(returns=returns, views=views)
        notionals = {k: float(v * total_capital) for k, v in blend.weights.items()}
        return CapitalAllocation(
            total_capital=float(total_capital),
            weights=blend.weights,
            notionals=notionals,
            method=blend.method,
        )
