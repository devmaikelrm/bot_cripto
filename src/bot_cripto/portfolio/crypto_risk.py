"""Crypto-specific portfolio risk engine."""

from __future__ import annotations

from bot_cripto.core.config import Settings
from bot_cripto.risk.engine import RiskEngine, RiskLimits


class CryptoRiskEngine(RiskEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            limits=RiskLimits(
                risk_per_trade=float(settings.risk_per_trade),
                max_daily_drawdown=float(settings.max_daily_drawdown),
                max_weekly_drawdown=float(settings.max_weekly_drawdown),
                max_position_size=float(settings.max_position_size),
                risk_score_block_threshold=float(settings.risk_score_block_threshold),
                position_size_multiplier=float(settings.risk_position_size_multiplier),
                cooldown_minutes=int(settings.risk_cooldown_minutes),
                enable_kelly=bool(settings.risk_enable_kelly),
                kelly_fraction=float(settings.risk_kelly_fraction),
                cvar_enabled=bool(settings.risk_cvar_enabled),
                cvar_alpha=float(settings.risk_cvar_alpha),
                cvar_min_samples=int(settings.risk_cvar_min_samples),
                cvar_limit=float(settings.risk_cvar_limit),
                circuit_breaker_minutes=int(settings.risk_circuit_breaker_minutes),
            )
        )
