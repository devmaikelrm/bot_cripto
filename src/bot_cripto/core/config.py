"""Centralized runtime settings."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from datetime import time
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    """Supported logging formats."""

    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Global configuration loaded from env vars and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    exchange: str = "binance"
    data_provider: str = "binance"
    symbols: str = "BTC/USDT"
    timeframe: str = "5m"

    pred_horizon_steps: int = 5
    encoder_length: int = 60

    data_dir_raw: Path = Path("./data/raw")
    data_dir_processed: Path = Path("./data/processed")
    models_dir: Path = Path("./models")
    logs_dir: Path = Path("./logs")

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enable_commands: bool = False
    telegram_allowed_chat_ids: str = ""
    telegram_poll_seconds: Annotated[int, Field(ge=1, le=60)] = 2

    paper_mode: bool = True
    live_mode: bool = False

    risk_max: Annotated[float, Field(ge=0.0, le=1.0)] = 0.30
    prob_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.60
    min_expected_return: Annotated[float, Field(ge=0.0)] = 0.002
    label_edge_return: Annotated[float, Field(ge=0.0)] = 0.0005
    fees_bps: Annotated[int, Field(ge=0)] = 10
    regime_adx_trend_min: float = 18.0
    regime_atr_high_vol_pct: float = 0.02
    macro_event_crisis_enabled: bool = True
    macro_event_crisis_windows_utc: str = "13:20-14:10"
    macro_event_crisis_weekdays: str = "0,1,2,3,4"
    macro_block_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70
    orderbook_sell_wall_threshold: Annotated[float, Field(ge=-1.0, le=1.0)] = -0.20
    social_sentiment_bull_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.60
    social_sentiment_bear_max: Annotated[float, Field(ge=0.0, le=1.0)] = 0.40
    context_prob_adjust_max: Annotated[float, Field(ge=0.0, le=0.3)] = 0.05
    social_sentiment_source: str = "auto"
    social_sentiment_endpoint: str = ""
    cryptopanic_api_key: str = ""
    social_sentiment_nlp_enabled: bool = True
    social_sentiment_nlp_model_id: str = "ProsusAI/finbert"
    social_sentiment_nlp_max_texts: Annotated[int, Field(ge=5, le=500)] = 120
    social_sentiment_weight_x: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    social_sentiment_weight_news: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    social_sentiment_weight_telegram: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    social_sentiment_ema_alpha: Annotated[float, Field(ge=0.01, le=1.0)] = 0.35
    x_bearer_token: str = ""
    x_query_template: str = "({coin} OR ${coin} OR #{coin}) lang:en -is:retweet"
    x_max_results: Annotated[int, Field(ge=10, le=100)] = 50
    telegram_sentiment_chat_ids: str = ""
    telegram_sentiment_lookback_limit: Annotated[int, Field(ge=10, le=500)] = 100
    risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.03
    max_weekly_drawdown: float = 0.07
    max_position_size: float = 1.0
    risk_score_block_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    risk_position_size_multiplier: Annotated[float, Field(ge=0.0)] = 10.0
    model_risk_vol_ref: Annotated[float, Field(gt=0.0)] = 0.02
    model_risk_spread_ref: Annotated[float, Field(gt=0.0)] = 0.05
    initial_equity: float = 10_000.0
    spread_bps: Annotated[int, Field(ge=0)] = 2
    slippage_bps: Annotated[int, Field(ge=0)] = 3
    stop_loss_buffer: Annotated[float, Field(ge=0.0)] = 0.0
    take_profit_buffer: Annotated[float, Field(ge=0.0)] = 0.0

    live_confirm_token: str = ""
    live_max_daily_loss: float = 0.03
    probability_calibration_method: str = "isotonic"
    enable_probability_calibration: bool = True
    tft_calibration_max_samples: Annotated[int, Field(ge=10)] = 32
    tft_calibration_holdout_ratio: Annotated[float, Field(ge=0.05, le=0.5)] = 0.2
    hard_stop_max_loss: Annotated[float, Field(ge=0.0, le=1.0)] = 0.03

    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    watchtower_db_path: Path = Path("./logs/watchtower.db")
    dashboard_refresh_seconds: Annotated[int, Field(ge=1, le=300)] = 10
    dashboard_target_start: str = "2017-01-01"
    fetch_log_every_batches: Annotated[int, Field(ge=1, le=500)] = 1
    fetch_checkpoint_every_batches: Annotated[int, Field(ge=0, le=500)] = 25

    @property
    def symbols_list(self) -> list[str]:
        """Parse SYMBOLS comma-separated list."""
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    @property
    def macro_event_weekdays(self) -> list[int]:
        """Parse weekday list for macro event crisis windows (0=Mon ... 6=Sun)."""
        out: list[int] = []
        for raw in self.macro_event_crisis_weekdays.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                day = int(raw)
            except ValueError:
                continue
            if 0 <= day <= 6:
                out.append(day)
        return out or [0, 1, 2, 3, 4]

    @property
    def macro_event_windows(self) -> list[tuple[time, time]]:
        """Parse UTC windows from HH:MM-HH:MM comma-separated format."""
        windows: list[tuple[time, time]] = []
        for part in self.macro_event_crisis_windows_utc.split(","):
            part = part.strip()
            if not part or "-" not in part:
                continue
            left, right = part.split("-", 1)
            try:
                sh, sm = left.strip().split(":")
                eh, em = right.strip().split(":")
                start = time(hour=int(sh), minute=int(sm))
                end = time(hour=int(eh), minute=int(em))
                windows.append((start, end))
            except Exception:
                continue
        return windows

    @property
    def telegram_allowed_chat_ids_list(self) -> list[str]:
        """Parse TELEGRAM_ALLOWED_CHAT_IDS comma-separated list (keep as strings)."""
        return [s.strip() for s in self.telegram_allowed_chat_ids.split(",") if s.strip()]

    @property
    def telegram_sentiment_chat_ids_list(self) -> list[str]:
        """Parse TELEGRAM_SENTIMENT_CHAT_IDS comma-separated list."""
        return [s.strip() for s in self.telegram_sentiment_chat_ids.split(",") if s.strip()]

    @property
    def fees_decimal(self) -> float:
        """Convert basis points to decimal (10 bps -> 0.001)."""
        return self.fees_bps / 10_000

    @property
    def execution_cost_bps(self) -> int:
        """Total non-fee execution costs in basis points."""
        return self.spread_bps + self.slippage_bps

    def ensure_dirs(self) -> None:
        """Create data/model/log directories if missing."""
        for directory in (
            self.data_dir_raw,
            self.data_dir_processed,
            self.models_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
