from __future__ import annotations

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.jobs.common import build_version_dir, load_feature_dataset, write_model_metadata
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.monitoring.watchtower_store import WatchtowerStore

logger = get_logger("jobs.train_trend")


def run(symbol: str | None = None, timeframe: str | None = None) -> str:
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    df = load_feature_dataset(settings, target, timeframe=tf)

    model = BaselineModel(objective="trend")
    metadata = model.train(df, target_col="close")
    out_dir = build_version_dir(settings, "trend", target, metadata, timeframe=tf)
    model.save(out_dir)
    write_model_metadata(out_dir, metadata)
    WatchtowerStore(settings.watchtower_db_path).log_training_metrics(
        ts=metadata.trained_at,
        model_name=f"trend:{target}:{tf}",
        metrics=metadata.metrics,
    )

    logger.info(
        "train_trend_done",
        symbol=target,
        timeframe=tf,
        output=str(out_dir),
        metrics=metadata.metrics,
    )
    return str(out_dir)


if __name__ == "__main__":
    print(run())
