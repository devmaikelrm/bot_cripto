from __future__ import annotations

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.jobs.common import build_version_dir, load_feature_dataset_for_training, write_model_metadata
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.tft import TFTPredictor
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.notifications.telegram import TelegramNotifier

logger = get_logger("jobs.train_trend")


def run(symbol: str | None = None, timeframe: str | None = None) -> str:
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    notifier = TelegramNotifier(settings=settings)
    job_name = f"train-trend:{target}:{tf}"
    notifier.notify_job_start(job_name)

    try:
        df = load_feature_dataset_for_training(
            settings,
            target,
            timeframe=tf,
            prefer_triple_barrier=True,
        )

        # If triple-barrier labels are present, train trend objective directly on them.
        if "tb_label" in df.columns:
            model = BaselineModel(objective="trend")
            logger.info("train_trend_using_triple_barrier_labels", symbol=target, timeframe=tf)
        else:
            model = TFTPredictor()  # Fallback to TFT when TB labels are unavailable.
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
        notifier.notify_job_end(job_name, status="ok")
        return str(out_dir)
    except Exception as exc:
        notifier.notify_error(job_name, str(exc))
        logger.exception("train_trend_failed", symbol=target, timeframe=tf, error=str(exc))
        raise


if __name__ == "__main__":
    print(run())
