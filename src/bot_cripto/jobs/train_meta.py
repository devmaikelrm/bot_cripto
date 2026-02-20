from __future__ import annotations

import json

import pandas as pd

from bot_cripto.backtesting.meta_cpcv import run_meta_cpcv_validation
from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.jobs.common import (
    build_version_dir,
    load_feature_dataset_for_training,
    write_model_metadata,
)
from bot_cripto.models.base import ModelMetadata, PredictionOutput
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.meta import MetaModel
from bot_cripto.monitoring.meta_metrics_store import MetaMetricsStore
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.notifications.telegram import TelegramNotifier

logger = get_logger("jobs.train_meta")


def _infer_regime_from_row(row: pd.Series, pred: PredictionOutput, adx_thr: float) -> str:
    adx = float(row.get("adx", 0.0))
    if adx >= adx_thr and pred.expected_return >= 0:
        return "BULL_TREND"
    if adx >= adx_thr and pred.expected_return < 0:
        return "BEAR_TREND"
    return "RANGE_SIDEWAYS"


def _build_meta_dataset(
    df: pd.DataFrame,
    primary_model: BaselineModel,
    split_idx: int,
    edge_return: float,
    adx_thr: float,
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[list[float]] = []
    labels: list[int] = []
    close = df["close"].astype(float)
    has_tb_label = "tb_label" in df.columns
    feature_builder = MetaModel()

    for idx in range(split_idx, len(df) - 1):
        window = df.iloc[: idx + 1]
        pred = primary_model.predict(window)
        now_row = df.iloc[idx]
        regime = _infer_regime_from_row(now_row, pred, adx_thr=adx_thr)

        if has_tb_label:
            tb = int(df["tb_label"].iloc[idx])
            if tb == 0:
                continue
            pred_dir = 1 if pred.prob_up >= 0.5 else -1
            y = int(pred_dir == tb)
        else:
            c0 = float(close.iloc[idx])
            c1 = float(close.iloc[idx + 1])
            realized = (c1 - c0) / c0 if c0 != 0 else 0.0
            if abs(realized) < edge_return:
                continue
            pred_dir = 1 if pred.prob_up >= 0.5 else -1
            real_dir = 1 if realized >= 0 else -1
            y = int(pred_dir == real_dir)

        feature = feature_builder._prepare_meta_features(
            tft_pred={
                "prob_up": float(pred.prob_up),
                "expected_return": float(pred.expected_return),
                "risk_score": float(pred.risk_score),
                "confidence": float(abs(pred.prob_up - 0.5) * 2.0),
            },
            regime_str=regime,
            quant_signals={
                "funding_rate": float(now_row.get("funding_rate", 0.0)),
                "fear_greed": float(now_row.get("fear_greed", 0.5)),
                "orderbook_imbalance": float(now_row.get("orderbook_imbalance", 0.0)),
                "social_sentiment": float(now_row.get("social_sentiment", 0.5)),
                "social_sentiment_anomaly": float(now_row.get("social_sentiment_anomaly", 0.0)),
                "macro_risk_off_score": float(now_row.get("macro_risk_off_score", 0.5)),
                "corr_btc_sp500": float(now_row.get("corr_btc_sp500", 0.0)),
                "corr_btc_dxy": float(now_row.get("corr_btc_dxy", 0.0)),
                "volatility": float(now_row.get("volatility", 0.0)),
                "rel_vol": float(now_row.get("rel_vol", 1.0)),
                "adx": float(now_row.get("adx", 0.0)),
            },
        )[0]
        rows.append(feature.tolist())
        labels.append(y)

    x = pd.DataFrame(rows, columns=list(MetaModel.FEATURE_COLUMNS))
    y = pd.Series(labels, dtype=int)
    return x, y


def run(symbol: str | None = None, timeframe: str | None = None) -> str:
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    notifier = TelegramNotifier(settings=settings)
    job_name = f"train-meta:{target}:{tf}"
    notifier.notify_job_start(job_name)

    try:
        df = load_feature_dataset_for_training(
            settings=settings,
            symbol=target,
            timeframe=tf,
            prefer_triple_barrier=True,
        )
        if "close" not in df.columns:
            raise ValueError("Meta training requires 'close' column")
        if len(df) < 400:
            raise ValueError("Meta training requires at least 400 rows")

        split = int(len(df) * 0.70)
        split = max(split, 250)
        split = min(split, len(df) - 20)
        if split <= 200:
            raise ValueError("Not enough data for stable train/validation meta split")

        primary = BaselineModel(objective="trend")
        primary.train(df.iloc[:split], target_col="close")

        x_meta, y_meta = _build_meta_dataset(
            df=df,
            primary_model=primary,
            split_idx=split,
            edge_return=float(settings.label_edge_return),
            adx_thr=float(settings.regime_adx_trend_min),
        )
        if len(x_meta) < 100:
            raise ValueError("Insufficient meta samples; need >= 100")
        if int(y_meta.nunique()) < 2:
            raise ValueError("Meta labels have a single class; cannot train classifier")
        if settings.meta_model_threshold_min > settings.meta_model_threshold_max:
            raise ValueError("META_MODEL_THRESHOLD_MIN must be <= META_MODEL_THRESHOLD_MAX")

        x_meta = MetaModel.ensure_feature_columns(x_meta)

        holdout_ratio = float(settings.meta_model_holdout_ratio)
        val_size = max(30, int(len(x_meta) * holdout_ratio))
        val_size = min(val_size, len(x_meta) - 20)
        train_size = len(x_meta) - val_size
        if train_size < 50:
            raise ValueError("Meta training split too small")

        x_train = x_meta.iloc[:train_size]
        y_train = y_meta.iloc[:train_size]
        x_val = x_meta.iloc[train_size:]
        y_val = y_meta.iloc[train_size:]

        meta_model = MetaModel(min_prob_success=settings.meta_model_min_prob_success)
        meta_model.fit(x_train, y_train)
        if not meta_model.is_fitted:
            raise ValueError("Meta model did not fit (low samples or single-class labels)")

        train_pred = meta_model.model.predict(x_train)
        train_acc = float((train_pred == y_train.to_numpy()).mean())
        val_probs = meta_model.predict_success_prob_batch(x_val)
        best_thr = MetaModel.optimize_threshold(
            probs=val_probs,
            labels=y_val.to_numpy(dtype=int),
            threshold_min=float(settings.meta_model_threshold_min),
            threshold_max=float(settings.meta_model_threshold_max),
            threshold_step=float(settings.meta_model_threshold_step),
            min_positive_predictions=int(settings.meta_model_min_positive_predictions),
        )
        meta_model.min_prob_success = float(best_thr["threshold"])
        success_rate = float(y_meta.mean())

        cpcv_report = run_meta_cpcv_validation(
            x_meta=x_meta,
            y_meta=y_meta,
            n_groups=6,
            n_test_groups=2,
            purge_size=5,
            embargo_size=5,
            threshold_min=float(settings.meta_model_threshold_min),
            threshold_max=float(settings.meta_model_threshold_max),
            threshold_step=float(settings.meta_model_threshold_step),
            min_positive_predictions=int(settings.meta_model_min_positive_predictions),
        )

        metrics = {
            "samples_meta": float(len(x_meta)),
            "samples_meta_train": float(len(x_train)),
            "samples_meta_val": float(len(x_val)),
            "label_success_rate": success_rate,
            "accuracy_train": train_acc,
            "threshold_opt": float(best_thr["threshold"]),
            "val_precision": float(best_thr["precision"]),
            "val_recall": float(best_thr["recall"]),
            "val_f1": float(best_thr["f1"]),
            "val_accuracy": float(best_thr["accuracy"]),
            "cpcv_f1_mean": float(cpcv_report.f1_mean),
            "cpcv_f1_p5": float(cpcv_report.f1_p5),
            "cpcv_precision_mean": float(cpcv_report.precision_mean),
            "cpcv_recall_mean": float(cpcv_report.recall_mean),
            "cpcv_accuracy_mean": float(cpcv_report.accuracy_mean),
        }
        metadata = ModelMetadata.create(
            model_type="meta_rf",
            version="0.1.0",
            metrics=metrics,
        )

        out_dir = build_version_dir(settings, "meta", target, metadata, timeframe=tf)
        meta_model.save(out_dir)
        write_model_metadata(out_dir, metadata)
        cpcv_payload = {
            "symbol": target,
            "timeframe": tf,
            "trained_at": metadata.trained_at,
            "summary": {
                "f1_mean": cpcv_report.f1_mean,
                "f1_p5": cpcv_report.f1_p5,
                "precision_mean": cpcv_report.precision_mean,
                "recall_mean": cpcv_report.recall_mean,
                "accuracy_mean": cpcv_report.accuracy_mean,
            },
            "fold_results": [f.__dict__ for f in cpcv_report.fold_results],
        }
        cpcv_path = out_dir / "meta_cpcv_report.json"
        cpcv_path.write_text(json.dumps(cpcv_payload, indent=2), encoding="utf-8")
        WatchtowerStore(settings.watchtower_db_path).log_training_metrics(
            ts=metadata.trained_at,
            model_name=f"meta:{target}:{tf}",
            metrics=metadata.metrics,
        )
        history_store = MetaMetricsStore(settings.logs_dir / "meta_metrics_history.json")
        prev_rows = history_store.records(symbol=target, timeframe=tf)
        prev_f1 = float(prev_rows[-1].get("val_f1", 0.0)) if prev_rows else None
        delta_f1 = None if prev_f1 is None else float(metrics["val_f1"] - prev_f1)
        history_store.append(
            {
                "ts": metadata.trained_at,
                "symbol": target,
                "timeframe": tf,
                **metrics,
                "val_f1_delta_prev": delta_f1,
            }
        )

        logger.info(
            "train_meta_done",
            symbol=target,
            timeframe=tf,
            output=str(out_dir),
            metrics=metrics,
            cpcv_report=str(cpcv_path),
        )
        notifier.notify_job_end(job_name, status="ok")
        return str(out_dir)
    except Exception as exc:
        notifier.notify_error(job_name, str(exc))
        logger.exception("train_meta_failed", symbol=target, timeframe=tf, error=str(exc))
        raise


if __name__ == "__main__":
    print(run())
