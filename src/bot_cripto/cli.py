"""CLI entrypoint for Bot Cripto."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

import pandas as pd
import typer

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import setup_logging

app = typer.Typer(name="bot-cripto", help="Crypto market prediction system", add_completion=False)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@app.callback()
def main(log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level")) -> None:
    setup_logging(level=log_level)


@app.command()
def info() -> None:
    settings = get_settings()
    typer.echo("=" * 50)
    typer.echo("Bot Cripto - Active Configuration")
    typer.echo("=" * 50)
    typer.echo(f"Exchange: {settings.exchange}")
    typer.echo(f"Symbols: {settings.symbols}")
    typer.echo(f"Timeframe: {settings.timeframe}")
    typer.echo(f"Horizon: {settings.pred_horizon_steps}")
    typer.echo(f"Paper mode: {settings.paper_mode}")
    typer.echo(f"Live mode: {settings.live_mode}")
    typer.echo(f"Risk max: {settings.risk_max}")
    typer.echo(f"Prob min: {settings.prob_min}")
    typer.echo(f"Min expected return: {settings.min_expected_return}")
    typer.echo(f"Fees bps: {settings.fees_bps} ({settings.fees_decimal:.4f})")
    typer.echo(f"Data raw: {settings.data_dir_raw}")
    typer.echo(f"Data processed: {settings.data_dir_processed}")
    typer.echo(f"Models dir: {settings.models_dir}")
    typer.echo(f"Logs dir: {settings.logs_dir}")
    tg_status = "configured" if settings.telegram_bot_token else "missing token"
    typer.echo(f"Telegram: {tg_status}")


@app.command("architecture-status")
def architecture_status(
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated symbols to classify by layer routing, e.g. BTC/USDT,EUR/USD",
    ),
) -> None:
    """Show current layered architecture routing (data/model/risk/execution domains)."""
    from bot_cripto.core.market import market_domain

    settings = get_settings()
    target = _split_csv(symbols) or settings.symbols_list
    out_rows: list[dict[str, str]] = []
    for sym in target:
        domain = market_domain(sym)
        out_rows.append(
            {
                "symbol": sym,
                "market_domain": domain,
                "data_layer": "crypto_feeds" if domain == "crypto" else "forex_feeds",
                "model_layer": "models.crypto" if domain == "crypto" else "models.forex",
                "portfolio_layer": "portfolio.crypto_risk" if domain == "crypto" else "portfolio.forex_risk",
                "execution_layer": "execution_router",
            }
        )

    typer.echo(json.dumps({"symbols": out_rows}, indent=2))


@app.command()
def fetch(
    days: int = typer.Option(30, help="Days to download"),
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    provider: str | None = typer.Option(
        None, help="Data provider: binance|bybit|coinbase|kraken|okx|yfinance (overrides DATA_PROVIDER)"
    ),
) -> None:
    from bot_cripto.data.ingestion import BinanceFetcher

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    target_provider = provider or settings.data_provider
    effective_settings = settings.model_copy(
        update={"timeframe": target_timeframe, "data_provider": target_provider}
    )

    fetcher = BinanceFetcher(effective_settings)
    df = fetcher.fetch_history(
        target_symbol,
        target_timeframe,
        days=days,
        log_every_batches=effective_settings.fetch_log_every_batches,
        checkpoint_every_batches=effective_settings.fetch_checkpoint_every_batches,
    )
    if df.empty:
        typer.echo("No data downloaded")
        raise typer.Exit(code=1)

    # Capture real-time microstructure snapshot (always, regardless of days)
    obi = fetcher.fetch_order_book_imbalance(target_symbol)
    whale_p = fetcher.fetch_whale_pressure(target_symbol)

    sentiment_score = 0.0
    try:
        from bot_cripto.data.sentiment import SentimentFetcher
        sent_fetcher = SentimentFetcher(effective_settings)
        sentiment_score = sent_fetcher.fetch_sentiment(target_symbol.split("/")[0])
    except Exception:
        pass

    fetcher.save_microstructure_snapshot(target_symbol, obi, whale_p, sentiment_score)
    typer.echo(f"Micro snapshot: OBI={obi:.4f}, Whale={whale_p:.2f}, Sent={sentiment_score:.2f}")

    path = fetcher.save_data(df, target_symbol, target_timeframe)
    typer.echo(f"Saved raw data: {path}")
    typer.echo(f"Rows: {len(df)}")


@app.command("fetch-batch")
def fetch_batch(
    days: int = typer.Option(30, help="Days to download"),
    symbols: str | None = typer.Option(
        None, help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT (defaults to SYMBOLS)"
    ),
    timeframes: str | None = typer.Option(
        None, help="Comma-separated timeframes, e.g. 5m,15m,1h (defaults to TIMEFRAME)"
    ),
    provider: str | None = typer.Option(
        None, help="Data provider: binance|bybit|coinbase|kraken|okx|yfinance (overrides DATA_PROVIDER)"
    ),
) -> None:
    """Download multiple symbols/timeframes sequentially."""
    from bot_cripto.data.ingestion import BinanceFetcher

    settings = get_settings()
    target_symbols = _split_csv(symbols) or settings.symbols_list
    target_timeframes = _split_csv(timeframes) or [settings.timeframe]
    target_provider = provider or settings.data_provider

    if not target_symbols:
        typer.echo("No symbols provided")
        raise typer.Exit(code=2)

    for sym in target_symbols:
        for tf in target_timeframes:
            typer.echo(f"Fetching {sym} {tf} provider={target_provider} days={days} ...")
            effective_settings = settings.model_copy(update={"timeframe": tf, "data_provider": target_provider})
            fetcher = BinanceFetcher(effective_settings)
            df = fetcher.fetch_history(
                sym,
                tf,
                days=days,
                log_every_batches=effective_settings.fetch_log_every_batches,
                checkpoint_every_batches=effective_settings.fetch_checkpoint_every_batches,
            )
            if df.empty:
                typer.echo(f"No data downloaded for {sym} {tf}")
                continue
            path = fetcher.save_data(df, sym, tf)
            typer.echo(f"Saved raw data: {path} rows={len(df)}")


@app.command("validate-data")
def validate_data(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 1h, 5m"),
    days: int = typer.Option(7, help="Days of history to compare"),
    providers: str = typer.Option(
        "binance,coinbase,kraken",
        help="Comma-separated exchanges to cross-check",
    ),
    mad_threshold: float = typer.Option(3.5, help="MAD z-score threshold for outlier detection"),
) -> None:
    """Cross-exchange OHLCV validation with outlier detection."""
    from bot_cripto.data.aggregator import RobustDataAggregator

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_tf = timeframe or settings.timeframe
    prov_list = _split_csv(providers) or ["binance", "coinbase", "kraken"]

    typer.echo(f"Validating {target_symbol} {target_tf} across {prov_list} ({days}d) ...")

    agg = RobustDataAggregator(
        providers=prov_list,
        mad_threshold=mad_threshold,
    )
    report = agg.fetch_and_validate(target_symbol, target_tf, days=days)

    typer.echo(f"\nProviders OK: {report.providers_ok}/{report.providers_requested}")
    if report.providers_failed:
        typer.echo(f"Failed: {report.providers_failed}")

    typer.echo(f"Total bars: {report.total_bars}")
    typer.echo(f"Outlier bars: {report.outlier_bars} ({report.outlier_ratio:.2%})")
    typer.echo(f"Consensus rows: {len(report.consensus_df)}")

    if report.outliers:
        typer.echo(f"\nTop outliers (max 20):")
        for o in report.outliers[:20]:
            typer.echo(
                f"  {o.timestamp} | {o.column:5s} | {o.provider:10s} | "
                f"value={o.value:.2f} median={o.median:.2f} z={o.z_score:.1f}"
            )

    for prov, info in report.per_source.items():
        status = "OK" if info["ok"] else f"FAIL: {info['error']}"
        typer.echo(f"  {prov:12s}: {info['rows']:>6} rows  [{status}]")

    if report.outlier_ratio > 0.05:
        typer.echo("\nWARNING: >5% outlier bars detected — data quality suspect")
    elif report.providers_ok >= 2:
        typer.echo("\nData quality: GOOD (cross-exchange consensus)")
    else:
        typer.echo("\nData quality: UNCHECKED (single source only)")


@app.command("fetch-macro")
def fetch_macro(
    days: int = typer.Option(60, help="Days to download"),
    tickers: str = typer.Option("SPY,DX-Y.NYB", help="Comma-separated tickers"),
) -> None:
    """Download macro data (SPY, DXY) for correlation analysis."""
    from bot_cripto.data.macro import MacroFetcher

    settings = get_settings()
    fetcher = MacroFetcher(settings)
    ticker_list = _split_csv(tickers)
    fetcher.fetch_macro_data(tickers=ticker_list, days=days)
    typer.echo("Macro data download complete")


@app.command("stream-capture")
def stream_capture(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    duration: int = typer.Option(60, help="Capture duration in seconds"),
    source: str = typer.Option("cryptofeed", help="Stream source: cryptofeed|poll"),
    snapshot_every: int | None = typer.Option(None, help="Snapshot period seconds"),
) -> None:
    """Capture realtime microstructure snapshots into parquet."""
    from bot_cripto.data.streaming import RealtimeStreamCollector

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    collector = RealtimeStreamCollector(settings)
    path = collector.capture(
        symbol=target_symbol,
        duration_seconds=duration,
        source=source,
        snapshot_every_seconds=snapshot_every,
    )
    typer.echo(f"Stream saved: {path}")


@app.command("fetch-sentiment")
def fetch_sentiment(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    source: str | None = typer.Option(
        None,
        help=(
            "Source: auto|blend|nlp|api|x|telegram|gnews|reddit|cryptopanic|rss|local "
            "(overrides SOCIAL_SENTIMENT_SOURCE)"
        ),
    ),
) -> None:
    """Fetch social sentiment score and print normalized values."""
    from bot_cripto.data.quant_signals import QuantSignalFetcher

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    effective_settings = (
        settings.model_copy(update={"social_sentiment_source": source}) if source else settings
    )
    fetcher = QuantSignalFetcher(effective_settings)
    score01 = fetcher.fetch_social_sentiment(target_symbol)
    score11 = (score01 * 2.0) - 1.0

    typer.echo(f"Symbol: {target_symbol}")
    typer.echo(f"Source: {effective_settings.social_sentiment_source}")
    typer.echo(f"Score [0,1]: {score01:.4f}")
    typer.echo(f"Score [-1,1]: {score11:.4f}")


@app.command("fetch-sentiment-nlp")
def fetch_sentiment_nlp(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
) -> None:
    """Force NLP sentiment path for quick validation."""
    from bot_cripto.data.quant_signals import QuantSignalFetcher

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    effective_settings = settings.model_copy(update={"social_sentiment_source": "nlp"})

    fetcher = QuantSignalFetcher(effective_settings)
    score01 = fetcher.fetch_social_sentiment(target_symbol)
    score11 = (score01 * 2.0) - 1.0

    typer.echo(f"Symbol: {target_symbol}")
    typer.echo("Source: nlp")
    typer.echo(f"Score [0,1]: {score01:.4f}")
    typer.echo(f"Score [-1,1]: {score11:.4f}")


@app.command("api-smoke")
def api_smoke(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 1h"),
) -> None:
    """Run connector smoke checks and persist a report in logs/."""
    from bot_cripto.ops.api_smoke import run_api_smoke

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    out = run_api_smoke(settings=settings, symbol=target_symbol, timeframe=target_timeframe)
    typer.echo(json.dumps(out, indent=2))


@app.command("dual-pipeline")
def dual_pipeline(
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated symbols, e.g. BTC/USDT,EUR/USD (defaults to SYMBOLS)",
    ),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 1h"),
    days: int = typer.Option(30, help="Days for data fetch when fetch is enabled"),
    include_meta: bool = typer.Option(True, help="Run train-meta inside the pipeline"),
    skip_fetch: bool = typer.Option(False, help="Skip market data fetch stage"),
    skip_train: bool = typer.Option(False, help="Skip model training stage"),
    skip_inference: bool = typer.Option(False, help="Skip inference stage"),
) -> None:
    """Run end-to-end pipeline for mixed crypto+forex symbols in one command."""
    from bot_cripto.ops.dual_pipeline import run_dual_pipeline

    settings = get_settings()
    out = run_dual_pipeline(
        settings=settings,
        symbols=_split_csv(symbols) or settings.symbols_list,
        timeframe=timeframe or settings.timeframe,
        days=days,
        include_meta=include_meta,
        skip_fetch=skip_fetch,
        skip_train=skip_train,
        skip_inference=skip_inference,
    )
    typer.echo(json.dumps(out, indent=2))


@app.command()
def features(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.features.engineering import FeaturePipeline

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")

    input_path = settings.data_dir_raw / f"{safe_symbol}_{target_timeframe}.parquet"
    output_path = (
        settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    )

    if not input_path.exists():
        typer.echo(f"Raw file not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)
    processed_df = FeaturePipeline().transform(df)
    if processed_df.empty:
        typer.echo("Feature dataset is empty")
        raise typer.Exit(code=1)

    settings.ensure_dirs()
    processed_df.to_parquet(output_path, compression="snappy")
    typer.echo(f"Saved features: {output_path}")
    typer.echo(f"Rows: {len(processed_df)} cols: {processed_df.shape[1]}")


@app.command("build-triple-barrier-labels")
def build_triple_barrier_labels_cmd(
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 1h"),
    pt_mult: float = typer.Option(2.0, help="Profit-take volatility multiplier"),
    sl_mult: float = typer.Option(2.0, help="Stop-loss volatility multiplier"),
    horizon_bars: int = typer.Option(20, help="Vertical barrier in bars"),
    vol_span: int = typer.Option(100, help="EWM volatility span"),
) -> None:
    """Build triple-barrier labels on top of processed feature dataset."""
    from bot_cripto.labels.triple_barrier import TripleBarrierConfig, build_triple_barrier_labels

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")
    input_path = settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    output_path = settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features_tb.parquet"
    if not input_path.exists():
        typer.echo(f"Feature dataset not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)
    cfg = TripleBarrierConfig(
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        horizon_bars=horizon_bars,
        vol_span=vol_span,
    )
    labeled = build_triple_barrier_labels(df, price_col="close", config=cfg)
    labeled.to_parquet(output_path, compression="snappy")
    counts = labeled["tb_label"].value_counts().to_dict() if "tb_label" in labeled.columns else {}
    typer.echo(f"Saved labeled dataset: {output_path}")
    typer.echo(json.dumps({"label_counts": counts}, indent=2))


@app.command()
def train(
    symbol: str | None = typer.Option(None, help="Pair"),
    model_type: str = typer.Option("baseline", help="Model type: baseline|tft"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    checkpoint: str | None = typer.Option(None, help="Path to checkpoint to resume training"),
    start_date: str | None = typer.Option(None, help="Filter data from this date (e.g. 2024-01-01)"),
) -> None:
    from bot_cripto.jobs.common import write_model_metadata
    from bot_cripto.models.base import BasePredictor
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")
    input_path = (
        settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    )
    model_path = settings.models_dir / model_type / safe_symbol

    if not input_path.exists():
        typer.echo(f"Feature dataset not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)

    if start_date:
        try:
            initial_rows = len(df)
            df = df[df.index >= start_date]
            typer.echo(f"Filtered data from {start_date}: {initial_rows} -> {len(df)} rows")
        except Exception as e:
            typer.echo(f"Error filtering by date: {e}")
            raise typer.Exit(code=1)

    model: BasePredictor
    if model_type == "baseline":
        model = BaselineModel()
    elif model_type == "tft":
        from bot_cripto.models.tft import TFTPredictor

        model = TFTPredictor()
    elif model_type == "nbeats":
        from bot_cripto.models.nbeats import NBeatsPredictor

        model = NBeatsPredictor()
    else:
        typer.echo(f"Unsupported model type: {model_type}")
        raise typer.Exit(code=1)

    # Pass checkpoint if provided
    if checkpoint and hasattr(model, "train_with_checkpoint"):
        metadata = model.train_with_checkpoint(df, checkpoint, target_col="close")
    else:
        metadata = model.train(df, target_col="close")
    
    model.save(model_path)
    write_model_metadata(model_path, metadata)
    typer.echo(f"Model saved: {model_path}")
    typer.echo(json.dumps(metadata.metrics, indent=2))


@app.command()
def predict(
    symbol: str | None = typer.Option(None, help="Pair"),
    model_type: str = typer.Option("baseline", help="Model type"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.decision.engine import DecisionEngine
    from bot_cripto.models.base import BasePredictor
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")

    input_path = (
        settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    )
    model_path = settings.models_dir / model_type / safe_symbol

    if not input_path.exists() or not model_path.exists():
        typer.echo("Missing features dataset or model")
        raise typer.Exit(code=1)

    model: BasePredictor
    if model_type == "baseline":
        model = BaselineModel()
    elif model_type == "tft":
        from bot_cripto.models.tft import TFTPredictor

        model = TFTPredictor()
    elif model_type == "nbeats":
        from bot_cripto.models.nbeats import NBeatsPredictor

        model = NBeatsPredictor()
    else:
        typer.echo(f"Unsupported model type: {model_type}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)
    model.load(model_path)
    prediction = model.predict(df)
    signal = DecisionEngine().decide(prediction, current_price=float(df["close"].iloc[-1]))

    typer.echo(f"Decision: {signal.action.value} conf={signal.confidence:.2f}")
    typer.echo(f"Reason: {signal.reason}")
    typer.echo(json.dumps(prediction.to_dict(), indent=2))


@app.command("train-trend")
def train_trend(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.jobs.train_trend import run

    typer.echo(run(symbol=symbol, timeframe=timeframe))


@app.command("train-return")
def train_return(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.jobs.train_return import run

    typer.echo(run(symbol=symbol, timeframe=timeframe))


@app.command("train-risk")
def train_risk(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.jobs.train_risk import run

    typer.echo(run(symbol=symbol, timeframe=timeframe))


@app.command("train-nbeats")
def train_nbeats(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    """Train N-BEATS time-series model."""
    from bot_cripto.jobs.train_nbeats import run

    typer.echo(run(symbol=symbol, timeframe=timeframe))


@app.command("train-meta")
def train_meta(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    """Train meta-labeling filter model on primary signal correctness."""
    from bot_cripto.jobs.train_meta import run

    typer.echo(run(symbol=symbol, timeframe=timeframe))


@app.command("meta-metrics-report")
def meta_metrics_report(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    window: int = typer.Option(10, help="Last N records to summarize"),
) -> None:
    """Summarize historical meta-model metrics and degradation trend."""
    from bot_cripto.monitoring.meta_metrics_store import MetaMetricsStore

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_tf = timeframe or settings.timeframe
    store = MetaMetricsStore(settings.logs_dir / "meta_metrics_history.json")
    rows = store.records(symbol=target_symbol, timeframe=target_tf)
    if not rows:
        typer.echo("No meta metrics history for this symbol/timeframe")
        raise typer.Exit(code=2)

    tail = rows[-max(1, int(window)) :]
    f1_vals = [float(r.get("val_f1", 0.0)) for r in tail]
    cpcv_f1_vals = [float(r.get("cpcv_f1_mean", 0.0)) for r in tail]
    out = {
        "symbol": target_symbol,
        "timeframe": target_tf,
        "records_total": len(rows),
        "window": len(tail),
        "last_ts": str(tail[-1].get("ts", "")),
        "latest": tail[-1],
        "summary": {
            "val_f1_mean_window": float(sum(f1_vals) / len(f1_vals)),
            "val_f1_latest": float(f1_vals[-1]),
            "val_f1_min_window": float(min(f1_vals)),
            "cpcv_f1_mean_window": float(sum(cpcv_f1_vals) / len(cpcv_f1_vals)),
            "cpcv_f1_latest": float(cpcv_f1_vals[-1]),
            "degrading_vs_prev": bool(float(tail[-1].get("val_f1_delta_prev", 0.0)) < 0.0),
        },
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("run-inference")
def run_inference(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
) -> None:
    from bot_cripto.jobs.inference import run

    typer.echo(json.dumps(run(symbol=symbol, timeframe=timeframe), indent=2))


@app.command("backtest")
def backtest(
    symbol: str | None = typer.Option(None, help="Pair"),
    folds: int = typer.Option(4, help="Walk-forward folds (ignored when --train-size set)"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    train_size: int | None = typer.Option(None, help="Training window in candles (e.g. 25920 for 90d@5m)"),
    test_size: int | None = typer.Option(None, help="Test window in candles (e.g. 8640 for 30d@5m)"),
    step_size: int | None = typer.Option(None, help="Step between folds in candles (e.g. 4320 for 15d@5m)"),
    anchored: bool = typer.Option(True, help="Anchored (expanding) window vs rolling"),
) -> None:
    from bot_cripto.backtesting.walk_forward import WalkForwardBacktester
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")
    input_path = (
        settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    )
    if not input_path.exists():
        typer.echo(f"Feature dataset not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)
    report = WalkForwardBacktester(
        n_folds=folds,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=anchored,
    ).run(df, model_factory=BaselineModel)

    # Build serializable output (fold_results contains dataclasses)
    out = {k: v for k, v in report.__dict__.items() if k != "fold_results"}
    out["fold_results"] = [f.__dict__ for f in report.fold_results]
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("backtest-purged-cv")
def backtest_purged_cv(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    splits: int = typer.Option(5, help="Number of purged CV folds"),
    purge_size: int = typer.Option(5, help="Bars purged before test fold"),
    embargo_size: int = typer.Option(5, help="Bars embargoed after test fold"),
    use_tb_dataset: bool = typer.Option(
        True,
        help="Prefer *_features_tb.parquet when available",
    ),
) -> None:
    """Run purged temporal cross-validation (anti-leakage robustness check)."""
    from bot_cripto.backtesting.purged_cv import run_purged_cv_backtest
    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=use_tb_dataset,
    )

    report = run_purged_cv_backtest(
        df=df,
        model_factory=BaselineModel,
        target_col="close",
        n_splits=splits,
        purge_size=purge_size,
        embargo_size=embargo_size,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
    )
    out = {k: v for k, v in report.__dict__.items() if k != "fold_results"}
    out["fold_results"] = [f.__dict__ for f in report.fold_results]
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("backtest-cpcv")
def backtest_cpcv(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    groups: int = typer.Option(6, help="Number of contiguous time groups"),
    test_groups: int = typer.Option(2, help="Number of groups used as test in each combination"),
    purge_size: int = typer.Option(5, help="Bars purged before each test group"),
    embargo_size: int = typer.Option(5, help="Bars embargoed after each test group"),
    use_tb_dataset: bool = typer.Option(
        True,
        help="Prefer *_features_tb.parquet when available",
    ),
) -> None:
    """Run combinatorial purged CV (CPCV-lite) and report distribution metrics."""
    from bot_cripto.backtesting.purged_cv import run_cpcv_backtest
    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=use_tb_dataset,
    )

    report = run_cpcv_backtest(
        df=df,
        model_factory=BaselineModel,
        target_col="close",
        n_groups=groups,
        n_test_groups=test_groups,
        purge_size=purge_size,
        embargo_size=embargo_size,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
    )

    out = {k: v for k, v in report.__dict__.items() if k != "fold_results"}
    out["fold_results"] = [f.__dict__ for f in report.fold_results]
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("phase1-kpi-report")
def phase1_kpi_report(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h"),
    folds: int = typer.Option(4, help="Walk-forward folds"),
    train_size: int | None = typer.Option(None, help="WF train window candles"),
    test_size: int | None = typer.Option(None, help="WF test window candles"),
    step_size: int | None = typer.Option(None, help="WF step candles"),
    anchored: bool = typer.Option(True, help="WF anchored vs rolling"),
    groups: int = typer.Option(6, help="CPCV contiguous groups"),
    test_groups: int = typer.Option(2, help="CPCV groups used as test"),
    purge_size: int = typer.Option(5, help="CPCV purge bars"),
    embargo_size: int = typer.Option(5, help="CPCV embargo bars"),
    use_tb_dataset: bool = typer.Option(True, help="Prefer *_features_tb.parquet when available"),
) -> None:
    """Generate unified Compass Phase 1 KPI report (WF efficiency + CPCV Sharpe dist)."""
    from bot_cripto.backtesting.phase1_kpi import (
        build_phase1_kpi_report,
        estimate_in_sample_sharpe,
    )
    from bot_cripto.backtesting.purged_cv import run_cpcv_backtest
    from bot_cripto.backtesting.walk_forward import WalkForwardBacktester
    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.models.baseline import BaselineModel

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=use_tb_dataset,
    )

    wf = WalkForwardBacktester(
        n_folds=folds,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=anchored,
    )
    wf_report = wf.run(df, model_factory=BaselineModel)

    cpcv_report = run_cpcv_backtest(
        df=df,
        model_factory=BaselineModel,
        target_col="close",
        n_groups=groups,
        n_test_groups=test_groups,
        purge_size=purge_size,
        embargo_size=embargo_size,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
    )

    split_idx = train_size or max(200, int(len(df) * 0.6))
    split_idx = min(split_idx, len(df))
    is_sharpe = estimate_in_sample_sharpe(
        df=df,
        model=BaselineModel(),
        train_size=split_idx,
        target_col="close",
        roundtrip_cost=(2 * settings.fees_bps + settings.spread_bps + settings.slippage_bps) / 10_000,
    )
    kpi = build_phase1_kpi_report(
        symbol=target_symbol,
        timeframe=target_timeframe,
        walk_forward_report=wf_report,
        cpcv_report=cpcv_report,
        in_sample_sharpe=is_sharpe,
    )

    ts_compact = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_symbol = target_symbol.replace("/", "_")
    out_file = settings.logs_dir / f"phase1_kpi_{safe_symbol}_{target_timeframe}_{ts_compact}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "symbol": target_symbol,
        "timeframe": target_timeframe,
        "walk_forward": {k: v for k, v in wf_report.__dict__.items() if k != "fold_results"},
        "cpcv": {k: v for k, v in cpcv_report.__dict__.items() if k != "fold_results"},
        "phase1_kpi": kpi.__dict__,
        "output_file": str(out_file),
    }
    out_file.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("benchmark-models")
def benchmark_models_cmd(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    models: str = typer.Option(
        "baseline,tft,nbeats,itransformer,patchtst",
        help="Comma-separated model families to compare",
    ),
    folds: int = typer.Option(4, help="Walk-forward folds"),
    train_size: int | None = typer.Option(None, help="Training window candles"),
    test_size: int | None = typer.Option(None, help="Test window candles"),
    step_size: int | None = typer.Option(None, help="Step candles"),
    anchored: bool = typer.Option(True, help="Anchored window vs rolling"),
) -> None:
    """Benchmark multiple model families with the same walk-forward setup."""
    from bot_cripto.backtesting.model_benchmark import benchmark_models, build_benchmark_summary
    from bot_cripto.backtesting.walk_forward import WalkForwardBacktester
    from bot_cripto.jobs.common import load_feature_dataset_for_training

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    model_list = _split_csv(models)
    if not model_list:
        typer.echo("No models provided")
        raise typer.Exit(code=2)

    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=True,
    )

    backtester = WalkForwardBacktester(
        n_folds=folds,
        fees_bps=settings.fees_bps,
        spread_bps=settings.spread_bps,
        slippage_bps=settings.slippage_bps,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=anchored,
    )
    results = benchmark_models(df=df, model_names=model_list, backtester=backtester)
    summary = build_benchmark_summary(results)
    ts_compact = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_symbol = target_symbol.replace("/", "_")
    out_file = settings.logs_dir / f"benchmark_{safe_symbol}_{target_timeframe}_{ts_compact}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "symbol": target_symbol,
        "timeframe": target_timeframe,
        "models": model_list,
        "summary": summary,
        "results": [r.__dict__ for r in results],
        "output_file": str(out_file),
    }
    out_file.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("phase2-sota-run")
def phase2_sota_run(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h"),
    models: str = typer.Option(
        "baseline,tft,nbeats,itransformer,patchtst",
        help="Comma-separated SOTA model families",
    ),
    train_frac: float = typer.Option(0.7, help="Train fraction for OOS evaluation"),
    use_tb_dataset: bool = typer.Option(True, help="Prefer *_features_tb.parquet when available"),
    strict_complete: bool = typer.Option(
        True,
        help="Fail command if any model is skipped or errors (enforces full GPU run)",
    ),
) -> None:
    """Run Phase 2 SOTA train+save+OOS benchmark and write final table artifacts."""
    from bot_cripto.backtesting.phase2_sota import run_phase2_sota, write_phase2_artifacts
    from bot_cripto.jobs.common import load_feature_dataset_for_training

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    model_list = _split_csv(models)
    if not model_list:
        typer.echo("No models provided")
        raise typer.Exit(code=2)

    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=use_tb_dataset,
    )
    rows, summary = run_phase2_sota(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        df=df,
        model_names=model_list,
        train_frac=train_frac,
    )
    files = write_phase2_artifacts(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        rows=rows,
        summary=summary,
    )

    out = {
        "symbol": target_symbol,
        "timeframe": target_timeframe,
        "summary": summary,
        "results": [r.__dict__ for r in rows],
        "artifacts": files,
    }
    typer.echo(json.dumps(out, indent=2, default=str))

    if strict_complete and not bool(summary.get("complete_success")):
        raise typer.Exit(code=2)


@app.command("tune-thresholds")
def tune_thresholds_cmd(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    train_frac: float = typer.Option(0.7, help="Training split fraction"),
    prob_grid: str = typer.Option("0.55,0.60,0.65,0.70", help="Comma-separated prob_min values"),
    return_grid: str = typer.Option("0.001,0.002,0.003,0.004", help="Comma-separated min_expected_return values"),
    apply_env: bool = typer.Option(False, help="Apply recommended thresholds into .env"),
    env_path: str = typer.Option(".env", help="Env file path to update when --apply-env is true"),
) -> None:
    """Grid-search decision thresholds from historical predictions."""
    from bot_cripto.backtesting.threshold_tuner import tune_thresholds
    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.models.baseline import BaselineModel
    from bot_cripto.ops.env_tools import apply_env_values, backup_env_file
    from pathlib import Path

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=True,
    )
    if len(df) < 300:
        typer.echo("Dataset too small for threshold tuning")
        raise typer.Exit(code=2)

    split = int(len(df) * train_frac)
    split = max(200, min(split, len(df) - 20))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:].copy()

    model = BaselineModel()
    model.train(train_df, target_col="close")

    rows: list[dict[str, float]] = []
    for i in range(split, len(df) - 1):
        window = df.iloc[: i + 1]
        pred = model.predict(window)
        c0 = float(df["close"].iloc[i])
        c1 = float(df["close"].iloc[i + 1])
        realized = (c1 - c0) / c0 if c0 != 0 else 0.0
        rows.append(
            {
                "prob_up": float(pred.prob_up),
                "expected_return": float(pred.expected_return),
                "realized_return": float(realized),
            }
        )
    pred_frame = pd.DataFrame(rows)
    if pred_frame.empty:
        typer.echo("No predictions generated for threshold tuning")
        raise typer.Exit(code=2)

    try:
        p_grid = [float(x) for x in _split_csv(prob_grid)]
        r_grid = [float(x) for x in _split_csv(return_grid)]
    except Exception:
        typer.echo("Invalid grid values")
        raise typer.Exit(code=2)

    if not p_grid or not r_grid:
        typer.echo("Empty threshold grids")
        raise typer.Exit(code=2)

    roundtrip_cost = (2 * settings.fees_bps + settings.spread_bps + settings.slippage_bps) / 10_000
    best, all_rows = tune_thresholds(
        pred_frame=pred_frame,
        prob_grid=p_grid,
        return_grid=r_grid,
        roundtrip_cost=roundtrip_cost,
    )

    out = {
        "symbol": target_symbol,
        "timeframe": target_timeframe,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "roundtrip_cost": roundtrip_cost,
        "current": {
            "prob_min": settings.prob_min,
            "min_expected_return": settings.min_expected_return,
        },
        "recommended": best.__dict__,
        "top5": [row.__dict__ for row in all_rows[:5]],
        "env_applied": False,
        "env_backup": None,
    }
    if apply_env:
        env_file = Path(env_path)
        backup = backup_env_file(env_file)
        apply_env_values(
            env_file,
            {
                "PROB_MIN": f"{best.prob_min:.6f}",
                "MIN_EXPECTED_RETURN": f"{best.min_expected_return:.6f}",
            },
        )
        out["env_applied"] = True
        out["env_backup"] = str(backup) if backup is not None else None
        out["env_path"] = str(env_file)
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("rollback-thresholds-env")
def rollback_thresholds_env(
    env_path: str = typer.Option(".env", help="Env file path to restore"),
    backup_file: str | None = typer.Option(None, help="Specific backup file; latest is used if omitted"),
) -> None:
    """Rollback .env threshold keys from backup."""
    from pathlib import Path

    from bot_cripto.ops.env_tools import find_latest_backup, restore_env_backup

    env_file = Path(env_path)
    backup = Path(backup_file) if backup_file else find_latest_backup(env_file)
    if backup is None:
        typer.echo("No backup found for rollback")
        raise typer.Exit(code=2)
    restore_env_backup(env_file, backup)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "env_path": str(env_file),
                "restored_from": str(backup),
            },
            indent=2,
        )
    )


@app.command("realistic-backtest")
def realistic_backtest(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    maker_fee: float = typer.Option(2.0, help="Maker fee in bps"),
    taker_fee: float = typer.Option(4.0, help="Taker fee in bps"),
    base_slippage: float = typer.Option(1.0, help="Base slippage in bps"),
    volume_impact: float = typer.Option(0.1, help="Volume impact factor for dynamic slippage"),
    latency: int = typer.Option(1, help="Execution latency in bars"),
    max_fill: float = typer.Option(0.10, help="Max fill ratio of bar volume (0.0-1.0)"),
    equity: float = typer.Option(10_000.0, help="Initial equity"),
    position_frac: float = typer.Option(0.02, help="Position size as fraction of equity"),
) -> None:
    """Run realistic backtest with dynamic costs, partial fills, and latency."""
    from bot_cripto.backtesting.realistic import CostModel, RealisticBacktester
    from bot_cripto.models.baseline import BaselineModel
    from bot_cripto.models.base import PredictionOutput

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")
    input_path = (
        settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    )
    if not input_path.exists():
        typer.echo(f"Feature dataset not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)

    # Train baseline model and generate signals
    model = BaselineModel()
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    model.train(train_df, target_col="close")

    signals = []
    for i in range(train_size, len(df)):
        window = df.iloc[:i]
        pred: PredictionOutput = model.predict(window)
        if pred.expected_return > 0 and pred.prob_up >= 0.55:
            signals.append(1)
        elif pred.expected_return < 0 and pred.prob_up < 0.45:
            signals.append(-1)
        else:
            signals.append(0)

    test_df = df.iloc[train_size:].copy()
    test_df["signal"] = signals

    cost_model = CostModel(
        maker_fee_bps=maker_fee,
        taker_fee_bps=taker_fee,
        base_slippage_bps=base_slippage,
        volume_impact_factor=volume_impact,
        latency_bars=latency,
        max_fill_ratio=max_fill,
    )

    report = RealisticBacktester(
        cost_model=cost_model,
        initial_equity=equity,
        position_size_frac=position_frac,
    ).run(test_df)

    out = {k: v for k, v in report.__dict__.items() if k != "trades"}
    out["sample_trades"] = [t.__dict__ for t in report.trades[:10]]
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("hrp-allocate")
def hrp_allocate_cmd(
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT,SOL/USDT (defaults to SYMBOLS)",
    ),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 1h"),
    lookback: int = typer.Option(1000, help="Lookback bars for return matrix"),
    use_tb_dataset: bool = typer.Option(True, help="Prefer *_features_tb.parquet when available"),
) -> None:
    """Compute HRP allocation weights (Phase 4 MVP)."""
    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.risk.hrp import hrp_allocate

    settings = get_settings()
    target_timeframe = timeframe or settings.timeframe
    target_symbols = _split_csv(symbols) or settings.symbols_list
    if len(target_symbols) < 2:
        typer.echo("Need at least 2 symbols for HRP allocation")
        raise typer.Exit(code=2)

    returns_data: dict[str, pd.Series] = {}
    for sym in target_symbols:
        df = load_feature_dataset_for_training(
            settings=settings,
            symbol=sym,
            timeframe=target_timeframe,
            prefer_triple_barrier=use_tb_dataset,
        )
        if "close" not in df.columns:
            typer.echo(f"Missing close column for {sym}")
            raise typer.Exit(code=2)
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        ret = close.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(ret) < 30:
            typer.echo(f"Insufficient return history for {sym}")
            raise typer.Exit(code=2)
        returns_data[sym] = ret.tail(lookback)

    aligned = pd.concat(returns_data, axis=1).dropna(how="any")
    report = hrp_allocate(aligned)

    ts_compact = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_file = settings.logs_dir / f"hrp_allocation_{target_timeframe}_{ts_compact}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "timeframe": target_timeframe,
        "rows_used": len(aligned),
        "symbols": target_symbols,
        "method": report.method,
        "ordered_assets": report.ordered_assets,
        "weights": report.weights,
        "output_file": str(out_file),
    }
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2))


@app.command("blend-allocate")
def blend_allocate_cmd(
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT,SOL/USDT (defaults to SYMBOLS)",
    ),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 1h"),
    lookback: int = typer.Option(1000, help="Lookback bars for return matrix"),
    views_file: str | None = typer.Option(
        None,
        help="Optional JSON file with asset views in [-1,1], e.g. {\"BTC/USDT\": 0.6}",
    ),
    w_hrp: float = typer.Option(0.5, help="Blend weight for HRP component"),
    w_kelly: float = typer.Option(0.3, help="Blend weight for Kelly proxy component"),
    w_views: float = typer.Option(0.2, help="Blend weight for views component"),
    corr_threshold: float = typer.Option(0.45, help="Correlation threshold to start shrink"),
    corr_max_shrink: float = typer.Option(0.50, help="Max shrink toward equal weights"),
    use_tb_dataset: bool = typer.Option(True, help="Prefer *_features_tb.parquet when available"),
) -> None:
    """Compute blended allocation HRP+Kelly+Views with dynamic-correlation adjustment."""
    from pathlib import Path

    from bot_cripto.jobs.common import load_feature_dataset_for_training
    from bot_cripto.risk.allocation_blend import blend_allocations

    settings = get_settings()
    target_timeframe = timeframe or settings.timeframe
    target_symbols = _split_csv(symbols) or settings.symbols_list
    if len(target_symbols) < 2:
        typer.echo("Need at least 2 symbols for blended allocation")
        raise typer.Exit(code=2)

    views: dict[str, float] | None = None
    if views_file:
        vf = Path(views_file)
        if not vf.exists():
            typer.echo(f"Views file not found: {vf}")
            raise typer.Exit(code=2)
        parsed = json.loads(vf.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            typer.echo("Views file must be a JSON object {symbol: score}")
            raise typer.Exit(code=2)
        views = {str(k): float(v) for k, v in parsed.items()}

    returns_data: dict[str, pd.Series] = {}
    for sym in target_symbols:
        df = load_feature_dataset_for_training(
            settings=settings,
            symbol=sym,
            timeframe=target_timeframe,
            prefer_triple_barrier=use_tb_dataset,
        )
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        ret = close.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(ret) < 30:
            typer.echo(f"Insufficient return history for {sym}")
            raise typer.Exit(code=2)
        returns_data[sym] = ret.tail(lookback)

    aligned = pd.concat(returns_data, axis=1).dropna(how="any")
    result = blend_allocations(
        returns=aligned,
        views=views,
        w_hrp=w_hrp,
        w_kelly=w_kelly,
        w_views=w_views,
        corr_threshold=corr_threshold,
        corr_max_shrink=corr_max_shrink,
    )

    ts_compact = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_file = settings.logs_dir / f"blend_allocation_{target_timeframe}_{ts_compact}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "timeframe": target_timeframe,
        "rows_used": len(aligned),
        "symbols": target_symbols,
        "method": result.method,
        "weights": result.weights,
        "components": {
            "hrp": result.hrp_weights,
            "kelly": result.kelly_weights,
            "views": result.view_weights,
        },
        "dynamic_corr": {
            "mean_abs_corr": result.mean_abs_corr,
            "corr_shrink_applied": result.corr_shrink_applied,
            "corr_threshold": corr_threshold,
            "corr_max_shrink": corr_max_shrink,
        },
        "output_file": str(out_file),
    }
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2))


@app.command("backtest-ab-sentiment")
def backtest_ab_sentiment(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    maker_fee: float = typer.Option(2.0, help="Maker fee in bps"),
    taker_fee: float = typer.Option(4.0, help="Taker fee in bps"),
    base_slippage: float = typer.Option(1.0, help="Base slippage in bps"),
    volume_impact: float = typer.Option(0.1, help="Volume impact factor"),
    latency: int = typer.Option(1, help="Execution latency in bars"),
    max_fill: float = typer.Option(0.10, help="Max fill ratio of bar volume"),
    equity: float = typer.Option(10_000.0, help="Initial equity"),
    position_frac: float = typer.Option(0.02, help="Position size fraction"),
    train_frac: float = typer.Option(0.7, help="Training fraction"),
) -> None:
    """A/B backtest: baseline signals vs sentiment/context-adjusted signals."""
    from bot_cripto.backtesting.realistic import CostModel, RealisticBacktester
    from bot_cripto.jobs.inference import _apply_context_adjustments
    from bot_cripto.models.baseline import BaselineModel
    from bot_cripto.models.base import PredictionOutput

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")
    input_path = settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    if not input_path.exists():
        typer.echo(f"Feature dataset not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)
    if len(df) < 200:
        typer.echo("Dataset too small for A/B backtest")
        raise typer.Exit(code=2)

    model = BaselineModel()
    train_size = int(len(df) * train_frac)
    train_df = df.iloc[:train_size]
    model.train(train_df, target_col="close")

    def _signal_from_pred(pred: PredictionOutput) -> int:
        if pred.expected_return > settings.min_expected_return and pred.prob_up >= settings.prob_min:
            return 1
        if pred.expected_return < -settings.min_expected_return and pred.prob_up <= (1.0 - settings.prob_min):
            return -1
        return 0

    signals_a: list[int] = []
    signals_b: list[int] = []
    for i in range(train_size, len(df)):
        window = df.iloc[:i]
        pred_raw: PredictionOutput = model.predict(window)

        row = df.iloc[i]
        q_data = {
            "social_sentiment": float(row.get("social_sentiment", 0.5)),
            "social_sentiment_anomaly": float(row.get("social_sentiment_anomaly", 0.0)),
            "orderbook_imbalance": float(row.get("orderbook_imbalance", 0.0)),
            "macro_risk_off_score": float(row.get("macro_risk_off_score", 0.5)),
            "sp500_ret_1d": float(row.get("sp500_ret_1d", 0.0)),
            "dxy_ret_1d": float(row.get("dxy_ret_1d", 0.0)),
            "corr_btc_sp500": float(row.get("corr_btc_sp500", 0.0)),
            "corr_btc_dxy": float(row.get("corr_btc_dxy", 0.0)),
        }
        pred_adj, _ = _apply_context_adjustments(pred_raw, q_data, settings)
        signals_a.append(_signal_from_pred(pred_raw))
        signals_b.append(_signal_from_pred(pred_adj))

    test_df_a = df.iloc[train_size:].copy()
    test_df_a["signal"] = signals_a
    test_df_b = df.iloc[train_size:].copy()
    test_df_b["signal"] = signals_b

    cost_model = CostModel(
        maker_fee_bps=maker_fee,
        taker_fee_bps=taker_fee,
        base_slippage_bps=base_slippage,
        volume_impact_factor=volume_impact,
        latency_bars=latency,
        max_fill_ratio=max_fill,
    )
    runner = RealisticBacktester(
        cost_model=cost_model,
        initial_equity=equity,
        position_size_frac=position_frac,
    )
    report_a = runner.run(test_df_a)
    report_b = runner.run(test_df_b)

    out = {
        "symbol": target_symbol,
        "timeframe": target_timeframe,
        "rows_total": len(df),
        "rows_test": len(test_df_a),
        "signals_baseline_active": int(sum(1 for x in signals_a if x != 0)),
        "signals_with_sentiment_active": int(sum(1 for x in signals_b if x != 0)),
        "baseline": {
            "net_return_pct": report_a.net_return_pct,
            "sharpe": report_a.sharpe,
            "max_drawdown": report_a.max_drawdown,
            "win_rate": report_a.win_rate,
            "total_trades": report_a.total_trades,
            "total_net_pnl": report_a.total_net_pnl,
        },
        "with_sentiment": {
            "net_return_pct": report_b.net_return_pct,
            "sharpe": report_b.sharpe,
            "max_drawdown": report_b.max_drawdown,
            "win_rate": report_b.win_rate,
            "total_trades": report_b.total_trades,
            "total_net_pnl": report_b.total_net_pnl,
        },
        "delta": {
            "net_return_pct": report_b.net_return_pct - report_a.net_return_pct,
            "sharpe": report_b.sharpe - report_a.sharpe,
            "max_drawdown": report_b.max_drawdown - report_a.max_drawdown,
            "win_rate": report_b.win_rate - report_a.win_rate,
            "total_trades": report_b.total_trades - report_a.total_trades,
            "total_net_pnl": report_b.total_net_pnl - report_a.total_net_pnl,
        },
    }
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("check-retrain")
def check_retrain(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    interval_hours: float = typer.Option(24.0, help="Max hours between retrains"),
    perf_threshold: float = typer.Option(0.2, help="Performance relative drop threshold"),
    drift_ratio: float = typer.Option(0.3, help="Feature drift ratio threshold"),
    train_frac: float = typer.Option(0.7, help="Fraction of data used as training reference"),
) -> None:
    """Evaluate retrain triggers (time, performance, data drift) and recommend action."""
    from bot_cripto.adaptive.online_learner import OnlineLearningSystem
    from bot_cripto.adaptive.telemetry import log_adaptive_telemetry
    from bot_cripto.monitoring.performance_store import PerformanceStore

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe
    safe_symbol = target_symbol.replace("/", "_")

    # Load performance history
    perf_path = settings.logs_dir / "performance_history.json"
    perf_history: list[float] | None = None
    if perf_path.exists():
        store = PerformanceStore(perf_path)
        perf_history = store.metrics() or None

    # Load feature data for drift comparison
    feat_path = settings.data_dir_processed / f"{safe_symbol}_{target_timeframe}_features.parquet"
    ref_df = None
    cur_df = None
    if feat_path.exists():
        full_df = pd.read_parquet(feat_path)
        split = int(len(full_df) * train_frac)
        ref_df = full_df.iloc[:split]
        cur_df = full_df.iloc[split:]

    system = OnlineLearningSystem(
        settings=settings,
        retrain_interval_hours=interval_hours,
        perf_drop_threshold=perf_threshold,
        feature_drift_ratio=drift_ratio,
    )

    rec = system.evaluate(
        performance_history=perf_history,
        reference_features=ref_df,
        current_features=cur_df,
    )

    out = {
        "should_retrain": rec.should_retrain,
        "urgency": rec.urgency,
        "triggers_fired": rec.triggers_fired,
        "triggers_total": rec.triggers_total,
        "triggers": [
            {"name": r.name, "fired": r.fired, "reason": r.reason, "details": r.details}
            for r in rec.results
        ],
    }
    severity = "high" if rec.urgency == "high" else ("medium" if rec.urgency == "medium" else "info")
    log_adaptive_telemetry(
        settings,
        event_type="retrain_check",
        severity=severity,
        payload=out,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("auto-retrain")
def auto_retrain(
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    interval_hours: float = typer.Option(24.0, help="Max hours between retrains"),
    perf_threshold: float = typer.Option(0.2, help="Performance relative drop threshold"),
    drift_ratio: float = typer.Option(0.3, help="Feature drift ratio threshold"),
    train_frac: float = typer.Option(0.7, help="Fraction of data used as training reference"),
    include_meta: bool = typer.Option(True, help="Include meta-model retrain"),
    dry_run: bool = typer.Option(True, help="Plan only; do not execute retrain jobs"),
) -> None:
    """Evaluate retrain triggers and optionally execute retrain jobs."""
    from bot_cripto.adaptive.online_learner import OnlineLearningSystem
    from bot_cripto.adaptive.retrain_orchestrator import execute_retrain_plan
    from bot_cripto.adaptive.telemetry import log_adaptive_telemetry
    from bot_cripto.monitoring.performance_store import PerformanceStore

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe

    perf_path = settings.logs_dir / "performance_history.json"
    perf_history: list[float] | None = None
    if perf_path.exists():
        perf_history = PerformanceStore(perf_path).metrics() or None

    feat_path = settings.data_dir_processed / f"{target_symbol.replace('/', '_')}_{target_timeframe}_features.parquet"
    ref_df = None
    cur_df = None
    if feat_path.exists():
        full_df = pd.read_parquet(feat_path)
        split = int(len(full_df) * train_frac)
        ref_df = full_df.iloc[:split]
        cur_df = full_df.iloc[split:]

    system = OnlineLearningSystem(
        settings=settings,
        retrain_interval_hours=interval_hours,
        perf_drop_threshold=perf_threshold,
        feature_drift_ratio=drift_ratio,
    )
    rec = system.evaluate(
        performance_history=perf_history,
        reference_features=ref_df,
        current_features=cur_df,
    )

    jobs = []
    if rec.should_retrain:
        jobs = execute_retrain_plan(
            settings=settings,
            symbol=target_symbol,
            timeframe=target_timeframe,
            dry_run=dry_run,
            include_meta=include_meta,
        )
        # Mark retrain timestamp only if execution started (not dry-run).
        if not dry_run and jobs and all(j.status == "ok" for j in jobs):
            system.record_retrain()

    out = {
        "should_retrain": rec.should_retrain,
        "urgency": rec.urgency,
        "dry_run": dry_run,
        "triggers": [
            {"name": r.name, "fired": r.fired, "reason": r.reason, "details": r.details}
            for r in rec.results
        ],
        "jobs": [j.__dict__ for j in jobs],
    }
    severity = "high" if rec.urgency == "high" else ("medium" if rec.urgency == "medium" else "info")
    log_adaptive_telemetry(
        settings,
        event_type="auto_retrain",
        severity=severity,
        payload=out,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("champion-challenger-check")
def champion_challenger_check(
    model_name: str = typer.Option("trend", help="Model family: trend|return|risk|nbeats|baseline|tft"),
    symbol: str | None = typer.Option(None, help="Pair"),
    timeframe: str | None = typer.Option(None, help="Timeframe"),
    challenger_path: str | None = typer.Option(
        None,
        help="Specific challenger model directory; if omitted uses previous version",
    ),
    eval_window: int | None = typer.Option(None, help="Evaluation window bars"),
    promotion_margin: float | None = typer.Option(None, help="Required relative improvement"),
    min_trades: int | None = typer.Option(None, help="Minimum challenger trades to allow promotion"),
    promote: bool = typer.Option(False, help="Persist challenger as champion pointer when rule passes"),
) -> None:
    """Paper/offline champion-challenger comparison and optional promotion pointer."""
    from pathlib import Path

    from bot_cripto.adaptive.champion_challenger import run_champion_challenger_check
    from bot_cripto.adaptive.telemetry import log_adaptive_telemetry
    from bot_cripto.jobs.common import load_feature_dataset_for_training, model_version_dirs

    settings = get_settings()
    target_symbol = symbol or settings.symbols_list[0]
    target_timeframe = timeframe or settings.timeframe

    versions = model_version_dirs(settings, model_name, target_symbol, timeframe=target_timeframe)
    if not versions:
        typer.echo("No model versions found for champion/challenger check")
        raise typer.Exit(code=2)

    champ_pointer = (
        settings.models_dir
        / model_name
        / target_symbol.replace("/", "_")
        / target_timeframe
        / "champion.txt"
    )
    if champ_pointer.exists():
        champion_dir = Path(champ_pointer.read_text(encoding="utf-8").strip())
    else:
        champion_dir = versions[-1]

    if challenger_path:
        challenger_dir = Path(challenger_path)
    else:
        # Fallback challenger: previous version (if champion is latest) or latest.
        if len(versions) < 2:
            typer.echo("Need at least two versions when challenger_path is omitted")
            raise typer.Exit(code=2)
        challenger_dir = versions[-2] if champion_dir == versions[-1] else versions[-1]

    if not champion_dir.exists() or not challenger_dir.exists():
        typer.echo("Champion or challenger path does not exist")
        raise typer.Exit(code=2)
    if champion_dir.resolve() == challenger_dir.resolve():
        typer.echo("Champion and challenger cannot be the same path")
        raise typer.Exit(code=2)

    df = load_feature_dataset_for_training(
        settings=settings,
        symbol=target_symbol,
        timeframe=target_timeframe,
        prefer_triple_barrier=True,
    )
    report = run_champion_challenger_check(
        df=df,
        champion_path=champion_dir,
        challenger_path=challenger_dir,
        eval_window=eval_window or settings.cc_eval_window,
        prob_min=float(settings.prob_min),
        min_expected_return=float(settings.min_expected_return),
        roundtrip_cost=(2 * settings.fees_bps + settings.spread_bps + settings.slippage_bps) / 10_000,
        promotion_margin=float(
            promotion_margin if promotion_margin is not None else settings.cc_promotion_margin
        ),
        min_trades=int(min_trades if min_trades is not None else settings.cc_min_trades),
    )

    promoted = False
    if promote and report.promote:
        champ_pointer.parent.mkdir(parents=True, exist_ok=True)
        champ_pointer.write_text(str(challenger_dir), encoding="utf-8")
        promoted = True

    out = {
        **report.__dict__,
        "champion": report.champion.__dict__,
        "challenger": report.challenger.__dict__,
        "promote_requested": promote,
        "promoted": promoted,
        "champion_pointer": str(champ_pointer),
    }
    severity = "info" if not out["promoted"] else "medium"
    log_adaptive_telemetry(
        settings,
        event_type="champion_challenger",
        severity=severity,
        payload=out,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("detect-drift")
def detect_drift(
    history_file: str = typer.Option(
        "./logs/performance_history.json",
        help="JSON file with metric history list, e.g. [0.55,0.53,...]",
    ),
    threshold: float = typer.Option(0.2, help="Relative drop threshold"),
) -> None:
    from pathlib import Path

    from bot_cripto.monitoring.drift import detect_performance_drift
    from bot_cripto.monitoring.performance_store import PerformanceStore

    path = Path(history_file)
    if not path.exists():
        typer.echo(f"History file not found: {path}")
        raise typer.Exit(code=1)


@app.command("telegram-control")
def telegram_control() -> None:
    """Run Telegram control bot (group-only)."""
    from bot_cripto.notifications.telegram_control import TelegramControlBot

    bot = TelegramControlBot()
    if not bot.enabled:
        typer.echo(
            "Telegram control disabled. Set TELEGRAM_ENABLE_COMMANDS=true, "
            "TELEGRAM_BOT_TOKEN, and TELEGRAM_ALLOWED_CHAT_IDS."
        )
        raise typer.Exit(code=2)
    bot.loop_forever()

    store = PerformanceStore(path)
    values = store.metrics()
    if not values:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and all(isinstance(v, (int, float)) for v in raw):
            values = [float(v) for v in raw]
    if not values:
        typer.echo("Invalid history file. Expected list[float] or [{ts, metric}].")
        raise typer.Exit(code=1)

    result = detect_performance_drift(
        history=values,
        relative_drop_threshold=threshold,
    )
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command("dashboard")
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Streamlit host"),
    port: int = typer.Option(8501, help="Streamlit port"),
) -> None:
    from bot_cripto.ui import dashboard as dashboard_module

    module_path = dashboard_module.__file__
    if module_path is None:
        typer.echo("Dashboard module path not found.")
        raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        module_path,
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    typer.echo("Launching Watchtower dashboard...")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("evaluate")
def evaluate(
    symbol: str = typer.Option("BTC/USDT", help="Symbol to evaluate"),
    days: int = typer.Option(30, help="Days of history to analyze"),
) -> None:
    """Generate a professional quantitative performance report."""
    from bot_cripto.backtesting.evaluator import QuantEvaluator
    import pandas as pd
    import json

    settings = get_settings()
    trades_path = settings.logs_dir / "paper_state.json"
    
    if not trades_path.exists():
        typer.echo("No trades found to evaluate. Start the bot first.")
        raise typer.Exit(code=1)

    with open(trades_path, "r") as f:
        data = json.load(f)
    
    trades_df = pd.DataFrame(data.get("trades", []))
    if trades_df.empty:
        typer.echo("Trade history is empty.")
        return

    # Cargar precios para el benchmark
    safe_symbol = symbol.replace("/", "_")
    price_path = settings.data_dir_raw / f"{safe_symbol}_1h.parquet"
    if not price_path.exists():
        typer.echo(f"Price history not found for benchmark: {price_path}")
        return
    
    price_df = pd.read_parquet(price_path)
    
    evaluator = QuantEvaluator(initial_equity=settings.initial_equity)
    metrics = evaluator.calculate_metrics(trades_df, price_df["close"])
    
    # Simular regímenes (esto se puede mejorar con el MLRegimeEngine real)
    regimes = {"TREND": metrics, "RANGE": metrics} # Placeholder
    
    report = evaluator.generate_markdown_report(metrics, regimes)
    typer.echo(report)
    
    # Guardar reporte
    report_path = settings.logs_dir / f"performance_report_{safe_symbol}.md"
    report_path.write_text(report)
    typer.echo(f"Report saved to: {report_path}")


if __name__ == "__main__":
    app()
