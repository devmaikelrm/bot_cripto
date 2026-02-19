"""CLI entrypoint for Bot Cripto."""

from __future__ import annotations

import json
import subprocess
import sys

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


@app.command()
def fetch(
    days: int = typer.Option(30, help="Days to download"),
    symbol: str | None = typer.Option(None, help="Pair like BTC/USDT"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
    provider: str | None = typer.Option(None, help="Data provider: binance|yfinance (overrides DATA_PROVIDER)"),
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
    provider: str | None = typer.Option(None, help="Data provider: binance|yfinance (overrides DATA_PROVIDER)"),
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
        help="Source: auto|blend|nlp|api|x|telegram|cryptopanic|local (overrides SOCIAL_SENTIMENT_SOURCE)",
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
