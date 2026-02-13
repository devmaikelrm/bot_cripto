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


@app.command()
def train(
    symbol: str | None = typer.Option(None, help="Pair"),
    model_type: str = typer.Option("baseline", help="Model type: baseline|tft"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
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

    model: BasePredictor
    if model_type == "baseline":
        model = BaselineModel()
    elif model_type == "tft":
        from bot_cripto.models.tft import TFTPredictor

        model = TFTPredictor()
    else:
        typer.echo(f"Unsupported model type: {model_type}")
        raise typer.Exit(code=1)

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
    folds: int = typer.Option(4, help="Walk-forward folds"),
    timeframe: str | None = typer.Option(None, help="Timeframe like 5m, 15m, 1h (overrides TIMEFRAME)"),
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
    ).run(df, model_factory=BaselineModel)
    typer.echo(json.dumps(report.__dict__, indent=2))


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


if __name__ == "__main__":
    app()
