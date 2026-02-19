from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class LogTarget:
    key: str
    pair_tf: str
    path: Path


@dataclass
class RowStatus:
    pair_tf: str
    state: str
    progress: str
    detail: str
    epoch: int
    val_loss: str


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.ok
    except requests.RequestException:
        return False


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_row_status(content: str, pair_tf: str, total_epochs: int) -> RowStatus:
    if not content:
        return RowStatus(pair_tf, "⏳ Pendiente", "-", "Sin log todavía.", 0, "-")

    epochs = re.findall(r"Epoch\s+(\d+):", content)
    losses = re.findall(r"val_loss=([0-9eE+.\-]+)", content)
    last_epoch = int(epochs[-1]) + 1 if epochs else 0
    last_loss = losses[-1] if losses else "-"

    if "OutOfMemoryError" in content or "Traceback" in content:
        state = "❌ Error"
        detail = "Revisa log: excepcion detectada."
    elif "entrenamiento_completado" in content or "train_trend_done" in content or "train_return_done" in content or "train_risk_done" in content:
        state = "✅ Completado"
        progress = f"{total_epochs}/{total_epochs}"
        detail = f"val_loss final: {last_loss}" if last_loss != "-" else "Entrenamiento finalizado."
        return RowStatus(pair_tf, state, progress, detail, total_epochs, last_loss)
    elif last_epoch > 0:
        state = "🔄 En Curso"
        detail = f"val_loss: {last_loss}" if last_loss != "-" else "Entrenando..."
    else:
        state = "🚀 Iniciado"
        detail = "Inicializando dataloaders/modelo."

    progress = f"{last_epoch} / {total_epochs}" if last_epoch > 0 else "1 / {0}".format(total_epochs)
    return RowStatus(pair_tf, state, progress, detail, last_epoch, last_loss)


def _format_cell(text: str, width: int) -> str:
    clean = text.replace("\n", " ").strip()
    if len(clean) > width:
        clean = clean[: max(0, width - 1)] + "…"
    return clean.ljust(width)


def _build_board(rows: list[RowStatus]) -> str:
    c1, c2, c3, c4 = 17, 12, 16, 44
    top = "┌" + "─" * (c1 + 2) + "┬" + "─" * (c2 + 2) + "┬" + "─" * (c3 + 2) + "┬" + "─" * (c4 + 2) + "┐"
    mid = "├" + "─" * (c1 + 2) + "┼" + "─" * (c2 + 2) + "┼" + "─" * (c3 + 2) + "┼" + "─" * (c4 + 2) + "┤"
    bot = "└" + "─" * (c1 + 2) + "┴" + "─" * (c2 + 2) + "┴" + "─" * (c3 + 2) + "┴" + "─" * (c4 + 2) + "┘"

    lines = [
        "*Estado de Entrenamiento*",
        "```",
        top,
        f"│ {_format_cell('Par / Temporalidad', c1)} │ {_format_cell('Estado', c2)} │ {_format_cell('Progreso / Epoca', c3)} │ {_format_cell('Detalle Tecnico', c4)} │",
        mid,
    ]
    for row in rows:
        lines.append(
            f"│ {_format_cell(row.pair_tf, c1)} │ {_format_cell(row.state, c2)} │ {_format_cell(row.progress, c3)} │ {_format_cell(row.detail, c4)} │"
        )
    lines.extend([bot, "```"])
    return "\n".join(lines)


def _build_signature(rows: list[RowStatus]) -> dict[str, dict[str, str | int]]:
    return {
        row.pair_tf: {
            "state": row.state,
            "epoch": row.epoch,
            "val_loss": row.val_loss,
            "progress": row.progress,
        }
        for row in rows
    }


def run(env_file: Path, interval_seconds: int, state_file: Path, targets: list[LogTarget], total_epochs: int) -> int:
    _load_env(env_file)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return 2

    state_file.parent.mkdir(parents=True, exist_ok=True)
    previous_signature: dict[str, dict[str, str | int]] = {}
    if state_file.exists():
        try:
            previous_signature = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_signature = {}

    while True:
        rows: list[RowStatus] = []
        for target in targets:
            content = _read_text_safe(target.path)
            rows.append(_extract_row_status(content, target.pair_tf, total_epochs))

        current_signature = _build_signature(rows)
        if current_signature != previous_signature:
            msg = _build_board(rows)
            sent = _send_telegram(token, chat_id, msg)
            if sent:
                previous_signature = current_signature
                state_file.write_text(
                    json.dumps(current_signature, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--state-file", default="logs/telegram_monitor_state.json")
    parser.add_argument("--total-epochs", type=int, default=50)
    args = parser.parse_args()

    targets = [
        LogTarget("btc_1h", "BTC/USD 1h", Path("logs/training.log")),
        LogTarget("sol_1h", "SOL/USDT 1h", Path("logs/training_sol.log")),
        LogTarget("btc_5m", "BTC/USDT 5m", Path("logs/training_btc_5m.log")),
        LogTarget("sol_5m", "SOL/USDT 5m", Path("logs/training_sol_5m.log")),
    ]

    raise SystemExit(
        run(
            env_file=Path(args.env_file),
            interval_seconds=max(args.interval, 30),
            state_file=Path(args.state_file),
            targets=targets,
            total_epochs=max(args.total_epochs, 1),
        )
    )


if __name__ == "__main__":
    main()
