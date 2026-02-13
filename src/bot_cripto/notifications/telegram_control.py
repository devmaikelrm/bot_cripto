"""Telegram control bot (group-only) for operational commands.

Design goals:
- Runs on the VPS as a long-lived process (systemd recommended).
- Never executes arbitrary shell; only whitelisted actions.
- Responds only inside configured group chat(s).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.ops.operator_flags import OperatorFlags, default_flags_store

logger = get_logger("notifications.telegram_control")

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,12}/[A-Z0-9]{2,12}$")
_TF_ALLOWED = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _tail_file(path: Path, n: int = 50) -> str:
    if not path.exists():
        return f"missing: {path.name}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return "\n".join(lines) if lines else "(empty)"
    except OSError as exc:
        return f"error reading {path.name}: {exc}"


@dataclass(frozen=True)
class Update:
    update_id: int
    chat_id: str
    chat_type: str
    from_user: str
    text: str


class TelegramControlBot:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.flags_store = default_flags_store(self.settings)
        self.offset_path = self.settings.logs_dir / "telegram_offset.json"
        self.offset = self._load_offset()

    @property
    def enabled(self) -> bool:
        s = self.settings
        return bool(
            s.telegram_enable_commands
            and s.telegram_bot_token
            and s.telegram_allowed_chat_ids_list
        )

    def _api_url(self, method: str) -> str:
        token = self.settings.telegram_bot_token
        return f"https://api.telegram.org/bot{token}/{method}"

    def _load_offset(self) -> int:
        if not self.offset_path.exists():
            return 0
        try:
            raw = json.loads(self.offset_path.read_text(encoding="utf-8"))
            return int(raw.get("offset", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            return 0

    def _save_offset(self, offset: int) -> None:
        payload = {"offset": int(offset), "updated_at": _now_iso()}
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)
        self.offset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = self._api_url(method)
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("User-Agent", "bot-cripto/telegram-control")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
            parsed: dict[str, Any] = json.loads(body)
            return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.error("telegram_control_http_error", error=str(exc))
            return None

    def _ensure_long_polling_mode(self) -> None:
        # If a webhook is set, Telegram will return 409 Conflict for getUpdates.
        resp = self._post("deleteWebhook", {"drop_pending_updates": "true"})
        if not resp:
            return
        if resp.get("ok"):
            logger.info("telegram_webhook_deleted")
        else:
            logger.warning("telegram_webhook_delete_failed", body=str(resp))

    def _send(self, chat_id: str, text: str) -> None:
        self._post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )

    def _allowed_chat(self, upd: Update) -> bool:
        if upd.chat_id not in set(self.settings.telegram_allowed_chat_ids_list):
            return False
        # group-only: do not respond to private chats
        return upd.chat_type in {"group", "supergroup"}

    def _get_updates(self) -> list[Update]:
        parsed = self._post(
            "getUpdates",
            {"timeout": 15, "offset": self.offset, "allowed_updates": json.dumps(["message"])},
        )
        if not parsed or not parsed.get("ok"):
            return []
        out: list[Update] = []
        for item in parsed.get("result", []):
            try:
                msg = item.get("message", {})
                chat = msg.get("chat", {})
                frm = msg.get("from", {})
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue
                out.append(
                    Update(
                        update_id=int(item["update_id"]),
                        chat_id=str(chat.get("id", "")),
                        chat_type=str(chat.get("type", "")),
                        from_user=str(frm.get("username") or frm.get("id") or "unknown"),
                        text=text,
                    )
                )
            except Exception:
                continue
        return out

    def _run_cli(self, argv: list[str], timeout_s: int = 3600) -> tuple[int, str]:
        # No shell, no injection.
        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            return int(proc.returncode), combined.strip()
        except Exception as exc:
            return 1, f"error running command: {exc}"

    def _cli_base(self) -> list[str]:
        # Always run the CLI using the same interpreter running this bot (venv-safe).
        return [sys.executable, "-m", "bot_cripto.cli"]

    def _cmd_help(self) -> str:
        default_symbol = self.settings.symbols_list[0] if self.settings.symbols_list else "BTC/USDT"
        default_tf = self.settings.timeframe
        try:
            recent_logs = sorted(
                [p.name for p in self.settings.logs_dir.glob("*.log")],
                key=lambda n: (self.settings.logs_dir / n).stat().st_mtime,
                reverse=True,
            )[:5]
        except OSError:
            recent_logs = []
        recent_logs_text = ", ".join(recent_logs) if recent_logs else "(none)"

        return "\n".join(
            [
                "Bot-Cripto Control (solo este grupo)",
                "",
                "Orden recomendado (operacion diaria):",
                "1) /status",
                f"2) /infer-now {default_symbol}",
                "3) Si hay dudas: /pause 30",
                "4) Emergencia: /kill on",
                "5) Volver a normal: /kill off y /resume",
                "",
                "Estado y control:",
                "/status",
                "  Muestra modo paper/live, si esta pausado, kill-switch y si live esta armado.",
                "/pause <min>",
                "  Pausa entradas BUY por X minutos (SELL sigue permitido). Ej: /pause 30",
                "/resume",
                "  Quita la pausa.",
                "/kill on|off",
                "  Kill-switch: bloquea todo (incluye live). Recomendado ante dudas.",
                "",
                "Live (muy importante):",
                "/live arm <min>",
                "  Arma ejecucion live por X minutos. Ej: /live arm 10",
                "/live disarm",
                "  Desarma live inmediatamente.",
                "  Nota: Live tambien requiere LIVE_MODE=true y LIVE_CONFIRM_TOKEN=I_UNDERSTAND_LIVE_TRADING.",
                "",
                "Acciones rapidas:",
                "/infer-now [SYMBOL]",
                f"  Corre inference ahora y devuelve la senal. Ej: /infer-now {default_symbol}",
                "",
                "Datos / modelos:",
                "/data",
                "  Lista archivos .parquet en data/raw.",
                "/models",
                "  Lista carpetas de modelos guardados.",
                "/fetch <SYMBOL> <TF> <DAYS>",
                f"  Descarga historico. Ej: /fetch {default_symbol} {default_tf} 30",
                "/features <SYMBOL> <TF>",
                f"  Genera features. Ej: /features {default_symbol} {default_tf}",
                "/train <trend|return|risk|baseline|tft> <SYMBOL>",
                f"  Entrena un modelo. Ej: /train trend {default_symbol}",
                "/backtest <SYMBOL> <FOLDS>",
                f"  Walk-forward backtest. Ej: /backtest {default_symbol} 4",
                "",
                "Logs:",
                "/tail <logfile> [N]",
                "  Muestra ultimas N lineas (default 50). Ej: /tail fetch_crypto_multi_tf.log 40",
                f"  Logs recientes: {recent_logs_text}",
            ]
        )

    def _cmd_status(self) -> str:
        flags = self.flags_store.load()
        parts = [
            f"ts={_now_iso()}",
            f"live_mode={self.settings.live_mode}",
            f"paper_mode={self.settings.paper_mode}",
            f"paused={flags.is_paused()}",
            f"kill_switch={flags.kill_switch}",
            f"live_armed={flags.is_live_armed()}",
        ]
        sig = self.settings.logs_dir / "signal.json"
        if sig.exists():
            parts.append("signal=present")
        else:
            parts.append("signal=missing")
        return " | ".join(parts)

    def _cmd_pause(self, minutes: int) -> str:
        flags = self.flags_store.load()
        until = datetime.now(tz=UTC) + timedelta(minutes=max(1, minutes))
        flags.pause_until = until.isoformat()
        flags.note = f"paused by telegram at {_now_iso()}"
        self.flags_store.save(flags)
        return f"paused until {flags.pause_until}"

    def _cmd_resume(self) -> str:
        flags = self.flags_store.load()
        flags.pause_until = None
        flags.note = f"resumed by telegram at {_now_iso()}"
        self.flags_store.save(flags)
        return "resumed"

    def _cmd_kill(self, on: bool) -> str:
        flags = self.flags_store.load()
        flags.kill_switch = bool(on)
        if on:
            flags.pause_until = None
            flags.live_armed_until = None
        flags.note = f"kill_switch={on} by telegram at {_now_iso()}"
        self.flags_store.save(flags)
        return f"kill_switch set to {on}"

    def _cmd_live_arm(self, minutes: int) -> str:
        flags = self.flags_store.load()
        until = datetime.now(tz=UTC) + timedelta(minutes=max(1, minutes))
        flags.live_armed_until = until.isoformat()
        flags.note = f"live armed by telegram at {_now_iso()}"
        self.flags_store.save(flags)
        return f"live armed until {flags.live_armed_until}"

    def _cmd_live_disarm(self) -> str:
        flags = self.flags_store.load()
        flags.live_armed_until = None
        flags.note = f"live disarmed by telegram at {_now_iso()}"
        self.flags_store.save(flags)
        return "live disarmed"

    def _cmd_tail(self, name: str, n: int) -> str:
        # whitelist: only logs directory and only filenames (no paths)
        safe_name = Path(name).name
        return _tail_file(self.settings.logs_dir / safe_name, n=n)

    def _cmd_data(self) -> str:
        raw = self.settings.data_dir_raw
        if not raw.exists():
            return "data/raw missing"
        files = sorted([p.name for p in raw.glob("*.parquet")])
        return "data/raw:\n" + ("\n".join(files) if files else "(none)")

    def _cmd_models(self) -> str:
        base = self.settings.models_dir
        if not base.exists():
            return "models dir missing"
        files = sorted([str(p.relative_to(base)) for p in base.glob("**/*") if p.is_dir()])
        shown = "\n".join(files[:80]) if files else "(none)"
        return "models:\n" + shown

    def _cmd_infer_now(self, symbol: str | None) -> str:
        args = self._cli_base() + ["run-inference"]
        if symbol:
            args.extend(["--symbol", symbol])
        code, out = self._run_cli(args, timeout_s=900)
        if code != 0:
            return f"infer failed rc={code}\n{out[-1500:]}"
        return out[-1500:] if out else "ok"

    def _cmd_fetch(self, symbol: str, tf: str, days: int) -> str:
        if not _SYMBOL_RE.match(symbol):
            return "invalid symbol format"
        if tf not in _TF_ALLOWED:
            return f"invalid timeframe (allowed: {sorted(_TF_ALLOWED)})"
        args = self._cli_base() + [
            "fetch",
            "--days",
            str(days),
            "--symbol",
            symbol,
            "--timeframe",
            tf,
        ]
        code, out = self._run_cli(args, timeout_s=6 * 3600)
        return f"fetch rc={code}\n{out[-1500:]}"

    def _cmd_features(self, symbol: str, tf: str) -> str:
        if not _SYMBOL_RE.match(symbol):
            return "invalid symbol format"
        if tf not in _TF_ALLOWED:
            return f"invalid timeframe (allowed: {sorted(_TF_ALLOWED)})"
        args = self._cli_base() + ["features", "--symbol", symbol, "--timeframe", tf]
        code, out = self._run_cli(args, timeout_s=3600)
        return f"features rc={code}\n{out[-1500:]}"

    def _cmd_train(self, model: str, symbol: str) -> str:
        if model not in {"trend", "return", "risk", "baseline", "tft"}:
            return "invalid model"
        if not _SYMBOL_RE.match(symbol):
            return "invalid symbol format"
        cmd = model
        if model in {"trend", "return", "risk"}:
            cmd = f"train-{model}"
            args = self._cli_base() + [cmd, "--symbol", symbol]
        else:
            args = self._cli_base() + ["train", "--model-type", model, "--symbol", symbol]
        code, out = self._run_cli(args, timeout_s=6 * 3600)
        return f"train rc={code}\n{out[-1500:]}"

    def _cmd_backtest(self, symbol: str, folds: int) -> str:
        if not _SYMBOL_RE.match(symbol):
            return "invalid symbol format"
        args = self._cli_base() + ["backtest", "--symbol", symbol, "--folds", str(folds)]
        code, out = self._run_cli(args, timeout_s=6 * 3600)
        return f"backtest rc={code}\n{out[-1500:]}"

    def handle_message(self, upd: Update) -> None:
        if not self._allowed_chat(upd):
            return
        text = upd.text.strip()
        if not text.startswith("/"):
            return

        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        args = parts[1:]

        try:
            if cmd == "/help":
                self._send(upd.chat_id, self._cmd_help())
                return
            if cmd == "/status":
                self._send(upd.chat_id, self._cmd_status())
                return
            if cmd == "/pause":
                minutes = _parse_int(args[0] if args else "15", 15)
                self._send(upd.chat_id, self._cmd_pause(minutes))
                return
            if cmd == "/resume":
                self._send(upd.chat_id, self._cmd_resume())
                return
            if cmd == "/tail":
                if not args:
                    self._send(upd.chat_id, "usage: /tail <logfile> [N]")
                    return
                n = _parse_int(args[1] if len(args) > 1 else "50", 50)
                self._send(upd.chat_id, self._cmd_tail(args[0], n))
                return
            if cmd == "/data":
                self._send(upd.chat_id, self._cmd_data())
                return
            if cmd == "/models":
                self._send(upd.chat_id, self._cmd_models())
                return
            if cmd == "/infer-now":
                sym = args[0] if args else None
                if sym and not _SYMBOL_RE.match(sym):
                    self._send(upd.chat_id, "invalid symbol")
                    return
                self._send(upd.chat_id, self._cmd_infer_now(sym))
                return

            # Admin-only / dangerous commands
            if cmd == "/kill":
                if len(args) < 1:
                    self._send(upd.chat_id, "usage: /kill on|off")
                    return
                on = args[0].lower() == "on"
                self._send(upd.chat_id, self._cmd_kill(on))
                return

            if cmd == "/live":
                if len(args) < 1:
                    self._send(upd.chat_id, "usage: /live arm <minutes> | /live disarm")
                    return
                action = args[0].lower()
                if action == "arm":
                    minutes = _parse_int(args[1] if len(args) > 1 else "10", 10)
                    self._send(upd.chat_id, self._cmd_live_arm(minutes))
                    return
                if action == "disarm":
                    self._send(upd.chat_id, self._cmd_live_disarm())
                    return
                self._send(upd.chat_id, "usage: /live arm <minutes> | /live disarm")
                return

            if cmd == "/fetch":
                if len(args) < 3:
                    self._send(upd.chat_id, "usage: /fetch <SYMBOL> <TF> <DAYS>")
                    return
                self._send(upd.chat_id, self._cmd_fetch(args[0], args[1], _parse_int(args[2], 30)))
                return

            if cmd == "/features":
                if len(args) < 2:
                    self._send(upd.chat_id, "usage: /features <SYMBOL> <TF>")
                    return
                self._send(upd.chat_id, self._cmd_features(args[0], args[1]))
                return

            if cmd == "/train":
                if len(args) < 2:
                    self._send(upd.chat_id, "usage: /train <trend|return|risk|baseline|tft> <SYMBOL>")
                    return
                self._send(upd.chat_id, self._cmd_train(args[0], args[1]))
                return

            if cmd == "/backtest":
                if len(args) < 2:
                    self._send(upd.chat_id, "usage: /backtest <SYMBOL> <FOLDS>")
                    return
                self._send(upd.chat_id, self._cmd_backtest(args[0], _parse_int(args[1], 4)))
                return

            self._send(upd.chat_id, "unknown command. use /help")
        except Exception as exc:
            logger.error("telegram_control_handle_error", error=str(exc))
            self._send(upd.chat_id, f"error: {exc}")

    def loop_forever(self) -> None:
        if not self.enabled:
            logger.warning("telegram_control_disabled")
            return

        self._ensure_long_polling_mode()
        logger.info(
            "telegram_control_started",
            allowed_chat_ids=self.settings.telegram_allowed_chat_ids_list,
            poll_seconds=self.settings.telegram_poll_seconds,
        )
        while True:
            updates = self._get_updates()
            for upd in updates:
                self.offset = max(self.offset, upd.update_id + 1)
                self.handle_message(upd)
            if updates:
                self._save_offset(self.offset)
            time.sleep(self.settings.telegram_poll_seconds)


def main() -> None:
    TelegramControlBot().loop_forever()


if __name__ == "__main__":
    main()
