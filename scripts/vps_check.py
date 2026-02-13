"""
VPS health-check and quick-fix helper for Bot Cripto.

Goal:
- Run from Windows (or anywhere) without storing credentials.
- SSH into VPS, inspect systemd --user units, logs, symlinks, and recent journal output.
- Optionally apply a safe "fix" (enable/start timers, restart telegram control, force one inference run).

Usage (PowerShell):
  $env:VPS_HOST="100.71.91.32"
  $env:VPS_USER="maikelrm95"
  python scripts\\vps_check.py
  python scripts\\vps_check.py --fix
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from dataclasses import dataclass

import paramiko


def _safe_print(text: str) -> None:
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")


@dataclass(frozen=True)
class RemoteResult:
    rc: int
    out: str
    err: str


class Remote:
    def __init__(self, host: str, user: str, password: str) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.client: paramiko.SSHClient | None = None

    def __enter__(self) -> "Remote":
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, username=self.user, password=self.password)
        self.client = client
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self.client is not None:
            self.client.close()
            self.client = None

    def run(self, cmd: str, timeout_s: int = 120) -> RemoteResult:
        if self.client is None:
            raise RuntimeError("not connected")

        # systemctl --user over SSH can fail without these env vars.
        prefix = (
            "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
            "export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus; "
        )
        full = f"bash -lc {paramiko.util.shell_quote(prefix + cmd)}"
        stdin, stdout, stderr = self.client.exec_command(full, timeout=timeout_s)
        rc = int(stdout.channel.recv_exit_status())
        out = (stdout.read() or b"").decode("utf-8", errors="replace").strip()
        err = (stderr.read() or b"").decode("utf-8", errors="replace").strip()
        return RemoteResult(rc=rc, out=out, err=err)


def _section(title: str) -> None:
    _safe_print("")
    _safe_print("=" * 72)
    _safe_print(title)
    _safe_print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot Cripto VPS check (no credentials saved).")
    parser.add_argument("--fix", action="store_true", help="Apply safe fixes (enable/start timers, restart services).")
    parser.add_argument("--tail", type=int, default=120, help="Lines to tail from log files (default: 120).")
    parser.add_argument("--host", default=os.environ.get("VPS_HOST", "").strip())
    parser.add_argument("--user", default=os.environ.get("VPS_USER", "").strip())
    parser.add_argument("--remote-dir", default=os.environ.get("VPS_REMOTE_DIR", "~/bot-cripto").strip())
    parser.add_argument(
        "--state-dir", default=os.environ.get("VPS_STATE_DIR", "~/bot-cripto-state").strip()
    )
    args = parser.parse_args()

    if not args.host or not args.user:
        _safe_print("Missing VPS host/user.")
        _safe_print("Set env vars or pass flags:")
        _safe_print("  PowerShell:")
        _safe_print('    $env:VPS_HOST="100.71.91.32"; $env:VPS_USER="maikelrm95"')
        _safe_print("  Then:")
        _safe_print("    python scripts\\vps_check.py --fix")
        return 2

    password = os.environ.get("VPS_PASS", "").strip()
    if not password:
        password = getpass.getpass(f"Password for {args.user}@{args.host}: ")

    _section(f"Connecting: {args.user}@{args.host}")
    with Remote(args.host, args.user, password) as r:
        _safe_print("Connected.")

        _section("Release/State Paths + Symlinks")
        checks = [
            f"echo remote_dir={args.remote_dir}",
            f"echo state_dir={args.state_dir}",
            f"ls -la {args.remote_dir} || true",
            f"ls -la {args.state_dir} || true",
            f"ls -la {args.remote_dir}/logs || true",
            f"ls -la {args.remote_dir}/data || true",
            f"ls -la {args.remote_dir}/models || true",
            f"readlink -f {args.remote_dir}/logs || true",
            f"readlink -f {args.remote_dir}/data || true",
            f"readlink -f {args.remote_dir}/models || true",
        ]
        for c in checks:
            res = r.run(c, timeout_s=60)
            if res.out:
                _safe_print(res.out)
            if res.err:
                _safe_print(res.err)

        if args.fix:
            _section("Fix: Ensure state dirs + symlinks")
            fix_cmd = " && ".join(
                [
                    f"mkdir -p {args.state_dir}/data {args.state_dir}/models {args.state_dir}/logs",
                    f"ln -sfn {args.state_dir}/data {args.remote_dir}/data",
                    f"ln -sfn {args.state_dir}/models {args.remote_dir}/models",
                    f"ln -sfn {args.state_dir}/logs {args.remote_dir}/logs",
                    "echo ok_symlinks",
                ]
            )
            res = r.run(fix_cmd, timeout_s=60)
            _safe_print(res.out or "(no stdout)")
            if res.err:
                _safe_print(res.err)

        _section("systemd --user: timers/services")
        cmds = [
            "systemctl --user daemon-reload || true",
            "systemctl --user list-timers | grep bot-cripto || true",
            "systemctl --user status bot-cripto-inference.timer --no-pager || true",
            "systemctl --user status bot-cripto-retrain.timer --no-pager || true",
            "systemctl --user status bot-cripto-telegram-control.service --no-pager || true",
        ]
        if args.fix:
            cmds = (
                [
                    "systemctl --user enable --now bot-cripto-inference.timer bot-cripto-retrain.timer || true",
                    "systemctl --user restart bot-cripto-telegram-control.service || true",
                ]
                + cmds
                + [
                    # Force one run to generate logs and surface errors immediately.
                    "systemctl --user start bot-cripto-inference.service || true",
                    "sleep 2 || true",
                ]
            )

        for c in cmds:
            res = r.run(c, timeout_s=120)
            if res.out:
                _safe_print(res.out)
            if res.err:
                _safe_print(res.err)

        _section("Logs (tail)")
        tail = int(max(20, args.tail))
        log_cmds = [
            f"ls -la {args.remote_dir}/logs || true",
            f"tail -n {tail} {args.remote_dir}/logs/cycle.log 2>/dev/null || true",
            f"tail -n {tail} {args.remote_dir}/logs/cycle.err.log 2>/dev/null || true",
            f"tail -n {tail} {args.remote_dir}/logs/retrain.log 2>/dev/null || true",
            f"tail -n {tail} {args.remote_dir}/logs/retrain.err.log 2>/dev/null || true",
            f"tail -n {tail} {args.remote_dir}/logs/telegram_control.log 2>/dev/null || true",
        ]
        for c in log_cmds:
            res = r.run(c, timeout_s=60)
            if res.out:
                _safe_print(res.out)
            if res.err:
                _safe_print(res.err)

        _section("journalctl --user (last 120 lines per unit)")
        journal_cmds = [
            "journalctl --user -u bot-cripto-inference.service -n 120 --no-pager || true",
            "journalctl --user -u bot-cripto-retrain.service -n 120 --no-pager || true",
            "journalctl --user -u bot-cripto-telegram-control.service -n 120 --no-pager || true",
        ]
        for c in journal_cmds:
            res = r.run(c, timeout_s=60)
            if res.out:
                _safe_print(res.out)
            if res.err:
                _safe_print(res.err)

    _section("Done")
    _safe_print("If you see failures above, paste the 'journalctl --user' sections here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
