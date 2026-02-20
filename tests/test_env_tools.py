from __future__ import annotations

from pathlib import Path

from bot_cripto.ops.env_tools import (
    apply_env_values,
    backup_env_file,
    find_latest_backup,
    restore_env_backup,
)


def test_apply_env_values_updates_and_appends(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("FOO=1\nPROB_MIN=0.60\n", encoding="utf-8")
    apply_env_values(
        env,
        {
            "PROB_MIN": "0.650000",
            "MIN_EXPECTED_RETURN": "0.003000",
        },
    )
    text = env.read_text(encoding="utf-8")
    assert "PROB_MIN=0.650000" in text
    assert "MIN_EXPECTED_RETURN=0.003000" in text


def test_backup_and_restore_env(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("PROB_MIN=0.60\n", encoding="utf-8")
    backup = backup_env_file(env)
    assert backup is not None
    env.write_text("PROB_MIN=0.99\n", encoding="utf-8")
    restore_env_backup(env, backup)
    assert env.read_text(encoding="utf-8") == "PROB_MIN=0.60\n"


def test_find_latest_backup(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    b1 = tmp_path / ".env.bak.20260101T000000Z"
    b2 = tmp_path / ".env.bak.20260101T010000Z"
    b1.write_text("A=1\n", encoding="utf-8")
    b2.write_text("A=2\n", encoding="utf-8")
    latest = find_latest_backup(env)
    assert latest is not None
    assert latest.name.endswith("010000Z")
