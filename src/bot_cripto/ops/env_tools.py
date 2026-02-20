"""Utilities to update .env safely with backup/rollback."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def backup_env_file(env_path: Path) -> Path | None:
    """Create timestamped backup of env file if it exists."""
    if not env_path.exists():
        return None
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = env_path.with_name(f"{env_path.name}.bak.{stamp}")
    backup.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def apply_env_values(env_path: Path, updates: dict[str, str]) -> None:
    """Update or append KEY=VALUE entries in env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    remaining = {k: str(v) for k, v in updates.items()}
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, _value = line.split("=", 1)
        k = key.strip()
        if k in remaining:
            out.append(f"{k}={remaining[k]}")
            del remaining[k]
        else:
            out.append(line)

    for k, v in remaining.items():
        out.append(f"{k}={v}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def find_latest_backup(env_path: Path) -> Path | None:
    """Return latest backup for given env file name in same directory."""
    pattern = f"{env_path.name}.bak.*"
    backups = sorted(env_path.parent.glob(pattern), key=lambda p: p.name)
    return backups[-1] if backups else None


def restore_env_backup(env_path: Path, backup_path: Path) -> None:
    """Restore env content from a backup file."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
