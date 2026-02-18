from bot_cripto.core.config import Settings
from bot_cripto.notifications import telegram_control as tc


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    return settings


def test_cmd_training_non_linux_message(tmp_path, monkeypatch) -> None:
    bot = tc.TelegramControlBot(settings=_settings(tmp_path))
    monkeypatch.setattr(tc.sys, "platform", "win32")
    out = bot._cmd_training()
    assert "Linux hosts" in out


def test_cmd_training_linux_uses_dynamic_paths(tmp_path, monkeypatch) -> None:
    bot = tc.TelegramControlBot(settings=_settings(tmp_path))
    monkeypatch.setattr(tc.sys, "platform", "linux")
    (bot.settings.logs_dir / "retrain_institutional_v2.log").write_text("Epoch 7\n", encoding="utf-8")

    def _fake_run_output(argv: list[str]) -> str:
        if argv[:2] == ["ps", "-eo"]:
            return "123 45.0 12.0 00:10:00 /usr/bin/python -m bot_cripto.cli train-trend"
        if argv[:2] == ["free", "-h"]:
            return "              total        used\nMem:           16Gi        4Gi"
        if argv[:2] == ["du", "-sh"]:
            return "1.2G logs"
        if argv[:2] == ["tail", "-n"]:
            return "Epoch 7\n"
        return ""

    monkeypatch.setattr(tc, "_run_output", _fake_run_output)
    out = bot._cmd_training()

    assert "PID 123" in out
    assert "Epoca 7" in out
    assert "Disco (logs)" in out
    assert "1.2G" in out
