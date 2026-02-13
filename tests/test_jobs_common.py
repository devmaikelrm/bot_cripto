from bot_cripto.jobs.common import write_model_metadata
from bot_cripto.models.base import ModelMetadata


def test_write_model_metadata(tmp_path) -> None:
    metadata = ModelMetadata.create(
        model_type="baseline_rf",
        version="0.1.0",
        metrics={"brier_after": 0.12},
    )
    out = write_model_metadata(tmp_path, metadata)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert '"model_type": "baseline_rf"' in content
    assert '"brier_after": 0.12' in content
