"""Tests de integracion para TFT Model."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:.*LeafSpec.*")

try:
    from bot_cripto.models.tft import TFTPredictor

    has_tft = True
except ImportError:
    has_tft = False


@pytest.fixture
def tft_sample_df() -> pd.DataFrame:
    """DataFrame de prueba para TFT."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="5min")

    x = np.linspace(0, 8 * np.pi, 200)
    close = 100 + 10 * np.sin(x) + np.linspace(0, 20, 200)

    return pd.DataFrame(
        {
            "open": close + np.random.normal(0, 1, 200),
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.random.randint(100, 1000, 200).astype(float),
            "log_ret": np.random.normal(0, 0.001, 200),
            "rsi": np.random.uniform(30, 70, 200),
            "volatility": np.random.uniform(0.001, 0.005, 200),
            "hour_sin": np.sin(2 * np.pi * dates.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dates.hour / 24),
            "day_of_week": dates.dayofweek,
        },
        index=dates,
    )


@pytest.mark.skipif(not has_tft, reason="TFT dependencies not installed")
class TestTFTPredictor:
    """Tests para TFTPredictor."""

    def test_train_smoke(self, tft_sample_df: pd.DataFrame, tmp_path: object) -> None:
        """Smoke test: entrena 1 epoca y verifica artifacts."""
        model = TFTPredictor()
        model.max_epochs = 1
        model.batch_size = 32
        model.num_workers = 0
        model.enable_probability_calibration = False
        model.trainer_params["default_root_dir"] = str(tmp_path)

        meta = model.train(tft_sample_df, target_col="close")

        assert meta.model_type == "tft_pytorch_improved"
        assert "val_loss" in meta.metrics
        assert model.model is not None

        save_path = tmp_path / "model_tft"
        model.save(save_path)

        assert (save_path / "dataset_params.joblib").exists()
        assert (save_path / "model.pt").exists()

        loaded = TFTPredictor()
        loaded.num_workers = 0
        loaded.load(save_path)

        assert loaded.model is not None

        pred = loaded.predict(tft_sample_df)
        assert 0 <= pred.risk_score <= 1
        assert pred.p10 < pred.p90

    def test_insufficient_data(self) -> None:
        """Falla si hay menos datos que encoder_length."""
        model = TFTPredictor()
        df = pd.DataFrame(
            {"close": [100] * 10},
            index=pd.date_range("2023-01-01", periods=10, freq="5min"),
        )
        df["group_id"] = "0"
        df["day_of_week"] = 0

        with pytest.raises(ValueError, match="Datos insuficientes"):
            model.train(df)
