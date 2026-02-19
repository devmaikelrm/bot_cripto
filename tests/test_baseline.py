"""Tests para Baseline Model."""

import numpy as np
import pandas as pd
import pytest

from bot_cripto.models.base import PredictionOutput
from bot_cripto.models.baseline import BaselineModel


@pytest.fixture
def sample_df():
    """DataFrame de prueba con tendencia clara."""
    # 200 filas para tener suficientes windows para TimeSeriesSplit
    dates = pd.date_range(start="2023-01-01", periods=200, freq="5min")

    # Onda senoidal + tendencia para tener subidas y bajadas
    x = np.linspace(0, 4 * np.pi, 200)
    close = 100 + 10 * np.sin(x) + np.linspace(0, 20, 200)

    # Agregar features necesarios
    df = pd.DataFrame(
        {
            "close": close,
            "rsi": np.random.uniform(30, 70, 200),
            "volatility": np.random.uniform(0.001, 0.005, 200),
            "log_ret": np.random.normal(0, 0.001, 200),
        },
        index=dates,
    )
    return df


@pytest.fixture
def temp_models_dir(tmp_path):
    """Directorio temporal para modelos."""
    d = tmp_path / "models"
    d.mkdir()
    return d


class TestBaselineModel:
    """Tests para BaselineModel."""

    def test_train_structure(self, sample_df):
        """Entrenamiento retorna metadata válida."""
        model = BaselineModel()
        meta = model.train(sample_df, target_col="close")

        assert meta.model_type == "baseline_rf"
        assert "accuracy_in_sample" in meta.metrics
        # Con tendencia alcista perfecta, accuracy debería ser alta
        assert meta.metrics["accuracy_in_sample"] > 0.5

    def test_predict_structure(self, sample_df):
        """Predicción retorna PredictionOutput válido."""
        model = BaselineModel()
        model.train(sample_df)

        # Predecir sobre el mismo DF (solo probamos estructura)
        pred = model.predict(sample_df)

        assert isinstance(pred, PredictionOutput)
        assert 0 <= pred.prob_up <= 1
        assert 0 <= pred.risk_score <= 1
        assert pred.p10 <= pred.p50 <= pred.p90

        # En tendencia alcista, expected return debería ser positivo
        # (aunque RF puede overfittear o no, es in-sample)
        # assert pred.expected_return > 0  <-- No garantizado por random init

    def test_save_load(self, sample_df, temp_models_dir):
        """Persistencia funciona correctamente."""
        model = BaselineModel()
        model.train(sample_df)

        save_path = temp_models_dir / "test_model"
        model.save(save_path)

        assert (save_path / "direction.joblib").exists()
        assert (save_path / "features.joblib").exists()

        # Cargar en otra instancia
        loaded_model = BaselineModel()
        loaded_model.load(save_path)

        assert loaded_model.features == model.features
        # Verificar que predice igual (con tolerancia a float precision)
        pred1 = model.predict(sample_df)
        pred2 = loaded_model.predict(sample_df)

        for k, v in pred1.__dict__.items():
            assert v == pytest.approx(pred2.__dict__[k], rel=1e-9)

    def test_insufficient_data(self):
        """Falla con error informativo si no hay datos."""
        df = pd.DataFrame({"close": [100, 101]}, index=[0, 1])
        model = BaselineModel()
        # Horizon default es 5, así que 2 filas no alcanzan para targets
        with pytest.raises(ValueError, match="No hay datos suficientes"):
            model.train(df)

    def test_triple_barrier_direction_used_for_trend(self, sample_df):
        df = sample_df.copy()
        # Alternate +1/-1 labels and some 0 (which should be dropped).
        labels = np.where(np.arange(len(df)) % 3 == 0, 0, np.where(np.arange(len(df)) % 2 == 0, 1, -1))
        df["tb_label"] = labels
        df["tb_ret"] = np.where(labels > 0, 0.01, np.where(labels < 0, -0.01, 0.0))

        model = BaselineModel(objective="trend")
        meta = model.train(df, target_col="close")
        assert meta.metrics["using_tb_label"] == 1.0
        assert "accuracy_in_sample" in meta.metrics

    def test_triple_barrier_return_used_for_return_objective(self, sample_df):
        df = sample_df.copy()
        df["tb_ret"] = 0.02  # force positive target return
        model = BaselineModel(objective="return")
        meta = model.train(df, target_col="close")
        assert meta.metrics["using_tb_return"] == 1.0
        pred = model.predict(df)
        assert pred.expected_return > 0
