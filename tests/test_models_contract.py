"""Contract tests para BasePredictor.

Verifica que cualquier implementación concreta cumple el contrato:
- train() retorna ModelMetadata
- predict() retorna PredictionOutput
- save()/load() no levantan excepciones
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd

from bot_cripto.models.base import (
    BasePredictor,
    ModelMetadata,
    PredictionOutput,
)


class TestPredictionOutput:
    """Tests para la dataclass PredictionOutput."""

    def test_creation(self) -> None:
        """Se puede crear con todos los campos."""
        output = PredictionOutput(
            prob_up=0.67,
            expected_return=0.008,
            p10=-0.004,
            p50=0.006,
            p90=0.013,
            risk_score=0.22,
        )
        assert output.prob_up == 0.67
        assert output.risk_score == 0.22

    def test_frozen(self) -> None:
        """PredictionOutput es inmutable."""
        output = PredictionOutput(
            prob_up=0.5, expected_return=0.0, p10=0.0, p50=0.0, p90=0.0, risk_score=0.0
        )
        try:
            output.prob_up = 0.9  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except AttributeError:
            pass  # Correcto: es frozen

    def test_to_dict(self) -> None:
        """to_dict retorna todos los campos."""
        output = PredictionOutput(
            prob_up=0.67,
            expected_return=0.008,
            p10=-0.004,
            p50=0.006,
            p90=0.013,
            risk_score=0.22,
        )
        d = output.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 6
        assert d["prob_up"] == 0.67

    def test_has_all_signal_fields(self) -> None:
        """Verifica que PredictionOutput tiene los 6 campos del spec."""
        field_names = {f.name for f in fields(PredictionOutput)}
        required = {"prob_up", "expected_return", "p10", "p50", "p90", "risk_score"}
        assert required.issubset(field_names)


class TestModelMetadata:
    """Tests para la dataclass ModelMetadata."""

    def test_create_factory(self) -> None:
        """Factory method genera timestamp y git commit."""
        meta = ModelMetadata.create(
            model_type="baseline",
            version="0.1.0",
            metrics={"accuracy": 0.65},
        )
        assert meta.model_type == "baseline"
        assert meta.version == "0.1.0"
        assert meta.trained_at != ""
        assert isinstance(meta.metrics, dict)
        assert "accuracy" in meta.metrics


class TestBasePredictor:
    """Contract tests para la interfaz BasePredictor."""

    def test_is_abstract(self) -> None:
        """No se puede instanciar directamente."""
        try:
            BasePredictor()  # type: ignore[abstract]
            raise AssertionError("Should not instantiate ABC")
        except TypeError:
            pass  # Correcto: es abstracta

    def test_has_required_methods(self) -> None:
        """Verifica que la clase abstracta define los 4 métodos del contrato."""
        methods = {"train", "predict", "save", "load"}
        abstract_methods = set(BasePredictor.__abstractmethods__)
        assert methods.issubset(abstract_methods)

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta puede instanciarse."""

        class DummyModel(BasePredictor):
            def train(self, df: pd.DataFrame, target_col: str) -> ModelMetadata:
                return ModelMetadata.create("dummy", "0.0.1", {"test": 1.0})

            def predict(self, df: pd.DataFrame) -> PredictionOutput:
                return PredictionOutput(
                    prob_up=0.5,
                    expected_return=0.0,
                    p10=-0.01,
                    p50=0.0,
                    p90=0.01,
                    risk_score=0.5,
                )

            def save(self, path: Path) -> None:
                pass

            def load(self, path: Path) -> None:
                pass

        model = DummyModel()
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})

        meta = model.train(df, "close")
        assert isinstance(meta, ModelMetadata)

        pred = model.predict(df)
        assert isinstance(pred, PredictionOutput)
        assert 0.0 <= pred.prob_up <= 1.0
