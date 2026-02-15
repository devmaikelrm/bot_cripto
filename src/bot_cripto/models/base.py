"""Contrato base para todos los modelos de predicción.

Define la interfaz ABC que deben implementar BaselineModel, TFTModel,
y cualquier modelo futuro. Incluye dataclasses para la salida estándar.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PredictionOutput:
    """Salida estandarizada de un modelo de predicción.

    Corresponde a los campos del signal.json definido en el spec.
    """

    prob_up: float
    """Probabilidad de subida en el horizonte (0.0 a 1.0)."""

    expected_return: float
    """Retorno esperado en el horizonte (decimal, ej: 0.008 = 0.8%)."""

    p10: float
    """Percentil 10 del retorno (escenario pesimista)."""

    p50: float
    """Percentil 50 del retorno (mediana)."""

    p90: float
    """Percentil 90 del retorno (escenario optimista)."""

    risk_score: float
    """Probabilidad de evento adverso / drawdown (0.0 a 1.0)."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.prob_up <= 1.0):
            raise ValueError(f"prob_up must be in [0, 1], got {self.prob_up}")
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score must be in [0, 1], got {self.risk_score}")
        if self.p10 > self.p90:
            raise ValueError(
                f"p10 ({self.p10}) must be <= p90 ({self.p90})"
            )

    def to_dict(self) -> dict[str, float]:
        """Serializa a diccionario para JSON."""
        return {
            "prob_up": self.prob_up,
            "expected_return": self.expected_return,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "risk_score": self.risk_score,
        }


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata de un modelo entrenado.

    Se guarda junto con los artefactos del modelo para trazabilidad.
    """

    model_type: str
    """Tipo de modelo: 'baseline', 'tft', etc."""

    version: str
    """Versión semántica o timestamp del modelo."""

    git_commit: str
    """Hash del commit de Git en el momento del entrenamiento."""

    trained_at: str
    """Timestamp ISO del entrenamiento."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Métricas de evaluación del modelo (accuracy, rmse, etc.)."""

    @staticmethod
    def current_git_commit() -> str:
        """Obtiene el commit hash actual de Git."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return "unknown"

    @classmethod
    def create(cls, model_type: str, version: str, metrics: dict[str, float]) -> ModelMetadata:
        """Factory method con timestamp y git commit automáticos."""
        return cls(
            model_type=model_type,
            version=version,
            git_commit=cls.current_git_commit(),
            trained_at=datetime.now(tz=UTC).isoformat(),
            metrics=metrics,
        )


class BasePredictor(ABC):
    """Interfaz abstracta para todos los modelos de predicción.

    Cada modelo concreto (BaselineModel, TFTModel, etc.) debe implementar
    estos 4 métodos. Esto permite intercambiar modelos sin cambiar el pipeline.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, target_col: str) -> ModelMetadata:
        """Entrena el modelo con datos procesados.

        Args:
            df: DataFrame con features + target.
            target_col: Nombre de la columna objetivo.

        Returns:
            Metadata del modelo entrenado.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        """Genera predicción para los datos más recientes.

        Args:
            df: DataFrame con features (encoder_length filas mínimo).

        Returns:
            PredictionOutput con todos los campos.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Guarda el modelo entrenado en disco.

        Args:
            path: Directorio donde guardar artefactos.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Carga un modelo desde disco.

        Args:
            path: Directorio con los artefactos del modelo.
        """
