"""Champion-Challenger MVP evaluation for model promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bot_cripto.models.base import BasePredictor
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.tft import TFTPredictor


@dataclass(frozen=True)
class ModelEvalStats:
    trades: int
    total_net_return: float
    win_rate: float


@dataclass(frozen=True)
class ChampionChallengerReport:
    champion_path: str
    challenger_path: str
    champion: ModelEvalStats
    challenger: ModelEvalStats
    relative_improvement: float
    promote: bool
    reason: str


def _load_predictor_from_dir(path: Path) -> BasePredictor:
    if (path / "neuralforecast_adapter.joblib").exists():
        from bot_cripto.models.neuralforecast_adapter import NeuralForecastAdapter

        model = NeuralForecastAdapter(model_name="patchtst")
        model.load(path)
        return model
    if (path / "model.pt").exists():
        if (path / "residual_std.joblib").exists():
            from bot_cripto.models.nbeats import NBeatsPredictor

            model = NBeatsPredictor()
            model.load(path)
            return model
        model = TFTPredictor()
        model.load(path)
        return model
    model = BaselineModel()
    model.load(path)
    return model


def _signal_from_pred(prob_up: float, exp_ret: float, prob_min: float, min_ret: float) -> int:
    if exp_ret >= min_ret and prob_up >= prob_min:
        return 1
    if exp_ret <= -min_ret and prob_up <= (1.0 - prob_min):
        return -1
    return 0


def evaluate_predictor(
    predictor: BasePredictor,
    df: pd.DataFrame,
    start_idx: int,
    prob_min: float,
    min_expected_return: float,
    roundtrip_cost: float,
) -> ModelEvalStats:
    trades = 0
    wins = 0
    total_net = 0.0
    for idx in range(start_idx, len(df) - 1):
        window = df.iloc[: idx + 1]
        pred = predictor.predict(window)
        signal = _signal_from_pred(
            prob_up=float(pred.prob_up),
            exp_ret=float(pred.expected_return),
            prob_min=float(prob_min),
            min_ret=float(min_expected_return),
        )
        if signal == 0:
            continue
        c0 = float(df["close"].iloc[idx])
        c1 = float(df["close"].iloc[idx + 1])
        realized = (c1 - c0) / c0 if c0 != 0 else 0.0
        gross = signal * realized
        net = gross - roundtrip_cost
        total_net += net
        trades += 1
        wins += int(gross > 0.0)
    win_rate = float(wins / trades) if trades > 0 else 0.0
    return ModelEvalStats(trades=trades, total_net_return=float(total_net), win_rate=win_rate)


def run_champion_challenger_check(
    df: pd.DataFrame,
    champion_path: Path,
    challenger_path: Path,
    eval_window: int,
    prob_min: float,
    min_expected_return: float,
    roundtrip_cost: float,
    promotion_margin: float = 0.05,
    min_trades: int = 20,
) -> ChampionChallengerReport:
    champion_model = _load_predictor_from_dir(champion_path)
    challenger_model = _load_predictor_from_dir(challenger_path)

    start_idx = max(20, len(df) - int(eval_window))
    champ_stats = evaluate_predictor(
        predictor=champion_model,
        df=df,
        start_idx=start_idx,
        prob_min=prob_min,
        min_expected_return=min_expected_return,
        roundtrip_cost=roundtrip_cost,
    )
    chall_stats = evaluate_predictor(
        predictor=challenger_model,
        df=df,
        start_idx=start_idx,
        prob_min=prob_min,
        min_expected_return=min_expected_return,
        roundtrip_cost=roundtrip_cost,
    )

    base = champ_stats.total_net_return
    if abs(base) <= 1e-12:
        improvement = float("inf") if chall_stats.total_net_return > 0 else 0.0
    else:
        improvement = (chall_stats.total_net_return - base) / abs(base)

    promote = (
        chall_stats.trades >= min_trades
        and chall_stats.total_net_return > champ_stats.total_net_return
        and improvement >= promotion_margin
    )
    if chall_stats.trades < min_trades:
        reason = f"not_enough_trades:{chall_stats.trades}<{min_trades}"
    elif chall_stats.total_net_return <= champ_stats.total_net_return:
        reason = "challenger_not_better"
    elif improvement < promotion_margin:
        reason = f"improvement_below_margin:{improvement:.4f}<{promotion_margin:.4f}"
    else:
        reason = "promote_challenger"

    return ChampionChallengerReport(
        champion_path=str(champion_path),
        challenger_path=str(challenger_path),
        champion=champ_stats,
        challenger=chall_stats,
        relative_improvement=float(improvement),
        promote=bool(promote),
        reason=reason,
    )
