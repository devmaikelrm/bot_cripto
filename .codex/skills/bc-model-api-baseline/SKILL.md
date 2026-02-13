---
name: bc-model-api-baseline
description: Maintain Bot Cripto model contracts and baseline model implementations with consistent train/predict/save/load behavior. Use for model API and baseline improvements.
---

# bc-model-api-baseline

1. Keep model contract in `src/bot_cripto/models/base.py`.
2. Ensure every predictor implements:
   - `train(df, target_col)`
   - `predict(df)`
   - `save(path)`
   - `load(path)`
3. Keep baseline implementation in `src/bot_cripto/models/baseline.py`.
4. Return standardized `PredictionOutput` fields for all models.
5. Preserve contract tests in `tests/test_models_contract.py` and baseline tests in `tests/test_baseline.py`.
