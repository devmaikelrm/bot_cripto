---
name: bc-inference-decision-paper
description: Implement Bot Cripto inference orchestration, decision logic, and paper-mode execution. Use for runtime signal generation and paper trading behavior.
---

# bc-inference-decision-paper

1. Run inference from `src/bot_cripto/jobs/inference.py`.
2. Load latest trend/return/risk models and merge outputs into standard signal fields.
3. Apply decision rules from `src/bot_cripto/decision/engine.py`.
4. Keep paper execution behavior in `src/bot_cripto/execution/paper.py`.
5. Persist `logs/signal.json` and paper trade reports for auditability.
