---
name: bc-functional-roadmap
description: Evaluate Bot Cripto functional gaps and prioritize the next implementations using impact, risk, complexity, and dependency scoring. Use when deciding what to build next for trading quality, robustness, and production readiness.
---

# bc-functional-roadmap

1. Start from current runtime capabilities and identify missing functional blocks.
2. Score each candidate implementation on a 1-5 scale:
   - user_value (improves signal quality or operational reliability)
   - risk_reduction (reduces losses/failures/security incidents)
   - complexity (implementation cost; higher means harder)
   - dependency_blocker (unblocks other features)
   - observability_gain (improves debug/monitoring quality)
3. Compute priority score:
   - `priority = user_value*0.30 + risk_reduction*0.30 + dependency_blocker*0.20 + observability_gain*0.10 - complexity*0.10`
4. Group items by milestone:
   - M1 critical foundation
   - M2 performance and quality
   - M3 advanced optimization
5. For each prioritized item, output:
   - why it matters
   - expected measurable outcome
   - concrete implementation paths/files
   - tests and acceptance checks
   - rollback strategy
6. Enforce practical constraints:
   - keep config externalized (`.env` / settings)
   - preserve `signal.json` contract
   - avoid introducing live-trading by default
7. Produce final roadmap as a numbered implementation queue with effort estimate and sequencing dependencies.
