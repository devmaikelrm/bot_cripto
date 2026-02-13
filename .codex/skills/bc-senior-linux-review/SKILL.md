---
name: bc-senior-linux-review
description: Perform a senior-level critical review of Bot Cripto Linux-native production readiness across architecture, risk controls, failure handling, security, and observability with prioritized remediation.
---

# bc-senior-linux-review

1. Prioritize findings by severity (high/medium/low).
2. Focus review on:
   - inference correctness and output contract
   - regime/risk control behavior under stress
   - Linux service reliability (systemd units/timers)
   - secrets and host hardening
   - observability and incident debugging paths
   - test coverage and missing edge cases
3. Produce concrete remediation actions and suggested diffs.
4. Flag release blockers explicitly.
5. Keep style-only comments secondary to behavior and risk findings.
