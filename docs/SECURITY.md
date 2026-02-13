# Security

## Secrets

- Keep API and Telegram credentials only in `.env` (local) or Kubernetes `Secret`.
- Do not commit `.env`, private keys, or live exchange credentials.

## Runtime Controls

- `LIVE_MODE=false` by default.
- `PAPER_MODE=true` by default.
- Enforce explicit confirmation before wiring live trading logic.

## Kubernetes Hardening

- Isolated namespace: `ml`.
- Dedicated service account: `bot-cripto-sa`.
- Namespace-scoped RBAC with read-only permissions needed by jobs.
- PVC mounts restricted to data/models/logs.

## Network and Exposure

- No public service exposure required for cron-based inference.
- Restrict egress if cluster policy allows (only exchange + Telegram APIs).

## Operational Safeguards

- Keep image tags immutable in production (avoid `latest`).
- Rotate Telegram token and exchange keys regularly.
- Monitor failed jobs and alert on repeated errors.
