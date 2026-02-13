# SSH Credentials (Template)

This repository must not contain real credentials.

Use environment variables (recommended) for tooling like `vps.py` or `scripts/deploy_auto.py`:

- `VPS_HOST` (e.g. `100.64.x.x` if using Tailscale)
- `VPS_USER` (e.g. `ubuntu`)
- `VPS_PASS` (optional; if not set you'll be prompted)
- `VPS_REMOTE_DIR` (optional; default `~/bot-cripto`)

If you previously committed passwords, rotate them immediately.

