---
name: bc-linux-native-ops
description: Operate Bot Cripto on Linux native hosts using systemd services, timers, env files, and log rotation. Use for installation, rollout, upgrades, and runtime troubleshooting when Kubernetes is not used.
---

# bc-linux-native-ops

1. Use `scripts/deploy_linux_native.sh` for one-shot setup.
2. Install and verify systemd units in `systemd/`:
   - `bot-cripto-inference.service`
   - `bot-cripto-inference.timer`
   - `bot-cripto-retrain.service`
   - `bot-cripto-retrain.timer`
3. Keep runtime env in `/etc/bot-cripto/bot-cripto.env`.
4. Keep service logs under `/var/log/bot-cripto` with `systemd/bot-cripto.logrotate`.
5. Validate timers with:
   - `systemctl list-timers | grep bot-cripto`
   - `systemctl status bot-cripto-inference.timer`
   - `systemctl status bot-cripto-retrain.timer`
6. If deployment fails, check service logs and env permissions first.
