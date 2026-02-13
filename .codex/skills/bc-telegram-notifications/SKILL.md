---
name: bc-telegram-notifications
description: Implement and maintain Telegram notifications for Bot Cripto job lifecycle, errors, metrics, and signals. Use when modifying messaging or alert reliability.
---

# bc-telegram-notifications

1. Keep notification code in `src/bot_cripto/notifications/telegram.py`.
2. Provide `tg_send` and `tg_send_markdown` helpers.
3. Apply rate limiting to avoid Telegram API flood and duplicate alerts.
4. Do not fail pipeline hard if Telegram send fails.
5. Keep token/chat IDs externalized via `.env` or Kubernetes `Secret`.
