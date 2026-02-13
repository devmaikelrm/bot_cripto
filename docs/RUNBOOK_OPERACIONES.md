# Runbook de Operaciones (Linux Nativo)

Este documento define como operar Bot Cripto en produccion local/Linux, con rutinas diarias, controles previos, respuesta a incidentes y recuperacion.

## 1. Precondiciones

- Servicio instalado en host Linux (`systemd`).
- Proyecto en `/opt/bot-cripto` (o ruta equivalente).
- Variables en `/etc/bot-cripto/bot-cripto.env`.
- Usuario de servicio no privilegiado (ej. `botcripto`).

## 2. Checklist de arranque inicial

1. Verificar entorno:
```bash
sudo test -f /etc/bot-cripto/bot-cripto.env && echo ok
sudo ls -l /etc/bot-cripto/bot-cripto.env
```
2. Verificar permisos sensibles:
```bash
sudo chmod 600 /etc/bot-cripto/bot-cripto.env
```
3. Verificar binarios y venv:
```bash
cd /opt/bot-cripto
source .venv/bin/activate
python -m bot_cripto.cli --help
```
4. Verificar timers:
```bash
systemctl status bot-cripto-inference.timer
systemctl status bot-cripto-retrain.timer
```
5. Verificar write paths:
```bash
test -d /opt/bot-cripto/data && echo data_ok
test -d /opt/bot-cripto/models && echo models_ok
test -d /var/log/bot-cripto && echo logs_ok
```

## 3. Operacion diaria (checklist)

1. Estado de timers/servicios:
```bash
systemctl list-timers | grep bot-cripto
systemctl status bot-cripto-inference.service --no-pager
```
2. Ultima senal:
```bash
cat /var/log/bot-cripto/signal.json
```
3. Validar decision/riesgo:
- `decision` esperado: `LONG|SHORT|NO_TRADE`
- revisar `regime`, `risk_allowed`, `position_size`
4. Validar salud de historicos:
```bash
ls -lh /var/log/bot-cripto/performance_history_*.json
ls -lh /var/log/bot-cripto/risk_state_*.json
```
5. Drift semanal:
```bash
cd /opt/bot-cripto
source .venv/bin/activate
bot-cripto detect-drift --history-file /var/log/bot-cripto/performance_history_BTC_USDT.json
```

## 4. Operacion manual (on-demand)

Pipeline completo:
```bash
cd /opt/bot-cripto
source .venv/bin/activate
bot-cripto fetch --days 30
bot-cripto features
bot-cripto train-trend
bot-cripto train-return
bot-cripto train-risk
bot-cripto run-inference
```

Backtest:
```bash
bot-cripto backtest --folds 4
```

## 5. Politica de seguridad operativa

- Mantener `LIVE_MODE=false` por defecto.
- Para habilitar live, exigir:
  - `LIVE_MODE=true`
  - `LIVE_CONFIRM_TOKEN=I_UNDERSTAND_LIVE_TRADING`
- Rotar tokens (Telegram/exchange) periodicamente.
- Nunca guardar secretos en git.

## 6. Matriz de incidentes y respuesta

### Incidente A: No hay `signal.json`

Diagnostico:
```bash
journalctl -u bot-cripto-inference.service -n 100 --no-pager
```
Acciones:
1. Verificar dataset features existente.
2. Verificar modelos entrenados en `models/trend|return|risk/...`.
3. Ejecutar inferencia manual.

### Incidente B: Error "Feature dataset not found"

Acciones:
1. Rehacer ingestion:
```bash
bot-cripto fetch --days 30
```
2. Regenerar features:
```bash
bot-cripto features
```
3. Reintentar inferencia.

### Incidente C: Error "No trend/return/risk models or baseline model found"

Acciones:
```bash
bot-cripto train-trend
bot-cripto train-return
bot-cripto train-risk
bot-cripto run-inference
```

### Incidente D: `decision=NO_TRADE` persistente

Causas probables:
- Regimen `RANGE` o `HIGH_VOL`
- `risk_allowed=false`
- Umbrales demasiado estrictos

Acciones:
1. Revisar `regime`, `regime_reason`, `risk_allowed`.
2. Ajustar cuidadosamente:
   - `PROB_MIN`
   - `MIN_EXPECTED_RETURN`
   - `RISK_MAX`
3. Volver a medir con backtest antes de cambios permanentes.

### Incidente E: Drift detectado

Acciones:
1. Ejecutar retrain completo.
2. Comparar backtest antes/despues.
3. Si no mejora, volver al ultimo modelo estable (rollback de carpeta versionada).

### Incidente F: Telegram no envia

Diagnostico:
- Revisar `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.
- Revisar logs del job de inferencia.

Acciones:
1. Corregir credenciales.
2. Probar envio con una inferencia manual.

## 7. Procedimiento de rollback

1. Identificar version estable:
```bash
ls -1 /opt/bot-cripto/models/trend/BTC_USDT
ls -1 /opt/bot-cripto/models/return/BTC_USDT
ls -1 /opt/bot-cripto/models/risk/BTC_USDT
```
2. Renombrar version defectuosa o ajustar selector de ultima version.
3. Reejecutar inferencia y validar `signal.json`.

## 8. SLO/SLI operativos sugeridos

- SLI disponibilidad inferencia: porcentaje de ejecuciones con `signal.json` generado.
- SLI latencia inferencia: tiempo total por corrida.
- SLI calidad: tasa de drift detectado por ventana.

Objetivos iniciales:
- Disponibilidad inferencia >= 99%
- Falla continua maxima sin alerta < 15 minutos

## 9. Checklist de cambio en configuracion

Antes de cambiar `.env`:
1. Guardar backup del archivo actual.
2. Aplicar cambio minimo (una variable por vez si es sensible).
3. Ejecutar inferencia manual.
4. Validar salida en `signal.json`.
5. Registrar fecha, cambio y motivo.

## 10. Comandos de soporte rapido

Logs inferencia:
```bash
journalctl -u bot-cripto-inference.service -f
```

Disparar retrain manual:
```bash
sudo systemctl start bot-cripto-retrain.service
```

Disparar inferencia manual:
```bash
sudo systemctl start bot-cripto-inference.service
```

## 11. Deploy recomendado (release + swap via git)

Si operas en `systemd --user` (como en tu VPS), el deploy mas robusto es: clonar nuevo release y hacer swap, manteniendo `data/models/logs` fuera del codigo.

Desde tu PC (Windows/PowerShell):

```powershell
$env:VPS_HOST="100.64.x.x"
$env:VPS_USER="maikelrm95"
# Opcionales:
# $env:VPS_REMOTE_DIR="/home/maikelrm95/bot-cripto"
# $env:VPS_STATE_DIR="/home/maikelrm95/bot-cripto-state"
# $env:VPS_GIT_REPO="https://git.kl3d.uy/maikelrm95/Crypto.git"
# $env:VPS_GIT_BRANCH="main"
python scripts/deploy_auto.py --git-swap
```

Este modo detiene timers/servicios, hace backup del release anterior, mantiene estado persistente y reinicia todo.
