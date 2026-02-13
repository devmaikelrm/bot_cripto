# Como Funciona Todo (Bot Cripto)

Este documento explica el funcionamiento completo del sistema, desde la descarga de datos hasta la generacion de senales y ejecucion en modo paper/live (guardado).

## 1. Objetivo del sistema

- Mercado inicial: `BTC/USDT`
- Timeframe: `5m`
- Horizonte de prediccion: `5` velas
- Salida principal: `logs/signal.json`

Contrato de salida esperado:

- `ts`
- `symbol`
- `timeframe`
- `horizon_steps`
- `prob_up`
- `expected_return`
- `p10`, `p50`, `p90`
- `risk_score`
- `decision` (`LONG|SHORT|NO_TRADE`)
- `confidence`
- `reason`
- `regime`, `regime_reason`
- `position_size`, `risk_allowed`
- `version.git_commit`, `version.model_version`

## 2. Arquitectura por capas

Flujo logico:

`data -> features -> models -> ensemble -> regime -> risk -> decision -> execution -> notifications`

### 2.1 Data

Archivo principal:
- `src/bot_cripto/data/ingestion.py`

Responsabilidad:
- Descargar OHLCV desde exchange (CCXT)
- Validar estructura minima
- Guardar parquet en `data/raw`
- Soportar proveedores via adapter (`binance`/`yfinance`)

Salida:
- `data/raw/{symbol}_{timeframe}.parquet`

### 2.2 Features

Archivo principal:
- `src/bot_cripto/features/engineering.py`

Responsabilidad:
- Calcular indicadores: RSI, MACD, ATR, volatilidad, volumen relativo, features de tiempo
- Limpiar NaNs por ventanas rolling

Salida:
- `data/processed/{symbol}_{timeframe}_features.parquet`

### 2.3 Modelos

Archivos:
- `src/bot_cripto/models/base.py` (contrato)
- `src/bot_cripto/models/baseline.py` (RandomForest + modos por objetivo)
- `src/bot_cripto/models/tft.py` (modelo secuencial)
- `src/bot_cripto/models/calibration.py` (calibracion de probabilidad)

Contrato unificado (`PredictionOutput`):
- `prob_up`, `expected_return`, `p10`, `p50`, `p90`, `risk_score`

Calibracion:
- Se aplica calibracion formal (`isotonic` o `platt`) sobre `prob_up`.
- Objetivo: que probabilidades entre modelos (Baseline/TFT) sean comparables antes del ensemble.
- En TFT la calibracion usa holdout temporal estricto para evitar look-ahead bias.

### 2.4 Entrenamiento por jobs

Archivos:
- `src/bot_cripto/jobs/train_trend.py`
- `src/bot_cripto/jobs/train_return.py`
- `src/bot_cripto/jobs/train_risk.py`

Comportamiento:
- Cargan dataset de features
- Entrenan modelo baseline con objetivo especializado:
  - `trend`
  - `return`
  - `risk`
- Guardan version en `models/{job}/{symbol}/{timestamp_commit}`

### 2.5 Ensemble

Archivo:
- `src/bot_cripto/models/ensemble.py`

Responsabilidad:
- Combinar salidas de trend/return/risk en una prediccion unica (`WeightedEnsemble`)

### 2.6 Regime Engine

Archivo:
- `src/bot_cripto/regime/engine.py`

Responsabilidad:
- Clasificar mercado en:
  - `TREND`
  - `RANGE`
  - `HIGH_VOL`
- Usa reglas ADX/ATR

### 2.7 Risk Engine

Archivos:
- `src/bot_cripto/risk/engine.py`
- `src/bot_cripto/risk/state_store.py`

Responsabilidad:
- Calcular `position_size` dinamico
- Bloquear operacion por:
  - drawdown diario/semanal
  - riesgo de prediccion alto
  - regimen `HIGH_VOL`
- Persistir estado de equity entre ejecuciones

Estado persistente:
- `logs/risk_state_<symbol>.json`
- `logs/paper_risk_state.json`

### 2.8 Decision Engine

Archivo:
- `src/bot_cripto/decision/engine.py`

Responsabilidad:
- Transformar prediccion en accion (`BUY`, `SELL`, `HOLD`)
- Umbrales por env:
  - `PROB_MIN`
  - `MIN_EXPECTED_RETURN`
  - `RISK_MAX`

En inferencia final se mapea a:
- `BUY -> LONG`
- `SELL -> SHORT`
- `HOLD o bloqueos -> NO_TRADE`

### 2.9 Inference Job

Archivo:
- `src/bot_cripto/jobs/inference.py`

Responsabilidad:
- Cargar ultimos modelos (`trend`, `return`, `risk`, fallback `baseline`)
- Generar prediccion combinada
- Aplicar filtros de regimen y riesgo
- Emitir `signal.json`
- Persistir metrica de performance para drift
- Enviar notificacion Telegram (si esta configurado)

Archivos de salida:
- `logs/signal.json`
- `logs/performance_history_<symbol>.json`

### 2.10 Execution

Archivos:
- `src/bot_cripto/execution/paper.py`
- `src/bot_cripto/execution/live.py`

#### Paper
- Simula entradas/salidas
- Aplica costos realistas:
  - fees (`FEES_BPS`)
  - spread (`SPREAD_BPS`)
  - slippage (`SLIPPAGE_BPS`)
- Guarda equity y performance

#### Live (guardado)
- No ejecuta orden real todavia
- Requiere para habilitar:
  - `LIVE_MODE=true`
  - `LIVE_CONFIRM_TOKEN=I_UNDERSTAND_LIVE_TRADING`
- Bloquea si excede limite diario (`LIVE_MAX_DAILY_LOSS`)
- En `BUY`, exige validacion de hard stop por `p10` (`HARD_STOP_MAX_LOSS`) antes de marcar la orden como lista.

### 2.11 Monitoring y drift

Archivos:
- `src/bot_cripto/monitoring/drift.py`
- `src/bot_cripto/monitoring/performance_store.py`

Responsabilidad:
- Detectar caida relativa de performance entre ventana baseline y reciente
- Entrada desde historial persistido

### 2.12 Notificaciones

Archivo:
- `src/bot_cripto/notifications/telegram.py`

Responsabilidad:
- Envio de mensajes Telegram
- Rate limit simple
- Usado por inferencia y jobs

### 2.13 Dashboard Watchtower

Archivo:
- `src/bot_cripto/ui/dashboard.py`

Responsabilidad:
- Monitorear salud de datos (progreso, gaps, frescura)
- Monitorear ML Ops (curvas de metrica y calibracion)
- Monitorear ejecucion (equity, decisiones, drawdown)
- Monitorear salud de API (latencia y estado)

## 3. Estructura de artefactos

- Raw data: `data/raw/{symbol}_{tf}.parquet`
- Features: `data/processed/{symbol}_{tf}_features.parquet`
- Modelos: `models/{trend|return|risk|baseline}/{symbol}/{version}/`
- Senal: `logs/signal.json`
- Risk state: `logs/risk_state_<symbol>.json`
- Performance history: `logs/performance_history_<symbol>.json`

## 4. Configuracion (env)

Variables clave:

- Mercado: `EXCHANGE`, `SYMBOLS`, `TIMEFRAME`
- Prediccion: `PRED_HORIZON_STEPS`, `ENCODER_LENGTH`
- Umbrales: `PROB_MIN`, `MIN_EXPECTED_RETURN`, `RISK_MAX`
- Costos: `FEES_BPS`, `SPREAD_BPS`, `SLIPPAGE_BPS`
- Riesgo: `RISK_PER_TRADE`, `MAX_DAILY_DRAWDOWN`, `MAX_WEEKLY_DRAWDOWN`, `MAX_POSITION_SIZE`
- Regimen: `REGIME_ADX_TREND_MIN`, `REGIME_ATR_HIGH_VOL_PCT`
- Runtime: `PAPER_MODE`, `LIVE_MODE`, `LIVE_CONFIRM_TOKEN`, `LIVE_MAX_DAILY_LOSS`
- Paths: `DATA_DIR_RAW`, `DATA_DIR_PROCESSED`, `MODELS_DIR`, `LOGS_DIR`
- Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## 5. Flujo operativo recomendado (Linux nativo)

1. `bot-cripto fetch --days 30`
2. `bot-cripto features`
3. `bot-cripto train-trend`
4. `bot-cripto train-return`
5. `bot-cripto train-risk`
6. `bot-cripto run-inference`

Monitoreo:

- Backtest: `bot-cripto backtest --folds 4`
- Drift: `bot-cripto detect-drift --history-file ./logs/performance_history_BTC_USDT.json`

## 6. CLI disponible

Comandos:

- `info`
- `fetch`
- `features`
- `train`
- `predict`
- `train-trend`
- `train-return`
- `train-risk`
- `run-inference`
- `backtest`
- `detect-drift`

## 7. Seguridad operacional

- Mantener `LIVE_MODE=false` por defecto
- No versionar `.env`
- Restringir permisos del archivo de entorno (`chmod 600`)
- Ejecutar con usuario no privilegiado en Linux

## 8. Troubleshooting rapido

- Error "Feature dataset not found": ejecutar `fetch` y luego `features`
- Error de modelos faltantes: correr `train-trend`, `train-return`, `train-risk`
- `decision=NO_TRADE`: revisar `regime`, `risk_allowed` y umbrales
- Drift falso por poco historial: aumentar ventanas y guardar mas puntos

## 9. Archivos de referencia

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/SECURITY.md`
- `docs/IMPROVEMENTS_APPLIED.md`
