# BOT-CRIPTO: Manual Maestro de Operaciones y Estrategia

Documento de referencia operacional y técnica del proyecto, con foco en entrenamiento/inferencia, gestión de riesgo y roadmap de precisión.

## 1. Infraestructura y Acceso (RunPod)

### Detalles de conexión (ejemplo de pod anterior)
- Host: `213.173.107.85`
- Puerto SSH: `19355`
- Usuario: `root`
- Autenticación: clave SSH privada `id_ed25519`
- Hardware usado: NVIDIA RTX 4090 (24GB VRAM)

### Comandos rápidos
- Ver logs entrenamiento:
  - `ssh -p <PORT> root@<HOST> "tail -f /workspace/logs/training.log"`
- Ver uso de GPU:
  - `ssh -p <PORT> root@<HOST> "nvidia-smi"`
- Ver monitor Telegram:
  - `ssh -p <PORT> root@<HOST> "tail -f /workspace/logs/telegram_monitor.log"`

## 2. Modelo Base (TFT)

### Por qué TFT
1. Selección dinámica de variables relevantes.
2. Atención temporal para contexto histórico no lineal.
3. Salida probabilística (cuantiles) útil para gestión de riesgo.

### Configuración actual relevante
- Entrenamiento con GPU (`accelerator=gpu`, `devices=1`, mixed precision).
- Arquitectura ajustada para robustez/VRAM.
- Salida con `p10/p50/p90`, `prob_up`, `expected_return`, `risk_score`.

## 3. Estrategia de Datos

### Doble horizonte
- 1h: sesgo macro/tendencia.
- 5m: timing de entrada/ejecución.

### Ventana histórica amplia
- Cubre múltiples regímenes (alcista, bajista, lateral).
- Reduce sobreajuste a un solo periodo.

## 4. Gestión de Riesgo

- Position sizing dinámico por motor de riesgo.
- Límites de drawdown diarios/semanales.
- Bloqueos por riesgo y por reglas operativas (`NO_TRADE`).

## 5. Hoja de Ruta para Máxima Precisión

Estado real al 2026-02-19 (implementación):

1. Análisis de sentimiento en tiempo real
- Estado: `COMPLETADO`
- Implementado:
  - pipeline NLP inicial (`finBERT`) con fallback seguro a léxico.
  - enrutamiento `SOCIAL_SENTIMENT_SOURCE=auto` con prioridad `nlp`.
  - blend ponderado `x/news/telegram` con reponderación automática.
  - suavizado temporal por `EMA` y métrica de `velocity`.
  - `fear_greed`.
  - `social_sentiment` por fuente configurable:
    - endpoint externo (`SOCIAL_SENTIMENT_ENDPOINT`)
    - X nativo (`X_BEARER_TOKEN`)
    - Telegram nativo (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_SENTIMENT_CHAT_IDS`)
    - CryptoPanic (`CRYPTOPANIC_API_KEY`)
    - RSS news multifuente (`SOCIAL_SENTIMENT_NEWS_RSS_URLS`)
    - archivo local `data/raw/social_sentiment_<SYMBOL>.json`
    - fallback seguro a neutral/FnG

2. Orderbook Imbalance
- Estado: `COMPLETADO`
- Implementado:
  - captura de profundidad de Binance.
  - feature `orderbook_imbalance` en inferencia.
  - gating de compra ante pared de venta configurable.

3. Correlación con SP500 y DXY
- Estado: `COMPLETADO`
- Implementado:
  - `sp500_ret_1d`, `dxy_ret_1d`.
  - `corr_btc_sp500`, `corr_btc_dxy`.
  - `macro_risk_off_score`.
  - integración en inferencia y decisión.

4. Fine-tuning de volatilidad (modo crisis)
- Estado: `COMPLETADO`
- Implementado:
  - activación por volatilidad realizada.
  - activación por ventana macro horaria (UTC) configurable.
  - `effective_regime = CRISIS_HIGH_VOL` cuando aplica.

5. Ingesta realtime de microestructura
- Estado: `COMPLETADO`
- Implementado:
  - captura websocket con `cryptofeed` (opcional).
  - fallback automático a polling REST si `cryptofeed` no está disponible.
  - persistencia de snapshots en `data/raw/stream/{symbol}_stream.parquet`.
  - retención configurable y escritura segura con lock.

## 6. Variables de Entorno de Precisión (nuevas)

- `MACRO_EVENT_CRISIS_ENABLED`
- `MACRO_EVENT_CRISIS_WINDOWS_UTC`
- `MACRO_EVENT_CRISIS_WEEKDAYS`
- `MACRO_BLOCK_THRESHOLD`
- `ORDERBOOK_SELL_WALL_THRESHOLD`
- `SOCIAL_SENTIMENT_BULL_MIN`
- `SOCIAL_SENTIMENT_BEAR_MAX`
- `CONTEXT_PROB_ADJUST_MAX`
- `SOCIAL_SENTIMENT_SOURCE`
- `SOCIAL_SENTIMENT_ENDPOINT`
- `CRYPTOPANIC_API_KEY`
- `SOCIAL_SENTIMENT_NLP_ENABLED`
- `SOCIAL_SENTIMENT_NLP_MODEL_ID`
- `SOCIAL_SENTIMENT_NLP_MAX_TEXTS`
- `SOCIAL_SENTIMENT_NEWS_RSS_ENABLED`
- `SOCIAL_SENTIMENT_NEWS_RSS_URLS`
- `SOCIAL_SENTIMENT_NEWS_RSS_MAX_ITEMS`
- `SOCIAL_SENTIMENT_WEIGHT_X`
- `SOCIAL_SENTIMENT_WEIGHT_NEWS`
- `SOCIAL_SENTIMENT_WEIGHT_TELEGRAM`
- `SOCIAL_SENTIMENT_EMA_ALPHA`
- `X_BEARER_TOKEN`
- `X_QUERY_TEMPLATE`
- `X_MAX_RESULTS`
- `TELEGRAM_SENTIMENT_CHAT_IDS`
- `TELEGRAM_SENTIMENT_LOOKBACK_LIMIT`
- `STREAM_SNAPSHOT_INTERVAL_SECONDS`
- `STREAM_ORDERBOOK_DEPTH`
- `STREAM_RETENTION_DAYS`

## 7. Estado del Sistema

- Entrenamientos clave (1h y 5m) ya completados y respaldados en `artifacts/runpod_backups/incremental`.
- Monitor de Telegram funcional con tablero de progreso.
- Backup incremental automático listo para ejecutar antes de apagar GPU.

## 8. Próximos pasos sugeridos

1. Al levantar nueva GPU: deploy del repo actualizado y smoke de inferencia.
2. Activar fuentes reales de sentimiento (X/Telegram/API).
3. Validar impacto en backtesting walk-forward y drift para recalibrar umbrales.
   - Implementado: `bot-cripto tune-thresholds --symbol BTC/USDT --timeframe 5m`.
   - Aplicación automática a `.env`: agregar `--apply-env` (crea backup).
   - Rollback: `bot-cripto rollback-thresholds-env`.
4. Champion-Challenger (paper paralelo) para promoción controlada de modelos.
   - Implementado: `bot-cripto champion-challenger-check --model-name trend --symbol BTC/USDT --timeframe 5m`.
