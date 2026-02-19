# Sentiment Stack - Plan Implementable (Repo Actual)

Fecha: 2026-02-19  
Objetivo: convertir la visión de `docs/sentiment-stack-mejorado.md` en un plan ejecutable sin romper la arquitectura actual.

## Alcance real

Este plan mantiene la estructura existente:

- `src/bot_cripto/data/*`
- `src/bot_cripto/jobs/inference.py`
- `src/bot_cripto/features/engineering.py`
- `src/bot_cripto/cli.py`

No introduce en esta fase:

- migración a `src/bot_cripto/sentiment/*`
- Redis/PostgreSQL obligatorios
- cambio de framework CLI

## Estado actual ya implementado

1. Fuentes activas:
- `x`, `telegram`, `cryptopanic`, `api`, `local`, `fear_greed` fallback.

2. NLP:
- scorer con `finBERT` y fallback léxico.

3. Agregación:
- blend `x/news/telegram` con reponderación.
- `EMA` + `velocity`.

4. Realtime:
- `stream-capture` con `cryptofeed` (fallback `poll`).

## Próximas 3 implementaciones recomendadas

### 1) Source reliability scoring

Estado: Implementado

Objetivo: ponderar fuentes por calidad observada, no solo pesos fijos.

Archivos a tocar:
- `src/bot_cripto/data/quant_signals.py`
- `src/bot_cripto/core/config.py`
- `tests/test_quant_signals.py`

Cambios:
- Añadir score de confiabilidad por fuente (`0..1`) basado en:
  - volumen reciente
  - estabilidad (varianza extrema)
  - tasa de fallos por fuente
- Aplicar factor de confiabilidad sobre pesos base:
  - `peso_final = peso_config * confiabilidad`
- Re-normalizar pesos activos.

Acceptance criteria:
- si una fuente falla repetidamente, su contribución cae automáticamente.
- tests cubren reponderación por confiabilidad.

### 2) Sentiment anomaly score

Estado: Implementado

Objetivo: detectar picos anómalos de sentimiento y exponerlo a inferencia.

Archivos a tocar:
- `src/bot_cripto/data/quant_signals.py`
- `src/bot_cripto/jobs/inference.py`
- `tests/test_quant_signals.py`

Cambios:
- mantener una ventana corta de `social_sentiment_raw` por símbolo.
- calcular z-score robusto (MAD) o std si no hay MAD suficiente.
- exponer:
  - `social_sentiment_anomaly` (`0..1`)
  - `social_sentiment_zscore` (signed)

Acceptance criteria:
- en régimen normal, anomaly bajo.
- en shock de sentimiento, anomaly sube y queda auditado en `signal.json`.

### 3) Backtest A/B con y sin sentiment

Estado: Implementado (versión inicial)

Objetivo: validar impacto real de sentiment en métricas de trading.

Archivos a tocar:
- `src/bot_cripto/cli.py`
- `src/bot_cripto/backtesting/*` (mínimo posible)
- `docs/IMPROVEMENTS_APPLIED.md`

Cambios:
- comando nuevo:
  - `bot-cripto backtest-ab-sentiment --symbol BTC/USDT --timeframe 5m`
- ejecutar dos corridas equivalentes:
  - A: sentiment desactivado (neutral fijo)
  - B: sentiment activado (pipeline actual)
- reporte comparativo:
  - retorno neto
  - sharpe
  - max drawdown
  - win rate

Acceptance criteria:
- salida JSON clara con delta A vs B.
- decisión objetiva sobre mantener/ajustar sentiment.

## Orden de ejecución recomendado

1. Source reliability (bajo riesgo, impacto directo).
2. Anomaly score (mejora seguridad operativa).
3. Backtest A/B (validación cuantitativa).

## Variables sugeridas nuevas (cuando se implemente)

- `SOCIAL_SENTIMENT_RELIABILITY_ENABLED=true`
- `SOCIAL_SENTIMENT_RELIABILITY_MIN_WEIGHT=0.10`
- `SOCIAL_SENTIMENT_ANOMALY_WINDOW=96`
- `SOCIAL_SENTIMENT_ANOMALY_Z_CLIP=4.0`

## Comandos de validación

```bash
pytest -q tests/test_quant_signals.py tests/test_inference_context.py
bot-cripto fetch-sentiment --symbol BTC/USDT --source auto
bot-cripto run-inference --symbol BTC/USDT --timeframe 5m
```

## Decisión práctica

`docs/sentiment-stack-mejorado.md` queda como documento de visión.
Este archivo (`SENTIMENT_STACK_IMPLEMENTABLE_PLAN.md`) es el plan operativo real para tu código actual.
