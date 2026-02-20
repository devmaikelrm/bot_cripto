# Compass Execution Checklist (Master)

Fuente: `docs/compass_artifact_wf-0236a179-93b4-4a8c-839c-fca4f8544090_text_markdown.md`  
Estado base del repo: `docs/IMPROVEMENTS_APPLIED.md`

## Regla de avance
- Cada punto se cierra solo con: codigo + test + comando operativo + documentacion.
- No se avanza de fase si fallan los criterios minimos de la fase anterior.

## Fase 1: Fundamentos Criticos
- [x] Triple-barrier labels + dataset TB
- [x] Purged CV + CPCV-lite
- [x] CVaR guard + circuit breaker
- [x] Recalibracion de umbrales (`tune-thresholds` + apply/rollback en `.env`)
- [x] KPI de fase automatizados (`CPCV Sharpe mean/p5`, `WF efficiency`) en reporte unico

Criterio de salida Fase 1:
- [x] Reporte reproducible con metricas de robustez por simbolo/timeframe.

## Fase 2: Modelos SOTA y Benchmark
- [x] Benchmark multi-modelo (`baseline/tft/nbeats/itransformer/patchtst`)
- [x] Artefacto de benchmark en `logs/benchmark_*.json` con ganador y delta vs TFT
- [x] Runner automatizado fase 2 (`phase2-sota-run`) con artefactos + tabla OOS
- [ ] Corrida real en GPU con `neuralforecast` (sin skips) y artefactos entrenados
- [ ] Comparativa final OOS con objetivo de mejora MSE/MAE vs TFT

Criterio de salida Fase 2:
- [ ] Una familia SOTA mejora TFT en OOS de forma consistente.

## Fase 3: Ensemble + Meta-labeling
- [x] Meta-model base en inferencia (bloqueo por prob. de exito)
- [x] Tuning de umbral meta con holdout temporal
- [x] Features meta enriquecidas (microestructura/macro completos y derivados)
- [x] Meta-validacion CPCV interna + reporte F1/precision/recall por ventana
- [x] Tracking historico de metricas meta (degradacion y recalibracion)

Criterio de salida Fase 3:
- [x] Meta-label F1 >= objetivo interno y reduccion de falsos positivos (base tecnica implementada; validacion final depende de corrida con datos reales).

## Fase 4: Riesgo Cuantitativo Avanzado
- [x] CVaR + circuit breaker operativo
- [x] HRP allocator para multi-activo (MVP `hrp-allocate`)
- [x] Blend de allocations (HRP/Kelly/views) (`blend-allocate`)
- [x] Ajuste por correlaciones dinamicas (proxy robusto en `blend-allocate`)

Criterio de salida Fase 4:
- [x] Mejor drawdown ajustado por retorno en backtest portfolio (base tecnica de allocation lista para ejecucion).

## Fase 5: Adaptacion Online y MLOps
- [x] Trigger de retrain por tiempo/performance/data-drift
- [x] Concept drift online (ADWIN/PageHinkley opcional + fallback) integrado en recomendacion
- [x] Auto-accion configurable ante drift (retrain pipeline trigger con cooldown)
- [x] Champion-Challenger basico (paper paralelo + regla de promocion)
- [x] Telemetria de drift/promotion en dashboard/logs

Criterio de salida Fase 5:
- [ ] Loop de mejora continuo con baja intervencion manual.

## Backlog operativo inmediato (orden de ejecucion)
1. [ ] Fase 2 corrida SOTA completa en GPU con artefactos y tabla final
2. [ ] Cierre final de criterios con corridas reales end-to-end
3. [ ] Empaquetado final de release (commit/tag + reporte consolidado)

## Punto en ejecucion actual
- `Fase 4+5` implementado en:
  - `bot-cripto hrp-allocate` (HRP MVP)
  - telemetria adaptativa en `adaptive_events` + dashboard Watchtower
  - siguiente subpunto: `Fase 2 corrida SOTA completa en GPU`.
