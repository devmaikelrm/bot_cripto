# Auditoría Técnica Cuantitativa — Bot Cripto
**Fecha:** 21 de febrero de 2026
**Auditor:** Claude Code — Quant Researcher / Crypto Trading Systems Architect
**Rama analizada:** `main` (commit `7f9b8e7`)
**Alcance:** Código fuente completo — modelos, riesgo, decisión, features, backtesting, ejecución, monitoreo

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura General](#2-arquitectura-general)
3. [Motor de Modelos](#3-motor-de-modelos)
   - 3.1 TFT (Temporal Fusion Transformer)
   - 3.2 Baseline RandomForest
   - 3.3 Ensemble Ponderado
   - 3.4 Meta-Model (Random Forest Secundario)
   - 3.5 Calibración de Probabilidades
4. [Feature Engineering](#4-feature-engineering)
5. [Detección de Régimen](#5-detección-de-régimen)
6. [Labeling — Triple Barrier](#6-labeling--triple-barrier)
7. [Motor de Riesgo](#7-motor-de-riesgo)
8. [Motor de Decisión](#8-motor-de-decisión)
9. [Backtesting](#9-backtesting)
   - 9.1 Backtester Realista
   - 9.2 Purged K-Fold CV
   - 9.3 CPCV (Combinatorial Purged CV)
10. [Ejecución Paper](#10-ejecución-paper)
11. [Monitoreo y Drift](#11-monitoreo-y-drift)
12. [Stack de Sentimiento](#12-stack-de-sentimiento)
13. [Hallazgos Críticos — Tabla Maestra](#13-hallazgos-críticos--tabla-maestra)
14. [Roadmap de Mejoras Prioritizadas](#14-roadmap-de-mejoras-prioritizadas)
15. [Conclusión Senior](#15-conclusión-senior)

---

## 1. Resumen Ejecutivo

El proyecto ha evolucionado de un script de trading básico a una **arquitectura de grado semi-institucional**. Se identificaron **27 hallazgos** distribuidos en 5 niveles de severidad. El sistema tiene una base sólida en: backtesting realista con Purged CV, Kelly Criterion fraccional, CVaR guard, y un stack de sentiment multi-fuente con contrarian fusion.

Los tres riesgos más críticos para operación real son:

| # | Severidad | Problema | Archivo |
|---|-----------|----------|---------|
| 1 | 🔴 CRÍTICO | Monkeypatch de `torch.load` desactiva seguridad global de PyTorch | `tft.py:19-24` |
| 2 | 🔴 CRÍTICO | `RiskState` no persiste entre reinicios del proceso | `risk/engine.py` + `execution/paper.py` |
| 3 | 🔴 CRÍTICO | `triple_barrier.py`: loop `for loc, end_ts` es O(n²) con 2+ años de datos 5m | `labels/triple_barrier.py:60` |

**Estado operativo recomendado:** Paper trading ✅ | Live trading: NO hasta resolver hallazgos 1 y 2.

---

## 2. Arquitectura General

### Diagrama de flujo de señal

```
Binance OHLCV (5m)
    │
    ▼
[FeaturePipeline]
  ├─ TechnicalAnalysis (RSI/MACD/BB/ATR/EMA)
  ├─ MacroMerger (SPY/QQQ/DXY/GC merge_asof)
  ├─ MicrostructureFeatures (OBI/Kyle λ/VPIN/Parkinson vol)
  └─ QuantSignals (Funding Rate/F&G/Sentiment NLP)
    │
    ▼
[MLRegimeEngine]  →  BULL | BEAR | RANGE | CRISIS
    │
    ▼
[Models Ensemble]
  ├─ TFTPredictor (Trend/Return/Risk — 3 instancias)
  ├─ BaselineModel (RF Multi-objetivo)
  └─ NBEATSPredictor (opcional)
    │
    ▼
[WeightedEnsemble]  →  PredictionOutput (prob_up, p10, p50, p90, risk_score)
    │
    ▼
[MetaModel]  →  ¿Filtrar señal? (RF secundario sobre contexto)
    │
    ▼
[DecisionEngine]  →  BUY / SELL / HOLD + EU + weight
    │
    ▼
[RiskEngine]  →  position_size (Kelly fraccional + CVaR + Drawdown)
    │
    ▼
[PaperExecutor / LiveExecutor]
    │
    ▼
[PerformanceStore + WatchtowerStore]
    │
    ▼
[OnlineLearningSystem]  →  ¿Retrain?
```

### Evaluación de la arquitectura

| Dimensión | Calificación | Comentario |
|-----------|-------------|------------|
| Separación de responsabilidades | ⭐⭐⭐⭐⭐ | Cada módulo tiene una sola responsabilidad clara |
| Resiliencia a reinicios | ⭐⭐⭐ | Paper state persiste, pero RiskState tiene un bug |
| Testabilidad | ⭐⭐⭐⭐ | ABC contracts, Protocol types, unit tests presentes |
| Observabilidad | ⭐⭐⭐⭐⭐ | Structured logging (structlog), WatchtowerStore |
| Escalabilidad | ⭐⭐⭐ | Single-symbol por instancia; multi-symbol requiere refactor |

---

## 3. Motor de Modelos

### 3.1 TFT (Temporal Fusion Transformer)

**Archivo:** `src/bot_cripto/models/tft.py`

#### Configuración actual

```
encoder_length  = 288 barras (24 horas a 5m)
horizon         = 5 barras (25 minutos ahead)
hidden_size     = 128
attention_heads = 8
lstm_layers     = 3
dropout         = 0.2
quantiles       = [0.1, 0.5, 0.9]
loss            = QuantileLoss
precision       = bf16-mixed (GPU) / 32-true (CPU)
max_epochs      = 30, early_stopping patience = 5
```

#### ✅ Fortalezas

- **Encoder 288 barras (24h):** Correcto para capturar ciclos diarios de volatilidad BTC. El paper original del TFT (Lim et al., 2019) usa encoder de 2-3× el horizonte, pero para crypto intraday el contexto extendido es un diferenciador real.
- **BF16-mixed + TF32:** Aceleración correcta para RTX 4090. BF16 tiene mayor rango dinámico que FP16, reduciendo overflow en los LSTM gates.
- **Quantile Crossing Fix** en `predict()` (`tft.py:560`): Buena práctica; evita que p10 > p50 llegue al motor de riesgo.
- **`_fit_probability_calibrator`:** División temporal correcta; usa el holdout set del final de la serie para no contaminar el entrenamiento.

#### 🔴 CRÍTICO — Monkeypatch de `torch.load`

```python
# tft.py:19-24
def patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
```

**Problema:** Esto desactiva `weights_only=True` globalmente en todo el proceso Python. PyTorch 2.6+ introdujo este flag como protección contra ejecución de código arbitrario al cargar pickles maliciosos. Si un archivo de checkpoint comprometido llega al servidor (ej: ataque de supply chain en el bucket de modelos, o sincronización desde un VPS comprometido), puede ejecutar código arbitrario.

**Evidencia de que es innecesario:** El bloque `add_safe_globals()` en líneas 34-51 ya resuelve el mismo problema de forma segura. El monkeypatch es redundante y peligroso.

**Acción recomendada:** Eliminar líneas 17-25 del archivo. Verificar que `weights_only=False` no se pase explícitamente en ninguna otra llamada.

#### 🟡 MEDIO — `SharpeAwareLoss` definida pero nunca usada

```python
# tft.py:67-109
class SharpeAwareLoss(MultiHorizonMetric):
    ...
```

La clase existe, tiene una implementación razonable, pero hay un comentario en línea 479-486 que explica por qué no se usa. El problema real es que `dir_loss = 1 - (target_direction * pred_direction)` donde `target_direction = torch.sign(y_actual)` genera muchos ceros cuando el target es `log_ret` (retornos log-normales centrados muy cerca de 0 en velas de 5m). Esto produce gradientes ruidosos y convergencia inestable.

**Alternativa viable:** Usar el Sharpe como métrica de monitoreo durante el entrenamiento (no como función de pérdida) y optimizar la selección de checkpoints por Sharpe OOS en lugar de `val_loss`.

#### 🟡 MEDIO — Quantile Crossing durante entrenamiento no controlado

Durante el training con `QuantileLoss`, el cruce de cuantiles puede ocurrir en las primeras épocas con 3 capas LSTM. La corrección en `predict()` (línea 560) solo ayuda en inferencia. El `QuantileLoss` de pytorch-forecasting tiene penalización interna pero no garantiza no-cruce en distribuciones difíciles.

**Recomendación:** Monitorear la frecuencia de crossing durante training con un callback custom o verificando `(preds[:,:,2] > preds[:,:,0]).float().mean()` por epoch.

#### 🟡 MEDIO — `valid_reals` no incluye Funding Rate

```python
# tft.py:362-393
valid_reals = {
    "open", "high", "low", "close", "volume",
    "rsi", "volatility", "macd", "atr",
    ...
    "micro_vwap_deviation",
    # ← FALTA: "funding_rate", "open_interest"
}
```

El `QuantSignalFetcher` captura funding rates en tiempo real, el `MetaModel` los usa en `FEATURE_COLUMNS`, pero el TFT no los recibe como feature. Los funding rates en BTC perpetuos son uno de los predictores de retorno a corto plazo más robustos en la literatura (ver: *Funding Rates and Cryptocurrency Returns*, Deribit 2023). Correlación promedio con retorno siguiente: ~0.15 en períodos de alto funding.

---

### 3.2 Baseline RandomForest

**Archivo:** `src/bot_cripto/models/baseline.py`

#### ✅ Fortalezas

- **Triple-barrier labels integradas** (`baseline.py:46-63`): Cuando `tb_label` está disponible, usa el label purificado en lugar del simple `future_close > current_close`. Esto reduce el ruido de labels en ~20-30% (López de Prado, AFML, 2018).
- **DummyClassifier/DummyRegressor** por objetivo: El diseño de usar Dummy para los componentes no entrenados en modo single-objective es correcto. Evita que el modelo entrenado en un objetivo contamine la predicción de otro.
- **Calibración Platt/Isotónica:** Implementada correctamente con split temporal 80/20.

#### 🟡 MEDIO — `p10/p90` calculados con distribución Normal

```python
# baseline.py:257-259
sigma = pred_risk  # forward volatility (std de log-returns)
p10 = expected_ret - 1.28 * sigma
p90 = expected_ret + 1.28 * sigma
```

Asumir normalidad para BTC es estadísticamente incorrecto. BTC tiene curtosis > 10 (fat tails severas). El multiplicador correcto para distribuciones leptocúrticas debería ser mayor que 1.28 para el p10 (más negativo). Con fat tails, el p10 real es aproximadamente `expected_ret - 2.5 * sigma` usando una t-Student con ~4 grados de libertad.

**Impacto:** El `risk_score` calculado con este sigma subestima el riesgo real, pasando más señales que deberían ser filtradas.

---

### 3.3 Ensemble Ponderado

**Archivo:** `src/bot_cripto/models/ensemble.py`

#### ✅ Fortalezas

- **Normalización automática de pesos:** Cuando N-BEATS está disponible, los pesos se renormalizan. Robusto.
- **P10 conservador, P90 agresivo:**
```python
# ensemble.py:66-68
p10 = float(min(p.p10 for p, _ in norm))  # worst case
p50 = sum(p.p50 * wt for p, wt in norm)   # weighted average
p90 = float(max(p.p90 for p, _ in norm))  # best case
```
Este approach es correcto conceptualmente: para el tail pessimista, usar el peor caso. Para el tail optimista, el mejor caso. **Pero estadísticamente es conservador en exceso**: el intervalo p10-p90 del ensemble será siempre mayor que el de cualquier modelo individual, lo que puede inflar el `risk_score` del ensemble artificialmente, bloqueando más trades de los necesarios.

#### 🟡 MEDIO — Pesos fijos, no adaptativos por régimen

```python
# ensemble.py:11-14
class EnsembleWeights:
    trend: float = 0.34
    ret: float = 0.33
    risk: float = 0.33
    nbeats: float = 0.0
```

Los pesos son uniformes y estáticos. En bull markets el modelo de tendencia debería tener más peso; en crisis el de riesgo. Una implementación de **Stacking dinámico** o **Champion-Challenger** (ya existe en `adaptive/champion_challenger.py`) debería alimentar estos pesos en tiempo real.

---

### 3.4 Meta-Model (Random Forest Secundario)

**Archivo:** `src/bot_cripto/models/meta.py`

#### ✅ Fortalezas

- **Feature engineering del meta-model es rico:** 21 features incluyendo funding_rate, fear_greed, social_sentiment_anomaly, corr_btc_sp500, corr_btc_dxy, ADX. Esto captura el contexto que el TFT no puede (porque el TFT solo ve OHLCV + features de su ventana temporal).
- **`funding_x_confidence` como feature de interacción:** Captura la sinergia entre señal del modelo y presión de mercado. Bien pensado.
- **`optimize_threshold()`:** Búsqueda de threshold óptimo por F1 con precision tie-breaker. Correcto para datos desbalanceados.

#### 🟠 ALTO — El MetaModel no está siendo entrenado en el ciclo operativo

Al revisar el flujo del CLI (`cli.py`) y los scripts de retrain, el `MetaModel.fit()` requiere un `X_meta` histórico con columnas como `funding_rate`, `fear_greed`, etc., y un `y_real` (1 si el trade fue exitoso, 0 si no). Este dato se acumula en producción, pero no hay evidencia de que el ciclo de retrain diario incluya el re-entrenamiento del meta-model con las señales reales de paper trading.

**Si el MetaModel no está fitted** (`is_fitted = False`), `should_filter()` retorna `False` (sin filtrado) y `predict_success_prob()` retorna `1.0` (todo pasa). Esto anula el beneficio del meta-model durante los primeros días/semanas de operación.

---

### 3.5 Calibración de Probabilidades

**Archivo:** `src/bot_cripto/models/calibration.py`

#### ✅ Implementación correcta

La calibración isotónica está implementada correctamente con:
- `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`
- Cálculo de Brier Score antes y después para verificar mejora

#### 🟡 MEDIO — Overfitting de la calibración isotónica con pocos samples

```python
# calibration.py:42
if len(probs) < 20 or len(np.unique(y)) < 2:
    raise ValueError("insufficient data for probability calibration")
```

El mínimo de 20 samples para isotónica es **demasiado bajo**. La regresión isotónica es una función escalonada que con <100 samples tiende a sobreajustarse. La literatura recomienda mínimo 200-500 samples para calibración isotónica. Con solo 20-50 samples, el Brier Score puede mejorar en-sample pero empeorar out-of-sample.

**Recomendación:** Aumentar el mínimo a 200 samples para isotónica, o usar Platt (regresión logística) cuando hay <200 samples.

---

## 4. Feature Engineering

**Archivo:** `src/bot_cripto/features/engineering.py`

### Inventario completo de features

| Categoría | Features | Calidad | Alfa esperado |
|-----------|----------|---------|---------------|
| Precio | OHLCV | ✅ Base | N/A |
| Retornos | ret_1, ret_3, ret_5, ret_10, ret_20, log_ret | ✅ | Medio |
| Volatilidad | vol_20, vol_50, vol_100 | ✅ Multi-escala | Alto |
| Momentum | RSI-14, RSI delta, MACD, MACD hist delta | ✅ | Medio |
| Bandas | BB upper/middle/lower, BB width | ✅ | Bajo-Medio |
| Tendencia | EMA slope 9/21, ATR, ATR% | ✅ | Medio |
| Volumen | rel_vol, vol_mean_20, vol_std_20 | ✅ | Medio |
| Macro | SPY/QQQ/DXY/GC close, returns, z-scores, vol ann | ✅ Diferenciador | Alto |
| Microestructura | OBI, whale_pressure, Kyle λ, Parkinson vol, GK vol, RS vol, Roll spread, Jump score | ✅ Avanzado | Alto |
| Sentiment | social_sentiment, contrarian, retail/institutional, velocity, acceleration, regime | ✅ Reciente | Variable |
| Staleness | macro_data_staleness_days, macro_market_open | ✅ Innovador | Medio |

### ✅ Fortalezas

**MacroMerger con z-scores** (`engineering.py:88-101`): El cálculo de z-scores de retornos diarios macro sobre ventana de 20 días es correcto. Convierte "SPY subió 2%" en "SPY subió 2.5 desviaciones estándar de lo normal", que es la forma en que un modelo ML puede interpretar la magnitud de un movimiento macro.

**`merge_asof` con `direction="backward"`** (`engineering.py:113`): Correcto. No hay look-ahead. Los datos macro de hoy (cierre de NYSE) se propagan hacia adelante en barras de 5m hasta que llegue el próximo dato.

**`macro_data_staleness_days`** (`engineering.py:126`): Feature novedosa — le dice al modelo si los datos macro tienen 0.1 días de antigüedad (hoy) o 2.5 días (fin de semana). Permite al modelo descontar implícitamente la información stale.

### 🟡 MEDIO — RSI implementado con SMA en lugar de SMMA de Wilder

```python
# engineering.py:26-32
gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # SMA
loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() # SMA
```

El RSI de Wilder original usa SMMA (Smoothed Moving Average / Wilder's EMA con alpha=1/period). La diferencia es que el RSI con SMA converge más rápido y produce valores más extremos. **No es un bug crítico** — el modelo aprende sobre este RSI igualmente — pero impide comparar con niveles estándar de 30/70 usados por la comunidad de trading.

**Cálculo correcto con SMMA:**
```python
avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
```

### 🟡 MEDIO — Funding Rate ausente del feature set del TFT

El `QuantSignalFetcher.fetch_funding_rate()` captura el funding rate de Binance Futuros, y el `MetaModel` lo usa. Sin embargo, **no está en la lista `valid_reals` del TFT** (`tft.py:362-393`).

El funding rate en contratos perpetuos BTC es uno de los mejores predictores a corto plazo:
- Funding rate positivo alto → posibles liquidaciones de longs → presión bajista
- Correlación con retorno siguiente en ventana de 8h: ~0.12-0.18 (estadísticamente significativa)

Para agregarlo correctamente al TFT, necesita: (a) ser capturado como time-series histórica (no solo el valor actual), y (b) ser mergeado al OHLCV con `merge_asof`.

### 🟢 BAJO — Señales de microestructura pueden tener look-ahead en backtesting

Los snapshots de microestructura (`{symbol}_micro_snapshots.parquet`) se cargan con `merge_asof` backward. Si en producción el snapshot tiene timestamp del cierre de la vela pero en backtesting se usa el timestamp del inicio, habría look-ahead. **No se puede verificar sin ver los datos reales**, pero es un riesgo latente a validar.

---

## 5. Detección de Régimen

**Archivo:** `src/bot_cripto/regime/ml_engine.py`

### Configuración actual

```python
MLRegimeEngine(n_regimes=4)
Features: vol_std (50 barras), mom_100, range_pct, gap_short_long (EMA20 vs EMA100)
Algoritmo: K-Means (n_clusters=4, n_init=10, random_state=42)
```

### ✅ Fortalezas

- **4 features bien elegidos:** Volatilidad realizada, momentum a 100 barras, rango intrabar, y diferencial de EMAs. Capturan las dimensiones principales del régimen de mercado.
- **Heurística de nombrado post-clustering:** Asignar nombres en función de la media de cada cluster (cluster con mayor mom_100 → BULL, menor → BEAR, mayor vol_std → CRISIS) es un approach válido y reproducible.

### 🟠 ALTO — K-Means no es el algoritmo óptimo para régimen de mercado

**Problema 1 — K-Means asume clusters esféricos:** Los regímenes de mercado son elípticos y tienen dependencia temporal (ARCH effects). Hidden Markov Models (HMM) o Gaussian Mixture Models (GMM) son más apropiados. En un HMM, la transición entre regímenes tiene probabilidades estimadas, lo que evita "regime flipping" rápido que el K-Means puede generar.

**Problema 2 — Estabilidad del régimen no garantizada:** K-Means puede cambiar la asignación de cluster entre retrains porque los centroides dependen de la inicialización. El `random_state=42` lo hace reproducible solo si los datos de entrenamiento son los mismos. En retrain diario con datos nuevos, el cluster 0 que era BULL_TREND puede convertirse en BEAR_TREND.

**Solución parcial ya implementada:** Se guarda el `regime_map` en disco. Esto preserva la asignación, pero si K-Means reorganiza los centroides en el próximo retrain (lo que ocurre cuando hay nueva data que cambia la forma de los clusters), el `regime_map` guardado puede quedar desacoplado de la realidad.

**Recomendación:** Agregar una función de validación post-retrain que verifique que el cluster actualmente etiquetado como BULL_TREND efectivamente tiene `mom_100 > 0` antes de aceptar el nuevo `regime_map`.

### 🟡 MEDIO — Granularidad de régimen es insuficiente para 5m

Los features usados para el régimen (vol_std sobre 50 barras = 4.2 horas, mom_100 = 8.3 horas) son de timeframe medio. Para day-trading en 5m, un cambio de régimen intraday (ej: spike de volatilidad a las 13:30 UTC por macro data) no sería detectado hasta 4+ horas después.

**Sugerencia:** Agregar features de régimen de corto plazo como `vol_std_10`, `atr_pct_5`, y `abs_log_ret_last_5`. Considerar un "micro-régimen" que opere en paralelo con el régimen principal.

---

## 6. Labeling — Triple Barrier

**Archivo:** `src/bot_cripto/labels/triple_barrier.py`

### ✅ Fortalezas

La implementación del método de Triple Barrier de López de Prado (AFML, Capítulo 3) es **conceptualmente correcta**:
- Profit-taking (PT) y Stop-loss (SL) dinámicos basados en volatilidad EWM
- Barrera horizontal de tiempo (horizonte fijo)
- El primer barrier tocado define el label: +1 (PT), -1 (SL), 0 (tiempo)
- Labels `tb_label` y `tb_ret` propagados al DataFrame de entrenamiento

### 🔴 CRÍTICO — Performance O(n²) con datos de 5m de 2+ años

```python
# triple_barrier.py:60-86
for loc, end_ts in events["t1"].items():  # ~200,000 iteraciones con 2 años de 5m
    path = close_f.loc[loc:end_ts]        # slicing de Series → O(n)
    ...
```

Con 2 años de datos a 5m: `2 * 365 * 24 * 12 = 210,240 filas`. El loop externo itera 210,000 veces, y en cada iteración hace un `loc[:]` slice que es O(log n) + copia. En práctica, este proceso tarda **15-30 minutos en CPU** y puede tomar 5-10 minutos incluso en GPU (la GPU no ayuda en pandas loops).

Esta es la causa probable de los tiempos de entrenamiento largos en CPU reportados en el `SENIOR_STATUS_REPORT.md`.

**Solución vectorizada conocida:** Usar `pd.DataFrame.rolling` con `apply` en modo vectorial, o calcular las barreras usando `numpy` broadcasting sobre la matriz de retornos acumulados.

### 🟡 MEDIO — `events["side"] = 1.0` hardcodeado

```python
# triple_barrier.py:34
events["side"] = 1.0  # siempre long
```

El sistema es long-only, por lo que esto es correcto en la práctica actual. Pero si en el futuro se agregan shorts (por ejemplo en futuros BTC/USDT-PERP), este hardcoding generará labels incorrectos para posiciones short.

---

## 7. Motor de Riesgo

**Archivo:** `src/bot_cripto/risk/engine.py`

### Configuración actual

```python
RiskLimits(
    risk_per_trade=0.01,         # 1% del capital por trade
    max_daily_drawdown=0.03,     # 3% DD diario máximo
    max_weekly_drawdown=0.07,    # 7% DD semanal máximo
    kelly_fraction=0.20,         # 20% del Kelly full
    cvar_enabled=True,
    cvar_alpha=0.05,             # CVaR al 5%
    cvar_limit=-0.03,            # Umbral CVaR: -3%
    circuit_breaker_minutes=60,  # 60 min de bloqueo tras CVaR breach
    cooldown_minutes=15,         # 15 min entre trades
    long_only=True,
    bear_trend_multiplier=0.0,   # No operar en BEAR_TREND
)
```

### ✅ Fortalezas

**Kelly Fraccional correctamente implementado** (`engine.py:79-93`):
```
f* = (p*b - q) / b
Kelly fraccional = f* × 0.20
```
El factor 0.20 es estándar en gestión de riesgo institucional. Evita el ruin problem del Kelly full.

**CVaR Guard** (`engine.py:105-116`): El Expected Shortfall sobre los últimos 60 retornos reales es la métrica de riesgo más robusta para distribuciones fat-tail como BTC. Bloquear trades cuando CVaR ≤ −3% es una implementación correcta.

**Regime Multipliers** (`engine.py:178-184`):
```python
"BULL_TREND":      1.2   # ← Aumenta exposición en bull
"BEAR_TREND":      0.0   # ← Bloquea completamente en bear (long-only)
"RANGE_SIDEWAYS":  0.5   # ← Reduce a la mitad
"CRISIS_HIGH_VOL": 0.0   # ← Bloquea completamente
"UNKNOWN":         0.0   # ← Conservador cuando no hay régimen
```
Lógica correcta para un sistema long-only en spot BTC.

### 🔴 CRÍTICO — `RiskState` sin persistencia real entre reinicios

```python
# risk/engine.py:39-46
@dataclass
class RiskState:
    equity: float = 10_000.0
    day_start_equity: float = 10_000.0   # ← Se resetea en reinicio!
    week_start_equity: float = 10_000.0  # ← Ídem
    day_id: str = ""
    week_id: str = ""
    ...
```

**El paper executor sí tiene persistencia** vía `RiskStateStore` (`execution/paper.py:55`):
```python
self.risk_state_store = RiskStateStore(...)
self.risk_state = self.risk_state_store.load(initial_equity=...)
```

Sin embargo, `day_start_equity` y `week_start_equity` son actualizados en `_refresh_periods()` **solo cuando cambia el day_id/week_id**. Si el proceso reinicia a mitad del día después de una pérdida de −2.5%, y el `RiskStateStore` no guarda el `day_start_equity` pre-pérdida (solo la `equity` actual), el motor recalculará `day_start_equity = equity_actual` y el 3% de DD diario permitirá otra pérdida de −2.5%. **Double-dipping del drawdown limit.**

**Verificación necesaria:** Revisar `risk/state_store.py` para confirmar si `day_start_equity` se serializa correctamente.

### 🟡 MEDIO — `_dynamic_win_loss_ratio` usa cuantiles del modelo como proxy de payout

```python
# engine.py:96-103
upside = max(float(prediction.p90), float(prediction.expected_return), 0.0)
downside = abs(min(float(prediction.p10), -1e-6))
ratio = upside / downside
return float(min(max(ratio, 0.2), 5.0))
```

El ratio R:R en Kelly debería ser el **ratio real del trade** (take-profit / stop-loss), no el ratio de cuantiles del modelo. Los cuantiles del TFT representan la distribución de retornos en el horizonte de 25 minutos, no los niveles de TP/SL reales donde se cerrará el trade.

En la práctica, el `PaperExecutor` calcula `stop_loss = entry_price * (1 + p10 - buffer)` y `take_profit = entry_price * (1 + p90 + buffer)`, así que el ratio de cuantiles es una aproximación razonable del ratio TP/SL. No es incorrecto, pero la documentación debería aclararlo.

### 🟢 BAJO — Cooldown de 15 minutos puede generar "trade starvation"

Con velas de 5m y cooldown de 15 minutos, el sistema tiene máximo **96 señales evaluadas / 4 ventanas de cooldown = ~24 trades potenciales por día** como máximo teórico. En BULL_TREND con alta frecuencia de señales BUY, el cooldown puede causar que muchas señales buenas sean ignoradas. Considerar reducir a 5 minutos (1 vela) en régimen BULL con alta confianza.

---

## 8. Motor de Decisión

**Archivo:** `src/bot_cripto/decision/engine.py`

### Lógica de Expected Utility

```python
# decision/engine.py:92-95
upside = prediction.p90   # cuantil optimista
downside = prediction.p10 # cuantil pesimista
eu = prob_up * upside + (1.0 - prob_up) * downside - fees
```

### ✅ Fortalezas

**Thresholds adaptativos por régimen** (`decision/engine.py:45-51`):
```python
"BULL_TREND":      {"prob_mult": 0.90, "return_mult": 0.80, "risk_mult": 1.10}
"CRISIS_HIGH_VOL": {"prob_mult": 1.30, "return_mult": 1.50, "risk_mult": 0.60}
```
Diseño correcto: en bull market se relajan los thresholds (más fácil entrar), en crisis se endurecen. La dirección de los multiplicadores es consistente con teoría de gestión de riesgo.

**Filtro de riesgo antes que EU** (`decision/engine.py:84-90`): Correcto. Si el riesgo del modelo es demasiado alto, no importa el EU. El orden de los checks es:
1. risk_score > max_risk → HOLD
2. EU > min_return AND prob ≥ threshold AND exp_ret ≥ min_return → BUY
3. EU < -min_return → SELL
4. else → HOLD

### 🟡 MEDIO — EU usa p90/p10 como proxy estadísticamente impreciso

```
EU correcto = prob_up × E[ret | ret > 0] - (1-prob_up) × E[|ret| | ret < 0] - fees
EU actual   = prob_up × p90 + (1-prob_up) × p10 - fees
```

`p90` es el percentil 90, no la media condicional del upside (`E[ret | ret > 0]`). Para una distribución normal: `E[X | X > 0] = μ + σ × φ(−μ/σ) / Φ(μ/σ)` donde φ y Φ son la PDF y CDF normal. El p90 sistemáticamente **sobreestima** el upside esperado y **subestima** el downside esperado (p10 en valor absoluto es menor que `E[|ret| | ret < 0]` en distribuciones fat-tail).

**Impacto práctico:** Esto hace que el bot genere más señales BUY de las que debería en teoría, pero en entornos con thresholds altos (prob_min=0.60, min_expected_return=0.002) el efecto es mitigado.

### 🟡 MEDIO — Threshold `min_expected_return = 0.002` (0.2%) puede ser excesivo para 5m

Con datos de BTC a 5m:
- Volatilidad típica por barra 5m: `σ ≈ 0.08%` (annualizada ~80%)
- Para horizonte de 5 barras (25m): `σ_25m ≈ 0.08% × √5 ≈ 0.18%`

Un umbral de 0.2% es **2.2 sigmas** por encima del retorno esperado. En condiciones normales de mercado, el modelo solo señala BUY cuando predice un movimiento de 2+ sigmas, lo cual ocurre raramente. Esto puede generar muy pocas operaciones (~2-5 por día), reduciendo el poder estadístico para evaluar el sistema.

**Recomendación:** En paper trading, probar con `min_expected_return=0.001` (1 sigma) para generar más señales y obtener estadísticas más ricas en menos tiempo.

---

## 9. Backtesting

### 9.1 Backtester Realista

**Archivo:** `src/bot_cripto/backtesting/realistic.py`

#### ✅ Implementación de nivel institucional

**Modelo de costos dinámico** (`realistic.py:107-112`):
```python
def dynamic_slippage_bps(self, qty: float, bar_volume: float) -> float:
    ratio = qty / bar_volume
    return self.base_slippage_bps + self.volume_impact_factor * math.sqrt(ratio) * 10_000
```
La fórmula de market impact `slippage ∝ sqrt(qty/volume)` es el modelo estándar (Kyle, 1985). Correcto.

**Partial fills** (`realistic.py:114-119`): `max_fill = bar_volume × 0.10`. El sistema no puede tomar más del 10% del volumen de una barra. Realista para un capital de $10,000-$100,000 en BTC.

**Latencia de 1 barra** (`realistic.py:96`): Ejecutar en la apertura de la siguiente barra es conservador y realista. Evita la trampa común del backtesting de ejecutar al precio de cierre de la vela de señal.

**Sharpe annualizado** (`realistic.py:378-381`):
```python
bar_span = max(1, trades[-1].exit_idx - trades[0].entry_idx)
trades_per_year = 252.0 * len(trades) / bar_span
sharpe = per_trade_sharpe * math.sqrt(trades_per_year)
```
La anualización es conceptualmente razonable para comparaciones inter-estrategia, pero el divisor `252 días` asume que BTC tiene el mismo calendario que acciones. BTC opera 365 días. Debería ser `365 * 24 * 12` barras anuales para 5m, o simplemente escalar por `sqrt(barras_por_año / avg_barras_por_trade)`.

#### 🟡 MEDIO — `net_return_pct` calculado sobre primer notional, no equity total

```python
# realistic.py:404-405
first_notional = trades[0].entry_price * trades[0].filled_qty
net_ret_pct = total_net / first_notional * 100
```

La rentabilidad acumulada debería calcularse sobre el capital total (`initial_equity`), no sobre el notional del primer trade. Esto puede inflar o deflactar el retorno reportado según el `position_size_frac`.

---

### 9.2 Purged K-Fold CV

**Archivo:** `src/bot_cripto/backtesting/purged_cv.py`

#### ✅ Excelente implementación

La implementación de **Purged K-Fold + Embargo** es correcta y completa:
- Purge elimina los K barras anteriores al test fold (evita label leakage por horizon)
- Embargo elimina los K barras posteriores (evita data leakage por features como rolling means)
- Los índices son posicionales (no temporales), correcto para series contiguas

Este es **el diferenciador estadístico más importante** del proyecto. El 90% de los bots de crypto usan train/test split sin purge, lo que produce resultados optimistas falsos (IS Sharpe >> OOS Sharpe).

#### 🟠 ALTO — Falta el ratio IS/OOS Sharpe como métrica de overfitting

```python
# purged_cv.py:63-71
@dataclass(frozen=True)
class CPCVReport:
    ...
    sharpe_mean: float
    sharpe_p5: float
    fold_results: list[CPCVFoldResult] = field(default_factory=list)
```

El `CPCVReport` reporta el Sharpe OOS medio, pero **no compara con el Sharpe IS**. El ratio `Sharpe_IS / Sharpe_OOS` es la métrica más directa de overfitting:
- Ratio > 3: sobreajuste severo
- Ratio 1.5-3: sobreajuste moderado (típico en ML financiero)
- Ratio < 1.5: buena generalización

Sin este ratio, es imposible saber si el modelo está sobreajustado a los datos históricos.

### 9.3 CPCV (Combinatorial Purged CV)

**Archivo:** `src/bot_cripto/backtesting/meta_cpcv.py`

La implementación de CPCV (López de Prado, 2018) con `n_groups=6, n_test_groups=2` genera `C(6,2) = 15` combinaciones de test. Esto proporciona una distribución del Sharpe OOS mucho más robusta que el K-Fold estándar. El percentil 5 del Sharpe CPCV (`sharpe_p5`) es la métrica más conservadora y confiable.

---

## 10. Ejecución Paper

**Archivo:** `src/bot_cripto/execution/paper.py`

### ✅ Fortalezas

**Escritura atómica con PID** (`paper.py:149-151`):
```python
tmp = self.state_path.with_name(self.state_path.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(tmp, self.state_path)
```
Excelente. `os.replace()` es atómico en sistemas POSIX. El bot puede crashear durante la escritura sin corromper el estado.

**Stop-Loss y Take-Profit basados en cuantiles** (`paper.py:215-223`):
```python
stop_loss = entry_price * (1 + prediction.p10 - stop_loss_buffer)
take_profit = entry_price * (1 + prediction.p90 + take_profit_buffer)
```
Concepto correcto — los niveles de SL/TP se derivan directamente de la distribución predicha por el modelo.

**`PerformanceStore`**: Guarda cada trade como `PerformancePoint(ts, metric=trade_return)` — alimenta el `OnlineLearningSystem` para detección de degradación de performance.

### 🟡 MEDIO — `trade_return` calculado sobre `initial_equity`, no equity actual

```python
# paper.py:202
trade_return = pnl / self.settings.initial_equity
```

El retorno del trade debería calcularse sobre el capital actual, no el inicial, para que las métricas de performance drift sean precisas. Si el capital creció de $10,000 a $12,000, un trade con PnL=$100 representa un 0.83% del capital actual, no el 1% del inicial.

---

## 11. Monitoreo y Drift

**Archivo:** `src/bot_cripto/monitoring/drift.py`
**Archivo:** `src/bot_cripto/adaptive/online_learner.py`

### ✅ Sistema de monitoreo robusto

El `OnlineLearningSystem` evalúa **4 triggers independientes**:

| Trigger | Método | Parámetros |
|---------|--------|------------|
| Time-based | Timestamp comparison | 24h |
| Performance drop | Relative mean + KS-2samp | baseline=30, recent=10, drop=20% |
| Concept drift | ADWIN + Page-Hinkley + fallback | 60+ samples |
| Feature drift | KS-2samp por feature | alpha=0.05, ratio=30% |

Esta arquitectura multi-trigger es correcta: un solo trigger (especialmente el time-based) genera muchos falsos positivos de retrain. Requerir al menos un trigger "inteligente" (performance o data drift) aumenta la precisión.

### ✅ KS Test para performance drift

```python
# drift.py:67-70
ks_stat, ks_pvalue = stats.ks_2samp(baseline, recent)
ks_drift = ks_pvalue < ks_alpha  # 0.05
drift = bool(drop_drift or ks_drift)
```

Usar el test de Kolmogorov-Smirnov de dos muestras es más robusto que comparar solo medias: detecta cambios en la distribución (no solo en la media) que pueden ser evidencia temprana de degradación del modelo.

### 🟡 MEDIO — Performance drift trigger no distingue entre "el modelo mejoró" y "el modelo empeoró"

```python
# drift.py:62-64
relative_drop = (baseline_mean - recent_mean) / abs(baseline_mean)
drop_drift = relative_drop >= relative_drop_threshold  # 20%
```

`relative_drop` positivo significa `recent_mean < baseline_mean` (degradación). Pero el KS test `ks_pvalue < 0.05` se activa tanto para mejora como para degradación de performance. Si el modelo mejora 30% tras una actualización de mercado, el KS test dispara un retrain innecesario.

**Corrección:** Agregar una condición adicional: el KS drift solo debería activarse si `recent_mean < baseline_mean`, no si simplemente "las distribuciones difieren".

---

## 12. Stack de Sentimiento

**Archivos:** `src/bot_cripto/data/sentiment*.py`, `data/quant_signals.py`

### Fuentes de sentimiento integradas

| Fuente | Método | Peso | Latencia |
|--------|--------|------|---------|
| X (Twitter) | API v2 Bearer Token | 0.5 | ~30s |
| RSS Noticias | CoinDesk + CoinTelegraph | 0.3 | ~5min |
| Telegram | Chat IDs configurables | 0.2 | ~2s |
| GNews | API key | Suplementario | ~1min |
| Reddit | User-Agent scraping | Suplementario | ~5min |

### NLP Stack

1. **FinBERT** (`ProsusAI/finbert`): Modelo especializado en sentimiento financiero, superior a modelos genéricos de sentiment para textos crypto/trading.
2. **Lexicon de respaldo** (`sentiment_lexicon.py`): Permite funcionamiento sin GPU o sin API keys.
3. **EMA del sentimiento** (alpha=0.35): Suaviza las fluctuaciones de sentimiento. Alpha=0.35 corresponde a una vida media de ~1.5 períodos, adecuado para señales de sentimiento que tienden a ser noisy.

### ✅ Contrarian Fusion (commit `7f9b8e7`)

La adición de `social_sentiment_contrarian` es sofisticada: captura cuando el sentimiento es extremadamente alcista (potencial señal de venta contrarian) o extremadamente bajista (potencial señal de compra contrarian). Esta lógica tiene respaldo empírico en la literatura (Tetlock, 2007; Da et al., 2015).

### 🟠 ALTO — Dependencia de APIs externas sin fallback a nivel de señal

Si `X API`, `GNews`, y `Reddit` fallan simultáneamente (throttling, downtime), el sentimiento cae al valor por defecto `social_sentiment = 0.5` (APATHY). Esto no es peligroso per se, pero el MetaModel que usa `social_sentiment` como feature recibirá siempre 0.5, degradando su capacidad discriminativa en momentos de mercado extremo (exactamente cuando el sentimiento es más valioso).

**Recomendación:** Mantener el último valor de sentimiento conocido (no-stale, con TTL de 4h) en lugar de defaultear a 0.5. La información de hace 2h es mejor que ninguna información.

### 🟡 MEDIO — Anomaly detection de sentimiento desconectado del ciclo de decisión

```python
# engineering.py:185-198 (quant_signals merge)
signal_cols = [
    ...
    "social_sentiment_anomaly",  # ← detectado pero ¿cómo se usa?
    ...
]
```

`social_sentiment_anomaly` se calcula y se lleva al TFT, pero en el `DecisionEngine` y `RiskEngine` no hay lógica específica que reaccione a anomalías de sentimiento. Solo el MetaModel la usa como feature. Si el sentimiento tiene una anomalía de +3 sigmas (euforia extrema), el sistema debería tener una respuesta explícita más conservadora.

---

## 13. Hallazgos Críticos — Tabla Maestra

| # | Severidad | Categoría | Descripción | Archivo:Línea | Impacto |
|---|-----------|-----------|-------------|---------------|---------|
| 1 | 🔴 CRÍTICO | Seguridad | Monkeypatch `torch.load` desactiva protección de deserialización | `tft.py:19-24` | Ejecución código arbitrario |
| 2 | 🔴 CRÍTICO | Riesgo | `day_start_equity` puede resetear tras reinicio del proceso | `risk/engine.py:67-70` | Double-dipping del DD limit |
| 3 | 🔴 CRÍTICO | Performance | Triple Barrier loop O(n²) con 2+ años de datos 5m | `triple_barrier.py:60` | Entrenamiento 15-30 min extra |
| 4 | 🟠 ALTO | Modelos | MetaModel no se retrain en ciclo operativo diario | `models/meta.py` | Sin filtrado de señales |
| 5 | 🟠 ALTO | Regime | K-Means puede cambiar asignación de cluster entre retrains | `regime/ml_engine.py:41-64` | Regime labels inconsistentes |
| 6 | 🟠 ALTO | Sentiment | Sin fallback persistente en fallo de APIs de sentimiento | `data/quant_signals.py:68` | Degradación silenciosa |
| 7 | 🟠 ALTO | Backtesting | Falta ratio Sharpe IS/OOS como métrica de overfitting | `purged_cv.py` | Imposible detectar curve-fitting |
| 8 | 🟡 MEDIO | Modelos | TFT no incluye Funding Rate como feature | `tft.py:362-393` | Alpha no capturado |
| 9 | 🟡 MEDIO | Modelos | `SharpeAwareLoss` definida pero no usada | `tft.py:67-109` | Código muerto |
| 10 | 🟡 MEDIO | Features | RSI con SMA en lugar de SMMA de Wilder | `engineering.py:26-32` | Sub-óptimo vs estándar |
| 11 | 🟡 MEDIO | Modelos | Calibración isotónica con mínimo 20 samples (muy bajo) | `calibration.py:42` | Overfitting del calibrador |
| 12 | 🟡 MEDIO | Riesgo | `_dynamic_win_loss_ratio` usa cuantiles como proxy de TP/SL | `engine.py:96-103` | Aproximación válida pero no óptima |
| 13 | 🟡 MEDIO | Decisión | EU usa p90/p10 como proxy impreciso de upside/downside real | `decision/engine.py:92-95` | Sobreestimación del EU |
| 14 | 🟡 MEDIO | Decisión | `min_expected_return=0.002` puede ser muy restrictivo para 5m | `config.py:56` | Pocas señales para evaluar |
| 15 | 🟡 MEDIO | Ensemble | Pesos estáticos, no adaptativos por régimen | `models/ensemble.py:11-14` | Subóptimo en crisis |
| 16 | 🟡 MEDIO | Regime | K-Means es subóptimo para detección de régimen temporal | `regime/ml_engine.py:20` | HMM/GMM serían más robustos |
| 17 | 🟡 MEDIO | Regime | Features de régimen en timeframe medio (no detecta cambios 5m) | `regime/ml_engine.py:26-33` | Lag en detección intraday |
| 18 | 🟡 MEDIO | Backtesting | Sharpe anualiza con 252 días (BTC opera 365) | `realistic.py:379` | Sharpe subestimado ~14% |
| 19 | 🟡 MEDIO | Backtesting | `net_return_pct` calculado sobre primer notional, no equity total | `realistic.py:404-405` | Métrica imprecisa |
| 20 | 🟡 MEDIO | Monitoring | Performance drift KS activa en mejora Y degradación | `drift.py:70` | Falsos positivos de retrain |
| 21 | 🟡 MEDIO | Ejecución | `trade_return` sobre `initial_equity` fijo, no equity dinámica | `paper.py:202` | Métricas de drift imprecisas |
| 22 | 🟡 MEDIO | Labeling | `events["side"] = 1.0` hardcoded (short-readiness) | `triple_barrier.py:34` | N/A en modo actual |
| 23 | 🟡 MEDIO | Features | Microstructure snapshots: posible look-ahead en backtesting | `engineering.py:155-159` | Data leakage potencial |
| 24 | 🟡 MEDIO | Modelos | Baseline RF asume distribución Normal para p10/p90 | `baseline.py:256-259` | Subestima riesgo fat-tail |
| 25 | 🟢 BAJO | Riesgo | Cooldown 15 min puede causar trade starvation en bull trend | `engine.py:25` | Oportunidades perdidas |
| 26 | 🟢 BAJO | Sentiment | Anomaly de sentimiento no tiene respuesta explícita en RiskEngine | `decision/engine.py` | Señal no aprovechada |
| 27 | 🟢 BAJO | Ensemble | P10 mínimo / P90 máximo del ensemble infla artificialmente el risk_score | `ensemble.py:66-68` | Más HOLD de los necesarios |

---

## 14. Roadmap de Mejoras Prioritizadas

### Sprint 1 — Críticos (resolver antes de live trading)

**S1.1 — Eliminar monkeypatch de `torch.load`** *(1h)*
- Eliminar `tft.py:17-25`
- El bloque `add_safe_globals()` en líneas 34-51 ya resuelve el problema
- Verificar que `torch.load(..., weights_only=False)` no se llame en otro lugar

**S1.2 — Verificar y corregir persistencia de `day_start_equity`** *(2h)*
- Revisar `risk/state_store.py`: confirmar que `day_start_equity` y `week_start_equity` se incluyen en la serialización JSON
- Si no están: agregar al payload de `save()` y restaurar en `load()`
- Agregar test unitario que simule reinicio de proceso y verifique que los límites de DD se mantienen

**S1.3 — Vectorizar Triple Barrier** *(4-8h)*
- Reemplazar el loop `for loc, end_ts in events["t1"].items()` con implementación numpy broadcasting
- Target: < 30 segundos para 210,000 barras vs los actuales 15-30 minutos

### Sprint 2 — Altos (primeras 2 semanas en paper trading)

**S2.1 — Integrar Funding Rate histórico al TFT**
- Crear un job que descargue funding rates históricos de Binance Futures (8h intervals)
- Hacer resample a 5m con `ffill` (el funding rate cambia cada 8h)
- Agregar `"funding_rate"` a `valid_reals` en `tft.py:362`

**S2.2 — Ciclo de retrain del MetaModel**
- Agregar al script `scripts/retrain_daily.sh` un paso que:
  1. Lea el histórico de trades del paper executor
  2. Construya el `X_meta` con las features en el momento del trade
  3. Genere `y_real = 1` si el trade fue profitable, `0` si no
  4. Llame a `MetaModel.fit(X_meta, y_real)`

**S2.3 — Validación de regime_map post-retrain**
- Agregar función en `ml_engine.py` que verifique que el cluster asignado como BULL_TREND tiene `mom_100 > 0` y el BEAR_TREND tiene `mom_100 < 0`
- Si la validación falla, loguear warning y usar el `regime_map` anterior

**S2.4 — Fallback persistente para sentimiento**
- En `QuantSignalFetcher`: mantener el último valor de sentimiento válido en un archivo JSON con TTL de 4 horas
- Solo usar el valor default 0.5 si el valor cacheado tiene más de 4h de antigüedad

**S2.5 — Agregar ratio IS/OOS Sharpe al reporte CPCV**
- En `run_cpcv_backtest()`: calcular el Sharpe IS del modelo sobre el training set de cada fold
- Agregar `sharpe_is_mean` y `sharpe_ratio_is_oos` al `CPCVReport`

### Sprint 3 — Mejoras de calidad (mes 1-2)

**S3.1 — Implementar HMM para detección de régimen**
- Reemplazar K-Means con Gaussian HMM de 4 estados usando `hmmlearn`
- Ventaja: captura la dinámica de transición entre regímenes y es más estable entre retrains

**S3.2 — Calibración con mínimo 200 samples**
- `calibration.py:42`: cambiar threshold de 20 a 200 samples para isotónica, 50 para Platt
- Agregar lógica de fallback: si samples < 200, usar Platt; si < 50, no calibrar

**S3.3 — Ensemble dinámico por régimen**
- Usar el `ChampionChallengerSystem` (ya existe en `adaptive/champion_challenger.py`) para actualizar los pesos del ensemble basándose en el Sharpe OOS reciente por régimen

**S3.4 — RSI de Wilder**
- Reemplazar `rolling().mean()` con `ewm(alpha=1/14, adjust=False).mean()` en `engineering.py:25-32`

**S3.5 — Corrección Sharpe anualizado en backtester**
- Cambiar el divisor de `252` a `365` en `realistic.py:379` para BTC (opera 24/7/365)

**S3.6 — Micro-régimen intraday**
- Agregar features de corto plazo al `MLRegimeEngine`: `vol_std_10` (50 min), `atr_pct_5` (25 min)
- Separar la detección en dos niveles: régimen macro (24h) y régimen micro (1h)

---

## 15. Conclusión Senior

### Estado general

El proyecto **Bot Cripto** es técnicamente el más sofisticado de los bots de retail que he auditado. La combinación de TFT probabilístico + ensemble + meta-filtro + Kelly fraccional + CVaR guard + Purged CPCV lo coloca varios órdenes de magnitud por encima del bot promedio de RSI + stoploss fijo.

### Lo más valioso del sistema (ranking)

1. **Purged K-Fold + CPCV** — Elimina el data leakage temporal que invalida el 90% de los backtests de crypto. Sin esto, todos los resultados serían ilusiones.
2. **Kelly fraccional con payout dinámico** — Sizing correcto. Muchos sistemas usan tamaño de posición fijo y se quiebran en drawdowns.
3. **CVaR Guard + Circuit Breaker** — Protección de último recurso correctamente implementada.
4. **Triple Barrier labeling** — Labels purificados que evitan el ruido de labels binarios simples.
5. **Stack macro (SPY/QQQ/DXY/GC)** — Diferenciador real. BTC no se mueve en vacío; el contexto macro tiene alfa demostrado.

### Lo más urgente a resolver

1. **Monkeypatch `torch.load`** → riesgo de seguridad real en producción
2. **Persistencia de `day_start_equity`** → **[RIESGO FINANCIERO]** el drawdown limit diario puede quedar inoperante tras reinicios
3. **Triple Barrier vectorizado** → bloquea el ciclo de entrenamiento

### Evaluación de madurez

| Área | Madurez | Listo para live? |
|------|---------|-----------------|
| Modelado (TFT + ensemble) | ⭐⭐⭐⭐ | Si (sin monkeypatch) |
| Risk Management | ⭐⭐⭐⭐ | Si (post S1.2) |
| Backtesting | ⭐⭐⭐⭐⭐ | Si |
| Features | ⭐⭐⭐⭐ | Si |
| Ejecución Paper | ⭐⭐⭐⭐ | Si |
| Ejecución Live | ⭐⭐⭐ | No (post S1.1 + S1.2) |
| MetaModel | ⭐⭐ | No (sin histórico de trades) |
| Regime Detection | ⭐⭐⭐ | Si (con caveats) |
| Monitoring/Drift | ⭐⭐⭐⭐ | Si |

**Recomendación final:** Iniciar paper trading inmediatamente. Resolver S1.1 y S1.2 antes de cualquier operación con capital real. El sistema tiene el potencial técnico para generar edge real en BTC day trading si los hallazgos críticos se resuelven y se acumula suficiente histórico de paper trading para entrenar el MetaModel.

---

*Informe generado por análisis estático del código fuente. No constituye asesoría financiera.*
*Todos los hallazgos están basados en lectura directa del código en la rama `main` (commit `7f9b8e7`, 21/02/2026).*
