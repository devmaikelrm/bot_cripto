# Plan maestro para llevar un sistema de trading cuantitativo al nivel profesional

**Un sistema de trading ML que ya corre en paper trading con XGBoost/TFT puede alcanzar rendimiento profesional mediante cinco fases secuenciales: primero corrigir los fundamentos de etiquetado y validación (donde la mayoría de los sistemas fallan), luego incorporar modelos SOTA y features avanzados, después construir un ensemble inteligente con meta-labeling, integrar gestión de riesgo cuantitativa de nivel hedge fund, y finalmente automatizar la adaptación online.** La investigación muestra que la arquitectura del modelo por sí sola rara vez supera al mercado — tests de Diebold-Mariano confirman que modelos DL sofisticados frecuentemente no superan persistencia naive sin features y etiquetado correctos. Este plan prioriza las mejoras con mayor impacto comprobado primero.

---

## Fase 1: Fundamentos críticos que el 90% de los sistemas ignora (Semanas 1-5)

Esta fase corrige los errores más costosos en trading ML: etiquetado naive, validación con data leakage, y ausencia de gestión de riesgo base. **Cada mejora aquí tiene impacto multiplicativo sobre todo lo que viene después.**

### Triple-Barrier Method para generación de etiquetas

El etiquetado de retorno fijo a horizonte temporal constante introduce ruido masivo. La triple barrera de López de Prado establece tres condiciones dinámicas por trade: take-profit (barrera superior), stop-loss (barrera inferior), y expiración temporal (barrera vertical), **escaladas por volatilidad realizada**. Esto genera etiquetas que reflejan cómo opera un trader real.

```python
# Implementación core del Triple-Barrier Method
import numpy as np
import pandas as pd

def get_daily_vol(close, span=100):
    """Volatilidad exponencial para escalar barreras"""
    return close.pct_change().ewm(span=span).std()

def apply_triple_barrier(close, events, pt_sl=(2.0, 2.0), molecule=None):
    """
    pt_sl: multiplicadores de volatilidad para profit-take y stop-loss
    events: DataFrame con 't1' (expiración), 'trgt' (volatilidad target), 'side' (dirección)
    """
    out = events[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index, dtype=float)
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index, dtype=float)
    
    for loc, t1 in events['t1'].items():
        path = close[loc:t1]
        path = (path / close[loc] - 1) * events.at[loc, 'side']
        # Primera barrera tocada
        earliest_pt = path[path > pt[loc]].index.min() if pt_sl[0] > 0 else pd.NaT
        earliest_sl = path[path < sl[loc]].index.min() if pt_sl[1] > 0 else pd.NaT
        out.loc[loc, 'pt'] = earliest_pt
        out.loc[loc, 'sl'] = earliest_sl
    
    out['label'] = out[['pt', 'sl', 't1']].idxmin(axis=1)
    # +1 si PT primero, -1 si SL primero, 0 si expiró
    return out
```

**Librerías**: `mlfinlab` (Hudson & Thames, comercial) o `quantreo` (open-source). **Configuración recomendada**: pt_sl=(2.0, 2.0) para mercados trending, (1.5, 1.5) para ranging. Barrera temporal: **20 bars para 5m** (~1.5h), **48 bars para 1h** (~2 días).

### Purged K-Fold y Combinatorial Purged CV (CPCV)

La validación cruzada estándar en series financieras produce estimaciones **infladas en 30-50%** por data leakage temporal. La purga elimina observaciones de entrenamiento cuyo span de etiqueta solapa con el test set. El embargo añade un gap adicional por autocorrelación serial.

```python
# Opción 1: skfolio (moderno, mantenido, open-source)
from skfolio.model_selection import CombinatorialPurgedCV
cpcv = CombinatorialPurgedCV(
    n_folds=6,           # 6 bloques secuenciales
    n_test_folds=2,      # 2 bloques como test → C(6,2)=15 splits
    purged_size=10,      # bars purgados = horizonte máximo de etiqueta
    embargo_size=5       # bars de embargo post-test
)

# Opción 2: mlfinlab 
from mlfinlab.cross_validation import PurgedKFold
cv = PurgedKFold(n_splits=5, times=label_end_times, embargo=0.02)

# Uso con GridSearch temporal
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=xgb_model, param_grid=params, cv=cpcv, 
                  scoring='f1', n_jobs=-1)
```

**CPCV(6,2)** genera **15 combinaciones** y **5 backtest paths únicos**, permitiendo inferencia estadística sobre la robustez del modelo. **Tamaño de purga**: debe ser ≥ horizonte máximo de la triple barrera.

### Walk-Forward Optimization Framework

```python
def walk_forward_optimization(data, train_months, test_months, anchored=False):
    """
    5m data: train=2-4 semanas, test=1 semana, step=1 semana
    1h data: train=6-12 meses, test=1-3 meses, step=1 mes
    """
    results, models = [], []
    dates = data.index.unique()
    
    for test_start in pd.date_range(dates[0] + pd.DateOffset(months=train_months),
                                     dates[-1], freq=f'{test_months}M'):
        train_start = dates[0] if anchored else test_start - pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        
        train = data[train_start:test_start]
        test = data[test_start:test_end]
        
        model = train_and_optimize(train)  # Incluye CPCV interno
        preds = model.predict(test)
        
        metrics = evaluate_trading(preds, test)
        metrics['wf_efficiency'] = metrics['oos_sharpe'] / metrics['is_sharpe']
        results.append(metrics)
        models.append(model)
    
    return pd.DataFrame(results), models
```

**Walk-Forward Efficiency** (OOS Sharpe / IS Sharpe) debe estar entre **0.5 y 0.85**. Menor a 0.4 indica overfitting severo; mayor a 0.9 sugiere underfitting.

### Features de microestructura desde OHLCV

Sin order book completo, estas aproximaciones capturan señales de microestructura:

```python
def microstructure_features(df):
    """Features de microestructura aproximados desde OHLCV"""
    features = pd.DataFrame(index=df.index)
    
    # 1. Corwin-Schultz Spread Estimator (2012)
    beta = np.log(df['high'] / df['low']) ** 2
    gamma = np.log(df['high'].rolling(2).max() / df['low'].rolling(2).min()) ** 2
    alpha = (np.sqrt(2*beta.rolling(2).mean()) - np.sqrt(beta.rolling(2).mean())) / \
            (3 - 2*np.sqrt(2)) - np.sqrt(gamma / (3 - 2*np.sqrt(2)))
    features['spread_cs'] = (2*(np.exp(alpha)-1) / (1+np.exp(alpha))).clip(lower=0)
    
    # 2. Trade Imbalance Proxy (buy/sell volume estimation)
    buy_ratio = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    features['trade_imbalance'] = (2 * buy_ratio - 1)  # [-1, 1]
    
    # 3. VWAP Deviation
    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical * df['volume']).rolling(48).sum() / df['volume'].rolling(48).sum()
    features['vwap_dev'] = (df['close'] - vwap) / vwap
    
    # 4. Kyle's Lambda (price impact proxy)
    signed_vol = np.sign(df['close'].pct_change()) * df['volume']
    features['kyle_lambda'] = df['close'].pct_change().rolling(20).apply(
        lambda y: np.polyfit(signed_vol.loc[y.index].values, y.values, 1)[0]
    )
    
    # 5. Volume clock features
    features['volume_zscore'] = (df['volume'] - df['volume'].rolling(288).mean()) / \
                                 df['volume'].rolling(288).std()
    features['dollar_volume'] = df['close'] * df['volume']
    
    return features
```

### CVaR y Circuit Breakers de riesgo base

```python
from scipy.stats import norm, skew, kurtosis

class BaseRiskEngine:
    def __init__(self, max_cvar_95=-0.03, max_drawdown=-0.15, max_daily_loss=-0.03):
        self.max_cvar = max_cvar_95
        self.max_dd = max_drawdown
        self.max_daily = max_daily_loss
        self.peak_equity = 0
    
    def cornish_fisher_cvar(self, returns, alpha=0.95):
        """CVaR con corrección para colas pesadas (esencial crypto)"""
        z = norm.ppf(1 - alpha)
        s, k = skew(returns), kurtosis(returns, excess=True)
        z_cf = z + (z**2-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*(s**2)/36
        return returns.mean() + returns.std() * z_cf
    
    def position_scale(self, equity):
        self.peak_equity = max(self.peak_equity, equity)
        dd = (equity - self.peak_equity) / self.peak_equity
        if dd <= self.max_dd: return 0.0      # CIRCUIT BREAKER TOTAL
        if dd <= self.max_dd * 0.5:           # Zona de advertencia
            return max(0.1, 1.0 + dd/abs(self.max_dd))
        return 1.0
```

### Métricas de evaluación Fase 1

- **Walk-Forward Efficiency**: 0.5–0.85 (target)
- **CPCV Sharpe distribution**: media > 0.5, p5 > 0 (5º percentil positivo)
- **Improvement en F1** vs etiquetado fijo: esperado **+15-25%**
- **Reducción de overfitting**: IS/OOS gap < 30%
- **CVaR 95%** diario: monitoreo continuo, target < -3%

**Librerías Fase 1**: `mlfinlab`, `skfolio`, `quantreo`, `scikit-learn`, `scipy`

**Tiempo estimado**: 4-5 semanas

---

## Fase 2: Modelos SOTA y features cross-market que capturan alfa real (Semanas 6-12)

Con la base de etiquetado y validación correcta, ahora los modelos avanzados pueden demostrar su verdadero valor. **iTransformer es la incorporación más impactante** para un sistema multi-mercado porque trata cada activo como un token, modelando naturalmente las correlaciones entre crypto y forex.

### iTransformer para forecasting multivariate cross-asset

La innovación del iTransformer (ICLR 2024) es elegante: **invierte la arquitectura Transformer** para que cada variable sea un token (no cada timestamp). Self-attention modela correlaciones entre activos; FFN aprende representaciones temporales. Para 7 instrumentos (BTC, SOL, ETH, EUR/USD, GBP/USD, USD/JPY, XAU/USD), esto es ideal.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer, PatchTST, NHITS

# iTransformer: cada activo = token, cross-attention captura correlaciones
itransformer = iTransformer(
    h=12,                    # horizonte: 12 bars (1h para 5m data)
    input_size=96,           # lookback: 96 bars (8h para 5m)
    n_series=7,              # 7 instrumentos
    hidden_size=256,
    n_heads=8,
    e_layers=3,
    d_model=256,
    learning_rate=1e-4,
    max_steps=1000,
    batch_size=32,
    loss=DistributionLoss(distribution='StudentT', level=[80, 95]),
    scaler_type='robust',
    early_stop_patience_steps=10,
)

# PatchTST: parches de 16 timesteps, channel-independence
# 21% reducción MSE vs TFT, self-supervised pretraining disponible
patchtst = PatchTST(
    h=12,
    input_size=512,          # lookback largo (patching reduce a 32 tokens)
    patch_len=16,
    stride=8,
    hidden_size=128,
    n_heads=4,
    learning_rate=1e-4,
    max_steps=1000,
)

# N-HiTS: 50x más rápido que Transformers, multi-horizon jerárquico
nhits = NHITS(
    h=12,
    input_size=96,
    n_pool_kernel_size=[4, 4, 4],  # Multi-rate downsampling
    n_freq_downsample=[12, 4, 1],  # Hierarchical interpolation
    learning_rate=1e-3,
    max_steps=500,
)

# Entrenar todos con NeuralForecast
nf = NeuralForecast(models=[itransformer, patchtst, nhits], freq='5T')
nf.fit(df=train_data)       # df formato long: unique_id, ds, y, + exógenas
forecasts = nf.predict()
```

### Features on-chain y derivados crypto

Los **funding rates** son la señal de sentimiento más honesta porque están respaldados por dinero real. Tasas extremas positivas (>0.05%/8h) preceden correcciones el **70%+ de las veces**.

```python
import ccxt

class CryptoAlternativeFeatures:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
    
    def funding_rate_features(self, symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']):
        features = {}
        for symbol in symbols:
            rates = pd.DataFrame(self.exchange.fetch_funding_rate_history(symbol, limit=500))
            name = symbol.split('/')[0]
            features[f'{name}_funding'] = rates['fundingRate']
            features[f'{name}_funding_zscore'] = (
                rates['fundingRate'] - rates['fundingRate'].rolling(90).mean()
            ) / rates['fundingRate'].rolling(90).std()
            features[f'{name}_funding_extreme'] = (
                rates['fundingRate'].abs() > rates['fundingRate'].rolling(90).quantile(0.95)
            ).astype(int)
        return pd.DataFrame(features)
    
    def open_interest_features(self, symbol='BTC/USDT:USDT'):
        """OI + price divergencia = señal potente"""
        # Vía ccxt o CoinGlass API
        # Rising OI + Rising Price = trend fuerte
        # Rising OI + Falling Price = distribución (bearish)
        # Falling OI + Rising Price = short squeeze
        pass
    
    def onchain_features_glassnode(self, api_key, asset='BTC'):
        """MVRV, Exchange Flows, NUPL desde Glassnode API"""
        import requests
        base = "https://api.glassnode.com/v1/metrics"
        
        mvrv = requests.get(f"{base}/market/mvrv", 
                           params={'a': asset, 'i': '1h', 'api_key': api_key}).json()
        exchange_flow = requests.get(f"{base}/transactions/transfers_volume_exchanges_net",
                                    params={'a': asset, 'i': '1h', 'api_key': api_key}).json()
        
        return {
            'mvrv': pd.DataFrame(mvrv),           # >3.7 overvalued, <1.0 undervalued
            'exchange_netflow': pd.DataFrame(exchange_flow),  # positivo = presión venta
        }
```

### Features cross-asset y correlaciones dinámicas

```python
import itertools
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA

class CrossAssetFeatures:
    def __init__(self, returns_df):
        self.returns = returns_df  # Columnas: BTC, ETH, SOL, EURUSD, GBPUSD, USDJPY, XAUUSD
    
    def rolling_correlations(self, windows=[24, 72, 288]):
        """Correlaciones multi-escala entre todos los pares"""
        features = {}
        pairs = list(itertools.combinations(self.returns.columns, 2))
        for w in windows:
            for a1, a2 in pairs:
                corr = self.returns[a1].rolling(w).corr(self.returns[a2])
                features[f'corr_{a1}_{a2}_{w}'] = corr
                features[f'corr_chg_{a1}_{a2}_{w}'] = corr.diff(w//4)
        return pd.DataFrame(features)
    
    def pca_risk_factors(self, window=252, n_components=3):
        """PC1 = risk-on/off; explained_var alta = mercado correlacionado (stress)"""
        results = []
        for i in range(window, len(self.returns)):
            pca = PCA(n_components=n_components)
            pca.fit(self.returns.iloc[i-window:i])
            proj = pca.transform(self.returns.iloc[i:i+1])
            row = {f'PC{j+1}': proj[0,j] for j in range(n_components)}
            row['PC1_var_explained'] = pca.explained_variance_ratio_[0]
            results.append(row)
        return pd.DataFrame(results, index=self.returns.index[window:])
    
    def lead_lag_features(self, max_lag=12):
        """BTC lidera ETH/SOL por 5-30 min; DXY lidera forex inversamente"""
        features = {}
        for a1, a2 in [('BTC','ETH'), ('BTC','SOL'), ('XAUUSD','USDJPY')]:
            if a1 in self.returns.columns and a2 in self.returns.columns:
                for lag in range(1, max_lag+1):
                    features[f'{a1}_lead_{a2}_lag{lag}'] = self.returns[a1].shift(lag)
        return pd.DataFrame(features, index=self.returns.index)
```

### GARCH como generador de features (no como modelo standalone)

```python
from arch import arch_model

def garch_feature_generator(returns_series, name=''):
    """GJR-GARCH captura asimetría en volatilidad (bad news = más vol)"""
    scaled = returns_series * 100
    am = arch_model(scaled, vol='GARCH', p=1, o=1, q=1, power=2.0, dist='t')
    result = am.fit(disp='off', last_obs=len(scaled)-1)
    
    cond_vol = result.conditional_volatility / 100
    forecast = result.forecast(horizon=1)
    
    return pd.DataFrame({
        f'{name}_garch_vol': cond_vol,
        f'{name}_garch_zscore': (returns_series.abs() - cond_vol) / cond_vol,
        f'{name}_vol_persistence': result.params.get('beta[1]',0) + result.params.get('alpha[1]',0),
        f'{name}_vol_forecast': np.sqrt(forecast.variance.iloc[-1].values[0]) / 100,
    })
```

### Métricas de evaluación Fase 2

- **MSE/MAE** en walk-forward: iTransformer vs TFT baseline (target: **-15 a -25% MSE**)
- **Directional Accuracy**: >55% en 1h, >52% en 5m
- **Information Coefficient** (IC) de features nuevos: target >0.03 por feature
- **Feature Importance** (SHAP): verificar que features on-chain/cross-asset aportan
- **Latencia de inferencia**: <100ms para N-HiTS, <500ms para iTransformer

**Librerías Fase 2**: `neuralforecast`, `ccxt`, `arch`, `scikit-learn`, `glassnode API`, `transformers` (FinBERT)

**Tiempo estimado**: 6-7 semanas

---

## Fase 3: Ensemble inteligente con meta-labeling que filtra el ruido (Semanas 13-19)

Esta fase transforma múltiples modelos imperfectos en un sistema de precisión superior. **Meta-labeling es posiblemente la técnica con mayor impacto comprobado** en trading ML: Hudson & Thames confirmó mejoras en F1, precisión y rendimiento real en S&P500 E-mini Futures.

### Meta-labeling: el multiplicador de precisión

El concepto es elegante: un modelo primario decide la **dirección** (optimizado para recall alto), y un modelo secundario decide **si apostar** (optimizado para precisión). La probabilidad del meta-modelo dimensiona la posición.

```python
class MetaLabelingPipeline:
    def __init__(self, primary_model, meta_model=None):
        self.primary = primary_model      # TFT o iTransformer → dirección
        self.meta = meta_model or XGBClassifier(
            n_estimators=300, max_depth=4, 
            scale_pos_weight=1.5,    # Balancear clases
            learning_rate=0.05
        )
    
    def generate_meta_labels(self, close, primary_preds, vol_target, pt_sl=(2,2)):
        """
        1. Primary predice dirección (side)
        2. Triple-barrier genera labels CONDICIONADOS al side
        3. Meta-label = 1 si primary acertó, 0 si no
        """
        events = get_events(close, t_events, pt_sl=pt_sl, 
                           target=vol_target, side=primary_preds)
        meta_labels = (events['ret'] * events['side'] > 0).astype(int)
        return meta_labels
    
    def train_meta_model(self, X, primary_preds, meta_labels, sample_weights=None):
        """Features del meta-modelo = features originales + predicción primary"""
        meta_features = pd.concat([
            X, 
            pd.Series(primary_preds, name='primary_pred', index=X.index),
            pd.Series(np.abs(primary_preds), name='primary_confidence', index=X.index)
        ], axis=1)
        
        # Entrenar con Purged K-Fold
        self.meta.fit(meta_features, meta_labels, sample_weight=sample_weights)
    
    def predict(self, X, close_current):
        """Output: side × probability → position size"""
        primary_side = self.primary.predict(X)  # {-1, 1}
        meta_features = self.prepare_meta_features(X, primary_side)
        meta_prob = self.meta.predict_proba(meta_features)[:, 1]
        
        # Kelly sizing: position = side × meta_probability
        position_size = primary_side * meta_prob
        return position_size
```

### Stacking ensemble sin data leakage temporal

```python
from sklearn.model_selection import TimeSeriesSplit

class TemporalStackingEnsemble:
    def __init__(self, base_models, meta_learner=None):
        """
        base_models: dict de {nombre: modelo} (XGBoost, TFT, iTransformer, N-HiTS)
        meta_learner: LogisticRegression o Ridge (simple = menos overfit)
        """
        self.base_models = base_models
        self.meta = meta_learner or LogisticRegression(C=1.0, max_iter=1000)
    
    def generate_oof_predictions(self, X, y, n_splits=5):
        """Out-of-fold predictions con expanding window temporal"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            for j, (name, model) in enumerate(self.base_models.items()):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                if hasattr(model, 'predict_proba'):
                    oof[val_idx, j] = model.predict_proba(X.iloc[val_idx])[:, 1]
                else:
                    oof[val_idx, j] = model.predict(X.iloc[val_idx])
        
        return oof
    
    def fit(self, X, y):
        oof = self.generate_oof_predictions(X, y)
        valid_mask = ~np.isnan(oof).any(axis=1)
        self.meta.fit(oof[valid_mask], y.values[valid_mask])
        # Reentrenar base models en datos completos
        for model in self.base_models.values():
            model.fit(X, y)
    
    def predict_proba(self, X):
        base_preds = np.column_stack([
            m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') 
            else m.predict(X) for m in self.base_models.values()
        ])
        return self.meta.predict_proba(base_preds)[:, 1]
```

### Calibración de probabilidades para Kelly sizing

**Probabilidades sin calibrar → Kelly criterion sobre-apuesta o sub-apuesta → ruina o subóptimo.** XGBoost es notoriamente mal calibrado; TFT tiende a ser sobreconfiado.

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss

class CalibratedTradingModel:
    def __init__(self, base_model, method='isotonic'):
        """
        'sigmoid' (Platt): paramétrico, bueno con pocas muestras (<500)
        'isotonic': no-paramétrico, más flexible, necesita >1000 muestras
        """
        self.calibrated = CalibratedClassifierCV(
            estimator=base_model, 
            method=method,
            cv=TimeSeriesSplit(n_splits=5)
        )
    
    def fit(self, X, y):
        self.calibrated.fit(X, y)
    
    def predict_with_kelly(self, X, b_ratio=1.5):
        """
        b_ratio = avg_win / avg_loss (típicamente 1.2-2.0 con triple barrier)
        Kelly: f* = (p*b - q) / b donde p = prob calibrada
        """
        p = self.calibrated.predict_proba(X)[:, 1]
        q = 1 - p
        kelly = (p * b_ratio - q) / b_ratio
        return np.clip(kelly * 0.25, 0, 0.1)  # Quarter-Kelly, max 10%
    
    def evaluate_calibration(self, X, y):
        probs = self.calibrated.predict_proba(X)[:, 1]
        return {
            'brier_score': brier_score_loss(y, probs),     # Target < 0.20
            'log_loss': log_loss(y, probs),                 # Target < 0.65
            'ece': expected_calibration_error(y, probs, n_bins=10),
        }
```

### Detección de régimen con HMM de 3 estados

```python
from hmmlearn.hmm import GaussianHMM

class RegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm = GaussianHMM(
            n_components=n_regimes, 
            covariance_type='full',   # Captura correlaciones entre features
            n_iter=500, 
            random_state=42,
            tol=1e-4
        )
    
    def fit(self, returns, volatility, volume_zscore=None):
        """Features: log-returns + volatilidad realizada + volumen normalizado"""
        X = np.column_stack([returns.values, volatility.values])
        if volume_zscore is not None:
            X = np.column_stack([X, volume_zscore.values])
        X = np.nan_to_num(X)
        
        # Múltiples inicializaciones, seleccionar mejor por log-likelihood
        best_score, best_model = -np.inf, None
        for seed in range(10):
            model = GaussianHMM(n_components=self.n_regimes, covariance_type='full',
                               n_iter=500, random_state=seed)
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score, best_model = score, model
        self.hmm = best_model
        
        states = self.hmm.predict(X)
        # Etiquetar regímenes por volatilidad media
        regime_vols = {s: volatility[states == s].mean() for s in range(self.n_regimes)}
        self.regime_map = dict(zip(
            sorted(regime_vols, key=regime_vols.get),
            ['low_vol', 'normal', 'high_vol']
        ))
        return states
    
    def regime_conditional_predict(self, X, regime_models):
        """Soft-switching: ponderar predicciones por probabilidad de régimen"""
        probs = self.hmm.predict_proba(X)
        predictions = np.zeros(len(X))
        for regime_id, model in regime_models.items():
            predictions += probs[:, regime_id] * model.predict(X)
        return predictions
```

### Sample weighting por uniqueness

```python
def compute_sample_weights(events_t1, close, decay=0.5):
    """
    Uniqueness: para cada bar, contar labels concurrentes. 
    Weight = 1/concurrencia promedio durante el span del label.
    """
    # Concurrencia
    t1 = events_t1.dropna()
    idx = close.index
    concurrency = pd.Series(0, index=idx, dtype=float)
    for start, end in t1.items():
        concurrency[start:end] += 1
    
    # Uniqueness por sample
    uniqueness = pd.Series(index=t1.index, dtype=float)
    for start, end in t1.items():
        avg_conc = concurrency[start:end].mean()
        uniqueness[start] = 1.0 / avg_conc if avg_conc > 0 else 1.0
    
    # Time decay (exponential, favorece recientes)
    n = len(uniqueness)
    time_weights = np.exp(-decay * np.arange(n)[::-1] / n)
    
    return uniqueness * time_weights
```

### Métricas de evaluación Fase 3

- **Meta-labeling F1**: target >0.60 (vs ~0.45-0.50 sin meta-labeling)
- **Precision**: target >0.55 (key metric — trades más selectivos pero más rentables)
- **Brier Score** post-calibración: <0.20
- **Ensemble vs best individual model**: >5% mejora en Sharpe
- **Regime detection accuracy**: verificar via stability de labels en períodos conocidos

**Librerías Fase 3**: `mlfinlab`, `hmmlearn`, `scikit-learn`, `xgboost`, `skfolio`

**Tiempo estimado**: 6-7 semanas

---

## Fase 4: Gestión de riesgo cuantitativa de nivel hedge fund (Semanas 20-26)

La diferencia entre un sistema de trading rentable y uno profesional está en cómo gestiona el riesgo. **HRP supera a la optimización mean-variance en Sharpe OOS por ~31%** según tests de Monte Carlo de López de Prado, y Black-Litterman convierte predicciones ML en allocations coherentes.

### Black-Litterman con predicciones ML como views

```python
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import risk_models, EfficientFrontier

class MLBlackLitterman:
    def __init__(self, instruments, tau=0.05):
        self.instruments = instruments
        self.tau = tau
    
    def compute_allocation(self, returns_history, ml_predictions, model_confidences):
        """
        ml_predictions: dict {instrument: expected_return} del ensemble
        model_confidences: dict {instrument: 0-1} de calibración del modelo
        """
        # Covarianza exponencial (da más peso a datos recientes)
        cov_matrix = risk_models.exp_cov(returns_history, span=180)
        
        # Market-implied equilibrium (prior)
        # Para crypto+forex, usar equal-weight como proxy de market cap
        market_weights = pd.Series(1/len(self.instruments), index=self.instruments)
        
        bl = BlackLittermanModel(
            cov_matrix,
            absolute_views=ml_predictions,
            omega="idzorek",                    # Calibra Omega desde confidences
            view_confidences=list(model_confidences.values()),
            tau=self.tau,
            pi="equal"                          # O usar market_implied_risk_premium
        )
        
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()
        
        ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0.0, 0.30))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        cleaned = ef.clean_weights(cutoff=0.01)
        
        return cleaned, ef.portfolio_performance(verbose=False)
```

### HRP con CVaR como medida de riesgo

```python
import riskfolio as rp

class HRPAllocator:
    def __init__(self):
        pass
    
    def optimize(self, returns_df, risk_measure='CVaR'):
        """
        HRP: no requiere inversión de matriz de covarianza → robusto con 7 activos
        CVaR como risk measure → tail-aware (crucial para crypto)
        """
        port = rp.HCPortfolio(returns=returns_df)
        weights = port.optimization(
            model='HRP',              # Hierarchical Risk Parity
            codependence='pearson',   # O 'gerber2' para robustez
            rm='CVaR',               # Conditional Value at Risk
            rf=0.04/252,             # Risk-free diario
            linkage='ward',           # Clustering aglomerativo
            leaf_order=True           # Optimiza orden de hojas
        )
        return weights
    
    def blend_with_kelly(self, hrp_weights, kelly_weights, blend=0.6):
        """60% HRP (robustez) + 40% Kelly (growth) = sweet spot"""
        blended = blend * hrp_weights.values.flatten() + (1-blend) * kelly_weights
        return blended / blended.sum()
```

### Multi-Asset Kelly con constraints

```python
from scipy.optimize import minimize

def multi_asset_kelly(mu, cov_matrix, risk_free=0.04/252, 
                      max_leverage=1.0, kelly_fraction=0.25):
    """
    Kelly multidimensional: max g(f) = Σ fi(μi-r) - 0.5 Σij fi·fj·Σij
    Con constraints de apalancamiento y quarter-Kelly
    """
    n = len(mu)
    excess = mu - risk_free
    
    def neg_growth(f):
        return -(f @ excess - 0.5 * f @ cov_matrix @ f)
    
    bounds = [(0, max_leverage/n * 3)] * n  # Max per-asset
    constraints = [
        {'type': 'ineq', 'fun': lambda f: max_leverage - np.sum(np.abs(f))}
    ]
    
    result = minimize(neg_growth, np.ones(n)/n, bounds=bounds,
                     constraints=constraints, method='SLSQP')
    
    return result.x * kelly_fraction  # Quarter-Kelly para control de drawdown

def bayesian_kelly_update(prior_alpha, prior_beta, win, loss):
    """Kelly Bayesiano: actualiza posterior con resultados de trading"""
    post_alpha = prior_alpha + win
    post_beta = prior_beta + loss
    p_est = post_alpha / (post_alpha + post_beta)
    # Confianza crece con más observaciones
    confidence = 1 - 1/np.sqrt(post_alpha + post_beta)
    return p_est, confidence, post_alpha, post_beta
```

### DCC-GARCH para correlaciones dinámicas

```python
class DynamicCorrelationEngine:
    def __init__(self, returns_df):
        self.returns = returns_df
    
    def ewma_cov(self, halflife=63):
        """EWMA con half-life de ~3 meses (estándar RiskMetrics)"""
        return self.returns.ewm(halflife=halflife).cov()
    
    def correlation_spike_adjustment(self, base_weights, threshold=0.7):
        """Reduce posiciones cuando correlaciones entre activos superan umbral"""
        corr = self.returns.tail(60).corr()  # Ventana reciente
        adjusted = base_weights.copy()
        
        for i in range(len(adjusted)):
            for j in range(i+1, len(adjusted)):
                if abs(corr.iloc[i,j]) > threshold:
                    excess = abs(corr.iloc[i,j]) - threshold
                    scale = max(0.3, 1.0 - excess * 2)
                    adjusted.iloc[i] *= scale
                    adjusted.iloc[j] *= scale
        
        return adjusted / adjusted.sum()
```

### Framework integrado de riesgo profesional

```python
class ProfessionalRiskManager:
    def __init__(self, instruments, max_cvar=-0.05, max_dd=-0.15):
        self.bl = MLBlackLitterman(instruments)
        self.hrp = HRPAllocator()
        self.dd_manager = BaseRiskEngine(max_drawdown=max_dd)
        self.corr_engine = None
        self.max_cvar = max_cvar
    
    def compute_final_positions(self, returns_history, ml_predictions, 
                                 confidences, current_equity):
        """Pipeline completo: BL → HRP blend → CVaR check → correlation adj → DD scale"""
        
        # 1. Black-Litterman allocation con ML views
        bl_weights, _ = self.bl.compute_allocation(
            returns_history, ml_predictions, confidences)
        
        # 2. HRP allocation (robust, sin inversión de covarianza)
        hrp_weights = self.hrp.optimize(returns_history, risk_measure='CVaR')
        
        # 3. Multi-asset Kelly
        mu = returns_history.mean().values
        cov = returns_history.cov().values
        kelly_w = multi_asset_kelly(mu, cov, kelly_fraction=0.25)
        
        # 4. Blend: 40% BL + 35% HRP + 25% Kelly
        bl_arr = np.array([bl_weights.get(i, 0) for i in returns_history.columns])
        hrp_arr = hrp_weights.values.flatten()
        blended = 0.40 * bl_arr + 0.35 * hrp_arr + 0.25 * kelly_w
        blended = np.maximum(blended, 0)
        blended /= blended.sum()
        
        # 5. CVaR constraint check
        port_returns = (returns_history * blended).sum(axis=1)
        cvar_95 = np.percentile(port_returns, 5)
        if cvar_95 < self.max_cvar:
            blended *= abs(self.max_cvar / cvar_95)
        
        # 6. Correlation spike adjustment
        self.corr_engine = DynamicCorrelationEngine(returns_history)
        blended_series = pd.Series(blended, index=returns_history.columns)
        blended_series = self.corr_engine.correlation_spike_adjustment(blended_series)
        
        # 7. Drawdown scaling
        dd_scale = self.dd_manager.position_scale(current_equity)
        
        return blended_series * dd_scale
```

### Métricas de evaluación Fase 4

- **Portfolio Sharpe** (walk-forward): target >1.5 anualizado
- **Maximum Drawdown**: target <15%
- **CVaR 95% diario**: <-3% del portfolio
- **Calmar Ratio** (return/max_dd): target >2.0
- **Diversification ratio**: >1.3
- **Correlación de drawdowns** entre activos: monitoreo continuo

**Librerías Fase 4**: `pypfopt`, `riskfolio-lib`, `arch`, `cvxpy`, `scipy`

**Tiempo estimado**: 5-7 semanas

---

## Fase 5: Adaptación online y producción autónoma (Semanas 27-34)

El mercado cambia constantemente. Un sistema profesional detecta drift automáticamente, reentrena cuando es necesario, y opera con mínima intervención. Esta fase añade **foundation models como miembros del ensemble**, monitoreo continuo, y la infraestructura para que el sistema sea autosuficiente.

### Detección de concept drift con ADWIN

```python
from river import drift

class DriftMonitor:
    def __init__(self):
        self.detectors = {
            'prediction_error': drift.ADWIN(delta=0.002),
            'feature_drift': {},  # Un detector por feature crítico
            'return_regime': drift.PageHinkley(delta=0.005, threshold=50)
        }
        self.drift_log = []
    
    def update(self, prediction_error, feature_values=None, timestamp=None):
        """Llamar después de cada predicción cuando se conoce el resultado"""
        self.detectors['prediction_error'].update(prediction_error)
        
        if self.detectors['prediction_error'].drift_detected:
            self.drift_log.append({
                'timestamp': timestamp,
                'type': 'concept_drift',
                'detector': 'ADWIN',
                'action': 'RETRAIN_FULL'
            })
            self.detectors['prediction_error'] = drift.ADWIN(delta=0.002)
            return 'RETRAIN_FULL'
        
        # Feature drift via KS-test
        if feature_values is not None:
            from scipy.stats import ks_2samp
            for fname, recent_vals in feature_values.items():
                if fname not in self.historical_features:
                    continue
                stat, p_val = ks_2samp(self.historical_features[fname], recent_vals)
                if p_val < 0.01:  # Significant distribution shift
                    return 'RETRAIN_INCREMENTAL'
        
        return 'NO_ACTION'
```

### Champion-Challenger para deployment seguro

```python
class ChampionChallenger:
    def __init__(self, champion_model, evaluation_window=500):
        self.champion = champion_model
        self.challenger = None
        self.window = evaluation_window
        self.champion_scores = []
        self.challenger_scores = []
    
    def deploy_challenger(self, new_model):
        self.challenger = new_model
        self.challenger_scores = []
    
    def evaluate_and_promote(self, X, y_true):
        """Paper-trade challenger en paralelo; promover si supera por >5%"""
        champ_pred = self.champion.predict(X)
        champ_score = evaluate_trading_metric(champ_pred, y_true)
        self.champion_scores.append(champ_score)
        
        if self.challenger:
            chall_pred = self.challenger.predict(X)
            chall_score = evaluate_trading_metric(chall_pred, y_true)
            self.challenger_scores.append(chall_score)
            
            if len(self.challenger_scores) >= self.window:
                champ_mean = np.mean(self.champion_scores[-self.window:])
                chall_mean = np.mean(self.challenger_scores[-self.window:])
                
                if chall_mean > champ_mean * 1.05:  # 5% superior
                    self.champion = self.challenger
                    self.challenger = None
                    return 'PROMOTED'
                elif chall_mean < champ_mean * 0.90:  # 10% inferior
                    self.challenger = None
                    return 'REJECTED'
        
        return 'EVALUATING'
```

### Foundation Models como ensemble members (zero-shot)

```python
# TimesFM 2.5 — Zero-shot, 200M params, entrenado en 100B datapoints
import timesfm
tfm = timesfm.TimesFM(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.5-200m-pytorch"
    ),
)
# Forecast zero-shot (sin entrenamiento!)
forecast = tfm.forecast(context=price_history[-512:], horizon=12)

# Chronos-Bolt — 250x más rápido que Chronos original, 5% más preciso
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda",
    torch_dtype=torch.float32,
)
# Forecast probabilístico (percentiles)
forecast = pipeline.predict(
    context=torch.tensor(price_history[-512:]).unsqueeze(0),
    prediction_length=12,
    num_samples=100,
)
median = forecast.median(dim=1)
p10, p90 = forecast.quantile(0.1, dim=1), forecast.quantile(0.9, dim=1)
```

**Uso óptimo**: no como modelo principal sino como **miembro del ensemble** que aporta una perspectiva complementaria sin costo de entrenamiento. Especialmente valioso durante regime shifts cuando modelos entrenados se degradan.

### FinGPT/FinBERT como generador de features de sentimiento

```python
from transformers import pipeline as hf_pipeline

class SentimentFeatureGenerator:
    def __init__(self):
        self.finbert = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    def process_headlines(self, headlines_list):
        """Convierte headlines en features numéricas para el pipeline ML"""
        results = self.finbert(headlines_list, truncation=True, max_length=512)
        
        scores = []
        for r in results:
            if r['label'] == 'positive': scores.append(r['score'])
            elif r['label'] == 'negative': scores.append(-r['score'])
            else: scores.append(0)
        
        return {
            'sentiment_mean': np.mean(scores),
            'sentiment_std': np.std(scores),
            'bullish_ratio': sum(1 for s in scores if s > 0) / len(scores),
            'headline_count': len(scores),
            'extreme_sentiment': int(abs(np.mean(scores)) > 0.7)
        }
```

### Pipeline de reentrenamiento automatizado

```python
class AutoRetrainingPipeline:
    def __init__(self, models, drift_monitor, champion_challenger):
        self.models = models
        self.drift = drift_monitor
        self.cc = champion_challenger
        self.retrain_history = []
    
    def scheduled_check(self, new_data, timestamp):
        """
        Frecuencia: 5m models → check diario; 1h models → check semanal
        """
        action = self.drift.update(
            prediction_error=compute_recent_error(new_data),
            timestamp=timestamp
        )
        
        if action == 'RETRAIN_FULL':
            new_model = full_retrain(self.models, new_data)
            self.cc.deploy_challenger(new_model)
            self.retrain_history.append({
                'timestamp': timestamp, 'type': 'full', 'trigger': 'concept_drift'
            })
        
        elif action == 'RETRAIN_INCREMENTAL':
            # Fine-tune con datos recientes (1-2 epochs, LR bajo)
            updated = incremental_update(self.models, new_data[-1000:], lr=1e-5)
            self.cc.deploy_challenger(updated)
        
        # Promotion check
        promotion_status = self.cc.evaluate_and_promote(
            new_data.features, new_data.labels)
        
        return action, promotion_status
```

### MLOps y monitoreo

```python
import mlflow

class TradingMLOps:
    def __init__(self, experiment_name="trading_system"):
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(self, model, params, metrics, data_version):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics({
                'sharpe_oos': metrics['sharpe'],
                'max_drawdown': metrics['max_dd'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'f1_score': metrics['f1'],
                'brier_score': metrics['brier'],
                'wf_efficiency': metrics['wf_eff'],
            })
            mlflow.log_param('data_version', data_version)
            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name=f"ensemble_v{data_version}")
    
    # Grafana dashboards via Prometheus metrics:
    # - prediction_latency_ms (histogram)
    # - model_accuracy_rolling_24h (gauge)
    # - portfolio_drawdown_current (gauge)  
    # - drift_detected_count (counter)
    # - active_positions_count (gauge)
    # - daily_pnl (gauge)
```

### Métricas de evaluación Fase 5

- **Drift detection latency**: <24h para concept drift significativo
- **Retraining frequency**: 1-4 veces/mes (demasiado = unstable, muy poco = stale)
- **Champion-challenger promotion rate**: 20-40% (indica modelos mejoran regularmente)
- **Foundation model contribution**: verificar via ablation study en ensemble
- **System uptime**: >99.5%
- **End-to-end latency** (signal → order): <1s para 5m, <5s para 1h

**Librerías Fase 5**: `river`, `mlflow`, `timesfm`, `chronos-forecasting`, `transformers`, `feast`, `redis`, `grafana/prometheus`

**Tiempo estimado**: 7-8 semanas

---

## El stack tecnológico completo en una vista

La arquitectura final integra todas las fases en un pipeline coherente. La tabla siguiente resume las dependencias críticas:

| Capa | Componentes | Librerías principales |
|------|------------|----------------------|
| **Datos** | Parquet + DuckDB (offline), Redis (online cache), ccxt + WebSockets | `duckdb`, `redis`, `ccxt`, `feast` |
| **Features** | Microestructura OHLCV, on-chain, cross-asset, GARCH, sentimiento | `arch`, `hmmlearn`, `transformers`, `scipy` |
| **Etiquetado** | Triple-barrier, meta-labels, sample weights, volume bars | `mlfinlab`, `quantreo` |
| **Modelos** | iTransformer, PatchTST, N-HiTS, XGBoost, TimesFM, Chronos | `neuralforecast`, `xgboost`, `timesfm`, `chronos` |
| **Ensemble** | Stacking temporal, meta-labeling, BMA, regime-conditional | `scikit-learn`, `hmmlearn` |
| **Validación** | CPCV, walk-forward, calibración | `skfolio`, `scikit-learn` |
| **Riesgo** | BL + HRP + Kelly, CVaR, DCC correlaciones, circuit breakers | `pypfopt`, `riskfolio-lib`, `cvxpy` |
| **MLOps** | Drift detection, champion-challenger, experiment tracking | `river`, `mlflow`, `grafana` |

---

## Por qué este orden y no otro

La secuencia de fases no es arbitraria. **La Fase 1 es multiplicativa**: sin triple-barrier y purged CV, cualquier modelo SOTA parecerá bueno en backtest pero fallará en producción — el research confirma que modelos DL no superan persistencia naive sin etiquetado correcto. La Fase 2 introduce los modelos que capturan estructura real del mercado. La Fase 3 es donde la precisión da el salto más grande porque meta-labeling convierte recall en precisión accionable. La Fase 4 es lo que separa un bot rentable de un sistema profesional — la mayoría de los hedge funds ganan dinero por gestión de riesgo superior, no por predicción superior. La Fase 5 asegura que el sistema no se degrade con el tiempo.

**Presupuesto total estimado: 28-34 semanas** para un developer Python avanzado trabajando full-time. Las fases pueden solaparse parcialmente (features de Fase 2 mientras se valida Fase 1). El factor crítico de éxito no es la complejidad del modelo sino la **disciplina en validación**: si el Walk-Forward Efficiency cae por debajo de 0.5 en cualquier fase, hay que detenerse y diagnosticar antes de avanzar.