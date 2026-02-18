# 🤖 BOT-CRIPTO: Manual Maestro de Operaciones y Estrategia (RTX 4090 Edition)

Este documento resume la configuración técnica, la arquitectura del modelo y la estrategia de Day Trading implementada para el despliegue en la GPU RTX 4090.

---

## 1. Infraestructura y Acceso (GPU RunPod)

### Detalles de Conexión
*   **Host:** `213.173.107.85`
*   **Puerto SSH:** `19355`
*   **Usuario:** `root`
*   **Autenticación:** Clave SSH Privada (`id_ed25519`)
*   **Hardware:** NVIDIA RTX 4090 (24GB VRAM) + 48 Cores CPU.

### Comandos de Gestión Rápidos
*   **Ver Logs de Entrenamiento:** `ssh -p 19355 root@213.173.107.85 "tail -f /workspace/logs/training.log"`
*   **Ver Uso de GPU:** `ssh -p 19355 root@213.173.107.85 "nvidia-smi"`
*   **Estado del Entrenamiento:** `ssh -p 19355 root@213.173.107.85 "python /workspace/monitor_training.py"`

---

## 2. El Modelo: Temporal Fusion Transformer (TFT)

### ¿Por qué TFT?
A diferencia de las redes neuronales simples (como LSTM o GRU), el TFT es un modelo de **Deep Learning de última generación** diseñado específicamente para series temporales financieras por las siguientes razones:
1.  **Variable Selection:** Identifica automáticamente qué indicadores (RSI, MACD, Volumen, etc.) son importantes en cada momento y descarta el "ruido".
2.  **Mecanismo de Atención:** Permite al bot "mirar atrás" en puntos específicos del pasado (por ejemplo, qué pasó en la apertura de New York de ayer) para predecir el futuro.
3.  **Probabilidades (Quantiles):** No te da un solo precio, te da un rango (P10, P50, P90). Esto permite calcular el riesgo real antes de entrar.

### Optimización para RTX 4090
*   **Precisión BF16 (Bfloat16):** Utiliza los Tensor Cores de la 4090 para procesar datos el doble de rápido sin perder precisión por desbordamiento (overflow).
*   **Batch Size 1024:** Saturamos la memoria de la tarjeta para que el entrenamiento sea masivo y rápido.
*   **4 Capas LSTM:** Hemos profundizado la red para capturar patrones más complejos de 2 años de historia.

---

## 3. Estrategia de Datos: El "Edge" del Mercado

### ¿Por qué 1 Hora y 5 Minutos?
Hemos implementado una estrategia de **Doble Horizonte**:
*   **Velas de 1 Hora (El Cerebro):** Define la tendencia principal. Evita que el bot haga "Long" cuando el mercado está colapsando en macro.
*   **Velas de 5 Minutos (El Gatillo):** Define la entrada exacta. Minimiza el Stop Loss y maximiza el Risk/Reward.

### Los 2 Años de Historia (17,400+ velas)
Usar 2 años es el "punto dulce" porque:
*   Cubre mercados alcistas, bajistas y laterales.
*   Proporciona suficiente masa estadística para que el TFT aprenda a ignorar "mechazos" falsos.
*   Permite al modelo aprender la estacionalidad (ej: BTC suele ser más volátil los martes y miércoles).

---

## 4. Gestión de Riesgo Avanzada

### Kelly Criterion (Fraccional)
El bot no apuesta una cantidad fija. Usa la fórmula de Kelly para calcular el tamaño de la posición basándose en:
*   **Confianza del Modelo:** Si `prob_up` es 80%, la posición es mayor.
*   **Ratio de Pago:** Si el beneficio potencial es mucho mayor que el riesgo, aumenta la apuesta.
*   *Nota: Usamos un "Kelly Fraccional" (0.2) para ser conservadores y evitar quiebras por rachas negativas.*

### Trailing Stop Loss Dinámico
A medida que el precio sube, el Stop Loss "persigue" al precio a una distancia calculada por la volatilidad (ATR). Si el mercado se da la vuelta, sales con ganancias en lugar de en pérdidas.

---

## 5. Hoja de Ruta para Máxima Precisión

Para llevar este modelo al siguiente nivel de precisión (+80% de acierto), estas son las configuraciones adicionales recomendadas:

1.  **Análisis de Sentimiento en Tiempo Real:** Integrar el miedo/codicia (Fear & Greed Index) y el sentimiento de Twitter/Telegram para detectar movimientos irracionales.
2.  **Orderbook Imbalance:** No mirar solo el precio, sino cuántas órdenes de compra vs venta hay en el nivel actual. Esto detecta "paredes" de ballenas.
3.  **Correlación con el SP500 y DXY:** BTC ya no se mueve solo. Integrar el índice del dólar y el mercado de acciones ayuda al modelo a entender el contexto global.
4.  **Fine-Tuning de Volatilidad:** Configurar el modelo para que cambie de parámetros automáticamente cuando la volatilidad sube de un umbral (modo "crisis").

---
**Estado del Sistema:**
*   **BTC/USDT 1h:** Entrenamiento al 40%.
*   **SOL/USDT:** Datos preparados, entrenamiento programado a continuación.
*   **Bot de Ejecución:** Actualizado con Kelly Criterion.
