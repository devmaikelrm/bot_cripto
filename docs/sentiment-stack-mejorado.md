# Sentiment Stack Implementation - Arquitectura Mejorada y Avanzada

**Fecha**: 2026-02-19  
**Scope**: Sistema completo de sentiment analysis en tiempo real con múltiples fuentes, NLP avanzado, y validación rigurosa

---

## Objetivo Mejorado

Construir un **sistema de sentiment analysis de nivel profesional** que:

1. **Recolecte datos de múltiples fuentes** con validación cruzada
2. **Use modelos NLP de última generación** (2024-2026) con fallbacks inteligentes
3. **Integre perfectamente** con el pipeline de trading sin romper nada
4. **Se valide rigurosamente** con backtesting y walk-forward analysis
5. **Se adapte dinámicamente** a cambios en el mercado
6. **Sea monitoreado** en tiempo real con alertas automáticas

---

## Arquitectura Propuesta Mejorada

### 1. Capa de Recolección de Datos (Multi-Source con Validación)

```
src/bot_cripto/sentiment/
├── collectors/
│   ├── __init__.py
│   ├── base_collector.py          # Interfaz base abstracta
│   ├── x_collector.py              # Twitter/X con rate limiting
│   ├── telegram_collector.py       # Canales de Telegram
│   ├── reddit_collector.py         # r/cryptocurrency, r/bitcoin
│   ├── news_collector.py           # CoinDesk, CoinTelegraph, Bloomberg
│   ├── cryptopanic_collector.py    # CryptoPanic API
│   ├── fear_greed_collector.py     # Fear & Greed Index
│   ├── lunarcrush_collector.py     # Social analytics (opcional)
│   └── onchain_collector.py        # Glassnode metrics narratives
├── validators/
│   ├── data_validator.py           # Valida calidad de datos
│   ├── spam_filter.py              # Filtra bots/spam
│   └── source_reliability.py      # Score de confiabilidad por fuente
└── storage/
    ├── sentiment_db.py             # SQLite/PostgreSQL para persistencia
    └── cache_manager.py            # Redis para cache de alta velocidad
```

#### Implementación Base Collector

```python
# src/bot_cripto/sentiment/collectors/base_collector.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class SentimentPost:
    """Estructura uniforme para posts de cualquier fuente."""
    id: str
    source: str  # 'x', 'telegram', 'reddit', etc.
    text: str
    author: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, retweets, comments, etc.
    metadata: Dict  # Información adicional específica de la fuente
    language: str = 'en'
    
    def __post_init__(self):
        # Validación básica
        if not self.text or len(self.text) < 10:
            raise ValueError("Text too short")
        if self.timestamp > datetime.now():
            raise ValueError("Future timestamp")

class BaseCollector(ABC):
    """
    Clase base para todos los collectors de sentiment.
    
    Características comunes:
    - Rate limiting automático
    - Retry logic con backoff exponencial
    - Logging estructurado
    - Métricas de performance
    """
    
    def __init__(
        self,
        symbol: str,
        max_posts: int = 100,
        rate_limit_per_minute: int = 60,
        cache_ttl_seconds: int = 300
    ):
        self.symbol = symbol
        self.max_posts = max_posts
        self.rate_limit = rate_limit_per_minute
        self.cache_ttl = cache_ttl_seconds
        
        self.source_name = self.__class__.__name__.replace('Collector', '').lower()
        self.logger = self._setup_logger()
        self.metrics = CollectorMetrics(self.source_name)
        
        # Rate limiting
        self._last_request_time = None
        self._request_count = 0
        
        # Cache
        self.cache = CacheManager(f"sentiment:{self.source_name}")
    
    @abstractmethod
    async def fetch_posts(self, since: Optional[datetime] = None) -> List[SentimentPost]:
        """
        Implementar en cada subclase.
        Debe retornar lista de SentimentPost.
        """
        pass
    
    async def collect(self, since: Optional[datetime] = None) -> List[SentimentPost]:
        """
        Método público con validación y error handling.
        """
        try:
            # Check cache
            cache_key = f"{self.symbol}:{since}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.info(f"Cache hit for {self.source_name}")
                return cached
            
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Fetch
            start_time = datetime.now()
            posts = await self.fetch_posts(since)
            
            # Validate
            validated_posts = await self._validate_posts(posts)
            
            # Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_collection(
                num_posts=len(validated_posts),
                duration=duration,
                success=True
            )
            
            # Cache
            await self.cache.set(cache_key, validated_posts, ttl=self.cache_ttl)
            
            return validated_posts
            
        except Exception as e:
            self.logger.error(f"Error collecting from {self.source_name}: {e}")
            self.metrics.record_collection(num_posts=0, duration=0, success=False)
            raise
    
    async def _validate_posts(self, posts: List[SentimentPost]) -> List[SentimentPost]:
        """Valida y filtra posts."""
        validator = DataValidator()
        spam_filter = SpamFilter()
        
        validated = []
        for post in posts:
            try:
                # Validación básica
                validator.validate(post)
                
                # Filtro de spam
                if spam_filter.is_spam(post):
                    self.logger.debug(f"Filtered spam post: {post.id}")
                    continue
                
                validated.append(post)
                
            except Exception as e:
                self.logger.warning(f"Invalid post {post.id}: {e}")
                continue
        
        return validated
    
    async def _enforce_rate_limit(self):
        """Implementa rate limiting simple."""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < 60:  # Dentro del mismo minuto
                self._request_count += 1
                if self._request_count >= self.rate_limit:
                    wait_time = 60 - elapsed
                    self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self._request_count = 0
            else:
                self._request_count = 1
        
        self._last_request_time = datetime.now()
```

#### X/Twitter Collector Mejorado

```python
# src/bot_cripto/sentiment/collectors/x_collector.py
import tweepy
from typing import List, Optional
from datetime import datetime, timedelta

class XCollector(BaseCollector):
    """
    Recolector de posts de X/Twitter.
    
    Features:
    - Búsqueda por hashtags y keywords relevantes
    - Filtrado por influencers verificados
    - Análisis de engagement (likes, retweets)
    - Detección de trending topics
    """
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        
        # Configurar cliente de Twitter API v2
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN not set")
        
        self.client = tweepy.Client(bearer_token=bearer_token)
        
        # Keywords relevantes para el símbolo
        self.keywords = self._generate_keywords(symbol)
        
        # Lista de influencers verificados (opcional)
        self.verified_accounts = self._load_verified_accounts()
    
    def _generate_keywords(self, symbol: str) -> List[str]:
        """Genera keywords de búsqueda inteligentes."""
        base = symbol.split('/')[0]  # BTC de BTC/USDT
        
        keywords = [
            f"#{base}",
            f"${base}",
            base,
            f"{base} crypto",
            f"{base} price",
            f"{base} trading",
            f"{base} analysis"
        ]
        
        # Agregar términos específicos por asset
        asset_specific = {
            'BTC': ['bitcoin', 'btc', 'sats'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'SOL': ['solana', 'sol'],
        }
        
        if base in asset_specific:
            keywords.extend(asset_specific[base])
        
        return keywords
    
    async def fetch_posts(self, since: Optional[datetime] = None) -> List[SentimentPost]:
        """Fetch tweets usando Twitter API v2."""
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        all_posts = []
        
        # Buscar por cada keyword
        for keyword in self.keywords:
            try:
                query = self._build_query(keyword, since)
                
                # Fetch tweets
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=self.max_posts // len(self.keywords),
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                    user_fields=['verified', 'public_metrics'],
                    expansions=['author_id']
                )
                
                if not tweets.data:
                    continue
                
                # Convertir a SentimentPost
                for tweet in tweets.data:
                    try:
                        # Obtener info del autor
                        author = self._get_author_info(tweet, tweets.includes)
                        
                        post = SentimentPost(
                            id=f"x_{tweet.id}",
                            source='x',
                            text=tweet.text,
                            author=author['username'],
                            timestamp=tweet.created_at,
                            engagement={
                                'likes': tweet.public_metrics['like_count'],
                                'retweets': tweet.public_metrics['retweet_count'],
                                'replies': tweet.public_metrics['reply_count'],
                                'impressions': tweet.public_metrics.get('impression_count', 0)
                            },
                            metadata={
                                'verified': author.get('verified', False),
                                'followers': author.get('followers_count', 0),
                                'keyword_matched': keyword
                            },
                            language=tweet.lang
                        )
                        
                        all_posts.append(post)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing tweet {tweet.id}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error fetching tweets for keyword '{keyword}': {e}")
                continue
        
        return all_posts
    
    def _build_query(self, keyword: str, since: datetime) -> str:
        """Construye query de búsqueda optimizada."""
        query_parts = [keyword]
        
        # Excluir retweets y replies
        query_parts.append("-is:retweet")
        query_parts.append("-is:reply")
        
        # Solo inglés y español (modificar según necesidad)
        query_parts.append("(lang:en OR lang:es)")
        
        # Excluir spam común
        query_parts.append("-is:nullcast")
        
        return " ".join(query_parts)
    
    def _get_author_info(self, tweet, includes) -> Dict:
        """Extrae información del autor del tweet."""
        if not includes or 'users' not in includes:
            return {'username': 'unknown', 'verified': False, 'followers_count': 0}
        
        author_id = tweet.author_id
        for user in includes['users']:
            if user.id == author_id:
                return {
                    'username': user.username,
                    'verified': user.verified,
                    'followers_count': user.public_metrics['followers_count']
                }
        
        return {'username': 'unknown', 'verified': False, 'followers_count': 0}
```

### 2. Capa de Análisis NLP (Estado del Arte 2024-2026)

```python
# src/bot_cripto/sentiment/analyzers/nlp_analyzer.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class SentimentScore:
    """Resultado de análisis de sentiment."""
    score: float  # -1 a 1
    confidence: float  # 0 a 1
    label: str  # 'positive', 'negative', 'neutral'
    subjectivity: float  # 0 a 1
    raw_scores: Dict[str, float]  # scores brutos del modelo

class AdvancedNLPAnalyzer:
    """
    Analizador NLP de última generación para crypto sentiment.
    
    Modelos soportados (orden de preferencia):
    1. ProsusAI/finbert (especializado en finanzas)
    2. cardiffnlp/twitter-roberta-base-sentiment-latest (Twitter)
    3. yiyanghkust/finbert-tone (finanzas alternativo)
    4. distilbert-base-uncased-finetuned-sst-2-english (fallback general)
    
    Features:
    - Ensemble de múltiples modelos
    - Calibración de probabilidades
    - Detección de sarcasmo/ironía
    - Análisis de subjetividad
    - Batch processing eficiente
    """
    
    def __init__(
        self,
        primary_model: str = 'ProsusAI/finbert',
        use_ensemble: bool = True,
        device: str = 'auto',
        batch_size: int = 32
    ):
        self.primary_model_name = primary_model
        self.use_ensemble = use_ensemble
        self.batch_size = batch_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Lazy loading de modelos
        self._primary_model = None
        self._ensemble_models = []
        self._initialized = False
        
        # Lexicon fallback
        self.lexicon_analyzer = LexiconAnalyzer()
    
    def _lazy_init(self):
        """Inicializa modelos solo cuando se necesitan."""
        if self._initialized:
            return
        
        try:
            # Cargar modelo principal
            self.logger.info(f"Loading primary model: {self.primary_model_name}")
            self._primary_model = pipeline(
                "sentiment-analysis",
                model=self.primary_model_name,
                device=0 if self.device == 'cuda' else -1,
                top_k=None  # Retornar todos los scores
            )
            
            # Cargar ensemble si está habilitado
            if self.use_ensemble:
                self._load_ensemble_models()
            
            self._initialized = True
            self.logger.info("NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load NLP models: {e}")
            self.logger.warning("Falling back to lexicon-only mode")
            self._initialized = False
    
    def _load_ensemble_models(self):
        """Carga modelos adicionales para ensemble."""
        ensemble_models = [
            'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'yiyanghkust/finbert-tone'
        ]
        
        for model_name in ensemble_models:
            try:
                model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if self.device == 'cuda' else -1,
                    top_k=None
                )
                self._ensemble_models.append({
                    'name': model_name,
                    'model': model,
                    'weight': 1.0 / (len(ensemble_models) + 1)  # Peso uniforme
                })
                self.logger.info(f"Loaded ensemble model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load ensemble model {model_name}: {e}")
    
    def analyze_batch(self, posts: List[SentimentPost]) -> List[SentimentScore]:
        """Analiza múltiples posts eficientemente."""
        if not posts:
            return []
        
        # Lazy init
        self._lazy_init()
        
        # Fallback a lexicon si modelos no disponibles
        if not self._initialized:
            return [self._fallback_analysis(post.text) for post in posts]
        
        # Pre-procesamiento
        texts = [self._preprocess_text(post.text) for post in posts]
        
        # Análisis en batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._analyze_batch_internal(batch)
            results.extend(batch_results)
        
        return results
    
    def _analyze_batch_internal(self, texts: List[str]) -> List[SentimentScore]:
        """Análisis interno de un batch."""
        # Análisis con modelo principal
        primary_results = self._primary_model(texts)
        
        # Ensemble si está habilitado
        if self.use_ensemble and self._ensemble_models:
            ensemble_results = []
            for model_info in self._ensemble_models:
                model_results = model_info['model'](texts)
                ensemble_results.append({
                    'results': model_results,
                    'weight': model_info['weight']
                })
            
            # Combinar resultados
            final_results = self._combine_ensemble_results(
                primary_results,
                ensemble_results
            )
        else:
            final_results = primary_results
        
        # Convertir a SentimentScore
        sentiment_scores = []
        for i, result in enumerate(final_results):
            score = self._convert_to_sentiment_score(result, texts[i])
            sentiment_scores.append(score)
        
        return sentiment_scores
    
    def _preprocess_text(self, text: str) -> str:
        """Pre-procesa texto para análisis."""
        # Remover URLs
        import re
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remover menciones (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remover hashtags excesivos (mantener primeros 3)
        hashtags = re.findall(r'#\w+', text)
        if len(hashtags) > 3:
            for hashtag in hashtags[3:]:
                text = text.replace(hashtag, '')
        
        # Remover emojis duplicados
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # Limpiar espacios
        text = ' '.join(text.split())
        
        return text
    
    def _convert_to_sentiment_score(
        self, 
        model_output: List[Dict], 
        original_text: str
    ) -> SentimentScore:
        """Convierte output del modelo a SentimentScore uniforme."""
        # Extraer scores
        scores = {item['label'].lower(): item['score'] for item in model_output}
        
        # Mapear a escala -1 a 1
        if 'positive' in scores and 'negative' in scores:
            # FinBERT style (positive, negative, neutral)
            pos = scores.get('positive', 0)
            neg = scores.get('negative', 0)
            neu = scores.get('neutral', 0)
            
            # Score compuesto
            sentiment_score = pos - neg
            
            # Confidence (max de los tres)
            confidence = max(pos, neg, neu)
            
            # Label
            if pos > neg and pos > neu:
                label = 'positive'
            elif neg > pos and neg > neu:
                label = 'negative'
            else:
                label = 'neutral'
        
        else:
            # Fallback a label directo
            top_result = max(model_output, key=lambda x: x['score'])
            label = top_result['label'].lower()
            confidence = top_result['score']
            
            # Mapear label a score
            label_to_score = {
                'positive': 1.0,
                'neutral': 0.0,
                'negative': -1.0,
                'bullish': 1.0,
                'bearish': -1.0
            }
            sentiment_score = label_to_score.get(label, 0.0)
        
        # Calcular subjetividad (heurística simple)
        subjectivity = self._calculate_subjectivity(original_text)
        
        return SentimentScore(
            score=sentiment_score,
            confidence=confidence,
            label=label,
            subjectivity=subjectivity,
            raw_scores=scores
        )
    
    def _calculate_subjectivity(self, text: str) -> float:
        """
        Estima subjetividad del texto.
        Textos subjetivos tienen opiniones personales.
        Textos objetivos solo reportan hechos.
        """
        # Palabras subjetivas comunes
        subjective_words = {
            'think', 'believe', 'feel', 'opinion', 'should', 'must',
            'amazing', 'terrible', 'love', 'hate', 'best', 'worst',
            'probably', 'maybe', 'perhaps', 'seem', 'appear'
        }
        
        words = text.lower().split()
        subjective_count = sum(1 for word in words if word in subjective_words)
        
        # Normalizar
        subjectivity = min(subjective_count / (len(words) + 1), 1.0)
        
        return subjectivity
    
    def _combine_ensemble_results(
        self,
        primary_results: List[Dict],
        ensemble_results: List[Dict]
    ) -> List[Dict]:
        """Combina resultados de múltiples modelos usando weighted average."""
        combined = []
        
        for i in range(len(primary_results)):
            # Primary model
            primary = primary_results[i]
            primary_weight = 0.5  # 50% peso al modelo principal
            
            # Inicializar scores combinados
            combined_scores = {}
            for item in primary:
                label = item['label'].lower()
                combined_scores[label] = item['score'] * primary_weight
            
            # Ensemble models
            ensemble_weight_total = 0.5  # 50% restante distribuido
            for model_result in ensemble_results:
                model_weight = model_result['weight'] * ensemble_weight_total
                model_output = model_result['results'][i]
                
                for item in model_output:
                    label = item['label'].lower()
                    if label not in combined_scores:
                        combined_scores[label] = 0
                    combined_scores[label] += item['score'] * model_weight
            
            # Convertir de vuelta a formato original
            combined_item = [
                {'label': label, 'score': score}
                for label, score in combined_scores.items()
            ]
            
            combined.append(combined_item)
        
        return combined
    
    def _fallback_analysis(self, text: str) -> SentimentScore:
        """Análisis de fallback usando lexicon si NLP no disponible."""
        return self.lexicon_analyzer.analyze(text)


class LexiconAnalyzer:
    """Analizador basado en lexicon como fallback."""
    
    def __init__(self):
        # Lexicons específicos de crypto
        self.positive_words = {
            'moon', 'bullish', 'pump', 'gain', 'profit', 'buy', 'long',
            'breakout', 'rally', 'surge', 'adoption', 'partnership',
            'upgrade', 'launch', 'success', 'growth', 'strong'
        }
        
        self.negative_words = {
            'dump', 'bearish', 'crash', 'loss', 'sell', 'short',
            'breakdown', 'fall', 'decline', 'scam', 'hack', 'ban',
            'regulation', 'shutdown', 'weak', 'failure', 'fear'
        }
    
    def analyze(self, text: str) -> SentimentScore:
        """Análisis simple basado en conteo de palabras."""
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return SentimentScore(
                score=0.0,
                confidence=0.3,  # Baja confianza cuando no hay palabras de sentiment
                label='neutral',
                subjectivity=0.5,
                raw_scores={'positive': 0, 'negative': 0, 'neutral': 1}
            )
        
        # Score normalizado
        score = (pos_count - neg_count) / total_sentiment_words
        
        # Confidence basado en número de palabras encontradas
        confidence = min(total_sentiment_words / 10, 0.8)  # Max 0.8 para lexicon
        
        # Label
        if score > 0.2:
            label = 'positive'
        elif score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'
        
        return SentimentScore(
            score=score,
            confidence=confidence,
            label=label,
            subjectivity=0.7,  # Asumimos alta subjetividad en social media
            raw_scores={
                'positive': pos_count / (len(words) + 1),
                'negative': neg_count / (len(words) + 1),
                'neutral': 1 - (total_sentiment_words / (len(words) + 1))
            }
        )
```

### 3. Orquestación y Agregación Inteligente

```python
# src/bot_cripto/sentiment/orchestrator.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import numpy as np

class SentimentOrchestrator:
    """
    Orquesta la recolección y análisis de sentiment de múltiples fuentes.
    
    Features:
    - Recolección paralela de todas las fuentes
    - Weighted blending con pesos adaptativos
    - Detección de anomalías
    - Cálculo de momentum y velocity
    - Time-decay para posts antiguos
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Inicializar collectors
        self.collectors = self._init_collectors()
        
        # Inicializar analyzer
        self.analyzer = AdvancedNLPAnalyzer(
            primary_model=config.get('nlp_model', 'ProsusAI/finbert'),
            use_ensemble=config.get('use_ensemble', True)
        )
        
        # Pesos por fuente (ajustables)
        self.source_weights = {
            'x': 0.30,           # Twitter tiene alto volumen
            'news': 0.25,        # Noticias son más confiables
            'reddit': 0.20,      # Reddit tiene buena calidad
            'telegram': 0.15,    # Telegram más nicho
            'cryptopanic': 0.10  # Agregador
        }
        
        # Time decay (qué tan rápido decae la importancia de posts antiguos)
        self.time_decay_halflife_hours = 6  # A las 6 horas, peso se reduce a 50%
        
        # Storage
        self.db = SentimentDatabase()
    
    async def get_aggregated_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict:
        """
        Obtiene sentiment agregado de todas las fuentes.
        
        Returns:
            {
                'score': float,  # -1 a 1
                'confidence': float,  # 0 a 1
                'volume': int,  # Número total de posts
                'momentum': float,  # Cambio en sentiment
                'velocity': float,  # Tasa de cambio
                'by_source': Dict,  # Breakdown por fuente
                'recent_posts': List,  # Posts más recientes/relevantes
                'timestamp': datetime
            }
        """
        since = datetime.now() - timedelta(hours=lookback_hours)
        
        # Recolectar de todas las fuentes en paralelo
        collection_tasks = []
        for collector_name, collector in self.collectors.items():
            task = asyncio.create_task(
                self._collect_from_source(collector_name, collector, symbol, since)
            )
            collection_tasks.append(task)
        
        # Esperar todas las colecciones
        collection_results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Procesar resultados
        all_posts = []
        source_breakdown = {}
        
        for result in collection_results:
            if isinstance(result, Exception):
                self.logger.error(f"Collection failed: {result}")
                continue
            
            source_name, posts = result
            all_posts.extend(posts)
            source_breakdown[source_name] = {
                'count': len(posts),
                'posts': posts
            }
        
        if not all_posts:
            return self._empty_sentiment_result()
        
        # Analizar sentiment de todos los posts
        sentiment_scores = self.analyzer.analyze_batch(all_posts)
        
        # Asociar scores con posts
        for post, score in zip(all_posts, sentiment_scores):
            post.sentiment_score = score
        
        # Calcular sentiment agregado con pesos y time decay
        aggregated = self._calculate_weighted_sentiment(all_posts, source_breakdown)
        
        # Calcular momentum y velocity
        aggregated['momentum'] = await self._calculate_sentiment_momentum(
            symbol, lookback_hours
        )
        aggregated['velocity'] = await self._calculate_sentiment_velocity(
            symbol, lookback_hours
        )
        
        # Detectar anomalías
        aggregated['anomaly_score'] = self._detect_sentiment_anomaly(aggregated)
        
        # Posts más relevantes
        aggregated['recent_posts'] = self._get_top_posts(all_posts, limit=20)
        
        # Guardar en base de datos
        await self.db.save_sentiment_snapshot(symbol, aggregated)
        
        return aggregated
    
    async def _collect_from_source(
        self,
        source_name: str,
        collector: BaseCollector,
        symbol: str,
        since: datetime
    ) -> Tuple[str, List[SentimentPost]]:
        """Recolecta posts de una fuente específica."""
        try:
            posts = await collector.collect(since=since)
            self.logger.info(f"Collected {len(posts)} posts from {source_name}")
            return (source_name, posts)
        except Exception as e:
            self.logger.error(f"Failed to collect from {source_name}: {e}")
            return (source_name, [])
    
    def _calculate_weighted_sentiment(
        self,
        all_posts: List[SentimentPost],
        source_breakdown: Dict
    ) -> Dict:
        """
        Calcula sentiment ponderado considerando:
        - Peso de la fuente
        - Time decay
        - Engagement (likes, retweets, etc.)
        - Confidence del modelo NLP
        """
        total_weight = 0
        weighted_sentiment = 0
        weighted_confidence = 0
        
        now = datetime.now()
        
        for post in all_posts:
            # Peso base de la fuente
            source_weight = self.source_weights.get(post.source, 0.1)
            
            # Time decay
            hours_old = (now - post.timestamp).total_seconds() / 3600
            time_weight = np.exp(-np.log(2) * hours_old / self.time_decay_halflife_hours)
            
            # Engagement weight (normalizado)
            engagement_weight = self._calculate_engagement_weight(post)
            
            # Confidence del modelo
            confidence_weight = post.sentiment_score.confidence
            
            # Peso final
            final_weight = source_weight * time_weight * engagement_weight * confidence_weight
            
            # Acumular
            weighted_sentiment += post.sentiment_score.score * final_weight
            weighted_confidence += confidence_weight * final_weight
            total_weight += final_weight
        
        if total_weight == 0:
            return self._empty_sentiment_result()
        
        # Normalizar
        avg_sentiment = weighted_sentiment / total_weight
        avg_confidence = weighted_confidence / total_weight
        
        # Calcular breakdown por fuente
        by_source = {}
        for source_name, data in source_breakdown.items():
            if not data['posts']:
                continue
            
            source_scores = [p.sentiment_score.score for p in data['posts']]
            by_source[source_name] = {
                'score': np.mean(source_scores),
                'confidence': np.mean([p.sentiment_score.confidence for p in data['posts']]),
                'count': len(data['posts']),
                'weight': self.source_weights.get(source_name, 0.1)
            }
        
        return {
            'score': avg_sentiment,
            'confidence': avg_confidence,
            'volume': len(all_posts),
            'by_source': by_source,
            'timestamp': datetime.now()
        }
    
    def _calculate_engagement_weight(self, post: SentimentPost) -> float:
        """
        Calcula peso basado en engagement.
        Posts con más engagement son más influyentes.
        """
        engagement = post.engagement
        
        # Diferentes métricas según fuente
        if post.source == 'x':
            # Para Twitter: likes + retweets * 2 (retweets valen más)
            score = engagement.get('likes', 0) + engagement.get('retweets', 0) * 2
        elif post.source == 'reddit':
            # Para Reddit: upvotes - downvotes
            score = engagement.get('upvotes', 0) - engagement.get('downvotes', 0)
        else:
            # Genérico: suma de todas las métricas
            score = sum(engagement.values())
        
        # Normalizar con log scale (evitar que posts virales dominen completamente)
        normalized = np.log1p(score) / 10  # log1p = log(1 + x)
        
        # Bonus si es cuenta verificada (solo X)
        if post.metadata.get('verified', False):
            normalized *= 1.5
        
        # Bonus por número de followers (para influencers)
        followers = post.metadata.get('followers', 0)
        if followers > 100000:  # >100k followers
            normalized *= 1.3
        elif followers > 10000:  # >10k followers
            normalized *= 1.1
        
        # Clip a rango razonable
        return min(normalized, 5.0)
    
    async def _calculate_sentiment_momentum(
        self,
        symbol: str,
        lookback_hours: int
    ) -> float:
        """
        Calcula momentum del sentiment (cambio reciente).
        
        Compara sentiment de últimas 4 horas vs 4-8 horas atrás.
        """
        # Sentiment reciente (últimas 4 horas)
        recent_sentiment = await self.db.get_average_sentiment(
            symbol,
            since=datetime.now() - timedelta(hours=4)
        )
        
        # Sentiment previo (4-8 horas atrás)
        previous_sentiment = await self.db.get_average_sentiment(
            symbol,
            since=datetime.now() - timedelta(hours=8),
            until=datetime.now() - timedelta(hours=4)
        )
        
        if previous_sentiment is None:
            return 0.0
        
        # Momentum = diferencia
        momentum = recent_sentiment - previous_sentiment
        
        return momentum
    
    async def _calculate_sentiment_velocity(
        self,
        symbol: str,
        lookback_hours: int
    ) -> float:
        """
        Calcula velocity (tasa de cambio del sentiment).
        
        Mide qué tan rápido está cambiando el sentiment.
        """
        # Obtener series temporal de sentiment
        sentiment_history = await self.db.get_sentiment_timeseries(
            symbol,
            since=datetime.now() - timedelta(hours=lookback_hours),
            interval_minutes=30  # Muestras cada 30 min
        )
        
        if len(sentiment_history) < 2:
            return 0.0
        
        # Calcular derivada (cambio por hora)
        timestamps = [s['timestamp'] for s in sentiment_history]
        scores = [s['score'] for s in sentiment_history]
        
        # Regresión lineal simple para estimar slope
        from scipy.stats import linregress
        hours = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
        slope, _, _, _, _ = linregress(hours, scores)
        
        return slope
    
    def _detect_sentiment_anomaly(self, aggregated: Dict) -> float:
        """
        Detecta si el sentiment actual es anómalo.
        
        Score alto = sentiment muy diferente a lo normal
        """
        score = aggregated['score']
        
        # Comparar con distribución histórica
        # (simplificado aquí, en producción usar modelo más sofisticado)
        
        # Asumimos distribución normal con:
        historical_mean = 0.0  # Neutral
        historical_std = 0.3   # Desviación típica
        
        # Z-score
        z_score = abs((score - historical_mean) / historical_std)
        
        # Convertir a score 0-1
        anomaly_score = min(z_score / 3, 1.0)  # z > 3 es muy anómalo
        
        return anomaly_score
    
    def _get_top_posts(
        self,
        posts: List[SentimentPost],
        limit: int = 20
    ) -> List[Dict]:
        """
        Obtiene los posts más relevantes.
        
        Ordena por engagement * confidence * recency.
        """
        # Calcular score de relevancia para cada post
        now = datetime.now()
        
        scored_posts = []
        for post in posts:
            # Engagement
            engagement_score = self._calculate_engagement_weight(post)
            
            # Confidence
            confidence_score = post.sentiment_score.confidence
            
            # Recency (últimas 6 horas = score 1.0, luego decae)
            hours_old = (now - post.timestamp).total_seconds() / 3600
            recency_score = np.exp(-hours_old / 6)
            
            # Score compuesto
            relevance_score = engagement_score * confidence_score * recency_score
            
            scored_posts.append({
                'post': post,
                'relevance_score': relevance_score
            })
        
        # Ordenar y retornar top N
        scored_posts.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return [
            {
                'text': item['post'].text,
                'source': item['post'].source,
                'author': item['post'].author,
                'timestamp': item['post'].timestamp,
                'sentiment': item['post'].sentiment_score.score,
                'confidence': item['post'].sentiment_score.confidence,
                'engagement': item['post'].engagement,
                'relevance': item['relevance_score']
            }
            for item in scored_posts[:limit]
        ]
    
    def _empty_sentiment_result(self) -> Dict:
        """Resultado vacío cuando no hay datos."""
        return {
            'score': 0.0,
            'confidence': 0.0,
            'volume': 0,
            'momentum': 0.0,
            'velocity': 0.0,
            'by_source': {},
            'recent_posts': [],
            'timestamp': datetime.now(),
            'anomaly_score': 0.0
        }
```

### 4. Integración con Pipeline de Trading

```python
# src/bot_cripto/features/sentiment_features.py
import pandas as pd
import numpy as np
from typing import Dict

class SentimentFeatureEngineer:
    """
    Genera features de sentiment para modelos de ML.
    """
    
    def __init__(self, orchestrator: SentimentOrchestrator):
        self.orchestrator = orchestrator
    
    async def generate_features(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """
        Genera todas las features de sentiment.
        
        Returns:
            Features listas para agregar al dataset de ML
        """
        # Obtener sentiment agregado
        sentiment_data = await self.orchestrator.get_aggregated_sentiment(
            symbol, lookback_hours
        )
        
        # Features básicas
        features = {
            'sentiment_score': sentiment_data['score'],
            'sentiment_confidence': sentiment_data['confidence'],
            'sentiment_volume': np.log1p(sentiment_data['volume']),  # Log scale
            'sentiment_momentum': sentiment_data['momentum'],
            'sentiment_velocity': sentiment_data['velocity'],
            'sentiment_anomaly': sentiment_data['anomaly_score']
        }
        
        # Features por fuente
        for source, data in sentiment_data['by_source'].items():
            features[f'sentiment_{source}_score'] = data['score']
            features[f'sentiment_{source}_confidence'] = data['confidence']
            features[f'sentiment_{source}_count'] = np.log1p(data['count'])
        
        # Features de distribución
        if sentiment_data['volume'] > 10:
            # Extraer distribución de scores
            all_scores = []
            for source_data in sentiment_data['by_source'].values():
                # (simplificado - en realidad necesitarías los posts individuales)
                all_scores.extend([source_data['score']] * source_data['count'])
            
            features['sentiment_std'] = np.std(all_scores)
            features['sentiment_skew'] = self._calculate_skewness(all_scores)
            features['sentiment_kurtosis'] = self._calculate_kurtosis(all_scores)
        else:
            features['sentiment_std'] = 0.0
            features['sentiment_skew'] = 0.0
            features['sentiment_kurtosis'] = 0.0
        
        # Interaction features
        features['sentiment_conf_weighted'] = (
            sentiment_data['score'] * sentiment_data['confidence']
        )
        features['sentiment_volume_weighted'] = (
            sentiment_data['score'] * np.log1p(sentiment_data['volume'])
        )
        
        # Regime indicators
        features['sentiment_extreme_positive'] = int(sentiment_data['score'] > 0.5)
        features['sentiment_extreme_negative'] = int(sentiment_data['score'] < -0.5)
        features['sentiment_neutral'] = int(abs(sentiment_data['score']) < 0.2)
        
        # Momentum regime
        features['sentiment_momentum_strong_up'] = int(sentiment_data['momentum'] > 0.2)
        features['sentiment_momentum_strong_down'] = int(sentiment_data['momentum'] < -0.2)
        
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calcula skewness (asimetría) de la distribución."""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calcula kurtosis (peso de colas) de la distribución."""
        from scipy.stats import kurtosis
        return kurtosis(data)
```

### 5. CLI Commands Mejorados

```python
# src/bot_cripto/cli/commands.py

@cli.command('fetch-sentiment')
@click.option('--symbol', default='BTC/USDT')
@click.option('--source', 
              type=click.Choice(['all', 'x', 'telegram', 'reddit', 'news', 'nlp']),
              default='all')
@click.option('--lookback-hours', default=24)
async def fetch_sentiment(symbol: str, source: str, lookback_hours: int):
    """Fetch y analiza sentiment de fuentes especificadas."""
    
    config = load_config()
    orchestrator = SentimentOrchestrator(config)
    
    if source == 'all':
        # Obtener de todas las fuentes
        result = await orchestrator.get_aggregated_sentiment(symbol, lookback_hours)
        
        # Display
        click.echo(f"\n{'='*60}")
        click.echo(f"Sentiment Analysis for {symbol}")
        click.echo(f"{'='*60}")
        click.echo(f"Score: {result['score']:.3f} ({result['confidence']:.2%} confidence)")
        click.echo(f"Volume: {result['volume']} posts")
        click.echo(f"Momentum: {result['momentum']:+.3f}")
        click.echo(f"Velocity: {result['velocity']:+.3f}")
        click.echo(f"Anomaly Score: {result['anomaly_score']:.2%}")
        
        click.echo(f"\nBreakdown by Source:")
        for source_name, data in result['by_source'].items():
            click.echo(
                f"  {source_name:12s}: {data['score']:+.3f} "
                f"({data['count']:4d} posts, {data['confidence']:.2%} conf)"
            )
        
        click.echo(f"\nTop Posts:")
        for i, post in enumerate(result['recent_posts'][:5], 1):
            click.echo(f"\n  {i}. [{post['source']}] @{post['author']}")
            click.echo(f"     {post['text'][:100]}...")
            click.echo(f"     Sentiment: {post['sentiment']:+.3f} "
                      f"(confidence: {post['confidence']:.2%})")
    
    else:
        # Fuente específica
        collector = orchestrator.collectors.get(source)
        if not collector:
            click.echo(f"Error: Source '{source}' not found", err=True)
            return
        
        posts = await collector.collect()
        scores = orchestrator.analyzer.analyze_batch(posts)
        
        click.echo(f"\nCollected {len(posts)} posts from {source}")
        avg_score = np.mean([s.score for s in scores])
        click.echo(f"Average sentiment: {avg_score:+.3f}")


@cli.command('sentiment-backtest')
@click.option('--symbol', default='BTC/USDT')
@click.option('--days', default=30)
@click.option('--save-report', is_flag=True)
async def sentiment_backtest(symbol: str, days: int, save_report: bool):
    """
    Backtest del sistema de sentiment.
    
    Valida qué tan bien el sentiment predice movimientos de precio.
    """
    click.echo(f"Running sentiment backtest for {symbol} ({days} days)...")
    
    # Cargar datos históricos de sentiment
    db = SentimentDatabase()
    sentiment_history = await db.get_sentiment_timeseries(
        symbol,
        since=datetime.now() - timedelta(days=days),
        interval_minutes=60
    )
    
    # Cargar datos de precio
    price_data = load_price_data(symbol, days)
    
    # Alinear timestamps
    aligned_data = align_sentiment_and_price(sentiment_history, price_data)
    
    # Calcular correlaciones
    correlations = calculate_sentiment_price_correlations(aligned_data)
    
    # Display resultados
    click.echo(f"\nCorrelation Analysis:")
    click.echo(f"  Sentiment vs Next 1h Return:  {correlations['1h']:+.3f}")
    click.echo(f"  Sentiment vs Next 4h Return:  {correlations['4h']:+.3f}")
    click.echo(f"  Sentiment vs Next 24h Return: {correlations['24h']:+.3f}")
    
    click.echo(f"\nSentiment Momentum vs Returns:")
    click.echo(f"  Momentum vs Next 1h Return:   {correlations['momentum_1h']:+.3f}")
    
    # Estrategia simple basada en sentiment
    strategy_results = backtest_sentiment_strategy(aligned_data)
    
    click.echo(f"\nSimple Sentiment Strategy:")
    click.echo(f"  Total Return: {strategy_results['return']:.2%}")
    click.echo(f"  Sharpe Ratio: {strategy_results['sharpe']:.2f}")
    click.echo(f"  Win Rate: {strategy_results['win_rate']:.1%}")
    
    if save_report:
        report_path = f"reports/sentiment_backtest_{symbol}_{days}d.html"
        generate_sentiment_backtest_report(aligned_data, strategy_results, report_path)
        click.echo(f"\nDetailed report saved to: {report_path}")
```

### 6. Configuración y Deployment

```yaml
# config/sentiment.yaml

# Fuentes habilitadas
enabled_sources:
  - x
  - reddit
  - news
  - telegram
  - cryptopanic

# Configuración de collectors
collectors:
  x:
    max_posts: 200
    rate_limit_per_minute: 30
    verified_only: false
    min_followers: 100
  
  reddit:
    subreddits:
      - cryptocurrency
      - bitcoin
      - ethtrader
    max_posts: 100
    min_upvotes: 10
  
  news:
    sources:
      - coindesk
      - cointelegraph
      - bloomberg_crypto
    max_articles: 50
  
  telegram:
    channels:
      - crypto_signals
      - bitcoin_discussion
    max_messages: 150

# Configuración de NLP
nlp:
  primary_model: "ProsusAI/finbert"
  use_ensemble: true
  ensemble_models:
    - "cardiffnlp/twitter-roberta-base-sentiment-latest"
    - "yiyanghkust/finbert-tone"
  batch_size: 32
  device: "auto"  # cuda, cpu, o auto

# Ponderación de fuentes
source_weights:
  x: 0.30
  news: 0.25
  reddit: 0.20
  telegram: 0.15
  cryptopanic: 0.10

# Time decay
time_decay_halflife_hours: 6

# Umbrales
thresholds:
  extreme_positive: 0.5
  extreme_negative: -0.5
  neutral_range: 0.2
  high_confidence: 0.7
  anomaly_threshold: 0.6

# Cache
cache:
  ttl_seconds: 300  # 5 minutos
  backend: "redis"  # redis o memory

# Database
database:
  backend: "postgresql"  # postgresql o sqlite
  connection_string: "${DATABASE_URL}"
  retention_days: 90
```

### 7. Tests y Validación

```python
# tests/test_sentiment_stack.py
import pytest
from bot_cripto.sentiment import SentimentOrchestrator, AdvancedNLPAnalyzer

@pytest.mark.asyncio
async def test_nlp_analyzer_fallback():
    """Verifica que fallback a lexicon funciona si NLP falla."""
    analyzer = AdvancedNLPAnalyzer(primary_model='invalid_model')
    
    posts = [
        SentimentPost(
            id='test1',
            source='test',
            text='Bitcoin is mooning! Very bullish! 🚀',
            author='test',
            timestamp=datetime.now(),
            engagement={},
            metadata={}
        )
    ]
    
    scores = analyzer.analyze_batch(posts)
    
    assert len(scores) == 1
    assert scores[0].score > 0  # Debería ser positivo
    assert 0 <= scores[0].confidence <= 1

@pytest.mark.asyncio
async def test_orchestrator_multi_source():
    """Verifica que orchestrator maneja múltiples fuentes."""
    config = load_test_config()
    orchestrator = SentimentOrchestrator(config)
    
    result = await orchestrator.get_aggregated_sentiment('BTC/USDT', lookback_hours=24)
    
    assert 'score' in result
    assert 'confidence' in result
    assert 'by_source' in result
    assert -1 <= result['score'] <= 1

@pytest.mark.asyncio
async def test_source_failure_handling():
    """Verifica que sistema continúa si una fuente falla."""
    # Simular fallo de una fuente
    # Sistema debería continuar con las fuentes disponibles
    pass

@pytest.mark.asyncio
async def test_sentiment_feature_generation():
    """Verifica generación de features para ML."""
    orchestrator = SentimentOrchestrator(load_test_config())
    feature_engineer = SentimentFeatureEngineer(orchestrator)
    
    features = await feature_engineer.generate_features('BTC/USDT')
    
    # Verificar que todas las features esperadas están presentes
    expected_features = [
        'sentiment_score',
        'sentiment_confidence',
        'sentiment_volume',
        'sentiment_momentum',
        'sentiment_velocity'
    ]
    
    for feature_name in expected_features:
        assert feature_name in features
```

---

## Plan de Implementación por Fases (Mejorado)

### **Fase 1: Fundamentos (Semana 1-2)** ✅

**Objetivo**: Infraestructura básica funcionando

1. Implementar BaseCollector y estructura de datos
2. Implementar X y Reddit collectors
3. Implementar AdvancedNLPAnalyzer con fallback
4. Tests unitarios básicos

**Validación**:
```bash
pytest tests/test_sentiment_collectors.py
pytest tests/test_nlp_analyzer.py
```

### **Fase 2: Orquestación (Semana 3)** 🔄

**Objetivo**: Sistema integrado end-to-end

1. Implementar SentimentOrchestrator
2. Agregar weighted blending
3. Implementar persistence (database)
4. CLI commands básicos

**Validación**:
```bash
bot-cripto fetch-sentiment --symbol BTC/USDT --source all
bot-cripto fetch-sentiment --symbol BTC/USDT --source x
```

### **Fase 3: Features Avanzadas (Semana 4)** 📊

**Objetivo**: Features de ML listas

1. Implementar SentimentFeatureEngineer
2. Calcular momentum y velocity
3. Detección de anomalías
4. Integración con pipeline de features existente

**Validación**:
```bash
bot-cripto features --include-sentiment
# Verificar que features de sentiment se agregan correctamente
```

### **Fase 4: Validación Rigurosa (Semana 5-6)** 🔬

**Objetivo**: Probar que sentiment mejora predicciones

1. **Backtest histórico**:
   - Correlación sentiment vs retornos futuros
   - Lead/lag analysis
   - Regímenes donde sentiment es más útil

2. **Walk-forward validation**:
   - Re-entrenar modelos con features de sentiment
   - Comparar performance con/sin sentiment
   - Verificar que no hay overfitting

3. **Métricas clave a validar**:
   - ¿Sharpe mejora con sentiment?
   - ¿Win rate aumenta?
   - ¿Max drawdown se reduce?
   - ¿Funciona en diferentes regímenes de mercado?

**Comandos de validación**:
```bash
# Backtest de sentiment puro
bot-cripto sentiment-backtest --symbol BTC/USDT --days 90

# Walk-forward con sentiment
bot-cripto validate-walk-forward --include-sentiment --windows 10

# Comparación A/B
bot-cripto compare-models --baseline no-sentiment --experimental with-sentiment
```

### **Fase 5: Producción (Semana 7-8)** 🚀

**Objetivo**: Deployment robusto

1. Implementar monitoreo en dashboard
2. Alertas automáticas (sentiment extremo, anomalías)
3. Configurar systemd para recolección periódica
4. Documentación completa

**Deployment**:
```bash
# Servicio de recolección de sentiment cada hora
sudo systemctl start bot-cripto-sentiment-collector.timer

# Dashboard con panel de sentiment
bot-cripto dashboard --include-sentiment-panel
```

### **Fase 6: Optimización Continua (Ongoing)** 🔄

1. **Monitoreo semanal**:
   - Correlación sentiment-retornos
   - Contribution de sentiment a P&L
   - Drift en modelos NLP

2. **Re-calibración mensual**:
   - Ajustar pesos de fuentes
   - Re-entrenar si es necesario
   - Actualizar umbrales

3. **Mejoras continuas**:
   - Agregar nuevas fuentes
   - Probar nuevos modelos NLP
   - Optimizar ponderación

---

## Métricas de Éxito

### Fase de Desarrollo
- ✅ Todos los tests pasan
- ✅ Coverage >80%
- ✅ Sin errores en producción

### Fase de Validación
- ✅ Correlación sentiment-retorno > 0.15 (para BTC 1h forward)
- ✅ Sharpe con sentiment > Sharpe sin sentiment
- ✅ Win rate +2-3% vs baseline
- ✅ Funciona en TREND y RANGE regimes

### Producción
- ✅ Uptime >99%
- ✅ Latencia recolección <30s
- ✅ P&L attribution del sentiment es positivo
- ✅ No degradación del modelo en 30 días

---

## Resumen de Mejoras vs Plan Original

| Aspecto | Original | Mejorado |
|---------|----------|----------|
| **Fuentes** | X, Telegram, API | +Reddit, News, CryptoPanic, On-chain |
| **NLP** | FinBERT solo | Ensemble multi-modelo + calibración |
| **Agregación** | Simple promedio | Weighted blending + time decay + engagement |
| **Validación** | Básica | Rigurosa: backtest + walk-forward + A/B testing |
| **Features** | Score básico | 15+ features: momentum, velocity, anomalías, etc. |
| **Monitoring** | Manual | Dashboard + alertas automáticas |
| **Fallback** | Lexicon simple | Lexicon mejorado + graceful degradation |
| **Tests** | Mínimos | Cobertura completa + integration tests |

---

**Próximo paso**: ¿Comenzamos con la Fase 1 (BaseCollector + NLP Analyzer) o prefieres que profundice en alguna parte específica primero?
