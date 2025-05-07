# Глава 50: Трансформеры с расширенной памятью для трейдинга

Эта глава исследует **Трансформеры с расширенной памятью** (Memory-Augmented Transformers) — класс архитектур трансформеров, которые дополняют стандартные механизмы внимания внешней памятью. Эти модели могут хранить и извлекать долгосрочные паттерны из исторических данных, что делает их особенно эффективными для прогнозирования финансовых временных рядов, где рыночные режимы и паттерны могут повторяться на длительных временных интервалах.

<p align="center">
<img src="https://i.imgur.com/8YqKvPz.png" width="70%" alt="Диаграмма архитектуры трансформера с расширенной памятью: входная последовательность, внешний банк памяти, слои энкодера с локальным вниманием и kNN извлечением из памяти, механизм гейтинга и голова предсказания">
</p>

## Содержание

1. [Введение в трансформеры с расширенной памятью](#введение-в-трансформеры-с-расширенной-памятью)
    * [Зачем нужна внешняя память?](#зачем-нужна-внешняя-память)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение со стандартными трансформерами](#сравнение-со-стандартными-трансформерами)
2. [Обзор архитектуры](#обзор-архитектуры)
    * [Внешний банк памяти](#внешний-банк-памяти)
    * [kNN извлечение из памяти](#knn-извлечение-из-памяти)
    * [Внимание с использованием памяти](#внимание-с-использованием-памяти)
    * [Интеграция с локальным вниманием](#интеграция-с-локальным-вниманием)
3. [Финансовые применения](#финансовые-применения)
    * [Распознавание долгосрочных паттернов](#распознавание-долгосрочных-паттернов)
    * [Определение рыночного режима](#определение-рыночного-режима)
    * [Торговля на основе исторического сходства](#торговля-на-основе-исторического-сходства)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Создание банка памяти](#02-создание-банка-памяти)
    * [03: Архитектура модели](#03-архитектура-модели)
    * [04: Пайплайн обучения](#04-пайплайн-обучения)
    * [05: Бэктестинг стратегии](#05-бэктестинг-стратегии)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в трансформеры с расширенной памятью

Трансформеры с расширенной памятью расширяют стандартную архитектуру трансформера, добавляя внешний банк памяти, который может хранить представления из гораздо более длинных контекстов, чем позволяют типичные механизмы внимания. В отличие от рекуррентных сетей, которые сжимают историю в скрытое состояние фиксированного размера, модели с расширенной памятью хранят явные пары (ключ, значение), которые могут быть извлечены во время вывода.

### Зачем нужна внешняя память?

Стандартные трансформеры имеют фундаментальное ограничение: их механизм внимания имеет сложность O(L²), где L — длина последовательности. Это делает вычислительно затратным прямое внимание на очень длинные последовательности.

**Проблема:**
```
Контекст традиционного трансформера:
[------ 512 токенов ------]  ← Ограниченное окно

Но финансовые паттерны могут охватывать:
[------ 50,000+ исторических точек данных ------]
        ↑ Бычьи рынки, обвалы, смены режимов
```

**Решение:**
```
Трансформер с расширенной памятью:
Локальный контекст: [--- 512 токенов ---] + Внешняя память: [100,000+ пар (ключ, значение)]
                           ↓                              ↓
                    Быстрое внимание             kNN извлечение
                           ↓                              ↓
                           └──────────┬───────────────────┘
                                      ↓
                           Комбинированное предсказание
```

### Ключевые преимущества

1. **Огромное контекстное окно**
   - Хранение 262K+ токенов во внешней памяти
   - Извлечение релевантных исторических паттернов за O(log N)
   - Без потока градиентов через память (масштабируемость)

2. **Точное извлечение**
   - В отличие от усреднения внимания, kNN извлекает точные исторические представления
   - Особенно полезно для редких, но важных рыночных событий
   - "Когда мы видели этот паттерн раньше?"

3. **Обучение во время вывода**
   - Можно добавлять новые знания просто добавляя в память
   - Не требуется переобучение для новых рыночных режимов
   - Непрерывная адаптация к изменяющимся рынкам

4. **Интерпретируемые решения**
   - Веса внимания показывают, какие исторические моменты влияют на предсказания
   - "Это похоже на март 2020" можно объяснить
   - Полезно для управления рисками и комплаенса

### Сравнение со стандартными трансформерами

| Характеристика | Стандартный трансформер | С расширенной памятью |
|----------------|------------------------|----------------------|
| Длина контекста | 512-4096 токенов | 100K+ токенов |
| Сложность | O(L²) | O(L² + k·log(M)) |
| Доступ к истории | Ограничен окном | Неограничен (размер памяти) |
| Сопоставление паттернов | Неявно в весах | Явно через извлечение |
| Адаптация | Требует переобучения | Просто добавить в память |
| Редкие события | Могут быть забыты | Явно сохранены |

## Обзор архитектуры

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ТРАНСФОРМЕР С РАСШИРЕННОЙ ПАМЯТЬЮ                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Входная последовательность          Внешний банк памяти                     │
│  [x₁, x₂, ..., xₙ]                   [m₁, m₂, ..., mₘ]  (M >> N)             │
│        │                                     │                                │
│        ▼                                     │                                │
│  ┌─────────────────┐                        │                                │
│  │ Токен-эмбеддинг │                        │                                │
│  │   + Позиция     │                        │                                │
│  └────────┬────────┘                        │                                │
│           │                                  │                                │
│           ▼                                  │                                │
│  ┌────────────────────────────────────────────────────────────┐              │
│  │                    СЛОЙ ЭНКОДЕРА (×N)                       │              │
│  │  ┌──────────────────────┐    ┌─────────────────────────┐  │              │
│  │  │  Локальное внимание  │    │   Извлечение из памяти  │  │              │
│  │  │   (Стандартное)      │    │   (kNN поиск)           │◄─┼──────────────┤
│  │  │   Q·K^T / √d         │    │   top-k соседей         │  │              │
│  │  └──────────┬───────────┘    └───────────┬─────────────┘  │              │
│  │             │                            │                 │              │
│  │             └───────────┬────────────────┘                 │              │
│  │                         │                                  │              │
│  │                    ┌────▼────┐                             │              │
│  │                    │  Гейт   │                             │              │
│  │                    │ α·local + (1-α)·memory               │              │
│  │                    └────┬────┘                             │              │
│  │                         │                                  │              │
│  │                    ┌────▼────┐                             │              │
│  │                    │   FFN   │                             │              │
│  │                    └────┬────┘                             │              │
│  └─────────────────────────┼──────────────────────────────────┘              │
│                            │                                                  │
│                            │  Сохранение новых (ключ, значение)              │
│                            ├──────────────────────────────────►  Обновление  │
│                            │                                     памяти      │
│                            ▼                                                  │
│                   ┌────────────────┐                                         │
│                   │ Голова предсказ.│                                         │
│                   │ (Цена/Сигнал)   │                                         │
│                   └────────────────┘                                         │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Внешний банк памяти

Банк памяти хранит пары (ключ, значение) из исторических данных:

```python
class ExternalMemory:
    """
    Внешний банк памяти с приближённым поиском ближайших соседей.

    Хранит исторические представления для эффективного извлечения.
    """

    def __init__(self, memory_size: int, dim: int, n_neighbors: int = 32):
        self.memory_size = memory_size
        self.dim = dim
        self.n_neighbors = n_neighbors

        # Хранилище для ключей и значений
        self.keys = np.zeros((memory_size, dim), dtype=np.float32)
        self.values = np.zeros((memory_size, dim), dtype=np.float32)

        # Отслеживание заполненности памяти
        self.current_size = 0

        # Индекс FAISS для быстрого kNN поиска
        self.index = faiss.IndexFlatIP(dim)  # Скалярное произведение (косинусное сходство)

    def add(self, keys: np.ndarray, values: np.ndarray):
        """Добавление новых пар (ключ, значение) в память"""
        n_new = keys.shape[0]

        if self.current_size + n_new > self.memory_size:
            # FIFO: удаление старейших записей
            self._remove_oldest(n_new)

        # Добавление в хранилище
        start = self.current_size
        self.keys[start:start+n_new] = keys
        self.values[start:start+n_new] = values
        self.current_size += n_new

        # Обновление индекса
        self.index.add(keys)

    def search(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск k ближайших соседей для каждого запроса.

        Возвращает:
            distances: [n_queries, k]
            indices: [n_queries, k]
        """
        distances, indices = self.index.search(queries, self.n_neighbors)
        retrieved_values = self.values[indices]
        return distances, retrieved_values
```

**Ключевые проектные решения:**
- **Без потока градиентов**: Память недифференцируема — градиенты не проходят через извлечение
- **FIFO обновления**: Старейшие воспоминания заменяются при заполнении памяти
- **Приближённый поиск**: Использование FAISS или ScaNN для извлечения за O(log M)

### kNN извлечение из памяти

Механизм извлечения находит похожие исторические моменты:

```python
class KNNMemoryAttention(nn.Module):
    """
    Слой внимания на основе kNN памяти.

    Извлекает релевантные исторические представления и
    комбинирует их с локальным контекстом через внимание.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_neighbors = config.n_neighbors
        self.temperature = config.temperature

        # Проекции для запросов и извлечённых значений
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: ExternalMemory
    ) -> torch.Tensor:
        """
        Аргументы:
            x: Текущие представления [batch, seq_len, d_model]
            memory: Внешний банк памяти

        Возвращает:
            Представления с расширенной памятью [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Проекция запросов
        queries = self.query_proj(x)  # [batch, seq_len, d_model]

        # Извлечение из памяти
        queries_np = queries.detach().cpu().numpy().reshape(-1, d_model)
        distances, retrieved = memory.search(queries_np)

        # Конвертация обратно в тензоры
        retrieved = torch.from_numpy(retrieved).to(x.device)
        retrieved = retrieved.view(batch, seq_len, self.n_neighbors, d_model)
        distances = torch.from_numpy(distances).to(x.device)
        distances = distances.view(batch, seq_len, self.n_neighbors)

        # Проекция извлечённых значений
        retrieved_v = self.value_proj(retrieved)

        # Внимание над извлечёнными соседями
        # distances - это сходства (скалярные произведения), используем как веса внимания
        attn_weights = F.softmax(distances / self.temperature, dim=-1)

        # Взвешенная сумма извлечённых значений
        context = torch.einsum('bsnk,bsnd->bsd', attn_weights.unsqueeze(-1), retrieved_v)

        return self.out_proj(context)
```

### Внимание с использованием памяти

Комбинирование локального и основанного на памяти внимания:

```python
class MemoryAugmentedAttention(nn.Module):
    """
    Комбинирует стандартное самовнимание с kNN извлечением из памяти.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()

        # Локальное самовнимание
        self.local_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Извлечение из памяти
        self.memory_attention = KNNMemoryAttention(config)

        # Механизм гейтирования
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: ExternalMemory,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Комбинирование локального внимания с извлечением из памяти.

        Гейт учится, когда полагаться на локальный контекст,
        а когда — на исторические паттерны.
        """
        # Локальное внимание
        local_out, local_attn = self.local_attention(x, x, x)

        # Извлечение из памяти
        memory_out = self.memory_attention(x, memory)

        # Гейтированная комбинация
        gate_input = torch.cat([local_out, memory_out], dim=-1)
        gate = self.gate(gate_input)

        output = gate * local_out + (1 - gate) * memory_out

        if return_attention:
            return output, local_attn
        return output, None
```

### Интеграция с локальным вниманием

```
Локальное внимание (Недавние данные):
┌──────────────────────────────────────────────┐
│ Сегодня ← Вчера ← 2 дня назад ← ... ← 7 дней │
│  x₁     ←   x₂  ←    x₃       ← ... ←   x₇   │
│                   ↓                          │
│           Плотная матрица внимания           │
│           (Все пары взаимодействуют)         │
└──────────────────────────────────────────────┘

Извлечение из памяти (Исторические данные):
┌──────────────────────────────────────────────┐
│        Запрос: "Текущий рынок похож на..."   │
│                      ↓                       │
│              kNN поиск в памяти              │
│                      ↓                       │
│   Извлечено: [кризис 2008, кризис 2020, ...] │
│                      ↓                       │
│        Внимание над извлечёнными моментами   │
└──────────────────────────────────────────────┘

Комбинированный выход:
┌──────────────────────────────────────────────┐
│    α · локальный_контекст + (1-α) · память   │
│                      ↓                       │
│            Финальное представление           │
└──────────────────────────────────────────────┘
```

## Финансовые применения

### Распознавание долгосрочных паттернов

Трансформеры с расширенной памятью превосходно распознают паттерны, охватывающие длительные периоды:

```python
# Пример: Определение рыночных режимов через сравнение с историческими паттернами

def detect_regime(model, current_data, memory):
    """
    Использование извлечения из памяти для идентификации текущего рыночного режима.

    Возвращает:
        regime: Предсказанный режим (бычий/медвежий/боковой)
        similar_periods: Исторические периоды с похожими паттернами
    """
    # Кодирование текущего состояния рынка
    encoded = model.encode(current_data)

    # Извлечение похожих исторических моментов
    distances, retrieved_indices = memory.search(encoded[-1:])

    # Анализ извлечённых периодов
    similar_periods = []
    for idx in retrieved_indices[0]:
        period_info = {
            'date': memory.metadata[idx]['date'],
            'regime': memory.metadata[idx]['regime'],
            'subsequent_return': memory.metadata[idx]['future_30d_return']
        }
        similar_periods.append(period_info)

    # Голосование по текущему режиму на основе похожих периодов
    regime_votes = Counter([p['regime'] for p in similar_periods])
    predicted_regime = regime_votes.most_common(1)[0][0]

    return predicted_regime, similar_periods
```

### Определение рыночного режима

```python
# Хранение представлений с метками режимов в памяти

class RegimeAwareMemory(ExternalMemory):
    """
    Память, которая отслеживает рыночные режимы для извлечённых паттернов.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}

    def add_with_metadata(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        dates: List[str],
        regimes: List[str],
        returns: List[float]
    ):
        """Добавление записей с ассоциированными метаданными"""
        start_idx = self.current_size
        self.add(keys, values)

        for i, (date, regime, ret) in enumerate(zip(dates, regimes, returns)):
            self.metadata[start_idx + i] = {
                'date': date,
                'regime': regime,
                'future_30d_return': ret
            }

    def get_regime_distribution(self, indices: np.ndarray) -> Dict[str, float]:
        """Получение распределения режимов для извлечённых индексов"""
        regimes = [self.metadata[i]['regime'] for i in indices.flatten()]
        counts = Counter(regimes)
        total = len(regimes)
        return {r: c/total for r, c in counts.items()}
```

### Торговля на основе исторического сходства

```python
class HistoricalSimilarityStrategy:
    """
    Торговая стратегия на основе сопоставления исторических паттернов.

    Логика: "Если текущий рынок похож на X, и за X последовало Y,
    то позиционируемся на повторение Y."
    """

    def __init__(
        self,
        model: MemoryTransformer,
        memory: RegimeAwareMemory,
        n_similar: int = 10,
        confidence_threshold: float = 0.7
    ):
        self.model = model
        self.memory = memory
        self.n_similar = n_similar
        self.confidence_threshold = confidence_threshold

    def generate_signal(self, current_data: torch.Tensor) -> Dict:
        """
        Генерация торгового сигнала на основе исторического сходства.
        """
        # Кодирование текущего состояния
        with torch.no_grad():
            encoded = self.model.encode(current_data)

        # Поиск похожих исторических моментов
        distances, indices = self.memory.search(
            encoded[-1:].numpy(),
            k=self.n_similar
        )

        # Анализ того, что произошло после похожих моментов
        future_returns = [
            self.memory.metadata[i]['future_30d_return']
            for i in indices[0]
        ]

        # Расчёт ожидаемой доходности и уверенности
        avg_return = np.mean(future_returns)
        return_std = np.std(future_returns)
        positive_ratio = np.mean([r > 0 for r in future_returns])

        # Генерация сигнала
        if positive_ratio > self.confidence_threshold:
            signal = 'LONG'
            confidence = positive_ratio
        elif positive_ratio < (1 - self.confidence_threshold):
            signal = 'SHORT'
            confidence = 1 - positive_ratio
        else:
            signal = 'HOLD'
            confidence = 0.5

        return {
            'signal': signal,
            'confidence': confidence,
            'expected_return': avg_return,
            'return_std': return_std,
            'similar_dates': [self.memory.metadata[i]['date'] for i in indices[0]]
        }
```

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
import yfinance as yf
from pybit.unified_trading import HTTP
from typing import List, Dict, Tuple

def load_stock_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1h'
) -> Dict[str, pd.DataFrame]:
    """
    Загрузка данных акций из yfinance.

    Аргументы:
        symbols: Список тикеров акций (например, ['AAPL', 'MSFT'])
        start_date: Дата начала в формате 'YYYY-MM-DD'
        end_date: Дата окончания в формате 'YYYY-MM-DD'
        interval: Частота данных ('1h', '1d' и т.д.)

    Возвращает:
        Словарь, сопоставляющий символы с DataFrame
    """
    data = {}

    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Добавление признаков
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()

        data[symbol] = df.dropna()

    return data


def load_bybit_data(
    symbols: List[str],
    interval: str = '60',  # 60 минут
    limit: int = 1000
) -> Dict[str, pd.DataFrame]:
    """
    Загрузка данных криптовалют из Bybit.

    Аргументы:
        symbols: Список торговых пар (например, ['BTCUSDT', 'ETHUSDT'])
        interval: Интервал свечей в минутах
        limit: Количество свечей для загрузки

    Возвращает:
        Словарь, сопоставляющий символы с DataFrame
    """
    client = HTTP(testnet=False)
    data = {}

    for symbol in symbols:
        response = client.get_kline(
            category='linear',
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        if response['retCode'] == 0:
            df = pd.DataFrame(response['result']['list'])
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

            # Конвертация типов
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

            # Добавление признаков
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

            data[symbol] = df.dropna().sort_values('timestamp')

    return data


def create_sequences(
    data: pd.DataFrame,
    seq_len: int = 96,
    features: List[str] = ['returns', 'volatility', 'volume_change']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создание последовательностей для обучения.

    Аргументы:
        data: DataFrame с OHLCV и признаками
        seq_len: Длина последовательности
        features: Столбцы признаков для использования

    Возвращает:
        X: [n_samples, seq_len, n_features]
        y: [n_samples, 1] (доходность следующего периода)
    """
    X, y = [], []

    values = data[features].values
    returns = data['returns'].values

    for i in range(seq_len, len(data) - 1):
        X.append(values[i-seq_len:i])
        y.append(returns[i+1])  # Предсказываем следующую доходность

    return np.array(X), np.array(y)
```

### 02: Создание банка памяти

```python
# python/02_memory_bank.py

import numpy as np
import faiss
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MemoryConfig:
    """Конфигурация внешней памяти"""
    memory_size: int = 100000
    dim: int = 64
    n_neighbors: int = 32
    use_gpu: bool = True


class ExternalMemoryBank:
    """
    Внешний банк памяти с использованием FAISS для эффективного kNN поиска.

    Особенности:
    - FIFO замена при заполнении памяти
    - GPU ускорение при наличии
    - Хранение метаданных для интерпретируемости
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_size = config.memory_size
        self.dim = config.dim
        self.n_neighbors = config.n_neighbors

        # Хранилище
        self.keys = np.zeros((self.memory_size, self.dim), dtype=np.float32)
        self.values = np.zeros((self.memory_size, self.dim), dtype=np.float32)
        self.current_size = 0
        self.write_pos = 0

        # Метаданные для интерпретируемости
        self.timestamps = [None] * self.memory_size
        self.returns = np.zeros(self.memory_size, dtype=np.float32)

        # Индекс FAISS
        self._build_index(config.use_gpu)

    def _build_index(self, use_gpu: bool):
        """Построение индекса FAISS для быстрого поиска"""
        # Используем скалярное произведение (эквивалентно косинусному сходству для нормализованных векторов)
        self.index = faiss.IndexFlatIP(self.dim)

        if use_gpu and faiss.get_num_gpus() > 0:
            # Перенос на GPU
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.on_gpu = True
        else:
            self.on_gpu = False

    def add(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        timestamps: Optional[List] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Добавление записей в память.

        Аргументы:
            keys: [n, dim] векторы ключей
            values: [n, dim] векторы значений
            timestamps: Опциональный список временных меток
            returns: Опциональный массив будущих доходностей
        """
        n = keys.shape[0]

        # Нормализация ключей для косинусного сходства
        keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)

        for i in range(n):
            pos = self.write_pos % self.memory_size

            self.keys[pos] = keys[i]
            self.values[pos] = values[i]

            if timestamps is not None:
                self.timestamps[pos] = timestamps[i]
            if returns is not None:
                self.returns[pos] = returns[i]

            self.write_pos += 1
            self.current_size = min(self.current_size + 1, self.memory_size)

        # Перестроение индекса
        self.index.reset()
        self.index.add(self.keys[:self.current_size])

    def search(
        self,
        queries: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Поиск k ближайших соседей.

        Аргументы:
            queries: [n, dim] векторы запросов
            k: Количество соседей (по умолчанию: self.n_neighbors)

        Возвращает:
            distances: [n, k] оценки сходства
            indices: [n, k] индексы в памяти
            values: [n, k, dim] извлечённые значения
        """
        if k is None:
            k = self.n_neighbors

        # Нормализация запросов
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

        # Поиск
        distances, indices = self.index.search(queries.astype(np.float32), k)

        # Получение значений
        values = self.values[indices]

        return distances, indices, values

    def get_metadata(self, indices: np.ndarray) -> dict:
        """Получение метаданных для извлечённых индексов"""
        return {
            'timestamps': [[self.timestamps[i] for i in row] for row in indices],
            'returns': self.returns[indices]
        }
```

### 03: Архитектура модели

Полная реализация модели трансформера с расширенной памятью доступна в файле `python/model.py`.

### 04: Пайплайн обучения

Скрипт обучения с обновлением памяти доступен в файле `python/train.py`.

### 05: Бэктестинг стратегии

Утилиты для бэктестинга стратегий на основе памяти доступны в файле `python/backtest.py`.

## Реализация на Rust

Смотрите [rust_memory_transformer](rust_memory_transformer/) для полной реализации на Rust с данными Bybit.

```
rust_memory_transformer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Экспорты библиотеки
│   ├── api/                # Клиент API Bybit
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   └── types.rs        # Типы API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Загрузка данных
│   │   └── features.rs     # Инженерия признаков
│   ├── memory/             # Внешняя память
│   │   ├── mod.rs
│   │   ├── bank.rs         # Банк памяти
│   │   └── search.rs       # kNN поиск
│   ├── model/              # Архитектура модели
│   │   ├── mod.rs
│   │   ├── embedding.rs    # Токен-эмбеддинг
│   │   ├── attention.rs    # Внимание с памятью
│   │   └── transformer.rs  # Полная модель
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Бэктестинг
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust_memory_transformer

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Обучить модель
cargo run --example train -- --epochs 50 --memory-size 50000

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── model.py               # Трансформер с расширенной памятью
├── memory.py              # Внешний банк памяти
├── data.py                # Загрузка данных (yfinance, Bybit)
├── train.py               # Скрипт обучения
├── backtest.py            # Утилиты бэктестинга
├── requirements.txt       # Зависимости
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_memory_bank.ipynb
    ├── 03_model_training.ipynb
    ├── 04_historical_similarity.ipynb
    └── 05_backtesting.ipynb
```

### Быстрый старт (Python)

```bash
# Установить зависимости
pip install -r requirements.txt

# Загрузить данные
python data.py --symbols BTCUSDT,ETHUSDT,AAPL,MSFT

# Обучить модель
python train.py --epochs 100 --memory-size 100000

# Запустить бэктест
python backtest.py --model checkpoints/best.pt
```

## Лучшие практики

### Когда использовать трансформеры с расширенной памятью

**Подходящие случаи:**
- Распознавание долгосрочных паттернов (месяцы-годы)
- Определение рыночного режима и стратегии смены режимов
- Моделирование редких событий (обвалы, сквизы)
- Торговля на основе исторического сходства
- Непрерывное обучение без переобучения

**Не идеально для:**
- Ультра-высокочастотной торговли (задержка извлечения)
- Очень краткосрочных предсказаний (накладные расходы памяти не оправданы)
- Полностью новых рыночных условий (нет похожих воспоминаний)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендация | Примечания |
|----------|--------------|------------|
| `memory_size` | 50K-200K | Больше = больше истории, медленнее извлечение |
| `n_neighbors` | 16-64 | Больше соседей = более плавные предсказания |
| `d_model` | 64-128 | Согласовать с размерностью памяти |
| `temperature` | 0.5-2.0 | Ниже = резче внимание, выше = плавнее |
| `gate_bias` | 0.0 | Настроить, если модель слишком полагается на память |

### Управление памятью

1. **FIFO vs. Замена по важности**
   ```python
   # FIFO (простой, быстрый)
   # Старые записи заменяются первыми

   # На основе важности (лучше сохранение редких событий)
   # Записи с высоким градиентом или доходностью сохраняются дольше
   ```

2. **Прогрев памяти**
   ```python
   # Перед торговлей заполнить память историческими данными
   for historical_batch in historical_data:
       with torch.no_grad():
           hidden = model.encode(historical_batch)
           memory.add(hidden, returns)
   ```

3. **Очистка памяти**
   ```python
   # Периодическое удаление низкокачественных записей
   # Например, записей, которые никогда не извлекались
   ```

### Распространённые ошибки

1. **Устаревание памяти**: Старые паттерны могут быть нерелевантны
   - Решение: Использовать взвешенное по времени извлечение или затухание памяти

2. **Задержка извлечения**: kNN поиск может быть медленным для большой памяти
   - Решение: Использовать приближённый поиск (FAISS, ScaNN)

3. **Холодный старт**: Нет полезных воспоминаний в начале
   - Решение: Предварительно заполнить историческими данными перед живой торговлей

4. **Сдвиг распределения**: Рынок меняется, но старые воспоминания остаются
   - Решение: Использовать адаптивную замену памяти или память с учётом режимов

## Ресурсы

### Статьи

- [Memorizing Transformers](https://arxiv.org/abs/2203.08913) - Оригинальная статья о kNN памяти для трансформеров
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Оригинальная статья о трансформерах
- [RETRO](https://arxiv.org/abs/2112.04426) - Трансформер с расширенным извлечением
- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Расширенный контекст для трансформеров

### Реализации

- [memorizing-transformers-pytorch](https://github.com/lucidrains/memorizing-transformers-pytorch) - Реализация на PyTorch
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Масштабируемый поиск ближайших соседей от Google

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) - Мульти-горизонтное прогнозирование
- [Глава 28: Regime Detection HMM](../28_regime_detection_hmm) - Определение рыночного режима
- [Глава 49: Multi-Scale Attention](../49_multi_scale_attention) - Многомасштабное внимание

---

## Уровень сложности

**Продвинутый**

Предварительные требования:
- Архитектура трансформеров и механизмы внимания
- Алгоритмы поиска ближайших соседей
- Основы прогнозирования временных рядов
- Библиотеки ML для PyTorch или Rust
