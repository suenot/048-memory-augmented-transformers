# Chapter 50: Memory-Augmented Transformers for Trading

This chapter explores **Memory-Augmented Transformers**, a class of transformer architectures that augment standard attention mechanisms with external memory. These models can store and retrieve long-term patterns from historical data, making them particularly powerful for financial time series prediction where market regimes and patterns may repeat over extended periods.

<p align="center">
<img src="https://i.imgur.com/8YqKvPz.png" width="70%" alt="Memory-Augmented Transformer Architecture Diagram showing input sequence, external memory bank, encoder layers with local attention and kNN memory retrieval, gating mechanism, and prediction head">
</p>

## Contents

1. [Introduction to Memory-Augmented Transformers](#introduction-to-memory-augmented-transformers)
    * [Why External Memory?](#why-external-memory)
    * [Key Advantages](#key-advantages)
    * [Comparison with Standard Transformers](#comparison-with-standard-transformers)
2. [Architecture Overview](#architecture-overview)
    * [External Memory Bank](#external-memory-bank)
    * [kNN Memory Retrieval](#knn-memory-retrieval)
    * [Memory-Augmented Attention](#memory-augmented-attention)
    * [Integration with Local Attention](#integration-with-local-attention)
3. [Financial Applications](#financial-applications)
    * [Long-Term Pattern Recognition](#long-term-pattern-recognition)
    * [Market Regime Detection](#market-regime-detection)
    * [Historical Similarity Trading](#historical-similarity-trading)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Memory Bank Construction](#02-memory-bank-construction)
    * [03: Model Architecture](#03-model-architecture)
    * [04: Training Pipeline](#04-training-pipeline)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Memory-Augmented Transformers

Memory-Augmented Transformers extend the standard transformer architecture by adding an external memory bank that can store representations from much longer contexts than typical attention mechanisms allow. Unlike recurrent networks that compress history into a fixed-size hidden state, memory-augmented models store explicit (key, value) pairs that can be retrieved at inference time.

### Why External Memory?

Standard transformers have a fundamental limitation: their attention mechanism has O(L²) complexity, where L is the sequence length. This makes it computationally expensive to attend to very long sequences directly.

**The Problem:**
```
Traditional Transformer Context:
[------ 512 tokens ------]  ← Limited window

But financial patterns may span:
[------ 50,000+ historical data points ------]
        ↑ Bull markets, crashes, regime changes
```

**The Solution:**
```
Memory-Augmented Transformer:
Local Context: [--- 512 tokens ---] + External Memory: [100,000+ (key,value) pairs]
                     ↓                         ↓
              Fast attention            kNN retrieval
                     ↓                         ↓
                     └──────────┬──────────────┘
                                ↓
                      Combined prediction
```

### Key Advantages

1. **Massive Context Window**
   - Store 262K+ tokens in external memory
   - Retrieve relevant historical patterns in O(log N) time
   - No gradient flow through memory (scalable)

2. **Exact Retrieval**
   - Unlike attention averaging, kNN retrieves exact historical representations
   - Particularly useful for rare but important market events
   - "When have we seen this pattern before?"

3. **Inference-Time Learning**
   - Can add new knowledge by simply appending to memory
   - No retraining required for new market regimes
   - Continuous adaptation to changing markets

4. **Interpretable Decisions**
   - Attention weights show which historical moments influence predictions
   - "This looks like March 2020" is explainable
   - Useful for risk management and compliance

### Comparison with Standard Transformers

| Feature | Standard Transformer | Memory-Augmented |
|---------|---------------------|------------------|
| Context length | 512-4096 tokens | 100K+ tokens |
| Complexity | O(L²) | O(L² + k·log(M)) |
| Historical access | Limited by window | Unlimited (memory size) |
| Pattern matching | Implicit in weights | Explicit via retrieval |
| Adaptation | Requires retraining | Just add to memory |
| Rare events | May be forgotten | Explicitly stored |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY-AUGMENTED TRANSFORMER                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Input Sequence                    External Memory Bank                       │
│  [x₁, x₂, ..., xₙ]                [m₁, m₂, ..., mₘ]  (M >> N)               │
│        │                                 │                                    │
│        ▼                                 │                                    │
│  ┌─────────────────┐                    │                                    │
│  │ Token Embedding │                    │                                    │
│  │   + Position    │                    │                                    │
│  └────────┬────────┘                    │                                    │
│           │                              │                                    │
│           ▼                              │                                    │
│  ┌────────────────────────────────────────────────────────────┐              │
│  │                    ENCODER LAYER (×N)                       │              │
│  │  ┌──────────────────────┐    ┌─────────────────────────┐  │              │
│  │  │   Local Attention    │    │   Memory Retrieval      │  │              │
│  │  │   (Standard)         │    │   (kNN Lookup)          │◄─┼──────────────┤
│  │  │   Q·K^T / √d         │    │   top-k neighbors       │  │              │
│  │  └──────────┬───────────┘    └───────────┬─────────────┘  │              │
│  │             │                            │                 │              │
│  │             └───────────┬────────────────┘                 │              │
│  │                         │                                  │              │
│  │                    ┌────▼────┐                             │              │
│  │                    │  Gate   │                             │              │
│  │                    │ α·local + (1-α)·memory               │              │
│  │                    └────┬────┘                             │              │
│  │                         │                                  │              │
│  │                    ┌────▼────┐                             │              │
│  │                    │   FFN   │                             │              │
│  │                    └────┬────┘                             │              │
│  └─────────────────────────┼──────────────────────────────────┘              │
│                            │                                                  │
│                            │  Store new (key, value)                         │
│                            ├──────────────────────────────────►  Memory      │
│                            │                                     Update      │
│                            ▼                                                  │
│                   ┌────────────────┐                                         │
│                   │ Prediction Head │                                         │
│                   │ (Price/Signal)  │                                         │
│                   └────────────────┘                                         │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### External Memory Bank

The memory bank stores (key, value) pairs from historical data:

```python
class ExternalMemory:
    """
    External memory bank with approximate nearest neighbor search.

    Stores historical representations for efficient retrieval.
    """

    def __init__(self, memory_size: int, dim: int, n_neighbors: int = 32):
        self.memory_size = memory_size
        self.dim = dim
        self.n_neighbors = n_neighbors

        # Storage for keys and values
        self.keys = np.zeros((memory_size, dim), dtype=np.float32)
        self.values = np.zeros((memory_size, dim), dtype=np.float32)

        # Track how much memory is filled
        self.current_size = 0

        # FAISS index for fast kNN search
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim)

    def add(self, keys: np.ndarray, values: np.ndarray):
        """Add new (key, value) pairs to memory"""
        n_new = keys.shape[0]

        if self.current_size + n_new > self.memory_size:
            # FIFO: Remove oldest entries
            self._remove_oldest(n_new)

        # Add to storage
        start = self.current_size
        self.keys[start:start+n_new] = keys
        self.values[start:start+n_new] = values
        self.current_size += n_new

        # Update index
        self.index.add(keys)

    def search(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k-nearest neighbors for each query.

        Returns:
            distances: [n_queries, k]
            indices: [n_queries, k]
        """
        distances, indices = self.index.search(queries, self.n_neighbors)
        retrieved_values = self.values[indices]
        return distances, retrieved_values
```

**Key Design Choices:**
- **No Gradient Flow**: Memory is not differentiable - gradients don't flow through retrieval
- **FIFO Updates**: Oldest memories are replaced when memory is full
- **Approximate Search**: Use FAISS or ScaNN for O(log M) retrieval

### kNN Memory Retrieval

The retrieval mechanism finds similar historical moments:

```python
class KNNMemoryAttention(nn.Module):
    """
    kNN-based memory attention layer.

    Retrieves relevant historical representations and
    combines them with local context using attention.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_neighbors = config.n_neighbors
        self.temperature = config.temperature

        # Projections for query and retrieved values
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
        Args:
            x: Current representations [batch, seq_len, d_model]
            memory: External memory bank

        Returns:
            Memory-augmented representations [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project queries
        queries = self.query_proj(x)  # [batch, seq_len, d_model]

        # Retrieve from memory
        queries_np = queries.detach().cpu().numpy().reshape(-1, d_model)
        distances, retrieved = memory.search(queries_np)

        # Convert back to tensors
        retrieved = torch.from_numpy(retrieved).to(x.device)
        retrieved = retrieved.view(batch, seq_len, self.n_neighbors, d_model)
        distances = torch.from_numpy(distances).to(x.device)
        distances = distances.view(batch, seq_len, self.n_neighbors)

        # Project retrieved values
        retrieved_v = self.value_proj(retrieved)

        # Attention over retrieved neighbors
        # distances are similarities (inner products), use as attention scores
        attn_weights = F.softmax(distances / self.temperature, dim=-1)

        # Weighted sum of retrieved values
        context = torch.einsum('bsnk,bsnd->bsd', attn_weights.unsqueeze(-1), retrieved_v)

        return self.out_proj(context)
```

### Memory-Augmented Attention

Combining local and memory-based attention:

```python
class MemoryAugmentedAttention(nn.Module):
    """
    Combines standard self-attention with kNN memory retrieval.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()

        # Local self-attention
        self.local_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Memory retrieval
        self.memory_attention = KNNMemoryAttention(config)

        # Gating mechanism
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
        Combine local attention with memory retrieval.

        The gate learns when to rely on local context vs. historical patterns.
        """
        # Local attention
        local_out, local_attn = self.local_attention(x, x, x)

        # Memory retrieval
        memory_out = self.memory_attention(x, memory)

        # Gated combination
        gate_input = torch.cat([local_out, memory_out], dim=-1)
        gate = self.gate(gate_input)

        output = gate * local_out + (1 - gate) * memory_out

        if return_attention:
            return output, local_attn
        return output, None
```

### Integration with Local Attention

```
Local Attention (Recent Data):
┌──────────────────────────────────────────────┐
│ Today ← Yesterday ← 2 days ago ← ... ← 7 days│
│  x₁   ←   x₂      ←    x₃     ← ... ←   x₇   │
│                   ↓                          │
│           Dense attention matrix             │
│           (All pairs interact)               │
└──────────────────────────────────────────────┘

Memory Retrieval (Historical Data):
┌──────────────────────────────────────────────┐
│        Query: "Current market looks like..." │
│                      ↓                       │
│              kNN Search in Memory            │
│                      ↓                       │
│   Retrieved: [2008 crash, 2020 crash, ...]  │
│                      ↓                       │
│        Attention over retrieved moments      │
└──────────────────────────────────────────────┘

Combined Output:
┌──────────────────────────────────────────────┐
│    α · local_context + (1-α) · memory_context │
│                      ↓                       │
│            Final representation              │
└──────────────────────────────────────────────┘
```

## Financial Applications

### Long-Term Pattern Recognition

Memory-augmented transformers excel at recognizing patterns that span long time periods:

```python
# Example: Detecting market regimes by comparing to historical patterns

def detect_regime(model, current_data, memory):
    """
    Use memory retrieval to identify current market regime.

    Returns:
        regime: Predicted regime (bull/bear/sideways)
        similar_periods: Historical periods with similar patterns
    """
    # Encode current market state
    encoded = model.encode(current_data)

    # Retrieve similar historical moments
    distances, retrieved_indices = memory.search(encoded[-1:])

    # Analyze retrieved periods
    similar_periods = []
    for idx in retrieved_indices[0]:
        period_info = {
            'date': memory.metadata[idx]['date'],
            'regime': memory.metadata[idx]['regime'],
            'subsequent_return': memory.metadata[idx]['future_30d_return']
        }
        similar_periods.append(period_info)

    # Vote on current regime based on similar periods
    regime_votes = Counter([p['regime'] for p in similar_periods])
    predicted_regime = regime_votes.most_common(1)[0][0]

    return predicted_regime, similar_periods
```

### Market Regime Detection

```python
# Store representations with regime labels in memory

class RegimeAwareMemory(ExternalMemory):
    """
    Memory that tracks market regimes for retrieved patterns.
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
        """Add entries with associated metadata"""
        start_idx = self.current_size
        self.add(keys, values)

        for i, (date, regime, ret) in enumerate(zip(dates, regimes, returns)):
            self.metadata[start_idx + i] = {
                'date': date,
                'regime': regime,
                'future_30d_return': ret
            }

    def get_regime_distribution(self, indices: np.ndarray) -> Dict[str, float]:
        """Get distribution of regimes for retrieved indices"""
        regimes = [self.metadata[i]['regime'] for i in indices.flatten()]
        counts = Counter(regimes)
        total = len(regimes)
        return {r: c/total for r, c in counts.items()}
```

### Historical Similarity Trading

```python
class HistoricalSimilarityStrategy:
    """
    Trading strategy based on historical pattern matching.

    Logic: "If the current market looks like X, and X was followed by Y,
    then position for Y to happen again."
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
        Generate trading signal based on historical similarity.
        """
        # Encode current state
        with torch.no_grad():
            encoded = self.model.encode(current_data)

        # Find similar historical moments
        distances, indices = self.memory.search(
            encoded[-1:].numpy(),
            k=self.n_similar
        )

        # Analyze what happened after similar moments
        future_returns = [
            self.memory.metadata[i]['future_30d_return']
            for i in indices[0]
        ]

        # Calculate expected return and confidence
        avg_return = np.mean(future_returns)
        return_std = np.std(future_returns)
        positive_ratio = np.mean([r > 0 for r in future_returns])

        # Generate signal
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

## Practical Examples

### 01: Data Preparation

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
    Load stock data from yfinance.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data frequency ('1h', '1d', etc.)

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    data = {}

    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Add features
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()

        data[symbol] = df.dropna()

    return data


def load_bybit_data(
    symbols: List[str],
    interval: str = '60',  # 60 minutes
    limit: int = 1000
) -> Dict[str, pd.DataFrame]:
    """
    Load cryptocurrency data from Bybit.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        interval: Candle interval in minutes
        limit: Number of candles to fetch

    Returns:
        Dictionary mapping symbols to DataFrames
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

            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

            # Add features
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
    Create sequences for training.

    Args:
        data: DataFrame with OHLCV and features
        seq_len: Sequence length
        features: Feature columns to use

    Returns:
        X: [n_samples, seq_len, n_features]
        y: [n_samples, 1] (next period return)
    """
    X, y = [], []

    values = data[features].values
    returns = data['returns'].values

    for i in range(seq_len, len(data) - 1):
        X.append(values[i-seq_len:i])
        y.append(returns[i+1])  # Predict next return

    return np.array(X), np.array(y)
```

### 02: Memory Bank Construction

```python
# python/02_memory_bank.py

import numpy as np
import faiss
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MemoryConfig:
    """Configuration for external memory"""
    memory_size: int = 100000
    dim: int = 64
    n_neighbors: int = 32
    use_gpu: bool = True


class ExternalMemoryBank:
    """
    External memory bank using FAISS for efficient kNN search.

    Features:
    - FIFO replacement when memory is full
    - GPU acceleration if available
    - Metadata storage for interpretability
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_size = config.memory_size
        self.dim = config.dim
        self.n_neighbors = config.n_neighbors

        # Storage
        self.keys = np.zeros((self.memory_size, self.dim), dtype=np.float32)
        self.values = np.zeros((self.memory_size, self.dim), dtype=np.float32)
        self.current_size = 0
        self.write_pos = 0

        # Metadata for interpretability
        self.timestamps = [None] * self.memory_size
        self.returns = np.zeros(self.memory_size, dtype=np.float32)

        # FAISS index
        self._build_index(config.use_gpu)

    def _build_index(self, use_gpu: bool):
        """Build FAISS index for fast search"""
        # Use inner product (equivalent to cosine sim with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)

        if use_gpu and faiss.get_num_gpus() > 0:
            # Move to GPU
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
        Add entries to memory.

        Args:
            keys: [n, dim] key vectors
            values: [n, dim] value vectors
            timestamps: Optional list of timestamps
            returns: Optional array of future returns
        """
        n = keys.shape[0]

        # Normalize keys for cosine similarity
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

        # Rebuild index
        self.index.reset()
        self.index.add(self.keys[:self.current_size])

    def search(
        self,
        queries: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for k-nearest neighbors.

        Args:
            queries: [n, dim] query vectors
            k: Number of neighbors (default: self.n_neighbors)

        Returns:
            distances: [n, k] similarity scores
            indices: [n, k] memory indices
            values: [n, k, dim] retrieved values
        """
        if k is None:
            k = self.n_neighbors

        # Normalize queries
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

        # Search
        distances, indices = self.index.search(queries.astype(np.float32), k)

        # Get values
        values = self.values[indices]

        return distances, indices, values

    def get_metadata(self, indices: np.ndarray) -> dict:
        """Get metadata for retrieved indices"""
        return {
            'timestamps': [[self.timestamps[i] for i in row] for row in indices],
            'returns': self.returns[indices]
        }
```

### 03: Model Architecture

```python
# python/03_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum


class OutputType(Enum):
    REGRESSION = "regression"
    DIRECTION = "direction"
    PORTFOLIO = "portfolio"


@dataclass
class MemoryTransformerConfig:
    """Configuration for Memory-Augmented Transformer"""
    # Architecture
    input_dim: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # Memory
    memory_size: int = 100000
    n_neighbors: int = 32
    temperature: float = 1.0
    gate_bias: float = 0.0  # Positive = prefer local, negative = prefer memory

    # Sequence
    seq_len: int = 96

    # Output
    output_type: OutputType = OutputType.REGRESSION


class TokenEmbedding(nn.Module):
    """1D CNN token embedding"""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.norm(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class KNNMemoryLayer(nn.Module):
    """
    kNN memory retrieval layer.

    Retrieves similar historical representations and attends to them.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_neighbors = config.n_neighbors
        self.temperature = config.temperature

        # Projections
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory_values: torch.Tensor,
        memory_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            memory_values: [batch, seq_len, k, d_model] retrieved values
            memory_scores: [batch, seq_len, k] similarity scores

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Project values
        values = self.value_proj(memory_values)

        # Attention weights from similarity scores
        attn = F.softmax(memory_scores / self.temperature, dim=-1)

        # Weighted sum
        context = torch.einsum('bsk,bskd->bsd', attn, values)

        return self.out_proj(context)


class MemoryAugmentedEncoderLayer(nn.Module):
    """Encoder layer with memory augmentation"""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()

        # Local self-attention
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Memory attention
        self.memory_layer = KNNMemoryLayer(config)

        # Gating
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, config.gate_bias)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_values: Optional[torch.Tensor] = None,
        memory_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Local attention
        local_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(local_out))

        # Memory attention (if available)
        if memory_values is not None and memory_scores is not None:
            memory_out = self.memory_layer(x, memory_values, memory_scores)

            # Gated combination
            gate_input = torch.cat([local_out, memory_out], dim=-1)
            gate = self.gate(gate_input)

            combined = gate * local_out + (1 - gate) * memory_out
            x = self.norm2(x + self.dropout(combined))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x


class MemoryAugmentedTransformer(nn.Module):
    """
    Memory-Augmented Transformer for financial time series.

    Combines local attention with kNN retrieval from external memory
    to capture both recent and long-term patterns.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = PositionalEncoding(config.d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            MemoryAugmentedEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # Output head
        if config.output_type == OutputType.REGRESSION:
            self.output_head = nn.Linear(config.d_model, 1)
        elif config.output_type == OutputType.DIRECTION:
            self.output_head = nn.Linear(config.d_model, 3)
        elif config.output_type == OutputType.PORTFOLIO:
            self.output_head = nn.Linear(config.d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to representations"""
        x = self.token_embedding(x)
        x = self.position_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        memory_values: Optional[torch.Tensor] = None,
        memory_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim]
            memory_values: [batch, seq_len, k, d_model] (optional)
            memory_scores: [batch, seq_len, k] (optional)

        Returns:
            Dictionary with predictions and representations
        """
        # Encode
        x = self.token_embedding(x)
        x = self.position_encoding(x)

        # Encoder layers with memory
        for layer in self.encoder_layers:
            x = layer(x, memory_values, memory_scores)

        # Use last position for prediction
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Output
        logits = self.output_head(last_hidden)

        if self.config.output_type == OutputType.PORTFOLIO:
            predictions = torch.tanh(logits)  # [-1, 1]
        elif self.config.output_type == OutputType.DIRECTION:
            predictions = F.softmax(logits, dim=-1)
        else:
            predictions = logits

        return {
            'predictions': predictions,
            'representations': x,
            'last_hidden': last_hidden
        }
```

### 04: Training Pipeline

```python
# python/04_train.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryTransformerTrainer:
    """
    Trainer for Memory-Augmented Transformer.

    Handles:
    - Training with memory updates
    - Memory population during training
    - Evaluation with memory retrieval
    """

    def __init__(
        self,
        model: nn.Module,
        memory: 'ExternalMemoryBank',
        config: Dict
    ):
        self.model = model
        self.memory = memory
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(
        self,
        train_loader: DataLoader,
        populate_memory: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Get memory-augmented predictions
            memory_values, memory_scores = self._retrieve_from_memory(batch_x)

            # Forward pass
            outputs = self.model(batch_x, memory_values, memory_scores)
            predictions = outputs['predictions'].squeeze()

            # Loss
            loss = self.loss_fn(predictions, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Populate memory with new representations
            if populate_memory:
                self._add_to_memory(
                    outputs['last_hidden'].detach(),
                    batch_y.detach()
                )

        return {'loss': total_loss / n_batches}

    def _retrieve_from_memory(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from memory for current batch"""
        batch_size, seq_len, _ = x.shape

        if self.memory.current_size == 0:
            # No memory yet, return None
            return None, None

        # Encode without gradients
        with torch.no_grad():
            encoded = self.model.encode(x)

        # Flatten for retrieval
        queries = encoded.view(-1, self.config['d_model']).cpu().numpy()

        # Search memory
        scores, indices, values = self.memory.search(queries)

        # Reshape
        k = values.shape[1]
        values = torch.from_numpy(values).to(self.device)
        values = values.view(batch_size, seq_len, k, -1)

        scores = torch.from_numpy(scores).to(self.device)
        scores = scores.view(batch_size, seq_len, k)

        return values, scores

    def _add_to_memory(
        self,
        representations: torch.Tensor,
        returns: torch.Tensor
    ):
        """Add new representations to memory"""
        keys = representations.cpu().numpy()
        values = representations.cpu().numpy()
        returns = returns.cpu().numpy()

        self.memory.add(keys, values, returns=returns)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                memory_values, memory_scores = self._retrieve_from_memory(batch_x)
                outputs = self.model(batch_x, memory_values, memory_scores)

                predictions = outputs['predictions'].squeeze()
                loss = self.loss_fn(predictions, batch_y)

                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        # Metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))

        # Direction accuracy
        pred_dir = np.sign(all_preds)
        true_dir = np.sign(all_targets)
        direction_acc = np.mean(pred_dir == true_dir)

        return {
            'val_loss': total_loss / len(val_loader),
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_acc
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> List[Dict]:
        """Full training loop"""
        history = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            history.append(metrics)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Dir Acc: {val_metrics['direction_accuracy']:.4f}"
            )

            # Save best
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(self.model.state_dict(), 'best_model.pt')

        return history
```

### 05: Backtesting Strategy

```python
# python/05_backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    transaction_cost: float = 0.001  # 0.1%
    max_position: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MemoryTradingStrategy:
    """
    Trading strategy using Memory-Augmented Transformer.

    Uses both model predictions and historical similarity
    to generate trading signals.
    """

    def __init__(
        self,
        model,
        memory,
        config: BacktestConfig
    ):
        self.model = model
        self.memory = memory
        self.config = config

    def generate_signals(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Generate trading signals for the dataset.

        Returns DataFrame with signals and confidence scores.
        """
        import torch

        self.model.eval()
        signals = []

        seq_len = self.model.config.seq_len

        with torch.no_grad():
            for i in range(seq_len, len(data)):
                # Prepare input
                x = data[feature_cols].iloc[i-seq_len:i].values
                x = torch.FloatTensor(x).unsqueeze(0)

                # Get prediction
                outputs = self.model(x)
                prediction = outputs['predictions'].item()

                # Get memory-based confidence
                hidden = outputs['last_hidden'].cpu().numpy()
                distances, indices, _ = self.memory.search(hidden)

                # Analyze historical outcomes
                historical_returns = self.memory.returns[indices[0]]
                avg_historical = np.mean(historical_returns)
                std_historical = np.std(historical_returns)

                # Combine model and memory signals
                model_signal = np.sign(prediction)
                memory_signal = np.sign(avg_historical)

                # Agreement increases confidence
                if model_signal == memory_signal:
                    confidence = min(abs(prediction) + abs(avg_historical), 1.0)
                    final_signal = model_signal
                else:
                    # Conflicting signals - use model but lower confidence
                    confidence = abs(prediction) * 0.5
                    final_signal = model_signal

                signals.append({
                    'date': data.index[i],
                    'model_prediction': prediction,
                    'memory_avg_return': avg_historical,
                    'memory_std': std_historical,
                    'signal': final_signal,
                    'confidence': confidence,
                    'position_size': final_signal * confidence
                })

        return pd.DataFrame(signals).set_index('date')


class Backtester:
    """
    Backtesting engine for memory-augmented trading strategies.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        signals: pd.DataFrame,
        returns: pd.Series
    ) -> Dict:
        """
        Run backtest.

        Args:
            signals: DataFrame with 'position_size' column
            returns: Series of actual returns

        Returns:
            Dictionary with performance metrics
        """
        # Align signals with returns
        aligned = signals.join(returns.to_frame('actual_return'), how='inner')

        # Calculate strategy returns
        strategy_returns = aligned['position_size'].shift(1) * aligned['actual_return']
        strategy_returns = strategy_returns.fillna(0)

        # Transaction costs
        position_changes = aligned['position_size'].diff().abs().fillna(0)
        costs = position_changes * self.config.transaction_cost

        net_returns = strategy_returns - costs

        # Cumulative returns
        cumulative = (1 + net_returns).cumprod()

        # Metrics
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(cumulative)) - 1

        volatility = net_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino
        downside = net_returns[net_returns < 0].std() * np.sqrt(252)
        sortino = annual_return / downside if downside > 0 else 0

        # Win rate
        winning_trades = (net_returns > 0).sum()
        total_trades = (net_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'cumulative_returns': cumulative,
            'daily_returns': net_returns
        }

    def plot_results(self, results: Dict, title: str = "Strategy Performance"):
        """Plot backtest results"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Cumulative returns
        ax = axes[0, 0]
        results['cumulative_returns'].plot(ax=ax)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Growth of $1')

        # Drawdown
        ax = axes[0, 1]
        running_max = results['cumulative_returns'].cummax()
        drawdown = (results['cumulative_returns'] - running_max) / running_max
        drawdown.plot(ax=ax, color='red')
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown %')

        # Returns distribution
        ax = axes[1, 0]
        results['daily_returns'].hist(bins=50, ax=ax)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Return')

        # Rolling Sharpe
        ax = axes[1, 1]
        rolling_sharpe = (
            results['daily_returns'].rolling(60).mean() /
            results['daily_returns'].rolling(60).std()
        ) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title('Rolling 60-Day Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        plt.close()

        # Print summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return:     {results['total_return']*100:.2f}%")
        print(f"Annual Return:    {results['annual_return']*100:.2f}%")
        print(f"Volatility:       {results['volatility']*100:.2f}%")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:    {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
        print(f"Win Rate:         {results['win_rate']*100:.2f}%")
        print("="*50)
```

## Rust Implementation

See [rust_memory_transformer](rust_memory_transformer/) for complete Rust implementation using Bybit data.

```
rust_memory_transformer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client
│   │   └── types.rs        # API types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading
│   │   └── features.rs     # Feature engineering
│   ├── memory/             # External memory
│   │   ├── mod.rs
│   │   ├── bank.rs         # Memory bank
│   │   └── search.rs       # kNN search
│   ├── model/              # Model architecture
│   │   ├── mod.rs
│   │   ├── embedding.rs    # Token embedding
│   │   ├── attention.rs    # Memory attention
│   │   └── transformer.rs  # Full model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_memory_transformer

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 50 --memory-size 50000

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── model.py               # Memory-Augmented Transformer
├── memory.py              # External memory bank
├── data.py                # Data loading (yfinance, Bybit)
├── train.py               # Training script
├── backtest.py            # Backtesting utilities
├── requirements.txt       # Dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_memory_bank.ipynb
    ├── 03_model_training.ipynb
    ├── 04_historical_similarity.ipynb
    └── 05_backtesting.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python data.py --symbols BTCUSDT,ETHUSDT,AAPL,MSFT

# Train model
python train.py --epochs 100 --memory-size 100000

# Run backtest
python backtest.py --model checkpoints/best.pt
```

## Best Practices

### When to Use Memory-Augmented Transformers

**Good use cases:**
- Long-term pattern recognition (months to years)
- Market regime detection and regime-switching strategies
- Rare event modeling (crashes, squeezes)
- Historical similarity trading
- Continuous learning without retraining

**Not ideal for:**
- Ultra-high-frequency trading (retrieval latency)
- Very short-term predictions (memory overhead not justified)
- Completely novel market conditions (no similar memories)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `memory_size` | 50K-200K | Larger = more history, slower retrieval |
| `n_neighbors` | 16-64 | More neighbors = smoother predictions |
| `d_model` | 64-128 | Match with memory dimension |
| `temperature` | 0.5-2.0 | Lower = sharper attention, higher = smoother |
| `gate_bias` | 0.0 | Adjust if model over-relies on memory |

### Memory Management

1. **FIFO vs. Importance-Based Replacement**
   ```python
   # FIFO (simple, fast)
   # Old entries are replaced first

   # Importance-based (better retention of rare events)
   # Keep high-gradient or high-return entries longer
   ```

2. **Memory Warming**
   ```python
   # Before trading, populate memory with historical data
   for historical_batch in historical_data:
       with torch.no_grad():
           hidden = model.encode(historical_batch)
           memory.add(hidden, returns)
   ```

3. **Memory Pruning**
   ```python
   # Remove low-quality entries periodically
   # E.g., entries that never got retrieved
   ```

### Common Pitfalls

1. **Memory Staleness**: Old patterns may not be relevant
   - Solution: Use time-weighted retrieval or memory decay

2. **Retrieval Latency**: kNN search can be slow for large memories
   - Solution: Use approximate search (FAISS, ScaNN)

3. **Cold Start**: No useful memories at the beginning
   - Solution: Pre-populate with historical data before live trading

4. **Distribution Shift**: Market changes but old memories remain
   - Solution: Use adaptive memory replacement or regime-aware memories

## Resources

### Papers

- [Memorizing Transformers](https://arxiv.org/abs/2203.08913) - Original paper on kNN memory for transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RETRO](https://arxiv.org/abs/2112.04426) - Retrieval-Enhanced Transformer
- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Extended context for transformers

### Implementations

- [memorizing-transformers-pytorch](https://github.com/lucidrains/memorizing-transformers-pytorch) - PyTorch implementation
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Google's scalable nearest neighbors

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) - Multi-horizon forecasting
- [Chapter 28: Regime Detection HMM](../28_regime_detection_hmm) - Market regime detection
- [Chapter 49: Multi-Scale Attention](../49_multi_scale_attention) - Multi-resolution attention

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and attention mechanisms
- Nearest neighbor search algorithms
- Time series forecasting fundamentals
- PyTorch or Rust ML libraries
