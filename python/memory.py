"""
External Memory Bank Implementation

Provides:
- MemoryConfig: Memory configuration
- ExternalMemoryBank: FAISS-based memory with kNN search
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for external memory"""
    memory_size: int = 100000
    dim: int = 64
    n_neighbors: int = 32
    use_gpu: bool = False  # FAISS GPU support


class ExternalMemoryBank:
    """
    External memory bank using FAISS for efficient kNN search.

    Features:
    - FIFO replacement when memory is full
    - GPU acceleration if available
    - Metadata storage for interpretability

    Example:
        config = MemoryConfig(memory_size=10000, dim=64)
        memory = ExternalMemoryBank(config)

        # Add entries
        keys = np.random.randn(100, 64).astype(np.float32)
        values = np.random.randn(100, 64).astype(np.float32)
        memory.add(keys, values)

        # Search
        queries = np.random.randn(10, 64).astype(np.float32)
        distances, indices, values = memory.search(queries)
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
        self.timestamps: List[Optional[Any]] = [None] * self.memory_size
        self.returns = np.zeros(self.memory_size, dtype=np.float32)
        self.metadata: Dict[int, Dict] = {}

        # FAISS index
        self._build_index(config.use_gpu)

    def _build_index(self, use_gpu: bool):
        """Build FAISS index for fast search"""
        try:
            import faiss

            # Use inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dim)

            if use_gpu:
                try:
                    if faiss.get_num_gpus() > 0:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                        self.on_gpu = True
                        logger.info("FAISS index moved to GPU")
                    else:
                        self.on_gpu = False
                        logger.info("No GPU available, using CPU")
                except Exception as e:
                    logger.warning(f"Failed to use GPU: {e}")
                    self.on_gpu = False
            else:
                self.on_gpu = False

            self.use_faiss = True
            logger.info("FAISS index initialized")

        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self.index = None
            self.use_faiss = False
            self.on_gpu = False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return vectors / norms

    def add(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        timestamps: Optional[List] = None,
        returns: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add entries to memory.

        Args:
            keys: [n, dim] key vectors
            values: [n, dim] value vectors
            timestamps: Optional list of timestamps
            returns: Optional array of future returns
            metadata: Optional list of metadata dicts
        """
        n = keys.shape[0]
        keys = self._normalize(keys.astype(np.float32))
        values = values.astype(np.float32)

        for i in range(n):
            pos = self.write_pos % self.memory_size

            self.keys[pos] = keys[i]
            self.values[pos] = values[i]

            if timestamps is not None and i < len(timestamps):
                self.timestamps[pos] = timestamps[i]
            if returns is not None and i < len(returns):
                self.returns[pos] = returns[i]
            if metadata is not None and i < len(metadata):
                self.metadata[pos] = metadata[i]

            self.write_pos += 1
            self.current_size = min(self.current_size + 1, self.memory_size)

        # Rebuild index with current entries
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index with current entries"""
        if self.use_faiss and self.current_size > 0:
            import faiss

            # Create new index
            if self.on_gpu:
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(self.dim)
                cpu_index.add(self.keys[:self.current_size])
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
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
        if self.current_size == 0:
            # Empty memory, return zeros
            n = queries.shape[0]
            k = k or self.n_neighbors
            return (
                np.zeros((n, k), dtype=np.float32),
                np.zeros((n, k), dtype=np.int64),
                np.zeros((n, k, self.dim), dtype=np.float32)
            )

        if k is None:
            k = self.n_neighbors
        k = min(k, self.current_size)

        queries = self._normalize(queries.astype(np.float32))

        if self.use_faiss:
            distances, indices = self.index.search(queries, k)
        else:
            # Numpy fallback
            distances, indices = self._numpy_search(queries, k)

        # Clamp indices to valid range
        indices = np.clip(indices, 0, self.current_size - 1)

        # Get values
        values = self.values[indices]

        return distances, indices, values

    def _numpy_search(
        self,
        queries: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback kNN search using numpy"""
        # Compute similarities: [n_queries, current_size]
        keys = self.keys[:self.current_size]
        similarities = queries @ keys.T

        # Get top-k
        indices = np.argsort(-similarities, axis=1)[:, :k]
        distances = np.take_along_axis(similarities, indices, axis=1)

        return distances, indices

    def get_metadata(self, indices: np.ndarray) -> Dict[str, Any]:
        """
        Get metadata for retrieved indices.

        Args:
            indices: [n, k] array of memory indices

        Returns:
            Dictionary with timestamps and returns
        """
        return {
            'timestamps': [
                [self.timestamps[i] for i in row if i < self.current_size]
                for row in indices
            ],
            'returns': self.returns[indices],
            'metadata': [
                [self.metadata.get(i, {}) for i in row if i < self.current_size]
                for row in indices
            ]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'current_size': self.current_size,
            'memory_size': self.memory_size,
            'fill_ratio': self.current_size / self.memory_size,
            'write_pos': self.write_pos,
            'use_faiss': self.use_faiss,
            'on_gpu': self.on_gpu,
        }

    def clear(self):
        """Clear all memory entries"""
        self.keys.fill(0)
        self.values.fill(0)
        self.returns.fill(0)
        self.timestamps = [None] * self.memory_size
        self.metadata.clear()
        self.current_size = 0
        self.write_pos = 0
        self._rebuild_index()

    def save(self, path: str):
        """Save memory to disk"""
        np.savez(
            path,
            keys=self.keys[:self.current_size],
            values=self.values[:self.current_size],
            returns=self.returns[:self.current_size],
            current_size=self.current_size,
            write_pos=self.write_pos
        )
        logger.info(f"Memory saved to {path}")

    def load(self, path: str):
        """Load memory from disk"""
        data = np.load(path)
        n = data['current_size']

        self.keys[:n] = data['keys']
        self.values[:n] = data['values']
        self.returns[:n] = data['returns']
        self.current_size = int(data['current_size'])
        self.write_pos = int(data['write_pos'])

        self._rebuild_index()
        logger.info(f"Memory loaded from {path}, {self.current_size} entries")


class RegimeAwareMemory(ExternalMemoryBank):
    """
    Memory that tracks market regimes for retrieved patterns.

    Extends ExternalMemoryBank with regime-specific functionality.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.regimes: List[Optional[str]] = [None] * self.memory_size

    def add_with_regime(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        regimes: List[str],
        timestamps: Optional[List] = None,
        returns: Optional[np.ndarray] = None
    ):
        """Add entries with regime labels"""
        n = keys.shape[0]

        # Capture write_pos before adding
        start_pos = self.write_pos

        # Call parent add
        self.add(keys, values, timestamps, returns)

        # Store regimes at the positions where entries were actually written
        for i in range(n):
            pos = (start_pos + i) % self.memory_size
            if i < len(regimes):
                self.regimes[pos] = regimes[i]

    def get_regime_distribution(self, indices: np.ndarray) -> Dict[str, float]:
        """
        Get distribution of regimes for retrieved indices.

        Args:
            indices: [n, k] array of memory indices

        Returns:
            Dictionary mapping regime names to proportions
        """
        from collections import Counter

        regimes = []
        for row in indices:
            for i in row:
                if i < self.current_size and self.regimes[i] is not None:
                    regimes.append(self.regimes[i])

        if not regimes:
            return {}

        counts = Counter(regimes)
        total = len(regimes)
        return {r: c / total for r, c in counts.items()}

    def predict_regime(self, indices: np.ndarray) -> Tuple[str, float]:
        """
        Predict current regime based on retrieved historical patterns.

        Args:
            indices: [n, k] array of memory indices

        Returns:
            Tuple of (predicted_regime, confidence)
        """
        distribution = self.get_regime_distribution(indices)

        if not distribution:
            return 'unknown', 0.0

        # Most common regime
        predicted = max(distribution, key=distribution.get)
        confidence = distribution[predicted]

        return predicted, confidence


if __name__ == "__main__":
    # Test the memory bank
    print("Testing External Memory Bank...")

    config = MemoryConfig(memory_size=1000, dim=32, n_neighbors=5)
    memory = ExternalMemoryBank(config)

    # Add some entries
    keys = np.random.randn(100, 32).astype(np.float32)
    values = np.random.randn(100, 32).astype(np.float32)
    returns = np.random.randn(100).astype(np.float32) * 0.02

    memory.add(keys, values, returns=returns)
    print(f"Added 100 entries. Stats: {memory.get_statistics()}")

    # Search
    queries = np.random.randn(10, 32).astype(np.float32)
    distances, indices, retrieved_values = memory.search(queries, k=5)

    print(f"Distances shape: {distances.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Values shape: {retrieved_values.shape}")

    # Test regime-aware memory
    print("\nTesting Regime-Aware Memory...")

    regime_memory = RegimeAwareMemory(config)
    regimes = ['bull'] * 50 + ['bear'] * 50
    regime_memory.add_with_regime(keys, values, regimes, returns=returns)

    # Search and get regime distribution
    distances, indices, _ = regime_memory.search(queries[:1], k=20)
    distribution = regime_memory.get_regime_distribution(indices)
    print(f"Regime distribution: {distribution}")

    regime, confidence = regime_memory.predict_regime(indices)
    print(f"Predicted regime: {regime} (confidence: {confidence:.2f})")

    print("\nAll tests passed!")
