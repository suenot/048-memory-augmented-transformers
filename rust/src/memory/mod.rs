//! External Memory Bank with kNN Retrieval
//!
//! Implements the external memory component of the Memory-Augmented Transformer.
//! Uses brute-force kNN search for small-scale applications.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

/// Memory-related errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory is empty, cannot search")]
    EmptyMemory,
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Configuration for the external memory bank
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Dimension of stored vectors
    pub dim: usize,
    /// Maximum number of entries to store
    pub max_entries: usize,
    /// Number of neighbors to retrieve (k in kNN)
    pub k: usize,
    /// Whether to normalize vectors before storage
    pub normalize: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            max_entries: 100_000,
            k: 10,
            normalize: true,
        }
    }
}

/// A single memory entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: usize,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Timestamp of the entry
    pub timestamp: i64,
    /// Market regime label (optional)
    pub regime: Option<String>,
    /// Future return after this pattern (for training)
    pub future_return: Option<f32>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Search result from kNN query
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index of the matching entry
    pub index: usize,
    /// Distance to query vector
    pub distance: f32,
    /// The memory entry
    pub entry: MemoryEntry,
}

/// External Memory Bank with kNN retrieval
///
/// This implements the memory component that stores historical patterns
/// and retrieves similar ones during inference.
#[derive(Debug)]
pub struct ExternalMemoryBank {
    config: MemoryConfig,
    entries: Vec<MemoryEntry>,
    vectors: Option<Array2<f32>>,
    next_id: usize,
}

impl ExternalMemoryBank {
    /// Create a new memory bank
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            vectors: None,
            next_id: 0,
        }
    }

    /// Get the number of entries in memory
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a single entry to memory
    pub fn add(&mut self, embedding: Vec<f32>, timestamp: i64, future_return: Option<f32>) -> Result<usize, MemoryError> {
        if embedding.len() != self.config.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.len(),
            });
        }

        // Normalize if configured
        let embedding = if self.config.normalize {
            normalize_vector(&embedding)
        } else {
            embedding
        };

        let id = self.next_id;
        self.next_id += 1;

        let entry = MemoryEntry {
            id,
            embedding: embedding.clone(),
            timestamp,
            regime: None,
            future_return,
            metadata: HashMap::new(),
        };

        self.entries.push(entry);

        // Rebuild the vector matrix (in production, use incremental update)
        self.rebuild_vectors();

        // Evict oldest entries if over capacity
        if self.entries.len() > self.config.max_entries {
            self.evict_oldest(self.entries.len() - self.config.max_entries);
        }

        Ok(id)
    }

    /// Add multiple entries at once (more efficient)
    pub fn add_batch(
        &mut self,
        embeddings: &Array2<f32>,
        timestamps: &[i64],
        future_returns: Option<&[f32]>,
    ) -> Result<Vec<usize>, MemoryError> {
        if embeddings.ncols() != self.config.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.dim,
                got: embeddings.ncols(),
            });
        }

        let mut ids = Vec::with_capacity(embeddings.nrows());

        for (i, row) in embeddings.rows().into_iter().enumerate() {
            let embedding: Vec<f32> = row.to_vec();
            let embedding = if self.config.normalize {
                normalize_vector(&embedding)
            } else {
                embedding
            };

            let id = self.next_id;
            self.next_id += 1;
            ids.push(id);

            let entry = MemoryEntry {
                id,
                embedding,
                timestamp: timestamps.get(i).copied().unwrap_or(0),
                regime: None,
                future_return: future_returns.and_then(|fr| fr.get(i).copied()),
                metadata: HashMap::new(),
            };

            self.entries.push(entry);
        }

        self.rebuild_vectors();

        // Evict if necessary
        if self.entries.len() > self.config.max_entries {
            self.evict_oldest(self.entries.len() - self.config.max_entries);
        }

        Ok(ids)
    }

    /// Search for k-nearest neighbors
    pub fn search(&self, query: &[f32], k: Option<usize>) -> Result<Vec<SearchResult>, MemoryError> {
        if self.is_empty() {
            return Err(MemoryError::EmptyMemory);
        }

        if query.len() != self.config.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }

        let k = k.unwrap_or(self.config.k).min(self.entries.len());

        // Normalize query if configured
        let query = if self.config.normalize {
            normalize_vector(query)
        } else {
            query.to_vec()
        };

        let query_arr = Array1::from(query);

        // Compute distances to all entries
        let vectors = self.vectors.as_ref().unwrap();
        let mut distances: Vec<(usize, f32)> = vectors
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let diff = &row - &query_arr;
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                (i, dist)
            })
            .collect();

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let results: Vec<SearchResult> = distances
            .into_iter()
            .take(k)
            .map(|(idx, dist)| SearchResult {
                index: idx,
                distance: dist,
                entry: self.entries[idx].clone(),
            })
            .collect();

        Ok(results)
    }

    /// Get aggregated memory context from search results
    pub fn get_memory_context(&self, query: &[f32], k: Option<usize>) -> Result<Array1<f32>, MemoryError> {
        let results = self.search(query, k)?;

        if results.is_empty() {
            return Err(MemoryError::EmptyMemory);
        }

        // Weight by inverse distance
        let total_weight: f32 = results.iter().map(|r| 1.0 / (r.distance + 1e-8)).sum();

        let mut context = Array1::<f32>::zeros(self.config.dim);
        for result in &results {
            let weight = 1.0 / (result.distance + 1e-8) / total_weight;
            let entry_arr = Array1::from(result.entry.embedding.clone());
            context = context + entry_arr * weight;
        }

        Ok(context)
    }

    /// Predict future return based on similar patterns
    pub fn predict_from_similar(&self, query: &[f32], k: Option<usize>) -> Result<Option<f32>, MemoryError> {
        let results = self.search(query, k)?;

        // Filter entries that have future return data
        let with_returns: Vec<&SearchResult> = results
            .iter()
            .filter(|r| r.entry.future_return.is_some())
            .collect();

        if with_returns.is_empty() {
            return Ok(None);
        }

        // Weight-averaged prediction
        let total_weight: f32 = with_returns.iter().map(|r| 1.0 / (r.distance + 1e-8)).sum();
        let prediction: f32 = with_returns
            .iter()
            .map(|r| {
                let weight = 1.0 / (r.distance + 1e-8) / total_weight;
                r.entry.future_return.unwrap() * weight
            })
            .sum();

        Ok(Some(prediction))
    }

    /// Save memory to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), MemoryError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &(&self.config, &self.entries, self.next_id))
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load memory from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let (config, entries, next_id): (MemoryConfig, Vec<MemoryEntry>, usize) =
            bincode::deserialize_from(reader)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        let mut bank = Self {
            config,
            entries,
            vectors: None,
            next_id,
        };
        bank.rebuild_vectors();
        Ok(bank)
    }

    /// Rebuild the internal vector matrix
    fn rebuild_vectors(&mut self) {
        if self.entries.is_empty() {
            self.vectors = None;
            return;
        }

        let n = self.entries.len();
        let d = self.config.dim;
        let mut data = Vec::with_capacity(n * d);

        for entry in &self.entries {
            data.extend_from_slice(&entry.embedding);
        }

        self.vectors = Some(Array2::from_shape_vec((n, d), data).unwrap());
    }

    /// Evict the oldest entries
    fn evict_oldest(&mut self, count: usize) {
        if count >= self.entries.len() {
            self.entries.clear();
        } else {
            self.entries.drain(0..count);
        }
        self.rebuild_vectors();
    }
}

/// Normalize a vector to unit length
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_add_and_search() {
        let config = MemoryConfig {
            dim: 4,
            max_entries: 100,
            k: 3,
            normalize: true,
        };
        let mut memory = ExternalMemoryBank::new(config);

        // Add some entries
        memory.add(vec![1.0, 0.0, 0.0, 0.0], 1000, Some(0.05)).unwrap();
        memory.add(vec![0.0, 1.0, 0.0, 0.0], 2000, Some(-0.02)).unwrap();
        memory.add(vec![0.0, 0.0, 1.0, 0.0], 3000, Some(0.10)).unwrap();

        assert_eq!(memory.len(), 3);

        // Search for similar to first entry
        let results = memory.search(&[1.0, 0.1, 0.0, 0.0], None).unwrap();
        assert_eq!(results.len(), 3);

        // First result should be closest to [1, 0, 0, 0]
        assert_eq!(results[0].entry.timestamp, 1000);
    }

    #[test]
    fn test_predict_from_similar() {
        let config = MemoryConfig {
            dim: 2,
            max_entries: 100,
            k: 3,
            normalize: false,
        };
        let mut memory = ExternalMemoryBank::new(config);

        memory.add(vec![1.0, 0.0], 1000, Some(0.10)).unwrap();
        memory.add(vec![0.9, 0.1], 2000, Some(0.08)).unwrap();
        memory.add(vec![0.0, 1.0], 3000, Some(-0.05)).unwrap();

        let prediction = memory.predict_from_similar(&[1.0, 0.0], Some(2)).unwrap();
        assert!(prediction.is_some());
        // Should be close to 0.10 since first entry is exact match
        let pred = prediction.unwrap();
        assert!(pred > 0.05);
    }

    #[test]
    fn test_memory_eviction() {
        let config = MemoryConfig {
            dim: 2,
            max_entries: 3,
            k: 2,
            normalize: false,
        };
        let mut memory = ExternalMemoryBank::new(config);

        memory.add(vec![1.0, 0.0], 1000, None).unwrap();
        memory.add(vec![0.0, 1.0], 2000, None).unwrap();
        memory.add(vec![1.0, 1.0], 3000, None).unwrap();
        memory.add(vec![0.0, 0.0], 4000, None).unwrap(); // Should evict first

        assert_eq!(memory.len(), 3);
        // First entry (timestamp 1000) should be evicted
        let has_1000 = memory.entries.iter().any(|e| e.timestamp == 1000);
        assert!(!has_1000);
    }
}
