//! Memory-Augmented Transformer Model
//!
//! Implements the transformer architecture with external memory attention,
//! based on the "Memorizing Transformers" paper.

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::memory::ExternalMemoryBank;

/// Output type for the model
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutputType {
    /// Continuous value regression (e.g., price prediction)
    Regression,
    /// Binary classification (up/down direction)
    Direction,
    /// Portfolio weights (softmax output)
    Portfolio,
    /// Quantile predictions
    Quantile,
}

/// Configuration for the Memory-Augmented Transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformerConfig {
    /// Number of input features
    pub input_dim: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feedforward dimension
    pub d_ff: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Dropout rate (for reference, not used in inference)
    pub dropout: f32,
    /// Output type
    pub output_type: OutputType,
    /// Number of output dimensions
    pub output_dim: usize,
    /// Memory dimension (should match d_model)
    pub memory_dim: usize,
    /// Number of memory neighbors to retrieve
    pub memory_k: usize,
    /// Gate initialization bias (higher = trust memory more initially)
    pub gate_bias: f32,
}

impl Default for MemoryTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,  // OHLCV
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            seq_len: 20,
            dropout: 0.1,
            output_type: OutputType::Direction,
            output_dim: 1,
            memory_dim: 64,
            memory_k: 10,
            gate_bias: 0.0,
        }
    }
}

/// Linear layer
#[derive(Debug, Clone)]
struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier initialization
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let weight = Array2::random((out_features, in_features), normal);
        let bias = Array1::zeros(out_features);
        Self { weight, bias }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x: (batch, in_features) -> (batch, out_features)
        x.dot(&self.weight.t()) + &self.bias
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let var = x.mapv(|v| v * v).mean_axis(Axis(1)).unwrap().insert_axis(Axis(1))
            - mean.mapv(|m| m * m);
        let std = var.mapv(|v| (v + self.eps).sqrt());

        let normalized = (x - &mean) / &std;
        &normalized * &self.gamma + &self.beta
    }
}

/// Multi-head self-attention
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MultiHeadAttention {
    n_heads: usize,
    d_head: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_heads: usize) -> Self {
        let d_head = d_model / n_heads;
        Self {
            n_heads,
            d_head,
            query: Linear::new(d_model, d_model),
            key: Linear::new(d_model, d_model),
            value: Linear::new(d_model, d_model),
            out: Linear::new(d_model, d_model),
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let _seq_len = x.nrows();
        let _d_model = x.ncols();

        let q = self.query.forward(x);
        let k = self.key.forward(x);
        let v = self.value.forward(x);

        // Simple single-head attention for efficiency
        // In production, implement proper multi-head
        let scale = (self.d_head as f32).sqrt();
        let scores = q.dot(&k.t()) / scale;

        // Softmax
        let max_scores = scores.map_axis(Axis(1), |row| {
            row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        });
        let exp_scores = (&scores - &max_scores.insert_axis(Axis(1))).mapv(|x| x.exp());
        let sum_exp = exp_scores.sum_axis(Axis(1)).insert_axis(Axis(1));
        let attn_weights = &exp_scores / &sum_exp;

        let attn_output = attn_weights.dot(&v);
        self.out.forward(&attn_output)
    }
}

/// Feedforward network
#[derive(Debug, Clone)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff),
            linear2: Linear::new(d_ff, d_model),
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // GELU approximation
        let h = self.linear1.forward(x);
        let h = h.mapv(|x| x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()));
        self.linear2.forward(&h)
    }
}

/// Encoder layer with optional memory attention
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    memory_gate: Array1<f32>,
}

impl EncoderLayer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, gate_bias: f32) -> Self {
        let mut memory_gate = Array1::zeros(d_model);
        memory_gate.fill(gate_bias);

        Self {
            self_attn: MultiHeadAttention::new(d_model, n_heads),
            ff: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            memory_gate,
        }
    }

    fn forward(&self, x: &Array2<f32>, memory_context: Option<&Array1<f32>>) -> Array2<f32> {
        // Self-attention with residual
        let attn_out = self.self_attn.forward(x);
        let x = &self.norm1.forward(&(x + &attn_out));

        // Combine with memory context if available
        let x = if let Some(mem_ctx) = memory_context {
            let gate = self.memory_gate.mapv(|g| 1.0 / (1.0 + (-g).exp())); // sigmoid
            let mut result = x.clone();
            for mut row in result.rows_mut() {
                let local = row.to_owned();
                let combined = &local * (1.0 - &gate) + mem_ctx * &gate;
                row.assign(&combined);
            }
            result
        } else {
            x.clone()
        };

        // Feedforward with residual
        let ff_out = self.ff.forward(&x);
        self.norm2.forward(&(&x + &ff_out))
    }
}

/// Positional encoding
fn positional_encoding(seq_len: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((seq_len, d_model));

    for pos in 0..seq_len {
        for i in 0..d_model {
            let angle = pos as f32 / (10000.0_f32).powf(2.0 * (i / 2) as f32 / d_model as f32);
            if i % 2 == 0 {
                pe[[pos, i]] = angle.sin();
            } else {
                pe[[pos, i]] = angle.cos();
            }
        }
    }

    pe
}

/// Memory-Augmented Transformer
#[derive(Debug)]
pub struct MemoryAugmentedTransformer {
    config: MemoryTransformerConfig,
    input_projection: Linear,
    layers: Vec<EncoderLayer>,
    output_layer: Linear,
    pos_encoding: Array2<f32>,
}

impl MemoryAugmentedTransformer {
    /// Create a new Memory-Augmented Transformer
    pub fn new(config: MemoryTransformerConfig) -> Self {
        let input_projection = Linear::new(config.input_dim, config.d_model);

        let layers: Vec<EncoderLayer> = (0..config.n_layers)
            .map(|_| EncoderLayer::new(config.d_model, config.n_heads, config.d_ff, config.gate_bias))
            .collect();

        let output_dim = match config.output_type {
            OutputType::Direction => 1,
            OutputType::Quantile => 4,
            _ => config.output_dim,
        };
        let output_layer = Linear::new(config.d_model, output_dim);

        let pos_encoding = positional_encoding(config.seq_len, config.d_model);

        Self {
            config,
            input_projection,
            layers,
            output_layer,
            pos_encoding,
        }
    }

    /// Encode input sequence to hidden representation
    pub fn encode(&self, x: &Array2<f32>) -> Array2<f32> {
        // Project input to model dimension
        let mut h = self.input_projection.forward(x);

        // Add positional encoding
        let seq_len = h.nrows().min(self.pos_encoding.nrows());
        for i in 0..seq_len {
            for j in 0..h.ncols() {
                h[[i, j]] += self.pos_encoding[[i, j]];
            }
        }

        // Pass through encoder layers (without memory for encoding)
        for layer in &self.layers {
            h = layer.forward(&h, None);
        }

        h
    }

    /// Get the embedding of the last token (for memory storage)
    pub fn get_embedding(&self, x: &Array2<f32>) -> Array1<f32> {
        let encoded = self.encode(x);
        encoded.row(encoded.nrows() - 1).to_owned()
    }

    /// Forward pass with optional memory retrieval
    pub fn forward(
        &self,
        x: &Array2<f32>,
        memory_bank: Option<&ExternalMemoryBank>,
    ) -> Array1<f32> {
        // Project input
        let mut h = self.input_projection.forward(x);

        // Add positional encoding
        let seq_len = h.nrows().min(self.pos_encoding.nrows());
        for i in 0..seq_len {
            for j in 0..h.ncols() {
                h[[i, j]] += self.pos_encoding[[i, j]];
            }
        }

        // Get memory context if available
        let memory_context = if let Some(bank) = memory_bank {
            if !bank.is_empty() {
                // Use last token's embedding as query
                let query: Vec<f32> = h.row(h.nrows() - 1).to_vec();
                bank.get_memory_context(&query, Some(self.config.memory_k)).ok()
            } else {
                None
            }
        } else {
            None
        };

        // Pass through encoder layers
        for layer in &self.layers {
            h = layer.forward(&h, memory_context.as_ref());
        }

        // Take last token for prediction
        let last_hidden = h.row(h.nrows() - 1).to_owned();
        let output = self.output_layer.forward(&last_hidden.insert_axis(Axis(0)));
        let output = output.row(0).to_owned();

        // Apply output activation based on output type
        match self.config.output_type {
            OutputType::Direction => output.mapv(|x| 1.0 / (1.0 + (-x).exp())), // sigmoid
            OutputType::Portfolio => {
                // softmax
                let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals = output.mapv(|x| (x - max_val).exp());
                let sum = exp_vals.sum();
                exp_vals / sum
            }
            _ => output,
        }
    }

    /// Predict direction (up/down) with confidence
    pub fn predict_direction(
        &self,
        x: &Array2<f32>,
        memory_bank: Option<&ExternalMemoryBank>,
    ) -> (bool, f32) {
        let output = self.forward(x, memory_bank);
        let prob = output[0];
        (prob > 0.5, (prob - 0.5).abs() * 2.0)  // confidence 0-1
    }

    /// Get configuration
    pub fn config(&self) -> &MemoryTransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_model_forward() {
        let config = MemoryTransformerConfig {
            input_dim: 5,
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            d_ff: 64,
            seq_len: 10,
            dropout: 0.0,
            output_type: OutputType::Direction,
            output_dim: 1,
            memory_dim: 32,
            memory_k: 5,
            gate_bias: 0.0,
        };

        let model = MemoryAugmentedTransformer::new(config);

        // Create dummy input: 10 timesteps, 5 features (OHLCV)
        let x = Array2::random((10, 5), Uniform::new(0.0, 1.0));

        let output = model.forward(&x, None);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0); // sigmoid output
    }

    #[test]
    fn test_model_with_memory() {
        use crate::memory::{ExternalMemoryBank, MemoryConfig};

        let model_config = MemoryTransformerConfig {
            input_dim: 5,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            seq_len: 5,
            dropout: 0.0,
            output_type: OutputType::Direction,
            output_dim: 1,
            memory_dim: 16,
            memory_k: 3,
            gate_bias: 0.0,
        };

        let model = MemoryAugmentedTransformer::new(model_config);

        let memory_config = MemoryConfig {
            dim: 16,
            max_entries: 100,
            k: 3,
            normalize: true,
        };
        let mut memory = ExternalMemoryBank::new(memory_config);

        // Add some patterns to memory
        for i in 0..10 {
            let x = Array2::random((5, 5), Uniform::new(0.0, 1.0));
            let embedding = model.get_embedding(&x);
            memory.add(embedding.to_vec(), i as i64 * 1000, Some(0.05)).unwrap();
        }

        // Forward with memory
        let x = Array2::random((5, 5), Uniform::new(0.0, 1.0));
        let output = model.forward(&x, Some(&memory));
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding(10, 32);
        assert_eq!(pe.shape(), &[10, 32]);

        // Check that values are in reasonable range
        for val in pe.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }
}
