"""
Memory-Augmented Transformer Model Implementation

Provides:
- MemoryTransformerConfig: Model configuration
- MemoryAugmentedTransformer: Main model with external memory support
- KNNMemoryLayer: kNN-based memory retrieval layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"
    DIRECTION = "direction"
    PORTFOLIO = "portfolio"
    QUANTILE = "quantile"


@dataclass
class MemoryTransformerConfig:
    """
    Configuration for Memory-Augmented Transformer

    Example:
        config = MemoryTransformerConfig(
            input_dim=6,
            d_model=64,
            n_heads=4
        )
    """
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
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.dropout >= 0 and self.dropout < 1, "dropout must be in [0, 1)"
        assert self.n_neighbors > 0, "n_neighbors must be positive"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution

    Converts [batch, seq_len, features] to [batch, seq_len, d_model]
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.activation(self.norm(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class KNNMemoryLayer(nn.Module):
    """
    kNN memory retrieval layer.

    Retrieves similar historical representations and attends to them.
    This layer does NOT perform the kNN search itself - it receives
    pre-computed retrieved values and similarity scores.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_neighbors = config.n_neighbors
        self.temperature = config.temperature

        # Projections
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        _x: torch.Tensor,
        memory_values: torch.Tensor,
        memory_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            _x: Current representations [batch, seq_len, d_model] (unused, for interface consistency)
            memory_values: Retrieved values [batch, seq_len, k, d_model]
            memory_scores: Similarity scores [batch, seq_len, k]

        Returns:
            Memory-augmented output [batch, seq_len, d_model]
        """
        # Project values
        values = self.value_proj(memory_values)

        # Attention weights from similarity scores
        attn = F.softmax(memory_scores / self.temperature, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: [batch, seq_len, d_model]
        context = torch.einsum('bsk,bskd->bsd', attn, values)

        return self.out_proj(context)


class MemoryAugmentedEncoderLayer(nn.Module):
    """
    Transformer encoder layer with memory augmentation.

    Combines:
    1. Standard self-attention on local context
    2. kNN-based memory retrieval for historical context
    3. Gating mechanism to balance local vs. memory information
    """

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

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )
        # Initialize gate bias (positive = prefer local, negative = prefer memory)
        if hasattr(self.gate[0], 'bias') and self.gate[0].bias is not None:
            nn.init.constant_(self.gate[0].bias, config.gate_bias)

        # Feed-forward network
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
        memory_scores: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            memory_values: Retrieved memory values [batch, seq_len, k, d_model]
            memory_scores: Memory similarity scores [batch, seq_len, k]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention_dict: Optional dictionary with attention weights
        """
        attention_dict = {}

        # Local self-attention
        local_out, local_attn = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(local_out))

        if return_attention:
            attention_dict['local'] = local_attn

        # Memory attention (if available)
        if memory_values is not None and memory_scores is not None:
            memory_out = self.memory_layer(x, memory_values, memory_scores)

            # Gated combination
            gate_input = torch.cat([local_out, memory_out], dim=-1)
            gate = self.gate(gate_input)

            combined = gate * local_out + (1 - gate) * memory_out
            x = self.norm2(x + self.dropout(combined))

            if return_attention:
                attention_dict['gate_values'] = gate.mean(dim=-1)  # [batch, seq_len]

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x, attention_dict if return_attention else None


class MemoryAugmentedTransformer(nn.Module):
    """
    Memory-Augmented Transformer for financial time series.

    Combines local attention with kNN retrieval from external memory
    to capture both recent and long-term patterns.

    Example:
        config = MemoryTransformerConfig(input_dim=6, d_model=64)
        model = MemoryAugmentedTransformer(config)

        x = torch.randn(2, 96, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 1] for regression
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embeddings
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = PositionalEncoding(
            config.d_model,
            max_len=config.seq_len * 2,
            dropout=config.dropout
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            MemoryAugmentedEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # Output head
        self.output_head = self._build_output_head(config)

    def _build_output_head(self, config: MemoryTransformerConfig) -> nn.Module:
        """Build output projection layer based on output type"""
        if config.output_type == OutputType.REGRESSION:
            return nn.Linear(config.d_model, 1)
        elif config.output_type == OutputType.DIRECTION:
            return nn.Linear(config.d_model, 3)  # Down, Neutral, Up
        elif config.output_type == OutputType.PORTFOLIO:
            return nn.Linear(config.d_model, 1)  # Position weight
        elif config.output_type == OutputType.QUANTILE:
            return nn.Linear(config.d_model, len(config.quantiles))
        return nn.Linear(config.d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to representations (without memory).

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Encoded representations [batch, seq_len, d_model]
        """
        x = self.token_embedding(x)
        x = self.position_encoding(x)

        for layer in self.encoder_layers:
            x, _ = layer(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        memory_values: Optional[torch.Tensor] = None,
        memory_scores: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            memory_values: Retrieved memory values [batch, seq_len, k, d_model]
            memory_scores: Memory similarity scores [batch, seq_len, k]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - predictions: Model predictions
                - representations: Encoded representations
                - last_hidden: Last position hidden state
                - attention_weights: Optional attention information
        """
        # Encode
        x = self.token_embedding(x)
        x = self.position_encoding(x)

        # Encoder layers with memory
        all_attention = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attn_dict = layer(
                x, memory_values, memory_scores, return_attention
            )
            if attn_dict:
                all_attention[f'layer_{i}'] = attn_dict

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

        result = {
            'predictions': predictions,
            'representations': x,
            'last_hidden': last_hidden,
        }

        if return_attention:
            result['attention_weights'] = all_attention

        # Add confidence for quantile regression
        if self.config.output_type == OutputType.QUANTILE:
            result['confidence'] = self._compute_confidence(predictions)

        return result

    def _compute_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence from quantile predictions (smaller interval = higher confidence)"""
        # predictions: [batch, n_quantiles]
        if predictions.size(-1) < 2:
            return torch.ones(predictions.size(0), device=predictions.device)

        interval_width = (predictions[:, -1] - predictions[:, 0]).abs()
        confidence = 1.0 / (1.0 + interval_width)
        return confidence

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            returns: Optional returns for portfolio loss

        Returns:
            Loss value
        """
        if self.config.output_type == OutputType.REGRESSION:
            return F.mse_loss(predictions.squeeze(), targets)

        elif self.config.output_type == OutputType.DIRECTION:
            # Convert continuous targets to classes
            target_classes = torch.zeros_like(targets, dtype=torch.long)
            target_classes[targets < -0.001] = 0  # Down
            target_classes[targets > 0.001] = 2   # Up
            target_classes[(targets >= -0.001) & (targets <= 0.001)] = 1  # Neutral
            # predictions are already softmaxed, use nll_loss with log probabilities
            log_probs = predictions.clamp_min(1e-9).log()
            return F.nll_loss(log_probs, target_classes)

        elif self.config.output_type == OutputType.PORTFOLIO:
            if returns is None:
                return F.mse_loss(predictions.squeeze(), targets)
            # Maximize portfolio return
            portfolio_return = (predictions.squeeze() * returns).mean()
            return -portfolio_return  # Negative because we minimize

        elif self.config.output_type == OutputType.QUANTILE:
            return self._quantile_loss(predictions, targets)

        return F.mse_loss(predictions.squeeze(), targets)

    def _quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute pinball loss for quantile regression"""
        losses = []
        targets = targets.unsqueeze(-1)  # [batch, 1]

        for i, q in enumerate(self.config.quantiles):
            errors = targets - predictions[:, i:i+1]
            losses.append(torch.max(q * errors, (q - 1) * errors))

        return torch.cat(losses, dim=-1).mean()


if __name__ == "__main__":
    # Test the model
    print("Testing Memory-Augmented Transformer...")

    config = MemoryTransformerConfig(
        input_dim=6,
        d_model=32,
        n_heads=4,
        n_layers=2,
        seq_len=48,
        n_neighbors=16
    )

    model = MemoryAugmentedTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass without memory
    x = torch.randn(2, 48, 6)
    output = model(x, return_attention=True)

    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Representations shape: {output['representations'].shape}")
    print(f"Last hidden shape: {output['last_hidden'].shape}")

    # Test forward pass with memory
    memory_values = torch.randn(2, 48, 16, 32)  # [batch, seq, k, d_model]
    memory_scores = torch.randn(2, 48, 16)       # [batch, seq, k]

    output_with_memory = model(x, memory_values, memory_scores)
    print(f"With memory - Predictions shape: {output_with_memory['predictions'].shape}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = MemoryAugmentedTransformer(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")

    print("\nAll tests passed!")
