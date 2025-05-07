# Memory-Augmented Transformer for Trading (Rust)

A Rust implementation of the Memory-Augmented Transformer for financial time series prediction.

## Features

- **External Memory Bank**: kNN-based memory with efficient retrieval
- **Transformer Model**: Multi-head attention with memory integration
- **Data Loaders**: Support for CSV (yfinance format) and Bybit API
- **Backtesting**: Full backtesting framework with walk-forward analysis

## Quick Start

### Building

```bash
cd rust
cargo build --release
```

### Running Examples

**Simple backtest with synthetic data:**
```bash
cargo run --example simple_backtest
```

**Live data from Bybit:**
```bash
cargo run --example bybit_live
```

## Usage

### Basic Usage

```rust
use memory_augmented_transformer::{
    MemoryAugmentedTransformer, MemoryTransformerConfig,
    ExternalMemoryBank, MemoryConfig,
    BacktestConfig, Backtester, TradingStrategy,
};

// Configure model
let model_config = MemoryTransformerConfig::default();

// Configure memory
let memory_config = MemoryConfig {
    dim: 64,
    max_entries: 10_000,
    k: 10,
    normalize: true,
};

// Configure backtesting
let backtest_config = BacktestConfig::default();

// Create strategy
let strategy = TradingStrategy::new(
    model_config,
    memory_config,
    backtest_config,
);
```

### Loading Data

```rust
use memory_augmented_transformer::data::{load_csv_data, BybitClient};

// From CSV (yfinance format)
let data = load_csv_data("AAPL.csv", "AAPL")?;

// From Bybit API
let client = BybitClient::new();
let data = client.get_klines("BTCUSDT", "15", 500).await?;
```

### Running Backtest

```rust
use memory_augmented_transformer::{Backtester, BacktestConfig};

let backtester = Backtester::new(BacktestConfig::default());
let result = backtester.run(&mut strategy, &data);

println!("Total Return: {:.2}%", result.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
```

## Module Structure

```text
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs          # Library entry point
│   ├── memory/         # External memory bank
│   │   └── mod.rs
│   ├── model/          # Transformer model
│   │   └── mod.rs
│   ├── data/           # Data loading
│   │   └── mod.rs
│   └── strategy/       # Trading strategy
│       └── mod.rs
└── examples/
    ├── simple_backtest.rs
    └── bybit_live.rs
```

## Configuration Options

### MemoryTransformerConfig

| Field | Default | Description |
|-------|---------|-------------|
| `input_dim` | 5 | Number of input features (OHLCV) |
| `d_model` | 64 | Hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 2 | Number of encoder layers |
| `d_ff` | 256 | Feedforward dimension |
| `seq_len` | 20 | Input sequence length |
| `memory_k` | 10 | Number of memory neighbors |

### BacktestConfig

| Field | Default | Description |
|-------|---------|-------------|
| `initial_capital` | 100,000 | Starting capital |
| `position_size` | 0.1 | Fraction of capital per trade |
| `stop_loss` | 0.02 | Stop loss percentage |
| `take_profit` | 0.04 | Take profit percentage |
| `fee_rate` | 0.001 | Trading fee rate |

## Tests

Run all tests:
```bash
cargo test
```

Run with output:
```bash
cargo test -- --nocapture
```

## Dependencies

- `ndarray`: N-dimensional array operations
- `tokio`: Async runtime for API calls
- `reqwest`: HTTP client for Bybit API
- `serde`: Serialization/deserialization
- `chrono`: Date/time handling

## License

MIT License
