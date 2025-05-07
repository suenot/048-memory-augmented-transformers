//! Memory-Augmented Transformer for Trading
//!
//! This crate implements the Memory-Augmented Transformer architecture
//! based on the "Memorizing Transformers" paper (arXiv:2203.08913).
//!
//! # Features
//!
//! - External memory bank with kNN retrieval
//! - Transformer encoder with memory attention
//! - Data loaders for yfinance CSV and Bybit API
//! - Trading strategy and backtesting utilities
//!
//! # Example
//!
//! ```rust,no_run
//! use memory_augmented_transformer::{
//!     MemoryAugmentedTransformer, MemoryTransformerConfig,
//!     ExternalMemoryBank, MemoryConfig,
//! };
//!
//! // Create configuration
//! let config = MemoryTransformerConfig::default();
//!
//! // Initialize model
//! let model = MemoryAugmentedTransformer::new(config);
//!
//! // Create memory bank
//! let memory_config = MemoryConfig::default();
//! let memory = ExternalMemoryBank::new(memory_config);
//! ```

pub mod memory;
pub mod model;
pub mod data;
pub mod strategy;

// Re-exports
pub use memory::{ExternalMemoryBank, MemoryConfig, MemoryEntry};
pub use model::{MemoryAugmentedTransformer, MemoryTransformerConfig, OutputType};
pub use data::{OHLCVBar, MarketData, load_csv_data, BybitClient};
pub use strategy::{BacktestConfig, TradingStrategy, Backtester, BacktestResult};
