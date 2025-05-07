//! Trading Strategy and Backtesting
//!
//! Provides trading strategy implementation using the Memory-Augmented Transformer
//! and a backtesting framework for evaluation.

use crate::data::MarketData;
use crate::memory::{ExternalMemoryBank, MemoryConfig};
use crate::model::{MemoryAugmentedTransformer, MemoryTransformerConfig};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f32,
    /// Position sizing (fraction of capital)
    pub position_size: f32,
    /// Stop loss percentage
    pub stop_loss: Option<f32>,
    /// Take profit percentage
    pub take_profit: Option<f32>,
    /// Trading fee percentage
    pub fee_rate: f32,
    /// Slippage percentage
    pub slippage: f32,
    /// Sequence length for model input
    pub seq_len: usize,
    /// Prediction horizon
    pub horizon: usize,
    /// Minimum confidence threshold for trading
    pub confidence_threshold: f32,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            fee_rate: 0.001,
            slippage: 0.0005,
            seq_len: 20,
            horizon: 5,
            confidence_threshold: 0.6,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f32,
    /// Exit price
    pub exit_price: f32,
    /// Position size (positive for long, negative for short)
    pub size: f32,
    /// Profit/loss
    pub pnl: f32,
    /// Return percentage
    pub return_pct: f32,
    /// Model confidence at entry
    pub confidence: f32,
    /// Exit reason
    pub exit_reason: String,
}

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Buy/long signal
    Long,
    /// Sell/short signal
    Short,
    /// No action
    Hold,
}

/// Trading strategy using Memory-Augmented Transformer
pub struct TradingStrategy {
    model: MemoryAugmentedTransformer,
    memory: ExternalMemoryBank,
    config: BacktestConfig,
    means: Option<Vec<f32>>,
    stds: Option<Vec<f32>>,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(
        model_config: MemoryTransformerConfig,
        memory_config: MemoryConfig,
        backtest_config: BacktestConfig,
    ) -> Self {
        Self {
            model: MemoryAugmentedTransformer::new(model_config),
            memory: ExternalMemoryBank::new(memory_config),
            config: backtest_config,
            means: None,
            stds: None,
        }
    }

    /// Fit normalization parameters from training data
    pub fn fit(&mut self, data: &MarketData) {
        let (_, means, stds) = data.normalize();
        self.means = Some(means);
        self.stds = Some(stds);
    }

    /// Normalize input features
    pub fn normalize_features(&self, features: &Array2<f32>) -> Array2<f32> {
        match (&self.means, &self.stds) {
            (Some(means), Some(stds)) => {
                let mut normalized = features.clone();
                for j in 0..features.ncols() {
                    // Guard against zero-variance features to avoid inf/NaN
                    let denom = stds[j].abs().max(1e-8);
                    for i in 0..features.nrows() {
                        normalized[[i, j]] = (features[[i, j]] - means[j]) / denom;
                    }
                }
                normalized
            }
            _ => features.clone(),
        }
    }

    /// Generate trading signal for current market state
    pub fn generate_signal(&self, window: &Array2<f32>) -> (Signal, f32) {
        let normalized = self.normalize_features(window);
        let (direction, confidence) = self.model.predict_direction(&normalized, Some(&self.memory));

        if confidence < self.config.confidence_threshold {
            return (Signal::Hold, confidence);
        }

        if direction {
            (Signal::Long, confidence)
        } else {
            (Signal::Short, confidence)
        }
    }

    /// Update memory with new pattern
    pub fn update_memory(&mut self, window: &Array2<f32>, future_return: f32, timestamp: i64) {
        let normalized = self.normalize_features(window);
        let embedding = self.model.get_embedding(&normalized);
        let _ = self.memory.add(embedding.to_vec(), timestamp, Some(future_return));
    }

    /// Get model reference
    pub fn model(&self) -> &MemoryAugmentedTransformer {
        &self.model
    }

    /// Get memory reference
    pub fn memory(&self) -> &ExternalMemoryBank {
        &self.memory
    }
}

/// Backtest result metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f32,
    /// Annualized return
    pub annual_return: f32,
    /// Sharpe ratio
    pub sharpe_ratio: f32,
    /// Maximum drawdown
    pub max_drawdown: f32,
    /// Win rate
    pub win_rate: f32,
    /// Profit factor
    pub profit_factor: f32,
    /// Total number of trades
    pub total_trades: usize,
    /// Average trade return
    pub avg_trade_return: f32,
    /// All trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f32>,
    /// Timestamps for equity curve
    pub timestamps: Vec<i64>,
}

/// Backtester for evaluating trading strategies
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on market data
    pub fn run(&self, strategy: &mut TradingStrategy, data: &MarketData) -> BacktestResult {
        let mut capital = self.config.initial_capital;
        let mut position: Option<(f32, f32, i64, f32)> = None; // (size, entry_price, entry_time, confidence)
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f32> = Vec::new();
        let mut timestamps: Vec<i64> = Vec::new();

        let features = data.to_features();
        let seq_len = self.config.seq_len;
        let horizon = self.config.horizon;

        if features.nrows() < seq_len + horizon {
            return self.empty_result();
        }

        // Fit normalization on data available before the first trade to avoid look-ahead bias
        let fit_end = seq_len.min(data.bars.len());
        let fit_data = MarketData::new(data.symbol.clone(), data.bars[..fit_end].to_vec());
        strategy.fit(&fit_data);

        for i in seq_len..(features.nrows() - horizon) {
            let window = features.slice(ndarray::s![i - seq_len..i, ..]).to_owned();
            let current_bar = &data.bars[i];
            let current_price = current_bar.close;

            // Record equity
            let unrealized_pnl = if let Some((size, entry_price, _, _)) = &position {
                size * (current_price - entry_price)
            } else {
                0.0
            };
            equity_curve.push(capital + unrealized_pnl);
            timestamps.push(current_bar.timestamp);

            // Check exit conditions for existing position
            if let Some((size, entry_price, entry_time, conf)) = position.take() {
                let mut should_exit = false;
                let mut exit_reason = String::new();

                let pnl_pct = if size > 0.0 {
                    (current_price - entry_price) / entry_price
                } else {
                    (entry_price - current_price) / entry_price
                };

                // Check stop loss
                if let Some(stop_loss) = self.config.stop_loss {
                    if pnl_pct < -stop_loss {
                        should_exit = true;
                        exit_reason = "stop_loss".to_string();
                    }
                }

                // Check take profit
                if let Some(take_profit) = self.config.take_profit {
                    if pnl_pct > take_profit {
                        should_exit = true;
                        exit_reason = "take_profit".to_string();
                    }
                }

                // Check holding period
                if i >= data.bars.len() - horizon - 1 {
                    should_exit = true;
                    exit_reason = "end_of_data".to_string();
                }

                if should_exit {
                    // Calculate PnL with fees and slippage
                    let exit_price = current_price * (1.0 - self.config.slippage * size.signum());
                    let raw_pnl = size * (exit_price - entry_price);
                    let fees = (size.abs() * entry_price + size.abs() * exit_price) * self.config.fee_rate;
                    let net_pnl = raw_pnl - fees;

                    capital += net_pnl;

                    trades.push(Trade {
                        entry_time,
                        exit_time: current_bar.timestamp,
                        entry_price,
                        exit_price,
                        size,
                        pnl: net_pnl,
                        return_pct: net_pnl / (size.abs() * entry_price),
                        confidence: conf,
                        exit_reason,
                    });
                } else {
                    position = Some((size, entry_price, entry_time, conf));
                }
            }

            // Generate signal if no position
            if position.is_none() {
                let (signal, confidence) = strategy.generate_signal(&window);

                match signal {
                    Signal::Long => {
                        let size = (capital * self.config.position_size) / current_price;
                        let entry_price = current_price * (1.0 + self.config.slippage);
                        position = Some((size, entry_price, current_bar.timestamp, confidence));
                    }
                    Signal::Short => {
                        let size = -(capital * self.config.position_size) / current_price;
                        let entry_price = current_price * (1.0 - self.config.slippage);
                        position = Some((size, entry_price, current_bar.timestamp, confidence));
                    }
                    Signal::Hold => {}
                }
            }

            // Update memory with future return
            if i + horizon < data.bars.len() {
                let future_return = (data.bars[i + horizon].close - current_price) / current_price;
                strategy.update_memory(&window, future_return, current_bar.timestamp);
            }
        }

        // Close any remaining position
        if let Some((size, entry_price, entry_time, conf)) = position {
            let last_bar = data.bars.last().unwrap();
            let exit_price = last_bar.close * (1.0 - self.config.slippage * size.signum());
            let raw_pnl = size * (exit_price - entry_price);
            let fees = (size.abs() * entry_price + size.abs() * exit_price) * self.config.fee_rate;
            let net_pnl = raw_pnl - fees;

            capital += net_pnl;

            trades.push(Trade {
                entry_time,
                exit_time: last_bar.timestamp,
                entry_price,
                exit_price,
                size,
                pnl: net_pnl,
                return_pct: net_pnl / (size.abs() * entry_price),
                confidence: conf,
                exit_reason: "end_of_backtest".to_string(),
            });
        }

        // Calculate metrics
        self.calculate_metrics(capital, trades, equity_curve, timestamps)
    }

    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            avg_trade_return: 0.0,
            trades: vec![],
            equity_curve: vec![self.config.initial_capital],
            timestamps: vec![],
        }
    }

    fn calculate_metrics(
        &self,
        final_capital: f32,
        trades: Vec<Trade>,
        equity_curve: Vec<f32>,
        timestamps: Vec<i64>,
    ) -> BacktestResult {
        let total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital;

        // Calculate max drawdown
        let mut max_equity = self.config.initial_capital;
        let mut max_drawdown = 0.0_f32;
        for &equity in &equity_curve {
            max_equity = max_equity.max(equity);
            let drawdown = (max_equity - equity) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calculate trade statistics
        let total_trades = trades.len();
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if total_trades > 0 {
            winning_trades.len() as f32 / total_trades as f32
        } else {
            0.0
        };

        let gross_profit: f32 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f32 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if total_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f32>() / total_trades as f32
        } else {
            0.0
        };

        // Calculate returns for Sharpe ratio
        let returns: Vec<f32> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f32>() / returns.len() as f32
        };

        let std_return = if returns.len() > 1 {
            let variance: f32 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f32>()
                / (returns.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let sharpe_ratio = if std_return > 0.0 {
            (mean_return / std_return) * (252.0_f32).sqrt() // Annualized
        } else {
            0.0
        };

        // Annualized return (assuming daily data, 252 trading days)
        let n_periods = equity_curve.len().max(1);
        let annual_return = ((1.0 + total_return).powf(252.0 / n_periods as f32)) - 1.0;

        BacktestResult {
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_return,
            trades,
            equity_curve,
            timestamps,
        }
    }
}

/// Walk-forward optimization/testing
pub fn walk_forward_backtest(
    data: &MarketData,
    model_config: MemoryTransformerConfig,
    memory_config: MemoryConfig,
    backtest_config: BacktestConfig,
    n_splits: usize,
    train_ratio: f32,
) -> Vec<BacktestResult> {
    let n = data.len();
    // Validate parameters to prevent divide-by-zero and invalid splits
    if n == 0 || n_splits == 0 || n_splits > n {
        return Vec::new();
    }
    if !(0.0..1.0).contains(&train_ratio) {
        return Vec::new();
    }
    let split_size = n / n_splits;
    let mut results = Vec::new();

    for i in 0..n_splits {
        let split_end = ((i + 1) * split_size).min(n);
        let split_data = MarketData::new(
            data.symbol.clone(),
            data.bars[..split_end].to_vec(),
        );

        let train_end = (split_data.len() as f32 * train_ratio) as usize;
        let train_data = MarketData::new(
            data.symbol.clone(),
            split_data.bars[..train_end].to_vec(),
        );
        let test_data = MarketData::new(
            data.symbol.clone(),
            split_data.bars[train_end..].to_vec(),
        );

        if test_data.len() < backtest_config.seq_len + backtest_config.horizon {
            continue;
        }

        // Create and train strategy
        let mut strategy = TradingStrategy::new(
            model_config.clone(),
            memory_config.clone(),
            backtest_config.clone(),
        );

        // Fit normalization on training data before populating memory
        strategy.fit(&train_data);

        // Populate memory from training data (use same normalization as backtest)
        let (train_seqs, train_targets) = train_data.create_sequences(
            backtest_config.seq_len,
            backtest_config.horizon,
        );

        for (seq, target) in train_seqs.iter().zip(train_targets.iter()) {
            // Normalize before embedding to ensure consistent scale with backtest
            let normalized = strategy.normalize_features(seq);
            let embedding = strategy.model().get_embedding(&normalized);
            let _ = strategy.memory.add(embedding.to_vec(), 0, Some(*target));
        }

        // Run backtest on test data
        let backtester = Backtester::new(backtest_config.clone());
        let result = backtester.run(&mut strategy, &test_data);
        results.push(result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;
    use crate::model::OutputType;

    #[test]
    fn test_trading_strategy() {
        let model_config = MemoryTransformerConfig {
            input_dim: 5,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            seq_len: 10,
            dropout: 0.0,
            output_type: OutputType::Direction,
            output_dim: 1,
            memory_dim: 16,
            memory_k: 5,
            gate_bias: 0.0,
        };

        let memory_config = MemoryConfig {
            dim: 16,
            max_entries: 100,
            k: 5,
            normalize: true,
        };

        let backtest_config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.1,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            fee_rate: 0.001,
            slippage: 0.0005,
            seq_len: 10,
            horizon: 3,
            confidence_threshold: 0.5,
        };

        let strategy = TradingStrategy::new(model_config, memory_config, backtest_config);
        assert!(strategy.memory().is_empty());
    }

    #[test]
    fn test_backtester() {
        let model_config = MemoryTransformerConfig {
            input_dim: 5,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            seq_len: 10,
            dropout: 0.0,
            output_type: OutputType::Direction,
            output_dim: 1,
            memory_dim: 16,
            memory_k: 5,
            gate_bias: 0.0,
        };

        let memory_config = MemoryConfig {
            dim: 16,
            max_entries: 100,
            k: 5,
            normalize: true,
        };

        let backtest_config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.1,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            fee_rate: 0.001,
            slippage: 0.0005,
            seq_len: 10,
            horizon: 3,
            confidence_threshold: 0.3, // Lower threshold for testing
        };

        let mut strategy = TradingStrategy::new(
            model_config,
            memory_config,
            backtest_config.clone(),
        );

        let data = generate_synthetic_data(100, 0.02);
        let backtester = Backtester::new(backtest_config);
        let result = backtester.run(&mut strategy, &data);

        // Basic sanity checks
        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0);
        assert!(result.max_drawdown <= 1.0);
    }
}
