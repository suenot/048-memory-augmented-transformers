//! Simple Backtest Example
//!
//! Demonstrates how to use the Memory-Augmented Transformer for backtesting
//! with synthetic market data.

use memory_augmented_transformer::{
    BacktestConfig, Backtester, MemoryConfig, MemoryTransformerConfig, OutputType, TradingStrategy,
};
use memory_augmented_transformer::data::generate_synthetic_data;

fn main() {
    println!("=== Memory-Augmented Transformer Backtest ===\n");

    // Configure the model
    let model_config = MemoryTransformerConfig {
        input_dim: 5,      // OHLCV
        d_model: 32,       // Hidden dimension
        n_heads: 4,        // Attention heads
        n_layers: 2,       // Transformer layers
        d_ff: 128,         // Feedforward dimension
        seq_len: 20,       // Sequence length
        dropout: 0.1,
        output_type: OutputType::Direction,
        output_dim: 1,
        memory_dim: 32,
        memory_k: 10,      // Retrieve 10 nearest neighbors
        gate_bias: 0.0,
    };

    // Configure memory bank
    let memory_config = MemoryConfig {
        dim: 32,
        max_entries: 10_000,
        k: 10,
        normalize: true,
    };

    // Configure backtesting
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,        // 10% per trade
        stop_loss: Some(0.02),     // 2% stop loss
        take_profit: Some(0.04),   // 4% take profit
        fee_rate: 0.001,           // 0.1% trading fee
        slippage: 0.0005,          // 0.05% slippage
        seq_len: 20,
        horizon: 5,
        confidence_threshold: 0.55, // Only trade when >55% confident
    };

    // Generate synthetic market data
    println!("Generating synthetic market data...");
    let data = generate_synthetic_data(500, 0.02); // 500 bars, 2% volatility
    println!("Generated {} bars of data\n", data.len());

    // Create strategy
    let mut strategy = TradingStrategy::new(
        model_config,
        memory_config,
        backtest_config.clone(),
    );

    // Run backtest
    println!("Running backtest...");
    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&mut strategy, &data);

    // Print results
    println!("\n=== Backtest Results ===\n");
    println!("Total Return:     {:.2}%", result.total_return * 100.0);
    println!("Annual Return:    {:.2}%", result.annual_return * 100.0);
    println!("Sharpe Ratio:     {:.2}", result.sharpe_ratio);
    println!("Max Drawdown:     {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate:         {:.2}%", result.win_rate * 100.0);
    println!("Profit Factor:    {:.2}", result.profit_factor);
    println!("Total Trades:     {}", result.total_trades);
    println!("Avg Trade Return: {:.2}%", result.avg_trade_return * 100.0);

    // Show some individual trades
    if !result.trades.is_empty() {
        println!("\n=== Sample Trades ===\n");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!(
                "Trade {}: {} @ {:.2} -> {:.2}, PnL: ${:.2} ({:.2}%), Reason: {}",
                i + 1,
                if trade.size > 0.0 { "LONG" } else { "SHORT" },
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.return_pct * 100.0,
                trade.exit_reason
            );
        }
    }

    // Show equity curve summary
    if !result.equity_curve.is_empty() {
        let initial = result.equity_curve.first().unwrap();
        let final_ = result.equity_curve.last().unwrap();
        let min = result.equity_curve.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.equity_curve.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("\n=== Equity Curve Summary ===\n");
        println!("Initial: ${:.2}", initial);
        println!("Final:   ${:.2}", final_);
        println!("Min:     ${:.2}", min);
        println!("Max:     ${:.2}", max);
    }

    println!("\n=== Backtest Complete ===");
}
