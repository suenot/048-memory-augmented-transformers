//! Bybit Live Data Example
//!
//! Demonstrates how to fetch live cryptocurrency data from Bybit
//! and use the Memory-Augmented Transformer for prediction.

use memory_augmented_transformer::{
    BacktestConfig, Backtester, BybitClient, MemoryConfig, MemoryTransformerConfig,
    OutputType, TradingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory-Augmented Transformer with Bybit Data ===\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Fetch recent klines for BTCUSDT
    println!("Fetching BTCUSDT data from Bybit...");
    let symbol = "BTCUSDT";
    let interval = "15";  // 15-minute candles
    let limit = 500;

    let data = match client.get_klines(symbol, interval, limit).await {
        Ok(data) => data,
        Err(e) => {
            println!("Error fetching data: {}", e);
            println!("\nNote: This example requires internet access to the Bybit API.");
            println!("If you're running offline, use the simple_backtest example instead.");
            return Ok(());
        }
    };

    println!("Fetched {} bars of {} data\n", data.len(), symbol);

    // Show some recent data
    println!("=== Recent Bars ===\n");
    for bar in data.bars.iter().rev().take(5) {
        let dt = chrono::DateTime::from_timestamp_millis(bar.timestamp)
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        println!(
            "{}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            dt, bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }

    // Configure model
    let model_config = MemoryTransformerConfig {
        input_dim: 5,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 30,
        dropout: 0.1,
        output_type: OutputType::Direction,
        output_dim: 1,
        memory_dim: 64,
        memory_k: 10,
        gate_bias: 0.0,
    };

    let memory_config = MemoryConfig {
        dim: 64,
        max_entries: 10_000,
        k: 10,
        normalize: true,
    };

    let backtest_config = BacktestConfig {
        initial_capital: 10_000.0,
        position_size: 0.05,       // 5% per trade (crypto is volatile!)
        stop_loss: Some(0.015),    // 1.5% stop loss
        take_profit: Some(0.03),   // 3% take profit
        fee_rate: 0.0004,          // Bybit taker fee
        slippage: 0.001,           // 0.1% slippage
        seq_len: 30,
        horizon: 4,                // Predict 4 bars ahead (1 hour for 15m candles)
        confidence_threshold: 0.6,
    };

    // Create strategy
    let mut strategy = TradingStrategy::new(
        model_config,
        memory_config,
        backtest_config.clone(),
    );

    // Run backtest
    println!("\n=== Running Backtest ===\n");
    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&mut strategy, &data);

    // Print results
    println!("=== Backtest Results for {} ===\n", symbol);
    println!("Total Return:     {:.2}%", result.total_return * 100.0);
    println!("Annual Return:    {:.2}%", result.annual_return * 100.0);
    println!("Sharpe Ratio:     {:.2}", result.sharpe_ratio);
    println!("Max Drawdown:     {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate:         {:.2}%", result.win_rate * 100.0);
    println!("Profit Factor:    {:.2}", result.profit_factor);
    println!("Total Trades:     {}", result.total_trades);

    // Show memory statistics
    println!("\n=== Memory Bank Statistics ===\n");
    println!("Memory entries:   {}", strategy.memory().len());

    // Generate current prediction
    if data.len() >= 30 {
        let features = data.to_features();
        let window = features.slice(ndarray::s![features.nrows() - 30.., ..]).to_owned();
        let (signal, confidence) = strategy.generate_signal(&window);

        println!("\n=== Current Prediction ===\n");
        println!("Signal:     {:?}", signal);
        println!("Confidence: {:.1}%", confidence * 100.0);

        let current_price = data.bars.last().unwrap().close;
        println!("Current price: ${:.2}", current_price);

        match signal {
            memory_augmented_transformer::strategy::Signal::Long => {
                println!("Prediction: Price likely to go UP");
            }
            memory_augmented_transformer::strategy::Signal::Short => {
                println!("Prediction: Price likely to go DOWN");
            }
            memory_augmented_transformer::strategy::Signal::Hold => {
                println!("Prediction: Uncertain - HOLD");
            }
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
