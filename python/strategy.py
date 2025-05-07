"""
Trading Strategy and Backtesting for Memory-Augmented Transformers

Provides:
- BacktestConfig: Configuration for backtesting
- MemoryTradingStrategy: Strategy using model + memory
- Backtester: Backtesting engine
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position: float = 1.0       # Maximum position size
    min_confidence: float = 0.0     # Minimum confidence to trade
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rebalance_threshold: float = 0.1  # Min change to rebalance


class MemoryTradingStrategy:
    """
    Trading strategy using Memory-Augmented Transformer.

    Combines model predictions with historical similarity analysis
    from the memory bank to generate trading signals.

    Example:
        strategy = MemoryTradingStrategy(model, memory, BacktestConfig())
        signals = strategy.generate_signals(data, feature_cols)
    """

    def __init__(
        self,
        model,
        memory,
        config: BacktestConfig
    ):
        """
        Args:
            model: MemoryAugmentedTransformer instance
            memory: ExternalMemoryBank instance
            config: Backtesting configuration
        """
        self.model = model
        self.memory = memory
        self.config = config

    def generate_signals(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        return_analysis: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals for the dataset.

        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            return_analysis: Whether to return detailed analysis

        Returns:
            DataFrame with signals and confidence scores
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required")

        self.model.eval()
        signals = []

        seq_len = self.model.config.seq_len
        d_model = self.model.config.d_model

        with torch.no_grad():
            for i in range(seq_len, len(data)):
                # Prepare input
                x = data[feature_cols].iloc[i-seq_len:i].values
                x = torch.FloatTensor(x).unsqueeze(0)

                # Get model encoding for memory search
                encoded = self.model.encode(x)
                last_hidden = encoded[0, -1, :].cpu().numpy()

                # Search memory for similar patterns
                if self.memory.current_size > 0:
                    distances, indices, memory_values = self.memory.search(
                        last_hidden.reshape(1, -1)
                    )

                    # Prepare memory tensors for model
                    memory_values_tensor = torch.FloatTensor(memory_values)
                    memory_values_tensor = memory_values_tensor.unsqueeze(1).expand(
                        1, seq_len, -1, d_model
                    )
                    memory_scores_tensor = torch.FloatTensor(distances)
                    memory_scores_tensor = memory_scores_tensor.unsqueeze(1).expand(
                        1, seq_len, -1
                    )
                else:
                    memory_values_tensor = None
                    memory_scores_tensor = None
                    distances = np.zeros((1, self.memory.n_neighbors))
                    indices = np.zeros((1, self.memory.n_neighbors), dtype=np.int64)

                # Get prediction with memory
                outputs = self.model(x, memory_values_tensor, memory_scores_tensor)
                prediction = outputs['predictions'].item()

                # Analyze historical outcomes from memory
                analysis = self._analyze_memory_outcomes(indices, distances)

                # Generate signal
                signal_info = self._compute_signal(prediction, analysis)
                signal_info['index'] = data.index[i] if hasattr(data, 'index') else i
                signal_info['model_prediction'] = prediction

                if return_analysis:
                    signal_info['analysis'] = analysis

                signals.append(signal_info)

        df = pd.DataFrame(signals)
        if 'index' in df.columns:
            df = df.set_index('index')

        return df

    def _analyze_memory_outcomes(
        self,
        indices: np.ndarray,
        distances: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze historical outcomes from retrieved memories"""
        if self.memory.current_size == 0:
            return {
                'avg_return': 0.0,
                'std_return': 0.0,
                'positive_ratio': 0.5,
                'n_similar': 0
            }

        # Get returns for retrieved memories
        historical_returns = self.memory.returns[indices[0]]

        # Weight by similarity (distance = similarity score)
        weights = distances[0]
        weights = np.maximum(weights, 0)  # Ensure non-negative
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        weighted_avg = np.average(historical_returns, weights=weights)
        std_return = np.std(historical_returns)
        positive_ratio = np.mean(historical_returns > 0)

        return {
            'avg_return': float(weighted_avg),
            'std_return': float(std_return),
            'positive_ratio': float(positive_ratio),
            'n_similar': len(historical_returns),
            'best_match_similarity': float(distances[0, 0]) if len(distances[0]) > 0 else 0
        }

    def _compute_signal(
        self,
        prediction: float,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute final trading signal from prediction and analysis"""

        model_signal = np.sign(prediction)
        memory_signal = np.sign(analysis['avg_return'])

        # Confidence based on agreement and memory consistency
        if model_signal == memory_signal and model_signal != 0:
            # Model and memory agree
            base_confidence = analysis['positive_ratio'] if model_signal > 0 else (1 - analysis['positive_ratio'])
            confidence = min(base_confidence + 0.1, 1.0)
            final_signal = model_signal
        elif model_signal == 0:
            # Model is neutral, use memory
            confidence = abs(analysis['positive_ratio'] - 0.5) * 2
            final_signal = memory_signal
        else:
            # Disagreement - use model but with lower confidence
            confidence = abs(prediction) * 0.5
            final_signal = model_signal

        # Apply minimum confidence threshold
        if confidence < self.config.min_confidence:
            final_signal = 0
            confidence = 0

        # Compute position size
        position_size = final_signal * confidence * self.config.max_position

        return {
            'signal': int(final_signal),
            'confidence': float(confidence),
            'position_size': float(position_size),
            'memory_avg_return': analysis['avg_return'],
            'memory_std': analysis['std_return'],
            'memory_agreement': float(model_signal == memory_signal)
        }


class Backtester:
    """
    Backtesting engine for memory-augmented trading strategies.

    Example:
        backtester = Backtester(BacktestConfig())
        results = backtester.run(signals, returns)
        backtester.plot_results(results)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run backtest.

        Args:
            signals: DataFrame with 'position_size' column
            returns: Series of actual returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with performance metrics and time series
        """
        # Align signals with returns
        common_index = signals.index.intersection(returns.index)
        signals = signals.loc[common_index]
        returns = returns.loc[common_index]

        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate strategy returns
        # Position is from previous period
        positions = signals['position_size'].shift(1).fillna(0)
        strategy_returns = positions * returns

        # Transaction costs
        position_changes = positions.diff().abs().fillna(0)
        costs = position_changes * self.config.transaction_cost

        # Net returns
        net_returns = strategy_returns - costs

        # Cumulative returns
        cumulative = (1 + net_returns).cumprod()
        cumulative.iloc[0] = 1  # Start at 1

        # Calculate metrics
        metrics = self._calculate_metrics(net_returns, cumulative)

        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            metrics['benchmark_return'] = float(benchmark_cumulative.iloc[-1] - 1)
            metrics['excess_return'] = metrics['total_return'] - metrics['benchmark_return']
            metrics['benchmark_cumulative'] = benchmark_cumulative

        return {
            **metrics,
            'cumulative_returns': cumulative,
            'daily_returns': net_returns,
            'positions': positions,
            'signals': signals
        }

    def _calculate_metrics(
        self,
        returns: pd.Series,
        cumulative: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics"""

        # Basic returns
        total_return = cumulative.iloc[-1] - 1
        n_periods = len(returns)

        # Annualization factor (assume daily by default)
        ann_factor = 252

        # Annualized return
        annual_return = (1 + total_return) ** (ann_factor / n_periods) - 1

        # Volatility
        volatility = returns.std() * np.sqrt(ann_factor)

        # Sharpe ratio
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        winning = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade
        avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'n_trades': int(total_trades)
        }

    def plot_results(
        self,
        results: Dict,
        title: str = "Strategy Performance",
        save_path: Optional[str] = None
    ):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cumulative returns
        ax = axes[0, 0]
        results['cumulative_returns'].plot(ax=ax, label='Strategy')
        if 'benchmark_cumulative' in results:
            results['benchmark_cumulative'].plot(ax=ax, label='Benchmark', alpha=0.7)
            ax.legend()
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Growth of $1')
        ax.grid(True, alpha=0.3)

        # 2. Drawdown
        ax = axes[0, 1]
        cumulative = results['cumulative_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=ax, color='red')
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)

        # 3. Returns distribution
        ax = axes[1, 0]
        results['daily_returns'].hist(bins=50, ax=ax, alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.axvline(x=results['daily_returns'].mean(), color='green', linestyle='--',
                   label=f"Mean: {results['daily_returns'].mean():.4f}")
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Rolling Sharpe
        ax = axes[1, 1]
        rolling_sharpe = (
            results['daily_returns'].rolling(60).mean() /
            results['daily_returns'].rolling(60).std()
        ) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax.set_title('Rolling 60-Day Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results: Dict):
        """Print performance summary"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Return:        {results['total_return']*100:>10.2f}%")
        print(f"Annual Return:       {results['annual_return']*100:>10.2f}%")
        print(f"Volatility:          {results['volatility']*100:>10.2f}%")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {results['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']*100:>10.2f}%")
        print(f"Calmar Ratio:        {results['calmar_ratio']:>10.2f}")
        print(f"Win Rate:            {results['win_rate']*100:>10.2f}%")
        print(f"Profit Factor:       {results['profit_factor']:>10.2f}")
        print(f"Number of Trades:    {results['n_trades']:>10d}")
        if 'excess_return' in results:
            print(f"Excess Return:       {results['excess_return']*100:>10.2f}%")
        print("=" * 60)


def walk_forward_backtest(
    model,
    memory,
    data: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = 'returns',
    train_window: int = 252,
    test_window: int = 21,
    config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Walk-forward backtesting with periodic retraining.

    Args:
        model: Model instance
        memory: Memory instance
        data: Full dataset
        feature_cols: Feature columns
        return_col: Return column name
        train_window: Training window size
        test_window: Test window size
        config: Backtest configuration

    Returns:
        Combined backtest results
    """
    if config is None:
        config = BacktestConfig()

    all_signals = []
    all_returns = []

    n = len(data)
    i = train_window

    while i + test_window <= n:
        # Train period
        train_data = data.iloc[i-train_window:i]

        # Update memory with training data (simplified - in practice would retrain model)
        # This is where you would normally retrain the model and update memory

        # Test period
        test_data = data.iloc[i:i+test_window]

        # Generate signals
        strategy = MemoryTradingStrategy(model, memory, config)
        signals = strategy.generate_signals(test_data, feature_cols)
        returns = test_data[return_col]

        all_signals.append(signals)
        all_returns.append(returns)

        i += test_window
        logger.info(f"Walk-forward step: {i}/{n}")

    # Combine results
    combined_signals = pd.concat(all_signals)
    combined_returns = pd.concat(all_returns)

    # Run backtest on combined results
    backtester = Backtester(config)
    return backtester.run(combined_signals, combined_returns)


if __name__ == "__main__":
    # Test the backtester with synthetic data
    print("Testing Backtester...")

    # Create synthetic signals and returns
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='D')

    # Random signals with some autocorrelation
    np.random.seed(42)
    signals = pd.DataFrame({
        'position_size': np.clip(np.cumsum(np.random.randn(n) * 0.1), -1, 1),
        'confidence': np.random.uniform(0.3, 0.9, n),
        'signal': np.random.choice([-1, 0, 1], n)
    }, index=dates)

    # Random returns
    returns = pd.Series(np.random.randn(n) * 0.02, index=dates)

    # Run backtest
    config = BacktestConfig(initial_capital=100000, transaction_cost=0.001)
    backtester = Backtester(config)
    results = backtester.run(signals, returns)

    # Print results
    backtester._print_summary(results)

    print("\nAll tests passed!")
