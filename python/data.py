"""
Data Loading Utilities for Memory-Augmented Transformers

Provides:
- load_stock_data: Load stock data from yfinance
- load_bybit_data: Load crypto data from Bybit
- create_sequences: Create sequences for training
- FinancialDataset: PyTorch dataset for training
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def load_stock_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1h'
) -> Dict[str, pd.DataFrame]:
    """
    Load stock data from yfinance.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data frequency ('1m', '5m', '15m', '30m', '1h', '1d')

    Returns:
        Dictionary mapping symbols to DataFrames

    Example:
        data = load_stock_data(['AAPL', 'MSFT'], '2024-01-01', '2024-06-01')
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    data = {}

    for symbol in symbols:
        logger.info(f"Loading {symbol}...")
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue

            df = df.reset_index()

            # Normalize column names
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]

            # Rename datetime column
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            # Add features
            df = _add_features(df)

            data[symbol] = df.dropna()
            logger.info(f"Loaded {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def load_bybit_data(
    symbols: List[str],
    interval: str = '60',
    limit: int = 1000,
    category: str = 'linear'
) -> Dict[str, pd.DataFrame]:
    """
    Load cryptocurrency data from Bybit.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        interval: Candle interval in minutes ('1', '5', '15', '30', '60', '240', 'D')
        limit: Number of candles to fetch (max 1000)
        category: Market category ('linear', 'inverse', 'spot')

    Returns:
        Dictionary mapping symbols to DataFrames

    Example:
        data = load_bybit_data(['BTCUSDT', 'ETHUSDT'], interval='60', limit=500)
    """
    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        raise ImportError("pybit is required. Install with: pip install pybit")

    client = HTTP(testnet=False)
    data = {}

    for symbol in symbols:
        logger.info(f"Loading {symbol} from Bybit...")
        try:
            response = client.get_kline(
                category=category,
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if response['retCode'] != 0:
                logger.error(f"Bybit API error for {symbol}: {response['retMsg']}")
                continue

            df = pd.DataFrame(response['result']['list'])

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue

            # Bybit returns: timestamp, open, high, low, close, volume, turnover
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

            # Sort by timestamp (Bybit returns newest first)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Add features
            df = _add_features(df)

            data[symbol] = df.dropna()
            logger.info(f"Loaded {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features to OHLCV data"""
    df = df.copy()

    # Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'] / df['volume_ma']
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(20).std()

    # Price features
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # Moving averages
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_ratio'] = df['ma_5'] / df['ma_20']

    # RSI
    df['rsi'] = _calculate_rsi(df['close'], 14)

    return df


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()

    rs = avg_gains / (avg_losses + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi / 100  # Normalize to [0, 1]


def create_sequences(
    data: pd.DataFrame,
    seq_len: int = 96,
    horizon: int = 1,
    features: Optional[List[str]] = None,
    target: str = 'returns'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training.

    Args:
        data: DataFrame with OHLCV and features
        seq_len: Sequence length (lookback window)
        horizon: Prediction horizon
        features: Feature columns to use (default: standard set)
        target: Target column

    Returns:
        X: [n_samples, seq_len, n_features]
        y: [n_samples,] (target values)

    Example:
        X, y = create_sequences(df, seq_len=96, horizon=1)
    """
    if features is None:
        features = [
            'returns', 'volatility', 'volume_change',
            'price_range', 'price_position', 'rsi'
        ]

    # Filter to available features
    available_features = [f for f in features if f in data.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing features: {missing}")

    X, y = [], []

    values = data[available_features].values
    targets = data[target].values

    for i in range(seq_len, len(data) - horizon):
        X.append(values[i-seq_len:i])
        y.append(targets[i + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_multiasset_sequences(
    data: Dict[str, pd.DataFrame],
    seq_len: int = 96,
    horizon: int = 1,
    features: Optional[List[str]] = None,
    target: str = 'returns'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create multi-asset sequences for training.

    Args:
        data: Dictionary of DataFrames by symbol
        seq_len: Sequence length
        horizon: Prediction horizon
        features: Feature columns
        target: Target column

    Returns:
        X: [n_samples, n_assets, seq_len, n_features]
        y: [n_samples, n_assets] (target values for each asset)
        symbols: List of symbol names
    """
    if features is None:
        features = ['returns', 'volatility', 'volume_change', 'price_range', 'rsi']

    symbols = list(data.keys())

    # Find common timestamps
    common_index = None
    for symbol, df in data.items():
        if common_index is None:
            common_index = set(df['timestamp'])
        else:
            common_index &= set(df['timestamp'])

    common_index = sorted(common_index)
    logger.info(f"Common timestamps: {len(common_index)}")

    if len(common_index) < seq_len + horizon:
        raise ValueError("Not enough common timestamps for sequences")

    # Align all dataframes
    aligned = {}
    for symbol, df in data.items():
        df_aligned = df[df['timestamp'].isin(common_index)].sort_values('timestamp')
        aligned[symbol] = df_aligned

    X, y = [], []
    n_features = len(features)

    for i in range(seq_len, len(common_index) - horizon):
        x_sample = []
        y_sample = []

        for symbol in symbols:
            df = aligned[symbol]
            available_features = [f for f in features if f in df.columns]
            x_asset = df[available_features].iloc[i-seq_len:i].values
            y_asset = df[target].iloc[i + horizon - 1]

            x_sample.append(x_asset)
            y_sample.append(y_asset)

        X.append(np.array(x_sample))
        y.append(np.array(y_sample))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), symbols


class FinancialDataset:
    """
    PyTorch-compatible dataset for financial data.

    Example:
        dataset = FinancialDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: Features [n_samples, seq_len, n_features]
            y: Targets [n_samples,]
            returns: Optional returns for portfolio optimization
        """
        self.X = X
        self.y = y
        self.returns = returns

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple:
        if self.returns is not None:
            return self.X[idx], self.y[idx], self.returns[idx]
        return self.X[idx], self.y[idx]

    def to_tensors(self):
        """Convert to PyTorch tensors"""
        try:
            import torch
            X = torch.FloatTensor(self.X)
            y = torch.FloatTensor(self.y)
            if self.returns is not None:
                returns = torch.FloatTensor(self.returns)
                return X, y, returns
            return X, y
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion")


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Split data into train/validation/test sets (time-based, no shuffle).

    Args:
        X: Features
        y: Targets
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading utilities...")

    # Test with synthetic data (no API calls)
    print("\nCreating synthetic test data...")

    # Create synthetic OHLCV data
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(n) * 0.1,
        'high': price + np.abs(np.random.randn(n) * 0.5),
        'low': price - np.abs(np.random.randn(n) * 0.5),
        'close': price,
        'volume': np.random.exponential(1000, n)
    })

    # Add features
    df = _add_features(df)
    print(f"Created DataFrame with {len(df)} rows and columns: {list(df.columns)}")

    # Create sequences
    X, y = create_sequences(df.dropna(), seq_len=48, horizon=1)
    print(f"Sequences: X shape = {X.shape}, y shape = {y.shape}")

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Test dataset
    dataset = FinancialDataset(X_train, y_train)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shapes: X = {sample[0].shape}, y = {sample[1].shape}")

    print("\nAll tests passed!")
