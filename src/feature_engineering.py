import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def fractional_differentiation(series: pd.Series, d: float = 0.5, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation to make series stationary while preserving memory.
    
    Args:
        series: Input time series
        d: Differentiation order (0 < d < 1)
        threshold: Threshold for stopping the expansion
        
    Returns:
        Fractionally differentiated series
    """
    if d >= 1:
        raise ValueError("d must be less than 1 for fractional differentiation")
    
    if len(series) < 10:
        # For very short series, return a simple difference
        return series.diff().dropna()
    
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    
    weights = np.array(weights[::-1])
    
    # Apply convolution
    res = np.convolve(series, weights, mode='valid')
    
    # Create index for the result
    # The result starts from index len(weights)-1
    start_idx = len(weights) - 1
    if start_idx >= len(series):
        # If series is too short, return simple difference
        return series.diff().dropna()
    
    result_index = series.index[start_idx:]
    
    # Ensure result length matches index length
    if len(res) > len(result_index):
        res = res[:len(result_index)]
    elif len(res) < len(result_index):
        # Pad with NaN if needed
        padding = len(result_index) - len(res)
        res = np.concatenate([res, np.full(padding, np.nan)])
    
    return pd.Series(res, index=result_index)


def calculate_vwap_enhanced(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate enhanced VWAP with rolling window and deviation metrics.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for VWAP calculation
        
    Returns:
        Enhanced VWAP series with deviation metrics
    """
    # Standard VWAP
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    
    # VWAP deviation
    vwap_deviation = (df['Close'] - vwap) / vwap
    
    # VWAP bands (similar to Bollinger Bands)
    vwap_std = vwap.rolling(window=window).std()
    vwap_upper = vwap + (2 * vwap_std)
    vwap_lower = vwap - (2 * vwap_std)
    
    # VWAP position (normalized position within bands)
    vwap_position = (df['Close'] - vwap_lower) / (vwap_upper - vwap_lower)
    
    return pd.Series({
        'VWAP': vwap,
        'VWAP_Deviation': vwap_deviation,
        'VWAP_Upper': vwap_upper,
        'VWAP_Lower': vwap_lower,
        'VWAP_Position': vwap_position
    })


def simulate_order_flow_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Simulate Order Flow Imbalance using volume and price action.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for calculation
        
    Returns:
        OFI simulation series
    """
    # Calculate volume-weighted price change
    price_change = df['Close'].diff()
    
    # Calculate net buying/selling pressure using pandas Series
    buying_pressure = pd.Series(np.where(price_change > 0, df['Volume'], 0), index=df.index)
    selling_pressure = pd.Series(np.where(price_change < 0, df['Volume'], 0), index=df.index)
    
    # OFI as net pressure
    ofi = (buying_pressure - selling_pressure).rolling(window=window).sum()
    
    # Normalize by total volume
    total_volume = df['Volume'].rolling(window=window).sum()
    ofi_normalized = ofi / total_volume
    
    # OFI momentum
    ofi_momentum = ofi_normalized.diff()
    
    return pd.Series({
        'OFI': ofi_normalized,
        'OFI_Momentum': ofi_momentum,
        'OFI_MA': ofi_normalized.rolling(window=window).mean(),
        'OFI_Std': ofi_normalized.rolling(window=window).std()
    })


def simulate_bid_ask_spread(df: pd.DataFrame, base_spread: float = 0.0001) -> pd.Series:
    """
    Simulate bid-ask spread based on volatility and volume.
    
    Args:
        df: DataFrame with OHLCV data
        base_spread: Base spread percentage
        
    Returns:
        Simulated spread series
    """
    # Calculate volatility-based spread
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Volume impact on spread
    volume_ma = df['Volume'].rolling(window=20).mean()
    volume_ratio = df['Volume'] / volume_ma
    
    # Spread increases with volatility and decreases with volume
    spread = base_spread * (1 + volatility * 10) * (1 - np.minimum(volume_ratio, 2) * 0.1)
    
    # Ensure spread is always positive
    spread = np.maximum(spread, base_spread * 0.1)
    
    # Spread momentum
    spread_momentum = spread.diff()
    
    # Spread bands
    spread_ma = spread.rolling(window=20).mean()
    spread_std = spread.rolling(window=20).std()
    
    # Spread position
    spread_position = (spread - spread_ma) / spread_std
    
    return pd.Series({
        'Bid_Ask_Spread': spread,
        'Spread_Momentum': spread_momentum,
        'Spread_MA': spread_ma,
        'Spread_Upper': spread_ma + 2 * spread_std,
        'Spread_Lower': spread_ma - 2 * spread_std,
        'Spread_Position': spread_position
    })


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to capture intraday and weekly patterns.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time-based features
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Hour of day (0-23)
    df['Hour'] = df.index.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['DayOfWeek'] = df.index.dayofweek
    
    # Month
    df['Month'] = df.index.month
    
    # Quarter
    df['Quarter'] = df.index.quarter
    
    # Time of day categories
    df['Is_Market_Open'] = ((df['Hour'] >= 9) & (df['Hour'] < 17)).astype(int)
    df['Is_Market_Close'] = ((df['Hour'] >= 15) & (df['Hour'] < 17)).astype(int)
    df['Is_Lunch_Hour'] = ((df['Hour'] >= 11) & (df['Hour'] < 13)).astype(int)
    
    # Day type
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Is_Monday'] = (df['DayOfWeek'] == 0).astype(int)
    df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding for hour and day
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add microstructure features including VWAP, OFI, and spread simulation.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with microstructure features
    """
    # Enhanced VWAP features
    vwap_features = calculate_vwap_enhanced(df)
    for col, series in vwap_features.items():
        df[f'VWAP_{col}'] = series
    
    # Order Flow Imbalance features
    ofi_features = simulate_order_flow_imbalance(df)
    for col, series in ofi_features.items():
        df[f'OFI_{col}'] = series
    
    # Bid-Ask Spread features
    spread_features = simulate_bid_ask_spread(df)
    for col, series in spread_features.items():
        df[f'Spread_{col}'] = series
    
    # Volume microstructure
    df['Volume_Price_Trend'] = df['Volume'] * df['Close'].pct_change()
    df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Volume_Std'] = df['Volume'].rolling(window=20).std()
    
    # Price microstructure
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Range'] = abs(df['Close'] - df['Open']) / df['Close']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
    
    return df


def add_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced technical indicators beyond the basic ones.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with advanced technical indicators
    """
    # Advanced momentum indicators
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    df['DMI_Plus'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
    df['DMI_Minus'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    
    # Advanced volatility indicators
    df['Keltner_Upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
    df['Keltner_Lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
    df['Keltner_Middle'] = ta.volatility.keltner_channel_mband(df['High'], df['Low'], df['Close'])
    
    # Advanced volume indicators
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    df['ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Price action patterns
    df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
    df['Hammer'] = ((df['Close'] - df['Low']) > 2 * (df['High'] - df['Close'])) & \
                   ((df['High'] - df['Low']) > 3 * (df['Close'] - df['Open']))
    df['Hammer'] = df['Hammer'].astype(int)
    
    return df


def add_fractional_features(df: pd.DataFrame, d_values: list = [0.3, 0.5, 0.7]) -> pd.DataFrame:
    """
    Add fractionally differentiated features for better stationarity.
    
    Args:
        df: DataFrame with price data
        d_values: List of differentiation orders
        
    Returns:
        DataFrame with fractional features
    """
    for d in d_values:
        try:
            # Fractional differentiation of price
            price_frac = fractional_differentiation(df['Close'], d=d)
            df[f'Price_FracDiff_{d}'] = price_frac
            
            # Fractional differentiation of volume
            volume_frac = fractional_differentiation(df['Volume'], d=d)
            df[f'Volume_FracDiff_{d}'] = volume_frac
            
            # Fractional differentiation of returns
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 10:  # Need enough data for fractional differentiation
                returns_frac = fractional_differentiation(returns, d=d)
                df[f'Returns_FracDiff_{d}'] = returns_frac
                
        except Exception as e:
            # Skip fractional differentiation if it fails
            print(f"Warning: Fractional differentiation with d={d} failed: {e}")
            continue
    
    return df


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced technical indicators function with microstructure and time-based features.
    
    Args:
        data (pd.DataFrame): Raw OHLCV data with columns: Open, High, Low, Close, Volume

    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    # Create a copy to avoid modifying the original
    df = data.copy()

    # Ensure we have the required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Basic Trend Indicators
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_Histogram"] = ta.trend.macd_diff(df["Close"])

    # Basic Momentum Indicators
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    df["Stoch_D"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"])
    df["Williams_R"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])

    # Basic Volatility Indicators
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"])
    df["BB_Width"] = ta.volatility.bollinger_wband(df["Close"])
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

    # Basic Volume Indicators
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["VWAP_Basic"] = ta.volume.volume_weighted_average_price(
        df["High"], df["Low"], df["Close"], df["Volume"]
    )

    # Price-based features
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["High_Low_Ratio"] = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]

    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f"Close_Std_{window}"] = df["Close"].rolling(window=window).std()
        df[f"Volume_Mean_{window}"] = df["Volume"].rolling(window=window).mean()
        df[f"Returns_Mean_{window}"] = df["Returns"].rolling(window=window).mean()

    # Add microstructure features
    df = add_microstructure_features(df)
    
    # Add advanced technical indicators
    df = add_advanced_technical_indicators(df)
    
    # Add time-based features
    df = add_time_based_features(df)
    
    # Add fractional differentiation features
    df = add_fractional_features(df)

    # Remove rows with NaN values (due to rolling calculations)
    df = df.dropna()

    return df
