import pandas as pd
import numpy as np
import ta


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the input DataFrame.
    
    Args:
        data (pd.DataFrame): Raw OHLCV data with columns: Open, High, Low, Close, Volume
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
    
    # Momentum Indicators
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    
    # Volatility Indicators
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Returns_Mean_{window}'] = df['Returns'].rolling(window=window).mean()
    
    # Remove rows with NaN values (due to rolling calculations)
    df = df.dropna()
    
    return df 