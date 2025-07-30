import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def download_data(ticker: str, start_date: str, end_date: str, 
                 interval: str = "1d", data_source: str = "yfinance") -> pd.DataFrame:
    """
    Download financial data for a given ticker and date range with enhanced options.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'BTC-USD')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        data_source (str): Data source ('yfinance')
        
    Returns:
        pd.DataFrame: Raw financial data with OHLCV columns
    """
    try:
        if data_source.lower() == "yfinance":
            return _download_yfinance_data(ticker, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {data_source}. Use 'yfinance' for now.")
            
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {str(e)}")


def _download_yfinance_data(ticker: str, start_date: str, end_date: str, 
                           interval: str = "1d") -> pd.DataFrame:
    """
    Download data using yfinance with enhanced interval support.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date
        end_date (str): End date
        interval (str): Data interval
        
    Returns:
        pd.DataFrame: Financial data
    """
    ticker_obj = yf.Ticker(ticker)
    
    # Validate interval for yfinance
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval for yfinance: {interval}. Valid intervals: {valid_intervals}")
    
    data = ticker_obj.history(start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range")
    
    return data


def get_intraday_data(ticker: str, start_date: str, end_date: str, 
                     interval: str = "30m") -> pd.DataFrame:
    """
    Get intraday data optimized for 30-minute charts and microstructure analysis.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date
        end_date (str): End date
        interval (str): Intraday interval ('1m', '5m', '15m', '30m', '1h')
        
    Returns:
        pd.DataFrame: Intraday financial data
    """
    # Validate intraday interval
    valid_intraday_intervals = ['1m', '5m', '15m', '30m', '1h']
    if interval not in valid_intraday_intervals:
        raise ValueError(f"Invalid intraday interval: {interval}. Valid intervals: {valid_intraday_intervals}")
    
    return download_data(ticker, start_date, end_date, interval, "yfinance")


def validate_data_quality(data: pd.DataFrame, min_data_points: int = 100) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        data (pd.DataFrame): Financial data
        min_data_points (int): Minimum required data points
        
    Returns:
        Dict[str, Any]: Quality metrics
    """
    quality_metrics = {
        'total_rows': len(data),
        'missing_values': {},
        'duplicates': data.index.duplicated().sum(),
        'date_range': {
            'start': data.index.min(),
            'end': data.index.max(),
            'duration_days': (data.index.max() - data.index.min()).days
        },
        'volume_zeros': (data['Volume'] == 0).sum(),
        'price_anomalies': 0,
        'quality_score': 0.0
    }
    
    # Check for missing values
    for col in data.columns:
        quality_metrics['missing_values'][col] = data[col].isnull().sum()
    
    # Check for price anomalies (negative prices, extreme values)
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in data.columns:
            anomalies = (data[col] <= 0) | (data[col] > data[col].mean() * 10)
            quality_metrics['price_anomalies'] += anomalies.sum()
    
    # Calculate quality score
    total_expected = len(data) * len(data.columns)
    total_missing = sum(quality_metrics['missing_values'].values())
    quality_metrics['quality_score'] = 1 - (total_missing / total_expected)
    
    # Validate minimum data points
    if len(data) < min_data_points:
        raise ValueError(f"Insufficient data points: {len(data)} < {min_data_points}")
    
    return quality_metrics


def resample_data(data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample data to a different interval.
    
    Args:
        data (pd.DataFrame): Input data with datetime index
        target_interval (str): Target interval ('1H', '4H', '1D', etc.)
        
    Returns:
        pd.DataFrame: Resampled data
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Define aggregation rules
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Resample
    resampled = data.resample(target_interval).agg(agg_rules)
    
    return resampled.dropna()


def add_market_hours_filter(data: pd.DataFrame, market_hours: tuple = (9, 17)) -> pd.DataFrame:
    """
    Filter data to include only market hours.
    
    Args:
        data (pd.DataFrame): Input data with datetime index
        market_hours (tuple): Market hours (start_hour, end_hour)
        
    Returns:
        pd.DataFrame: Filtered data
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    start_hour, end_hour = market_hours
    filtered_data = data[
        (data.index.hour >= start_hour) & 
        (data.index.hour < end_hour) &
        (data.index.dayofweek < 5)  # Monday to Friday
    ]
    
    return filtered_data
