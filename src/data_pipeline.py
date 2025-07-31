import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any, List
import warnings
from datetime import datetime, timedelta
import numpy as np

warnings.filterwarnings("ignore")


def download_multi_timeframe_data(
    ticker: str,
    start_date: str,
    end_date: str,
    timeframes: List[str] = ["1h", "4h", "1d", "1w"],
    data_source: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """
    Download financial data for multiple timeframes with enhanced data quality checks.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'BTC-USD')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        timeframes (List[str]): List of timeframes to download
        data_source (str): Data source ('yfinance')

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with timeframe as key and data as value
    """
    results = {}

    for timeframe in timeframes:
        print(f"📊 Downloading {timeframe} data for {ticker}...")
        try:
            data = download_data(ticker, start_date, end_date, timeframe, data_source)

            # Validate data quality and completeness
            quality_metrics = validate_data_quality(data, min_data_points=100)

            # Check for data gaps and fill if necessary
            data = fill_data_gaps(data, timeframe)

            # Final validation after gap filling
            final_quality = validate_data_quality(data, min_data_points=100)

            print(
                f"✅ {timeframe} data: {len(data)} points, quality score: {final_quality['quality_score']:.3f}"
            )

            results[timeframe] = data

        except Exception as e:
            print(f"❌ Failed to download {timeframe} data: {str(e)}")
            continue

    return results


def download_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    data_source: str = "yfinance",
) -> pd.DataFrame:
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
            raise ValueError(
                f"Unsupported data source: {data_source}. Use 'yfinance' for now."
            )

    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {str(e)}")


def _extend_start_date(start_date: str, interval: str) -> str:
    """
    Extend the start date to ensure we get enough data, especially for longer timeframes.
    Respects Yahoo Finance limitations for different intervals.

    Args:
        start_date (str): Original start date
        interval (str): Data interval

    Returns:
        str: Extended start date
    """
    start_dt = pd.to_datetime(start_date)
    current_date = datetime.now()

    # Yahoo Finance limitations:
    # - 1h data: only available for last 730 days (2 years)
    # - 1d data: available for much longer periods
    # - 1w data: available for much longer periods

    if interval in ["1h", "4h"]:
        # For intraday data, limit to 730 days (Yahoo Finance limit)
        max_days = 730
        extended_start = current_date - timedelta(days=max_days)
        # Use the later of the calculated start or requested start
        extended_start = max(extended_start, start_dt - timedelta(days=180))
    elif interval == "1d":
        # For daily data, we can go back much further
        extended_start = start_dt - timedelta(days=365)
    elif interval == "1w":
        # For weekly data, we can go back much further
        extended_start = start_dt - timedelta(days=730)
    else:
        extended_start = start_dt - timedelta(days=365)

    return extended_start.strftime("%Y-%m-%d")


def _download_yfinance_data(
    ticker: str, start_date: str, end_date: str, interval: str = "1d"
) -> pd.DataFrame:
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

    # Map timeframes to yfinance intervals
    interval_mapping = {
        "1h": "1h",
        "4h": "1h",  # yfinance doesn't support 4h directly, we'll resample
        "1d": "1d",
        "1w": "1wk",
    }

    yf_interval = interval_mapping.get(interval, interval)

    # Validate interval for yfinance
    valid_intervals = [
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ]
    if yf_interval not in valid_intervals:
        raise ValueError(
            f"Invalid interval for yfinance: {yf_interval}. Valid intervals: {valid_intervals}"
        )

    # Download data with extended range to ensure we get enough data
    extended_start = _extend_start_date(start_date, interval)

    try:
        data = ticker_obj.history(
            start=extended_start, end=end_date, interval=yf_interval
        )

        if data.empty:
            raise ValueError(
                f"No data found for ticker {ticker} in the specified date range"
            )

        # Filter to requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Handle timezone issues
        if data.index.tz is not None:
            start_dt = start_dt.tz_localize(data.index.tz)
            end_dt = end_dt.tz_localize(data.index.tz)

        data = data[(data.index >= start_dt) & (data.index <= end_dt)]

        # Resample to 4h if needed
        if interval == "4h":
            data = resample_data(data, "4H")

        return data

    except Exception as e:
        # If the original request fails, try with a shorter period for intraday data
        if interval in ["1h", "4h"]:
            print(f"⚠️  Retrying with shorter period for {interval} data...")
            # Try with last 60 days for intraday data
            retry_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            data = ticker_obj.history(
                start=retry_start, end=end_date, interval=yf_interval
            )

            if data.empty:
                raise ValueError(
                    f"No data found for ticker {ticker} even with shorter period"
                )

            # Filter to requested date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Handle timezone issues
            if data.index.tz is not None:
                start_dt = start_dt.tz_localize(data.index.tz)
                end_dt = end_dt.tz_localize(data.index.tz)

            data = data[(data.index >= start_dt) & (data.index <= end_dt)]

            # Resample to 4h if needed
            if interval == "4h":
                data = resample_data(data, "4H")

            return data
        else:
            raise e


def fill_data_gaps(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Fill gaps in the data using forward fill and interpolation.

    Args:
        data (pd.DataFrame): Input data
        timeframe (str): Data timeframe

    Returns:
        pd.DataFrame: Data with gaps filled
    """
    if data.empty:
        return data

    # Create a complete date range
    start_date = data.index.min()
    end_date = data.index.max()

    if timeframe == "1h":
        freq = "H"
    elif timeframe == "4h":
        freq = "4H"
    elif timeframe == "1d":
        freq = "D"
    elif timeframe == "1w":
        freq = "W"
    else:
        freq = "D"

    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Reindex data to complete range
    data_reindexed = data.reindex(complete_range)

    # Forward fill OHLC data
    ohlc_cols = ["Open", "High", "Low", "Close"]
    for col in ohlc_cols:
        if col in data_reindexed.columns:
            data_reindexed[col] = data_reindexed[col].fillna(method="ffill")

    # Fill volume with 0 for missing periods
    if "Volume" in data_reindexed.columns:
        data_reindexed["Volume"] = data_reindexed["Volume"].fillna(0)

    return data_reindexed


def get_intraday_data(
    ticker: str, start_date: str, end_date: str, interval: str = "30m"
) -> pd.DataFrame:
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
    valid_intraday_intervals = ["1m", "5m", "15m", "30m", "1h"]
    if interval not in valid_intraday_intervals:
        raise ValueError(
            f"Invalid intraday interval: {interval}. Valid intervals: {valid_intraday_intervals}"
        )

    return download_data(ticker, start_date, end_date, interval, "yfinance")


def validate_data_quality(
    data: pd.DataFrame, min_data_points: int = 100
) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.

    Args:
        data (pd.DataFrame): Financial data
        min_data_points (int): Minimum required data points

    Returns:
        Dict[str, Any]: Quality metrics
    """
    quality_metrics = {
        "total_rows": len(data),
        "missing_values": {},
        "duplicates": data.index.duplicated().sum(),
        "date_range": {
            "start": data.index.min(),
            "end": data.index.max(),
            "duration_days": (data.index.max() - data.index.min()).days,
        },
        "volume_zeros": (data["Volume"] == 0).sum() if "Volume" in data.columns else 0,
        "price_anomalies": 0,
        "quality_score": 0.0,
        "data_gaps": 0,
    }

    # Check for missing values
    for col in data.columns:
        quality_metrics["missing_values"][col] = data[col].isnull().sum()

    # Check for price anomalies (negative prices, extreme values)
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in data.columns:
            anomalies = (data[col] <= 0) | (data[col] > data[col].mean() * 10)
            quality_metrics["price_anomalies"] += anomalies.sum()

    # Check for data gaps
    if len(data) > 1:
        expected_periods = (data.index.max() - data.index.min()).days
        actual_periods = len(data)
        quality_metrics["data_gaps"] = max(0, expected_periods - actual_periods)

    # Calculate quality score
    total_expected = len(data) * len(data.columns)
    total_missing = sum(quality_metrics["missing_values"].values())
    quality_metrics["quality_score"] = 1 - (total_missing / total_expected)

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
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    # Resample
    resampled = data.resample(target_interval).agg(agg_rules)

    return resampled.dropna()


def add_market_hours_filter(
    data: pd.DataFrame, market_hours: tuple = (9, 17)
) -> pd.DataFrame:
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
        (data.index.hour >= start_hour)
        & (data.index.hour < end_hour)
        & (data.index.dayofweek < 5)  # Monday to Friday
    ]

    return filtered_data


def get_optimal_date_range(min_years: int = 1, preferred_years: int = 3) -> tuple:
    """
    Calculate optimal date range for data download.

    Args:
        min_years (int): Minimum years of data required
        preferred_years (int): Preferred years of data

    Returns:
        tuple: (start_date, end_date) as strings
    """
    end_date = datetime.now()

    # Calculate start date for preferred years
    preferred_start = end_date - timedelta(days=preferred_years * 365)

    # Ensure we have at least min_years
    min_start = end_date - timedelta(days=min_years * 365)

    # Use the earlier date to ensure we get enough data
    start_date = min(preferred_start, min_start)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def ensure_data_completeness(
    data_dict: Dict[str, pd.DataFrame], min_years: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Ensure all timeframes have complete data for the required period.
    Adjusts requirements based on Yahoo Finance limitations.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of data by timeframe
        min_years (int): Minimum years of data required

    Returns:
        Dict[str, pd.DataFrame]: Validated data dictionary
    """
    validated_data = {}

    # Define minimum data requirements for each timeframe
    min_requirements = {
        "1h": 60 * 24,  # 60 days of hourly data (Yahoo Finance limit)
        "4h": 60 * 6,  # 60 days of 4-hour data
        "1d": 365,  # 1 year of daily data
        "1w": 52,  # 1 year of weekly data
    }

    for timeframe, data in data_dict.items():
        if data.empty:
            print(f"⚠️  Warning: {timeframe} data is empty")
            continue

        # Check data duration
        duration_days = (data.index.max() - data.index.min()).days
        min_data_points = min_requirements.get(timeframe, 365)

        if len(data) < min_data_points:
            print(
                f"⚠️  Warning: {timeframe} data only has {len(data)} points, need at least {min_data_points}"
            )
            # For intraday data, we're more lenient due to Yahoo Finance limitations
            if (
                timeframe in ["1h", "4h"] and len(data) >= min_data_points * 0.5
            ):  # At least 50% of required
                print(
                    f"✅ Accepting {timeframe} data with {len(data)} points (Yahoo Finance limitation)"
                )
                validated_data[timeframe] = data
                continue
            else:
                continue

        # Check for data quality
        quality = validate_data_quality(data, min_data_points=min_data_points)

        if quality["quality_score"] < 0.95:
            print(
                f"⚠️  Warning: {timeframe} data quality score is {quality['quality_score']:.3f}"
            )
            # Still accept if quality is reasonable
            if quality["quality_score"] > 0.8:
                print(f"✅ Accepting {timeframe} data despite lower quality score")
                validated_data[timeframe] = data
                continue

        validated_data[timeframe] = data

    return validated_data
