import pandas as pd
import yfinance as yf
from typing import Optional


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download financial data for a given ticker and date range.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'BTC-USD')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: Raw financial data with OHLCV columns
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker} in the specified date range")

        return data
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {str(e)}")
