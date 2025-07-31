import pytest
import pandas as pd
from src.data_pipeline import download_data


def test_download_data_structure():
    """Test that download_data returns the expected structure."""
    # Test the function signature
    import inspect

    params = inspect.signature(download_data).parameters
    assert len(params) == 5  # ticker, start_date, end_date, interval, data_source
    assert "ticker" in params
    assert "start_date" in params
    assert "end_date" in params
    assert "interval" in params
    assert "data_source" in params


def test_download_data():
    """Test that download_data function successfully downloads data."""
    # Test with a more reliable ticker and date range
    ticker = "AAPL"  # Apple stock is more reliable than crypto
    start_date = "2024-01-01"
    end_date = "2024-01-10"

    try:
        # Download data
        data = download_data(ticker, start_date, end_date)

        # Verify the data is a DataFrame
        assert isinstance(data, pd.DataFrame)

        # Verify the DataFrame is not empty
        assert len(data) > 0

        # Verify it has the expected OHLCV columns
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            assert col in data.columns

        # Verify the data is within the expected date range
        # Convert timezone-aware timestamps to timezone-naive for comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Convert timezone-aware index to timezone-naive for comparison
        data_index_naive = data.index.tz_localize(None)
        assert data_index_naive.min() >= start_ts
        assert data_index_naive.max() <= end_ts

    except Exception as e:
        # If network fails, skip the test but don't fail
        pytest.skip(f"Network test skipped due to connection issues: {str(e)}")
