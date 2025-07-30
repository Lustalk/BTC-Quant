import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import add_technical_indicators


def test_add_technical_indicators():
    """Test that technical indicators are correctly calculated."""
    # Create a small, static DataFrame for testing
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)  # For reproducible results

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Ensure High >= Low and High >= Close >= Low
    data["High"] = data[["High", "Close"]].max(axis=1)
    data["Low"] = data[["Low", "Close"]].min(axis=1)

    # Add technical indicators
    result = add_technical_indicators(data)

    # Verify the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify it has more columns than the original (indicators added)
    assert len(result.columns) > len(data.columns)

    # Verify specific technical indicators are present
    expected_indicators = [
        "RSI_14",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Upper",
        "BB_Lower",
        "BB_Middle",
        "BB_Width",
        "ATR",
        "OBV",
        "VWAP",
        "Returns",
        "Log_Returns",
    ]

    for indicator in expected_indicators:
        assert indicator in result.columns, f"Missing indicator: {indicator}"

    # Verify RSI is within expected range (0-100)
    rsi_values = result["RSI_14"].dropna()
    assert len(rsi_values) > 0, "RSI values should not be all NaN"
    assert (rsi_values >= 0).all() and (
        rsi_values <= 100
    ).all(), "RSI should be between 0 and 100"

    # Verify SMA calculations are reasonable
    sma_20 = result["SMA_20"].dropna()
    assert len(sma_20) > 0, "SMA_20 should not be all NaN"
    assert (sma_20 > 0).all(), "SMA_20 should be positive"

    # Verify the result has no NaN values (except for the initial rows that were dropped)
    assert not result.isnull().any().any(), "Result should not contain NaN values"

    # Verify the original OHLCV columns are preserved
    original_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in original_columns:
        assert col in result.columns, f"Original column {col} should be preserved"


def test_add_technical_indicators_missing_columns():
    """Test that the function raises an error for missing required columns."""
    # Create DataFrame with missing columns
    data = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            # Missing 'Close' and 'Volume'
        }
    )

    with pytest.raises(ValueError, match="Missing required column"):
        add_technical_indicators(data)
