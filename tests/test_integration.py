import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Local imports
from src.data_pipeline import download_data  # noqa: E402
from src.feature_engineering import add_technical_indicators  # noqa: E402
from src.model import prepare_features_target, walk_forward_validation  # noqa: E402


def test_full_pipeline_integration():
    """Test that the entire pipeline runs without crashing."""
    # Create mock data to avoid network calls
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 200)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create mock OHLCV data
    mock_data = pd.DataFrame(
        {
            "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 200),
        },
        index=dates,
    )

    # Ensure High >= Low and High >= Close >= Low
    mock_data["High"] = mock_data[["High", "Close"]].max(axis=1)
    mock_data["Low"] = mock_data[["Low", "Close"]].min(axis=1)

    # Mock the download_data function to return our test data
    with patch("src.data_pipeline.download_data", return_value=mock_data):
        # Step 1: "Download" data - use the mocked function
        from src.data_pipeline import download_data as mocked_download_data

        data = mocked_download_data("BTC-USD", "2023-01-01", "2023-12-31")
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Step 2: Engineer features
        data_with_features = add_technical_indicators(data)
        assert isinstance(data_with_features, pd.DataFrame)
        assert len(data_with_features.columns) > len(data.columns)

        # Step 3: Prepare features and target
        X, y = prepare_features_target(data_with_features)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

        # Step 4: Test walk-forward validation (with reduced splits for speed)
        scores = walk_forward_validation(X, y, n_splits=2)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)


def test_main_function_import():
    """Test that main.py can be imported without errors."""
    try:
        from main import main

        assert callable(main)
    except ImportError as e:
        pytest.fail(f"Failed to import main function: {e}")


def test_pipeline_with_small_dataset():
    """Test pipeline with minimal data to ensure it handles edge cases."""
    # Create very small dataset
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)

    prices = [100]
    for _ in range(49):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

    small_data = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, 50),
        },
        index=dates,
    )

    # Test feature engineering with small dataset
    try:
        data_with_features = add_technical_indicators(small_data)
        assert isinstance(data_with_features, pd.DataFrame)
        assert len(data_with_features) > 0
    except Exception as e:
        pytest.fail(f"Feature engineering failed with small dataset: {e}")

    # Test feature preparation with small dataset
    try:
        X, y = prepare_features_target(data_with_features)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
    except Exception as e:
        pytest.fail(f"Feature preparation failed with small dataset: {e}")


def test_pipeline_error_handling():
    """Test that the pipeline handles errors gracefully."""
    # Test with invalid data
    invalid_data = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            # Missing 'Close' and 'Volume' columns
        }
    )

    # Should raise ValueError for missing columns
    with pytest.raises(ValueError, match="Missing required column"):
        add_technical_indicators(invalid_data)
