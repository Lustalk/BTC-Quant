import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    add_technical_indicators,
    fractional_differentiation,
    calculate_vwap_enhanced,
    simulate_order_flow_imbalance,
    simulate_bid_ask_spread,
    add_time_based_features,
    add_microstructure_features,
    add_advanced_technical_indicators,
    add_fractional_features
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='30min')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 100 + np.random.randn(100).cumsum() + 2,
        'Low': 100 + np.random.randn(100).cumsum() - 2,
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure High >= Low and High >= Close >= Low
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, 100)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, 100)
    
    return data


def test_fractional_differentiation():
    """Test fractional differentiation function."""
    # Create a longer series for proper testing
    series = pd.Series([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 
                       index=pd.date_range('2023-01-01', periods=12))
    
    # Test with different d values
    for d in [0.3, 0.5, 0.7]:
        result = fractional_differentiation(series, d=d)
        
        # Check that result is a pandas Series
        assert isinstance(result, pd.Series)
        
        # Check that result has fewer values than input (due to convolution)
        assert len(result) < len(series)
        
        # Check that result is not all NaN
        assert not result.isna().all()
        
        # Check that result has some non-zero values
        assert (result != 0).any()


def test_calculate_vwap_enhanced(sample_data):
    """Test enhanced VWAP calculation."""
    result = calculate_vwap_enhanced(sample_data)
    
    # Check that all expected columns are present
    expected_columns = ['VWAP', 'VWAP_Deviation', 'VWAP_Upper', 'VWAP_Lower', 'VWAP_Position']
    for col in expected_columns:
        assert col in result.index
    
    # Check that VWAP values are reasonable (accounting for NaN at start)
    assert result['VWAP'].notna().any()
    # Check that non-NaN values are positive
    non_nan_vwap = result['VWAP'].dropna()
    assert (non_nan_vwap > 0).all()


def test_simulate_order_flow_imbalance(sample_data):
    """Test Order Flow Imbalance simulation."""
    result = simulate_order_flow_imbalance(sample_data)
    
    # Check that all expected columns are present
    expected_columns = ['OFI', 'OFI_Momentum', 'OFI_MA', 'OFI_Std']
    for col in expected_columns:
        assert col in result.index
    
    # Check that OFI values are reasonable (accounting for NaN at start)
    assert result['OFI'].notna().any()
    # Check that non-NaN values are in reasonable range
    non_nan_ofi = result['OFI'].dropna()
    assert (non_nan_ofi >= -1).all() and (non_nan_ofi <= 1).all()


def test_simulate_bid_ask_spread(sample_data):
    """Test bid-ask spread simulation."""
    result = simulate_bid_ask_spread(sample_data)
    
    # Check that all expected columns are present
    expected_columns = ['Bid_Ask_Spread', 'Spread_Momentum', 'Spread_MA', 'Spread_Upper', 'Spread_Lower', 'Spread_Position']
    for col in expected_columns:
        assert col in result.index
    
    # Check that spread values are reasonable (accounting for NaN at start)
    assert result['Bid_Ask_Spread'].notna().any()
    # Check that non-NaN values are positive
    non_nan_spread = result['Bid_Ask_Spread'].dropna()
    assert (non_nan_spread > 0).all()


def test_add_time_based_features(sample_data):
    """Test time-based feature addition."""
    result = add_time_based_features(sample_data.copy())
    
    # Check that time-based columns are added
    expected_columns = ['Hour', 'DayOfWeek', 'Month', 'Quarter', 'Is_Market_Open', 
                       'Is_Market_Close', 'Is_Lunch_Hour', 'Is_Weekend', 'Is_Monday', 
                       'Is_Friday', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos']
    
    for col in expected_columns:
        assert col in result.columns
    
    # Check that hour values are in valid range
    assert (result['Hour'] >= 0).all() and (result['Hour'] <= 23).all()
    
    # Check that day of week values are in valid range
    assert (result['DayOfWeek'] >= 0).all() and (result['DayOfWeek'] <= 6).all()


def test_add_microstructure_features(sample_data):
    """Test microstructure feature addition."""
    result = add_microstructure_features(sample_data.copy())
    
    # Check that microstructure columns are added
    microstructure_columns = [
        'VWAP_VWAP', 'VWAP_VWAP_Deviation', 'VWAP_VWAP_Upper', 'VWAP_VWAP_Lower', 'VWAP_VWAP_Position',
        'OFI_OFI', 'OFI_OFI_Momentum', 'OFI_OFI_MA', 'OFI_OFI_Std',
        'Spread_Bid_Ask_Spread', 'Spread_Spread_Momentum', 'Spread_Spread_MA', 'Spread_Spread_Upper', 'Spread_Spread_Lower', 'Spread_Spread_Position',
        'Volume_Price_Trend', 'Volume_MA_Ratio', 'Volume_Std',
        'High_Low_Range', 'Open_Close_Range', 'Body_Size'
    ]
    
    for col in microstructure_columns:
        assert col in result.columns


def test_add_advanced_technical_indicators(sample_data):
    """Test advanced technical indicators addition."""
    result = add_advanced_technical_indicators(sample_data.copy())
    
    # Check that advanced technical indicators are added
    advanced_indicators = [
        'CCI', 'DMI_Plus', 'DMI_Minus', 'ADX',
        'Keltner_Upper', 'Keltner_Lower', 'Keltner_Middle',
        'MFI', 'ADI', 'CMF',
        'Doji', 'Hammer'
    ]
    
    for indicator in advanced_indicators:
        assert indicator in result.columns


def test_add_fractional_features(sample_data):
    """Test fractional differentiation features."""
    result = add_fractional_features(sample_data.copy())
    
    # Check that fractional features are added
    d_values = [0.3, 0.5, 0.7]
    series_names = ['Price', 'Volume', 'Returns']
    
    for d in d_values:
        for series in series_names:
            col_name = f'{series}_FracDiff_{d}'
            assert col_name in result.columns


def test_add_technical_indicators_integration(sample_data):
    """Test the complete technical indicators function."""
    result = add_technical_indicators(sample_data)
    
    # Check that basic indicators are present
    basic_indicators = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'Williams_R',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'ATR',
        'OBV', 'VWAP_Basic'
    ]
    
    for indicator in basic_indicators:
        assert indicator in result.columns
    
    # Check that microstructure features are present
    microstructure_features = [
        'VWAP_VWAP', 'OFI_OFI', 'Spread_Bid_Ask_Spread',
        'Hour', 'DayOfWeek', 'Hour_Sin', 'Hour_Cos'
    ]
    
    for feature in microstructure_features:
        assert feature in result.columns
    
    # Check that fractional features are present
    fractional_features = ['Price_FracDiff_0.5', 'Volume_FracDiff_0.5']
    for feature in fractional_features:
        assert feature in result.columns
    
    # Check that no NaN values remain (except for fractional features which may have some)
    non_fractional_cols = [col for col in result.columns if 'FracDiff' not in col]
    assert not result[non_fractional_cols].isna().all().any()


def test_feature_engineering_with_intraday_data():
    """Test feature engineering with intraday data."""
    # Create intraday data (30-minute intervals)
    dates = pd.date_range('2023-01-01 09:00:00', periods=50, freq='30min')
    np.random.seed(42)
    
    intraday_data = pd.DataFrame({
        'Open': 100 + np.random.randn(50).cumsum(),
        'High': 100 + np.random.randn(50).cumsum() + 2,
        'Low': 100 + np.random.randn(50).cumsum() - 2,
        'Close': 100 + np.random.randn(50).cumsum(),
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Ensure High >= Low and High >= Close >= Low
    intraday_data['High'] = intraday_data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, 50)
    intraday_data['Low'] = intraday_data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, 50)
    
    result = add_technical_indicators(intraday_data)
    
    # Check that time-based features work with intraday data
    assert 'Hour' in result.columns
    assert 'Is_Market_Open' in result.columns
    
    # Check that microstructure features are calculated
    assert 'VWAP_VWAP' in result.columns
    assert 'OFI_OFI' in result.columns
    
    # Verify data quality
    assert len(result) > 0
    assert not result.isna().all().any()


def test_error_handling():
    """Test error handling in feature engineering."""
    # Test with missing required columns
    invalid_data = pd.DataFrame({
        'Open': [1, 2, 3],
        'Close': [1, 2, 3]
        # Missing High, Low, Volume
    })
    
    with pytest.raises(ValueError, match="Missing required column"):
        add_technical_indicators(invalid_data)
    
    # Test fractional differentiation with invalid d value
    series = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        fractional_differentiation(series, d=1.5)  # d should be < 1
