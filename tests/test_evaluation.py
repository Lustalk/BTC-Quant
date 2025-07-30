import pytest
import numpy as np
from src.evaluation import (
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_volatility, calculate_auc, calculate_win_rate,
    calculate_profit_factor, calculate_all_metrics
)


def test_calculate_returns():
    """Test return calculation."""
    prices = [100, 110, 105, 120, 115]
    returns = calculate_returns(prices)
    
    expected_returns = [0.1, -0.0455, 0.1429, -0.0417]
    assert len(returns) == len(expected_returns)
    
    for actual, expected in zip(returns, expected_returns):
        assert abs(actual - expected) < 0.001


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Test with positive returns
    returns = [0.01, 0.02, 0.015, 0.025, 0.02]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0
    
    # Test with zero returns
    returns = [0.0, 0.0, 0.0]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0
    
    # Test with empty list
    sharpe = calculate_sharpe_ratio([])
    assert sharpe == 0.0


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    # Test with drawdown
    returns = [0.1, -0.05, -0.1, 0.2, -0.15]
    max_dd = calculate_max_drawdown(returns)
    assert max_dd > 0
    assert max_dd <= 1.0
    
    # Test with no drawdown (all positive)
    returns = [0.01, 0.02, 0.015, 0.025]
    max_dd = calculate_max_drawdown(returns)
    assert max_dd == 0.0
    
    # Test with empty list
    max_dd = calculate_max_drawdown([])
    assert max_dd == 0.0


def test_calculate_volatility():
    """Test volatility calculation."""
    returns = [0.01, -0.02, 0.015, -0.01, 0.025]
    vol = calculate_volatility(returns)
    assert vol > 0
    
    # Test with zero volatility
    returns = [0.01, 0.01, 0.01, 0.01]
    vol = calculate_volatility(returns)
    assert vol == 0.0


def test_calculate_auc():
    """Test AUC calculation."""
    actual = [1, 0, 1, 0, 1]
    predicted = [0.9, 0.1, 0.8, 0.2, 0.7]
    
    auc = calculate_auc(actual, predicted)
    assert 0 <= auc <= 1
    
    # Test with perfect prediction
    predicted = [0.9, 0.1, 0.8, 0.2, 0.7]
    auc = calculate_auc(actual, predicted)
    assert auc > 0.5  # Should be better than random
    
    # Test with single class
    actual = [1, 1, 1, 1, 1]
    predicted = [0.9, 0.8, 0.7, 0.6, 0.5]
    auc = calculate_auc(actual, predicted)
    assert auc == 0.5


def test_calculate_win_rate():
    """Test win rate calculation."""
    returns = [0.01, -0.02, 0.015, -0.01, 0.025]
    win_rate = calculate_win_rate(returns)
    assert 0 <= win_rate <= 1
    assert win_rate == 0.6  # 3 positive out of 5
    
    # Test with all positive
    returns = [0.01, 0.02, 0.015]
    win_rate = calculate_win_rate(returns)
    assert win_rate == 1.0
    
    # Test with all negative
    returns = [-0.01, -0.02, -0.015]
    win_rate = calculate_win_rate(returns)
    assert win_rate == 0.0


def test_calculate_profit_factor():
    """Test profit factor calculation."""
    returns = [0.01, -0.02, 0.015, -0.01, 0.025]
    pf = calculate_profit_factor(returns)
    assert pf > 0
    
    # Test with no losses
    returns = [0.01, 0.02, 0.015]
    pf = calculate_profit_factor(returns)
    assert pf == float('inf')
    
    # Test with no profits
    returns = [-0.01, -0.02, -0.015]
    pf = calculate_profit_factor(returns)
    assert pf == 0.0


def test_calculate_all_metrics():
    """Test calculation of all metrics."""
    returns = [0.01, -0.02, 0.015, -0.01, 0.025]
    actual = [1, 0, 1, 0, 1]
    predicted = [0.9, 0.1, 0.8, 0.2, 0.7]
    
    metrics = calculate_all_metrics(returns, actual, predicted)
    
    required_keys = [
        'sharpe_ratio', 'max_drawdown', 'volatility', 
        'win_rate', 'profit_factor', 'total_return', 'avg_return', 'auc'
    ]
    
    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


def test_calculate_all_metrics_no_classification():
    """Test calculation of all metrics without classification data."""
    returns = [0.01, -0.02, 0.015, -0.01, 0.025]
    
    metrics = calculate_all_metrics(returns)
    
    required_keys = [
        'sharpe_ratio', 'max_drawdown', 'volatility', 
        'win_rate', 'profit_factor', 'total_return', 'avg_return'
    ]
    
    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))
    
    # AUC should not be present
    assert 'auc' not in metrics 