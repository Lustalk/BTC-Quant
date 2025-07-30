import pytest
import numpy as np
from src.strategy_analysis import (
    analyze_strategy_performance,
    calculate_strategy_returns,
    calculate_buy_hold_returns,
    calculate_trade_statistics,
    generate_performance_report,
    print_performance_table,
)


def test_calculate_strategy_returns():
    """Test strategy returns calculation."""
    prices = [100, 110, 105, 120, 115, 125]
    signals = [1, 0, -1, 1, 0, -1]  # Buy, hold, sell, buy, hold, sell

    returns = calculate_strategy_returns(prices, signals)

    # Should have 5 returns (one less than prices)
    assert len(returns) == 5

    # First period: position=1, return should be (110-100)/100 = 0.1
    assert abs(returns[0] - 0.1) < 0.001

    # Second period: position=1, return should be (105-110)/110 = -0.045
    assert abs(returns[1] - (-0.045)) < 0.001

    # Third period: position=0, return should be 0
    assert returns[2] == 0.0


def test_calculate_buy_hold_returns():
    """Test buy-and-hold returns calculation."""
    prices = [100, 110, 105, 120, 115]
    returns = calculate_buy_hold_returns(prices)

    expected_returns = [0.1, -0.0455, 0.1429, -0.0417]
    assert len(returns) == len(expected_returns)

    for actual, expected in zip(returns, expected_returns):
        assert abs(actual - expected) < 0.001


def test_calculate_trade_statistics():
    """Test trade statistics calculation."""
    signals = [1, 0, 0, -1, 1, 0, -1, 0, 1, -1]

    stats = calculate_trade_statistics(signals)

    assert stats["total_trades"] == 3
    assert stats["avg_trade_duration"] > 0
    assert stats["max_trade_duration"] > 0
    assert stats["min_trade_duration"] > 0

    # Test with no trades
    no_trade_signals = [0, 0, 0, 0, 0]
    stats_no_trades = calculate_trade_statistics(no_trade_signals)

    assert stats_no_trades["total_trades"] == 0
    assert stats_no_trades["avg_trade_duration"] == 0
    assert stats_no_trades["max_trade_duration"] == 0
    assert stats_no_trades["min_trade_duration"] == 0


def test_analyze_strategy_performance():
    """Test strategy performance analysis."""
    prices = [100, 110, 105, 120, 115, 125, 130, 125, 120, 115]
    signals = [1, 0, -1, 1, 0, 0, -1, 1, 0, -1]

    metrics = analyze_strategy_performance(prices, signals)

    # Check that all required metrics are present
    required_metrics = [
        "sharpe_ratio",
        "max_drawdown",
        "volatility",
        "win_rate",
        "profit_factor",
        "total_return",
        "avg_return",
        "excess_return",
        "excess_sharpe",
    ]

    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_analyze_strategy_performance_invalid_input():
    """Test strategy performance analysis with invalid input."""
    prices = [100, 110, 105]
    signals = [1, 0]  # Different length

    with pytest.raises(ValueError, match="Prices and signals must have the same length"):
        analyze_strategy_performance(prices, signals)


def test_generate_performance_report():
    """Test performance report generation."""
    strategy_metrics = {
        "sharpe_ratio": 0.5,
        "max_drawdown": 0.15,
        "volatility": 0.2,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "total_return": 0.25,
        "avg_return": 0.02,
        "excess_return": 0.05,
        "excess_sharpe": 0.1,
    }

    buy_hold_metrics = {"sharpe_ratio": 0.4, "total_return": 0.2}

    trade_stats = {
        "total_trades": 10,
        "avg_trade_duration": 5.5,
        "max_trade_duration": 15,
        "min_trade_duration": 2,
    }

    report = generate_performance_report(strategy_metrics, buy_hold_metrics, trade_stats)

    # Check that report contains expected sections
    assert "BTC QUANT - PERFORMANCE REPORT" in report
    assert "STRATEGY PERFORMANCE:" in report
    assert "BUY-AND-HOLD COMPARISON:" in report
    assert "TRADING STATISTICS:" in report

    # Check that metrics are included
    assert "0.5000" in report  # Sharpe ratio
    assert "0.1500" in report  # Max drawdown
    assert "10" in report  # Total trades


def test_print_performance_table():
    """Test performance table printing."""
    metrics = {
        "sharpe_ratio": 0.5,
        "max_drawdown": 0.15,
        "volatility": 0.2,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "total_return": 0.25,
        "avg_return": 0.02,
        "auc": 0.65,
    }

    # This should not raise any exceptions
    print_performance_table(metrics)


def test_print_performance_table_with_auc():
    """Test performance table printing with AUC."""
    metrics = {
        "sharpe_ratio": 0.5,
        "max_drawdown": 0.15,
        "volatility": 0.2,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "total_return": 0.25,
        "avg_return": 0.02,
        "auc": 0.65,
    }

    # This should not raise any exceptions
    print_performance_table(metrics)


def test_strategy_returns_edge_cases():
    """Test strategy returns with edge cases."""
    # Test with no signals
    prices = [100, 110, 105]
    signals = [0, 0, 0]

    returns = calculate_strategy_returns(prices, signals)
    assert all(r == 0.0 for r in returns)

    # Test with single price
    prices = [100]
    signals = [1]

    returns = calculate_strategy_returns(prices, signals)
    assert len(returns) == 0  # No returns for single price


def test_trade_statistics_edge_cases():
    """Test trade statistics with edge cases."""
    # Test with no signals
    signals = []
    stats = calculate_trade_statistics(signals)

    assert stats["total_trades"] == 0
    assert stats["avg_trade_duration"] == 0
    assert stats["max_trade_duration"] == 0
    assert stats["min_trade_duration"] == 0

    # Test with only buy signal (no sell)
    signals = [1, 0, 0, 0]
    stats = calculate_trade_statistics(signals)

    assert stats["total_trades"] == 0  # No completed trades
