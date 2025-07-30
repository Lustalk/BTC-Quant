import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .evaluation import calculate_all_metrics


def analyze_strategy_performance(prices: List[float], signals: List[int], 
                               initial_capital: float = 10000.0) -> Dict[str, float]:
    """
    Analyze trading strategy performance.
    
    Args:
        prices (List[float]): List of prices
        signals (List[int]): List of trading signals (1 for buy, 0 for hold, -1 for sell)
        initial_capital (float): Initial capital
    
    Returns:
        Dict[str, float]: Performance metrics
    """
    if len(prices) != len(signals):
        raise ValueError("Prices and signals must have the same length")
    
    # Calculate strategy returns
    strategy_returns = calculate_strategy_returns(prices, signals)
    
    # Calculate buy-and-hold returns for comparison
    buy_hold_returns = calculate_buy_hold_returns(prices)
    
    # Calculate metrics
    strategy_metrics = calculate_all_metrics(strategy_returns)
    buy_hold_metrics = calculate_all_metrics(buy_hold_returns)
    
    # Add comparison metrics
    strategy_metrics['excess_return'] = strategy_metrics['total_return'] - buy_hold_metrics['total_return']
    strategy_metrics['excess_sharpe'] = strategy_metrics['sharpe_ratio'] - buy_hold_metrics['sharpe_ratio']
    
    return strategy_metrics


def calculate_strategy_returns(prices: List[float], signals: List[int]) -> List[float]:
    """
    Calculate returns based on trading signals.
    
    Args:
        prices (List[float]): List of prices
        signals (List[int]): List of trading signals
    
    Returns:
        List[float]: Strategy returns
    """
    returns = []
    position = 0  # 0: no position, 1: long position
    
    for i in range(1, len(prices)):
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        
        # Update position based on signal
        if signals[i-1] == 1 and position == 0:  # Buy signal
            position = 1
        elif signals[i-1] == -1 and position == 1:  # Sell signal
            position = 0
        
        # Calculate strategy return
        if position == 1:
            returns.append(price_return)
        else:
            returns.append(0.0)
    
    return returns


def calculate_buy_hold_returns(prices: List[float]) -> List[float]:
    """
    Calculate buy-and-hold returns.
    
    Args:
        prices (List[float]): List of prices
    
    Returns:
        List[float]: Buy-and-hold returns
    """
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    return returns


def calculate_trade_statistics(signals: List[int]) -> Dict[str, float]:
    """
    Calculate trading statistics.
    
    Args:
        signals (List[int]): List of trading signals
    
    Returns:
        Dict[str, float]: Trading statistics
    """
    trades = []
    position = 0
    entry_price = None
    
    for i, signal in enumerate(signals):
        if signal == 1 and position == 0:  # Enter long position
            position = 1
            entry_price = i
        elif signal == -1 and position == 1:  # Exit long position
            if entry_price is not None:
                trades.append(i - entry_price)
            position = 0
            entry_price = None
    
    if not trades:
        return {
            'total_trades': 0,
            'avg_trade_duration': 0,
            'max_trade_duration': 0,
            'min_trade_duration': 0
        }
    
    return {
        'total_trades': len(trades),
        'avg_trade_duration': np.mean(trades),
        'max_trade_duration': np.max(trades),
        'min_trade_duration': np.min(trades)
    }


def generate_performance_report(strategy_metrics: Dict[str, float], 
                              buy_hold_metrics: Dict[str, float],
                              trade_stats: Dict[str, float]) -> str:
    """
    Generate a formatted performance report.
    
    Args:
        strategy_metrics (Dict[str, float]): Strategy performance metrics
        buy_hold_metrics (Dict[str, float]): Buy-and-hold metrics
        trade_stats (Dict[str, float]): Trading statistics
    
    Returns:
        str: Formatted performance report
    """
    report = []
    report.append("=" * 60)
    report.append("BTC QUANT - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Strategy Performance
    report.append("STRATEGY PERFORMANCE:")
    report.append("-" * 30)
    report.append(f"Sharpe Ratio:        {strategy_metrics['sharpe_ratio']:.4f}")
    report.append(f"Max Drawdown:        {strategy_metrics['max_drawdown']:.4f}")
    report.append(f"Volatility:          {strategy_metrics['volatility']:.4f}")
    report.append(f"Win Rate:            {strategy_metrics['win_rate']:.4f}")
    report.append(f"Profit Factor:        {strategy_metrics['profit_factor']:.4f}")
    report.append(f"Total Return:        {strategy_metrics['total_return']:.4f}")
    report.append(f"Average Return:      {strategy_metrics['avg_return']:.4f}")
    
    if 'auc' in strategy_metrics:
        report.append(f"AUC:                 {strategy_metrics['auc']:.4f}")
    
    report.append("")
    
    # Buy-and-Hold Comparison
    report.append("BUY-AND-HOLD COMPARISON:")
    report.append("-" * 30)
    report.append(f"Buy-Hold Sharpe:     {buy_hold_metrics['sharpe_ratio']:.4f}")
    report.append(f"Buy-Hold Return:     {buy_hold_metrics['total_return']:.4f}")
    report.append(f"Excess Return:       {strategy_metrics['excess_return']:.4f}")
    report.append(f"Excess Sharpe:       {strategy_metrics['excess_sharpe']:.4f}")
    report.append("")
    
    # Trading Statistics
    report.append("TRADING STATISTICS:")
    report.append("-" * 30)
    report.append(f"Total Trades:        {trade_stats['total_trades']}")
    report.append(f"Avg Trade Duration:  {trade_stats['avg_trade_duration']:.1f} days")
    report.append(f"Max Trade Duration:  {trade_stats['max_trade_duration']} days")
    report.append(f"Min Trade Duration:  {trade_stats['min_trade_duration']} days")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def print_performance_table(metrics: Dict[str, float]) -> None:
    """
    Print a formatted performance metrics table.
    
    Args:
        metrics (Dict[str, float]): Performance metrics
    """
    print("=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    # Define the metrics to display and their labels
    metric_labels = {
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'volatility': 'Volatility',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor',
        'total_return': 'Total Return',
        'avg_return': 'Average Return'
    }
    
    # Print each metric
    for key, label in metric_labels.items():
        if key in metrics:
            value = metrics[key]
            if key in ['win_rate', 'max_drawdown']:
                print(f"{label:<20}: {value:.4f}")
            else:
                print(f"{label:<20}: {value:.4f}")
    
    if 'auc' in metrics:
        print(f"{'AUC':<20}: {metrics['auc']:.4f}")
    
    print("=" * 60) 