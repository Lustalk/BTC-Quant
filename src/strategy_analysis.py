import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .evaluation import calculate_all_metrics
from .transaction_costs import TransactionCostModel
from .position_sizing import PositionSizer


def analyze_strategy_performance(
    prices: List[float],
    signals: List[int],
    initial_capital: float = 10000.0,
    volumes: List[float] = None,
    atr_values: List[float] = None,
    include_transaction_costs: bool = True,
    position_sizing_strategy: str = "volatility_targeted",
) -> Dict[str, float]:
    """
    Analyze trading strategy performance with optional transaction costs and position sizing.

    Args:
        prices (List[float]): List of prices
        signals (List[int]): List of trading signals (1 for buy, 0 for hold, -1 for sell)
        initial_capital (float): Initial capital
        volumes (List[float]): List of trading volumes (optional)
        atr_values (List[float]): List of ATR values (optional)
        include_transaction_costs (bool): Whether to include transaction costs
        position_sizing_strategy (str): Position sizing strategy to use

    Returns:
        Dict[str, float]: Performance metrics
    """
    if len(prices) != len(signals):
        raise ValueError("Prices and signals must have the same length")

    # Initialize position sizer
    position_sizer = PositionSizer()

    # Calculate basic strategy returns
    strategy_returns = calculate_strategy_returns(prices, signals)

    # Calculate buy-and-hold returns for comparison
    buy_hold_returns = calculate_buy_hold_returns(prices)

    # Apply transaction costs if requested and data available
    if include_transaction_costs and volumes is not None and atr_values is not None:
        cost_model = TransactionCostModel()
        adjusted_returns, portfolio_values, cost_summary = (
            cost_model.apply_transaction_costs(
                prices, signals, volumes, atr_values, initial_capital
            )
        )

        # Calculate cost impact analysis
        cost_impact = cost_model.calculate_cost_impact_analysis(
            strategy_returns, adjusted_returns, cost_summary
        )

        # Calculate position sizing metrics
        position_sizes = calculate_position_sizes(
            prices,
            signals,
            atr_values,
            initial_capital,
            position_sizer,
            position_sizing_strategy,
        )

        # Calculate portfolio risk metrics
        portfolio_risk_metrics = position_sizer.calculate_portfolio_risk_metrics(
            position_sizes, adjusted_returns, initial_capital
        )

        # Use adjusted returns for metrics
        strategy_metrics = calculate_all_metrics(adjusted_returns)

        # Add transaction cost metrics
        strategy_metrics.update(
            {
                "total_fees": cost_summary["total_fees"],
                "total_slippage": cost_summary["total_slippage"],
                "total_costs": cost_summary["total_costs"],
                "trade_count": cost_summary["trade_count"],
                "cost_impact": cost_summary["cost_impact"],
                "return_degradation": cost_impact["return_degradation"],
                "degradation_percentage": cost_impact["degradation_percentage"],
            }
        )

        # Add position sizing metrics
        strategy_metrics.update(portfolio_risk_metrics)

    else:
        # Calculate metrics without transaction costs
        strategy_metrics = calculate_all_metrics(strategy_returns)

    buy_hold_metrics = calculate_all_metrics(buy_hold_returns)

    # Add comparison metrics
    strategy_metrics["excess_return"] = (
        strategy_metrics["total_return"] - buy_hold_metrics["total_return"]
    )
    strategy_metrics["excess_sharpe"] = (
        strategy_metrics["sharpe_ratio"] - buy_hold_metrics["sharpe_ratio"]
    )

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
        price_return = (prices[i] - prices[i - 1]) / prices[i - 1]

        # Update position based on signal
        if signals[i - 1] == 1 and position == 0:  # Buy signal
            position = 1
        elif signals[i - 1] == -1 and position == 1:  # Sell signal
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
        ret = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns.append(ret)
    return returns


def calculate_position_sizes(
    prices: List[float],
    signals: List[int],
    atr_values: List[float],
    initial_capital: float,
    position_sizer: PositionSizer,
    strategy: str = "volatility_targeted",
) -> List[float]:
    """
    Calculate position sizes for each trading signal.

    Args:
        prices (List[float]): List of prices
        signals (List[int]): List of trading signals
        atr_values (List[float]): List of ATR values
        initial_capital (float): Initial capital
        position_sizer (PositionSizer): Position sizing instance
        strategy (str): Position sizing strategy

    Returns:
        List[float]: Position sizes for each period (same length as returns)
    """
    position_sizes = []
    current_capital = initial_capital
    position = 0  # 0: no position, 1: long position

    # Start from index 1 to match the returns calculation
    for i in range(1, len(prices)):
        price = prices[i]
        signal = signals[i - 1]  # Use previous signal
        atr = (
            atr_values[i]
            if i < len(atr_values)
            else atr_values[-1] if atr_values else 0.01
        )

        # Update position based on signal
        if signal == 1 and position == 0:  # Buy signal
            position = 1
            # Calculate position size using the position sizer
            position_size = position_sizer.calculate_optimal_position_size(
                current_capital, price, atr, strategy
            )
        elif signal == -1 and position == 1:  # Sell signal
            position = 0
            position_size = 0.0
        else:
            # No position change, maintain current position size
            position_size = position_sizes[-1] if position_sizes else 0.0

        position_sizes.append(position_size)

        # Update capital (simplified - in reality this would be more complex)
        price_return = (price - prices[i - 1]) / prices[i - 1]
        if position == 1:
            current_capital *= 1 + price_return

    return position_sizes


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
            "total_trades": 0,
            "avg_trade_duration": 0,
            "max_trade_duration": 0,
            "min_trade_duration": 0,
        }

    return {
        "total_trades": len(trades),
        "avg_trade_duration": np.mean(trades),
        "max_trade_duration": np.max(trades),
        "min_trade_duration": np.min(trades),
    }


def generate_performance_report(
    strategy_metrics: Dict[str, float],
    buy_hold_metrics: Dict[str, float],
    trade_stats: Dict[str, float],
) -> str:
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

    if "auc" in strategy_metrics:
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

    # Position Sizing Metrics (if available)
    if "concentration_ratio" in strategy_metrics:
        report.append("POSITION SIZING METRICS:")
        report.append("-" * 30)
        report.append(
            f"Position Concentration: {strategy_metrics['concentration_ratio']:.4f}"
        )
        report.append(
            f"Total Position Value: {strategy_metrics['total_position_value']:.2f}"
        )
        report.append(f"Position Count:        {strategy_metrics['position_count']}")
        report.append(
            f"Portfolio Volatility: {strategy_metrics['portfolio_volatility']:.4f}"
        )
        report.append("")

    # Transaction Cost Metrics (if available)
    if "total_costs" in strategy_metrics:
        report.append("TRANSACTION COST ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Total Fees:           {strategy_metrics['total_fees']:.2f}")
        report.append(f"Total Slippage:      {strategy_metrics['total_slippage']:.2f}")
        report.append(f"Total Costs:          {strategy_metrics['total_costs']:.2f}")
        report.append(f"Cost Impact:          {strategy_metrics['cost_impact']:.4f}")
        report.append(
            f"Return Degradation:   {strategy_metrics['return_degradation']:.4f}"
        )
        report.append(
            f"Degradation %:        {strategy_metrics['degradation_percentage']:.2f}%"
        )
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)


def print_performance_table(metrics: Dict[str, float]) -> str:
    """
    Format performance metrics into a readable table.

    Args:
        metrics (Dict[str, float]): Performance metrics

    Returns:
        str: Formatted performance table
    """
    table_lines = []
    table_lines.append("=" * 60)
    table_lines.append("PERFORMANCE METRICS")
    table_lines.append("=" * 60)

    # Group metrics by category
    metric_categories = {
        "Returns": ["total_return", "avg_return", "annualized_return"],
        "Risk": ["volatility", "max_drawdown", "var_95"],
        "Ratios": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
        "Trading": ["win_rate", "profit_factor", "avg_win", "avg_loss"],
        "Model": ["auc", "precision", "recall", "f1_score"],
    }

    for category, metric_names in metric_categories.items():
        table_lines.append(f"\n{category.upper()} METRICS:")
        table_lines.append("-" * 30)

        for metric_name in metric_names:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, float):
                    if "ratio" in metric_name.lower() or "rate" in metric_name.lower():
                        table_lines.append(f"{metric_name:<20}: {value:.4f}")
                    else:
                        table_lines.append(f"{metric_name:<20}: {value:.4f}")
                else:
                    table_lines.append(f"{metric_name:<20}: {value}")

    table_lines.append("=" * 60)

    return "\n".join(table_lines)
