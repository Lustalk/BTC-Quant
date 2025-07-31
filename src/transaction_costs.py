"""
Transaction Costs and Market Friction Module

This module implements realistic transaction cost modeling including:
1. Dynamic transaction fees (maker/taker)
2. Volume-based slippage model
3. Volatility-adjusted slippage
4. Position sizing with risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from .position_sizing import PositionSizer


class TransactionCostModel:
    """Realistic transaction cost model for backtesting."""

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize the transaction cost model.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.fee_config = self.config.get("transaction_costs", {})
        self.slippage_config = self.fee_config.get("slippage", {})
        self.position_config = self.fee_config.get("position_sizing", {})
        self.position_sizer = PositionSizer(config_path)

    def calculate_transaction_fee(
        self, trade_value: float, fee_type: str = "taker"
    ) -> float:
        """
        Calculate transaction fee for a trade.

        Args:
            trade_value: Value of the trade
            fee_type: Type of fee ("maker", "taker", or "default")

        Returns:
            Transaction fee amount
        """
        if fee_type == "maker":
            fee_rate = self.fee_config.get("maker_fee", 0.0004)
        elif fee_type == "taker":
            fee_rate = self.fee_config.get("taker_fee", 0.0006)
        else:
            fee_rate = self.fee_config.get("default_fee", 0.0006)

        return trade_value * fee_rate

    def calculate_slippage(
        self, price: float, volume: float, atr: float, trade_size: float
    ) -> float:
        """
        Calculate slippage based on market conditions.

        Args:
            price: Current price
            volume: Trading volume
            atr: Average True Range (volatility measure)
            trade_size: Size of the trade in currency units

        Returns:
            Slippage as a percentage of price
        """
        base_slippage = self.slippage_config.get("base_slippage", 0.0001)
        volatility_multiplier = self.slippage_config.get("volatility_multiplier", 0.5)
        volume_impact = self.slippage_config.get("volume_impact", 0.00005)
        max_slippage = self.slippage_config.get("max_slippage", 0.005)

        # Volatility-based slippage
        volatility_slippage = (atr / price) * volatility_multiplier

        # Volume impact (larger trades have more impact)
        volume_slippage = (trade_size / 100000) * volume_impact  # Per $100k

        # Total slippage
        total_slippage = base_slippage + volatility_slippage + volume_slippage

        # Cap at maximum slippage
        return min(total_slippage, max_slippage)

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: float,
        strategy: str = "hybrid",
        **kwargs
    ) -> float:
        """
        Calculate position size using advanced position sizing strategies.

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Current volatility (ATR or similar)
            strategy: Position sizing strategy
            **kwargs: Additional parameters for position sizing

        Returns:
            Position size in currency units
        """
        return self.position_sizer.calculate_optimal_position_size(
            portfolio_value, price, volatility, strategy, **kwargs
        )

    def apply_transaction_costs(
        self,
        prices: List[float],
        signals: List[int],
        volumes: List[float],
        atr_values: List[float],
        initial_capital: float = 10000.0,
        fee_type: str = "taker",
    ) -> Tuple[List[float], List[float], Dict[str, float]]:
        """
        Apply transaction costs to strategy returns.

        Args:
            prices: List of prices
            signals: List of trading signals
            volumes: List of trading volumes
            atr_values: List of ATR values
            initial_capital: Initial capital
            fee_type: Type of transaction fee

        Returns:
            Tuple of (adjusted_returns, portfolio_values, cost_summary)
        """
        if len(prices) != len(signals) != len(volumes) != len(atr_values):
            raise ValueError("All input lists must have the same length")

        adjusted_returns = []
        portfolio_values = [initial_capital]
        total_fees = 0.0
        total_slippage = 0.0
        trade_count = 0

        position = 0  # 0: no position, 1: long position
        current_capital = initial_capital

        for i in range(1, len(prices)):
            price = prices[i]
            prev_price = prices[i - 1]
            signal = signals[i - 1]
            volume = volumes[i] if i < len(volumes) else volumes[-1]
            atr = atr_values[i] if i < len(atr_values) else atr_values[-1]

            # Calculate price return
            price_return = (price - prev_price) / prev_price

            # Initialize return for this period
            period_return = 0.0

            # Handle position changes
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                position_size = self.calculate_position_size(
                    current_capital, price, atr
                )

                # Calculate transaction costs
                trade_value = position_size
                fee = self.calculate_transaction_fee(trade_value, fee_type)
                slippage = self.calculate_slippage(price, volume, atr, trade_value)

                # Apply slippage to execution price
                execution_price = price * (1 + slippage)

                # Update position and costs
                position = 1
                total_fees += fee
                total_slippage += slippage * trade_value
                trade_count += 1

                # Calculate return with costs
                period_return = (execution_price - prev_price) / prev_price
                period_return -= (fee + slippage * trade_value) / current_capital

            elif signal == -1 and position == 1:  # Sell signal
                # Calculate transaction costs for selling
                trade_value = current_capital * 0.95  # Approximate position value
                fee = self.calculate_transaction_fee(trade_value, fee_type)
                slippage = self.calculate_slippage(price, volume, atr, trade_value)

                # Apply slippage to execution price
                execution_price = price * (1 - slippage)

                # Update position and costs
                position = 0
                total_fees += fee
                total_slippage += slippage * trade_value
                trade_count += 1

                # Calculate return with costs
                period_return = (execution_price - prev_price) / prev_price
                period_return -= (fee + slippage * trade_value) / current_capital

            elif position == 1:  # Hold long position
                period_return = price_return

            # Update capital
            current_capital *= 1 + period_return
            portfolio_values.append(current_capital)
            adjusted_returns.append(period_return)

        # Calculate cost summary
        cost_summary = {
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "total_costs": total_fees + total_slippage,
            "trade_count": trade_count,
            "avg_fee_per_trade": total_fees / max(trade_count, 1),
            "avg_slippage_per_trade": total_slippage / max(trade_count, 1),
            "cost_impact": (total_fees + total_slippage) / initial_capital,
        }

        return adjusted_returns, portfolio_values, cost_summary

    def calculate_cost_impact_analysis(
        self,
        original_returns: List[float],
        adjusted_returns: List[float],
        cost_summary: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Analyze the impact of transaction costs on performance.

        Args:
            original_returns: Returns without transaction costs
            adjusted_returns: Returns with transaction costs
            cost_summary: Summary of transaction costs

        Returns:
            Dictionary with cost impact analysis
        """
        if len(original_returns) != len(adjusted_returns):
            raise ValueError("Return lists must have the same length")

        # Calculate performance metrics
        original_total_return = np.prod([1 + r for r in original_returns]) - 1
        adjusted_total_return = np.prod([1 + r for r in adjusted_returns]) - 1

        # Calculate return degradation
        return_degradation = original_total_return - adjusted_total_return

        # Calculate cost-adjusted metrics
        cost_impact = {
            "original_total_return": original_total_return,
            "adjusted_total_return": adjusted_total_return,
            "return_degradation": return_degradation,
            "degradation_percentage": (
                (return_degradation / original_total_return) * 100
                if original_total_return != 0
                else 0
            ),
            "cost_impact_ratio": cost_summary["cost_impact"],
            "fee_impact": (
                cost_summary["total_fees"] / sum([abs(r) for r in original_returns])
                if sum([abs(r) for r in original_returns]) > 0
                else 0
            ),
            "slippage_impact": (
                cost_summary["total_slippage"] / sum([abs(r) for r in original_returns])
                if sum([abs(r) for r in original_returns]) > 0
                else 0
            ),
        }

        return cost_impact
