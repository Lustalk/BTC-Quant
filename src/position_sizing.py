"""
Dynamic Position Sizing Module

This module implements sophisticated position sizing strategies including:
1. Volatility targeting
2. Risk-based position sizing
3. Kelly criterion
4. Dynamic risk adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml


class PositionSizer:
    """Advanced position sizing with multiple strategies."""

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize the position sizer.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.position_config = self.config.get("transaction_costs", {}).get(
            "position_sizing", {}
        )

    def volatility_targeted_sizing(
        self,
        portfolio_value: float,
        price: float,
        volatility: float,
        target_volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate position size using volatility targeting.

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Current volatility (ATR or similar)
            target_volatility: Target volatility (optional, uses config default)

        Returns:
            Position size in currency units
        """
        if target_volatility is None:
            target_volatility = self.position_config.get("volatility_target", 0.15)

        max_position_size = self.position_config.get("max_position_size", 0.10)

        # Annualize volatility
        annualized_volatility = volatility * np.sqrt(252)

        # Calculate volatility ratio
        volatility_ratio = target_volatility / max(annualized_volatility, 0.01)

        # Base position size
        position_size = portfolio_value * volatility_ratio

        # Cap at maximum position size
        max_size = portfolio_value * max_position_size
        position_size = min(position_size, max_size)

        return max(0, position_size)

    def risk_based_sizing(
        self,
        portfolio_value: float,
        price: float,
        volatility: float,
        risk_per_trade: Optional[float] = None,
        stop_loss_pct: float = 0.02,
    ) -> float:
        """
        Calculate position size based on risk per trade.

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Current volatility
            risk_per_trade: Risk per trade (optional, uses config default)
            stop_loss_pct: Stop loss percentage

        Returns:
            Position size in currency units
        """
        if risk_per_trade is None:
            risk_per_trade = self.position_config.get("default_risk_per_trade", 0.02)

        max_position_size = self.position_config.get("max_position_size", 0.10)

        # Calculate risk amount
        risk_amount = portfolio_value * risk_per_trade

        # Calculate position size based on stop loss
        position_size = risk_amount / stop_loss_pct

        # Cap at maximum position size
        max_size = portfolio_value * max_position_size
        position_size = min(position_size, max_size)

        return max(0, position_size)

    def kelly_criterion_sizing(
        self, portfolio_value: float, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Args:
            portfolio_value: Current portfolio value
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return

        Returns:
            Position size as percentage of portfolio
        """
        if avg_loss == 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Apply fractional Kelly (50% of full Kelly)
        fractional_kelly = kelly_fraction * 0.5

        # Cap at reasonable limits
        position_size_pct = max(0, min(fractional_kelly, 0.25))

        return portfolio_value * position_size_pct

    def dynamic_risk_adjustment(
        self,
        base_position_size: float,
        market_regime: str,
        volatility_regime: str,
        confidence_score: float,
    ) -> float:
        """
        Adjust position size based on market conditions and confidence.

        Args:
            base_position_size: Base position size
            market_regime: Market regime ("bull", "bear", "sideways")
            volatility_regime: Volatility regime ("low", "medium", "high")
            confidence_score: Model confidence score (0-1)

        Returns:
            Adjusted position size
        """
        # Market regime adjustments
        regime_multipliers = {"bull": 1.2, "sideways": 1.0, "bear": 0.7}

        # Volatility regime adjustments
        volatility_multipliers = {"low": 1.1, "medium": 1.0, "high": 0.8}

        # Confidence adjustments
        confidence_multiplier = 0.5 + (confidence_score * 0.5)  # 0.5 to 1.0

        # Apply all adjustments
        adjusted_size = base_position_size
        adjusted_size *= regime_multipliers.get(market_regime, 1.0)
        adjusted_size *= volatility_multipliers.get(volatility_regime, 1.0)
        adjusted_size *= confidence_multiplier

        return max(0, adjusted_size)

    def calculate_optimal_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: float,
        strategy: str = "volatility_targeted",
        **kwargs,
    ) -> float:
        """
        Calculate optimal position size using specified strategy.

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Current volatility
            strategy: Position sizing strategy
            **kwargs: Additional parameters for specific strategies

        Returns:
            Optimal position size in currency units
        """
        if strategy == "volatility_targeted":
            return self.volatility_targeted_sizing(
                portfolio_value, price, volatility, kwargs.get("target_volatility")
            )

        elif strategy == "risk_based":
            return self.risk_based_sizing(
                portfolio_value,
                price,
                volatility,
                kwargs.get("risk_per_trade"),
                kwargs.get("stop_loss_pct", 0.02),
            )

        elif strategy == "kelly":
            return self.kelly_criterion_sizing(
                portfolio_value,
                kwargs.get("win_rate", 0.5),
                kwargs.get("avg_win", 0.02),
                kwargs.get("avg_loss", -0.01),
            )

        elif strategy == "hybrid":
            # Combine multiple strategies
            vol_target_size = self.volatility_targeted_sizing(
                portfolio_value, price, volatility
            )
            risk_based_size = self.risk_based_sizing(portfolio_value, price, volatility)

            # Use the smaller of the two for conservative approach
            return min(vol_target_size, risk_based_size)

        else:
            raise ValueError(f"Unknown position sizing strategy: {strategy}")

    def calculate_portfolio_risk_metrics(
        self, position_sizes: List[float], returns: List[float], portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.

        Args:
            position_sizes: List of position sizes
            returns: List of returns
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of risk metrics
        """
        if len(position_sizes) != len(returns):
            raise ValueError("Position sizes and returns must have same length")

        # Calculate position concentration
        total_position_value = sum(position_sizes)
        concentration_ratio = (
            total_position_value / portfolio_value if portfolio_value > 0 else 0
        )

        # Calculate weighted portfolio return
        weighted_returns = [
            pos_size * ret for pos_size, ret in zip(position_sizes, returns)
        ]
        portfolio_return = (
            sum(weighted_returns) / portfolio_value if portfolio_value > 0 else 0
        )

        # Calculate portfolio volatility
        portfolio_volatility = (
            np.std(weighted_returns) if len(weighted_returns) > 1 else 0
        )

        # Calculate maximum drawdown
        cumulative_returns = np.cumprod([1 + r for r in weighted_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        return {
            "concentration_ratio": concentration_ratio,
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "max_drawdown": abs(max_drawdown),
            "total_position_value": total_position_value,
            "position_count": len([s for s in position_sizes if s > 0]),
        }
