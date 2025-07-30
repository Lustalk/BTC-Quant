"""
Risk Management Module
Position sizing and risk controls for trading strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from config import DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management and position sizing for trading strategy
    """

    def __init__(self):
        """Initialize risk manager"""
        self.position_sizes = None
        self.risk_metrics = {}
        self.risk_limits = {
            "max_position_size": 1.0,
            "max_drawdown": 0.20,
            "volatility_target": 0.15,
            "max_correlation": 0.7
        }

    def calculate_position_size(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        method: str = "volatility_targeting",
        target_volatility: float = None,
        max_position: float = None,
    ) -> pd.Series:
        """
        Calculate position sizes based on risk management rules
        
        Args:
            returns: Historical returns
            predictions: Model predictions
            method: Position sizing method
            target_volatility: Target annualized volatility
            max_position: Maximum position size
            
        Returns:
            Series with position sizes
        """
        logger.info(f"Calculating position sizes using {method} method...")
        
        target_volatility = target_volatility or self.risk_limits["volatility_target"]
        max_position = max_position or self.risk_limits["max_position_size"]
        
        if method == "volatility_targeting":
            position_sizes = self._volatility_targeting(
                returns, predictions, target_volatility, max_position
            )
        elif method == "kelly_criterion":
            position_sizes = self._kelly_criterion(returns, predictions, max_position)
        elif method == "fixed_fraction":
            position_sizes = self._fixed_fraction(predictions, max_position)
        elif method == "adaptive":
            position_sizes = self._adaptive_position_sizing(
                returns, predictions, target_volatility, max_position
            )
        else:
            position_sizes = self._volatility_targeting(
                returns, predictions, target_volatility, max_position
            )
        
        self.position_sizes = position_sizes
        return position_sizes
    
    def _volatility_targeting(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        target_volatility: float,
        max_position: float,
    ) -> pd.Series:
        """Volatility targeting position sizing"""
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Calculate position size based on volatility targeting
        position_sizes = target_volatility / rolling_vol
        
        # Apply maximum position limit
        position_sizes = np.minimum(position_sizes, max_position)
        
        # Apply prediction-based scaling
        position_sizes = position_sizes * predictions
        
        # Ensure non-negative positions
        position_sizes = np.maximum(position_sizes, 0)
        
        return position_sizes
    
    def _kelly_criterion(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        max_position: float,
    ) -> pd.Series:
        """Kelly criterion position sizing"""
        
        # Calculate win rate and average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1
        
        # Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        
        # Apply maximum position limit
        kelly_fraction = min(kelly_fraction, max_position)
        
        # Scale by predictions
        position_sizes = kelly_fraction * predictions
        
        return position_sizes
    
    def _fixed_fraction(
        self,
        predictions: pd.Series,
        max_position: float,
    ) -> pd.Series:
        """Fixed fraction position sizing"""
        
        # Simple fixed fraction based on predictions
        position_sizes = max_position * predictions
        
        return position_sizes
    
    def _adaptive_position_sizing(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        target_volatility: float,
        max_position: float,
    ) -> pd.Series:
        """Adaptive position sizing based on market conditions"""
        
        # Calculate market regime indicators
        rolling_vol = returns.rolling(window=60).std() * np.sqrt(252)
        rolling_sharpe = returns.rolling(window=60).mean() / returns.rolling(window=60).std()
        
        # Base position size from volatility targeting
        base_position = target_volatility / rolling_vol
        
        # Adjust based on Sharpe ratio
        sharpe_adjustment = np.clip(rolling_sharpe / 2, 0.5, 2.0)
        
        # Final position size
        position_sizes = base_position * sharpe_adjustment * predictions
        
        # Apply limits
        position_sizes = np.clip(position_sizes, 0, max_position)
        
        return position_sizes
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        position_sizes: pd.Series,
    ) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Strategy returns
            position_sizes: Position sizes
            
        Returns:
            Dictionary with risk metrics
        """
        logger.info("Calculating risk metrics...")
        
        # Strategy returns
        strategy_returns = returns * position_sizes
        
        # Basic risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Position sizing metrics
        avg_position_size = position_sizes.mean()
        max_position_size = position_sizes.max()
        position_volatility = position_sizes.std()
        
        # Risk-adjusted metrics
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Downside risk
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = strategy_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        risk_metrics = {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "avg_position_size": avg_position_size,
            "max_position_size": max_position_size,
            "position_volatility": position_volatility
        }
        
        self.risk_metrics = risk_metrics
        return risk_metrics
    
    def check_risk_limits(
        self,
        returns: pd.Series,
        position_sizes: pd.Series,
    ) -> Dict:
        """
        Check if strategy violates risk limits
        
        Args:
            returns: Strategy returns
            position_sizes: Position sizes
            
        Returns:
            Dictionary with risk limit violations
        """
        logger.info("Checking risk limits...")
        
        # Calculate current metrics
        risk_metrics = self.calculate_risk_metrics(returns, position_sizes)
        
        # Check limits
        violations = {}
        
        # Position size limit
        if risk_metrics["max_position_size"] > self.risk_limits["max_position_size"]:
            violations["position_size"] = {
                "limit": self.risk_limits["max_position_size"],
                "actual": risk_metrics["max_position_size"]
            }
        
        # Drawdown limit
        if abs(risk_metrics["max_drawdown"]) > self.risk_limits["max_drawdown"]:
            violations["drawdown"] = {
                "limit": self.risk_limits["max_drawdown"],
                "actual": abs(risk_metrics["max_drawdown"])
            }
        
        # Volatility limit
        if risk_metrics["volatility"] > self.risk_limits["volatility_target"]:
            violations["volatility"] = {
                "limit": self.risk_limits["volatility_target"],
                "actual": risk_metrics["volatility"]
            }
        
        return violations
    
    def apply_risk_controls(
        self,
        position_sizes: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Apply risk controls to position sizes
        
        Args:
            position_sizes: Original position sizes
            returns: Historical returns
            
        Returns:
            Adjusted position sizes
        """
        logger.info("Applying risk controls...")
        
        # Check for violations
        violations = self.check_risk_limits(returns, position_sizes)
        
        adjusted_positions = position_sizes.copy()
        
        # Apply position size limits
        if "position_size" in violations:
            max_allowed = self.risk_limits["max_position_size"]
            adjusted_positions = np.minimum(adjusted_positions, max_allowed)
        
        # Apply volatility targeting if needed
        if "volatility" in violations:
            target_vol = self.risk_limits["volatility_target"]
            current_vol = returns.std() * np.sqrt(252)
            vol_adjustment = target_vol / current_vol
            adjusted_positions = adjusted_positions * vol_adjustment
        
        # Apply drawdown protection
        if "drawdown" in violations:
            # Reduce position sizes during high drawdown periods
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Reduce positions when drawdown exceeds 10%
            drawdown_factor = np.where(drawdown < -0.1, 0.5, 1.0)
            adjusted_positions = adjusted_positions * drawdown_factor
        
        return adjusted_positions
    
    def export_risk_analysis(self, filepath: str = None) -> str:
        """Export risk analysis results"""
        
        if not filepath:
            filepath = "exports/risk_analysis.csv"
        
        if hasattr(self, 'risk_metrics'):
            risk_df = pd.DataFrame([self.risk_metrics])
            risk_df.to_csv(filepath, index=False)
            logger.info(f"Risk analysis exported to: {filepath}")
        
        return filepath


def main():
    """Main function for standalone execution"""
    logger.info("Risk management module loaded successfully")


if __name__ == "__main__":
    main() 