"""
Threshold Optimization Module
Optimizes probability thresholds for trading decisions
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

from config import DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes probability thresholds for trading decisions
    """

    def __init__(self):
        """Initialize threshold optimizer"""
        self.optimal_thresholds = {}
        self.threshold_analysis = {}

    def calculate_optimal_threshold_for_returns(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        method: str = "sharpe_max",
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        step: float = 0.01,
    ) -> float:
        """
        Calculate optimal threshold based on financial performance
        
        Args:
            returns: Actual returns
            predictions: Model probability predictions
            method: Optimization method ('sharpe_max', 'sortino_max', 'return_max')
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            step: Step size for threshold testing
            
        Returns:
            Optimal threshold value
        """
        logger.info(f"Calculating optimal threshold using {method} method...")
        
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        results = []
        
        for threshold in thresholds:
            # Create signals based on threshold
            signals = (predictions > threshold).astype(int)
            
            # Calculate strategy returns
            strategy_returns = signals * returns
            
            # Calculate metrics
            metrics = self._calculate_threshold_metrics(strategy_returns, threshold)
            results.append(metrics)
        
        # Find optimal threshold based on method
        results_df = pd.DataFrame(results)
        
        if method == "sharpe_max":
            optimal_idx = results_df["sharpe_ratio"].idxmax()
        elif method == "sortino_max":
            optimal_idx = results_df["sortino_ratio"].idxmax()
        elif method == "return_max":
            optimal_idx = results_df["total_return"].idxmax()
        else:
            optimal_idx = results_df["sharpe_ratio"].idxmax()
        
        optimal_threshold = results_df.loc[optimal_idx, "threshold"]
        
        # Store results
        self.threshold_analysis = {
            "method": method,
            "optimal_threshold": optimal_threshold,
            "all_results": results_df,
            "optimal_metrics": results_df.loc[optimal_idx].to_dict()
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"Optimal Sharpe: {results_df.loc[optimal_idx, 'sharpe_ratio']:.3f}")
        
        return optimal_threshold
    
    def _calculate_threshold_metrics(
        self, strategy_returns: pd.Series, threshold: float
    ) -> Dict:
        """Calculate performance metrics for a given threshold"""
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                "threshold": threshold,
                "total_return": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "signal_count": 0
            }
        
        # Basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        # Signal count
        signal_count = (strategy_returns != 0).sum()
        
        return {
            "threshold": threshold,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "signal_count": signal_count
        }
    
    def optimize_threshold_by_period(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        period_length: int = 252,
    ) -> Dict:
        """
        Optimize thresholds for different time periods
        
        Args:
            data: DataFrame with returns and other data
            predictions: Model predictions
            period_length: Length of each period in days
            
        Returns:
            Dictionary with period-specific optimal thresholds
        """
        logger.info("Optimizing thresholds by period...")
        
        results = {}
        returns = data["returns"]
        
        # Split data into periods
        periods = len(data) // period_length
        
        for i in range(periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length
            
            period_returns = returns.iloc[start_idx:end_idx]
            period_predictions = predictions.iloc[start_idx:end_idx]
            
            # Calculate optimal threshold for this period
            optimal_threshold = self.calculate_optimal_threshold_for_returns(
                period_returns, period_predictions
            )
            
            results[f"period_{i+1}"] = {
                "start_date": data.index[start_idx],
                "end_date": data.index[end_idx-1],
                "optimal_threshold": optimal_threshold,
                "metrics": self.threshold_analysis["optimal_metrics"]
            }
        
        return results
    
    def calculate_dynamic_threshold(
        self,
        predictions: pd.Series,
        window: int = 60,
        method: str = "rolling_percentile",
    ) -> pd.Series:
        """
        Calculate dynamic thresholds that adapt over time
        
        Args:
            predictions: Model predictions
            window: Rolling window size
            method: Method for dynamic threshold ('rolling_percentile', 'rolling_mean')
            
        Returns:
            Series with dynamic thresholds
        """
        logger.info("Calculating dynamic thresholds...")
        
        if method == "rolling_percentile":
            # Use 75th percentile as threshold
            dynamic_thresholds = predictions.rolling(window=window).quantile(0.75)
        elif method == "rolling_mean":
            # Use mean + 1 standard deviation
            rolling_mean = predictions.rolling(window=window).mean()
            rolling_std = predictions.rolling(window=window).std()
            dynamic_thresholds = rolling_mean + rolling_std
        else:
            # Default to rolling percentile
            dynamic_thresholds = predictions.rolling(window=window).quantile(0.75)
        
        # Fill NaN values with median
        dynamic_thresholds = dynamic_thresholds.fillna(predictions.median())
        
        return dynamic_thresholds
    
    def export_threshold_analysis(self, filepath: str = None) -> str:
        """Export threshold analysis results"""
        
        if not filepath:
            filepath = "exports/threshold_analysis.csv"
        
        if hasattr(self, 'threshold_analysis') and 'all_results' in self.threshold_analysis:
            self.threshold_analysis['all_results'].to_csv(filepath, index=False)
            logger.info(f"Threshold analysis exported to: {filepath}")
        
        return filepath


def main():
    """Main function for standalone execution"""
    logger.info("Threshold optimization module loaded successfully")


if __name__ == "__main__":
    main() 