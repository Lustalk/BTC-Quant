"""
Simplified Strategy Analysis Module
Focus on core functionality without over-engineering
"""

import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from scipy import stats
import os

warnings.filterwarnings("ignore")

from config import EVALUATION_CONFIG, DATA_CONFIG, ADVANCED_CONFIG
from src.evaluation import PerformanceEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    """
    Simplified strategy analysis focusing on core functionality
    """

    def __init__(self):
        """Initialize strategy analyzer"""
        self.results = {}
        
    def analyze_strategy(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str = "target",
    ) -> Dict:
        """
        Core strategy analysis with essential functionality
        
        Args:
            data: Full dataset with features and target
            model: Trained model object
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("=" * 80)
        logger.info("CORE STRATEGY ANALYSIS")
        logger.info("=" * 80)
        
        # Get model predictions
        X = data[feature_columns]
        predictions = model.predict(X)
        
        # Calculate strategy returns
        signals = pd.Series((predictions > 0.5).astype(int), index=data.index)
        strategy_returns = signals.shift(1) * data['returns'].fillna(0)
        strategy_returns = strategy_returns.fillna(0)
        
        # Calculate buy & hold returns
        buy_hold_returns = data['returns'].fillna(0)
        
        # Calculate performance metrics
        strategy_sharpe = self._calculate_sharpe_ratio(strategy_returns)
        buy_hold_sharpe = self._calculate_sharpe_ratio(buy_hold_returns)
        
        strategy_return = strategy_returns.sum()
        buy_hold_return = buy_hold_returns.sum()
        
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        buy_hold_volatility = buy_hold_returns.std() * np.sqrt(252)
        
        # Calculate maximum drawdown
        strategy_cumulative = (1 + strategy_returns).cumprod()
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        
        strategy_drawdown = (strategy_cumulative / strategy_cumulative.cummax() - 1).min()
        buy_hold_drawdown = (buy_hold_cumulative / buy_hold_cumulative.cummax() - 1).min()
        
        # Simple statistical test
        returns_diff = strategy_returns - buy_hold_returns
        t_stat, p_value = stats.ttest_1samp(returns_diff, 0)
        
        # Compile results
        self.results = {
            "strategy_sharpe": strategy_sharpe,
            "buy_hold_sharpe": buy_hold_sharpe,
            "sharpe_improvement": strategy_sharpe - buy_hold_sharpe,
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "return_improvement": strategy_return - buy_hold_return,
            "strategy_volatility": strategy_volatility,
            "buy_hold_volatility": buy_hold_volatility,
            "strategy_drawdown": strategy_drawdown,
            "buy_hold_drawdown": buy_hold_drawdown,
            "t_statistic": t_stat,
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log results
        logger.info(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
        logger.info(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.4f}")
        logger.info(f"Sharpe Improvement: {strategy_sharpe - buy_hold_sharpe:.4f}")
        logger.info(f"Strategy Return: {strategy_return:.4f}")
        logger.info(f"Buy & Hold Return: {buy_hold_return:.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        logger.info(f"Statistically Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        return self.results
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def export_analysis_results(self, filepath: str = None) -> str:
        """
        Export analysis results to file
        """
        if filepath is None:
            os.makedirs("results", exist_ok=True)
            filepath = f"results/strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filepath, 'w') as f:
            f.write("BTC Strategy Analysis Results\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Strategy Sharpe Ratio: {self.results['strategy_sharpe']:.4f}\n")
            f.write(f"Buy & Hold Sharpe Ratio: {self.results['buy_hold_sharpe']:.4f}\n")
            f.write(f"Sharpe Improvement: {self.results['sharpe_improvement']:.4f}\n\n")
            
            f.write(f"Strategy Total Return: {self.results['strategy_return']:.4f}\n")
            f.write(f"Buy & Hold Total Return: {self.results['buy_hold_return']:.4f}\n")
            f.write(f"Return Improvement: {self.results['return_improvement']:.4f}\n\n")
            
            f.write(f"Strategy Volatility: {self.results['strategy_volatility']:.4f}\n")
            f.write(f"Buy & Hold Volatility: {self.results['buy_hold_volatility']:.4f}\n\n")
            
            f.write(f"Strategy Max Drawdown: {self.results['strategy_drawdown']:.4f}\n")
            f.write(f"Buy & Hold Max Drawdown: {self.results['buy_hold_drawdown']:.4f}\n\n")
            
            f.write(f"T-statistic: {self.results['t_statistic']:.4f}\n")
            f.write(f"P-value: {self.results['p_value']:.4f}\n")
            f.write(f"Statistically Significant: {'YES' if self.results['statistically_significant'] else 'NO'}\n")
        
        logger.info(f"Results exported to {filepath}")
        return filepath
    
    def print_comprehensive_report(self):
        """
        Print comprehensive analysis report
        """
        print("\n" + "="*80)
        print("STRATEGY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nPerformance Metrics:")
        print(f"  Strategy Sharpe Ratio: {self.results['strategy_sharpe']:.4f}")
        print(f"  Buy & Hold Sharpe Ratio: {self.results['buy_hold_sharpe']:.4f}")
        print(f"  Sharpe Improvement: {self.results['sharpe_improvement']:.4f}")
        
        print(f"\nReturn Analysis:")
        print(f"  Strategy Total Return: {self.results['strategy_return']:.4f}")
        print(f"  Buy & Hold Total Return: {self.results['buy_hold_return']:.4f}")
        print(f"  Return Improvement: {self.results['return_improvement']:.4f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Strategy Volatility: {self.results['strategy_volatility']:.4f}")
        print(f"  Buy & Hold Volatility: {self.results['buy_hold_volatility']:.4f}")
        print(f"  Strategy Max Drawdown: {self.results['strategy_drawdown']:.4f}")
        print(f"  Buy & Hold Max Drawdown: {self.results['buy_hold_drawdown']:.4f}")
        
        print(f"\nStatistical Significance:")
        print(f"  T-statistic: {self.results['t_statistic']:.4f}")
        print(f"  P-value: {self.results['p_value']:.4f}")
        print(f"  Statistically Significant: {'YES' if self.results['statistically_significant'] else 'NO'}")
        
        if self.results['statistically_significant']:
            print(f"\n✅ CONCLUSION: Strategy shows statistically significant improvement over buy & hold")
        else:
            print(f"\n⚠️  CONCLUSION: Strategy does not show statistically significant improvement")


def main():
    """
    Test the strategy analyzer
    """
    logger.info("Testing Strategy Analyzer...")
    
    # This would be called from main.py
    pass


if __name__ == "__main__":
    main() 