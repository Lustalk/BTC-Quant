"""
Dynamic Threshold Optimization Module
Optimizes probability thresholds for trading signals based on recent performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
from typing import Tuple, Optional
from config import STRATEGY_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    Dynamic threshold optimization for trading signals
    """
    
    def __init__(self, optimization_window: int = 252):
        """
        Initialize threshold optimizer
        
        Args:
            optimization_window: Number of days to use for threshold optimization
        """
        self.optimization_window = optimization_window
        self.current_threshold = 0.52  # Default threshold
        
    def optimize_threshold(self, 
                          y_true: pd.Series, 
                          y_proba: pd.Series,
                          method: str = 'f1_max') -> float:
        """
        Optimize probability threshold based on recent performance
        
        Args:
            y_true: True labels (1 for positive, 0 for negative)
            y_proba: Predicted probabilities
            method: Optimization method ('f1_max', 'precision_max', 'recall_max')
            
        Returns:
            Optimized threshold value
        """
        logger.info(f"Optimizing threshold using {method} method...")
        
        # Use only recent data for optimization
        if len(y_true) > self.optimization_window:
            y_true = y_true.iloc[-self.optimization_window:]
            y_proba = y_proba.iloc[-self.optimization_window:]
        
        # Test different thresholds
        thresholds = np.arange(0.3, 0.8, 0.01)
        best_score = 0
        best_threshold = 0.52
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if method == 'f1_max':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif method == 'precision_max':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif method == 'recall_max':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.current_threshold = best_threshold
        logger.info(f"Optimized threshold: {best_threshold:.3f} (score: {best_score:.3f})")
        
        return best_threshold
    
    def get_threshold(self) -> float:
        """Get current threshold value"""
        return self.current_threshold
    
    def update_threshold(self, new_threshold: float):
        """Update current threshold value"""
        self.current_threshold = new_threshold
        logger.info(f"Updated threshold to: {new_threshold:.3f}")
    
    def calculate_optimal_threshold_for_returns(self,
                                              returns: pd.Series,
                                              predictions: pd.Series,
                                              method: str = 'sharpe_max') -> float:
        """
        Optimize threshold based on financial performance metrics
        
        Args:
            returns: Actual returns
            predictions: Predicted probabilities
            method: Optimization method ('sharpe_max', 'return_max', 'sortino_max')
            
        Returns:
            Optimized threshold value
        """
        logger.info(f"Optimizing threshold for financial performance using {method}...")
        
        # Use only recent data for optimization
        if len(returns) > self.optimization_window:
            returns = returns.iloc[-self.optimization_window:]
            predictions = predictions.iloc[-self.optimization_window:]
        
        thresholds = np.arange(0.3, 0.8, 0.01)
        best_score = float('-inf')
        best_threshold = 0.52
        
        for threshold in thresholds:
            # Generate trading signals
            signals = (predictions >= threshold).astype(int)
            
            # Calculate strategy returns
            strategy_returns = signals * returns
            
            if method == 'sharpe_max':
                if strategy_returns.std() > 0:
                    score = strategy_returns.mean() / strategy_returns.std()
                else:
                    score = float('-inf')
            elif method == 'return_max':
                score = strategy_returns.sum()
            elif method == 'sortino_max':
                negative_returns = strategy_returns[strategy_returns < 0]
                if len(negative_returns) > 0 and negative_returns.std() > 0:
                    score = strategy_returns.mean() / negative_returns.std()
                else:
                    score = float('-inf')
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.current_threshold = best_threshold
        logger.info(f"Optimized threshold: {best_threshold:.3f} (score: {best_score:.3f})")
        
        return best_threshold

def main():
    """Test threshold optimization"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample predictions and returns
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_proba = np.random.beta(2, 2, n_samples)
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Test threshold optimization
    optimizer = ThresholdOptimizer(optimization_window=252)
    
    # Test classification-based optimization
    threshold_clf = optimizer.optimize_threshold(
        pd.Series(y_true), 
        pd.Series(y_proba), 
        method='f1_max'
    )
    
    # Test financial performance-based optimization
    threshold_fin = optimizer.calculate_optimal_threshold_for_returns(
        pd.Series(returns),
        pd.Series(y_proba),
        method='sharpe_max'
    )
    
    print(f"Classification-optimized threshold: {threshold_clf:.3f}")
    print(f"Financial-optimized threshold: {threshold_fin:.3f}")

if __name__ == "__main__":
    main() 