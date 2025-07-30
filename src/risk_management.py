"""
Risk Management and Regime Detection Module
Implements volatility clustering identification and position sizing based on volatility targeting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from config import ADVANCED_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes based on volatility clustering
    """
    
    def __init__(self, window: int = 60):
        """
        Initialize regime detector
        
        Args:
            window: Window size for regime detection
        """
        self.window = window
        self.regimes = None
        
    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect market regimes based on volatility clustering
        
        Args:
            returns: Asset returns
            
        Returns:
            Series with regime labels (0: low volatility, 1: high volatility)
        """
        logger.info("Detecting market regimes...")
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.window).std() * np.sqrt(252)
        
        # Calculate volatility percentile
        vol_percentile = volatility.rolling(window=self.window).rank(pct=True)
        
        # Define regimes based on volatility percentile
        # Low volatility regime: bottom 40%
        # High volatility regime: top 40%
        # Transition regime: middle 20%
        
        regimes = pd.Series(index=returns.index, dtype=int)
        regimes[vol_percentile <= 0.4] = 0  # Low volatility
        regimes[vol_percentile >= 0.6] = 1  # High volatility
        regimes[(vol_percentile > 0.4) & (vol_percentile < 0.6)] = 2  # Transition
        
        self.regimes = regimes
        logger.info(f"Regime detection complete. Regimes: {regimes.value_counts().to_dict()}")
        
        return regimes
    
    def get_regime_statistics(self, returns: pd.Series) -> Dict:
        """
        Calculate statistics for each regime
        
        Args:
            returns: Asset returns
            
        Returns:
            Dictionary with regime statistics
        """
        if self.regimes is None:
            self.detect_regimes(returns)
        
        stats = {}
        for regime in [0, 1, 2]:
            regime_returns = returns[self.regimes == regime]
            if len(regime_returns) > 0:
                stats[f'regime_{regime}'] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': regime_returns.mean() / (regime_returns.std() + 1e-8),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns)
                }
        
        return stats
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_current_regime(self) -> Optional[int]:
        """Get the most recent regime"""
        if self.regimes is not None:
            return self.regimes.iloc[-1]
        return None

class RiskManager:
    """
    Implements risk management features including volatility targeting and position sizing
    """
    
    def __init__(self, volatility_target: float = 0.15, max_position_size: float = 1.0):
        """
        Initialize risk manager
        
        Args:
            volatility_target: Target annualized volatility (default 15%)
            max_position_size: Maximum position size (default 100%)
        """
        self.volatility_target = volatility_target
        self.max_position_size = max_position_size
        self.regime_detector = RegimeDetector()
        
    def calculate_position_size(self, 
                              returns: pd.Series,
                              predictions: pd.Series,
                              method: str = 'volatility_targeting') -> pd.Series:
        """
        Calculate position sizes based on risk management rules
        
        Args:
            returns: Asset returns
            predictions: Model predictions
            method: Position sizing method
            
        Returns:
            Series with position sizes
        """
        logger.info(f"Calculating position sizes using {method} method...")
        
        if method == 'volatility_targeting':
            return self._volatility_targeted_sizing(returns, predictions)
        elif method == 'regime_based':
            return self._regime_based_sizing(returns, predictions)
        elif method == 'kelly_criterion':
            return self._kelly_criterion_sizing(returns, predictions)
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def _volatility_targeted_sizing(self, 
                                   returns: pd.Series, 
                                   predictions: pd.Series) -> pd.Series:
        """
        Volatility targeting position sizing
        
        Args:
            returns: Asset returns
            predictions: Model predictions
            
        Returns:
            Position sizes
        """
        # Calculate rolling volatility
        volatility = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Calculate position size based on volatility targeting
        # Position size = target_vol / current_vol
        position_size = self.volatility_target / (volatility + 1e-8)
        
        # Apply maximum position size constraint
        position_size = position_size.clip(0, self.max_position_size)
        
        # Only take positions when model predicts positive return
        position_size = position_size * (predictions >= 0.52).astype(float)
        
        return position_size
    
    def _regime_based_sizing(self, 
                            returns: pd.Series, 
                            predictions: pd.Series) -> pd.Series:
        """
        Regime-based position sizing
        
        Args:
            returns: Asset returns
            predictions: Model predictions
            
        Returns:
            Position sizes
        """
        # Detect regimes
        regimes = self.regime_detector.detect_regimes(returns)
        
        # Calculate base position size
        position_size = pd.Series(0.0, index=returns.index)
        
        # Different position sizes for different regimes
        # Low volatility regime: full position
        # High volatility regime: reduced position
        # Transition regime: moderate position
        
        position_size[regimes == 0] = self.max_position_size  # Low volatility
        position_size[regimes == 1] = self.max_position_size * 0.5  # High volatility
        position_size[regimes == 2] = self.max_position_size * 0.75  # Transition
        
        # Only take positions when model predicts positive return
        position_size = position_size * (predictions >= 0.52).astype(float)
        
        return position_size
    
    def _kelly_criterion_sizing(self, 
                               returns: pd.Series, 
                               predictions: pd.Series) -> pd.Series:
        """
        Kelly Criterion position sizing
        
        Args:
            returns: Asset returns
            predictions: Model predictions
            
        Returns:
            Position sizes
        """
        # Calculate Kelly fraction based on win rate and average win/loss
        # Kelly = (p * b - q) / b
        # where p = win rate, b = avg_win/avg_loss, q = 1-p
        
        # Use rolling windows to calculate Kelly fraction
        window = 252  # 1 year
        position_size = pd.Series(0.0, index=returns.index)
        
        for i in range(window, len(returns)):
            # Get recent data
            recent_returns = returns.iloc[i-window:i]
            recent_predictions = predictions.iloc[i-window:i]
            
            # Calculate win rate and average win/loss for predicted positive returns
            positive_predictions = recent_predictions >= 0.52
            if positive_predictions.sum() > 0:
                predicted_returns = recent_returns[positive_predictions]
                
                if len(predicted_returns) > 0:
                    win_rate = (predicted_returns > 0).mean()
                    avg_win = predicted_returns[predicted_returns > 0].mean()
                    avg_loss = abs(predicted_returns[predicted_returns < 0].mean())
                    
                    if avg_loss > 0:
                        b = avg_win / avg_loss
                        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
                        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
                        
                        # Apply to current position if model predicts positive
                        if predictions.iloc[i] >= 0.52:
                            position_size.iloc[i] = kelly_fraction
        
        return position_size
    
    def calculate_risk_metrics(self, 
                              returns: pd.Series, 
                              strategy_returns: pd.Series) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Asset returns
            strategy_returns: Strategy returns
            
        Returns:
            Dictionary with risk metrics
        """
        # Basic risk metrics
        metrics = {
            'volatility': strategy_returns.std() * np.sqrt(252),
            'sharpe_ratio': strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252),
            'sortino_ratio': strategy_returns.mean() / (strategy_returns[strategy_returns < 0].std() + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'var_95': strategy_returns.quantile(0.05),
            'cvar_95': strategy_returns[strategy_returns <= strategy_returns.quantile(0.05)].mean(),
            'win_rate': (strategy_returns > 0).mean(),
            'avg_win': strategy_returns[strategy_returns > 0].mean(),
            'avg_loss': strategy_returns[strategy_returns < 0].mean(),
        }
        
        # Regime analysis
        if ADVANCED_CONFIG['regime_detection']['enabled']:
            regimes = self.regime_detector.detect_regimes(returns)
            regime_stats = self.regime_detector.get_regime_statistics(strategy_returns)
            metrics['regime_analysis'] = regime_stats
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

def main():
    """Test risk management functionality"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample returns and predictions
    returns = np.random.normal(0.001, 0.02, n_samples)
    predictions = np.random.beta(2, 2, n_samples)
    
    # Test regime detection
    regime_detector = RegimeDetector()
    regimes = regime_detector.detect_regimes(pd.Series(returns))
    regime_stats = regime_detector.get_regime_statistics(pd.Series(returns))
    
    print("Regime Statistics:")
    for regime, stats in regime_stats.items():
        print(f"{regime}: {stats}")
    
    # Test risk management
    risk_manager = RiskManager()
    
    # Test different position sizing methods
    vol_targeted = risk_manager.calculate_position_size(
        pd.Series(returns), 
        pd.Series(predictions), 
        method='volatility_targeting'
    )
    
    regime_based = risk_manager.calculate_position_size(
        pd.Series(returns), 
        pd.Series(predictions), 
        method='regime_based'
    )
    
    print(f"\nVolatility targeted position size mean: {vol_targeted.mean():.3f}")
    print(f"Regime-based position size mean: {regime_based.mean():.3f}")

if __name__ == "__main__":
    main() 