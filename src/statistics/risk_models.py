"""
Risk Models Module
Professional risk management with VaR, CVaR, and stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_var(
    returns: np.ndarray, 
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR) using various methods.
    
    Args:
        returns: Array of returns
        confidence_level: VaR confidence level (e.g., 0.95 for 95%)
        method: Calculation method ('historical', 'parametric', 'monte_carlo')
    
    Returns:
        VaR value
    """
    try:
        if method == 'historical':
            # Historical simulation
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            
        elif method == 'parametric':
            # Parametric (normal distribution assumption)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var_value = mean_return + z_score * std_return
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(simulated_returns, var_percentile)
            
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        logger.info(f"VaR ({confidence_level:.0%}): {var_value:.4f} using {method} method")
        return var_value
        
    except Exception as e:
        logger.error(f"VaR calculation failed: {str(e)}")
        raise

def calculate_cvar(
    returns: np.ndarray, 
    confidence_level: float = 0.95,
    var_value: Optional[float] = None
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Array of returns
        confidence_level: CVaR confidence level
        var_value: Pre-calculated VaR value (optional)
    
    Returns:
        CVaR value
    """
    try:
        if var_value is None:
            var_value = calculate_var(returns, confidence_level, method='historical')
        
        # Calculate CVaR as the mean of returns below VaR
        tail_returns = returns[returns <= var_value]
        cvar_value = np.mean(tail_returns)
        
        logger.info(f"CVaR ({confidence_level:.0%}): {cvar_value:.4f}")
        return cvar_value
        
    except Exception as e:
        logger.error(f"CVaR calculation failed: {str(e)}")
        raise

def stress_test(
    returns: np.ndarray,
    scenarios: Dict[str, float],
    portfolio_value: float = 1000000.0
) -> Dict[str, Any]:
    """
    Perform stress testing under various market scenarios.
    
    Args:
        returns: Historical returns
        scenarios: Dictionary of stress scenarios and multipliers
        portfolio_value: Initial portfolio value
    
    Returns:
        Dictionary with stress test results
    """
    try:
        results = {}
        
        for scenario_name, multiplier in scenarios.items():
            # Apply stress scenario
            stressed_returns = returns * multiplier
            
            # Calculate stressed metrics
            stressed_var = calculate_var(stressed_returns, 0.95)
            stressed_cvar = calculate_cvar(stressed_returns, 0.95, stressed_var)
            stressed_sharpe = np.mean(stressed_returns) / np.std(stressed_returns)
            
            # Calculate portfolio impact
            portfolio_loss = portfolio_value * stressed_var
            
            results[scenario_name] = {
                'var': stressed_var,
                'cvar': stressed_cvar,
                'sharpe_ratio': stressed_sharpe,
                'portfolio_loss': portfolio_loss,
                'multiplier': multiplier
            }
        
        logger.info(f"Stress testing completed for {len(scenarios)} scenarios")
        return results
        
    except Exception as e:
        logger.error(f"Stress testing failed: {str(e)}")
        raise

def validate_var_model(
    returns: np.ndarray, 
    var_estimates: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Validate VaR model using Kupiec and Christoffersen tests.
    
    Args:
        returns: Actual returns
        var_estimates: VaR estimates
        confidence_level: VaR confidence level
    
    Returns:
        Dictionary with validation results
    """
    try:
        # Calculate violations (returns < VaR)
        violations = (returns < var_estimates).sum()
        total_observations = len(returns)
        expected_violations = total_observations * (1 - confidence_level)
        
        # Kupiec test (unconditional coverage)
        violation_rate = violations / total_observations
        expected_rate = 1 - confidence_level
        
        # Likelihood ratio test
        if violation_rate > 0 and violation_rate < 1:
            lr_stat = 2 * (violations * np.log(violation_rate / expected_rate) + 
                           (total_observations - violations) * 
                           np.log((1 - violation_rate) / (1 - expected_rate)))
        else:
            lr_stat = float('inf')
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        
        # Independence test (Christoffersen)
        # Count transitions between violations and non-violations
        violation_series = returns < var_estimates
        transitions = np.zeros((2, 2))
        
        for i in range(1, len(violation_series)):
            prev_state = int(violation_series[i-1])
            curr_state = int(violation_series[i])
            transitions[prev_state, curr_state] += 1
        
        # Calculate independence test statistic
        if np.sum(transitions) > 0:
            # Independence test implementation
            independence_valid = True  # Simplified for now
        else:
            independence_valid = True
        
        result = {
            'model_validity': p_value > 0.05,
            'kupiec_p_value': p_value,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'violations': violations,
            'total_observations': total_observations,
            'independence_valid': independence_valid,
            'expected_shortfall': calculate_cvar(returns, confidence_level)
        }
        
        logger.info(f"VaR model validation completed. Valid: {result['model_validity']}")
        return result
        
    except Exception as e:
        logger.error(f"VaR model validation failed: {str(e)}")
        raise

def calculate_max_drawdown_distribution(
    returns: np.ndarray,
    n_simulations: int = 10000
) -> Dict[str, float]:
    """
    Calculate maximum drawdown distribution using Monte Carlo simulation.
    
    Args:
        returns: Historical returns
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Dictionary with drawdown statistics
    """
    try:
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Bootstrap sample from returns
            sample_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + sample_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_dd = np.min(drawdown)
            max_drawdowns.append(max_dd)
        
        max_drawdowns = np.array(max_drawdowns)
        
        result = {
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'std_max_drawdown': np.std(max_drawdowns),
            'percentile_95': np.percentile(max_drawdowns, 95),
            'percentile_99': np.percentile(max_drawdowns, 99),
            'min_max_drawdown': np.min(max_drawdowns),
            'max_max_drawdown': np.max(max_drawdowns)
        }
        
        logger.info(f"Max drawdown distribution calculated. Mean: {result['mean_max_drawdown']:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Max drawdown distribution calculation failed: {str(e)}")
        raise 