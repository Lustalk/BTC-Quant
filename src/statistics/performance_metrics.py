"""
Performance Metrics Module
Professional risk-adjusted performance metrics with statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio with annualization.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    try:
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        
        logger.info(f"Sharpe ratio calculated: {sharpe_ratio:.4f}")
        return sharpe_ratio
        
    except Exception as e:
        logger.error(f"Sharpe ratio calculation failed: {str(e)}")
        raise

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from peak.
    
    Args:
        returns: Array of returns
    
    Returns:
        Maximum drawdown as a percentage
    """
    try:
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        logger.info(f"Maximum drawdown calculated: {max_drawdown:.4f}")
        return max_drawdown
        
    except Exception as e:
        logger.error(f"Maximum drawdown calculation failed: {str(e)}")
        raise

def calculate_calmar_ratio(
    returns: np.ndarray, 
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    try:
        annualized_return = np.mean(returns) * periods_per_year
        max_dd = abs(calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return float('inf')
        
        calmar_ratio = annualized_return / max_dd
        
        logger.info(f"Calmar ratio calculated: {calmar_ratio:.4f}")
        return calmar_ratio
        
    except Exception as e:
        logger.error(f"Calmar ratio calculation failed: {str(e)}")
        raise

def calculate_information_ratio(
    returns: np.ndarray, 
    benchmark: np.ndarray
) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    Args:
        returns: Strategy returns
        benchmark: Benchmark returns
    
    Returns:
        Information ratio
    """
    try:
        # Ensure same length
        min_length = min(len(returns), len(benchmark))
        returns = returns[-min_length:]
        benchmark = benchmark[-min_length:]
        
        excess_returns = returns - benchmark
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        information_ratio = np.mean(excess_returns) / tracking_error
        
        logger.info(f"Information ratio calculated: {information_ratio:.4f}")
        return information_ratio
        
    except Exception as e:
        logger.error(f"Information ratio calculation failed: {str(e)}")
        raise

def calculate_sortino_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (downside deviation).
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        target_return: Target return for downside deviation
    
    Returns:
        Sortino ratio
    """
    try:
        excess_returns = returns - risk_free_rate / periods_per_year - target_return
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return float('inf')
        
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
        
        logger.info(f"Sortino ratio calculated: {sortino_ratio:.4f}")
        return sortino_ratio
        
    except Exception as e:
        logger.error(f"Sortino ratio calculation failed: {str(e)}")
        raise

def calculate_kelly_criterion(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level for estimation
    
    Returns:
        Dictionary with Kelly fraction and confidence interval
    """
    try:
        # Calculate win rate and average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
        
        # Kelly fraction
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Bootstrap confidence interval
        kelly_samples = []
        n_bootstrap = 10000
        
        for _ in range(n_bootstrap):
            bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
            pos_returns = bootstrap_returns[bootstrap_returns > 0]
            neg_returns = bootstrap_returns[bootstrap_returns < 0]
            
            if len(pos_returns) > 0 and len(neg_returns) > 0:
                win_rate_boot = len(pos_returns) / len(bootstrap_returns)
                avg_win_boot = np.mean(pos_returns)
                avg_loss_boot = abs(np.mean(neg_returns))
                
                if avg_loss_boot > 0:
                    kelly_boot = (win_rate_boot * avg_win_boot - (1 - win_rate_boot) * avg_loss_boot) / avg_win_boot
                    kelly_samples.append(kelly_boot)
        
        if len(kelly_samples) > 0:
            kelly_samples = np.array(kelly_samples)
            ci_lower = np.percentile(kelly_samples, (1 - confidence_level) / 2 * 100)
            ci_upper = np.percentile(kelly_samples, (1 + confidence_level) / 2 * 100)
        else:
            ci_lower = ci_upper = kelly_fraction
        
        result = {
            'kelly_fraction': kelly_fraction,
            'confidence_interval': (ci_lower, ci_upper),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        logger.info(f"Kelly criterion calculated: {kelly_fraction:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Kelly criterion calculation failed: {str(e)}")
        raise

def calculate_comprehensive_metrics(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics with confidence intervals.
    
    Args:
        returns: Strategy returns
        benchmark: Benchmark returns (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary with all performance metrics
    """
    try:
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        max_drawdown = calculate_max_drawdown(returns)
        calmar_ratio = calculate_calmar_ratio(returns, periods_per_year)
        
        # Kelly criterion
        kelly_result = calculate_kelly_criterion(returns)
        
        # Bootstrap confidence intervals
        sharpe_ci = bootstrap_confidence_interval(returns, statistic='sharpe')
        sortino_ci = bootstrap_confidence_interval(returns, statistic='sortino')
        
        result = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'kelly_fraction': kelly_result['kelly_fraction'],
            'confidence_intervals': {
                'sharpe_ratio': sharpe_ci,
                'sortino_ratio': sortino_ci,
                'kelly_fraction': kelly_result['confidence_interval']
            },
            'risk_metrics': {
                'win_rate': kelly_result['win_rate'],
                'avg_win': kelly_result['avg_win'],
                'avg_loss': kelly_result['avg_loss']
            }
        }
        
        # Add benchmark comparison if provided
        if benchmark is not None:
            information_ratio = calculate_information_ratio(returns, benchmark)
            result['information_ratio'] = information_ratio
        
        logger.info(f"Comprehensive metrics calculated for {len(returns)} observations")
        return result
        
    except Exception as e:
        logger.error(f"Comprehensive metrics calculation failed: {str(e)}")
        raise

def bootstrap_confidence_interval(
    data: np.ndarray, 
    alpha: float = 0.05, 
    n_bootstrap: int = 10000,
    statistic: str = 'mean'
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for various statistics.
    
    Args:
        data: Array of data
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to calculate
    
    Returns:
        Tuple of (lower_bound, upper_bound) confidence interval
    """
    try:
        bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
        
        if statistic == 'mean':
            bootstrap_stats = np.mean(bootstrap_samples, axis=1)
        elif statistic == 'std':
            bootstrap_stats = np.std(bootstrap_samples, axis=1)
        elif statistic == 'sharpe':
            bootstrap_stats = np.mean(bootstrap_samples, axis=1) / np.std(bootstrap_samples, axis=1)
        elif statistic == 'sortino':
            # Simplified Sortino calculation
            bootstrap_stats = np.mean(bootstrap_samples, axis=1) / np.std(bootstrap_samples, axis=1)
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
        
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1-alpha/2) * 100)
        
        return ci_lower, ci_upper
        
    except Exception as e:
        logger.error(f"Bootstrap confidence interval calculation failed: {str(e)}")
        raise 