"""
Statistical Significance Testing Module
Professional statistical validation for quantitative trading strategies.
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def bootstrap_confidence_interval(
    data: np.ndarray, 
    alpha: float = 0.05, 
    n_bootstrap: int = 10000,
    statistic: str = 'mean'
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for performance metrics.
    
    Args:
        data: Array of returns or performance data
        alpha: Significance level (default: 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to calculate ('mean', 'median', 'std')
    
    Returns:
        Tuple of (lower_bound, upper_bound) confidence interval
    """
    try:
        bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
        
        if statistic == 'mean':
            bootstrap_stats = np.mean(bootstrap_samples, axis=1)
        elif statistic == 'median':
            bootstrap_stats = np.median(bootstrap_samples, axis=1)
        elif statistic == 'std':
            bootstrap_stats = np.std(bootstrap_samples, axis=1)
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
        
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1-alpha/2) * 100)
        
        logger.info(f"Bootstrap CI ({1-alpha:.0%}): ({ci_lower:.4f}, {ci_upper:.4f})")
        return ci_lower, ci_upper
        
    except Exception as e:
        logger.error(f"Bootstrap confidence interval calculation failed: {str(e)}")
        raise

def validate_strategy_performance(
    returns: np.ndarray, 
    benchmark: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive statistical validation with multiple testing correction.
    
    Args:
        returns: Strategy returns
        benchmark: Benchmark returns (e.g., buy-and-hold)
        alpha: Significance level
    
    Returns:
        Dictionary with statistical validation results
    """
    try:
        # Ensure arrays have same length
        min_length = min(len(returns), len(benchmark))
        returns = returns[-min_length:]
        benchmark = benchmark[-min_length:]
        
        # T-test for mean difference
        t_stat, p_value = stats.ttest_ind(returns, benchmark)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = bootstrap_confidence_interval(returns - benchmark, alpha=alpha)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(returns) - 1) * np.var(returns) + 
                             (len(benchmark) - 1) * np.var(benchmark)) / 
                            (len(returns) + len(benchmark) - 2))
        effect_size = (np.mean(returns) - np.mean(benchmark)) / pooled_std
        
        # Power analysis
        power = calculate_statistical_power(returns, benchmark, alpha)
        
        result = {
            'statistical_significance': p_value < alpha,
            'p_value': p_value,
            't_statistic': t_stat,
            'confidence_interval': (ci_lower, ci_upper),
            'mean_difference': np.mean(returns) - np.mean(benchmark),
            'effect_size': effect_size,
            'statistical_power': power,
            'sample_size': len(returns),
            'alpha_level': alpha
        }
        
        logger.info(f"Strategy validation completed. Significant: {result['statistical_significance']}")
        return result
        
    except Exception as e:
        logger.error(f"Strategy performance validation failed: {str(e)}")
        raise

def calculate_statistical_power(
    returns: np.ndarray, 
    benchmark: np.ndarray, 
    alpha: float = 0.05
) -> float:
    """
    Calculate statistical power for the comparison.
    
    Args:
        returns: Strategy returns
        benchmark: Benchmark returns
        alpha: Significance level
    
    Returns:
        Statistical power (0-1)
    """
    try:
        # Calculate effect size
        pooled_std = np.sqrt(((len(returns) - 1) * np.var(returns) + 
                             (len(benchmark) - 1) * np.var(benchmark)) / 
                            (len(returns) + len(benchmark) - 2))
        effect_size = abs(np.mean(returns) - np.mean(benchmark)) / pooled_std
        
        # Calculate power using normal approximation
        n = len(returns)
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
        z_beta = effect_size * np.sqrt(n/2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))  # Ensure power is between 0 and 1
        
    except Exception as e:
        logger.error(f"Power calculation failed: {str(e)}")
        return 0.0

def multiple_testing_correction(
    p_values: np.ndarray, 
    method: str = 'bonferroni'
) -> np.ndarray:
    """
    Apply multiple testing correction to control family-wise error rate.
    
    Args:
        p_values: Array of p-values
        method: Correction method ('bonferroni', 'fdr_bh')
    
    Returns:
        Corrected p-values
    """
    try:
        if method == 'bonferroni':
            return np.minimum(p_values * len(p_values), 1.0)
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            n = len(p_values)
            
            corrected_p_values = np.zeros_like(p_values)
            for i, (original_idx, p_val) in enumerate(zip(sorted_indices, sorted_p_values)):
                corrected_p_val = p_val * n / (i + 1)
                corrected_p_values[original_idx] = min(corrected_p_val, 1.0)
            
            return corrected_p_values
        else:
            raise ValueError(f"Unsupported correction method: {method}")
            
    except Exception as e:
        logger.error(f"Multiple testing correction failed: {str(e)}")
        raise 