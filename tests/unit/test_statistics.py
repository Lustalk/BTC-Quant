"""
Unit tests for statistics module.
"""

import pytest
import numpy as np
from src.statistics.significance_tests import validate_strategy_performance, bootstrap_confidence_interval
from src.statistics.risk_models import calculate_var, calculate_cvar
from src.statistics.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown

class TestStatistics:
    """Test class for statistics module."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_returns = np.random.normal(0.001, 0.02, 1000)
        self.benchmark_returns = np.random.normal(0.0008, 0.018, 1000)
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        ci_lower, ci_upper = bootstrap_confidence_interval(self.sample_returns, alpha=0.05)
        
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
    
    def test_validate_strategy_performance(self):
        """Test strategy performance validation."""
        result = validate_strategy_performance(self.sample_returns, self.benchmark_returns)
        
        assert 'statistical_significance' in result
        assert 'p_value' in result
        assert 'confidence_interval' in result
        assert isinstance(result['statistical_significance'], bool)
        assert 0 <= result['p_value'] <= 1
    
    def test_calculate_var(self):
        """Test VaR calculation."""
        var_value = calculate_var(self.sample_returns, confidence_level=0.95)
        
        assert isinstance(var_value, float)
        assert var_value < 0  # VaR should be negative for normal returns
    
    def test_calculate_cvar(self):
        """Test CVaR calculation."""
        cvar_value = calculate_cvar(self.sample_returns, confidence_level=0.95)
        
        assert isinstance(cvar_value, float)
        assert cvar_value < 0  # CVaR should be negative for normal returns
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(self.sample_returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd = calculate_max_drawdown(self.sample_returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Maximum drawdown should be negative or zero
        assert not np.isnan(max_dd)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty array
        with pytest.raises(Exception):
            bootstrap_confidence_interval(np.array([]))
        
        # Test with single value
        with pytest.raises(Exception):
            bootstrap_confidence_interval(np.array([1.0]))
        
        # Test with all zeros
        zero_returns = np.zeros(100)
        sharpe = calculate_sharpe_ratio(zero_returns)
        assert np.isnan(sharpe) or np.isinf(sharpe)
    
    def test_statistical_properties(self):
        """Test statistical properties of the calculations."""
        # Test that confidence intervals are reasonable
        ci_lower, ci_upper = bootstrap_confidence_interval(self.sample_returns, alpha=0.05)
        mean_return = np.mean(self.sample_returns)
        
        assert ci_lower <= mean_return <= ci_upper
        
        # Test that VaR is more negative than CVaR
        var_value = calculate_var(self.sample_returns, confidence_level=0.95)
        cvar_value = calculate_cvar(self.sample_returns, confidence_level=0.95)
        
        assert var_value >= cvar_value  # VaR should be >= CVaR
    
    def test_performance_metrics_consistency(self):
        """Test consistency of performance metrics."""
        # Test that metrics are consistent
        sharpe = calculate_sharpe_ratio(self.sample_returns)
        max_dd = calculate_max_drawdown(self.sample_returns)
        
        # Basic sanity checks
        assert isinstance(sharpe, float)
        assert isinstance(max_dd, float)
        assert max_dd <= 0
        
        # Test with different risk-free rates
        sharpe_high_rf = calculate_sharpe_ratio(self.sample_returns, risk_free_rate=0.05)
        sharpe_low_rf = calculate_sharpe_ratio(self.sample_returns, risk_free_rate=0.01)
        
        # Higher risk-free rate should result in lower Sharpe ratio
        assert sharpe_high_rf <= sharpe_low_rf 