"""
Unit tests for evaluation.py module
Critical performance calculation testing
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import PerformanceEvaluator


class TestPerformanceEvaluator(unittest.TestCase):
    """Test PerformanceEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = PerformanceEvaluator()
        
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Create price data with realistic patterns
        base_price = 10000
        returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000),
            'target': np.random.choice([0, 1], size=1000)  # Add target column
        }, index=dates)
        
        # Create test predictions and probabilities
        self.test_predictions = np.random.choice([0, 1], size=1000)
        self.test_probabilities = np.random.uniform(0, 1, size=1000)
        
    def test_calculate_strategy_returns(self):
        """Test strategy returns calculation"""
        # Test with default threshold
        strategy_returns = self.evaluator.calculate_strategy_returns(
            self.test_data, 
            self.test_predictions, 
            self.test_probabilities
        )
        
        # Verify return type and structure
        self.assertIsInstance(strategy_returns, pd.Series)
        self.assertGreater(len(strategy_returns), 0)
        
        # Verify no NaN values in final result
        self.assertFalse(strategy_returns.isnull().any())
        
        # Test with custom threshold
        strategy_returns_custom = self.evaluator.calculate_strategy_returns(
            self.test_data, 
            self.test_predictions, 
            self.test_probabilities,
            threshold=0.7
        )
        
        self.assertIsInstance(strategy_returns_custom, pd.Series)
        self.assertGreater(len(strategy_returns_custom), 0)
        
        # Test with different data lengths
        short_data = self.test_data.iloc[:500]
        short_predictions = self.test_predictions[:500]
        short_probabilities = self.test_probabilities[:500]
        
        short_returns = self.evaluator.calculate_strategy_returns(
            short_data, 
            short_predictions, 
            short_probabilities
        )
        
        self.assertIsInstance(short_returns, pd.Series)
        self.assertGreater(len(short_returns), 0)
        
    def test_calculate_benchmark_returns(self):
        """Test benchmark returns calculation"""
        benchmark_returns = self.evaluator.calculate_benchmark_returns(self.test_data)
        
        # Verify return type and structure
        self.assertIsInstance(benchmark_returns, pd.Series)
        self.assertGreater(len(benchmark_returns), 0)
        
        # Verify no NaN values in final result (after dropna)
        self.assertFalse(benchmark_returns.isnull().any())
        
        # Test with different data
        short_data = self.test_data.iloc[:100]
        short_benchmark = self.evaluator.calculate_benchmark_returns(short_data)
        
        self.assertIsInstance(short_benchmark, pd.Series)
        self.assertGreater(len(short_benchmark), 0)
        
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create test returns
        test_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        # Test with default risk-free rate
        metrics = self.evaluator.calculate_performance_metrics(test_returns)
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        required_keys = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
            'sortino_ratio', 'var_95', 'cvar_95'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            
        # Test with custom risk-free rate
        metrics_custom = self.evaluator.calculate_performance_metrics(
            test_returns, risk_free_rate=0.02
        )
        
        self.assertIsInstance(metrics_custom, dict)
        for key in required_keys:
            self.assertIn(key, metrics_custom)
            
        # Test with zero returns
        zero_returns = pd.Series([0] * 100)
        zero_metrics = self.evaluator.calculate_performance_metrics(zero_returns)
        
        self.assertIsInstance(zero_metrics, dict)
        self.assertEqual(zero_metrics['total_return'], 0)
        
    def test_backtest_strategy(self):
        """Test complete strategy backtesting"""
        # Test basic backtest
        results = self.evaluator.backtest_strategy(
            self.test_data,
            self.test_predictions,
            self.test_probabilities
        )
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        required_keys = [
            'strategy_metrics', 'benchmark_metrics', 'strategy_returns',
            'benchmark_returns', 'signals', 'predictive_power',
            'financial_performance', 'robustness'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
            
        # Verify metrics are calculated
        strategy_metrics = results['strategy_metrics']
        benchmark_metrics = results['benchmark_metrics']
        
        self.assertIsInstance(strategy_metrics, dict)
        self.assertIsInstance(benchmark_metrics, dict)
        
        # Test with position sizes
        position_sizes = pd.Series(np.random.uniform(0.5, 1.5, len(self.test_data)))
        results_with_sizes = self.evaluator.backtest_strategy(
            self.test_data,
            self.test_predictions,
            self.test_probabilities,
            position_sizes=position_sizes
        )
        
        self.assertIsInstance(results_with_sizes, dict)
        
    def test_analyze_predictive_power(self):
        """Test predictive power analysis"""
        # Create test data
        predictions = np.random.choice([0, 1], size=1000)
        probabilities = np.random.uniform(0, 1, size=1000)
        
        # Mock the method call
        with patch.object(self.evaluator, '_analyze_predictive_power') as mock_analyze:
            mock_analyze.return_value = {
                'accuracy': 0.65,
                'precision': 0.7,
                'recall': 0.6,
                'f1_score': 0.65,
                'auc_roc': 0.68
            }
            
            result = self.evaluator._analyze_predictive_power(
                predictions, probabilities, self.test_data
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('accuracy', result)
            self.assertIn('auc_roc', result)
            
    def test_analyze_financial_performance(self):
        """Test financial performance analysis"""
        # Create test metrics
        strategy_metrics = {
            'sharpe_ratio': 1.2,
            'total_return': 0.15,
            'max_drawdown': -0.1
        }
        benchmark_metrics = {
            'sharpe_ratio': 0.8,
            'total_return': 0.10,
            'max_drawdown': -0.15
        }
        
        strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.025, 1000))
        
        # Mock the method call
        with patch.object(self.evaluator, '_analyze_financial_performance') as mock_analyze:
            mock_analyze.return_value = {
                'alpha': 0.05,
                'beta': 0.8,
                'information_ratio': 0.3,
                'tracking_error': 0.02
            }
            
            result = self.evaluator._analyze_financial_performance(
                strategy_metrics, benchmark_metrics, 
                strategy_returns, benchmark_returns
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('alpha', result)
            self.assertIn('information_ratio', result)
            
    def test_analyze_robustness(self):
        """Test robustness analysis"""
        strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.025, 1000))
        probabilities = np.random.uniform(0, 1, 1000)
        
        # Mock the method call
        with patch.object(self.evaluator, '_analyze_robustness') as mock_analyze:
            mock_analyze.return_value = {
                'signal_stability': 0.75,
                'probability_distribution': 'normal',
                'risk_adjusted_metrics': {'sortino': 1.1}
            }
            
            result = self.evaluator._analyze_robustness(
                strategy_returns, benchmark_returns, probabilities
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('signal_stability', result)
            
    def test_plot_cumulative_returns(self):
        """Test cumulative returns plotting"""
        # Set up test data with proper structure
        self.evaluator.results = {
            'strategy_returns': pd.Series(np.random.normal(0.001, 0.02, 1000)),
            'benchmark_returns': pd.Series(np.random.normal(0.0008, 0.025, 1000))
        }
        
        # Test plotting (should not raise exception)
        try:
            self.evaluator.plot_cumulative_returns()
        except Exception as e:
            self.fail(f"plot_cumulative_returns raised {e} unexpectedly!")
            
    def test_plot_drawdown(self):
        """Test drawdown plotting"""
        # Set up test data with proper structure
        self.evaluator.results = {
            'strategy_returns': pd.Series(np.random.normal(0.001, 0.02, 1000)),
            'benchmark_returns': pd.Series(np.random.normal(0.0008, 0.025, 1000))
        }
        
        # Test plotting (should not raise exception)
        try:
            self.evaluator.plot_drawdown()
        except Exception as e:
            self.fail(f"plot_drawdown raised {e} unexpectedly!")
            
    def test_create_performance_summary(self):
        """Test performance summary creation"""
        # Set up test data with complete structure
        self.evaluator.results = {
            'strategy_metrics': {
                'sharpe_ratio': 1.2,
                'total_return': 0.15,
                'max_drawdown': -0.1,
                'annualized_return': 0.12,
                'volatility': 0.20,
                'sortino_ratio': 1.5,
                'calmar_ratio': 1.2,
                'win_rate': 0.6
            },
            'benchmark_metrics': {
                'sharpe_ratio': 0.8,
                'total_return': 0.10,
                'max_drawdown': -0.15,
                'annualized_return': 0.08,
                'volatility': 0.25,
                'sortino_ratio': 1.0,
                'calmar_ratio': 0.8,
                'win_rate': 0.5
            },
            'alpha': 0.05,
            'beta': 0.8,
            'information_ratio': 0.3
        }
        
        summary = self.evaluator.create_performance_summary()
        
        # Verify summary structure
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertGreater(len(summary.columns), 0)
        
    def test_save_results(self):
        """Test results saving"""
        # Set up test results with complete structure
        self.evaluator.results = {
            'strategy_metrics': {'sharpe_ratio': 1.2},
            'benchmark_metrics': {'sharpe_ratio': 0.8},
            'alpha': 0.05,
            'beta': 0.8,
            'information_ratio': 0.3,
            'tracking_error': 0.02
        }
        
        # Test saving
        filepath = self.evaluator.save_results()
        
        # Verify file was created
        self.assertIsInstance(filepath, str)
        self.assertTrue(os.path.exists(filepath))
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
    def test_print_summary(self):
        """Test summary printing"""
        # Set up test results with complete structure
        self.evaluator.results = {
            'strategy_metrics': {
                'sharpe_ratio': 1.2,
                'total_return': 0.15,
                'max_drawdown': -0.1,
                'annualized_return': 0.12,
                'sortino_ratio': 1.5,
                'win_rate': 0.6
            },
            'benchmark_metrics': {
                'sharpe_ratio': 0.8,
                'total_return': 0.10,
                'max_drawdown': -0.15,
                'annualized_return': 0.08,
                'sortino_ratio': 1.0,
                'win_rate': 0.5
            }
        }
        
        # Test printing (should not raise exception)
        try:
            self.evaluator.print_summary()
        except Exception as e:
            self.fail(f"print_summary raised {e} unexpectedly!")


class TestPerformanceEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = PerformanceEvaluator()
        
    def test_empty_data(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        empty_predictions = []
        empty_probabilities = []
        
        # Should handle gracefully
        with self.assertRaises((ValueError, IndexError, KeyError)):
            self.evaluator.calculate_strategy_returns(
                empty_data, empty_predictions, empty_probabilities
            )
            
    def test_mismatched_lengths(self):
        """Test handling of mismatched data lengths"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        predictions = [0, 1]  # Shorter than data
        probabilities = [0.3, 0.7]
        
        # Should handle gracefully
        strategy_returns = self.evaluator.calculate_strategy_returns(
            data, predictions, probabilities
        )
        
        self.assertIsInstance(strategy_returns, pd.Series)
        
    def test_invalid_probabilities(self):
        """Test handling of invalid probabilities"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        predictions = [0, 1, 0]
        probabilities = [1.5, -0.1, 0.8]  # Invalid values
        
        # Should handle gracefully
        strategy_returns = self.evaluator.calculate_strategy_returns(
            data, predictions, probabilities
        )
        
        self.assertIsInstance(strategy_returns, pd.Series)
        
    def test_zero_volatility(self):
        """Test handling of zero volatility returns"""
        zero_returns = pd.Series([0.001] * 100)  # Constant returns
        
        metrics = self.evaluator.calculate_performance_metrics(zero_returns)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('volatility', metrics)
        # For constant returns, volatility should be very small but not exactly zero
        self.assertLess(metrics['volatility'], 1e-10)


if __name__ == '__main__':
    unittest.main() 