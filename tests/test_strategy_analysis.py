"""
Unit tests for strategy_analysis.py module
Critical strategy analysis testing
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

from strategy_analysis import StrategyAnalyzer


class TestStrategyAnalyzer(unittest.TestCase):
    """Test StrategyAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StrategyAnalyzer()
        
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
            'returns': returns,
            'target': np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
        }, index=dates)
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.choice([0, 1], size=1000)
        
        # Create feature columns
        self.feature_columns = [f'feature_{i}' for i in range(10)]
        for col in self.feature_columns:
            self.test_data[col] = np.random.normal(0, 1, 1000)
            
    def test_analyze_strategy(self):
        """Test core strategy analysis"""
        # Test basic analysis
        results = self.analyzer.analyze_strategy(
            self.test_data,
            self.mock_model,
            self.feature_columns
        )
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        required_keys = [
            'strategy_sharpe', 'buy_hold_sharpe', 'sharpe_improvement',
            'strategy_return', 'buy_hold_return', 'return_improvement',
            'strategy_volatility', 'buy_hold_volatility',
            'strategy_drawdown', 'buy_hold_drawdown',
            'statistical_significance', 'p_value'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], (int, float))
            
        # Verify Sharpe ratios are reasonable
        self.assertGreater(results['strategy_sharpe'], -10)
        self.assertLess(results['strategy_sharpe'], 10)
        self.assertGreater(results['buy_hold_sharpe'], -10)
        self.assertLess(results['buy_hold_sharpe'], 10)
        
        # Verify drawdowns are negative or zero
        self.assertLessEqual(results['strategy_drawdown'], 0)
        self.assertLessEqual(results['buy_hold_drawdown'], 0)
        
        # Verify p-value is between 0 and 1
        self.assertGreaterEqual(results['p_value'], 0)
        self.assertLessEqual(results['p_value'], 1)
        
    def test_analyze_strategy_with_custom_target(self):
        """Test strategy analysis with custom target column"""
        # Add custom target column
        self.test_data['custom_target'] = np.random.choice([0, 1], size=1000)
        
        results = self.analyzer.analyze_strategy(
            self.test_data,
            self.mock_model,
            self.feature_columns,
            target_column='custom_target'
        )
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_sharpe', results)
        self.assertIn('buy_hold_sharpe', results)
        
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Test with normal returns
        normal_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        sharpe = self.analyzer._calculate_sharpe_ratio(normal_returns)
        
        # Verify Sharpe ratio is reasonable
        self.assertIsInstance(sharpe, (int, float))
        self.assertGreater(sharpe, -10)
        self.assertLess(sharpe, 10)
        
        # Test with zero returns
        zero_returns = pd.Series([0] * 100)
        zero_sharpe = self.analyzer._calculate_sharpe_ratio(zero_returns)
        
        # Should handle zero returns gracefully
        self.assertIsInstance(zero_sharpe, (int, float))
        
        # Test with constant positive returns
        positive_returns = pd.Series([0.001] * 100)
        positive_sharpe = self.analyzer._calculate_sharpe_ratio(positive_returns)
        
        self.assertIsInstance(positive_sharpe, (int, float))
        
        # Test with constant negative returns
        negative_returns = pd.Series([-0.001] * 100)
        negative_sharpe = self.analyzer._calculate_sharpe_ratio(negative_returns)
        
        self.assertIsInstance(negative_sharpe, (int, float))
        
    def test_export_analysis_results(self):
        """Test analysis results export"""
        # Set up test results
        self.analyzer.results = {
            'strategy_sharpe': 1.2,
            'buy_hold_sharpe': 0.8,
            'sharpe_improvement': 0.4,
            'strategy_return': 0.15,
            'buy_hold_return': 0.10,
            'return_improvement': 0.05,
            'strategy_volatility': 0.20,
            'buy_hold_volatility': 0.25,
            'strategy_drawdown': -0.10,
            'buy_hold_drawdown': -0.15,
            'statistical_significance': True,
            'p_value': 0.001
        }
        
        # Test export
        filepath = self.analyzer.export_analysis_results()
        
        # Verify file was created
        self.assertIsInstance(filepath, str)
        self.assertTrue(os.path.exists(filepath))
        
        # Verify file content
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn('strategy_sharpe', content)
            self.assertIn('buy_hold_sharpe', content)
            
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        # Test with custom filepath
        custom_filepath = 'custom_results.json'
        custom_path = self.analyzer.export_analysis_results(custom_filepath)
        
        self.assertIsInstance(custom_path, str)
        self.assertTrue(os.path.exists(custom_path))
        
        # Clean up
        if os.path.exists(custom_path):
            os.remove(custom_path)
            
    def test_print_comprehensive_report(self):
        """Test comprehensive report printing"""
        # Set up test results
        self.analyzer.results = {
            'strategy_sharpe': 1.2,
            'buy_hold_sharpe': 0.8,
            'sharpe_improvement': 0.4,
            'strategy_return': 0.15,
            'buy_hold_return': 0.10,
            'return_improvement': 0.05,
            'strategy_volatility': 0.20,
            'buy_hold_volatility': 0.25,
            'strategy_drawdown': -0.10,
            'buy_hold_drawdown': -0.15,
            'statistical_significance': True,
            'p_value': 0.001
        }
        
        # Test printing (should not raise exception)
        try:
            self.analyzer.print_comprehensive_report()
        except Exception as e:
            self.fail(f"print_comprehensive_report raised {e} unexpectedly!")
            
    def test_strategy_returns_calculation(self):
        """Test strategy returns calculation logic"""
        # Create test data with known patterns
        test_data = pd.DataFrame({
            'returns': [0.01, -0.01, 0.02, -0.02, 0.01],
            'target': [1, 0, 1, 0, 1]
        })
        
        # Mock model predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0, 1, 0, 1]
        
        # Analyze strategy
        results = self.analyzer.analyze_strategy(
            test_data,
            mock_model,
            ['feature_1', 'feature_2']
        )
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_sharpe', results)
        self.assertIn('buy_hold_sharpe', results)
        
    def test_drawdown_calculation(self):
        """Test drawdown calculation logic"""
        # Create test data with known drawdown pattern
        test_data = pd.DataFrame({
            'returns': [0.01, -0.05, -0.03, 0.02, 0.01],  # Creates drawdown
            'target': [1, 1, 1, 1, 1]
        })
        
        # Mock model predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 1, 1, 1, 1]
        
        # Analyze strategy
        results = self.analyzer.analyze_strategy(
            test_data,
            mock_model,
            ['feature_1']
        )
        
        # Verify drawdown is calculated
        self.assertIn('strategy_drawdown', results)
        self.assertIn('buy_hold_drawdown', results)
        
        # Drawdown should be negative or zero
        self.assertLessEqual(results['strategy_drawdown'], 0)
        self.assertLessEqual(results['buy_hold_drawdown'], 0)
        
    def test_statistical_significance(self):
        """Test statistical significance calculation"""
        # Create test data with clear difference
        test_data = pd.DataFrame({
            'returns': [0.01] * 50 + [-0.01] * 50,  # Clear pattern
            'target': [1] * 100
        })
        
        # Mock model predictions that create clear difference
        mock_model = MagicMock()
        mock_model.predict.return_value = [1] * 50 + [0] * 50
        
        # Analyze strategy
        results = self.analyzer.analyze_strategy(
            test_data,
            mock_model,
            ['feature_1']
        )
        
        # Verify statistical significance
        self.assertIn('statistical_significance', results)
        self.assertIn('p_value', results)
        
        # p_value should be between 0 and 1
        self.assertGreaterEqual(results['p_value'], 0)
        self.assertLessEqual(results['p_value'], 1)
        
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        # Create test data with known volatility
        test_data = pd.DataFrame({
            'returns': [0.01, -0.01, 0.02, -0.02, 0.01],
            'target': [1, 1, 1, 1, 1]
        })
        
        # Mock model predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 1, 1, 1, 1]
        
        # Analyze strategy
        results = self.analyzer.analyze_strategy(
            test_data,
            mock_model,
            ['feature_1']
        )
        
        # Verify volatility calculations
        self.assertIn('strategy_volatility', results)
        self.assertIn('buy_hold_volatility', results)
        
        # Volatility should be positive
        self.assertGreater(results['strategy_volatility'], 0)
        self.assertGreater(results['buy_hold_volatility'], 0)


class TestStrategyAnalyzerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StrategyAnalyzer()
        
    def test_empty_data(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        mock_model = MagicMock()
        
        # Should handle gracefully
        with self.assertRaises((ValueError, IndexError)):
            self.analyzer.analyze_strategy(
                empty_data, mock_model, ['feature_1']
            )
            
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        data_without_returns = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'target': [1, 0, 1]
        })
        mock_model = MagicMock()
        
        # Should handle gracefully
        with self.assertRaises(KeyError):
            self.analyzer.analyze_strategy(
                data_without_returns, mock_model, ['feature_1']
            )
            
    def test_single_row_data(self):
        """Test handling of single row data"""
        single_row_data = pd.DataFrame({
            'returns': [0.01],
            'target': [1],
            'feature_1': [1]
        })
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        
        # Should handle gracefully
        results = self.analyzer.analyze_strategy(
            single_row_data, mock_model, ['feature_1']
        )
        
        # Should return results even with limited data
        self.assertIsInstance(results, dict)
        
    def test_all_zero_returns(self):
        """Test handling of all zero returns"""
        zero_returns_data = pd.DataFrame({
            'returns': [0] * 100,
            'target': [1] * 100,
            'feature_1': [1] * 100
        })
        mock_model = MagicMock()
        mock_model.predict.return_value = [1] * 100
        
        # Should handle gracefully
        results = self.analyzer.analyze_strategy(
            zero_returns_data, mock_model, ['feature_1']
        )
        
        # Should return results
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_sharpe', results)
        
    def test_all_positive_returns(self):
        """Test handling of all positive returns"""
        positive_returns_data = pd.DataFrame({
            'returns': [0.01] * 100,
            'target': [1] * 100,
            'feature_1': [1] * 100
        })
        mock_model = MagicMock()
        mock_model.predict.return_value = [1] * 100
        
        # Should handle gracefully
        results = self.analyzer.analyze_strategy(
            positive_returns_data, mock_model, ['feature_1']
        )
        
        # Should return results
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_sharpe', results)
        
    def test_all_negative_returns(self):
        """Test handling of all negative returns"""
        negative_returns_data = pd.DataFrame({
            'returns': [-0.01] * 100,
            'target': [1] * 100,
            'feature_1': [1] * 100
        })
        mock_model = MagicMock()
        mock_model.predict.return_value = [1] * 100
        
        # Should handle gracefully
        results = self.analyzer.analyze_strategy(
            negative_returns_data, mock_model, ['feature_1']
        )
        
        # Should return results
        self.assertIsInstance(results, dict)
        self.assertIn('strategy_sharpe', results)
        
    def test_missing_feature_columns(self):
        """Test handling of missing feature columns"""
        test_data = pd.DataFrame({
            'returns': [0.01, -0.01, 0.02],
            'target': [1, 0, 1]
        })
        mock_model = MagicMock()
        
        # Should handle gracefully
        with self.assertRaises(KeyError):
            self.analyzer.analyze_strategy(
                test_data, mock_model, ['missing_feature']
            )
            
    def test_model_prediction_errors(self):
        """Test handling of model prediction errors"""
        test_data = pd.DataFrame({
            'returns': [0.01, -0.01, 0.02],
            'target': [1, 0, 1],
            'feature_1': [1, 2, 3]
        })
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        
        # Should handle gracefully
        with self.assertRaises(Exception):
            self.analyzer.analyze_strategy(
                test_data, mock_model, ['feature_1']
            )


class TestStrategyAnalyzerIntegration(unittest.TestCase):
    """Test integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StrategyAnalyzer()
        
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        test_data = pd.DataFrame({
            'returns': np.random.normal(0.001, 0.02, 500),
            'target': np.random.choice([0, 1], size=500),
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(0, 1, 500),
            'feature_3': np.random.normal(0, 1, 500)
        }, index=dates)
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.choice([0, 1], size=500)
        
        # Run complete analysis
        results = self.analyzer.analyze_strategy(
            test_data,
            mock_model,
            ['feature_1', 'feature_2', 'feature_3']
        )
        
        # Verify complete results
        self.assertIsInstance(results, dict)
        required_keys = [
            'strategy_sharpe', 'buy_hold_sharpe', 'sharpe_improvement',
            'strategy_return', 'buy_hold_return', 'return_improvement',
            'strategy_volatility', 'buy_hold_volatility',
            'strategy_drawdown', 'buy_hold_drawdown',
            'statistical_significance', 'p_value'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
            
        # Export results
        filepath = self.analyzer.export_analysis_results()
        self.assertTrue(os.path.exists(filepath))
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        # Print report
        try:
            self.analyzer.print_comprehensive_report()
        except Exception as e:
            self.fail(f"print_comprehensive_report raised {e} unexpectedly!")


if __name__ == '__main__':
    unittest.main() 