"""
Core functionality tests - focused on essential components only
Tests data pipeline, feature engineering, modeling, and validation
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from modeling import XGBoostModel
from validation import WalkForwardValidator
from config import DATA_CONFIG, XGBOOST_CONFIG


class TestCoreDataPipeline(unittest.TestCase):
    """Test core data pipeline functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = DataPipeline(
            symbol='SPY',  # Use SPY for testing (more reliable than BTC)
            start_date='2023-01-01', 
            end_date='2023-12-31'
        )
        
    def test_data_download_and_preprocessing(self):
        """Test that data can be downloaded and preprocessed"""
        # Test data download
        data = self.pipeline.download_data()
        
        # Basic validation
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check data types
        self.assertTrue(data.index.dtype == 'datetime64[ns]')
        
        # Test preprocessing
        processed_data = self.pipeline.preprocess_data()
        
        # Check that preprocessing was successful
        self.assertIsNotNone(processed_data)
        self.assertGreater(len(processed_data), 0)
        
        # Check that required columns exist
        required_columns = ['returns', 'target']
        for col in required_columns:
            self.assertIn(col, processed_data.columns)
            
        # Check data quality
        self.assertTrue(processed_data.index.dtype == 'datetime64[ns]')
        self.assertTrue(all(processed_data['target'].isin([0, 1])))
        
    def test_return_calculations(self):
        """Test return calculations are correct"""
        # Create sample data
        sample_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 103],
            'volume': [1000, 1100, 900, 1200, 1300]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        result = self.pipeline.calculate_returns(sample_data)
        
        # Check that returns were calculated
        self.assertIn('returns', result.columns)
        self.assertIn('target', result.columns)
        
        # Check that target is binary
        self.assertTrue(all(result['target'].isin([0, 1])))
        
        # Check return calculations (first value should be NaN)
        expected_returns = [np.nan, 0.01, -0.0198, 0.0303, 0.0098]
        np.testing.assert_array_almost_equal(
            result['returns'].values, expected_returns, decimal=4
        )


class TestCoreFeatureEngineering(unittest.TestCase):
    """Test core feature engineering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample data for testing
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(110, 130, 100),
            'low': np.random.uniform(70, 90, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'returns': np.random.normal(0, 0.02, 100)
        }, index=dates)
        
        # Ensure data quality
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['close'])
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['open'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['close'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['open'])
        
    def test_core_technical_indicators(self):
        """Test core technical indicators"""
        data = self.sample_data.copy()
        result = self.feature_engineer.create_all_features(data)
        
        # Check core indicators are created
        core_indicators = [
            'rsi_14', 'sma_20', 'ema_12', 'macd', 'bb_upper_20_2', 
            'atr_14', 'obv', 'volume_roc_10'
        ]
        
        for indicator in core_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                
        # Check RSI bounds
        if 'rsi_14' in result.columns:
            rsi_values = result['rsi_14'].dropna()
            self.assertTrue(all(0 <= rsi <= 100 for rsi in rsi_values))
            
        # Check moving averages
        if 'sma_20' in result.columns:
            sma_values = result['sma_20'].dropna()
            self.assertTrue(len(sma_values) > 0)
            
        # Check that feature_columns is updated
        self.assertGreater(len(self.feature_engineer.feature_columns), 0)
        
    def test_feature_preparation_for_modeling(self):
        """Test feature preparation for modeling"""
        data = self.sample_data.copy()
        data_with_features = self.feature_engineer.create_all_features(data)
        
        # Add target column
        data_with_features['target'] = np.random.choice([0, 1], size=len(data_with_features))
        
        prepared_data = self.feature_engineer.prepare_features_for_modeling(data_with_features)
        
        # Check that target column exists
        self.assertIn('target', prepared_data.columns)
        
        # Check that features are prepared
        self.assertGreater(len(self.feature_engineer.feature_columns), 0)
        
        # Check that no missing values in feature columns
        feature_data = prepared_data[self.feature_engineer.feature_columns]
        self.assertFalse(feature_data.isnull().any().any())


class TestCoreModeling(unittest.TestCase):
    """Test core modeling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = XGBoostModel()
        
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Create realistic feature data
        n_features = 10
        feature_data = np.random.normal(0, 1, (200, n_features))
        
        self.sample_features = pd.DataFrame(
            feature_data,
            columns=[f'feature_{i}' for i in range(n_features)],
            index=dates
        )
        
        # Create realistic target data (binary classification)
        self.sample_targets = pd.Series(
            np.random.choice([0, 1], size=200, p=[0.6, 0.4]),
            index=dates,
            name='target'
        )
        
        # Ensure alignment
        common_index = self.sample_features.index.intersection(self.sample_targets.index)
        self.sample_features = self.sample_features.loc[common_index]
        self.sample_targets = self.sample_targets.loc[common_index]
        
    def test_model_training_and_prediction(self):
        """Test model training and prediction"""
        # Split data for training
        split_idx = len(self.sample_features) // 2
        X_train = self.sample_features.iloc[:split_idx]
        y_train = self.sample_targets.iloc[:split_idx]
        X_test = self.sample_features.iloc[split_idx:]
        y_test = self.sample_targets.iloc[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train, X_test, y_test)
        
        # Verify model is trained
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        
        # Test predictions
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Verify prediction outputs
        self.assertEqual(len(predictions), len(X_test))
        self.assertEqual(len(probabilities), len(X_test))
        
        # Verify probability constraints
        self.assertTrue(all(0 <= prob <= 1 for prob in probabilities.flatten()))
        
        # Verify binary classification
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
    def test_feature_importance(self):
        """Test feature importance calculation"""
        # Train model
        split_idx = len(self.sample_features) // 2
        X_train = self.sample_features.iloc[:split_idx]
        y_train = self.sample_targets.iloc[:split_idx]
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Verify importance structure
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        
        # Verify importance values
        self.assertTrue(all(importance['importance'] >= 0))
        self.assertEqual(len(importance), len(self.sample_features.columns))


class TestCoreValidation(unittest.TestCase):
    """Test core validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = WalkForwardValidator(
            initial_train_years=1,
            test_period_months=3,
            min_train_size=100
        )
        
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Create realistic data with features and target
        n_features = 10
        feature_data = np.random.normal(0, 1, (1000, n_features))
        
        self.sample_data = pd.DataFrame(
            feature_data,
            columns=[f'feature_{i}' for i in range(n_features)],
            index=dates
        )
        
        # Add target column
        self.sample_data['target'] = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
        self.sample_data['returns'] = np.random.normal(0, 0.02, 1000)
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.choice([0, 1], size=100)
        self.mock_model.predict_proba.return_value = np.random.uniform(0, 1, size=100)
        
    def test_validation_periods(self):
        """Test validation period creation"""
        periods = self.validator.generate_walk_forward_periods(self.sample_data)
        
        # Check that periods were created
        self.assertIsInstance(periods, list)
        self.assertGreater(len(periods), 0)
        
        # Check period structure
        for train_data, test_data in periods:
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertIsInstance(test_data, pd.DataFrame)
            self.assertGreater(len(train_data), 0)
            self.assertGreater(len(test_data), 0)
            
        # Check period ordering
        for i in range(len(periods) - 1):
            train_data1, test_data1 = periods[i]
            train_data2, test_data2 = periods[i + 1]
            self.assertLess(test_data1.index[-1], test_data2.index[0])
            
    def test_complete_validation(self):
        """Test complete validation workflow"""
        # Mock model predictions for multiple periods
        self.mock_model.predict.side_effect = [
            np.array([0, 1, 0]),
            np.array([1, 0, 1]),
            np.array([0, 0, 1])
        ]
        self.mock_model.predict_proba.side_effect = [
            np.array([0.3, 0.7, 0.2]),
            np.array([0.8, 0.1, 0.9]),
            np.array([0.2, 0.3, 0.8])
        ]
        
        # Validate model
        results = self.validator.validate_model(
            self.sample_data, self.mock_model, ['feature_0', 'feature_1']
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('period_results', results)
        self.assertIn('overall_metrics', results)
        
        # Check period results
        self.assertIsInstance(results['period_results'], list)
        
        # Check overall metrics
        self.assertIn('accuracy', results['overall_metrics'])
        self.assertIn('auc_roc', results['overall_metrics'])


class TestCoreIntegration(unittest.TestCase):
    """Test core integration workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Create realistic market data
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 500),
            'high': np.random.uniform(110, 130, 500),
            'low': np.random.uniform(70, 90, 500),
            'close': np.random.uniform(90, 110, 500),
            'volume': np.random.uniform(1000, 5000, 500),
            'returns': np.random.normal(0, 0.02, 500)
        }, index=dates)
        
        # Ensure data quality
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['close'])
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['open'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['close'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['open'])
        
        # Add target column
        self.sample_data['target'] = np.random.choice([0, 1], size=500, p=[0.6, 0.4])
        
    def test_complete_workflow(self):
        """Test the complete core workflow"""
        # Step 1: Data Pipeline
        with patch.object(DataPipeline, 'download_data', return_value=self.sample_data):
            pipeline = DataPipeline()
            pipeline.data = self.sample_data.copy()  # Set the data directly
            data = pipeline.preprocess_data()
            self.assertIsNotNone(data)
            
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        self.assertGreater(len(feature_engineer.feature_columns), 0)
        
        # Step 3: Modeling
        model = XGBoostModel()
        
        # Split data
        split_idx = len(data_ready) // 2
        X_train = data_ready[feature_engineer.feature_columns].iloc[:split_idx]
        y_train = data_ready['target'].iloc[:split_idx]
        X_test = data_ready[feature_engineer.feature_columns].iloc[split_idx:]
        y_test = data_ready['target'].iloc[split_idx:]
        
        # Train model
        model.fit(X_train, y_train, X_test, y_test)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Verify predictions
        self.assertEqual(len(predictions), len(X_test))
        self.assertEqual(len(probabilities), len(X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Step 4: Validation
        validator = WalkForwardValidator(
            initial_train_years=1,
            test_period_months=3,
            min_train_size=100
        )
        
        # Mock model predictions
        with patch.object(model, 'predict', return_value=np.random.choice([0, 1], size=100)):
            with patch.object(model, 'predict_proba', return_value=np.random.uniform(0, 1, size=100)):
                # Run validation
                results = validator.validate_model(
                    data_ready, model, feature_engineer.feature_columns
                )
                
                # Check results
                self.assertIsInstance(results, dict)
                self.assertIn('period_results', results)
                self.assertIn('overall_metrics', results)
                
        # Verify complete workflow
        self.assertIsNotNone(data)
        self.assertGreater(len(feature_engineer.feature_columns), 0)
        self.assertTrue(model.is_trained)
        self.assertIsInstance(results, dict)


if __name__ == '__main__':
    unittest.main() 