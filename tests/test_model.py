"""
Test suite for XGBoost model functionality
Validates model loading, training, and prediction capabilities
"""

import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.modeling import XGBoostModel
from src.feature_engineering import FeatureEngineer

class TestXGBoostModel:
    """Test cases for XGBoost model functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for model testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 200)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 200)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sample_features(self, sample_data):
        """Generate sample features for model testing"""
        engineer = FeatureEngineer()
        features = engineer.create_all_features(sample_data.copy())
        prepared = engineer.prepare_features_for_modeling(features, normalize=True)
        return prepared
    
    @pytest.fixture
    def sample_targets(self, sample_data):
        """Generate sample targets for model testing"""
        # Create binary targets based on 5-day forward returns
        forward_returns = sample_data['close'].pct_change(5).shift(-5)
        targets = (forward_returns > 0).astype(int)
        return targets.dropna()
    
    @pytest.fixture
    def model(self):
        """Initialize XGBoost model"""
        return XGBoostModel()
    
    def test_model_creation(self, model):
        """Test model initialization and creation"""
        # Verify model can be created
        xgb_model = model.create_model()
        assert xgb_model is not None
        assert hasattr(xgb_model, 'fit')
        assert hasattr(xgb_model, 'predict')
        assert hasattr(xgb_model, 'predict_proba')
    
    def test_model_training(self, model, sample_features, sample_targets):
        """Test model training functionality"""
        # Align features and targets
        common_index = sample_features.index.intersection(sample_targets.index)
        X = sample_features.loc[common_index]
        y = sample_targets.loc[common_index]
        
        # Split for training
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Train model
        model.fit(X_train, y_train, X_test, y_test)
        
        # Verify model is trained
        assert model.is_trained
        assert model.model is not None
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Verify prediction outputs
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert probabilities.shape[1] == 2  # Binary classification
        
        # Verify probability constraints
        assert all(0 <= prob <= 1 for prob in probabilities.flatten())
        assert all(abs(prob.sum(axis=1) - 1) < 1e-10 for prob in probabilities)
    
    def test_feature_importance(self, model, sample_features, sample_targets):
        """Test feature importance calculation"""
        # Train model first
        common_index = sample_features.index.intersection(sample_targets.index)
        X = sample_features.loc[common_index]
        y = sample_targets.loc[common_index]
        
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        model.fit(X_train, y_train)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Verify importance structure
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0
        
        # Verify importance values are reasonable
        assert all(importance['importance'] >= 0)
        assert importance['importance'].sum() > 0
    
    def test_model_save_load(self, model, sample_features, sample_targets, tmp_path):
        """Test model persistence functionality"""
        # Train model
        common_index = sample_features.index.intersection(sample_targets.index)
        X = sample_features.loc[common_index]
        y = sample_targets.loc[common_index]
        
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        model.fit(X_train, y_train)
        
        # Save model
        save_path = tmp_path / "test_model.pkl"
        saved_path = model.save_model(str(save_path))
        
        # Verify file was created
        assert os.path.exists(saved_path)
        
        # Load model
        loaded_model = XGBoostModel()
        loaded_model.load_model(saved_path)
        
        # Verify loaded model works
        assert loaded_model.is_trained
        assert loaded_model.model is not None
        
        # Test predictions with loaded model
        X_test = X.iloc[split_idx:]
        predictions = loaded_model.predict(X_test)
        probabilities = loaded_model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
    
    def test_model_info(self, model, sample_features, sample_targets):
        """Test model information retrieval"""
        # Train model
        common_index = sample_features.index.intersection(sample_targets.index)
        X = sample_features.loc[common_index]
        y = sample_targets.loc[common_index]
        
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        model.fit(X_train, y_train)
        
        # Get model info
        info = model.get_model_info()
        
        # Verify info structure
        assert isinstance(info, dict)
        assert 'n_features' in info
        assert 'n_samples' in info
        assert 'feature_names' in info
        assert 'model_params' in info
        
        # Verify info values
        assert info['n_features'] > 0
        assert info['n_samples'] > 0
        assert len(info['feature_names']) == info['n_features']
    
    def test_prediction_consistency(self, model, sample_features, sample_targets):
        """Test prediction consistency across multiple calls"""
        # Train model
        common_index = sample_features.index.intersection(sample_targets.index)
        X = sample_features.loc[common_index]
        y = sample_targets.loc[common_index]
        
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        
        model.fit(X_train, y_train)
        
        # Make multiple predictions
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        prob1 = model.predict_proba(X_test)
        prob2 = model.predict_proba(X_test)
        
        # Verify consistency
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(prob1, prob2)
    
    def test_edge_cases(self, model):
        """Test model behavior with edge cases"""
        # Test with empty data
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=int)
        
        # Should handle gracefully or raise appropriate error
        try:
            model.fit(empty_df, empty_series)
            assert False, "Should not succeed with empty data"
        except (ValueError, IndexError):
            pass  # Expected behavior
        
        # Test with single sample
        single_sample = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        single_target = pd.Series([1])
        
        try:
            model.fit(single_sample, single_target)
            # Should work but may not be meaningful
        except Exception as e:
            # Some models require minimum samples
            assert "sample" in str(e).lower() or "minimum" in str(e).lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 