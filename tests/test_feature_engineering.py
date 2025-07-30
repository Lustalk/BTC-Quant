"""
Test suite for feature engineering module
Validates technical indicator calculations using sample data
"""

import pytest
import pandas as pd
import numpy as np
import os
from src.feature_engineering import FeatureEngineer

class TestFeatureEngineering:
    """Test cases for technical indicator calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Add returns column that feature engineering expects
        data['returns'] = data['close'].pct_change()
        
        return data
    
    @pytest.fixture
    def engineer(self):
        """Initialize feature engineer"""
        return FeatureEngineer()
    
    def test_momentum_indicators(self, engineer, sample_data):
        """Test momentum indicator calculations"""
        result = engineer.calculate_momentum_indicators(sample_data.copy())
        
        # Verify RSI calculation
        assert 'rsi_14' in result.columns
        assert not result['rsi_14'].isnull().all()
        assert all(0 <= rsi <= 100 for rsi in result['rsi_14'].dropna())
        
        # Verify Stochastic Oscillator
        assert 'stoch_k_14' in result.columns
        assert 'stoch_d_14' in result.columns
        assert not result['stoch_k_14'].isnull().all()
        assert not result['stoch_d_14'].isnull().all()
        
        # Verify Williams %R
        assert 'williams_r_14' in result.columns
        assert not result['williams_r_14'].isnull().all()
        
        # Verify Rate of Change
        assert 'roc_10' in result.columns
        assert not result['roc_10'].isnull().all()
    
    def test_trend_indicators(self, engineer, sample_data):
        """Test trend indicator calculations"""
        result = engineer.calculate_trend_indicators(sample_data.copy())
        
        # Verify SMA
        assert 'sma_20' in result.columns
        assert not result['sma_20'].isnull().all()
        assert all(result['sma_20'].dropna() > 0)
        
        # Verify EMA
        assert 'ema_12' in result.columns
        assert not result['ema_12'].isnull().all()
        assert all(result['ema_12'].dropna() > 0)
        
        # Verify MACD (check for any MACD-related columns)
        macd_columns = [col for col in result.columns if 'macd' in col.lower()]
        assert len(macd_columns) >= 2, "Expected at least 2 MACD columns"
        for col in macd_columns:
            assert not result[col].isnull().all()
    
    def test_volatility_indicators(self, engineer, sample_data):
        """Test volatility indicator calculations"""
        result = engineer.calculate_volatility_indicators(sample_data.copy())
        
        # Verify Bollinger Bands
        assert 'bb_upper_20_2' in result.columns
        assert 'bb_lower_20_2' in result.columns
        assert 'bb_middle_20_2' in result.columns
        assert not result['bb_upper_20_2'].isnull().all()
        
        # Verify ATR
        assert 'atr_14' in result.columns
        assert not result['atr_14'].isnull().all()
        assert all(result['atr_14'].dropna() >= 0)
    
    def test_volume_indicators(self, engineer, sample_data):
        """Test volume indicator calculations"""
        result = engineer.calculate_volume_indicators(sample_data.copy())
        
        # Verify OBV
        assert 'obv' in result.columns
        assert not result['obv'].isnull().all()
        
        # Verify Volume Rate of Change
        assert 'volume_roc_10' in result.columns
        assert not result['volume_roc_10'].isnull().all()
    
    def test_create_all_features(self, engineer, sample_data):
        """Test complete feature engineering pipeline"""
        result = engineer.create_all_features(sample_data.copy())
        
        # Verify core features are present (more flexible approach)
        core_features = [
            'rsi_14', 'stoch_k_14', 'stoch_d_14', 'williams_r_14', 'roc_10',
            'sma_20', 'ema_12', 'atr_14', 'obv'
        ]
        
        for feature in core_features:
            assert feature in result.columns, f"Missing core feature: {feature}"
            assert not result[feature].isnull().all(), f"All null values in {feature}"
        
        # Verify MACD features exist (any MACD column)
        macd_features = [col for col in result.columns if 'macd' in col.lower()]
        assert len(macd_features) >= 2, "Expected at least 2 MACD features"
        
        # Verify Bollinger Bands features exist
        bb_features = [col for col in result.columns if 'bb_' in col.lower()]
        assert len(bb_features) >= 2, "Expected at least 2 Bollinger Band features"
    
    def test_feature_normalization(self, engineer, sample_data):
        """Test feature normalization methods"""
        features = engineer.create_all_features(sample_data.copy())
        
        # Test expanding window normalization
        normalized = engineer.normalize_features(features, method='expanding')
        
        # Verify normalization properties
        for col in normalized.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip original price columns
            
            # Check that normalized features have reasonable ranges
            values = normalized[col].dropna()
            if len(values) > 0:
                assert values.std() > 0, f"Zero variance in {col}"
    
    def test_feature_preparation(self, engineer, sample_data):
        """Test feature preparation for modeling"""
        features = engineer.create_all_features(sample_data.copy())
        prepared = engineer.prepare_features_for_modeling(features, normalize=True)
        
        # Verify no missing values in prepared features
        assert not prepared.isnull().any().any(), "Missing values in prepared features"
        
        # Verify feature columns are available
        feature_cols = engineer.get_feature_importance_columns()
        assert len(feature_cols) > 0, "No feature importance columns available"
        
        # Verify all feature columns are in prepared data
        for col in feature_cols:
            assert col in prepared.columns, f"Missing feature column: {col}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 