"""
Feature Engineering Module
Calculates technical indicators for the technical indicator alpha project
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler

from config import FEATURE_CONFIG, FEATURE_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for calculating technical indicators
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        logger.info("Calculating momentum indicators...")
        
        # RSI
        data['rsi_14'] = ta.momentum.RSIIndicator(
            close=data['close'], 
            window=14
        ).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=14,
            smooth_window=3
        )
        data['stoch_k_14'] = stoch.stoch()
        data['stoch_d_14'] = stoch.stoch_signal()
        
        # Williams %R
        data['williams_r_14'] = ta.momentum.WilliamsRIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            lbp=14
        ).williams_r()
        
        # Rate of Change
        data['roc_10'] = ta.momentum.ROCIndicator(
            close=data['close'],
            window=10
        ).roc()
        
        return data
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        logger.info("Calculating trend indicators...")
        
        # Simple Moving Average
        data['sma_20'] = ta.trend.SMAIndicator(
            close=data['close'],
            window=FEATURE_CONFIG['trend_indicators']['sma']['window']
        ).sma_indicator()
        
        # Exponential Moving Average
        data['ema_12'] = ta.trend.EMAIndicator(
            close=data['close'],
            window=FEATURE_CONFIG['trend_indicators']['ema']['window']
        ).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(
            close=data['close'],
            window_fast=FEATURE_CONFIG['trend_indicators']['macd']['fast'],
            window_slow=FEATURE_CONFIG['trend_indicators']['macd']['slow'],
            window_sign=FEATURE_CONFIG['trend_indicators']['macd']['signal']
        )
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_hist'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=14
        )
        data['adx_14'] = adx.adx()
        data['di_plus'] = adx.adx_pos()
        data['di_minus'] = adx.adx_neg()
        
        # Parabolic SAR
        data['psar'] = ta.trend.PSARIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close']
        ).psar()
        
        return data
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators
        """
        logger.info("Calculating volatility indicators...")
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=data['close'],
            window=FEATURE_CONFIG['volatility_indicators']['bbands']['window'],
            window_dev=FEATURE_CONFIG['volatility_indicators']['bbands']['std']
        )
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_width'] = bb.bollinger_wband()
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Average True Range
        data['atr_14'] = ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=FEATURE_CONFIG['volatility_indicators']['atr']['window']
        ).average_true_range()
        
        # Realized Volatility (20-day rolling standard deviation of returns)
        data['realized_volatility_20'] = data['returns'].rolling(
            window=FEATURE_CONFIG['volatility_indicators']['realized_volatility']['window']
        ).std() * np.sqrt(252)  # Annualized
        
        return data
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        logger.info("Calculating volume indicators...")
        
        # On-Balance Volume
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=data['close'],
            volume=data['volume']
        ).on_balance_volume()
        
        # Volume Rate of Change (custom calculation)
        data['volume_roc_10'] = data['volume'].pct_change(periods=10)
        
        # Volume Weighted Average Price
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        ).volume_weighted_average_price()
        
        # VWAP Deviation (price relative to VWAP)
        data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
        
        return data
    
    def calculate_sentiment_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment-based indicators (Bitcoin Fear & Greed Index)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with sentiment indicators
        """
        logger.info("Calculating sentiment indicators...")
        
        # Bitcoin Fear & Greed Index (placeholder implementation)
        # In a real implementation, this would fetch data from an API
        # For now, we'll create a simple proxy based on volatility and momentum
        
        # Create a simple fear/greed proxy based on volatility and RSI
        if 'rsi_14' in data.columns and 'realized_volatility_20' in data.columns:
            # Normalize RSI to 0-100 scale
            rsi_normalized = data['rsi_14'] / 100.0
            
            # Normalize volatility (assuming max 100% annualized)
            vol_normalized = data['realized_volatility_20'] / 1.0
            vol_normalized = vol_normalized.clip(0, 1)  # Clip to 0-1 range
            
            # Fear & Greed Index: High volatility + low RSI = Fear
            # Low volatility + high RSI = Greed
            data['fear_greed_index'] = (1 - vol_normalized) * rsi_normalized
            
            # Scale to 0-100 range
            data['fear_greed_index'] = data['fear_greed_index'] * 100
        else:
            # Fallback: simple volatility-based proxy
            data['fear_greed_index'] = data['realized_volatility_20'] * 100
        
        return data
    
    def calculate_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price action features
        """
        logger.info("Calculating price action features...")
        
        # Price relative to 52-week high/low (already calculated in data pipeline)
        # These are already in the data from the data pipeline
        
        # Additional price action features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Price momentum
        data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
        data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
        data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
        
        # Volatility features (already calculated in data pipeline)
        # data['volatility_20d'] is already available
        
        return data
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators and features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all technical indicators...")
        
        # Calculate all indicator types
        data = self.calculate_momentum_indicators(data)
        data = self.calculate_trend_indicators(data)
        data = self.calculate_volatility_indicators(data)
        data = self.calculate_volume_indicators(data)
        data = self.calculate_sentiment_indicators(data)
        data = self.calculate_price_action_features(data)
        
        # Get list of feature columns (exclude OHLCV, returns, target, and non-technical indicators)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                       'log_returns', 'forward_returns', 'target', 'high_52w', 'low_52w',
                       'dividends', 'stock splits']
        
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        logger.info(f"Created {len(self.feature_columns)} technical indicators")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        return data
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'expanding') -> pd.DataFrame:
        """
        Normalize features to prevent lookahead bias
        
        Args:
            data: DataFrame with features
            method: Normalization method ('expanding' or 'standard')
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {method} method...")
        
        if method == 'expanding':
            # Use expanding window normalization to prevent lookahead bias
            for col in self.feature_columns:
                if col in data.columns:
                    # Calculate expanding mean and std
                    expanding_mean = data[col].expanding().mean()
                    expanding_std = data[col].expanding().std()
                    
                    # Normalize using expanding statistics
                    data[f'{col}_normalized'] = (data[col] - expanding_mean) / expanding_std
                    
                    # Replace original column with normalized version
                    data[col] = data[f'{col}_normalized']
                    data.drop(columns=[f'{col}_normalized'], inplace=True)
        
        elif method == 'standard':
            # Standard normalization (for training data only)
            scaler = StandardScaler()
            data[self.feature_columns] = scaler.fit_transform(data[self.feature_columns])
        
        return data
    
    def prepare_features_for_modeling(self, data: pd.DataFrame, 
                                   normalize: bool = True) -> pd.DataFrame:
        """
        Prepare features for modeling (remove NaN, normalize)
        
        Args:
            data: DataFrame with features
            normalize: Whether to normalize features
            
        Returns:
            DataFrame ready for modeling
        """
        # Remove rows with NaN values
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        
        logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        
        if normalize:
            data = self.normalize_features(data, method='expanding')
        
        return data
    
    def get_feature_importance_columns(self) -> List[str]:
        """
        Get list of feature columns for importance analysis
        
        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()

def main():
    """Test the feature engineering"""
    from data_pipeline import DataPipeline
    
    # Load data
    pipeline = DataPipeline()
    data = pipeline.preprocess_data()
    
    # Create features
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    
    # Prepare for modeling
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
    
    print(f"Original data shape: {data.shape}")
    print(f"Data with features shape: {data_with_features.shape}")
    print(f"Final modeling data shape: {data_ready.shape}")
    print(f"Number of features: {len(feature_engineer.feature_columns)}")
    print(f"Feature columns: {feature_engineer.feature_columns[:5]}...")

if __name__ == "__main__":
    main() 