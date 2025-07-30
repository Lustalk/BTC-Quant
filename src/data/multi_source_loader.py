"""
Multi-Source Data Loader
Professional data pipeline with multiple sources and caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import redis
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Professional data loader with multi-source support and caching.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize data loader.
        
        Args:
            redis_client: Redis client for caching (optional)
        """
        self.redis_client = redis_client
        self.cache_enabled = redis_client is not None
        
        if self.cache_enabled:
            logger.info("Data loader initialized with Redis caching")
        else:
            logger.info("Data loader initialized without caching")
    
    def get_bitcoin_data(
        self, 
        start_date: str, 
        end_date: str,
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load Bitcoin data with caching support.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1h', '1m')
            use_cache: Whether to use caching
        
        Returns:
            DataFrame with Bitcoin price data
        """
        try:
            cache_key = f"btc_data_{start_date}_{end_date}_{interval}"
            
            # Check cache first
            if use_cache and self.cache_enabled:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"Retrieved Bitcoin data from cache: {start_date} to {end_date}")
                    return cached_data
            
            # Fetch from Yahoo Finance
            logger.info(f"Fetching Bitcoin data from Yahoo Finance: {start_date} to {end_date}")
            btc = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
            
            if btc.empty:
                raise ValueError(f"No data found for period {start_date} to {end_date}")
            
            # Clean and validate data
            btc = self._clean_data(btc)
            
            # Cache the data
            if use_cache and self.cache_enabled:
                self._save_to_cache(cache_key, btc)
            
            logger.info(f"Successfully loaded {len(btc)} Bitcoin data points")
            return btc
            
        except Exception as e:
            logger.error(f"Failed to load Bitcoin data: {str(e)}")
            raise
    
    def get_multiple_assets(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple assets.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            Dictionary with asset data
        """
        try:
            results = {}
            
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}")
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                    if not data.empty:
                        data = self._clean_data(data)
                        results[symbol] = data
                    else:
                        logger.warning(f"No data found for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load data for {symbol}: {str(e)}")
            
            logger.info(f"Successfully loaded data for {len(results)} assets")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load multiple assets: {str(e)}")
            raise
    
    def get_market_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load comprehensive market data including Bitcoin and related assets.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary with market data
        """
        try:
            symbols = [
                'BTC-USD',  # Bitcoin
                'ETH-USD',  # Ethereum
                'SPY',      # S&P 500 ETF
                'GLD',      # Gold ETF
                'TLT',      # Treasury bonds
                'DXY'       # US Dollar Index
            ]
            
            market_data = self.get_multiple_assets(symbols, start_date, end_date)
            
            logger.info(f"Loaded market data for {len(market_data)} assets")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to load market data: {str(e)}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            data: Raw data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        try:
            # Remove rows with missing values
            data = data.dropna()
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Remove first row (NaN from pct_change)
            data = data.dropna()
            
            logger.info(f"Data cleaned: {len(data)} rows, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached DataFrame or None
        """
        try:
            if not self.cache_enabled:
                return None
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data_dict = json.loads(cached_data)
                return pd.DataFrame(data_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, ttl: int = 3600):
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key
            data: DataFrame to cache
            ttl: Time to live in seconds
        """
        try:
            if not self.cache_enabled:
                return
            
            data_dict = data.to_dict('records')
            self.redis_client.setex(cache_key, ttl, json.dumps(data_dict))
            logger.info(f"Data cached with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache save failed: {str(e)}")

def get_bitcoin_data(
    start_date: str, 
    end_date: str,
    interval: str = '1d',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to get Bitcoin data.
    
    Args:
        start_date: Start date
        end_date: End date
        interval: Data interval
        use_cache: Whether to use caching
    
    Returns:
        Bitcoin price data
    """
    try:
        # Initialize Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url)
        
        # Test Redis connection
        try:
            redis_client.ping()
            cache_enabled = True
        except:
            cache_enabled = False
            redis_client = None
        
        loader = DataLoader(redis_client if cache_enabled else None)
        return loader.get_bitcoin_data(start_date, end_date, interval, use_cache)
        
    except Exception as e:
        logger.error(f"Failed to get Bitcoin data: {str(e)}")
        raise

def get_sample_data() -> pd.DataFrame:
    """
    Get sample Bitcoin data for testing and development.
    
    Returns:
        Sample Bitcoin data
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        return get_bitcoin_data(start_date, end_date)
        
    except Exception as e:
        logger.error(f"Failed to get sample data: {str(e)}")
        raise 