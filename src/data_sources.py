"""
Multi-Data Source System for BTC Quant Trading System

This module implements a comprehensive data source system with:
- Abstract data source interface
- Factory pattern for data source creation
- Multiple data source implementations
- Data fusion and quality comparison
- Caching and performance metrics
- Error handling and fallback mechanisms
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import asyncio
import aiohttp
from functools import lru_cache
import hashlib
import json


@dataclass
class DataSourceMetrics:
    """Metrics for data source performance and quality."""
    source_name: str
    response_time: float
    data_quality_score: float
    success_rate: float
    last_update: datetime
    total_requests: int
    failed_requests: int
    cache_hit_rate: float


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


class AbstractDataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.metrics = DataSourceMetrics(
            source_name=name,
            response_time=0.0,
            data_quality_score=0.0,
            success_rate=1.0,
            last_update=datetime.now(),
            total_requests=0,
            failed_requests=0,
            cache_hit_rate=0.0
        )
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes default
    
    @abstractmethod
    def download_data(self, ticker: str, start_date: str, end_date: str, 
                     interval: str = "1d") -> pd.DataFrame:
        """Download data from the source."""
        pass
    
    @abstractmethod
    def get_supported_intervals(self) -> List[str]:
        """Get list of supported intervals."""
        pass
    
    @abstractmethod
    def get_max_history_days(self) -> int:
        """Get maximum history available in days."""
        pass
    
    def update_metrics(self, response_time: float, success: bool, 
                      quality_score: float = 1.0):
        """Update performance metrics."""
        self.metrics.response_time = response_time
        self.metrics.last_update = datetime.now()
        self.metrics.total_requests += 1
        
        if not success:
            self.metrics.failed_requests += 1
        
        self.metrics.success_rate = (
            (self.metrics.total_requests - self.metrics.failed_requests) / 
            self.metrics.total_requests
        )
        self.metrics.data_quality_score = quality_score
    
    def get_cache_key(self, ticker: str, start_date: str, end_date: str, 
                      interval: str) -> str:
        """Generate cache key for data request."""
        key_data = f"{ticker}_{start_date}_{end_date}_{interval}_{self.name}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available and not expired."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                self.metrics.cache_hit_rate = (
                    (self.metrics.cache_hit_rate * (self.metrics.total_requests - 1) + 1) /
                    self.metrics.total_requests
                )
                return cached_data
        return None
    
    def cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp."""
        self.cache[cache_key] = (data, datetime.now())


class YahooFinanceDataSource(AbstractDataSource):
    """Yahoo Finance data source implementation."""
    
    def __init__(self):
        super().__init__("YahooFinance")
        self.interval_mapping = {
            "1h": "1h",
            "4h": "1h",  # Will resample
            "1d": "1d",
            "1w": "1wk"
        }
    
    def get_supported_intervals(self) -> List[str]:
        return ["1h", "4h", "1d", "1w"]
    
    def get_max_history_days(self) -> int:
        return 730  # 2 years for intraday, much more for daily
    
    def download_data(self, ticker: str, start_date: str, end_date: str, 
                     interval: str = "1d") -> pd.DataFrame:
        """Download data from Yahoo Finance."""
        start_time = time.time()
        cache_key = self.get_cache_key(ticker, start_date, end_date, interval)
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            ticker_obj = yf.Ticker(ticker)
            yf_interval = self.interval_mapping.get(interval, interval)
            
            data = ticker_obj.history(start=start_date, end=end_date, interval=yf_interval)
            
            if data.empty:
                raise DataSourceError(f"No data found for {ticker}")
            
            # Resample to 4h if needed
            if interval == "4h":
                data = self._resample_data(data, "4H")
            
            # Cache the result
            self.cache_data(cache_key, data)
            
            response_time = time.time() - start_time
            self.update_metrics(response_time, True, 1.0)
            
            return data
            
        except Exception as e:
            response_time = time.time() - start_time
            self.update_metrics(response_time, False, 0.0)
            raise DataSourceError(f"Yahoo Finance error: {str(e)}")
    
    def _resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample data to target interval."""
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        resampled = data.resample(target_interval).agg(agg_rules)
        return resampled.dropna()


class AlphaVantageDataSource(AbstractDataSource):
    """Alpha Vantage data source implementation."""
    
    def __init__(self, api_key: str):
        super().__init__("AlphaVantage", api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.interval_mapping = {
            "1h": "60min",
            "4h": "60min",  # Will resample
            "1d": "daily",
            "1w": "weekly"
        }
    
    def get_supported_intervals(self) -> List[str]:
        return ["1d", "1w"]  # Alpha Vantage has limited intraday for free tier
    
    def get_max_history_days(self) -> int:
        return 365 * 20  # 20 years of historical data
    
    def download_data(self, ticker: str, start_date: str, end_date: str, 
                     interval: str = "1d") -> pd.DataFrame:
        """Download data from Alpha Vantage."""
        start_time = time.time()
        cache_key = self.get_cache_key(ticker, start_date, end_date, interval)
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            if not self.api_key:
                raise DataSourceError("Alpha Vantage API key required")
            
            # Alpha Vantage uses different ticker format
            av_ticker = self._convert_to_alpha_vantage_ticker(ticker)
            
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': av_ticker,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise DataSourceError(f"Alpha Vantage error: {data['Error Message']}")
            
            if 'Note' in data:
                # Rate limit hit
                raise DataSourceError(f"Alpha Vantage rate limit: {data['Note']}")
            
            # Parse the data
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                raise DataSourceError("No data returned from Alpha Vantage")
            
            # Convert to DataFrame
            df_data = []
            for date, values in time_series.items():
                df_data.append({
                    'Date': date,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['6. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Cache the result
            self.cache_data(cache_key, df)
            
            response_time = time.time() - start_time
            self.update_metrics(response_time, True, 1.0)
            
            return df
            
        except Exception as e:
            response_time = time.time() - start_time
            self.update_metrics(response_time, False, 0.0)
            raise DataSourceError(f"Alpha Vantage error: {str(e)}")
    
    def _convert_to_alpha_vantage_ticker(self, ticker: str) -> str:
        """Convert ticker to Alpha Vantage format."""
        # Remove -USD suffix for crypto
        if ticker.endswith('-USD'):
            return ticker.replace('-USD', '')
        return ticker


class CoinGeckoDataSource(AbstractDataSource):
    """CoinGecko data source implementation."""
    
    def __init__(self):
        super().__init__("CoinGecko")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.interval_mapping = {
            "1h": "hourly",
            "4h": "hourly",  # Will resample
            "1d": "daily",
            "1w": "weekly"
        }
    
    def get_supported_intervals(self) -> List[str]:
        return ["1d", "1w"]  # CoinGecko has limited intraday
    
    def get_max_history_days(self) -> int:
        return 365 * 5  # 5 years of historical data
    
    def download_data(self, ticker: str, start_date: str, end_date: str, 
                     interval: str = "1d") -> pd.DataFrame:
        """Download data from CoinGecko."""
        start_time = time.time()
        cache_key = self.get_cache_key(ticker, start_date, end_date, interval)
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Convert ticker to CoinGecko format
            coin_id = self._convert_to_coingecko_id(ticker)
            
            # Calculate days between dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' not in data:
                raise DataSourceError("No price data returned from CoinGecko")
            
            # Parse the data
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                date = datetime.fromtimestamp(timestamp / 1000)
                volume = volumes[i][1] if i < len(volumes) else 0
                
                df_data.append({
                    'Date': date,
                    'Open': price,  # CoinGecko doesn't provide OHLC, use close price
                    'High': price,
                    'Low': price,
                    'Close': price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Cache the result
            self.cache_data(cache_key, df)
            
            response_time = time.time() - start_time
            self.update_metrics(response_time, True, 0.8)  # Lower quality due to no OHLC
            
            return df
            
        except Exception as e:
            response_time = time.time() - start_time
            self.update_metrics(response_time, False, 0.0)
            raise DataSourceError(f"CoinGecko error: {str(e)}")
    
    def _convert_to_coingecko_id(self, ticker: str) -> str:
        """Convert ticker to CoinGecko ID."""
        # Map common tickers to CoinGecko IDs
        ticker_map = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'ADA-USD': 'cardano',
            'DOT-USD': 'polkadot',
            'LINK-USD': 'chainlink',
            'LTC-USD': 'litecoin',
            'BCH-USD': 'bitcoin-cash',
            'XRP-USD': 'ripple',
            'BNB-USD': 'binancecoin',
            'SOL-USD': 'solana'
        }
        
        return ticker_map.get(ticker, ticker.lower().replace('-usd', ''))


class DataSourceFactory:
    """Factory for creating data sources."""
    
    _sources = {}
    
    @classmethod
    def register_source(cls, name: str, source_class: type):
        """Register a data source class."""
        cls._sources[name] = source_class
    
    @classmethod
    def create_source(cls, name: str, **kwargs) -> AbstractDataSource:
        """Create a data source instance."""
        if name not in cls._sources:
            raise ValueError(f"Unknown data source: {name}")
        
        return cls._sources[name](**kwargs)
    
    @classmethod
    def get_available_sources(cls) -> List[str]:
        """Get list of available data sources."""
        return list(cls._sources.keys())


# Register data sources
DataSourceFactory.register_source("yahoo", YahooFinanceDataSource)
DataSourceFactory.register_source("alphavantage", AlphaVantageDataSource)
DataSourceFactory.register_source("coingecko", CoinGeckoDataSource)


class MultiDataSourceManager:
    """Manages multiple data sources with fallback and fusion capabilities."""
    
    def __init__(self, primary_source: str = "yahoo", 
                 fallback_sources: List[str] = None,
                 api_keys: Dict[str, str] = None):
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or ["coingecko"]
        self.api_keys = api_keys or {}
        
        # Initialize data sources
        self.sources = {}
        self._initialize_sources()
        
        # Performance tracking
        self.source_performance = {}
    
    def _initialize_sources(self):
        """Initialize all data sources."""
        try:
            self.sources["yahoo"] = DataSourceFactory.create_source("yahoo")
        except Exception as e:
            logging.warning(f"Failed to initialize Yahoo Finance: {e}")
        
        try:
            if "alphavantage" in [self.primary_source] + self.fallback_sources:
                api_key = self.api_keys.get("alphavantage")
                if api_key:
                    self.sources["alphavantage"] = DataSourceFactory.create_source(
                        "alphavantage", api_key=api_key
                    )
        except Exception as e:
            logging.warning(f"Failed to initialize Alpha Vantage: {e}")
        
        try:
            if "coingecko" in [self.primary_source] + self.fallback_sources:
                self.sources["coingecko"] = DataSourceFactory.create_source("coingecko")
        except Exception as e:
            logging.warning(f"Failed to initialize CoinGecko: {e}")
    
    def download_data_with_fallback(self, ticker: str, start_date: str, end_date: str,
                                  interval: str = "1d") -> Tuple[pd.DataFrame, str]:
        """Download data with automatic fallback to other sources."""
        sources_to_try = [self.primary_source] + self.fallback_sources
        
        for source_name in sources_to_try:
            if source_name not in self.sources:
                continue
            
            try:
                source = self.sources[source_name]
                data = source.download_data(ticker, start_date, end_date, interval)
                
                if not data.empty:
                    return data, source_name
                    
            except Exception as e:
                logging.warning(f"Failed to download from {source_name}: {e}")
                continue
        
        raise DataSourceError(f"All data sources failed for {ticker}")
    
    def download_data_fusion(self, ticker: str, start_date: str, end_date: str,
                           interval: str = "1d") -> pd.DataFrame:
        """Download data from multiple sources and fuse them for better quality."""
        available_data = {}
        
        # Download from all available sources
        for source_name, source in self.sources.items():
            try:
                data = source.download_data(ticker, start_date, end_date, interval)
                if not data.empty:
                    available_data[source_name] = data
            except Exception as e:
                logging.warning(f"Failed to download from {source_name}: {e}")
        
        if not available_data:
            raise DataSourceError(f"No data available from any source for {ticker}")
        
        # If only one source, return it
        if len(available_data) == 1:
            return list(available_data.values())[0]
        
        # Fuse data from multiple sources
        return self._fuse_data(available_data)
    
    def _fuse_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fuse data from multiple sources."""
        # Use the source with the highest quality score as primary
        best_source = max(data_dict.keys(), 
                         key=lambda x: self.sources[x].metrics.data_quality_score)
        
        primary_data = data_dict[best_source]
        
        # For now, return the best quality data
        # In a more sophisticated implementation, we could:
        # - Cross-validate data across sources
        # - Fill gaps using other sources
        # - Weight data by source reliability
        
        return primary_data
    
    def get_source_metrics(self) -> Dict[str, DataSourceMetrics]:
        """Get performance metrics for all sources."""
        return {name: source.metrics for name, source in self.sources.items()}
    
    def get_best_source(self) -> str:
        """Get the best performing source based on metrics."""
        if not self.sources:
            return self.primary_source
        
        best_source = max(self.sources.keys(),
                         key=lambda x: self.sources[x].metrics.success_rate)
        return best_source


# Convenience functions for easy usage
def create_data_manager(primary_source: str = "yahoo", 
                       fallback_sources: List[str] = None,
                       api_keys: Dict[str, str] = None) -> MultiDataSourceManager:
    """Create a data manager with specified sources."""
    return MultiDataSourceManager(primary_source, fallback_sources, api_keys)


def download_multi_source_data(ticker: str, start_date: str, end_date: str,
                             interval: str = "1d",
                             use_fusion: bool = False,
                             **kwargs) -> pd.DataFrame:
    """Download data using multiple sources."""
    manager = create_data_manager(**kwargs)
    
    if use_fusion:
        return manager.download_data_fusion(ticker, start_date, end_date, interval)
    else:
        data, source_name = manager.download_data_with_fallback(
            ticker, start_date, end_date, interval
        )
        return data 