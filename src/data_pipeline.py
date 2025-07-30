"""
Data Pipeline Module
Handles data acquisition, preprocessing, and storage for the technical indicator alpha project
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os

from config import DATA_CONFIG, PATHS_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Data pipeline for downloading and preprocessing financial market data
    """
    
    def __init__(self, symbol: str = None, start_date: str = None, end_date: str = None):
        """
        Initialize data pipeline
        
        Args:
            symbol: Stock symbol (default from config)
            start_date: Start date for data (default from config)
            end_date: End date for data (default from config)
        """
        self.symbol = symbol or DATA_CONFIG['symbol']
        self.start_date = start_date or DATA_CONFIG['start_date']
        self.end_date = end_date or DATA_CONFIG['end_date']
        self.data = None
        
    def download_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Download market data from Yahoo Finance
        
        Args:
            force_download: Force re-download even if file exists
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = os.path.join(PATHS_CONFIG['raw_data_dir'], f"{self.symbol}_data.csv")
        
        # Check if data already exists
        if not force_download and os.path.exists(file_path):
            logger.info(f"Loading existing data from {file_path}")
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return self.data
        
        logger.info(f"Downloading {self.symbol} data from {self.start_date} to {self.end_date}")
        
        try:
            # Download data using yfinance
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )
            
            # Clean column names
            self.data.columns = [col.lower() for col in self.data.columns]
            
            # Remove any rows with missing data
            self.data = self.data.dropna()
            
            # Save to file
            os.makedirs(PATHS_CONFIG['raw_data_dir'], exist_ok=True)
            self.data.to_csv(file_path)
            
            logger.info(f"Downloaded {len(self.data)} observations for {self.symbol}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def calculate_returns(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate various return metrics
        
        Args:
            data: DataFrame with OHLCV data (uses self.data if None)
            
        Returns:
            DataFrame with return calculations
        """
        if data is None:
            data = self.data.copy()
        
        # Calculate daily returns
        data['returns'] = data['close'].pct_change()
        
        # Calculate log returns
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate forward returns (for target variable)
        horizon = DATA_CONFIG['target_horizon']
        data['forward_returns'] = data['close'].shift(-horizon) / data['close'] - 1
        
        # Calculate target variable (binary classification)
        data['target'] = (data['forward_returns'] > 0).astype(int)
        
        # Calculate volatility
        data['volatility_20d'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return data
    
    def add_price_levels(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add price level indicators (52-week high/low)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price level indicators
        """
        if data is None:
            data = self.data.copy()
        
        # 52-week high and low
        data['high_52w'] = data['high'].rolling(window=252).max()
        data['low_52w'] = data['low'].rolling(window=252).min()
        
        # Price relative to 52-week levels
        data['price_to_52w_high'] = data['close'] / data['high_52w']
        data['price_to_52w_low'] = data['close'] / data['low_52w']
        
        return data
    
    def preprocess_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline
        
        Args:
            force_download: Force re-download of data
            
        Returns:
            Preprocessed DataFrame
        """
        # Download data
        self.download_data(force_download=force_download)
        
        # Calculate returns and target
        self.data = self.calculate_returns(self.data)
        
        # Add price levels
        self.data = self.add_price_levels(self.data)
        
        # Remove rows with NaN values (from rolling calculations)
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        final_rows = len(self.data)
        
        logger.info(f"Preprocessing complete: {initial_rows - final_rows} rows removed due to NaN values")
        logger.info(f"Final dataset: {final_rows} observations from {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
        return self.data
    
    def get_train_test_split(self, train_end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets for walk-forward validation
        
        Args:
            train_end_date: End date for training set (default: 2 years from start)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if train_end_date is None:
            # Default: use first 2 years for training
            train_end_date = pd.to_datetime(self.start_date) + timedelta(days=2*365)
        
        train_data = self.data[self.data.index <= train_end_date].copy()
        test_data = self.data[self.data.index > train_end_date].copy()
        
        logger.info(f"Train set: {len(train_data)} observations until {train_end_date.date()}")
        logger.info(f"Test set: {len(test_data)} observations from {(train_end_date + timedelta(days=1)).date()}")
        
        return train_data, test_data
    
    def save_processed_data(self, filename: str = None) -> str:
        """
        Save processed data to file
        
        Args:
            filename: Custom filename (default: symbol_processed.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.symbol}_processed.csv"
        
        file_path = os.path.join(PATHS_CONFIG['processed_data_dir'], filename)
        os.makedirs(PATHS_CONFIG['processed_data_dir'], exist_ok=True)
        
        self.data.to_csv(file_path)
        logger.info(f"Processed data saved to {file_path}")
        
        return file_path
    
    def load_processed_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load processed data from file
        
        Args:
            filename: Custom filename (default: symbol_processed.csv)
            
        Returns:
            DataFrame with processed data
        """
        if filename is None:
            filename = f"{self.symbol}_processed.csv"
        
        file_path = os.path.join(PATHS_CONFIG['processed_data_dir'], filename)
        
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded processed data from {file_path}")
            return self.data
        else:
            logger.warning(f"Processed data file not found: {file_path}")
            return None

def main():
    """Test the data pipeline"""
    pipeline = DataPipeline()
    data = pipeline.preprocess_data()
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Target distribution: {data['target'].value_counts().to_dict()}")

if __name__ == "__main__":
    main() 