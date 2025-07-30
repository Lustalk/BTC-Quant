"""
Test file for data pipeline module
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    """Test cases for DataPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = DataPipeline(symbol='SPY', start_date='2023-01-01', end_date='2023-12-31')
        
    def test_download_data(self):
        """Test data download functionality"""
        data = self.pipeline.download_data()
        
        # Check that data was downloaded
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check data types
        self.assertTrue(data.index.dtype == 'datetime64[ns]')
        
    def test_calculate_returns(self):
        """Test return calculations"""
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
        
    def test_add_price_levels(self):
        """Test price level calculations"""
        # Create sample data
        sample_data = pd.DataFrame({
            'high': [110, 111, 109, 112, 113],
            'low': [90, 91, 89, 92, 93],
            'close': [100, 101, 99, 102, 103]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        result = self.pipeline.add_price_levels(sample_data)
        
        # Check that price levels were added
        self.assertIn('price_to_52w_high', result.columns)
        self.assertIn('price_to_52w_low', result.columns)
        
    def test_preprocess_data(self):
        """Test complete preprocessing pipeline"""
        data = self.pipeline.preprocess_data()
        
        # Check that preprocessing was successful
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        
        # Check that required columns exist
        required_columns = ['returns', 'target', 'forward_returns']
        for col in required_columns:
            self.assertIn(col, data.columns)

if __name__ == '__main__':
    unittest.main() 