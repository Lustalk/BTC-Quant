"""
Data Quality Control Module
Professional data validation and cleaning for financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataQualityControl:
    """Professional data quality control for financial time series."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize quality control with configuration."""
        self.config = config or {
            'max_missing_pct': 0.05,
            'outlier_threshold': 3.0,
            'min_data_points': 100,
            'max_price_change': 0.5  # 50% daily change threshold
        }
    
    def validate_data_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate basic data structure requirements."""
        errors = []
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} is not numeric")
        
        # Check for empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")
        
        return len(errors) == 0, errors
    
    def detect_missing_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Detect and analyze missing data patterns."""
        missing_info = {
            'total_rows': len(df),
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'has_missing': df.isnull().any().any()
        }
        
        # Check if missing data exceeds threshold
        max_missing = max(missing_info['missing_percentages'].values()) if missing_info['missing_percentages'] else 0
        is_acceptable = max_missing <= self.config['max_missing_pct'] * 100
        
        return is_acceptable, missing_info
    
    def detect_outliers(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Detect outliers using statistical methods."""
        outliers_info = {}
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Calculate z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.config['outlier_threshold']
                
                outliers_info[col] = {
                    'outlier_count': outliers.sum(),
                    'outlier_percentage': (outliers.sum() / len(df)) * 100,
                    'outlier_indices': df[outliers].index.tolist()
                }
        
        # Check for extreme price changes
        if 'Close' in df.columns and len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            extreme_changes = price_changes > self.config['max_price_change']
            outliers_info['extreme_price_changes'] = {
                'count': extreme_changes.sum(),
                'percentage': (extreme_changes.sum() / len(df)) * 100,
                'indices': df[extreme_changes].index.tolist()
            }
        
        return True, outliers_info
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality validation."""
        results = {
            'is_valid': False,
            'structure_valid': False,
            'missing_data_acceptable': False,
            'outliers_detected': False,
            'errors': [],
            'warnings': []
        }
        
        # Structure validation
        structure_valid, structure_errors = self.validate_data_structure(df)
        results['structure_valid'] = structure_valid
        results['errors'].extend(structure_errors)
        
        if not structure_valid:
            return results
        
        # Missing data analysis
        missing_acceptable, missing_info = self.detect_missing_data(df)
        results['missing_data_acceptable'] = missing_acceptable
        
        if not missing_acceptable:
            results['warnings'].append(f"High missing data percentage: {max(missing_info['missing_percentages'].values()):.2f}%")
        
        # Outlier detection
        outliers_ok, outliers_info = self.detect_outliers(df)
        results['outliers_detected'] = outliers_ok
        
        # Overall validation
        results['is_valid'] = (
            structure_valid and 
            missing_acceptable and 
            len(df) >= self.config['min_data_points']
        )
        
        if len(df) < self.config['min_data_points']:
            results['warnings'].append(f"Insufficient data points: {len(df)} < {self.config['min_data_points']}")
        
        return results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        cleaned_df = df.copy()
        
        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Forward fill for small gaps in price data
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
        
        # Fill volume with 0 if missing
        if 'Volume' in cleaned_df.columns:
            cleaned_df['Volume'] = cleaned_df['Volume'].fillna(0)
        
        # Remove extreme outliers (optional)
        if 'Close' in cleaned_df.columns:
            z_scores = np.abs((cleaned_df['Close'] - cleaned_df['Close'].mean()) / cleaned_df['Close'].std())
            extreme_outliers = z_scores > 5.0  # Very strict threshold
            if extreme_outliers.sum() > 0:
                logger.warning(f"Removing {extreme_outliers.sum()} extreme outliers")
                cleaned_df = cleaned_df[~extreme_outliers]
        
        return cleaned_df

def validate_bitcoin_data(df: pd.DataFrame) -> Dict:
    """Convenience function for Bitcoin data validation."""
    qc = DataQualityControl()
    return qc.validate_data_quality(df)

def clean_bitcoin_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for Bitcoin data cleaning."""
    qc = DataQualityControl()
    return qc.clean_data(df) 