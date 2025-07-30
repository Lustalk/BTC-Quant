"""
Walk-Forward Validation Module
Implements expanding window walk-forward validation for time series modeling
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from config import VALIDATION_CONFIG, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Walk-forward validation for time series modeling
    Implements expanding window approach to prevent lookahead bias
    """
    
    def __init__(self, 
                 initial_train_years: int = None,
                 test_period_months: int = None,
                 min_train_size: int = None):
        """
        Initialize walk-forward validator
        
        Args:
            initial_train_years: Initial training period in years
            test_period_months: Test period length in months
            min_train_size: Minimum training samples required
        """
        self.initial_train_years = initial_train_years or VALIDATION_CONFIG['initial_train_years']
        self.test_period_months = test_period_months or VALIDATION_CONFIG['test_period_months']
        self.min_train_size = min_train_size or VALIDATION_CONFIG['min_train_size']
        
        self.train_periods = []
        self.test_periods = []
        self.predictions = []
        self.actuals = []
        self.probabilities = []
        
    def generate_walk_forward_periods(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train-test splits for walk-forward validation
        
        Args:
            data: Full dataset with datetime index
            
        Returns:
            List of (train_data, test_data) tuples
        """
        logger.info("Generating walk-forward validation periods...")
        
        # Calculate initial training end date
        start_date = data.index[0]
        initial_train_end = start_date + timedelta(days=365 * self.initial_train_years)
        
        # Calculate test period duration
        test_period_days = self.test_period_months * 30
        
        periods = []
        current_train_end = initial_train_end
        
        while current_train_end < data.index[-1]:
            # Define train and test periods
            train_data = data[data.index <= current_train_end].copy()
            test_start = current_train_end + timedelta(days=1)
            test_end = min(test_start + timedelta(days=test_period_days), data.index[-1])
            test_data = data[(data.index >= test_start) & (data.index <= test_end)].copy()
            
            # Check minimum training size
            if len(train_data) >= self.min_train_size and len(test_data) > 0:
                periods.append((train_data, test_data))
                
                # Store period information
                self.train_periods.append({
                    'start': train_data.index[0],
                    'end': train_data.index[-1],
                    'size': len(train_data)
                })
                
                self.test_periods.append({
                    'start': test_data.index[0],
                    'end': test_data.index[-1],
                    'size': len(test_data)
                })
            
            # Move to next period (expanding window)
            current_train_end = test_end
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        logger.info(f"Training periods range: {periods[0][0].index[0].date()} to {periods[-1][0].index[-1].date()}")
        logger.info(f"Testing periods range: {periods[0][1].index[0].date()} to {periods[-1][1].index[-1].date()}")
        
        return periods
    
    def validate_model(self, 
                      data: pd.DataFrame,
                      model,
                      feature_columns: List[str],
                      target_column: str = 'target') -> Dict:
        """
        Perform walk-forward validation
        
        Args:
            data: Full dataset
            model: Model object with fit() and predict_proba() methods
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting walk-forward validation...")
        
        # Generate periods
        periods = self.generate_walk_forward_periods(data)
        
        # Initialize results storage
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        period_results = []
        
        # Perform validation for each period
        for i, (train_data, test_data) in enumerate(periods):
            logger.info(f"Period {i+1}/{len(periods)}: "
                       f"Train: {len(train_data)} samples, "
                       f"Test: {len(test_data)} samples")
            
            try:
                # Prepare features
                X_train = train_data[feature_columns].dropna()
                y_train = train_data.loc[X_train.index, target_column]
                
                X_test = test_data[feature_columns].dropna()
                y_test = test_data.loc[X_test.index, target_column]
                
                logger.info(f"Period {i+1}: X_train shape after dropna: {X_train.shape}, X_test shape: {X_test.shape}")
                
                # Skip if insufficient data
                if len(X_train) < self.min_train_size or len(X_test) == 0:
                    logger.warning(f"Skipping period {i+1}: insufficient data (X_train: {len(X_train)}, min_required: {self.min_train_size})")
                    continue
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                probabilities = model.predict_proba(X_test)[:, 1]
                predictions = (probabilities > DATA_CONFIG['threshold']).astype(int)
                
                # Store results
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                all_probabilities.extend(probabilities)
                
                # Calculate period metrics
                period_metrics = self._calculate_period_metrics(y_test, predictions, probabilities)
                period_metrics['period'] = i + 1
                period_metrics['train_size'] = len(X_train)
                period_metrics['test_size'] = len(X_test)
                period_metrics['train_start'] = train_data.index[0].date()
                period_metrics['train_end'] = train_data.index[-1].date()
                period_metrics['test_start'] = test_data.index[0].date()
                period_metrics['test_end'] = test_data.index[-1].date()
                
                period_results.append(period_metrics)
                
            except Exception as e:
                logger.error(f"Error in period {i+1}: {e}")
                continue
        
        # Calculate overall metrics
        overall_results = self._calculate_overall_metrics(all_actuals, all_predictions, all_probabilities)
        
        # Store results
        self.predictions = all_predictions
        self.actuals = all_actuals
        self.probabilities = all_probabilities
        
        results = {
            'overall_metrics': overall_results,
            'period_results': period_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'probabilities': all_probabilities,
            'train_periods': self.train_periods,
            'test_periods': self.test_periods
        }
        
        logger.info("Walk-forward validation completed")
        logger.info(f"Total predictions: {len(all_predictions)}")
        logger.info(f"Overall accuracy: {overall_results['accuracy']:.3f}")
        logger.info(f"Overall AUC-ROC: {overall_results['auc_roc']:.3f}")
        
        return results
    
    def _calculate_period_metrics(self, 
                                y_true: pd.Series, 
                                y_pred: np.ndarray, 
                                probabilities: np.ndarray) -> Dict:
        """
        Calculate metrics for a single validation period
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Predicted probabilities
            
        Returns:
            Dictionary with period metrics
        """
        # Basic classification metrics
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-8)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true == 1) + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, probabilities)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
    
    def _calculate_overall_metrics(self, 
                                 y_true: List, 
                                 y_pred: List, 
                                 probabilities: List) -> Dict:
        """
        Calculate overall metrics across all validation periods
        
        Args:
            y_true: List of true labels
            y_pred: List of predicted labels
            probabilities: List of predicted probabilities
            
        Returns:
            Dictionary with overall metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        probabilities = np.array(probabilities)
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-8)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true == 1) + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, probabilities)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'classification_report': report
        }
    
    def get_period_summary(self) -> pd.DataFrame:
        """
        Get summary of all validation periods
        
        Returns:
            DataFrame with period summaries
        """
        if not self.train_periods:
            return pd.DataFrame()
        
        summary_data = []
        for i, (train_period, test_period) in enumerate(zip(self.train_periods, self.test_periods)):
            summary_data.append({
                'period': i + 1,
                'train_start': train_period['start'],
                'train_end': train_period['end'],
                'train_size': train_period['size'],
                'test_start': test_period['start'],
                'test_end': test_period['end'],
                'test_size': test_period['size']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_validation_periods(self, data: pd.DataFrame):
        """
        Plot validation periods for visualization
        
        Args:
            data: Full dataset with datetime index
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Plot price data
            ax.plot(data.index, data['close'], alpha=0.7, label='SPY Price')
            
            # Color train and test periods
            for i, (train_period, test_period) in enumerate(zip(self.train_periods, self.test_periods)):
                # Train period
                train_mask = (data.index >= train_period['start']) & (data.index <= train_period['end'])
                ax.fill_between(data.index, data['close'].min(), data['close'].max(), 
                              where=train_mask, alpha=0.3, color='blue', label='Train' if i == 0 else "")
                
                # Test period
                test_mask = (data.index >= test_period['start']) & (data.index <= test_period['end'])
                ax.fill_between(data.index, data['close'].min(), data['close'].max(), 
                              where=test_mask, alpha=0.3, color='red', label='Test' if i == 0 else "")
            
            ax.set_title('Walk-Forward Validation Periods')
            ax.set_xlabel('Date')
            ax.set_ylabel('SPY Price')
            ax.legend()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")

def main():
    """Test the walk-forward validator"""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    from modeling import XGBoostModel
    
    # Load and prepare data
    pipeline = DataPipeline()
    data = pipeline.preprocess_data()
    
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
    
    # Initialize validator and model
    validator = WalkForwardValidator()
    model = XGBoostModel()
    
    # Perform validation
    results = validator.validate_model(
        data=data_ready,
        model=model,
        feature_columns=feature_engineer.feature_columns
    )
    
    print("Validation Results:")
    print(f"Total periods: {len(results['period_results'])}")
    print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"Overall AUC-ROC: {results['overall_metrics']['auc_roc']:.3f}")

if __name__ == "__main__":
    main() 