"""
Enhanced Walk-Forward Validation Module
50+ out-of-sample periods with expanding window approach
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix, classification_report
)
import xgboost as xgb

logger = logging.getLogger(__name__)

class EnhancedWalkForwardValidator:
    """
    Enhanced walk-forward validation with 50+ out-of-sample periods
    """
    
    def __init__(self, initial_train_years=2, test_period_months=3, 
                 min_train_size=500, expanding_window=True):
        self.initial_train_years = initial_train_years
        self.test_period_months = test_period_months
        self.min_train_size = min_train_size
        self.expanding_window = expanding_window
        self.validation_periods = []
        self.period_metrics = []
        self.daily_predictions = []
        
    def create_validation_periods(self, data):
        """
        Create validation periods with expanding window approach
        """
        logger.info("Creating validation periods with expanding window...")
        
        # Calculate initial training end date
        start_date = data.index[0]
        initial_train_end = start_date + pd.DateOffset(years=self.initial_train_years)
        
        # Calculate test period duration
        test_period_days = self.test_period_months * 30  # Approximate
        
        current_train_end = initial_train_end
        periods = []
        
        while current_train_end < data.index[-1]:
            # Define test period
            test_start = current_train_end
            test_end = min(test_start + pd.DateOffset(days=test_period_days), data.index[-1])
            
            # Define training period (expanding window)
            if self.expanding_window:
                train_start = start_date
            else:
                # Rolling window approach
                train_start = test_start - pd.DateOffset(years=self.initial_train_years)
            
            train_end = test_start
            
            # Ensure minimum training size
            train_data = data[train_start:train_end]
            if len(train_data) < self.min_train_size:
                logger.warning(f"Training period too small: {len(train_data)} samples. Skipping.")
                current_train_end = test_end
                continue
            
            period = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(train_data),
                'test_size': len(data[test_start:test_end])
            }
            
            periods.append(period)
            current_train_end = test_end
            
            # Log progress
            if len(periods) % 10 == 0:
                logger.info(f"Created {len(periods)} validation periods...")
        
        self.validation_periods = periods
        logger.info(f"Created {len(periods)} validation periods")
        
        return periods
    
    def validate_model_period(self, data, model, feature_columns, period):
        """
        Validate model on a single period
        """
        # Split data for this period
        train_data = data[period['train_start']:period['train_end']]
        test_data = data[period['test_start']:period['test_end']]
        
        # Prepare features and target
        X_train = train_data[feature_columns]
        y_train = train_data['target']
        X_test = test_data[feature_columns]
        y_test = test_data['target']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'period': len(self.period_metrics) + 1,
            'train_start': period['train_start'],
            'train_end': period['train_end'],
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'train_size': period['train_size'],
            'test_size': period['test_size'],
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]
        
        # Calculate financial metrics
        returns = test_data['returns']
        strategy_returns = self.calculate_strategy_returns(returns, y_pred_proba)
        
        metrics.update({
            'buy_hold_return': (1 + returns).prod() - 1,
            'strategy_return': (1 + strategy_returns).prod() - 1,
            'excess_return': metrics['strategy_return'] - metrics['buy_hold_return'],
            'sharpe_ratio': self.calculate_sharpe_ratio(strategy_returns),
            'max_drawdown': self.calculate_max_drawdown(strategy_returns),
            'win_rate': (strategy_returns > 0).mean(),
        })
        
        # Store daily predictions
        daily_preds = pd.DataFrame({
            'date': test_data.index,
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'period': metrics['period']
        })
        
        self.daily_predictions.append(daily_preds)
        self.period_metrics.append(metrics)
        
        return metrics, daily_preds
    
    def calculate_strategy_returns(self, returns, probabilities, threshold=0.52):
        """
        Calculate strategy returns based on predictions
        """
        positions = (probabilities > threshold).astype(int)
        strategy_returns = positions * returns
        return strategy_returns
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def validate_model(self, data, model, feature_columns):
        """
        Perform enhanced walk-forward validation
        """
        logger.info("Starting enhanced walk-forward validation...")
        
        # Create validation periods
        periods = self.create_validation_periods(data)
        
        # Validate on each period
        for i, period in enumerate(periods):
            logger.info(f"Validating period {i+1}/{len(periods)}: "
                       f"{period['test_start'].date()} to {period['test_end'].date()}")
            
            try:
                metrics, daily_preds = self.validate_model_period(
                    data, model, feature_columns, period
                )
                
                # Log period results
                logger.info(f"  AUC: {metrics['auc_roc']:.3f}, "
                           f"F1: {metrics['f1_score']:.3f}, "
                           f"Return: {metrics['strategy_return']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in period {i+1}: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics()
        
        # Export results
        self.export_validation_results()
        
        logger.info(f"Enhanced validation completed: {len(self.period_metrics)} periods")
        logger.info(f"Overall AUC: {overall_metrics['auc_roc']:.3f}")
        logger.info(f"Overall Sharpe: {overall_metrics['sharpe_ratio']:.3f}")
        
        return {
            'period_metrics': self.period_metrics,
            'daily_predictions': pd.concat(self.daily_predictions, ignore_index=True),
            'overall_metrics': overall_metrics,
            'validation_periods': periods
        }
    
    def calculate_overall_metrics(self):
        """
        Calculate overall metrics across all periods
        """
        if not self.period_metrics:
            return {}
        
        df = pd.DataFrame(self.period_metrics)
        
        overall_metrics = {
            'total_periods': len(df),
            'auc_roc': df['auc_roc'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1_score': df['f1_score'].mean(),
            'accuracy': df['accuracy'].mean(),
            'sharpe_ratio': df['sharpe_ratio'].mean(),
            'strategy_return': df['strategy_return'].mean(),
            'buy_hold_return': df['buy_hold_return'].mean(),
            'excess_return': df['excess_return'].mean(),
            'max_drawdown': df['max_drawdown'].mean(),
            'win_rate': df['win_rate'].mean(),
        }
        
        # Calculate standard deviations
        for metric in ['auc_roc', 'precision', 'recall', 'f1_score', 'accuracy', 
                      'sharpe_ratio', 'strategy_return', 'excess_return']:
            overall_metrics[f'{metric}_std'] = df[metric].std()
        
        return overall_metrics
    
    def export_validation_results(self):
        """
        Export validation results to CSV files
        """
        # Export period metrics
        if self.period_metrics:
            period_df = pd.DataFrame(self.period_metrics)
            export_path = 'exports/performance_metrics.csv'
            period_df.to_csv(export_path, index=False)
            logger.info(f"Performance metrics exported to: {export_path}")
        
        # Export daily predictions
        if self.daily_predictions:
            daily_df = pd.concat(self.daily_predictions, ignore_index=True)
            export_path = 'exports/daily_predictions.csv'
            daily_df.to_csv(export_path, index=False)
            logger.info(f"Daily predictions exported to: {export_path}")
        
        # Export overall metrics
        overall_metrics = self.calculate_overall_metrics()
        if overall_metrics:
            overall_df = pd.DataFrame([overall_metrics])
            export_path = 'exports/overall_metrics.csv'
            overall_df.to_csv(export_path, index=False)
            logger.info(f"Overall metrics exported to: {export_path}")
    
    def plot_validation_results(self, save_path=None):
        """
        Plot validation results
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            if not self.period_metrics:
                logger.warning("No validation metrics available for plotting")
                return None
            
            df = pd.DataFrame(self.period_metrics)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('AUC-ROC by Period', 'Sharpe Ratio by Period',
                              'Strategy vs Buy & Hold Returns', 'Cumulative Returns',
                              'Win Rate by Period', 'Max Drawdown by Period'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # AUC-ROC by period
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['auc_roc'],
                    mode='lines+markers',
                    name='AUC-ROC'
                ),
                row=1, col=1
            )
            
            # Sharpe ratio by period
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['sharpe_ratio'],
                    mode='lines+markers',
                    name='Sharpe Ratio'
                ),
                row=1, col=2
            )
            
            # Strategy vs Buy & Hold returns
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['strategy_return'],
                    mode='lines+markers',
                    name='Strategy Return'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['buy_hold_return'],
                    mode='lines+markers',
                    name='Buy & Hold Return'
                ),
                row=2, col=1
            )
            
            # Cumulative returns
            cumulative_strategy = (1 + df['strategy_return']).cumprod()
            cumulative_buyhold = (1 + df['buy_hold_return']).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=cumulative_strategy,
                    mode='lines',
                    name='Cumulative Strategy'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=cumulative_buyhold,
                    mode='lines',
                    name='Cumulative Buy & Hold'
                ),
                row=2, col=2
            )
            
            # Win rate by period
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['win_rate'],
                    mode='lines+markers',
                    name='Win Rate'
                ),
                row=3, col=1
            )
            
            # Max drawdown by period
            fig.add_trace(
                go.Scatter(
                    x=df['period'],
                    y=df['max_drawdown'],
                    mode='lines+markers',
                    name='Max Drawdown'
                ),
                row=3, col=2
            )
            
            fig.update_layout(
                title='Enhanced Walk-Forward Validation Results',
                height=1000,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Validation plots saved to: {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available. Skipping validation plots.")
            return None
    
    def get_period_summary(self):
        """
        Get summary statistics of validation periods
        """
        if not self.period_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.period_metrics)
        
        summary = {
            'total_periods': len(df),
            'avg_train_size': df['train_size'].mean(),
            'avg_test_size': df['test_size'].mean(),
            'min_train_size': df['train_size'].min(),
            'max_train_size': df['train_size'].max(),
            'min_test_size': df['test_size'].min(),
            'max_test_size': df['test_size'].max(),
            'avg_auc': df['auc_roc'].mean(),
            'avg_sharpe': df['sharpe_ratio'].mean(),
            'avg_strategy_return': df['strategy_return'].mean(),
            'avg_excess_return': df['excess_return'].mean(),
        }
        
        return pd.DataFrame([summary]) 