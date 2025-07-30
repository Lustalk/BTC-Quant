"""
Advanced Hyperparameter Optimization Module
Combines Grid Search and Bayesian Optimization for XGBoost with Time Series CV
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
from datetime import datetime
import logging
import os
import json

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for XGBoost with time series cross-validation
    """
    
    def __init__(self, n_trials=100, cv_splits=5, random_state=42):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_params = None
        self.optimization_history = []
        
    def objective(self, trial, X, y, time_index):
        """
        Objective function for Optuna optimization
        """
        # Define hyperparameter search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Combined score (weighted average)
            combined_score = 0.4 * auc_score + 0.2 * precision + 0.2 * recall + 0.2 * f1
            scores.append(combined_score)
        
        # Return mean score across CV folds
        mean_score = np.mean(scores)
        
        # Store trial results
        trial_result = {
            'trial_number': trial.number,
            'params': params,
            'mean_score': mean_score,
            'cv_scores': scores,
            'timestamp': datetime.now().isoformat()
        }
        self.optimization_history.append(trial_result)
        
        return mean_score
    
    def optimize_hyperparameters(self, X, y, time_index):
        """
        Perform Bayesian optimization of hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, X, y, time_index),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state
        })
        
        logger.info(f"Best hyperparameters found: {self.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return self.best_params, study
    
    def grid_search_validation(self, X, y, time_index, param_grid=None):
        """
        Perform grid search validation on best parameters
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
            }
        
        logger.info("Performing grid search validation...")
        
        best_grid_score = -np.inf
        best_grid_params = None
        grid_results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state
            })
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
            
            mean_score = np.mean(scores)
            grid_results.append({
                'params': params,
                'mean_score': mean_score,
                'cv_scores': scores
            })
            
            if mean_score > best_grid_score:
                best_grid_score = mean_score
                best_grid_params = params
        
        logger.info(f"Grid search best score: {best_grid_score:.4f}")
        logger.info(f"Grid search best params: {best_grid_params}")
        
        return best_grid_params, grid_results
    
    def export_results(self, study, grid_results=None):
        """
        Export optimization results to CSV
        """
        # Prepare optimization history for export
        export_data = []
        
        for trial_result in self.optimization_history:
            row = {
                'trial_number': trial_result['trial_number'],
                'timestamp': trial_result['timestamp'],
                'mean_score': trial_result['mean_score'],
                'cv_std': np.std(trial_result['cv_scores']),
                'cv_min': np.min(trial_result['cv_scores']),
                'cv_max': np.max(trial_result['cv_scores']),
            }
            
            # Add hyperparameters
            for param, value in trial_result['params'].items():
                if param not in ['objective', 'eval_metric', 'random_state']:
                    row[f'param_{param}'] = value
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        
        # Add grid search results if available
        if grid_results:
            grid_data = []
            for i, result in enumerate(grid_results):
                row = {
                    'trial_number': f'grid_{i}',
                    'timestamp': datetime.now().isoformat(),
                    'mean_score': result['mean_score'],
                    'cv_std': np.std(result['cv_scores']),
                    'cv_min': np.min(result['cv_scores']),
                    'cv_max': np.max(result['cv_scores']),
                }
                
                for param, value in result['params'].items():
                    if param not in ['objective', 'eval_metric', 'random_state']:
                        row[f'param_{param}'] = value
                
                grid_data.append(row)
            
            grid_df = pd.DataFrame(grid_data)
            df = pd.concat([df, grid_df], ignore_index=True)
        
        # Export to CSV
        export_path = 'exports/hyperparameter_results.csv'
        df.to_csv(export_path, index=False)
        logger.info(f"Hyperparameter results exported to: {export_path}")
        
        # Export best parameters as JSON
        best_params_export = {
            'bayesian_optimization': {
                'best_params': self.best_params,
                'best_score': study.best_value,
                'optimization_trials': len(self.optimization_history)
            },
            'grid_search': {
                'best_params': grid_results[0]['params'] if grid_results else None,
                'best_score': grid_results[0]['mean_score'] if grid_results else None,
                'grid_combinations': len(grid_results) if grid_results else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = 'exports/best_hyperparameters.json'
        with open(json_path, 'w') as f:
            json.dump(best_params_export, f, indent=2)
        logger.info(f"Best hyperparameters exported to: {json_path}")
        
        return export_path, json_path
    
    def plot_optimization_history(self, study, save_path=None):
        """
        Plot optimization history
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Optimization History', 'Parameter Importance', 
                              'Score Distribution', 'Best Parameters'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "table"}]]
            )
            
            # Optimization history
            trials_df = study.trials_dataframe()
            fig.add_trace(
                go.Scatter(
                    x=trials_df['number'],
                    y=trials_df['value'],
                    mode='lines+markers',
                    name='Optimization History'
                ),
                row=1, col=1
            )
            
            # Parameter importance
            importance = optuna.importance.get_param_importances(study)
            fig.add_trace(
                go.Bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    name='Parameter Importance'
                ),
                row=1, col=2
            )
            
            # Score distribution
            fig.add_trace(
                go.Histogram(
                    x=trials_df['value'],
                    name='Score Distribution'
                ),
                row=2, col=1
            )
            
            # Best parameters table
            best_params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v} 
                for k, v in self.best_params.items()
                if k not in ['objective', 'eval_metric', 'random_state']
            ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Parameter', 'Value']),
                    cells=dict(values=[best_params_df['Parameter'], best_params_df['Value']])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Hyperparameter Optimization Results',
                height=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Optimization plots saved to: {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available. Skipping optimization plots.")
            return None 