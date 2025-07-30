"""
Hyperparameter Optimization Module
Bayesian optimization for XGBoost hyperparameters
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

from config import XGBOOST_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization for XGBoost model
    """

    def __init__(self, n_trials: int = 50):
        """
        Initialize hyperparameter optimizer

        Args:
            n_trials: Number of optimization trials
        """
        self.n_trials = n_trials
        self.best_params = None
        self.study = None

    def optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> Tuple[Dict, optuna.Study]:
        """
        Optimize hyperparameters using Bayesian optimization

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index for time series splitting

        Returns:
            Tuple of (best_params, study)
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")

        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "random_state": 42,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                import xgboost as xgb
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    verbose=False
                )

                # Predict
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate AUC
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            return np.mean(scores)

        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_params.update({
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })

        logger.info(f"Best AUC: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params, self.study

    def grid_search_validation(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Validate optimization results with grid search

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index

        Returns:
            Tuple of (best_grid_params, results_df)
        """
        logger.info("Running grid search validation...")

        # Define parameter grid
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.8, 0.9, 1.0],
        }

        results = []
        tscv = TimeSeriesSplit(n_splits=3)

        # Grid search
        for n_estimators in param_grid["n_estimators"]:
            for max_depth in param_grid["max_depth"]:
                for learning_rate in param_grid["learning_rate"]:
                    for subsample in param_grid["subsample"]:
                        params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "random_state": 42,
                            "objective": "binary:logistic",
                        }

                        scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                            import xgboost as xgb
                            model = xgb.XGBClassifier(**params)
                            model.fit(X_train, y_train, verbose=False)
                            
                            y_pred_proba = model.predict_proba(X_val)[:, 1]
                            auc = roc_auc_score(y_val, y_pred_proba)
                            scores.append(auc)

                        results.append({
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "mean_auc": np.mean(scores),
                            "std_auc": np.std(scores),
                        })

        results_df = pd.DataFrame(results)
        best_grid_params = results_df.loc[results_df["mean_auc"].idxmax()].to_dict()

        logger.info(f"Grid search best AUC: {best_grid_params['mean_auc']:.4f}")

        return best_grid_params, results_df

    def export_results(self, study: optuna.Study, grid_results: pd.DataFrame) -> None:
        """
        Export optimization results

        Args:
            study: Optuna study object
            grid_results: Grid search results DataFrame
        """
        # Export study results
        study_df = study.trials_dataframe()
        study_df.to_csv("exports/hyperparameter_optimization.csv", index=False)

        # Export grid search results
        grid_results.to_csv("exports/grid_search_validation.csv", index=False)

        # Export best parameters
        best_params_df = pd.DataFrame([self.best_params])
        best_params_df.to_csv("exports/best_hyperparameters.csv", index=False)

        logger.info("Hyperparameter optimization results exported")


def main():
    """Test hyperparameter optimization"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(n_trials=10)
    best_params, study = optimizer.optimize_hyperparameters(X, y, dates)
    
    # Run grid search validation
    grid_params, grid_results = optimizer.grid_search_validation(X, y, dates)
    
    print("Optimization completed successfully!")


if __name__ == "__main__":
    main() 