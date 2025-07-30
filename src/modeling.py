"""
Modeling Module
XGBoost model implementation for technical indicator alpha prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GridSearchCV
import joblib
import os

from config import XGBOOST_CONFIG, PATHS_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost model for technical indicator alpha prediction
    """

    def __init__(self, config: Dict = None):
        """
        Initialize XGBoost model

        Args:
            config: Model configuration dictionary
        """
        self.config = config or XGBOOST_CONFIG.copy()
        self.model = None
        self.feature_importance = None
        self.is_trained = False

    def set_hyperparameters(self, hyperparameters: Dict) -> None:
        """
        Set hyperparameters for the model

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.config.update(hyperparameters)
        logger.info(f"Updated hyperparameters: {hyperparameters}")

    def create_model(self) -> xgb.XGBClassifier:
        """
        Create XGBoost model with specified configuration

        Returns:
            Configured XGBoost classifier
        """
        model = xgb.XGBClassifier(
            objective=self.config["objective"],
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            random_state=self.config["random_state"],
            eval_metric=self.config["eval_metric"],
            early_stopping_rounds=self.config["early_stopping_rounds"],
        )

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> "XGBoostModel":
        """
        Train the XGBoost model

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Training XGBoost model with {len(X)} samples and {len(X.columns)} features"
        )

        # Create model
        self.model = self.create_model()

        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set with {len(X_val)} samples")

        # Train model
        if eval_set is not None:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            # Remove early stopping if no validation set
            model_config = self.config.copy()
            model_config["early_stopping_rounds"] = None
            self.model = xgb.XGBClassifier(
                objective=model_config["objective"],
                n_estimators=model_config["n_estimators"],
                max_depth=model_config["max_depth"],
                learning_rate=model_config["learning_rate"],
                subsample=model_config["subsample"],
                colsample_bytree=model_config["colsample_bytree"],
                reg_alpha=model_config["reg_alpha"],
                reg_lambda=model_config["reg_lambda"],
                random_state=model_config["random_state"],
                eval_metric=model_config["eval_metric"],
            )
            self.model.fit(X, y, verbose=False)

        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(X.columns)

        self.is_trained = True
        logger.info("Model training completed")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions

        Args:
            X: Features for prediction

        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions

        Args:
            X: Features for prediction

        Returns:
            Probability predictions [P(class=0), P(class=1)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            proba = self.model.predict_proba(X)
            # Ensure 2D array for binary classification
            if proba.ndim == 1:
                # If 1D, reshape to 2D with shape (n_samples, 1)
                proba = proba.reshape(-1, 1)
                # Add the complementary probability for binary classification
                proba = np.column_stack([1 - proba, proba])
            return proba
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            # Return default probabilities if prediction fails
            n_samples = len(X)
            return np.ones((n_samples, 2)) * 0.5

    def _calculate_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate and format feature importance

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            return pd.DataFrame()

        # Get feature importance scores
        importance_scores = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_scores}
        )

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Calculate percentage
        importance_df["percentage"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )

        return importance_df

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with top N features
        """
        if self.feature_importance is None:
            return pd.DataFrame()

        return self.feature_importance.head(top_n)

    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """
        Plot feature importance

        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        try:
            # import matplotlib.pyplot as plt
            import seaborn as sns

            if self.feature_importance is None:
                logger.warning("No feature importance available")
                return

            # Get top features
            top_features = self.feature_importance.head(top_n)

            # Create plot
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x="percentage", y="feature")
            plt.title(f"Top {top_n} Feature Importance (XGBoost)")
            plt.xlabel("Importance (%)")
            plt.ylabel("Feature")
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Feature importance plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")

    def save_model(self, filepath: str = None) -> str:
        """
        Save trained model to file

        Args:
            filepath: Path to save model (default: models/xgboost_model.pkl)

        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        if filepath is None:
            filepath = os.path.join(PATHS_CONFIG["models_dir"], "xgboost_model.pkl")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_importance": self.feature_importance,
            "config": self.config,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    def load_model(self, filepath: str = None) -> "XGBoostModel":
        """
        Load trained model from file

        Args:
            filepath: Path to model file (default: models/xgboost_model.pkl)

        Returns:
            Self for method chaining
        """
        if filepath is None:
            filepath = os.path.join(PATHS_CONFIG["models_dir"], "xgboost_model.pkl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model data
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.feature_importance = model_data["feature_importance"]
        self.config = model_data["config"]
        self.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {filepath}")

        return self

    def hyperparameter_tuning(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 3
    ) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X: Training features
            y: Training targets
            cv_folds: Number of CV folds

        Returns:
            Dictionary with best parameters and scores
        """
        logger.info("Starting hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        # Create base model
        base_model = self.create_model()

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y)

        # Update model with best parameters
        self.config.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the trained model

        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}

        info = {
            "status": "trained",
            "n_features": (
                len(self.feature_importance)
                if self.feature_importance is not None
                else 0
            ),
            "n_samples": (
                len(self.model.classes_) if hasattr(self.model, "classes_") else 0
            ),
            "config": self.config,
            "feature_importance_available": self.feature_importance is not None,
            "feature_names": (
                self.feature_importance["feature"].tolist()
                if self.feature_importance is not None
                else []
            ),
        }

        if self.feature_importance is not None:
            info["top_features"] = self.feature_importance.head(5)["feature"].tolist()

        return info


def main():
    """Test the XGBoost model"""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    pipeline = DataPipeline()
    data = pipeline.preprocess_data()

    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)

    # Split data for testing
    split_idx = int(len(data_ready) * 0.8)
    train_data = data_ready.iloc[:split_idx]
    test_data = data_ready.iloc[split_idx:]

    X_train = train_data[feature_engineer.feature_columns]
    y_train = train_data["target"]
    X_test = test_data[feature_engineer.feature_columns]
    y_test = test_data["target"]

    # Create and train model
    model = XGBoostModel()
    model.fit(X_train, y_train, X_test, y_test)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Print results
    print("Model Training Results:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {np.mean(predictions == y_test):.3f}")
    print(f"Top 5 features: {model.get_feature_importance(5)['feature'].tolist()}")


if __name__ == "__main__":
    main()
