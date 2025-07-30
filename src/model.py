import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from typing import Tuple, List


def prepare_features_target(
    data: pd.DataFrame, target_lookahead: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for the model.

    Args:
        data (pd.DataFrame): DataFrame with technical indicators
        target_lookahead (int): Number of periods to look ahead for target

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    # Create target: 1 if price goes up, 0 if down
    data = data.copy()
    data["Target"] = (data["Close"].shift(-target_lookahead) > data["Close"]).astype(
        int
    )

    # Remove the last few rows where we don't have targets
    data = data.dropna()

    # Select feature columns (exclude OHLCV and Target)
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Target"]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols]
    y = data["Target"]

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with default parameters.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets

    Returns:
        xgb.XGBClassifier: Trained model
    """
    model = xgb.XGBClassifier(
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def walk_forward_validation(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> List[float]:
    """
    Perform walk-forward validation on time series data.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Targets
        n_splits (int): Number of splits for time series cross-validation

    Returns:
        List[float]: List of accuracy scores for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"Processing fold {fold}/{n_splits}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model = train_model(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

        print(f"Fold {fold} accuracy: {accuracy:.4f}")

    return scores
