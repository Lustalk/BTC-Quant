"""
Feature Selection Module
Recursive feature elimination and importance analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection using recursive feature elimination and importance analysis
    """

    def __init__(self):
        """Initialize feature selector"""
        self.feature_importance = None
        self.selected_features = []
        self.rfe_results = None

    def calculate_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index

        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance...")

        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_

        # Handle NaN values before F-statistic calculation
        X_clean = X.fillna(method='ffill').fillna(method='bfill')
        
        # F-statistic importance
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(X_clean, y)
        f_scores = f_selector.scores_

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf_importance,
            'f_score': f_scores,
            'combined_score': (rf_importance + f_scores / f_scores.max()) / 2
        })

        # Sort by combined score
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        self.feature_importance = importance_df
        logger.info(f"Feature importance calculated for {len(X.columns)} features")

        return importance_df

    def recursive_feature_elimination(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> Dict:
        """
        Perform recursive feature elimination

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index

        Returns:
            Dictionary with RFE results
        """
        logger.info("Performing recursive feature elimination...")

        # Use XGBoost for RFE
        import xgboost as xgb
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )

        # RFE with cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)

        # Test different numbers of features
        n_features_range = range(5, min(21, len(X.columns) + 1))
        rfe_results = []

        for n_features in n_features_range:
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=1
            )

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Fit RFE
                rfe.fit(X_train, y_train)
                
                # Transform features
                X_train_selected = rfe.transform(X_train)
                X_val_selected = rfe.transform(X_val)

                # Train model on selected features
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_train_selected, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            rfe_results.append({
                'n_features': n_features,
                'mean_auc': np.mean(scores),
                'std_auc': np.std(scores),
                'selected_features': X.columns[rfe.support_].tolist()
            })

        self.rfe_results = pd.DataFrame(rfe_results)
        logger.info(f"RFE completed for {len(n_features_range)} feature counts")

        return {
            'results': self.rfe_results,
            'best_n_features': self.rfe_results.loc[self.rfe_results['mean_auc'].idxmax(), 'n_features'],
            'best_auc': self.rfe_results['mean_auc'].max()
        }

    def get_optimal_feature_set(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex, max_features: int = None
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Get optimal feature set based on importance and RFE

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index
            max_features: Maximum number of features to select

        Returns:
            Tuple of (optimal_features, combined_results)
        """
        logger.info("Determining optimal feature set...")

        # Calculate feature importance
        importance_df = self.calculate_feature_importance(X, y, dates)

        # Perform RFE
        rfe_results = self.recursive_feature_elimination(X, y, dates)

        # Get best RFE result
        best_rfe = rfe_results['results'].loc[rfe_results['results']['mean_auc'].idxmax()]
        best_features = best_rfe['selected_features']

        # Combine importance and RFE results
        combined_df = importance_df.copy()
        combined_df['in_rfe_best'] = combined_df['feature'].isin(best_features)
        combined_df['rfe_rank'] = combined_df['feature'].apply(
            lambda x: best_features.index(x) + 1 if x in best_features else np.nan
        )

        # Select optimal features
        if max_features:
            optimal_features = best_features[:max_features]
        else:
            optimal_features = best_features

        self.selected_features = optimal_features

        logger.info(f"Selected {len(optimal_features)} optimal features")
        logger.info(f"Best AUC with selected features: {best_rfe['mean_auc']:.4f}")

        return optimal_features, combined_df

    def export_feature_importance(self, filepath: str = None) -> str:
        """
        Export feature importance results

        Args:
            filepath: Output file path

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = "exports/feature_importance.csv"

        if self.feature_importance is not None:
            self.feature_importance.to_csv(filepath, index=False)
            logger.info(f"Feature importance exported to {filepath}")
            return filepath
        else:
            logger.warning("No feature importance data to export")
            return ""

    def plot_feature_importance(self, save_path: str = None) -> None:
        """
        Plot feature importance

        Args:
            save_path: Path to save the plot
        """
        if self.feature_importance is None:
            logger.warning("No feature importance data to plot")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))
        
        # Top 15 features
        top_features = self.feature_importance.head(15)
        
        plt.subplot(2, 1, 1)
        sns.barplot(data=top_features, x='combined_score', y='feature')
        plt.title('Feature Importance (Combined Score)')
        plt.xlabel('Importance Score')

        plt.subplot(2, 1, 2)
        sns.barplot(data=top_features, x='rf_importance', y='feature')
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance Score')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()

    def validate_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> Dict:
        """
        Validate feature selection with different feature sets

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date index

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating feature selection...")

        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb

        # Test different feature sets
        feature_sets = {
            'all_features': X.columns.tolist(),
            'top_10_importance': self.feature_importance.head(10)['feature'].tolist(),
            'top_15_importance': self.feature_importance.head(15)['feature'].tolist(),
            'rfe_selected': self.selected_features,
        }

        results = []
        tscv = TimeSeriesSplit(n_splits=3)

        for set_name, features in feature_sets.items():
            X_subset = X[features]
            
            scores = []
            for train_idx, val_idx in tscv.split(X_subset):
                X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            results.append({
                'feature_set': set_name,
                'n_features': len(features),
                'mean_auc': np.mean(scores),
                'std_auc': np.std(scores),
                'features': features
            })

        validation_results = pd.DataFrame(results)
        logger.info("Feature selection validation completed")

        return {
            'results': validation_results,
            'best_set': validation_results.loc[validation_results['mean_auc'].idxmax(), 'feature_set']
        }


def main():
    """Test feature selection"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    
    # Run feature selection
    selector = FeatureSelector()
    optimal_features, combined_df = selector.get_optimal_feature_set(X, y, dates, max_features=10)
    
    # Validate selection
    validation_results = selector.validate_feature_selection(X, y, dates)
    
    print("Feature selection completed successfully!")
    print(f"Selected {len(optimal_features)} features")


if __name__ == "__main__":
    main() 