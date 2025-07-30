"""
Feature Selection Module
Recursive Feature Elimination (RFE) with cross-validation for time series data
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Advanced feature selection using RFE with time series cross-validation
    """
    
    def __init__(self, cv_splits=5, random_state=42):
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.feature_importance = None
        self.selected_features = None
        self.rfe_results = None
        
    def recursive_feature_elimination(self, X, y, time_index, n_features_to_select=None):
        """
        Perform Recursive Feature Elimination with time series cross-validation
        """
        logger.info(f"Starting RFE with {X.shape[1]} features...")
        
        # Use XGBoost as the estimator for RFE
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        if n_features_to_select is None:
            # Use RFECV to automatically determine optimal number of features
            rfe = RFECV(
                estimator=estimator,
                step=1,
                cv=tscv,
                scoring='roc_auc',
                min_features_to_select=5,
                n_jobs=-1
            )
        else:
            # Use RFE with specified number of features
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=1
            )
        
        # Fit RFE
        rfe.fit(X, y)
        
        # Store results
        self.rfe_results = {
            'support': rfe.support_,
            'ranking': rfe.ranking_,
            'n_features': rfe.n_features_,
            'selected_features': X.columns[rfe.support_].tolist(),
            'eliminated_features': X.columns[~rfe.support_].tolist(),
            'feature_ranking': dict(zip(X.columns, rfe.ranking_))
        }
        
        if hasattr(rfe, 'cv_results_'):
            self.rfe_results['cv_results'] = rfe.cv_results_
        
        logger.info(f"RFE completed. Selected {len(self.rfe_results['selected_features'])} features")
        logger.info(f"Selected features: {self.rfe_results['selected_features']}")
        
        return self.rfe_results
    
    def calculate_feature_importance(self, X, y, time_index):
        """
        Calculate feature importance using XGBoost
        """
        logger.info("Calculating feature importance...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        importance_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Get feature importance
            importance = model.feature_importances_
            importance_scores.append(importance)
        
        # Average importance across CV folds
        avg_importance = np.mean(importance_scores, axis=0)
        
        # Create feature importance DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': avg_importance,
            'importance_std': np.std(importance_scores, axis=0),
            'importance_rank': np.argsort(np.argsort(-avg_importance)) + 1
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance calculated successfully")
        return self.feature_importance
    
    def stability_analysis(self, X, y, time_index, n_iterations=10):
        """
        Perform stability analysis of feature selection
        """
        logger.info(f"Performing stability analysis with {n_iterations} iterations...")
        
        stability_results = []
        feature_counts = {}
        
        for i in range(n_iterations):
            # Bootstrap sampling for stability analysis
            bootstrap_indices = np.random.choice(
                len(X), size=len(X), replace=True
            )
            
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            
            # Perform RFE on bootstrap sample
            estimator = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state + i
            )
            
            tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for bootstrap
            rfe = RFECV(
                estimator=estimator,
                step=1,
                cv=tscv,
                scoring='roc_auc',
                min_features_to_select=5
            )
            
            rfe.fit(X_bootstrap, y_bootstrap)
            
            # Store selected features
            selected_features = X.columns[rfe.support_].tolist()
            stability_results.append(selected_features)
            
            # Count feature occurrences
            for feature in selected_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Calculate stability metrics
        total_features = len(X.columns)
        stability_metrics = {}
        
        for feature in X.columns:
            count = feature_counts.get(feature, 0)
            stability_metrics[feature] = {
                'selection_frequency': count / n_iterations,
                'selection_count': count,
                'stability_rank': np.argsort(np.argsort([-count])) + 1
            }
        
        stability_df = pd.DataFrame(stability_metrics).T
        stability_df = stability_df.sort_values('selection_frequency', ascending=False)
        
        logger.info("Stability analysis completed")
        return stability_df, stability_results
    
    def export_feature_importance(self):
        """
        Export feature importance rankings to CSV
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not calculated. Run calculate_feature_importance first.")
            return None
        
        # Export feature importance
        export_path = 'exports/feature_importance.csv'
        self.feature_importance.to_csv(export_path, index=False)
        logger.info(f"Feature importance exported to: {export_path}")
        
        # Export RFE results if available
        if self.rfe_results:
            rfe_df = pd.DataFrame({
                'feature': list(self.rfe_results['feature_ranking'].keys()),
                'ranking': list(self.rfe_results['feature_ranking'].values()),
                'selected': [f in self.rfe_results['selected_features'] for f in self.rfe_results['feature_ranking'].keys()]
            }).sort_values('ranking')
            
            rfe_path = 'exports/rfe_results.csv'
            rfe_df.to_csv(rfe_path, index=False)
            logger.info(f"RFE results exported to: {rfe_path}")
        
        return export_path
    
    def plot_feature_importance(self, save_path=None, top_n=20):
        """
        Plot feature importance
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not calculated. Run calculate_feature_importance first.")
            return None
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Get top features
            top_features = self.feature_importance.head(top_n)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Feature Importance', 'Importance Distribution', 
                              'RFE Ranking', 'Stability Analysis'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # Feature importance bar chart
            fig.add_trace(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    name='Importance Score',
                    error_x=dict(type='data', array=top_features['importance_std'])
                ),
                row=1, col=1
            )
            
            # Importance distribution
            fig.add_trace(
                go.Histogram(
                    x=self.feature_importance['importance'],
                    name='Importance Distribution'
                ),
                row=1, col=2
            )
            
            # RFE ranking if available
            if self.rfe_results:
                rfe_df = pd.DataFrame({
                    'feature': list(self.rfe_results['feature_ranking'].keys()),
                    'ranking': list(self.rfe_results['feature_ranking'].values())
                }).sort_values('ranking')
                
                fig.add_trace(
                    go.Scatter(
                        x=rfe_df['ranking'],
                        y=rfe_df['feature'],
                        mode='markers',
                        name='RFE Ranking',
                        marker=dict(
                            size=10,
                            color=rfe_df['ranking'],
                            colorscale='Viridis'
                        )
                    ),
                    row=2, col=1
                )
            
            # Feature correlation heatmap (top features)
            if len(top_features) > 1:
                top_feature_data = self.feature_importance.head(top_n)
                correlation_matrix = np.random.rand(len(top_feature_data), len(top_feature_data))  # Placeholder
                
                fig.add_trace(
                    go.Heatmap(
                        z=correlation_matrix,
                        x=top_feature_data['feature'],
                        y=top_feature_data['feature'],
                        colorscale='RdBu',
                        name='Feature Correlation'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Feature Selection Analysis',
                height=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Feature selection plots saved to: {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available. Using matplotlib instead.")
            
            # Fallback to matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Feature importance
            top_features = self.feature_importance.head(top_n)
            axes[0, 0].barh(range(len(top_features)), top_features['importance'])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features['feature'])
            axes[0, 0].set_title('Feature Importance')
            axes[0, 0].set_xlabel('Importance Score')
            
            # Importance distribution
            axes[0, 1].hist(self.feature_importance['importance'], bins=20)
            axes[0, 1].set_title('Importance Distribution')
            axes[0, 1].set_xlabel('Importance Score')
            
            # RFE ranking
            if self.rfe_results:
                rfe_df = pd.DataFrame({
                    'feature': list(self.rfe_results['feature_ranking'].keys()),
                    'ranking': list(self.rfe_results['feature_ranking'].values())
                }).sort_values('ranking')
                
                axes[1, 0].scatter(rfe_df['ranking'], range(len(rfe_df)))
                axes[1, 0].set_yticks(range(len(rfe_df)))
                axes[1, 0].set_yticklabels(rfe_df['feature'])
                axes[1, 0].set_title('RFE Ranking')
                axes[1, 0].set_xlabel('Ranking')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
                logger.info(f"Feature selection plots saved to: {save_path.replace('.html', '.png')}")
            
            return fig
    
    def get_optimal_feature_set(self, X, y, time_index, max_features=20):
        """
        Get optimal feature set based on importance and stability
        """
        # Calculate feature importance
        self.calculate_feature_importance(X, y, time_index)
        
        # Perform RFE
        self.recursive_feature_elimination(X, y, time_index)
        
        # Get stability analysis
        stability_df, _ = self.stability_analysis(X, y, time_index)
        
        # Combine metrics for optimal selection
        combined_df = self.feature_importance.copy()
        combined_df['stability_score'] = combined_df['feature'].map(
            stability_df['selection_frequency']
        ).fillna(0)
        
        # Calculate combined score
        combined_df['combined_score'] = (
            0.6 * combined_df['importance'] + 
            0.4 * combined_df['stability_score']
        )
        
        # Select optimal features
        optimal_features = combined_df.nlargest(max_features, 'combined_score')['feature'].tolist()
        
        logger.info(f"Optimal feature set selected: {len(optimal_features)} features")
        logger.info(f"Optimal features: {optimal_features}")
        
        return optimal_features, combined_df 