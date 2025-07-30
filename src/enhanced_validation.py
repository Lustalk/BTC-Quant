"""
Enhanced Walk-Forward Validation Module
Advanced validation with statistical significance testing
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

from config import VALIDATION_CONFIG, DATA_CONFIG
from src.validation import WalkForwardValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedWalkForwardValidator(WalkForwardValidator):
    """
    Enhanced walk-forward validation with statistical significance testing
    """

    def __init__(
        self,
        initial_train_years: int = None,
        test_period_months: int = None,
        min_train_size: int = None,
    ):
        """
        Initialize enhanced walk-forward validator

        Args:
            initial_train_years: Initial training period in years
            test_period_months: Test period length in months
            min_train_size: Minimum training samples required
        """
        super().__init__(initial_train_years, test_period_months, min_train_size)
        
        # Enhanced metrics storage
        self.statistical_tests = []
        self.period_analysis = []
        self.performance_degradation = []

    def validate_model(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str = "target",
    ) -> Dict:
        """
        Enhanced model validation with statistical significance testing

        Args:
            data: Full dataset with features and target
            model: Trained model object
            feature_columns: List of feature column names
            target_column: Target column name

        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info("Starting enhanced walk-forward validation...")

        # Generate validation periods
        periods = self.generate_walk_forward_periods(data)
        
        if not periods:
            logger.error("No validation periods generated")
            return {}

        # Initialize results storage
        all_predictions = []
        all_probabilities = []
        all_actuals = []
        period_metrics = []
        daily_predictions = []

        # Validate each period
        for i, (train_data, test_data) in enumerate(periods):
            logger.info(f"Validating period {i+1}/{len(periods)}")
            
            # Prepare data
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate period metrics
            period_metric = self._calculate_enhanced_period_metrics(
                y_test, y_pred, y_pred_proba, i+1
            )
            period_metrics.append(period_metric)

            # Store predictions for daily analysis
            for j, (idx, actual) in enumerate(y_test.items()):
                daily_predictions.append({
                    'period': i+1,
                    'date': idx,
                    'predicted': y_pred[j],
                    'probability': y_pred_proba[j],
                    'actual': actual
                })

            # Store for overall analysis
            all_predictions.extend(y_pred)
            all_probabilities.extend(y_pred_proba)
            all_actuals.extend(y_test.values)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            all_actuals, all_predictions, all_probabilities
        )

        # Perform statistical significance tests
        statistical_tests = self._perform_statistical_tests(
            all_actuals, all_predictions, all_probabilities
        )

        # Analyze performance degradation
        degradation_analysis = self._analyze_performance_degradation(period_metrics)

        # Create comprehensive results
        results = {
            'period_metrics': period_metrics,
            'overall_metrics': overall_metrics,
            'statistical_tests': statistical_tests,
            'degradation_analysis': degradation_analysis,
            'daily_predictions': pd.DataFrame(daily_predictions),
            'validation_periods': len(periods),
            'total_predictions': len(all_predictions)
        }

        logger.info(f"Enhanced validation completed: {len(periods)} periods")
        logger.info(f"Overall AUC: {overall_metrics['auc_roc']:.4f}")
        logger.info(f"Statistical significance: {statistical_tests['is_significant']}")

        return results

    def _calculate_enhanced_period_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray, period: int
    ) -> Dict:
        """
        Calculate enhanced metrics for a single period

        Args:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities
            period: Period number

        Returns:
            Dictionary with enhanced metrics
        """
        # Basic classification metrics
        auc = roc_auc_score(y_true, probabilities)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        # Financial metrics (if returns available)
        financial_metrics = self._calculate_financial_metrics(y_true, probabilities)

        # Statistical metrics
        statistical_metrics = self._calculate_statistical_metrics(y_true, probabilities)

        return {
            'period': period,
            'auc_roc': auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'n_samples': len(y_true),
            'positive_rate': y_true.mean(),
            'prediction_positive_rate': y_pred.mean(),
            **financial_metrics,
            **statistical_metrics
        }

    def _calculate_financial_metrics(self, y_true: pd.Series, probabilities: np.ndarray) -> Dict:
        """
        Calculate financial performance metrics

        Args:
            y_true: True labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary with financial metrics
        """
        # Simple strategy: long when probability > 0.5
        signals = (probabilities > 0.5).astype(int)
        
        # Calculate strategy returns (assuming y_true represents positive returns)
        strategy_returns = signals * y_true
        buy_hold_returns = y_true

        # Calculate metrics
        strategy_mean = strategy_returns.mean()
        strategy_std = strategy_returns.std()
        buy_hold_mean = buy_hold_returns.mean()
        buy_hold_std = buy_hold_returns.std()

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_strategy = strategy_mean / strategy_std if strategy_std > 0 else 0
        sharpe_buy_hold = buy_hold_mean / buy_hold_std if buy_hold_std > 0 else 0

        # Excess return
        excess_return = strategy_mean - buy_hold_mean

        return {
            'strategy_return': strategy_mean,
            'strategy_volatility': strategy_std,
            'buy_hold_return': buy_hold_mean,
            'buy_hold_volatility': buy_hold_std,
            'sharpe_strategy': sharpe_strategy,
            'sharpe_buy_hold': sharpe_buy_hold,
            'excess_return': excess_return,
            'excess_sharpe': sharpe_strategy - sharpe_buy_hold
        }

    def _calculate_statistical_metrics(self, y_true: pd.Series, probabilities: np.ndarray) -> Dict:
        """
        Calculate statistical significance metrics

        Args:
            y_true: True labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary with statistical metrics
        """
        # Chi-square test for independence
        contingency_table = pd.crosstab(y_true, (probabilities > 0.5))
        if contingency_table.shape == (2, 2):
            chi2_stat, chi2_pvalue = stats.chi2_contingency(contingency_table)[:2]
        else:
            chi2_stat, chi2_pvalue = np.nan, np.nan

        # T-test for mean difference
        high_prob_returns = y_true[probabilities > 0.5]
        low_prob_returns = y_true[probabilities <= 0.5]
        
        if len(high_prob_returns) > 0 and len(low_prob_returns) > 0:
            t_stat, t_pvalue = stats.ttest_ind(high_prob_returns, low_prob_returns)
        else:
            t_stat, t_pvalue = np.nan, np.nan

        # Information coefficient (correlation between predictions and actuals)
        ic = np.corrcoef(probabilities, y_true)[0, 1] if len(y_true) > 1 else np.nan

        return {
            'chi2_statistic': chi2_stat,
            'chi2_pvalue': chi2_pvalue,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'information_coefficient': ic,
            'is_significant_chi2': chi2_pvalue < 0.05 if not np.isnan(chi2_pvalue) else False,
            'is_significant_t': t_pvalue < 0.05 if not np.isnan(t_pvalue) else False
        }

    def _perform_statistical_tests(
        self, y_true: List, y_pred: List, probabilities: List
    ) -> Dict:
        """
        Perform comprehensive statistical significance tests

        Args:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary with statistical test results
        """
        logger.info("Performing statistical significance tests...")

        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        probabilities = np.array(probabilities)

        # 1. Test if AUC is significantly different from 0.5
        auc = roc_auc_score(y_true, probabilities)
        auc_se = self._calculate_auc_standard_error(y_true, probabilities)
        auc_z_score = (auc - 0.5) / auc_se if auc_se > 0 else 0
        auc_p_value = 2 * (1 - stats.norm.cdf(abs(auc_z_score)))

        # 2. Test if accuracy is significantly different from random
        accuracy = accuracy_score(y_true, y_pred)
        random_accuracy = 0.5
        accuracy_se = np.sqrt(accuracy * (1 - accuracy) / len(y_true))
        accuracy_z_score = (accuracy - random_accuracy) / accuracy_se if accuracy_se > 0 else 0
        accuracy_p_value = 2 * (1 - stats.norm.cdf(abs(accuracy_z_score)))

        # 3. Test if positive prediction rate is different from actual positive rate
        actual_positive_rate = y_true.mean()
        predicted_positive_rate = y_pred.mean()
        
        if actual_positive_rate > 0 and actual_positive_rate < 1:
            rate_se = np.sqrt(actual_positive_rate * (1 - actual_positive_rate) / len(y_true))
            rate_z_score = (predicted_positive_rate - actual_positive_rate) / rate_se if rate_se > 0 else 0
            rate_p_value = 2 * (1 - stats.norm.cdf(abs(rate_z_score)))
        else:
            rate_z_score, rate_p_value = np.nan, np.nan

        # 4. Overall significance assessment
        p_values = [auc_p_value, accuracy_p_value, rate_p_value]
        significant_tests = sum(1 for p in p_values if p < 0.05 and not np.isnan(p))
        is_significant = significant_tests >= 2  # At least 2 out of 3 tests significant

        return {
            'auc': auc,
            'auc_standard_error': auc_se,
            'auc_z_score': auc_z_score,
            'auc_p_value': auc_p_value,
            'accuracy': accuracy,
            'accuracy_z_score': accuracy_z_score,
            'accuracy_p_value': accuracy_p_value,
            'predicted_positive_rate': predicted_positive_rate,
            'actual_positive_rate': actual_positive_rate,
            'rate_z_score': rate_z_score,
            'rate_p_value': rate_p_value,
            'significant_tests': significant_tests,
            'total_tests': len([p for p in p_values if not np.isnan(p)]),
            'is_significant': is_significant,
            'overall_p_value': np.mean([p for p in p_values if not np.isnan(p)])
        }

    def _calculate_auc_standard_error(self, y_true: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Calculate standard error for AUC

        Args:
            y_true: True labels
            probabilities: Prediction probabilities

        Returns:
            Standard error of AUC
        """
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return np.inf
        
        # Hanley and McNeil formula
        auc = roc_auc_score(y_true, probabilities)
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        
        se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + 
                     (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
        
        return se

    def _analyze_performance_degradation(self, period_metrics: List[Dict]) -> Dict:
        """
        Analyze performance degradation over time

        Args:
            period_metrics: List of period metrics

        Returns:
            Dictionary with degradation analysis
        """
        if len(period_metrics) < 2:
            return {}

        # Extract metrics over time
        periods = [m['period'] for m in period_metrics]
        aucs = [m['auc_roc'] for m in period_metrics]
        accuracies = [m['accuracy'] for m in period_metrics]
        excess_returns = [m.get('excess_return', 0) for m in period_metrics]

        # Calculate trends
        auc_trend = np.polyfit(periods, aucs, 1)[0]
        accuracy_trend = np.polyfit(periods, accuracies, 1)[0]
        excess_return_trend = np.polyfit(periods, excess_returns, 1)[0]

        # Calculate stability metrics
        auc_stability = np.std(aucs)
        accuracy_stability = np.std(accuracies)
        excess_return_stability = np.std(excess_returns)

        # Test for significant degradation
        auc_degradation_p = stats.pearsonr(periods, aucs)[1] if len(periods) > 2 else np.nan
        accuracy_degradation_p = stats.pearsonr(periods, accuracies)[1] if len(periods) > 2 else np.nan

        return {
            'auc_trend': auc_trend,
            'accuracy_trend': accuracy_trend,
            'excess_return_trend': excess_return_trend,
            'auc_stability': auc_stability,
            'accuracy_stability': accuracy_stability,
            'excess_return_stability': excess_return_stability,
            'auc_degradation_significant': auc_degradation_p < 0.05 if not np.isnan(auc_degradation_p) else False,
            'accuracy_degradation_significant': accuracy_degradation_p < 0.05 if not np.isnan(accuracy_degradation_p) else False,
            'overall_stable': auc_stability < 0.1 and accuracy_stability < 0.1
        }

    def export_validation_results(self, results: Dict, filepath: str = None) -> str:
        """
        Export comprehensive validation results

        Args:
            results: Validation results dictionary
            filepath: Output file path

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = "exports/enhanced_validation_results.csv"

        # Create comprehensive results DataFrame
        period_df = pd.DataFrame(results['period_metrics'])
        period_df.to_csv(filepath, index=False)

        # Export statistical tests
        stats_df = pd.DataFrame([results['statistical_tests']])
        stats_df.to_csv(filepath.replace('.csv', '_statistical_tests.csv'), index=False)

        # Export degradation analysis
        if results['degradation_analysis']:
            deg_df = pd.DataFrame([results['degradation_analysis']])
            deg_df.to_csv(filepath.replace('.csv', '_degradation.csv'), index=False)

        logger.info(f"Enhanced validation results exported to {filepath}")
        return filepath

    def print_enhanced_summary(self, results: Dict) -> None:
        """
        Print comprehensive validation summary

        Args:
            results: Validation results dictionary
        """
        print("=" * 80)
        print("ENHANCED WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 80)
        
        # Overall metrics
        overall = results['overall_metrics']
        print(f"Total Periods: {results['validation_periods']}")
        print(f"Total Predictions: {results['total_predictions']}")
        print(f"Overall AUC: {overall['auc_roc']:.4f}")
        print(f"Overall Accuracy: {overall['accuracy']:.4f}")
        print(f"Overall F1-Score: {overall['f1_score']:.4f}")
        
        # Statistical significance
        stats = results['statistical_tests']
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"AUC p-value: {stats['auc_p_value']:.4f}")
        print(f"Accuracy p-value: {stats['accuracy_p_value']:.4f}")
        print(f"Significant tests: {stats['significant_tests']}/{stats['total_tests']}")
        print(f"Overall significant: {stats['is_significant']}")
        
        # Performance degradation
        if results['degradation_analysis']:
            deg = results['degradation_analysis']
            print(f"\nPERFORMANCE DEGRADATION:")
            print(f"AUC trend: {deg['auc_trend']:.6f}")
            print(f"Accuracy trend: {deg['accuracy_trend']:.6f}")
            print(f"Overall stable: {deg['overall_stable']}")
        
        print("=" * 80)


def main():
    """Test enhanced validation"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    
    data = pd.DataFrame(X, index=dates)
    data['target'] = y
    
    # Run enhanced validation
    validator = EnhancedWalkForwardValidator(
        initial_train_years=1,
        test_period_months=3,
        min_train_size=200
    )
    
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    results = validator.validate_model(
        data=data,
        model=model,
        feature_columns=list(range(n_features))
    )
    
    # Print summary
    validator.print_enhanced_summary(results)
    
    print("Enhanced validation completed successfully!")


if __name__ == "__main__":
    main() 