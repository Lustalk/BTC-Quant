"""
Comprehensive Strategy Analysis Module
Answers the primary question: Does the model generate statistically significant positive risk-adjusted returns?

This module addresses three sub-questions:
1. Technical Question: Does the model have predictive power?
2. Financial Question: Does predictive power translate to profit?
3. Robustness Question: Is the result real or just luck?
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import norm
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings("ignore")

from config import EVALUATION_CONFIG, DATA_CONFIG, ADVANCED_CONFIG
from src.enhanced_validation import EnhancedWalkForwardValidator
from src.evaluation import PerformanceEvaluator
from src.monte_carlo_simulation import MonteCarloSimulator
from src.professional_visualizations import ProfessionalVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    """
    Comprehensive strategy analysis addressing the three key questions
    """

    def __init__(self):
        """Initialize strategy analyzer"""
        self.results = {}
        self.technical_evidence = {}
        self.financial_evidence = {}
        self.robustness_evidence = {}
        
    def analyze_strategy(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str = "target",
    ) -> Dict:
        """
        Comprehensive strategy analysis addressing all three sub-questions
        
        Args:
            data: Full dataset with features and target
            model: Trained model object
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE STRATEGY ANALYSIS")
        logger.info("=" * 80)
        
        # Question 1: Technical Analysis - Does the model have predictive power?
        logger.info("\n1. TECHNICAL QUESTION: Does the model have predictive power?")
        technical_results = self._analyze_predictive_power(data, model, feature_columns, target_column)
        
        # Question 2: Financial Analysis - Does predictive power translate to profit?
        logger.info("\n2. FINANCIAL QUESTION: Does predictive power translate to profit?")
        financial_results = self._analyze_financial_performance(data, model, feature_columns, target_column)
        
        # Question 3: Robustness Analysis - Is the result real or just luck?
        logger.info("\n3. ROBUSTNESS QUESTION: Is the result real or just luck?")
        robustness_results = self._analyze_statistical_significance(data, model, feature_columns, target_column)
        
        # Final Answer: Primary Question
        logger.info("\n" + "=" * 80)
        logger.info("PRIMARY QUESTION ANSWER")
        logger.info("=" * 80)
        final_answer = self._answer_primary_question(technical_results, financial_results, robustness_results)
        
        # Compile comprehensive results
        self.results = {
            "technical_analysis": technical_results,
            "financial_analysis": financial_results,
            "robustness_analysis": robustness_results,
            "final_answer": final_answer,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.results
    
    def _analyze_predictive_power(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str,
    ) -> Dict:
        """
        Analyze whether the model has predictive power beyond random chance
        
        Returns:
            Dictionary with technical evidence
        """
        logger.info("Analyzing predictive power using walk-forward validation...")
        
        # Enhanced walk-forward validation
        validator = EnhancedWalkForwardValidator(
            initial_train_years=2,
            test_period_months=3,
            min_train_size=500
        )
        
        validation_results = validator.validate_model(
            data=data,
            model=model,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        # Extract key metrics
        overall_metrics = validation_results.get("overall_metrics", {})
        period_metrics = validation_results.get("period_metrics", [])
        
        # Calculate aggregate AUC across all periods
        auc_scores = [period.get("auc_roc", 0) for period in period_metrics]
        mean_auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        
        # Statistical test for AUC > 0.5
        t_stat, p_value = stats.ttest_1samp(auc_scores, 0.5)
        
        # Calculate confidence interval for mean AUC
        confidence_interval = stats.t.interval(
            alpha=0.95,
            df=len(auc_scores) - 1,
            loc=mean_auc,
            scale=stats.sem(auc_scores)
        )
        
        # Additional classification metrics
        precision_scores = [period.get("precision", 0) for period in period_metrics]
        f1_scores = [period.get("f1_score", 0) for period in period_metrics]
        
        mean_precision = np.mean(precision_scores)
        mean_f1 = np.mean(f1_scores)
        
        # Determine if model has predictive power
        has_predictive_power = (
            mean_auc > 0.5 and 
            p_value < 0.05 and 
            confidence_interval[0] > 0.5
        )
        
        technical_evidence = {
            "mean_auc": mean_auc,
            "auc_std": auc_std,
            "auc_p_value": p_value,
            "auc_confidence_interval": confidence_interval,
            "mean_precision": mean_precision,
            "mean_f1_score": mean_f1,
            "has_predictive_power": has_predictive_power,
            "period_metrics": period_metrics,
            "overall_metrics": overall_metrics
        }
        
        # Log results
        logger.info(f"Mean AUC: {mean_auc:.3f} ± {auc_std:.3f}")
        logger.info(f"AUC p-value: {p_value:.4f}")
        logger.info(f"AUC 95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        logger.info(f"Has predictive power: {'YES' if has_predictive_power else 'NO'}")
        
        return technical_evidence
    
    def _analyze_financial_performance(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str,
    ) -> Dict:
        """
        Analyze whether predictive power translates to profitable trading
        
        Returns:
            Dictionary with financial evidence
        """
        logger.info("Analyzing financial performance vs buy-and-hold benchmark...")
        
        # Get predictions from walk-forward validation
        validator = EnhancedWalkForwardValidator(
            initial_train_years=2,
            test_period_months=3,
            min_train_size=500
        )
        
        validation_results = validator.validate_model(
            data=data,
            model=model,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        # Extract predictions and probabilities
        daily_predictions = validation_results.get("daily_predictions", {})
        if not daily_predictions:
            logger.error("No daily predictions available")
            return {}
        
        predictions = daily_predictions.get("predicted", [])
        probabilities = daily_predictions.get("probability", [])
        
        # Calculate strategy returns
        evaluator = PerformanceEvaluator()
        backtest_results = evaluator.backtest_strategy(
            data=data,
            predictions=predictions,
            probabilities=probabilities
        )
        
        # Extract key financial metrics
        strategy_metrics = backtest_results.get("strategy_metrics", {})
        benchmark_metrics = backtest_results.get("benchmark_metrics", {})
        
        # Calculate excess performance
        excess_sharpe = strategy_metrics.get("sharpe_ratio", 0) - benchmark_metrics.get("sharpe_ratio", 0)
        excess_return = strategy_metrics.get("total_return", 0) - benchmark_metrics.get("total_return", 0)
        excess_sortino = strategy_metrics.get("sortino_ratio", 0) - benchmark_metrics.get("sortino_ratio", 0)
        
        # Risk-adjusted performance analysis
        strategy_sharpe = strategy_metrics.get("sharpe_ratio", 0)
        benchmark_sharpe = benchmark_metrics.get("sharpe_ratio", 0)
        
        # Determine if strategy outperforms benchmark
        outperforms_benchmark = (
            strategy_sharpe > benchmark_sharpe and
            excess_sharpe > 0.1  # Minimum meaningful excess
        )
        
        financial_evidence = {
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "excess_sharpe": excess_sharpe,
            "excess_return": excess_return,
            "excess_sortino": excess_sortino,
            "outperforms_benchmark": outperforms_benchmark,
            "backtest_results": backtest_results
        }
        
        # Log results
        logger.info(f"Strategy Sharpe: {strategy_sharpe:.3f}")
        logger.info(f"Benchmark Sharpe: {benchmark_sharpe:.3f}")
        logger.info(f"Excess Sharpe: {excess_sharpe:.3f}")
        logger.info(f"Outperforms benchmark: {'YES' if outperforms_benchmark else 'NO'}")
        
        return financial_evidence
    
    def _analyze_statistical_significance(
        self,
        data: pd.DataFrame,
        model,
        feature_columns: List[str],
        target_column: str,
    ) -> Dict:
        """
        Analyze statistical significance using Monte Carlo simulation
        
        Returns:
            Dictionary with robustness evidence
        """
        logger.info("Analyzing statistical significance with Monte Carlo simulation...")
        
        # Get predictions and returns
        validator = EnhancedWalkForwardValidator(
            initial_train_years=2,
            test_period_months=3,
            min_train_size=500
        )
        
        validation_results = validator.validate_model(
            data=data,
            model=model,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        daily_predictions = validation_results.get("daily_predictions", {})
        if not daily_predictions:
            logger.error("No daily predictions available")
            return {}
        
        probabilities = daily_predictions.get("probability", [])
        returns = data["returns"]
        
        # Run Monte Carlo simulation
        monte_carlo = MonteCarloSimulator(n_simulations=1000)
        simulation_results = monte_carlo.simulate_strategy_performance(
            returns=returns,
            probabilities=probabilities
        )
        
        # Extract key statistical measures
        actual_sharpe = simulation_results.get("actual_sharpe", 0)
        simulation_sharpes = simulation_results.get("simulation_sharpes", [])
        
        # Calculate p-value for Sharpe ratio
        p_value_sharpe = simulation_results.get("p_value_sharpe", 1.0)
        
        # Calculate probability of achieving actual performance by chance
        better_than_actual = sum(1 for s in simulation_sharpes if s >= actual_sharpe)
        probability_by_chance = better_than_actual / len(simulation_sharpes) if simulation_sharpes else 1.0
        
        # Statistical significance threshold
        is_statistically_significant = p_value_sharpe < 0.05
        
        # Calculate confidence intervals
        confidence_interval = np.percentile(simulation_sharpes, [2.5, 97.5])
        
        robustness_evidence = {
            "actual_sharpe": actual_sharpe,
            "p_value_sharpe": p_value_sharpe,
            "probability_by_chance": probability_by_chance,
            "is_statistically_significant": is_statistically_significant,
            "confidence_interval": confidence_interval,
            "simulation_results": simulation_results
        }
        
        # Log results
        logger.info(f"Actual Sharpe: {actual_sharpe:.3f}")
        logger.info(f"Sharpe p-value: {p_value_sharpe:.4f}")
        logger.info(f"Probability by chance: {probability_by_chance:.1%}")
        logger.info(f"Statistically significant: {'YES' if is_statistically_significant else 'NO'}")
        
        return robustness_evidence
    
    def _answer_primary_question(
        self,
        technical_results: Dict,
        financial_results: Dict,
        robustness_results: Dict,
    ) -> Dict:
        """
        Answer the primary question based on all three sub-analyses
        
        Returns:
            Dictionary with final answer and evidence
        """
        logger.info("Synthesizing evidence to answer primary question...")
        
        # Extract key evidence
        has_predictive_power = technical_results.get("has_predictive_power", False)
        outperforms_benchmark = financial_results.get("outperforms_benchmark", False)
        is_statistically_significant = robustness_results.get("is_statistically_significant", False)
        
        # Additional evidence
        mean_auc = technical_results.get("mean_auc", 0)
        excess_sharpe = financial_results.get("excess_sharpe", 0)
        p_value = robustness_results.get("p_value_sharpe", 1.0)
        
        # Decision criteria
        technical_criterion = has_predictive_power and mean_auc > 0.55
        financial_criterion = outperforms_benchmark and excess_sharpe > 0.1
        robustness_criterion = is_statistically_significant and p_value < 0.05
        
        # Final answer
        primary_answer = (
            technical_criterion and
            financial_criterion and
            robustness_criterion
        )
        
        # Confidence level based on evidence strength
        confidence_factors = []
        
        if mean_auc > 0.6:
            confidence_factors.append("Strong predictive power (AUC > 0.6)")
        elif mean_auc > 0.55:
            confidence_factors.append("Moderate predictive power (AUC > 0.55)")
        else:
            confidence_factors.append("Weak predictive power")
            
        if excess_sharpe > 0.2:
            confidence_factors.append("Strong financial outperformance")
        elif excess_sharpe > 0.1:
            confidence_factors.append("Moderate financial outperformance")
        else:
            confidence_factors.append("Weak financial outperformance")
            
        if p_value < 0.01:
            confidence_factors.append("Highly statistically significant")
        elif p_value < 0.05:
            confidence_factors.append("Statistically significant")
        else:
            confidence_factors.append("Not statistically significant")
        
        # Overall confidence
        if primary_answer and len([f for f in confidence_factors if "Strong" in f or "Highly" in f]) >= 2:
            confidence_level = "HIGH"
        elif primary_answer:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
        
        final_answer = {
            "answer": primary_answer,
            "confidence_level": confidence_level,
            "technical_criterion_met": technical_criterion,
            "financial_criterion_met": financial_criterion,
            "robustness_criterion_met": robustness_criterion,
            "confidence_factors": confidence_factors,
            "summary": self._generate_summary(technical_results, financial_results, robustness_results)
        }
        
        # Log final answer
        logger.info(f"\nFINAL ANSWER: {'YES' if primary_answer else 'NO'}")
        logger.info(f"Confidence Level: {confidence_level}")
        logger.info(f"Technical Criterion: {'MET' if technical_criterion else 'NOT MET'}")
        logger.info(f"Financial Criterion: {'MET' if financial_criterion else 'NOT MET'}")
        logger.info(f"Robustness Criterion: {'MET' if robustness_criterion else 'NOT MET'}")
        
        return final_answer
    
    def _generate_summary(
        self,
        technical_results: Dict,
        financial_results: Dict,
        robustness_results: Dict,
    ) -> str:
        """Generate a comprehensive summary of the analysis"""
        
        mean_auc = technical_results.get("mean_auc", 0)
        excess_sharpe = financial_results.get("excess_sharpe", 0)
        p_value = robustness_results.get("p_value_sharpe", 1.0)
        
        summary = f"""
STRATEGY ANALYSIS SUMMARY

Technical Evidence:
- Mean AUC: {mean_auc:.3f} (threshold: 0.55)
- Predictive Power: {'YES' if technical_results.get('has_predictive_power') else 'NO'}

Financial Evidence:
- Excess Sharpe Ratio: {excess_sharpe:.3f} (threshold: 0.1)
- Outperforms Benchmark: {'YES' if financial_results.get('outperforms_benchmark') else 'NO'}

Robustness Evidence:
- Sharpe p-value: {p_value:.4f} (threshold: 0.05)
- Statistically Significant: {'YES' if robustness_results.get('is_statistically_significant') else 'NO'}

CONCLUSION:
The model {'DOES' if self.results.get('final_answer', {}).get('answer') else 'DOES NOT'} generate 
statistically significant positive risk-adjusted returns on out-of-sample data when compared 
to a simple buy-and-hold benchmark.
        """
        
        return summary
    
    def export_analysis_results(self, filepath: str = None) -> str:
        """Export comprehensive analysis results"""
        
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"exports/strategy_analysis_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Analysis results exported to: {filepath}")
        return filepath
    
    def print_comprehensive_report(self):
        """Print a comprehensive analysis report"""
        
        if not self.results:
            logger.error("No analysis results available. Run analyze_strategy() first.")
            return
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE STRATEGY ANALYSIS REPORT")
        print("=" * 80)
        
        # Technical Analysis
        tech = self.results.get("technical_analysis", {})
        print(f"\n1. TECHNICAL ANALYSIS (Predictive Power)")
        print(f"   Mean AUC: {tech.get('mean_auc', 0):.3f}")
        print(f"   AUC p-value: {tech.get('auc_p_value', 1):.4f}")
        print(f"   Has Predictive Power: {'YES' if tech.get('has_predictive_power') else 'NO'}")
        
        # Financial Analysis
        fin = self.results.get("financial_analysis", {})
        print(f"\n2. FINANCIAL ANALYSIS (Profit Translation)")
        print(f"   Strategy Sharpe: {fin.get('strategy_metrics', {}).get('sharpe_ratio', 0):.3f}")
        print(f"   Benchmark Sharpe: {fin.get('benchmark_metrics', {}).get('sharpe_ratio', 0):.3f}")
        print(f"   Excess Sharpe: {fin.get('excess_sharpe', 0):.3f}")
        print(f"   Outperforms Benchmark: {'YES' if fin.get('outperforms_benchmark') else 'NO'}")
        
        # Robustness Analysis
        rob = self.results.get("robustness_analysis", {})
        print(f"\n3. ROBUSTNESS ANALYSIS (Statistical Significance)")
        print(f"   Actual Sharpe: {rob.get('actual_sharpe', 0):.3f}")
        print(f"   Sharpe p-value: {rob.get('p_value_sharpe', 1):.4f}")
        print(f"   Probability by Chance: {rob.get('probability_by_chance', 1):.1%}")
        print(f"   Statistically Significant: {'YES' if rob.get('is_statistically_significant') else 'NO'}")
        
        # Final Answer
        final = self.results.get("final_answer", {})
        print(f"\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)
        print(f"Answer: {'YES' if final.get('answer') else 'NO'}")
        print(f"Confidence Level: {final.get('confidence_level', 'UNKNOWN')}")
        print(f"\n{final.get('summary', 'No summary available')}")


def main():
    """Main function for standalone execution"""
    
    # This would be called from the main analysis pipeline
    logger.info("Strategy analysis module loaded successfully")
    logger.info("Use StrategyAnalyzer.analyze_strategy() to run comprehensive analysis")


if __name__ == "__main__":
    main() 