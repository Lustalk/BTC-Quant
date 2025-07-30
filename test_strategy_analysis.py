#!/usr/bin/env python3
"""
Test Script for Comprehensive Strategy Analysis
Demonstrates how the system answers the primary question:
"Can you prove, with statistically significant evidence, that your model's trading strategy 
generates positive risk-adjusted returns on out-of-sample data when compared to a simple 
buy-and-hold benchmark?"
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import create_directories
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.hyperparameter_optimization import HyperparameterOptimizer
from src.feature_selection import FeatureSelector
from src.modeling import XGBoostModel
from src.strategy_analysis import StrategyAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/strategy_analysis_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_comprehensive_test():
    """
    Run comprehensive test to demonstrate the three sub-questions and primary answer
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE STRATEGY ANALYSIS TEST")
    logger.info("=" * 80)
    logger.info("Testing the system's ability to answer the primary question:")
    logger.info("Does the model generate statistically significant positive risk-adjusted returns?")
    logger.info("=" * 80)
    
    try:
        # Step 1: Setup and Data Preparation
        logger.info("\nStep 1: Data preparation...")
        create_directories()
        
        # Load and preprocess data
        pipeline = DataPipeline(start_date="2020-01-01", end_date="2024-12-31")
        data = pipeline.preprocess_data()
        logger.info(f"Data loaded: {len(data)} observations")
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        logger.info(f"Features created: {len(feature_engineer.feature_columns)} technical indicators")
        
        # Step 2: Model Development
        logger.info("\nStep 2: Model development...")
        
        # Hyperparameter optimization
        hyperopt = HyperparameterOptimizer(n_trials=20)  # Reduced for testing
        X = data_ready[feature_engineer.feature_columns]
        y = data_ready["target"]
        best_params, study = hyperopt.optimize_hyperparameters(X, y, data_ready.index)
        
        # Feature selection
        feature_selector = FeatureSelector()
        optimal_features, _ = feature_selector.get_optimal_feature_set(
            X, y, data_ready.index, max_features=15
        )
        
        # Model training
        model = XGBoostModel()
        model.set_hyperparameters(best_params)
        
        # Step 3: Comprehensive Strategy Analysis
        logger.info("\nStep 3: Running comprehensive strategy analysis...")
        
        strategy_analyzer = StrategyAnalyzer()
        analysis_results = strategy_analyzer.analyze_strategy(
            data=data_ready,
            model=model,
            feature_columns=optimal_features,
            target_column="target"
        )
        
        # Step 4: Display Results
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        # Print comprehensive report
        strategy_analyzer.print_comprehensive_report()
        
        # Step 5: Export Results
        logger.info("\nStep 5: Exporting results...")
        analysis_file = strategy_analyzer.export_analysis_results()
        
        # Step 6: Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        final_answer = analysis_results.get("final_answer", {})
        answer = final_answer.get("answer", False)
        confidence = final_answer.get("confidence_level", "UNKNOWN")
        
        logger.info(f"Primary Question Answer: {'YES' if answer else 'NO'}")
        logger.info(f"Confidence Level: {confidence}")
        
        # Check each sub-question
        technical = analysis_results.get("technical_analysis", {})
        financial = analysis_results.get("financial_analysis", {})
        robustness = analysis_results.get("robustness_analysis", {})
        
        logger.info("\nSub-Question Results:")
        logger.info(f"1. Technical (Predictive Power): {'YES' if technical.get('has_predictive_power') else 'NO'}")
        logger.info(f"2. Financial (Profit Translation): {'YES' if financial.get('outperforms_benchmark') else 'NO'}")
        logger.info(f"3. Robustness (Statistical Significance): {'YES' if robustness.get('is_statistically_significant') else 'NO'}")
        
        # Detailed metrics
        logger.info("\nKey Metrics:")
        logger.info(f"  Mean AUC: {technical.get('mean_auc', 0):.3f}")
        logger.info(f"  Excess Sharpe: {financial.get('excess_sharpe', 0):.3f}")
        logger.info(f"  Sharpe p-value: {robustness.get('p_value_sharpe', 1):.4f}")
        
        logger.info(f"\nResults exported to: {analysis_file}")
        logger.info("=" * 80)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False


def demonstrate_three_questions():
    """
    Demonstrate how each of the three sub-questions is addressed
    """
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING THE THREE SUB-QUESTIONS")
    logger.info("=" * 80)
    
    logger.info("\n1. TECHNICAL QUESTION: Does the model have predictive power?")
    logger.info("   - Uses walk-forward validation with AUC-ROC analysis")
    logger.info("   - Tests if mean AUC > 0.5 with statistical significance")
    logger.info("   - Analyzes precision, F1-score, and classification metrics")
    logger.info("   - Proves model is better than a coin flip")
    
    logger.info("\n2. FINANCIAL QUESTION: Does predictive power translate to profit?")
    logger.info("   - Compares strategy Sharpe ratio vs buy-and-hold benchmark")
    logger.info("   - Analyzes excess returns, Sortino ratio, maximum drawdown")
    logger.info("   - Tests if strategy outperforms simple buy-and-hold")
    logger.info("   - Ensures predictive power creates financial value")
    
    logger.info("\n3. ROBUSTNESS QUESTION: Is the result real or just luck?")
    logger.info("   - Uses Monte Carlo simulation with 1000+ trials")
    logger.info("   - Calculates p-value for observed Sharpe ratio")
    logger.info("   - Tests probability of achieving results by random chance")
    logger.info("   - Provides statistical significance testing")
    
    logger.info("\nPRIMARY QUESTION ANSWER:")
    logger.info("   - Combines all three sub-analyses")
    logger.info("   - Requires ALL criteria to be met:")
    logger.info("     * Technical: AUC > 0.55 with p < 0.05")
    logger.info("     * Financial: Excess Sharpe > 0.1")
    logger.info("     * Robustness: p-value < 0.05")
    logger.info("   - Provides confidence level (HIGH/MODERATE/LOW)")
    logger.info("   - Gives definitive YES/NO answer with evidence")


if __name__ == "__main__":
    logger.info("Starting Comprehensive Strategy Analysis Test")
    logger.info("This test demonstrates how the system answers your primary question")
    
    # Demonstrate the three sub-questions
    demonstrate_three_questions()
    
    # Run the comprehensive test
    success = run_comprehensive_test()
    
    if success:
        logger.info("✅ Test completed successfully!")
        logger.info("📊 Check the logs and exports for detailed results")
    else:
        logger.error("❌ Test failed!")
        sys.exit(1) 