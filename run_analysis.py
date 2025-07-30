#!/usr/bin/env python3
"""
Simple script to run comprehensive strategy analysis
Demonstrates how the system answers your primary question
"""

import sys
import os
import logging

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Run comprehensive strategy analysis to answer the primary question
    """
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY ANALYSIS")
    print("=" * 80)
    print("Primary Question:")
    print("Can you prove, with statistically significant evidence, that your model's")
    print("trading strategy generates positive risk-adjusted returns on out-of-sample")
    print("data when compared to a simple buy-and-hold benchmark?")
    print("=" * 80)
    
    try:
        # Step 1: Setup
        logger.info("Setting up analysis...")
        create_directories()
        
        # Step 2: Data preparation
        logger.info("Loading and preprocessing data...")
        pipeline = DataPipeline(start_date="2020-01-01", end_date="2024-12-31")
        data = pipeline.preprocess_data()
        
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        
        # Step 3: Model development
        logger.info("Developing model...")
        hyperopt = HyperparameterOptimizer(n_trials=20)
        X = data_ready[feature_engineer.feature_columns]
        y = data_ready["target"]
        best_params, _ = hyperopt.optimize_hyperparameters(X, y, data_ready.index)
        
        feature_selector = FeatureSelector()
        optimal_features, _ = feature_selector.get_optimal_feature_set(
            X, y, data_ready.index, max_features=15
        )
        
        model = XGBoostModel()
        model.set_hyperparameters(best_params)
        
        # Step 4: Comprehensive analysis
        logger.info("Running comprehensive analysis...")
        strategy_analyzer = StrategyAnalyzer()
        results = strategy_analyzer.analyze_strategy(
            data=data_ready,
            model=model,
            feature_columns=optimal_features,
            target_column="target"
        )
        
        # Step 5: Display results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        strategy_analyzer.print_comprehensive_report()
        
        # Step 6: Export results
        analysis_file = strategy_analyzer.export_analysis_results()
        print(f"\nResults exported to: {analysis_file}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 