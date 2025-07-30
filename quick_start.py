#!/usr/bin/env python3
"""
Quick Start Script for Technical Indicator Alpha Project
Runs a simplified version of the analysis for testing purposes
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import create_directories
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.validation import WalkForwardValidator
from src.modeling import XGBoostModel
from src.evaluation import PerformanceEvaluator
from src.threshold_optimization import ThresholdOptimizer
from src.risk_management import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_start():
    """
    Quick start function with limited data for testing
    """
    logger.info("🚀 QUICK START - Technical Indicator Alpha Analysis")
    logger.info("="*50)
    
    try:
        # Create directories
        create_directories()
        
        # Load limited data (2020-2023 for faster processing)
        logger.info("📊 Loading data (2020-2023)...")
        pipeline = DataPipeline(start_date='2020-01-01', end_date='2023-12-31')
        data = pipeline.preprocess_data()
        logger.info(f"✅ Data loaded: {len(data)} observations")
        
        # Feature engineering
        logger.info("🔧 Creating technical indicators...")
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        logger.info(f"✅ Features created: {len(feature_engineer.feature_columns)} indicators")
        logger.info(f"📊 Final dataset shape: {data_ready.shape}")
        logger.info(f"📊 Target distribution: {data_ready['target'].value_counts().to_dict()}")
        
        # Quick validation (shorter periods)
        logger.info("🔄 Running walk-forward validation...")
        validator = WalkForwardValidator(initial_train_years=0.5, test_period_months=3, min_train_size=50)
        model = XGBoostModel()
        
        validation_results = validator.validate_model(
            data=data_ready,
            model=model,
            feature_columns=feature_engineer.feature_columns
        )
        
        logger.info(f"✅ Validation completed: {len(validation_results['predictions'])} predictions")
        logger.info(f"📈 Accuracy: {validation_results['overall_metrics']['accuracy']:.3f}")
        logger.info(f"📊 AUC-ROC: {validation_results['overall_metrics']['auc_roc']:.3f}")
        
        # Advanced features
        logger.info("⚙️ Applying advanced features...")
        
        # Threshold optimization
        threshold_optimizer = ThresholdOptimizer()
        optimized_threshold = threshold_optimizer.calculate_optimal_threshold_for_returns(
            returns=data_ready['returns'],
            predictions=pd.Series(validation_results['probabilities'], index=data_ready.index[-len(validation_results['probabilities']):]),
            method='sharpe_max'
        )
        logger.info(f"🎯 Optimized threshold: {optimized_threshold:.3f}")
        
        # Risk management
        risk_manager = RiskManager()
        position_sizes = risk_manager.calculate_position_size(
            returns=data_ready['returns'],
            predictions=pd.Series(validation_results['probabilities'], index=data_ready.index[-len(validation_results['probabilities']):]),
            method='volatility_targeting'
        )
        logger.info("🛡️ Position sizing calculated")
        
        # Performance evaluation
        logger.info("💰 Evaluating performance...")
        evaluator = PerformanceEvaluator()
        
        # Align data with predictions (take only the last 977 rows to match predictions)
        aligned_data = data_ready.iloc[-len(validation_results['predictions']):]
        aligned_position_sizes = position_sizes.iloc[-len(validation_results['predictions']):]
        
        backtest_results = evaluator.backtest_strategy(
            data=aligned_data,
            predictions=validation_results['predictions'],
            probabilities=validation_results['probabilities'],
            position_sizes=aligned_position_sizes
        )
        
        # Print results
        logger.info("📋 RESULTS SUMMARY")
        logger.info("="*50)
        evaluator.print_summary()
        
        # Feature importance
        top_features = model.get_feature_importance(5)
        logger.info("\n🏆 Top 5 Most Important Features:")
        for idx, row in top_features.iterrows():
            logger.info(f"   {row['feature']}: {row['percentage']:.2f}%")
        
        logger.info("\n🎉 Quick start completed successfully!")
        logger.info("💡 Run 'python main.py' for full analysis with 2010-2024 data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick start failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_start()
    sys.exit(0 if success else 1) 