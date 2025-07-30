#!/usr/bin/env python3
"""
Main Script for Technical Indicator Alpha Project
Orchestrates the complete analysis pipeline from data acquisition to performance evaluation
"""

import sys
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import create_directories, PATHS_CONFIG
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.validation import WalkForwardValidator
from src.modeling import XGBoostModel
from src.evaluation import PerformanceEvaluator
from src.threshold_optimization import ThresholdOptimizer
from src.risk_management import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/technical_alpha.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for the technical indicator alpha analysis
    """
    logger.info("="*60)
    logger.info("TECHNICAL INDICATOR ALPHA ANALYSIS")
    logger.info("="*60)
    logger.info(f"Analysis started at: {datetime.now()}")
    
    try:
        # Step 1: Create necessary directories
        logger.info("Step 1: Creating project directories...")
        create_directories()
        
        # Step 2: Data Pipeline
        logger.info("Step 2: Data acquisition and preprocessing...")
        pipeline = DataPipeline()
        data = pipeline.preprocess_data()
        logger.info(f"Data loaded: {len(data)} observations from {data.index[0].date()} to {data.index[-1].date()}")
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Feature engineering...")
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        logger.info(f"Features created: {len(feature_engineer.feature_columns)} technical indicators")
        
        # Step 4: Walk-Forward Validation
        logger.info("Step 4: Walk-forward validation...")
        validator = WalkForwardValidator()
        model = XGBoostModel()
        
        validation_results = validator.validate_model(
            data=data_ready,
            model=model,
            feature_columns=feature_engineer.feature_columns
        )
        
        logger.info(f"Validation completed: {len(validation_results['predictions'])} predictions")
        logger.info(f"Overall accuracy: {validation_results['overall_metrics']['accuracy']:.3f}")
        logger.info(f"Overall AUC-ROC: {validation_results['overall_metrics']['auc_roc']:.3f}")
        
        # Step 5: Advanced Features (Threshold Optimization & Risk Management)
        logger.info("Step 5: Advanced features - Threshold optimization and risk management...")
        
        # Threshold optimization
        threshold_optimizer = ThresholdOptimizer()
        optimized_threshold = threshold_optimizer.calculate_optimal_threshold_for_returns(
            returns=data_ready['returns'],
            predictions=validation_results['probabilities'],
            method='sharpe_max'
        )
        logger.info(f"Optimized threshold: {optimized_threshold:.3f}")
        
        # Risk management
        risk_manager = RiskManager()
        position_sizes = risk_manager.calculate_position_size(
            returns=data_ready['returns'],
            predictions=validation_results['probabilities'],
            method='volatility_targeting'
        )
        logger.info(f"Position sizing calculated using volatility targeting")
        
        # Step 6: Performance Evaluation
        logger.info("Step 6: Performance evaluation and backtesting...")
        evaluator = PerformanceEvaluator()
        backtest_results = evaluator.backtest_strategy(
            data=data_ready,
            predictions=validation_results['predictions'],
            probabilities=validation_results['probabilities'],
            position_sizes=position_sizes  # Include position sizing
        )
        
        # Step 6: Generate Visualizations and Save Results
        logger.info("Step 6: Generating visualizations and saving results...")
        
        # Save model
        model_path = model.save_model()
        logger.info(f"Model saved to: {model_path}")
        
        # Save results
        results_path = evaluator.save_results()
        logger.info(f"Results saved to: {results_path}")
        
        # Generate plots
        evaluator.plot_cumulative_returns(
            save_path=os.path.join(PATHS_CONFIG['results_dir'], 'cumulative_returns.png')
        )
        
        evaluator.plot_drawdown(
            save_path=os.path.join(PATHS_CONFIG['results_dir'], 'drawdown_analysis.png')
        )
        
        # Feature importance plot
        model.plot_feature_importance(
            save_path=os.path.join(PATHS_CONFIG['results_dir'], 'feature_importance.png')
        )
        
        # Step 7: Print Summary
        logger.info("Step 7: Analysis summary...")
        evaluator.print_summary()
        
        # Print feature importance
        top_features = model.get_feature_importance(10)
        logger.info("Top 10 Feature Importance:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['percentage']:.2f}%")
        
        # Print validation summary
        period_summary = validator.get_period_summary()
        logger.info(f"Validation periods: {len(period_summary)}")
        logger.info(f"Average training size: {period_summary['train_size'].mean():.0f}")
        logger.info(f"Average test size: {period_summary['test_size'].mean():.0f}")
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Analysis completed at: {datetime.now()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        logger.error("Please check the logs for details")
        return False

def run_quick_test():
    """
    Run a quick test with smaller dataset for development
    """
    logger.info("Running quick test with limited data...")
    
    # Use shorter time period for testing
    pipeline = DataPipeline(start_date='2020-01-01', end_date='2023-12-31')
    data = pipeline.preprocess_data()
    
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
    
    # Quick validation with fewer periods
    validator = WalkForwardValidator(initial_train_years=1, test_period_months=6)
    model = XGBoostModel()
    
    validation_results = validator.validate_model(
        data=data_ready,
        model=model,
        feature_columns=feature_engineer.feature_columns
    )
    
    evaluator = PerformanceEvaluator()
    backtest_results = evaluator.backtest_strategy(
        data=data_ready,
        predictions=validation_results['predictions'],
        probabilities=validation_results['probabilities']
    )
    
    evaluator.print_summary()
    logger.info("Quick test completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Technical Indicator Alpha Analysis')
    parser.add_argument('--test', action='store_true', 
                       help='Run quick test with limited data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        run_quick_test()
    else:
        success = main()
        sys.exit(0 if success else 1) 