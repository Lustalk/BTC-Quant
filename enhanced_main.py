#!/usr/bin/env python3
"""
Enhanced Main Script for BTC Quant Project
Integrates all professional deliverables with advanced features
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
from src.hyperparameter_optimization import HyperparameterOptimizer
from src.feature_selection import FeatureSelector
from src.enhanced_validation import EnhancedWalkForwardValidator
from src.monte_carlo_simulation import MonteCarloSimulator
from src.professional_visualizations import ProfessionalVisualizer
from src.modeling import XGBoostModel
from src.evaluation import PerformanceEvaluator
from src.threshold_optimization import ThresholdOptimizer
from src.risk_management import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_enhanced_analysis():
    """
    Run enhanced analysis with all professional deliverables
    """
    logger.info("="*80)
    logger.info("ENHANCED BTC QUANT ANALYSIS - PROFESSIONAL DELIVERABLES")
    logger.info("="*80)
    logger.info(f"Analysis started at: {datetime.now()}")
    
    try:
        # Step 1: Create necessary directories
        logger.info("Step 1: Creating project directories...")
        create_directories()
        os.makedirs('exports', exist_ok=True)
        os.makedirs('exports/visualizations', exist_ok=True)
        
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
        
        # Step 4: Advanced Hyperparameter Optimization
        logger.info("Step 4: Advanced hyperparameter optimization...")
        hyperopt = HyperparameterOptimizer(n_trials=50)  # Reduced for demo
        
        # Prepare data for optimization
        X = data_ready[feature_engineer.feature_columns]
        y = data_ready['target']
        
        # Run Bayesian optimization
        best_params, study = hyperopt.optimize_hyperparameters(X, y, data_ready.index)
        
        # Run grid search validation
        grid_params, grid_results = hyperopt.grid_search_validation(X, y, data_ready.index)
        
        # Export hyperparameter results
        hyperopt.export_results(study, grid_results)
        
        # Step 5: Feature Selection
        logger.info("Step 5: Feature selection with RFE...")
        feature_selector = FeatureSelector()
        
        # Calculate feature importance
        feature_importance = feature_selector.calculate_feature_importance(X, y, data_ready.index)
        
        # Perform RFE
        rfe_results = feature_selector.recursive_feature_elimination(X, y, data_ready.index)
        
        # Get optimal feature set
        optimal_features, combined_df = feature_selector.get_optimal_feature_set(X, y, data_ready.index)
        
        # Export feature importance
        feature_selector.export_feature_importance()
        
        # Step 6: Enhanced Walk-Forward Validation
        logger.info("Step 6: Enhanced walk-forward validation...")
        enhanced_validator = EnhancedWalkForwardValidator(
            initial_train_years=2,
            test_period_months=3,
            min_train_size=500
        )
        
        # Use optimal features and hyperparameters
        model = XGBoostModel()
        model.set_hyperparameters(best_params)
        
        validation_results = enhanced_validator.validate_model(
            data=data_ready,
            model=model,
            feature_columns=optimal_features
        )
        
        logger.info(f"Enhanced validation completed: {len(validation_results['period_metrics'])} periods")
        
        # Step 7: Monte Carlo Simulation
        logger.info("Step 7: Monte Carlo simulation...")
        monte_carlo = MonteCarloSimulator(n_simulations=500)  # Reduced for demo
        
        # Run Monte Carlo simulation
        simulation_results = monte_carlo.simulate_strategy_performance(
            data_ready['returns'],
            validation_results['daily_predictions']['probability']
        )
        
        # Calculate rolling metrics
        rolling_metrics = monte_carlo.calculate_rolling_metrics(
            data_ready['returns'],
            validation_results['daily_predictions']['probability']
        )
        
        # Perform regime analysis
        regime_data, regime_stats = monte_carlo.regime_analysis(
            data_ready['returns'],
            validation_results['daily_predictions']['probability']
        )
        
        # Export Monte Carlo results
        monte_carlo.export_simulation_results()
        
        # Step 8: Professional Visualizations
        logger.info("Step 8: Creating professional visualizations...")
        visualizer = ProfessionalVisualizer()
        
        # Create all visualizations
        viz_dir = visualizer.export_all_visualizations(
            data=data_ready,
            predictions=validation_results['daily_predictions']['predicted'],
            probabilities=validation_results['daily_predictions']['probability'],
            feature_importance=feature_importance,
            regime_data=regime_data
        )
        
        # Step 9: Dynamic Threshold Optimization
        logger.info("Step 9: Dynamic threshold optimization...")
        threshold_optimizer = ThresholdOptimizer()
        
        # Optimize threshold for each period
        optimized_thresholds = []
        for period_metrics in validation_results['period_metrics']:
            period_data = data_ready[
                (data_ready.index >= period_metrics['test_start']) & 
                (data_ready.index <= period_metrics['test_end'])
            ]
            
            optimal_threshold = threshold_optimizer.calculate_optimal_threshold_for_returns(
                returns=period_data['returns'],
                predictions=validation_results['daily_predictions'][
                    validation_results['daily_predictions']['period'] == period_metrics['period']
                ]['probability'],
                method='sharpe_max'
            )
            optimized_thresholds.append(optimal_threshold)
        
        # Step 10: Risk Management
        logger.info("Step 10: Risk management analysis...")
        risk_manager = RiskManager()
        
        # Calculate position sizes
        position_sizes = risk_manager.calculate_position_size(
            returns=data_ready['returns'],
            predictions=validation_results['daily_predictions']['probability'],
            method='volatility_targeting'
        )
        
        # Step 11: Performance Evaluation
        logger.info("Step 11: Performance evaluation...")
        evaluator = PerformanceEvaluator()
        
        backtest_results = evaluator.backtest_strategy(
            data=data_ready,
            predictions=validation_results['daily_predictions']['predicted'],
            probabilities=validation_results['daily_predictions']['probability'],
            position_sizes=position_sizes
        )
        
        # Step 12: Export All Results
        logger.info("Step 12: Exporting all results...")
        
        # Export trade analysis
        trade_analysis = pd.DataFrame({
            'date': data_ready.index,
            'signal': validation_results['daily_predictions']['predicted'],
            'probability': validation_results['daily_predictions']['probability'],
            'position_size': position_sizes,
            'returns': data_ready['returns'],
            'strategy_returns': backtest_results['strategy_returns']
        })
        trade_analysis.to_csv('exports/trade_analysis.csv', index=False)
        
        # Export regime analysis
        if regime_data is not None:
            regime_data.to_csv('exports/regime_analysis.csv', index=False)
            regime_stats.to_csv('exports/regime_stats.csv', index=False)
        
        # Export cumulative returns
        cumulative_returns = pd.DataFrame({
            'date': data_ready.index,
            'strategy_cumulative': (1 + backtest_results['strategy_returns']).cumprod(),
            'buy_hold_cumulative': (1 + data_ready['returns']).cumprod(),
            'excess_cumulative': (1 + backtest_results['strategy_returns']).cumprod() / (1 + data_ready['returns']).cumprod()
        })
        cumulative_returns.to_csv('exports/cumulative_returns.csv', index=False)
        
        # Step 13: Print Summary
        logger.info("Step 13: Analysis summary...")
        
        # Print overall metrics
        overall_metrics = validation_results['overall_metrics']
        logger.info("="*60)
        logger.info("OVERALL PERFORMANCE METRICS")
        logger.info("="*60)
        logger.info(f"Total Periods: {overall_metrics['total_periods']}")
        logger.info(f"Average AUC-ROC: {overall_metrics['auc_roc']:.3f}")
        logger.info(f"Average Sharpe Ratio: {overall_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Average Strategy Return: {overall_metrics['strategy_return']:.3f}")
        logger.info(f"Average Excess Return: {overall_metrics['excess_return']:.3f}")
        logger.info(f"Average Win Rate: {overall_metrics['win_rate']:.3f}")
        
        # Print feature importance
        logger.info("="*60)
        logger.info("TOP 10 FEATURE IMPORTANCE")
        logger.info("="*60)
        top_features = feature_importance.head(10)
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Print Monte Carlo summary
        if simulation_results:
            sim_df = pd.DataFrame(simulation_results)
            logger.info("="*60)
            logger.info("MONTE CARLO SIMULATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Simulations: {len(sim_df)}")
            logger.info(f"Mean Strategy Return: {sim_df['total_return'].mean():.3f}")
            logger.info(f"Mean Sharpe Ratio: {sim_df['sharpe_ratio'].mean():.3f}")
            logger.info(f"Positive Excess Returns: {(sim_df['excess_return'] > 0).mean():.1%}")
        
        # Print export summary
        logger.info("="*60)
        logger.info("EXPORT SUMMARY")
        logger.info("="*60)
        logger.info("CSV Exports:")
        logger.info("  - exports/hyperparameter_results.csv")
        logger.info("  - exports/feature_importance.csv")
        logger.info("  - exports/performance_metrics.csv")
        logger.info("  - exports/daily_predictions.csv")
        logger.info("  - exports/rolling_metrics.csv")
        logger.info("  - exports/cumulative_returns.csv")
        logger.info("  - exports/trade_analysis.csv")
        logger.info("  - exports/regime_analysis.csv")
        logger.info("  - exports/monte_carlo_results.csv")
        logger.info("HTML Visualizations:")
        logger.info(f"  - {viz_dir}/comprehensive_dashboard.html")
        logger.info(f"  - {viz_dir}/cumulative_returns.html")
        logger.info(f"  - {viz_dir}/feature_importance.html")
        logger.info(f"  - {viz_dir}/risk_return_scatter.html")
        
        logger.info("="*80)
        logger.info("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Analysis completed at: {datetime.now()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed with error: {e}")
        logger.error("Please check the logs for details")
        return False

def run_quick_enhanced_test():
    """
    Run a quick enhanced test with limited data for development
    """
    logger.info("Running quick enhanced test with limited data...")
    
    # Use shorter time period for testing
    pipeline = DataPipeline(start_date='2020-01-01', end_date='2023-12-31')
    data = pipeline.preprocess_data()
    
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
    
    # Quick hyperparameter optimization
    hyperopt = HyperparameterOptimizer(n_trials=10)
    X = data_ready[feature_engineer.feature_columns]
    y = data_ready['target']
    best_params, study = hyperopt.optimize_hyperparameters(X, y, data_ready.index)
    
    # Quick feature selection
    feature_selector = FeatureSelector()
    feature_importance = feature_selector.calculate_feature_importance(X, y, data_ready.index)
    optimal_features, _ = feature_selector.get_optimal_feature_set(X, y, data_ready.index, max_features=10)
    
    # Quick validation
    enhanced_validator = EnhancedWalkForwardValidator(
        initial_train_years=1,
        test_period_months=6,
        min_train_size=200
    )
    
    model = XGBoostModel()
    model.set_hyperparameters(best_params)
    
    validation_results = enhanced_validator.validate_model(
        data=data_ready,
        model=model,
        feature_columns=optimal_features
    )
    
    # Quick Monte Carlo
    monte_carlo = MonteCarloSimulator(n_simulations=100)
    simulation_results = monte_carlo.simulate_strategy_performance(
        data_ready['returns'],
        validation_results['daily_predictions']['probability']
    )
    
    # Quick visualizations
    visualizer = ProfessionalVisualizer()
    visualizer.create_cumulative_returns_plot(
        data_ready,
        validation_results['daily_predictions']['predicted'],
        validation_results['daily_predictions']['probability'],
        save_path='exports/quick_test_cumulative_returns.html'
    )
    
    logger.info("Quick enhanced test completed successfully!")
    logger.info(f"Validation periods: {len(validation_results['period_metrics'])}")
    logger.info(f"Monte Carlo simulations: {len(simulation_results)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced BTC Quant Analysis')
    parser.add_argument('--test', action='store_true', 
                       help='Run quick enhanced test with limited data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        run_quick_enhanced_test()
    else:
        success = run_enhanced_analysis()
        sys.exit(0 if success else 1) 