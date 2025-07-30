#!/usr/bin/env python3
"""
Simplified BTC Quant Project - Core Analysis
Focus on essential functionality without over-engineering
"""

import sys
import os
import logging
import argparse
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import create_directories
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.modeling import XGBoostModel
from src.evaluation import PerformanceEvaluator
from src.strategy_analysis import StrategyAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_core_analysis():
    """
    Run core analysis with essential functionality only
    """
    logger.info("=" * 80)
    logger.info("CORE BTC QUANT ANALYSIS - ESSENTIAL FUNCTIONALITY")
    logger.info("=" * 80)
    logger.info(f"Analysis started at: {datetime.now()}")

    try:
        # Step 1: Setup
        logger.info("Step 1: Creating project directories...")
        create_directories()
        os.makedirs("exports", exist_ok=True)
        os.makedirs("exports/visualizations", exist_ok=True)

        # Step 2: Data Pipeline
        logger.info("Step 2: Data acquisition and preprocessing...")
        pipeline = DataPipeline()
        data = pipeline.preprocess_data()
        logger.info(
            f"Data loaded: {len(data)} observations from {data.index[0].date()} to {data.index[-1].date()}"
        )

        # Step 3: Feature Engineering
        logger.info("Step 3: Feature engineering...")
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_all_features(data)
        data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)
        logger.info(
            f"Features created: {len(feature_engineer.feature_columns)} technical indicators"
        )

        # Step 4: Model Training with Default Parameters
        logger.info("Step 4: Training XGBoost model with default parameters...")
        model = XGBoostModel()
        
        # Use default XGBoost parameters - no hyperparameter optimization
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        model.set_hyperparameters(default_params)
        
        # Train on all features - no feature selection
        X = data_ready[feature_engineer.feature_columns]
        y = data_ready["target"]
        model.fit(X, y)

        # Step 5: Strategy Analysis
        logger.info("Step 5: Running strategy analysis...")
        strategy_analyzer = StrategyAnalyzer()
        results = strategy_analyzer.analyze_strategy(
            data=data_ready,
            model=model,
            feature_columns=feature_engineer.feature_columns,
            target_column="target"
        )

        # Step 6: Simple Validation
        logger.info("Step 6: Running validation analysis...")
        validation_results = run_validation_analysis(data_ready, model, feature_engineer.feature_columns)

        # Step 7: Export Results
        logger.info("Step 7: Exporting results...")
        export_analysis_results(results, validation_results)

        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


def run_validation_analysis(data, model, feature_columns):
    """
    Simple validation analysis without over-engineering
    """
    logger.info("Running validation analysis...")
    
    # Get strategy returns
    strategy_returns = get_strategy_returns(data, model, feature_columns)
    buy_hold_returns = data['returns'].fillna(0)
    
    # Simple bootstrap analysis for Sharpe ratio
    validation_results = bootstrap_sharpe_analysis(strategy_returns, buy_hold_returns)
    
    # Transaction cost analysis
    cost_analysis = analyze_transaction_costs(strategy_returns, buy_hold_returns)
    validation_results.update(cost_analysis)
    
    print_validation_results(validation_results)
    return validation_results


def get_strategy_returns(data, model, feature_columns):
    """
    Calculate strategy returns using model predictions
    """
    X = data[feature_columns]
    predictions = model.predict(X)
    
    # Simple strategy: buy when prediction > 0.5
    signals = pd.Series((predictions > 0.5).astype(int), index=data.index)
    strategy_returns = signals.shift(1) * data['returns'].fillna(0)
    
    return strategy_returns.fillna(0)


def bootstrap_sharpe_analysis(strategy_returns, buy_hold_returns, n_bootstrap=1000):
    """
    Simple bootstrap analysis for Sharpe ratio comparison
    """
    def calculate_sharpe(returns):
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    # Calculate actual Sharpe ratios
    strategy_sharpe = calculate_sharpe(strategy_returns)
    buy_hold_sharpe = calculate_sharpe(buy_hold_returns)
    
    # Bootstrap to get confidence intervals
    strategy_bootstrap = []
    buy_hold_bootstrap = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        strategy_sample = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
        buy_hold_sample = np.random.choice(buy_hold_returns, size=len(buy_hold_returns), replace=True)
        
        strategy_bootstrap.append(calculate_sharpe(strategy_sample))
        buy_hold_bootstrap.append(calculate_sharpe(buy_hold_sample))
    
    strategy_bootstrap = np.array(strategy_bootstrap)
    buy_hold_bootstrap = np.array(buy_hold_bootstrap)
    
    return {
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe_ci': (np.percentile(strategy_bootstrap, 2.5), np.percentile(strategy_bootstrap, 97.5)),
        'buy_hold_sharpe_ci': (np.percentile(buy_hold_bootstrap, 2.5), np.percentile(buy_hold_bootstrap, 97.5)),
        'sharpe_improvement': strategy_sharpe - buy_hold_sharpe
    }


def analyze_transaction_costs(strategy_returns, buy_hold_returns, costs=[0.0001, 0.0005, 0.001]):
    """
    Analyze impact of transaction costs
    """
    results = {}
    
    for cost in costs:
        # Simple transaction cost model
        # Assume cost applies to each trade (signal change)
        signals = (strategy_returns > 0).astype(int)
        trades = signals.diff().abs().fillna(0)
        total_cost = trades.sum() * cost
        
        net_strategy_returns = strategy_returns - total_cost / len(strategy_returns)
        
        results[f'transaction_cost_{cost}'] = {
            'total_trades': trades.sum(),
            'total_cost': total_cost,
            'net_sharpe': net_strategy_returns.mean() / net_strategy_returns.std() * np.sqrt(252) if net_strategy_returns.std() > 0 else 0
        }
    
    return results


def print_validation_results(validation_results):
    """
    Print validation results in a clear format
    """
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    print(f"Strategy Sharpe Ratio: {validation_results['strategy_sharpe']:.4f}")
    print(f"Buy & Hold Sharpe Ratio: {validation_results['buy_hold_sharpe']:.4f}")
    print(f"Improvement: {validation_results['sharpe_improvement']:.4f}")
    
    print(f"\nStrategy Sharpe 95% CI: {validation_results['strategy_sharpe_ci']}")
    print(f"Buy & Hold Sharpe 95% CI: {validation_results['buy_hold_sharpe_ci']}")
    
    print("\nTransaction Cost Analysis:")
    for cost_key, cost_data in validation_results.items():
        if cost_key.startswith('transaction_cost_'):
            cost_rate = cost_key.split('_')[-1]
            print(f"  Cost {cost_rate}: {cost_data['total_trades']} trades, "
                  f"Net Sharpe: {cost_data['net_sharpe']:.4f}")


def export_analysis_results(results, validation_results):
    """
    Export analysis results to files
    """
    # Export validation results
    validation_file = "results/validation_results.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(validation_file, 'w') as f:
        f.write("BTC Quant Strategy Validation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Strategy Sharpe Ratio: {validation_results['strategy_sharpe']:.4f}\n")
        f.write(f"Buy & Hold Sharpe Ratio: {validation_results['buy_hold_sharpe']:.4f}\n")
        f.write(f"Improvement: {validation_results['sharpe_improvement']:.4f}\n")
        
        f.write("\nTransaction Cost Analysis:\n")
        for cost_key, cost_data in validation_results.items():
            if cost_key.startswith('transaction_cost_'):
                cost_rate = cost_key.split('_')[-1]
                f.write(f"  Cost {cost_rate}: {cost_data['total_trades']} trades, "
                       f"Net Sharpe: {cost_data['net_sharpe']:.4f}\n")
    
    logger.info(f"Results exported to {validation_file}")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="BTC Quant Strategy Analysis")
    parser.add_argument("--test", action="store_true", help="Run quick test analysis")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running test analysis...")
        # For test mode, we could add some shortcuts
        pass
    
    run_core_analysis()


if __name__ == "__main__":
    main()
