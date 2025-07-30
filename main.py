#!/usr/bin/env python3
"""
BTC Quant - Main Backtesting Engine

This script implements a comprehensive backtesting engine that:
1. Downloads financial data
2. Engineers technical indicators
3. Uses ML optimization to find best parameters
4. Trains an XGBoost model with optimized parameters
5. Performs walk-forward validation
"""

import argparse
import sys
from datetime import datetime
import pandas as pd

from src.data_pipeline import download_data
from src.feature_engineering import add_technical_indicators
from src.model import prepare_features_target, walk_forward_validation
from src.evaluation import calculate_all_metrics
from src.strategy_analysis import analyze_strategy_performance, calculate_trade_statistics, print_performance_table
from src.parameter_optimization import ParameterOptimizer


def main():
    """Main function for the BTC Quant backtesting engine."""
    parser = argparse.ArgumentParser(
        description="BTC Quant - Quantitative Trading Backtesting Engine"
    )
    parser.add_argument(
        "--ticker", 
        type=str, 
        default="BTC-USD",
        help="Ticker symbol (default: BTC-USD)"
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format (default: 2023-01-01)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default="2024-01-01",
        help="End date in YYYY-MM-DD format (default: 2024-01-01)"
    )
    parser.add_argument(
        "--n-splits", 
        type=int, 
        default=5,
        help="Number of splits for walk-forward validation (default: 5)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable ML optimization to find best parameters"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    
    args = parser.parse_args()
    
    print(f"BTC Quant Backtesting Engine")
    print(f"Ticker: {args.ticker}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Validation Splits: {args.n_splits}")
    if args.optimize:
        print(f"ML Optimization: Enabled ({args.n_trials} trials)")
    print("-" * 50)
    
    try:
        # Step 1: Download data
        print("Step 1: Downloading financial data...")
        data = download_data(args.ticker, args.start_date, args.end_date)
        print(f"Downloaded {len(data)} data points")
        
        # Step 2: ML Optimization (if enabled)
        if args.optimize:
            print("Step 2: Running ML optimization...")
            optimizer = ParameterOptimizer(data, n_trials=args.n_trials)
            optimization_results = optimizer.optimize()
            print(f"Optimization completed. Best score: {optimization_results['best_score']:.4f}")
            
            # Get optimized data with best parameters
            optimized_data = optimizer.create_indicators_with_params(optimization_results['best_params'])
            print(f"Optimized dataset shape: {optimized_data.shape}")
            
            # Use optimized data for the rest of the pipeline
            data_with_features = optimized_data
        else:
            # Step 2: Engineer features (default)
            print("Step 2: Engineering technical indicators...")
            data_with_features = add_technical_indicators(data)
            print(f"Added technical indicators. Dataset shape: {data_with_features.shape}")
        
        # Step 3: Prepare features and target
        print("Step 3: Preparing features and target...")
        X, y = prepare_features_target(data_with_features)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Step 4: Perform walk-forward validation
        print("Step 4: Running walk-forward validation...")
        scores = walk_forward_validation(X, y, n_splits=args.n_splits)
        
        # Step 5: Calculate performance metrics
        print("\n" + "=" * 50)
        print("PERFORMANCE EVALUATION")
        print("=" * 50)
        
        # Calculate model performance metrics
        model_metrics = {
            'average_accuracy': sum(scores) / len(scores),
            'best_fold_accuracy': max(scores),
            'worst_fold_accuracy': min(scores),
            'fold_accuracies': scores
        }
        
        print(f"Model Performance:")
        print(f"  Average Accuracy: {model_metrics['average_accuracy']:.4f}")
        print(f"  Best Fold: {model_metrics['best_fold_accuracy']:.4f}")
        print(f"  Worst Fold: {model_metrics['worst_fold_accuracy']:.4f}")
        print(f"  Individual Folds: {[f'{s:.4f}' for s in scores]}")
        
        # Calculate strategy performance
        if len(data_with_features) > 0:
            prices = data_with_features['Close'].tolist()
            
            if args.optimize:
                # Use optimized signals from the optimizer
                signals, _ = optimizer.generate_signals_with_tp_sl(data_with_features, optimization_results['best_params'])
                print("\nOptimized Strategy Performance:")
            else:
                # Simple strategy: buy when RSI < 30, sell when RSI > 70
                signals = []
                for _, row in data_with_features.iterrows():
                    rsi = row.get('RSI_14', 50)
                    if rsi < 30:
                        signals.append(1)  # Buy
                    elif rsi > 70:
                        signals.append(-1)  # Sell
                    else:
                        signals.append(0)  # Hold
                print("\nStrategy Performance:")
            
            if len(signals) == len(prices):
                strategy_metrics = analyze_strategy_performance(prices, signals)
                trade_stats = calculate_trade_statistics(signals)
                
                print_performance_table(strategy_metrics)
                
                print(f"\nTrading Statistics:")
                print(f"  Total Trades: {trade_stats['total_trades']}")
                print(f"  Avg Trade Duration: {trade_stats['avg_trade_duration']:.1f} days")
        
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 