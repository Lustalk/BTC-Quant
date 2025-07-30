#!/usr/bin/env python3
"""
BTC Quant - Main Backtesting Engine

This script implements a comprehensive backtesting engine that:
1. Downloads financial data
2. Engineers technical indicators
3. Uses ML optimization to find best parameters
4. Trains an XGBoost model with optimized parameters
5. Performs walk-forward validation
6. Generates comprehensive visualizations and reports
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
from src.transaction_costs import TransactionCostModel
from src.visualization import plot_model_accuracy, plot_performance_metrics, plot_price_and_signals


def print_optimization_summary(optimization_results):
    """Print a detailed summary of optimization results."""
    print("\n" + "=" * 60)
    print("🏆 PARAMETER OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    best_params = optimization_results['best_params']
    best_score = optimization_results['best_score']
    
    print(f"🎯 Best Optimization Score: {best_score:.4f}")
    print(f"📊 Total Parameters Optimized: {len(best_params)}")
    
    # Group parameters by category
    param_categories = {
        'Technical Indicators': ['sma_short', 'sma_long', 'ema_short', 'ema_long', 'rsi_window', 
                               'rsi_oversold', 'rsi_overbought', 'stoch_window', 'williams_window', 
                               'bb_window', 'atr_window', 'macd_fast', 'macd_slow'],
        'Risk Management': ['take_profit', 'stop_loss', 'risk_per_trade', 'stop_loss_pct'],
        'Position Sizing': ['position_sizing_strategy', 'target_volatility'],
        'ML Model': ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'colsample_bytree'],
        'Feature Engineering': ['lag_1', 'lag_2', 'lag_3', 'roll_short', 'roll_medium', 'roll_long'],
        'Transaction Costs': ['fee_type']
    }
    
    for category, params in param_categories.items():
        category_params = {k: v for k, v in best_params.items() if k in params}
        if category_params:
            print(f"\n📈 {category}:")
            for param, value in category_params.items():
                print(f"   {param}: {value}")
    
    print("=" * 60)


def print_strategy_comparison(basic_metrics, optimized_metrics):
    """Print comparison between basic and optimized strategy performance."""
    print("\n" + "=" * 60)
    print("📊 STRATEGY PERFORMANCE COMPARISON")
    print("=" * 60)
    
    metrics_to_compare = [
        ('total_return', 'Total Return'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('sortino_ratio', 'Sortino Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('win_rate', 'Win Rate'),
        ('profit_factor', 'Profit Factor')
    ]
    
    print(f"{'Metric':<20} {'Basic':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for metric_key, metric_name in metrics_to_compare:
        basic_val = basic_metrics.get(metric_key, 0)
        optimized_val = optimized_metrics.get(metric_key, 0)
        
        if basic_val != 0:
            improvement = ((optimized_val - basic_val) / basic_val) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric_name:<20} {basic_val:<15.4f} {optimized_val:<15.4f} {improvement_str:<15}")
    
    print("=" * 60)


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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare basic vs optimized strategy performance"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 BTC Quant Backtesting Engine")
    print(f"📊 Ticker: {args.ticker}")
    print(f"📅 Date Range: {args.start_date} to {args.end_date}")
    print(f"🔧 Validation Splits: {args.n_splits}")
    if args.optimize:
        print(f"🤖 ML Optimization: Enabled ({args.n_trials} trials)")
    if args.compare:
        print(f"📈 Performance Comparison: Enabled")
    print("-" * 50)
    
    try:
        # Step 1: Download data
        print("Step 1: Downloading financial data...")
        data = download_data(args.ticker, args.start_date, args.end_date)
        print(f"✅ Downloaded {len(data)} data points")
        
        basic_metrics = None
        optimized_metrics = None
        
        if args.compare:
            # Run basic pipeline for comparison
            print("\nStep 2a: Running basic pipeline...")
            data_with_features_basic = add_technical_indicators(data)
            X_basic, y_basic = prepare_features_target(data_with_features_basic)
            scores_basic = walk_forward_validation(X_basic, y_basic, n_splits=args.n_splits)
            
            # Calculate basic strategy performance
            prices_basic = data_with_features_basic['Close'].tolist()
            signals_basic = []
            for _, row in data_with_features_basic.iterrows():
                rsi = row.get('RSI_14', 50)
                if rsi < 30:
                    signals_basic.append(1)  # Buy
                elif rsi > 70:
                    signals_basic.append(-1)  # Sell
                else:
                    signals_basic.append(0)  # Hold
            
            if len(signals_basic) == len(prices_basic):
                volumes_basic = data_with_features_basic['Volume'].tolist()
                atr_values_basic = data_with_features_basic['ATR'].tolist() if 'ATR' in data_with_features_basic.columns else [0.01] * len(prices_basic)
                
                basic_metrics = analyze_strategy_performance(
                    prices_basic, signals_basic, 
                    volumes=volumes_basic, 
                    atr_values=atr_values_basic,
                    include_transaction_costs=True,
                    position_sizing_strategy="volatility_targeted"
                )
        
        # Step 2: ML Optimization (if enabled) or Feature Engineering
        if args.optimize:
            print("Step 2: Running ML optimization...")
            optimizer = ParameterOptimizer(data, n_trials=args.n_trials)
            optimization_results = optimizer.optimize()
            
            # Print detailed optimization summary
            print_optimization_summary(optimization_results)
            
            # Get optimized data with best parameters
            optimized_data = optimizer.create_indicators_with_params(optimization_results['best_params'])
            print(f"✅ Optimized dataset shape: {optimized_data.shape}")
            
            # Use optimized data for the rest of the pipeline
            data_with_features = optimized_data
        else:
            # Step 2: Engineer features (default)
            print("Step 2: Engineering technical indicators...")
            data_with_features = add_technical_indicators(data)
            print(f"✅ Added technical indicators. Dataset shape: {data_with_features.shape}")
        
        # Step 3: Prepare features and target
        print("Step 3: Preparing features and target...")
        X, y = prepare_features_target(data_with_features)
        print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")
        
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
        
        print(f"🤖 Model Performance:")
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
                print("\n🚀 Optimized Strategy Performance:")
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
                print("\n📊 Strategy Performance:")
            
            if len(signals) == len(prices):
                # Get volume and ATR data for transaction cost analysis
                volumes = data_with_features['Volume'].tolist()
                atr_values = data_with_features['ATR'].tolist() if 'ATR' in data_with_features.columns else [0.01] * len(prices)
                
                # Analyze strategy with transaction costs and position sizing
                strategy_metrics = analyze_strategy_performance(
                    prices, signals, 
                    volumes=volumes, 
                    atr_values=atr_values,
                    include_transaction_costs=True,
                    position_sizing_strategy="volatility_targeted"
                )
                trade_stats = calculate_trade_statistics(signals)
                
                optimized_metrics = strategy_metrics
                print_performance_table(strategy_metrics)
                
                # Print transaction cost analysis
                if 'total_costs' in strategy_metrics:
                    print(f"\n💰 Transaction Cost Analysis:")
                    print(f"  Total Fees: ${strategy_metrics['total_fees']:.2f}")
                    print(f"  Total Slippage: ${strategy_metrics['total_slippage']:.2f}")
                    print(f"  Total Costs: ${strategy_metrics['total_costs']:.2f}")
                    print(f"  Cost Impact: {strategy_metrics['cost_impact']:.4f}")
                    print(f"  Return Degradation: {strategy_metrics['degradation_percentage']:.2f}%")
                
                print(f"\n📈 Trading Statistics:")
                print(f"  Total Trades: {trade_stats['total_trades']}")
                print(f"  Avg Trade Duration: {trade_stats['avg_trade_duration']:.1f} days")
                
                # Compare basic vs optimized if requested
                if args.compare and basic_metrics:
                    print_strategy_comparison(basic_metrics, optimized_metrics)
                
                # Generate visualizations
                if args.optimize:
                    # Get optimized results for visualization
                    results = optimizer.get_optimized_results()
                    fold_scores = results['fold_scores']
                    
                    # Plot model accuracy with optimized parameters
                    plot_model_accuracy(
                        fold_scores,
                        "ML Model Accuracy - Optimized Parameters",
                        "output/model_accuracy_detailed.png"
                    )
                    print("✅ Model accuracy visualization saved")
                    
                    # Plot performance metrics
                    plot_performance_metrics(
                        strategy_metrics,
                        "output/performance_metrics_detailed.png"
                    )
                    print("✅ Performance metrics visualization saved")
                    
                    # Plot price and signals
                    plot_price_and_signals(
                        prices, signals,
                        "Optimized Trading Strategy",
                        "output/optimized_strategy.png"
                    )
                    print("✅ Strategy visualization saved")
                else:
                    # Plot model accuracy with baseline parameters
                    plot_model_accuracy(
                        scores,
                        "ML Model Accuracy - Baseline Parameters",
                        "output/model_accuracy_detailed.png"
                    )
                    print("✅ Model accuracy visualization saved")
        
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 