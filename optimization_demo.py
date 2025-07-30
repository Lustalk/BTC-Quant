#!/usr/bin/env python3
"""
BTC Quant - Parameter Optimization Demo

This script demonstrates the intelligent parameter optimization system that:
1. Tests multiple indicator parameters (RSI window, SMA periods, etc.)
2. Optimizes take-profit and stop-loss levels
3. Optimizes ML model hyperparameters
4. Uses Optuna for efficient hyperparameter search
"""

import sys
import time
from datetime import datetime
import pandas as pd

from src.data_pipeline import download_data
from src.parameter_optimization import ParameterOptimizer
from src.visualization import plot_price_and_signals, plot_performance_metrics


def main():
    """Main function for the parameter optimization demo."""
    print("=" * 60)
    print("🚀 BTC Quant - Parameter Optimization Demo")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    try:
        # Step 1: Download data
        print("\n[1/4] 🔧 Data Acquisition")
        print("-" * 40)
        start_time = time.time()
        
        data = download_data("BTC-USD", "2023-01-01", "2024-01-01")
        download_time = time.time() - start_time
        
        print(f"   ✅ Downloaded {len(data)} data points in {download_time:.2f}s")
        print(f"   📊 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Step 2: Initialize optimizer
        print("\n[2/4] 🔧 Parameter Optimization Setup")
        print("-" * 40)
        
        # Use fewer trials for demo (100 instead of 500+ for production)
        optimizer = ParameterOptimizer(data, n_trials=50)
        
        print(f"   ✅ Optimizer initialized with {optimizer.n_trials} trials")
        print(f"   🎯 Optimizing 25+ parameters:")
        print(f"      • Indicator parameters (RSI, SMA, EMA, MACD, etc.)")
        print(f"      • Take-profit and stop-loss levels")
        print(f"      • ML model hyperparameters")
        
        # Step 3: Run optimization
        print("\n[3/4] 🔧 Running Parameter Optimization")
        print("-" * 40)
        print("   ⏳ This may take a few minutes...")
        print("   📊 Testing multiple parameter combinations...")
        
        start_time = time.time()
        optimization_results = optimizer.optimize()
        optimization_time = time.time() - start_time
        
        print(f"   ✅ Optimization completed in {optimization_time:.2f}s")
        print(f"   🎯 Best score: {optimization_results['best_score']:.4f}")
        
        # Step 4: Get optimized results
        print("\n[4/4] 🔧 Analyzing Optimized Results")
        print("-" * 40)
        
        final_results = optimizer.get_optimized_results()
        
        # Display best parameters
        print("\n📊 BEST OPTIMIZED PARAMETERS:")
        print("-" * 40)
        best_params = optimization_results['best_params']
        
        print("🎯 Indicator Parameters:")
        print(f"   • RSI Window: {best_params['rsi_window']}")
        print(f"   • RSI Oversold: {best_params['rsi_oversold']}")
        print(f"   • RSI Overbought: {best_params['rsi_overbought']}")
        print(f"   • SMA Short: {best_params['sma_short']}")
        print(f"   • SMA Long: {best_params['sma_long']}")
        print(f"   • MACD Fast: {best_params['macd_fast']}")
        print(f"   • MACD Slow: {best_params['macd_slow']}")
        
        print("\n💰 Risk Management:")
        print(f"   • Take Profit: {best_params['take_profit']:.1%}")
        print(f"   • Stop Loss: {best_params['stop_loss']:.1%}")
        
        print("\n🤖 ML Model Parameters:")
        print(f"   • Learning Rate: {best_params['learning_rate']:.3f}")
        print(f"   • Max Depth: {best_params['max_depth']}")
        print(f"   • N Estimators: {best_params['n_estimators']}")
        
        # Display performance metrics
        strategy_metrics = final_results['strategy_metrics']
        ml_score = final_results['ml_score']
        
        print("\n📈 OPTIMIZED PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"   📊 Strategy Performance:")
        print(f"      • Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"      • Total Return: {strategy_metrics.get('total_return', 0):.2%}")
        print(f"      • Max Drawdown: {strategy_metrics.get('max_drawdown', 0):.2%}")
        print(f"      • Win Rate: {strategy_metrics.get('win_rate', 0):.2%}")
        print(f"      • Profit Factor: {strategy_metrics.get('profit_factor', 0):.2f}")
        
        print(f"\n   🤖 ML Model Performance:")
        print(f"      • F1 Score: {ml_score:.4f}")
        
        # Generate visualizations
        print("\n📊 Generating Optimized Visualizations...")
        
        signals = final_results['signals']
        prices = final_results['optimized_data']['Close'].tolist()
        
        # Plot optimized strategy
        plot_price_and_signals(
            prices, signals, 
            title="Optimized Strategy with TP/SL",
            save_path="output/optimized_strategy.png"
        )
        
        # Plot performance metrics
        plot_performance_metrics(
            strategy_metrics,
            save_path="output/optimized_performance.png"
        )
        
        print("   ✅ Visualizations saved to output/ directory")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Intelligent parameter search completed")
        print("✅ Take-profit and stop-loss levels optimized")
        print("✅ ML model hyperparameters tuned")
        print("✅ Professional visualizations generated")
        print("✅ Production-ready trading system created")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 