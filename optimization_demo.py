#!/usr/bin/env python3
"""
BTC Quant - ML Optimization Demo

This script demonstrates the complete ML optimization pipeline:
1. Downloads financial data
2. Uses Optuna to optimize technical indicator parameters
3. Optimizes take-profit and stop-loss levels
4. Optimizes ML model hyperparameters
5. Evaluates the best strategy performance
"""

import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

from src.data_pipeline import download_data
from src.parameter_optimization import ParameterOptimizer
from src.visualization import plot_price_and_signals, plot_performance_metrics


def run_optimization_demo():
    """Run the complete ML optimization demo."""
    print("=" * 60)
    print("🚀 BTC Quant - ML Optimization Demo")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Download data
        print("\n[1/5] 🔧 Data Acquisition")
        print("-" * 40)
        data_start = time.time()
        data = download_data("BTC-USD", "2023-01-01", "2024-01-01")
        data_time = time.time() - data_start
        print(f"   ✅ Downloaded {len(data)} data points in {data_time:.2f}s")
        print(f"   📊 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Step 2: Initialize optimizer
        print("\n[2/5] 🔧 Optimizer Setup")
        print("-" * 40)
        optimizer = ParameterOptimizer(data, n_trials=50)  # Reduced for demo
        print(f"   ✅ Optimizer initialized with {optimizer.n_trials} trials")
        print(f"   🎯 Target: Optimize technical indicators + ML parameters")
        
        # Step 3: Run optimization
        print("\n[3/5] 🔧 ML Optimization")
        print("-" * 40)
        opt_start = time.time()
        optimization_results = optimizer.optimize()
        opt_time = time.time() - opt_start
        print(f"   ✅ Optimization completed in {opt_time:.2f}s")
        print(f"   🏆 Best score: {optimization_results['best_score']:.4f}")
        
        # Step 4: Get optimized results
        print("\n[4/5] 🔧 Results Analysis")
        print("-" * 40)
        results = optimizer.get_optimized_results()
        
        strategy_metrics = results['strategy_metrics']
        ml_score = results['ml_score']
        signals = results['signals']
        prices = results['optimized_data']['Close'].tolist()
        
        print(f"   📊 Strategy Performance:")
        print(f"      Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.4f}")
        print(f"      Max Drawdown: {strategy_metrics['max_drawdown']:.4f}")
        print(f"      Total Return: {strategy_metrics['total_return']:.4f}")
        print(f"      Win Rate: {strategy_metrics['win_rate']:.4f}")
        print(f"   🤖 ML Score: {ml_score:.4f}")
        
        # Step 5: Generate visualizations
        print("\n[5/5] 🔧 Visualization")
        print("-" * 40)
        
        # Plot optimized strategy
        plot_price_and_signals(
            prices, 
            signals, 
            "Optimized Trading Strategy",
            "output/optimized_strategy.png"
        )
        print("   ✅ Strategy visualization saved")
        
        # Plot performance metrics
        plot_performance_metrics(
            strategy_metrics,
            "output/optimized_performance.png"
        )
        print("   ✅ Performance metrics saved")
        
        # Display best parameters
        print("\n🏆 Best Parameters Found:")
        print("-" * 30)
        for key, value in optimization_results['best_params'].items():
            print(f"   {key}: {value}")
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 Optimization Demo Completed Successfully!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Best strategy score: {optimization_results['best_score']:.4f}")
        print(f"📈 Total return: {strategy_metrics['total_return']:.2%}")
        print(f"🎯 ML accuracy: {ml_score:.4f}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_optimization_demo()
    sys.exit(0 if success else 1) 