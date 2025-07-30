#!/usr/bin/env python3
"""
Test script to analyze optimization results and parameter ranges.
"""

import pandas as pd
import numpy as np
from src.data_pipeline import download_data
from src.parameter_optimization import ParameterOptimizer
from src.strategy_analysis import analyze_strategy_performance

def test_parameter_ranges():
    """Test different parameter ranges to see their impact."""
    print("=" * 60)
    print("🔍 PARAMETER RANGE ANALYSIS")
    print("=" * 60)
    
    # Download data
    data = download_data("BTC-USD", "2023-01-01", "2024-01-01")
    print(f"📊 Data loaded: {len(data)} points")
    
    # Test with default parameters
    optimizer = ParameterOptimizer(data, n_trials=1)
    
    # Test different parameter combinations
    test_params = [
        {
            'name': 'Conservative',
            'params': {
                'sma_short': 10, 'sma_long': 50,
                'ema_short': 12, 'ema_long': 26,
                'rsi_window': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'stoch_window': 14, 'williams_window': 14,
                'bb_window': 20, 'atr_window': 14,
                'macd_fast': 12, 'macd_slow': 26,
                'lag_1': 1, 'lag_2': 2, 'lag_3': 3,
                'roll_short': 5, 'roll_medium': 10, 'roll_long': 20,
                'take_profit': 0.05, 'stop_loss': 0.03,
                'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100,
                'subsample': 0.8, 'colsample_bytree': 0.8
            }
        },
        {
            'name': 'Aggressive',
            'params': {
                'sma_short': 5, 'sma_long': 20,
                'ema_short': 5, 'ema_long': 13,
                'rsi_window': 7, 'rsi_oversold': 25, 'rsi_overbought': 75,
                'stoch_window': 7, 'williams_window': 7,
                'bb_window': 10, 'atr_window': 7,
                'macd_fast': 8, 'macd_slow': 21,
                'lag_1': 1, 'lag_2': 2, 'lag_3': 3,
                'roll_short': 3, 'roll_medium': 7, 'roll_long': 14,
                'take_profit': 0.10, 'stop_loss': 0.05,
                'learning_rate': 0.2, 'max_depth': 8, 'n_estimators': 200,
                'subsample': 0.9, 'colsample_bytree': 0.9
            }
        },
        {
            'name': 'Balanced',
            'params': {
                'sma_short': 15, 'sma_long': 40,
                'ema_short': 10, 'ema_long': 20,
                'rsi_window': 10, 'rsi_oversold': 28, 'rsi_overbought': 72,
                'stoch_window': 10, 'williams_window': 10,
                'bb_window': 15, 'atr_window': 10,
                'macd_fast': 10, 'macd_slow': 24,
                'lag_1': 1, 'lag_2': 3, 'lag_3': 5,
                'roll_short': 5, 'roll_medium': 10, 'roll_long': 20,
                'take_profit': 0.07, 'stop_loss': 0.04,
                'learning_rate': 0.15, 'max_depth': 7, 'n_estimators': 150,
                'subsample': 0.85, 'colsample_bytree': 0.85
            }
        }
    ]
    
    results = []
    
    for test in test_params:
        print(f"\n🧪 Testing {test['name']} Strategy...")
        
        # Create indicators with test parameters
        df_with_indicators = optimizer.create_indicators_with_params(test['params'])
        
        # Generate signals
        signals, entry_prices = optimizer.generate_signals_with_tp_sl(df_with_indicators, test['params'])
        
        # Calculate performance
        prices = df_with_indicators['Close'].tolist()
        if len(signals) == len(prices):
            strategy_metrics = analyze_strategy_performance(prices, signals)
            
            # Count signals
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == -1)
            total_signals = buy_signals + sell_signals
            
            results.append({
                'name': test['name'],
                'metrics': strategy_metrics,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': total_signals,
                'signal_rate': total_signals / len(signals) if len(signals) > 0 else 0
            })
            
            print(f"   📊 Buy signals: {buy_signals}")
            print(f"   📊 Sell signals: {sell_signals}")
            print(f"   📊 Total signals: {total_signals}")
            print(f"   📊 Signal rate: {total_signals / len(signals) * 100:.2f}%")
            print(f"   📊 Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"   📊 Total Return: {strategy_metrics.get('total_return', 0):.2%}")
            print(f"   📊 Win Rate: {strategy_metrics.get('win_rate', 0):.2%}")
        else:
            print(f"   ❌ Signal length mismatch!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY OF PARAMETER RANGE TESTING")
    print("=" * 60)
    
    for result in results:
        print(f"\n🎯 {result['name']} Strategy:")
        print(f"   • Signal Rate: {result['signal_rate']*100:.2f}%")
        print(f"   • Total Signals: {result['total_signals']}")
        print(f"   • Sharpe Ratio: {result['metrics'].get('sharpe_ratio', 0):.4f}")
        print(f"   • Total Return: {result['metrics'].get('total_return', 0):.2%}")
        print(f"   • Win Rate: {result['metrics'].get('win_rate', 0):.2%}")
    
    return results

if __name__ == "__main__":
    test_parameter_ranges() 