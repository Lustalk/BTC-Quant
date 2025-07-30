#!/usr/bin/env python3
"""
BTC Quant - Main Backtesting Engine

This script implements a minimum viable backtesting engine that:
1. Downloads financial data
2. Engineers technical indicators
3. Trains an XGBoost model
4. Performs walk-forward validation
"""

import argparse
import sys
from datetime import datetime
import pandas as pd

from src.data_pipeline import download_data
from src.feature_engineering import add_technical_indicators
from src.model import prepare_features_target, walk_forward_validation


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
    
    args = parser.parse_args()
    
    print(f"BTC Quant Backtesting Engine")
    print(f"Ticker: {args.ticker}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Validation Splits: {args.n_splits}")
    print("-" * 50)
    
    try:
        # Step 1: Download data
        print("Step 1: Downloading financial data...")
        data = download_data(args.ticker, args.start_date, args.end_date)
        print(f"Downloaded {len(data)} data points")
        
        # Step 2: Engineer features
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
        
        # Step 5: Report results
        print("\n" + "=" * 50)
        print("BACKTESTING RESULTS")
        print("=" * 50)
        print(f"Average Accuracy: {sum(scores) / len(scores):.4f}")
        print(f"Individual Fold Accuracies: {[f'{s:.4f}' for s in scores]}")
        print(f"Best Fold: {max(scores):.4f}")
        print(f"Worst Fold: {min(scores):.4f}")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 