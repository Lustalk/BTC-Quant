#!/usr/bin/env python3
"""
Test Real Bitcoin Analysis
Test the complete analysis pipeline with real Bitcoin data.
"""

import sys
import os
sys.path.append('.')

from src.statistics.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from src.statistics.risk_models import calculate_var
from src.data.multi_source_loader import get_bitcoin_data
import numpy as np
from datetime import datetime, timedelta

def test_real_bitcoin_analysis():
    """Test real Bitcoin analysis with live data."""
    print("🚀 Testing Real Bitcoin Analysis")
    print("=" * 50)
    
    try:
        # Get Bitcoin data
        print("📊 Fetching Bitcoin data...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_bitcoin_data(start_date, end_date)
        
        if data is None or data.empty:
            print("❌ Failed to fetch Bitcoin data")
            return False
        
        print(f"✅ Data loaded: {len(data)} data points")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        returns_array = returns.values  # Convert to numpy array
        print(f"📈 Returns calculated: {len(returns)} observations")
        
        # Performance metrics
        sharpe = calculate_sharpe_ratio(returns_array)
        max_dd = calculate_max_drawdown(returns_array)
        var_95 = calculate_var(returns_array, confidence_level=0.95)
        
        print("\n📊 Performance Metrics:")
        print(f"   Sharpe Ratio: {float(sharpe):.4f}")
        print(f"   Max Drawdown: {float(max_dd):.4f}")
        print(f"   VaR (95%): {float(var_95):.4f}")
        
        # Price statistics
        current_price = float(data['Close'].iloc[-1])
        initial_price = float(data['Close'].iloc[0])
        price_change = ((current_price - initial_price) / initial_price) * 100
        volatility = float(returns.std() * np.sqrt(252) * 100)
        
        print(f"\n💰 Price Analysis:")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   Annual Return: {price_change:.2f}%")
        print(f"   Volatility: {volatility:.2f}%")
        
        print("\n✅ Real Bitcoin analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_real_bitcoin_analysis()
    sys.exit(0 if success else 1) 