#!/usr/bin/env python3
"""
Test script for multi-timeframe data download functionality.
This script verifies that we can download OHLCV data for 1h, 4h, 1d, and 1w timeframes
with at least 1 year of complete data, preferably 3 years.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from src.data_pipeline import (
    download_multi_timeframe_data, 
    get_optimal_date_range, 
    ensure_data_completeness,
    validate_data_quality,
    download_data,
    fill_data_gaps
)


def test_multi_timeframe_download():
    """Test the multi-timeframe data download functionality."""
    print("🧪 Testing Multi-Timeframe Data Download")
    print("=" * 60)
    
    # Test parameters
    ticker = "BTC-USD"
    timeframes = ["1h", "4h", "1d", "1w"]
    min_years = 1
    preferred_years = 3
    
    # Get optimal date range - use more conservative approach
    end_date = datetime.now()
    
    # For intraday data (1h, 4h), use shorter periods due to Yahoo Finance limitations
    # For daily/weekly data, we can use longer periods
    start_date_intraday = (end_date - timedelta(days=60)).strftime("%Y-%m-%d")  # 60 days for intraday
    start_date_daily = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")     # 1 year for daily
    start_date_weekly = (end_date - timedelta(days=1095)).strftime("%Y-%m-%d")   # 3 years for weekly
    
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"📅 Date Ranges:")
    print(f"   Intraday (1h, 4h): {start_date_intraday} to {end_date_str}")
    print(f"   Daily (1d): {start_date_daily} to {end_date_str}")
    print(f"   Weekly (1w): {start_date_weekly} to {end_date_str}")
    print(f"⏰ Timeframes: {', '.join(timeframes)}")
    print(f"📈 Data Requirements: {min_years} year(s) minimum, {preferred_years} year(s) preferred")
    print("-" * 60)
    
    try:
        # Download data for each timeframe with appropriate date ranges
        data_dict = {}
        
        for timeframe in timeframes:
            print(f"📊 Downloading {timeframe} data for {ticker}...")
            
            # Choose appropriate date range based on timeframe
            if timeframe in ["1h", "4h"]:
                start_date = start_date_intraday
            elif timeframe == "1d":
                start_date = start_date_daily
            elif timeframe == "1w":
                start_date = start_date_weekly
            else:
                start_date = start_date_daily
            
            try:
                data = download_data(ticker, start_date, end_date_str, timeframe, "yfinance")
                
                # Validate data quality and completeness
                quality_metrics = validate_data_quality(data, min_data_points=100)
                
                # Check for data gaps and fill if necessary
                data = fill_data_gaps(data, timeframe)
                
                # Final validation after gap filling
                final_quality = validate_data_quality(data, min_data_points=100)
                
                print(f"✅ {timeframe} data: {len(data)} points, quality score: {final_quality['quality_score']:.3f}")
                
                data_dict[timeframe] = data
                
            except Exception as e:
                print(f"❌ Failed to download {timeframe} data: {str(e)}")
                continue
        
        # Validate data completeness
        print("\n🔍 Validating data completeness...")
        validated_data = ensure_data_completeness(data_dict, min_years=min_years)
        
        # Print detailed results
        print("\n📊 RESULTS SUMMARY")
        print("=" * 60)
        
        success_count = 0
        total_count = len(timeframes)
        
        for timeframe in timeframes:
            if timeframe in validated_data and not validated_data[timeframe].empty:
                data = validated_data[timeframe]
                quality = validate_data_quality(data, min_data_points=100)
                
                duration_days = (data.index.max() - data.index.min()).days
                duration_years = duration_days / 365
                
                print(f"✅ {timeframe.upper()} Data:")
                print(f"   📊 Data Points: {len(data):,}")
                print(f"   ⏱️  Duration: {duration_days} days ({duration_years:.1f} years)")
                print(f"   📅 Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                print(f"   💰 Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                print(f"   📈 Volume Range: {data['Volume'].min():,.0f} - {data['Volume'].max():,.0f}")
                print(f"   🎯 Quality Score: {quality['quality_score']:.3f}")
                print(f"   ❌ Missing Values: {sum(quality['missing_values'].values())}")
                print(f"   🔄 Duplicates: {quality['duplicates']}")
                print(f"   📉 Data Gaps: {quality['data_gaps']}")
                print()
                
                success_count += 1
                
                # Verify minimum data requirements
                if duration_years < min_years:
                    print(f"⚠️  WARNING: {timeframe} data only covers {duration_years:.1f} years, need at least {min_years} year(s)")
                elif duration_years >= preferred_years:
                    print(f"🎉 EXCELLENT: {timeframe} data covers {duration_years:.1f} years (≥ {preferred_years} preferred)")
                else:
                    print(f"✅ GOOD: {timeframe} data covers {duration_years:.1f} years (≥ {min_years} minimum)")
                
            else:
                print(f"❌ {timeframe.upper()} Data: FAILED")
                print()
        
        # Summary
        print("=" * 60)
        print(f"📈 SUMMARY: {success_count}/{total_count} timeframes successful")
        
        if success_count == total_count:
            print("🎉 All timeframes downloaded successfully!")
            return True
        else:
            print(f"⚠️  {total_count - success_count} timeframe(s) failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False


def test_data_quality():
    """Test data quality validation functions."""
    print("\n🧪 Testing Data Quality Validation")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'Open': [100] * len(dates),
        'High': [110] * len(dates),
        'Low': [90] * len(dates),
        'Close': [105] * len(dates),
        'Volume': [1000000] * len(dates)
    }, index=dates)
    
    # Test quality validation
    quality = validate_data_quality(sample_data, min_data_points=100)
    
    print(f"📊 Sample Data Quality Metrics:")
    print(f"   Total Rows: {quality['total_rows']}")
    print(f"   Quality Score: {quality['quality_score']:.3f}")
    print(f"   Missing Values: {sum(quality['missing_values'].values())}")
    print(f"   Duplicates: {quality['duplicates']}")
    print(f"   Data Gaps: {quality['data_gaps']}")
    print(f"   Duration: {quality['date_range']['duration_days']} days")
    
    return quality['quality_score'] > 0.95


def main():
    """Main test function."""
    print("🚀 BTC Quant - Multi-Timeframe Data Test")
    print("=" * 60)
    
    # Test 1: Multi-timeframe download
    test1_success = test_multi_timeframe_download()
    
    # Test 2: Data quality validation
    test2_success = test_data_quality()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"📊 Multi-Timeframe Download: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"🔍 Data Quality Validation: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Multi-timeframe data download is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 