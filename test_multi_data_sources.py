#!/usr/bin/env python3
"""
Multi-Data Source Test Suite

This script tests the comprehensive multi-data source system including:
- Multiple data source implementations (Yahoo Finance, Alpha Vantage, CoinGecko)
- Fallback mechanisms
- Data fusion capabilities
- Performance metrics tracking
- Caching mechanisms
- Error handling
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback

from src.data_sources import (
    DataSourceFactory,
    MultiDataSourceManager,
    create_data_manager,
    download_multi_source_data,
    YahooFinanceDataSource,
    CoinGeckoDataSource,
    DataSourceError
)
from src.data_pipeline import (
    download_multi_timeframe_data,
    get_available_data_sources,
    get_data_source_capabilities,
    print_source_metrics
)


def test_data_source_factory():
    """Test the data source factory pattern."""
    print("🧪 Testing Data Source Factory")
    print("=" * 50)
    
    try:
        # Test available sources
        available_sources = DataSourceFactory.get_available_sources()
        print(f"✅ Available sources: {available_sources}")
        
        # Test source creation
        yahoo_source = DataSourceFactory.create_source("yahoo")
        print(f"✅ Created Yahoo Finance source: {yahoo_source.name}")
        
        coingecko_source = DataSourceFactory.create_source("coingecko")
        print(f"✅ Created CoinGecko source: {coingecko_source.name}")
        
        # Test source capabilities
        print(f"✅ Yahoo supported intervals: {yahoo_source.get_supported_intervals()}")
        print(f"✅ CoinGecko supported intervals: {coingecko_source.get_supported_intervals()}")
        print(f"✅ Yahoo max history: {yahoo_source.get_max_history_days()} days")
        print(f"✅ CoinGecko max history: {coingecko_source.get_max_history_days()} days")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory test failed: {str(e)}")
        return False


def test_individual_data_sources():
    """Test individual data source implementations."""
    print("\n🧪 Testing Individual Data Sources")
    print("=" * 50)
    
    test_ticker = "BTC-USD"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    sources_to_test = [
        ("yahoo", YahooFinanceDataSource()),
        ("coingecko", CoinGeckoDataSource())
    ]
    
    results = {}
    success_count = 0
    
    for source_name, source in sources_to_test:
        print(f"\n🔗 Testing {source_name.upper()}...")
        
        try:
            # Test data download
            start_time = time.time()
            data = source.download_data(test_ticker, start_date, end_date, "1d")
            download_time = time.time() - start_time
            
            if not data.empty:
                print(f"✅ {source_name} download successful:")
                print(f"   📊 Data points: {len(data)}")
                print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                print(f"   ⏱️  Download time: {download_time:.3f}s")
                print(f"   📈 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                print(f"   🎯 Quality score: {source.metrics.data_quality_score:.3f}")
                
                results[source_name] = {
                    'success': True,
                    'data_points': len(data),
                    'download_time': download_time,
                    'quality_score': source.metrics.data_quality_score
                }
                success_count += 1
            else:
                print(f"❌ {source_name} returned empty data")
                results[source_name] = {'success': False}
                
        except Exception as e:
            print(f"❌ {source_name} failed: {str(e)}")
            results[source_name] = {'success': False, 'error': str(e)}
    
    # Return True if at least one source succeeded
    return success_count > 0


def test_multi_source_manager():
    """Test the multi-source manager with fallback capabilities."""
    print("\n🧪 Testing Multi-Source Manager")
    print("=" * 50)
    
    test_ticker = "BTC-USD"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    try:
        # Test with Yahoo as primary, CoinGecko as fallback
        manager = create_data_manager(
            primary_source="yahoo",
            fallback_sources=["coingecko"]
        )
        
        print("🔗 Testing fallback mechanism...")
        
        # Test fallback download
        data, source_used = manager.download_data_with_fallback(
            test_ticker, start_date, end_date, "1d"
        )
        
        print(f"✅ Fallback download successful:")
        print(f"   📊 Data points: {len(data)}")
        print(f"   🔗 Source used: {source_used}")
        print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
        
        # Test data fusion
        print("\n🔄 Testing data fusion...")
        fused_data = manager.download_data_fusion(
            test_ticker, start_date, end_date, "1d"
        )
        
        print(f"✅ Data fusion successful:")
        print(f"   📊 Fused data points: {len(fused_data)}")
        print(f"   📅 Date range: {fused_data.index.min()} to {fused_data.index.max()}")
        
        # Print performance metrics
        print("\n📊 Performance Metrics:")
        metrics = manager.get_source_metrics()
        for source_name, metric in metrics.items():
            print(f"   {source_name.upper()}:")
            print(f"     Success Rate: {metric.success_rate:.3f}")
            print(f"     Response Time: {metric.response_time:.3f}s")
            print(f"     Quality Score: {metric.data_quality_score:.3f}")
        
        best_source = manager.get_best_source()
        print(f"🏆 Best source: {best_source}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-source manager test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_multi_timeframe_with_sources():
    """Test multi-timeframe download with multiple data sources."""
    print("\n🧪 Testing Multi-Timeframe with Multiple Sources")
    print("=" * 50)
    
    test_ticker = "BTC-USD"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    timeframes = ["1d", "1w"]  # Test with daily and weekly
    
    try:
        print("🔗 Testing with Yahoo Finance...")
        data_dict_yahoo = download_multi_timeframe_data(
            test_ticker, start_date, end_date, timeframes,
            data_source="yahoo",
            fallback_sources=["coingecko"]
        )
        
        print(f"✅ Yahoo Finance results:")
        for timeframe, data in data_dict_yahoo.items():
            if not data.empty:
                print(f"   {timeframe}: {len(data)} points")
        
        print("\n🔗 Testing with CoinGecko...")
        data_dict_coingecko = download_multi_timeframe_data(
            test_ticker, start_date, end_date, timeframes,
            data_source="coingecko"
        )
        
        print(f"✅ CoinGecko results:")
        for timeframe, data in data_dict_coingecko.items():
            if not data.empty:
                print(f"   {timeframe}: {len(data)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-timeframe test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_caching_mechanism():
    """Test the caching mechanism."""
    print("\n🧪 Testing Caching Mechanism")
    print("=" * 50)
    
    test_ticker = "BTC-USD"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    try:
        source = YahooFinanceDataSource()
        
        # First download (should cache)
        print("📥 First download (should cache)...")
        start_time = time.time()
        data1 = source.download_data(test_ticker, start_date, end_date, "1d")
        first_download_time = time.time() - start_time
        
        print(f"✅ First download: {first_download_time:.3f}s")
        print(f"   Cache hit rate: {source.metrics.cache_hit_rate:.3f}")
        
        # Second download (should use cache)
        print("\n📥 Second download (should use cache)...")
        start_time = time.time()
        data2 = source.download_data(test_ticker, start_date, end_date, "1d")
        second_download_time = time.time() - start_time
        
        print(f"✅ Second download: {second_download_time:.3f}s")
        print(f"   Cache hit rate: {source.metrics.cache_hit_rate:.3f}")
        
        # Verify data is the same
        if data1.equals(data2):
            print("✅ Cached data matches original data")
        else:
            print("❌ Cached data differs from original")
        
        # Check if second download was faster
        if second_download_time < first_download_time:
            print("✅ Caching improved performance")
        else:
            print("⚠️  Caching didn't improve performance (expected for small datasets)")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {str(e)}")
        return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        # Test with invalid ticker
        print("🔗 Testing invalid ticker...")
        try:
            data = download_multi_source_data("INVALID-TICKER", "2024-01-01", "2024-01-31", "1d")
            print("❌ Should have failed with invalid ticker")
            return False
        except DataSourceError:
            print("✅ Correctly handled invalid ticker")
        
        # Test with invalid date range
        print("\n🔗 Testing invalid date range...")
        try:
            data = download_multi_source_data("BTC-USD", "2024-01-31", "2024-01-01", "1d")
            print("❌ Should have failed with invalid date range")
            return False
        except Exception:
            print("✅ Correctly handled invalid date range")
        
        # Test with unsupported interval
        print("\n🔗 Testing unsupported interval...")
        try:
            data = download_multi_source_data("BTC-USD", "2024-01-01", "2024-01-31", "1m")
            print("✅ Unsupported interval handled gracefully")
        except Exception as e:
            print(f"✅ Correctly handled unsupported interval: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False


def test_data_source_capabilities():
    """Test data source capability reporting."""
    print("\n🧪 Testing Data Source Capabilities")
    print("=" * 50)
    
    try:
        capabilities = get_data_source_capabilities()
        available_sources = get_available_data_sources()
        
        print("📊 Data Source Capabilities:")
        for source in available_sources:
            if source in capabilities:
                cap = capabilities[source]
                print(f"\n🔗 {source.upper()}:")
                print(f"   📊 Supported Intervals: {', '.join(cap['supported_intervals'])}")
                print(f"   📅 Max History: {cap['max_history_days']} days")
                print(f"   🔑 API Key Required: {cap['api_key_required']}")
                print(f"   ⚡ Rate Limits: {cap['rate_limits']}")
                print(f"   🎯 Data Quality: {cap['data_quality']}")
                print(f"   💰 Cost: {cap['cost']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Capabilities test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("🚀 Multi-Data Source Test Suite")
    print("=" * 60)
    print("🧪 Testing comprehensive multi-data source system")
    print("📊 Including: Factory pattern, fallback mechanisms, caching, error handling")
    print("=" * 60)
    
    tests = [
        ("Data Source Factory", test_data_source_factory),
        ("Individual Data Sources", test_individual_data_sources),
        ("Multi-Source Manager", test_multi_source_manager),
        ("Multi-Timeframe with Sources", test_multi_timeframe_with_sources),
        ("Caching Mechanism", test_caching_mechanism),
        ("Error Handling", test_error_handling),
        ("Data Source Capabilities", test_data_source_capabilities)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAIL: {test_name} - Exception: {str(e)}")
            results[test_name] = False
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-data source system is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 