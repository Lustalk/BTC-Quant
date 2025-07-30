"""
Test script to verify Phase 1 setup.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.statistics.significance_tests import validate_strategy_performance
        from src.statistics.risk_models import calculate_var, calculate_cvar
        from src.statistics.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
        print("✅ Statistics modules imported successfully")
    except Exception as e:
        print(f"❌ Statistics import failed: {e}")
        return False
    
    try:
        from src.data.multi_source_loader import DataLoader, get_bitcoin_data
        from src.data.quality_control import validate_bitcoin_data, clean_bitcoin_data
        print("✅ Data modules imported successfully")
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    return True

def test_statistical_functions():
    """Test statistical functions with sample data."""
    print("\nTesting statistical functions...")
    
    try:
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        benchmark = np.random.normal(0.0008, 0.018, 1000)
        
        # Test significance validation
        from src.statistics.significance_tests import validate_strategy_performance
        result = validate_strategy_performance(returns, benchmark)
        print(f"✅ Strategy validation: {result['statistical_significance']}")
        
        # Test VaR calculation
        from src.statistics.risk_models import calculate_var
        var_value = calculate_var(returns, confidence_level=0.95)
        print(f"✅ VaR calculation: {var_value:.4f}")
        
        # Test Sharpe ratio
        from src.statistics.performance_metrics import calculate_sharpe_ratio
        sharpe = calculate_sharpe_ratio(returns)
        print(f"✅ Sharpe ratio: {sharpe:.4f}")
        
        # Test maximum drawdown
        from src.statistics.performance_metrics import calculate_max_drawdown
        max_dd = calculate_max_drawdown(returns)
        print(f"✅ Max drawdown: {max_dd:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical functions test failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")
    
    try:
        from src.data.multi_source_loader import DataLoader
        from src.data.quality_control import validate_bitcoin_data, clean_bitcoin_data
        
        # Test with mock data (no actual API calls)
        loader = DataLoader(redis_client=None)
        print("✅ DataLoader initialized successfully")
        
        # Test data quality control
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        })
        
        # Test quality validation
        quality_result = validate_bitcoin_data(mock_data)
        print(f"✅ Data quality validation: {quality_result['is_valid']}")
        
        # Test data cleaning
        cleaned_data = clean_bitcoin_data(mock_data)
        print(f"✅ Data cleaning: {len(cleaned_data)} rows processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'src/statistics',
        'src/data',
        'src/validation',
        'src/optimization',
        'src/execution',
        'src/monitoring',
        'src/dashboard',
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/statistical',
        'tests/performance',
        'docs',
        'api',
        'docker',
        'kubernetes',
        'monitoring',
        'ci_cd',
        'logs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def test_config_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")
    
    required_files = [
        'requirements.txt',
        'docker-compose.yml',
        'Dockerfile',
        'main.py',
        '.gitignore',
        'env.example'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required configuration files exist")
        return True

def main():
    """Run all tests."""
    print("🚀 Testing Phase 1 Setup - Professional Bitcoin Trading Analysis")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration Files", test_config_files),
        ("Module Imports", test_imports),
        ("Statistical Functions", test_statistical_functions),
        ("Data Loader", test_data_loader)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 1 setup is complete and working correctly!")
        print("\nNext steps:")
        print("1. Run: docker-compose up -d postgres redis")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run: python main.py")
        print("4. Visit: http://localhost:8000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 