#!/usr/bin/env python3
"""
Test Script for Enhanced BTC Quant Features
Verifies all professional deliverables are working correctly
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.hyperparameter_optimization import HyperparameterOptimizer
from src.feature_selection import FeatureSelector
from src.enhanced_validation import EnhancedWalkForwardValidator
from src.monte_carlo_simulation import MonteCarloSimulator
from src.professional_visualizations import ProfessionalVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """
    Create synthetic test data for verification
    """
    logger.info("Creating test data...")
    
    # Generate synthetic time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate synthetic price data
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = 100 * (1 + returns).cumprod()
    
    # Create feature data
    features = {}
    for i in range(20):  # 20 synthetic features
        features[f'feature_{i}'] = np.random.normal(0, 1, n_days)
    
    # Create target (binary classification)
    target = (returns > np.median(returns)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'returns': returns,
        'target': target,
        **features
    }, index=dates)
    
    logger.info(f"Test data created: {len(data)} observations, {len(features)} features")
    return data

def test_hyperparameter_optimization(data):
    """
    Test hyperparameter optimization
    """
    logger.info("Testing hyperparameter optimization...")
    
    # Prepare data
    feature_columns = [col for col in data.columns if col.startswith('feature_')]
    X = data[feature_columns]
    y = data['target']
    
    # Run optimization
    hyperopt = HyperparameterOptimizer(n_trials=5)  # Small number for testing
    best_params, study = hyperopt.optimize_hyperparameters(X, y, data.index)
    
    # Verify results
    assert best_params is not None, "Best parameters should not be None"
    assert study is not None, "Study should not be None"
    assert len(hyperopt.optimization_history) > 0, "Optimization history should not be empty"
    
    logger.info("✅ Hyperparameter optimization test passed")
    return best_params, study

def test_feature_selection(data):
    """
    Test feature selection
    """
    logger.info("Testing feature selection...")
    
    # Prepare data
    feature_columns = [col for col in data.columns if col.startswith('feature_')]
    X = data[feature_columns]
    y = data['target']
    
    # Run feature selection
    selector = FeatureSelector()
    feature_importance = selector.calculate_feature_importance(X, y, data.index)
    rfe_results = selector.recursive_feature_elimination(X, y, data.index)
    optimal_features, combined_df = selector.get_optimal_feature_set(X, y, data.index, max_features=10)
    
    # Verify results
    assert feature_importance is not None, "Feature importance should not be None"
    assert rfe_results is not None, "RFE results should not be None"
    assert len(optimal_features) > 0, "Optimal features should not be empty"
    assert combined_df is not None, "Combined dataframe should not be None"
    
    logger.info("✅ Feature selection test passed")
    return feature_importance, optimal_features

def test_enhanced_validation(data, feature_columns):
    """
    Test enhanced walk-forward validation
    """
    logger.info("Testing enhanced walk-forward validation...")
    
    # Create a simple model for testing
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Run validation
    validator = EnhancedWalkForwardValidator(
        initial_train_years=1,
        test_period_months=3,
        min_train_size=100
    )
    
    results = validator.validate_model(data, model, feature_columns)
    
    # Verify results
    assert 'period_metrics' in results, "Period metrics should be in results"
    assert 'daily_predictions' in results, "Daily predictions should be in results"
    assert 'overall_metrics' in results, "Overall metrics should be in results"
    assert len(results['period_metrics']) > 0, "Should have validation periods"
    
    logger.info("✅ Enhanced validation test passed")
    return results

def test_monte_carlo_simulation(data, probabilities):
    """
    Test Monte Carlo simulation
    """
    logger.info("Testing Monte Carlo simulation...")
    
    # Run simulation
    simulator = MonteCarloSimulator(n_simulations=50)  # Small number for testing
    simulation_results = simulator.simulate_strategy_performance(
        data['returns'], 
        probabilities
    )
    
    # Calculate rolling metrics
    rolling_metrics = simulator.calculate_rolling_metrics(
        data['returns'], 
        probabilities
    )
    
    # Verify results
    assert len(simulation_results) > 0, "Simulation results should not be empty"
    assert rolling_metrics is not None, "Rolling metrics should not be None"
    
    logger.info("✅ Monte Carlo simulation test passed")
    return simulation_results, rolling_metrics

def test_visualizations(data, predictions, probabilities, feature_importance):
    """
    Test professional visualizations
    """
    logger.info("Testing professional visualizations...")
    
    # Create visualizer
    visualizer = ProfessionalVisualizer()
    
    # Test individual plots
    fig1 = visualizer.create_cumulative_returns_plot(data, predictions, probabilities)
    fig2 = visualizer.create_feature_importance_plot(feature_importance)
    fig3 = visualizer.create_probability_distribution_plot(probabilities, predictions)
    
    # Verify plots were created
    assert fig1 is not None, "Cumulative returns plot should be created"
    assert fig2 is not None, "Feature importance plot should be created"
    assert fig3 is not None, "Probability distribution plot should be created"
    
    logger.info("✅ Professional visualizations test passed")
    return visualizer

def test_exports():
    """
    Test that exports are created
    """
    logger.info("Testing exports...")
    
    # Check if exports directory exists
    assert os.path.exists('exports'), "Exports directory should exist"
    
    # Check if visualization directory exists
    viz_dir = 'exports/visualizations'
    if os.path.exists(viz_dir):
        logger.info(f"✅ Visualization directory exists: {viz_dir}")
    
    logger.info("✅ Exports test passed")

def run_all_tests():
    """
    Run all tests
    """
    logger.info("="*60)
    logger.info("RUNNING ENHANCED FEATURES TEST SUITE")
    logger.info("="*60)
    
    try:
        # Create test data
        data = create_test_data()
        
        # Test 1: Hyperparameter optimization
        best_params, study = test_hyperparameter_optimization(data)
        
        # Test 2: Feature selection
        feature_importance, optimal_features = test_feature_selection(data)
        
        # Test 3: Enhanced validation
        feature_columns = [col for col in data.columns if col.startswith('feature_')]
        validation_results = test_enhanced_validation(data, feature_columns)
        
        # Test 4: Monte Carlo simulation
        probabilities = np.random.random(len(data))  # Synthetic probabilities
        simulation_results, rolling_metrics = test_monte_carlo_simulation(data, probabilities)
        
        # Test 5: Visualizations
        predictions = np.random.randint(0, 2, len(data))  # Synthetic predictions
        visualizer = test_visualizations(data, predictions, probabilities, feature_importance)
        
        # Test 6: Exports
        test_exports()
        
        logger.info("="*60)
        logger.info("ALL TESTS PASSED SUCCESSFULLY! ✅")
        logger.info("="*60)
        
        # Print summary
        logger.info("Test Summary:")
        logger.info(f"- Data: {len(data)} observations, {len(feature_columns)} features")
        logger.info(f"- Validation periods: {len(validation_results['period_metrics'])}")
        logger.info(f"- Monte Carlo simulations: {len(simulation_results)}")
        logger.info(f"- Optimal features: {len(optimal_features)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 