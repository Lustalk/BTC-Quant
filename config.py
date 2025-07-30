"""
Configuration file for Technical Indicator Alpha Project
Contains all model parameters, data settings, and validation parameters
"""

import os
from datetime import datetime, timedelta

# Data Configuration
DATA_CONFIG = {
    'symbol': 'BTC-USD',
    'start_date': '2017-01-01',
    'end_date': '2024-12-31',
    'data_source': 'yfinance',
    'target_horizon': 5,  # 5-day forward return prediction
    'threshold': 0.52,  # Probability threshold for trading signal (will be optimized dynamically)
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'momentum_indicators': {
        'rsi': {'window': 14},
        'stoch': {'window': 14, 'smooth_window': 3},
        'williams_r': {'window': 14},
        'roc': {'window': 10},
    },
    'trend_indicators': {
        'sma': {'window': 20},
        'ema': {'window': 12},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'adx': {'window': 14},
        'psar': {},
    },
    'volatility_indicators': {
        'bbands': {'window': 20, 'std': 2},
        'atr': {'window': 14},
        'realized_volatility': {'window': 20},  # New: Realized volatility
    },
    'volume_indicators': {
        'obv': {},
        'volume_roc': {'window': 10},
        'vwap': {'window': 20},  # New: Volume Weighted Average Price
        'vwap_deviation': {'window': 20},  # New: VWAP deviation
    },
    'price_action': {
        'price_to_52w_high': {},
        'price_to_52w_low': {},
        'volatility_20d': {'window': 20},
    },
    'sentiment_indicators': {
        'fear_greed_index': {},  # New: Bitcoin Fear & Greed Index (conditional)
    }
}

# XGBoost Model Configuration
XGBOOST_CONFIG = {
    'objective': 'binary:logistic',
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50,
}

# Walk-Forward Validation Configuration
VALIDATION_CONFIG = {
    'initial_train_years': 2,  # Initial training period in years
    'test_period_months': 3,   # Test period length in months
    'min_train_size': 500,     # Minimum training samples
    'expanding_window': True,  # Use expanding window approach
}

# Performance Evaluation Configuration
EVALUATION_CONFIG = {
    'risk_free_rate': 0.02,  # Annual risk-free rate
    'benchmark': 'buy_hold',
    'metrics': ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio'],
    'plot_style': 'seaborn-v0_8',
}

# File Paths Configuration
PATHS_CONFIG = {
    'data_dir': 'data',
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'results_dir': 'results',
    'models_dir': 'models',
    'logs_dir': 'logs',
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project"""
    for path in PATHS_CONFIG.values():
        os.makedirs(path, exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/technical_alpha.log',
}

# Feature names for importance analysis
FEATURE_NAMES = [
    'rsi_14', 'stoch_k_14', 'williams_r_14', 'roc_10',
    'sma_20', 'ema_12', 'macd', 'macd_signal', 'macd_hist',
    'adx_14', 'psar', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'realized_volatility_20', 'obv', 'volume_roc_10',
    'vwap', 'vwap_deviation', 'price_to_52w_high',
    'price_to_52w_low', 'volatility_20d', 'fear_greed_index'
]

# Trading strategy parameters
STRATEGY_CONFIG = {
    'position_size': 1.0,  # Full position size
    'rebalance_frequency': 'daily',
    'transaction_costs': 0.0,  # No transaction costs for simplicity
    'slippage': 0.0,  # No slippage for simplicity
    'dynamic_threshold': True,  # Enable dynamic threshold optimization
    'threshold_optimization_window': 252,  # 1 year for threshold optimization
}

# Advanced Features Configuration
ADVANCED_CONFIG = {
    'regime_detection': {
        'enabled': True,
        'volatility_clustering': True,
        'regime_window': 60,  # 60-day regime detection
    },
    'risk_management': {
        'enabled': True,
        'volatility_targeting': True,
        'position_sizing': True,
        'max_position_size': 1.0,
        'volatility_target': 0.15,  # 15% annualized volatility target
    },
    'transaction_costs': {
        'enabled': False,  # Set to True for realistic modeling
        'commission_rate': 0.001,  # 0.1% commission
        'slippage_rate': 0.0005,  # 0.05% slippage
    },
    'monte_carlo': {
        'enabled': False,  # Set to True for Monte Carlo simulation
        'n_simulations': 1000,
        'simulation_years': 5,
    }
} 