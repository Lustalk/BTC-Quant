"""
Simplified Configuration for BTC Trading Strategy
Essential configuration only - no bloat
"""

import os
from typing import Dict, Any


# Environment-based configuration
def get_env_var(key: str, default: Any) -> Any:
    """Get environment variable with type conversion"""
    value = os.getenv(key, default)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    return value


# Data Configuration
DATA_CONFIG = {
    "symbol": get_env_var("BTC_SYMBOL", "BTC-USD"),
    "start_date": get_env_var("BTC_START_DATE", "2017-01-01"),
    "end_date": get_env_var("BTC_END_DATE", "2024-12-31"),
    "data_source": get_env_var("BTC_DATA_SOURCE", "yfinance"),
    "target_horizon": get_env_var("BTC_TARGET_HORIZON", 5),  # 5-day forward return prediction
    "threshold": get_env_var("BTC_THRESHOLD", 0.52),  # Probability threshold for trading signal
}

# Feature Engineering Configuration - Core indicators only
FEATURE_CONFIG = {
    "momentum_indicators": {
        "rsi": {"window": 14},
        "roc": {"window": 10},
    },
    "trend_indicators": {
        "sma": {"window": 20},
        "ema": {"window": 12},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
    },
    "volatility_indicators": {
        "bbands": {"window": 20, "std": 2},
        "atr": {"window": 14},
    },
    "volume_indicators": {
        "obv": {},
        "volume_roc": {"window": 10},
    },
}

# XGBoost Model Configuration - Simple, robust defaults
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "n_estimators": get_env_var("XGB_N_ESTIMATORS", 200),
    "max_depth": get_env_var("XGB_MAX_DEPTH", 6),
    "learning_rate": get_env_var("XGB_LEARNING_RATE", 0.1),
    "subsample": get_env_var("XGB_SUBSAMPLE", 0.8),
    "colsample_bytree": get_env_var("XGB_COLSAMPLE_BYTREE", 0.8),
    "reg_alpha": get_env_var("XGB_REG_ALPHA", 0.1),
    "reg_lambda": get_env_var("XGB_REG_LAMBDA", 1.0),
    "random_state": get_env_var("XGB_RANDOM_STATE", 42),
    "eval_metric": "logloss",
    "early_stopping_rounds": get_env_var("XGB_EARLY_STOPPING", 50),
}

# Walk-Forward Validation Configuration
VALIDATION_CONFIG = {
    "initial_train_years": get_env_var("VAL_INITIAL_TRAIN_YEARS", 2),
    "test_period_months": get_env_var("VAL_TEST_PERIOD_MONTHS", 3),
    "min_train_size": get_env_var("VAL_MIN_TRAIN_SIZE", 500),
    "expanding_window": get_env_var("VAL_EXPANDING_WINDOW", True),
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "risk_free_rate": get_env_var("EVAL_RISK_FREE_RATE", 0.02),
    "benchmark_symbol": get_env_var("EVAL_BENCHMARK", "SPY"),
    "transaction_costs": get_env_var("EVAL_TRANSACTION_COSTS", 0.0),
    "slippage": get_env_var("EVAL_SLIPPAGE", 0.0),
    "rebalance_frequency": get_env_var("EVAL_REBALANCE", "daily"),
}

# Advanced Configuration
ADVANCED_CONFIG = {
    "monte_carlo": {
        "n_simulations": get_env_var("MC_N_SIMULATIONS", 1000),
        "confidence_level": get_env_var("MC_CONFIDENCE_LEVEL", 0.95),
    },
    "regime_analysis": {
        "window": get_env_var("REGIME_WINDOW", 60),
        "volatility_threshold": get_env_var("REGIME_VOL_THRESHOLD", 0.02),
    },
}

# File Paths Configuration
PATHS_CONFIG = {
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "results_dir": "results",
    "models_dir": "models",
    "logs_dir": "logs",
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project"""
    for path in PATHS_CONFIG.values():
        os.makedirs(path, exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/btc_trading.log",
}

# Core feature names for importance analysis
FEATURE_NAMES = [
    "rsi_14",
    "roc_10",
    "sma_20",
    "ema_12",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper_20_2",
    "bb_middle_20_2",
    "bb_lower_20_2",
    "atr_14",
    "obv",
    "volume_roc_10",
]

# Trading strategy parameters - Simple and robust
STRATEGY_CONFIG = {
    "position_size": 1.0,  # Full position size
    "rebalance_frequency": "daily",
    "transaction_costs": 0.0,  # No transaction costs for simplicity
    "slippage": 0.0,  # No slippage for simplicity
    "threshold": 0.52,  # Fixed threshold - no optimization
}
