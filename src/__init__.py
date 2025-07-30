"""
BTC Technical Alpha Project - Source Package
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"

# Import main classes for easy access
from .data_pipeline import DataPipeline
from .feature_engineering import FeatureEngineer
from .validation import WalkForwardValidator
from .modeling import XGBoostModel
from .evaluation import PerformanceEvaluator
__all__ = [
    "DataPipeline",
    "FeatureEngineer",
    "WalkForwardValidator",
    "XGBoostModel",
    "PerformanceEvaluator",
]
