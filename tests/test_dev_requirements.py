"""
Test for development requirements.

This module tests that all development dependencies can be imported
and work correctly for development tasks.
"""

import pytest
import sys
import importlib


def test_core_dependencies():
    """Test that core dependencies can be imported."""
    core_modules = [
        "pandas",
        "numpy",
        "yfinance",
        "xgboost",
        "sklearn",
        "pytest",
        "pytest_cov",
        "flake8",
        "ta",
    ]

    for module in core_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_development_tools():
    """Test that development tools can be imported."""
    dev_modules = [
        "pytest_mock",
        "xdist",  # pytest_xdist is imported as xdist
        "pytest_html",
        "coverage",
        "black",
        "isort",
        "mypy",
    ]

    for module in dev_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_visualization_dependencies():
    """Test that visualization dependencies can be imported."""
    viz_modules = ["matplotlib", "seaborn"]

    for module in viz_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_documentation_tools():
    """Test that documentation tools can be imported."""
    doc_modules = ["sphinx", "sphinx_rtd_theme"]

    for module in doc_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_code_quality_tools():
    """Test that code quality tools can be imported."""
    quality_modules = ["pre_commit", "bandit"]

    for module in quality_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_jupyter_tools():
    """Test that Jupyter tools can be imported."""
    jupyter_modules = ["jupyter", "ipykernel"]

    for module in jupyter_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_data_science_tools():
    """Test that data science tools can be imported."""
    ds_modules = ["scipy", "plotly"]

    for module in ds_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_development_utilities():
    """Test that development utilities can be imported."""
    util_modules = ["dotenv", "click", "rich"]

    for module in util_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


def test_requirements_file_exists():
    """Test that requirements-dev.txt file exists and is readable."""
    import os

    requirements_file = "requirements-dev.txt"

    assert os.path.exists(requirements_file), f"{requirements_file} does not exist"

    with open(requirements_file, "r") as f:
        content = f.read()
        assert len(content) > 0, f"{requirements_file} is empty"
        assert "pandas" in content, "pandas not found in requirements-dev.txt"
        assert "matplotlib" in content, "matplotlib not found in requirements-dev.txt"


def test_version_compatibility():
    """Test that key dependencies have compatible versions."""
    import pandas as pd
    import numpy as np
    import matplotlib as mpl
    from packaging import version

    # Test pandas version
    assert version.parse(pd.__version__) >= version.parse(
        "2.0.0"
    ), f"pandas version {pd.__version__} is too old"

    # Test numpy version
    assert version.parse(np.__version__) >= version.parse(
        "1.20.0"
    ), f"numpy version {np.__version__} is too old"

    # Test matplotlib version
    assert version.parse(mpl.__version__) >= version.parse(
        "3.5.0"
    ), f"matplotlib version {mpl.__version__} is too old"


def test_development_environment():
    """Test that the development environment is properly configured."""
    # Test that we can import our own modules
    try:
        from src.visualization import plot_price_and_signals

        assert callable(plot_price_and_signals)
    except ImportError as e:
        pytest.fail(f"Failed to import visualization module: {e}")

    # Test that pytest is working
    assert pytest.__version__ >= "7.0.0", f"pytest version {pytest.__version__} is too old"

    # Test that we can use mocking
    from unittest.mock import patch, MagicMock

    assert callable(patch)
    assert callable(MagicMock)
