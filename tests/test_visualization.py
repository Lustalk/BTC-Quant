"""
Tests for the visualization module.

This module tests all visualization functions to ensure they work correctly
with various input data and edge cases.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# flake8: noqa: E402
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.visualization import (
    plot_price_and_signals,
    plot_performance_metrics,
    plot_model_accuracy,
    create_performance_dashboard,
)


class TestVisualization:
    """Test class for visualization functions."""

    def setup_method(self):
        """Set up test data."""
        self.sample_prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
        self.sample_signals = [0, 1, 0, 0, -1, 0, 1, 0, -1, 0]
        self.sample_metrics = {
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.15,
            "volatility": 0.2,
            "win_rate": 0.6,
            "profit_factor": 1.5,
            "total_return": 0.25,
            "avg_return": 0.02,
            "auc": 0.56,
        }
        self.sample_scores = [0.52, 0.48, 0.54, 0.51, 0.49]

    @patch("matplotlib.pyplot.close")
    def test_plot_price_and_signals(self, mock_close):
        """Test plot_price_and_signals function."""
        # Test with valid data
        plot_price_and_signals(self.sample_prices, self.sample_signals)
        mock_close.assert_called_once()

        # Test with empty signals
        plot_price_and_signals(self.sample_prices, [0] * len(self.sample_prices))
        assert mock_close.call_count == 2

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_price_and_signals(
                self.sample_prices, self.sample_signals, save_path="test_plot.png"
            )
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.close")
    def test_plot_performance_metrics(self, mock_close):
        """Test plot_performance_metrics function."""
        # Test with valid metrics
        plot_performance_metrics(self.sample_metrics)
        mock_close.assert_called_once()

        # Test with empty metrics
        plot_performance_metrics({})
        assert mock_close.call_count == 2

        # Test with non-numeric metrics
        mixed_metrics = {"text": "not_numeric", "number": 42}
        plot_performance_metrics(mixed_metrics)
        assert mock_close.call_count == 3

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_performance_metrics(self.sample_metrics, save_path="test_metrics.png")
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.close")
    def test_plot_model_accuracy(self, mock_close):
        """Test plot_model_accuracy function."""
        # Test with valid scores
        plot_model_accuracy(self.sample_scores)
        mock_close.assert_called_once()

        # Test with single score
        plot_model_accuracy([0.5])
        assert mock_close.call_count == 2

        # Test with custom title
        plot_model_accuracy(self.sample_scores, title="Custom Title")
        assert mock_close.call_count == 3

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_model_accuracy(self.sample_scores, save_path="test_accuracy.png")
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.close")
    def test_create_performance_dashboard(self, mock_close):
        """Test create_performance_dashboard function."""
        # Test with valid data
        create_performance_dashboard(
            self.sample_prices, self.sample_signals, self.sample_metrics, self.sample_scores
        )
        mock_close.assert_called_once()

        # Test with empty metrics
        create_performance_dashboard(
            self.sample_prices, self.sample_signals, {}, self.sample_scores
        )
        assert mock_close.call_count == 2

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            create_performance_dashboard(
                self.sample_prices,
                self.sample_signals,
                self.sample_metrics,
                self.sample_scores,
                save_path="test_dashboard.png",
            )
            mock_save.assert_called_once()

    def test_edge_cases(self):
        """Test visualization functions with edge cases."""
        # Test with very short data
        short_prices = [100, 101]
        short_signals = [0, 1]

        with patch("matplotlib.pyplot.close"):
            plot_price_and_signals(short_prices, short_signals)
            plot_performance_metrics({"single_metric": 0.5})
            plot_model_accuracy([0.5])
            create_performance_dashboard(short_prices, short_signals, {"metric": 0.5}, [0.5])

    def test_input_validation(self):
        """Test input validation for visualization functions."""
        # Test with mismatched lengths
        with pytest.raises(IndexError):
            with patch("matplotlib.pyplot.show"):
                plot_price_and_signals([1, 2, 3], [1, 2])  # Different lengths

        # Test with invalid data types
        with patch("matplotlib.pyplot.show"):
            # These should handle gracefully
            plot_performance_metrics({"invalid": "not_numeric"})
            plot_model_accuracy([0.5, "invalid", 0.6])

    def test_matplotlib_backend(self):
        """Test that matplotlib backend is properly configured."""
        # Ensure we can create plots without display
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_price_and_signals(self.sample_prices, self.sample_signals, save_path="test.png")
            mock_save.assert_called_once()


def test_visualization_integration():
    """Integration test for visualization with real data."""
    # Create realistic test data
    prices = [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(50)]
    signals = [np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]) for _ in range(50)]
    metrics = {
        "sharpe_ratio": 0.52,
        "max_drawdown": 0.12,
        "volatility": 0.18,
        "win_rate": 0.58,
        "profit_factor": 1.42,
        "total_return": 0.23,
        "avg_return": 0.001,
        "auc": 0.54,
    }
    scores = [0.51, 0.53, 0.49, 0.52, 0.50]

    # Test all visualization functions
    with patch("matplotlib.pyplot.show"):
        plot_price_and_signals(prices, signals)
        plot_performance_metrics(metrics)
        plot_model_accuracy(scores)
        create_performance_dashboard(prices, signals, metrics, scores)

    # If we get here without errors, the integration test passes
    assert True
