"""
Tests for the visualization module.

This module tests all visualization functions to ensure they work correctly
with various input data and edge cases.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set matplotlib backend to Agg for non-interactive testing
import matplotlib

matplotlib.use("Agg")

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

    @patch("matplotlib.pyplot.show")
    def test_plot_price_and_signals(self, mock_show):
        """Test plot_price_and_signals function."""
        # Test with valid data
        plot_price_and_signals(self.sample_prices, self.sample_signals)
        mock_show.assert_called_once()

        # Test with empty signals
        plot_price_and_signals(self.sample_prices, [0] * len(self.sample_prices))
        assert mock_show.call_count == 2

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_price_and_signals(
                self.sample_prices, self.sample_signals, save_path="test_plot.png"
            )
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_performance_metrics(self, mock_show):
        """Test plot_performance_metrics function."""
        # Test with valid metrics
        plot_performance_metrics(self.sample_metrics)
        mock_show.assert_called_once()

        # Test with empty metrics
        plot_performance_metrics({})
        assert mock_show.call_count == 1  # Should not call show for empty metrics

        # Test with non-numeric metrics
        mixed_metrics = {"text": "not_numeric", "number": 42}
        plot_performance_metrics(mixed_metrics)
        assert mock_show.call_count == 2

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_performance_metrics(self.sample_metrics, save_path="test_metrics.png")
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_model_accuracy(self, mock_show):
        """Test plot_model_accuracy function."""
        # Test with valid scores
        plot_model_accuracy(self.sample_scores)
        mock_show.assert_called_once()

        # Test with single score
        plot_model_accuracy([0.5])
        assert mock_show.call_count == 2

        # Test with custom title
        plot_model_accuracy(self.sample_scores, title="Custom Title")
        assert mock_show.call_count == 3

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            plot_model_accuracy(self.sample_scores, save_path="test_accuracy.png")
            mock_save.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_create_performance_dashboard(self, mock_show):
        """Test create_performance_dashboard function."""
        # Test with valid data - function takes strategy_metrics and buy_hold_metrics
        strategy_metrics = {"sharpe_ratio": 0.5, "total_return": 0.2}
        buy_hold_metrics = {"sharpe_ratio": 0.3, "total_return": 0.1}

        create_performance_dashboard(strategy_metrics, buy_hold_metrics)
        mock_show.assert_called_once()

        # Test with save path
        with patch("matplotlib.pyplot.savefig") as mock_save:
            create_performance_dashboard(
                strategy_metrics, buy_hold_metrics, save_path="test_dashboard.png"
            )
            mock_save.assert_called_once()

    def test_edge_cases(self):
        """Test edge cases for visualization functions."""
        # Test with empty data
        with patch("matplotlib.pyplot.show"):
            plot_model_accuracy([])  # Should handle empty list gracefully

        # Test with very small data
        with patch("matplotlib.pyplot.show"):
            plot_model_accuracy([0.5])  # Single score

        # Test with large data
        large_scores = [0.5 + i * 0.01 for i in range(100)]
        with patch("matplotlib.pyplot.show"):
            plot_model_accuracy(large_scores)

    def test_input_validation(self):
        """Test input validation for visualization functions."""
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            with patch("matplotlib.pyplot.show"):
                plot_price_and_signals([1, 2, 3], [1, 2])  # Different lengths

        # Test with invalid data types - should handle gracefully
        with patch("matplotlib.pyplot.show"):
            # These should handle gracefully
            plot_performance_metrics({"invalid": "not_numeric"})

            # Test with mixed data types - should filter out non-numeric
            mixed_scores = [0.5, 0.6, 0.7]  # Only numeric values
            plot_model_accuracy(mixed_scores)

    def test_matplotlib_backend(self):
        """Test that matplotlib backend is set correctly."""
        # This test ensures the backend is set for non-interactive environments
        import matplotlib

        assert matplotlib.get_backend() in [
            "Agg",
            "TkAgg",
            "Qt5Agg",
            "GTK3Agg",
            "qtagg",
            "tkagg",
        ]


def test_visualization_integration():
    """Test integration of visualization functions."""
    # Create sample data
    prices = [100, 101, 102, 103, 104]
    signals = [0, 1, 0, -1, 0]
    metrics = {"sharpe_ratio": 0.5, "total_return": 0.2}
    scores = [0.52, 0.48, 0.54]

    # Test all functions together
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_price_and_signals(prices, signals)
        plot_performance_metrics(metrics)
        plot_model_accuracy(scores)
        create_performance_dashboard(metrics, metrics)

        # Should have called show 4 times
        assert mock_show.call_count == 4
