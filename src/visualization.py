"""
Visualization module for BTC Quant trading system.

This module provides simple, professional charts for displaying
trading results and system performance.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns

# Set professional style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_price_and_signals(
    prices: List[float],
    signals: List[int],
    title: str = "Price Movement and Trading Signals",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot price movement with buy/sell signals.

    Args:
        prices: List of price values
        signals: List of trading signals (1=buy, -1=sell, 0=hold)
        title: Chart title
        save_path: Optional path to save the plot
    """
    # Validate input lengths
    if len(prices) != len(signals):
        raise IndexError(
            f"Length mismatch: prices has {len(prices)} elements, signals has {len(signals)} elements"
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Plot price
    ax1.plot(prices, label="Price", color="#2E86AB", linewidth=1.5)
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot signals
    buy_points = [i for i, s in enumerate(signals) if s == 1]
    sell_points = [i for i, s in enumerate(signals) if s == -1]

    if buy_points:
        ax1.scatter(
            buy_points,
            [prices[i] for i in buy_points],
            color="green",
            s=50,
            marker="^",
            label="Buy Signal",
            zorder=5,
        )
    if sell_points:
        ax1.scatter(
            sell_points,
            [prices[i] for i in sell_points],
            color="red",
            s=50,
            marker="v",
            label="Sell Signal",
            zorder=5,
        )

    # Plot signal strength
    signal_strength = np.array(signals)
    ax2.fill_between(range(len(signals)), signal_strength, alpha=0.6, color="orange")
    ax2.set_ylabel("Signal", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_performance_metrics(metrics: Dict[str, float], save_path: Optional[str] = None) -> None:
    """
    Plot performance metrics in a professional format.

    Args:
        metrics: Dictionary of performance metrics
        save_path: Optional path to save the plot
    """
    # Filter out non-numeric metrics
    numeric_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)
    }

    if not numeric_metrics:
        print("No numeric metrics to plot")
        # Create an empty plot to show the message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No numeric metrics to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Performance Metrics", fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Trading System Performance Metrics", fontsize=16, fontweight="bold")

    # Create subplots for different metric types
    metric_groups = {
        "Returns": ["sharpe_ratio", "total_return", "avg_return"],
        "Risk": ["max_drawdown", "volatility"],
        "Trading": ["win_rate", "profit_factor"],
        "Model": ["auc"],
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    for idx, (group_name, metric_names) in enumerate(metric_groups.items()):
        ax = axes[idx // 2, idx % 2]

        # Get available metrics for this group
        available_metrics = {k: v for k, v in numeric_metrics.items() if k in metric_names}

        if available_metrics:
            bars = ax.bar(
                available_metrics.keys(),
                available_metrics.values(),
                color=colors[idx % len(colors)],
                alpha=0.7,
            )
            ax.set_title(f"{group_name} Metrics", fontweight="bold")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(
                0.5,
                0.5,
                f"No {group_name} metrics available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{group_name} Metrics", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_accuracy(
    scores: List[float], title: str = "Model Accuracy by Fold", save_path: Optional[str] = None
) -> None:
    """
    Plot model accuracy scores across validation folds.

    Args:
        scores: List of accuracy scores
        title: Chart title
        save_path: Optional path to save the plot
    """
    # Filter out non-numeric values
    numeric_scores = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]

    if not numeric_scores:
        print("No numeric scores to plot")
        # Create an empty plot to show the message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No numeric scores to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = range(1, len(numeric_scores) + 1)
    bars = ax.bar(x_pos, numeric_scores, color="#2E86AB", alpha=0.7)

    # Add average line
    avg_score = np.mean(numeric_scores)
    ax.axhline(y=avg_score, color="red", linestyle="--", label=f"Average: {avg_score:.3f}")

    # Add value labels on bars
    for bar, score in zip(bars, numeric_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{score:.3f}", ha="center", va="bottom"
        )

    ax.set_xlabel("Validation Fold", fontsize=12)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_performance_dashboard(
    prices: List[float],
    signals: List[int],
    metrics: Dict[str, float],
    scores: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Create a comprehensive performance dashboard.

    Args:
        prices: List of price values
        signals: List of trading signals
        metrics: Dictionary of performance metrics
        scores: List of model accuracy scores
        save_path: Optional path to save the dashboard
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Price and signals plot (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(prices, label="Price", color="#2E86AB", linewidth=1.5)
    buy_points = [i for i, s in enumerate(signals) if s == 1]
    sell_points = [i for i, s in enumerate(signals) if s == -1]

    if buy_points:
        ax1.scatter(
            buy_points,
            [prices[i] for i in buy_points],
            color="green",
            s=50,
            marker="^",
            label="Buy Signal",
            zorder=5,
        )
    if sell_points:
        ax1.scatter(
            sell_points,
            [prices[i] for i in sell_points],
            color="red",
            s=50,
            marker="v",
            label="Sell Signal",
            zorder=5,
        )

    ax1.set_title("Price Movement and Trading Signals", fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Model accuracy plot (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    x_pos = range(1, len(scores) + 1)
    bars = ax2.bar(x_pos, scores, color="#A23B72", alpha=0.7)
    avg_score = np.mean(scores)
    ax2.axhline(y=avg_score, color="red", linestyle="--", label=f"Avg: {avg_score:.3f}")
    ax2.set_title("Model Accuracy by Fold", fontweight="bold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Performance metrics (bottom, spans all columns)
    ax3 = fig.add_subplot(gs[1:, :])
    numeric_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)
    }

    if numeric_metrics:
        bars = ax3.bar(numeric_metrics.keys(), numeric_metrics.values(), color="#F18F01", alpha=0.7)
        ax3.set_title("Performance Metrics", fontweight="bold")
        ax3.set_ylabel("Value")
        ax3.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    plt.suptitle("BTC Quant Trading System Dashboard", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
