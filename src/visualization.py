"""
Visualization module for BTC Quant trading system.

This module provides simple, professional charts for displaying
trading results and system performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_price_and_signals(prices: List[float], signals: List[int], 
                          title: str = "Trading Strategy", 
                          save_path: Optional[str] = None) -> None:
    """
    Plot price data with buy/sell signals.
    
    Args:
        prices (List[float]): List of prices
        signals (List[int]): List of trading signals (1: buy, -1: sell, 0: hold)
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    if len(prices) != len(signals):
        raise ValueError("Prices and signals must have the same length")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot prices
    ax1.plot(prices, label='Price', alpha=0.7, linewidth=1)
    
    # Plot buy/sell signals
    buy_points = [i for i, s in enumerate(signals) if s == 1]
    sell_points = [i for i, s in enumerate(signals) if s == -1]
    
    if buy_points:
        ax1.scatter(buy_points, [prices[i] for i in buy_points], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    if sell_points:
        ax1.scatter(sell_points, [prices[i] for i in sell_points], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot signal strength
    signal_strength = np.array(signals)
    ax2.fill_between(range(len(signals)), signal_strength, alpha=0.7, 
                     color='blue', label='Signal Strength')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics(metrics: Dict[str, float], 
                           save_path: Optional[str] = None) -> None:
    """
    Plot performance metrics as a bar chart.
    
    Args:
        metrics (Dict[str, float]): Performance metrics
        save_path (Optional[str]): Path to save the plot
    """
    # Filter numeric metrics
    numeric_metrics = {k: v for k, v in metrics.items() 
                      if isinstance(v, (int, float)) and not k.startswith('total_')}
    
    if not numeric_metrics:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    metric_names = list(numeric_metrics.keys())
    metric_values = list(numeric_metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_accuracy(scores: List[float], 
                       title: str = "Model Accuracy", 
                       save_path: Optional[str] = None) -> None:
    """
    Plot model accuracy scores.
    
    Args:
        scores (List[float]): List of accuracy scores
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    if not scores:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    fold_labels = [f'Fold {i+1}' for i in range(len(scores))]
    bars = ax.bar(fold_labels, scores, color='lightcoral', alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add average line
    avg_score = np.mean(scores)
    ax.axhline(y=avg_score, color='red', linestyle='--', alpha=0.8, 
               label=f'Average: {avg_score:.4f}')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_performance_dashboard(strategy_metrics: Dict[str, float], 
                               buy_hold_metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive performance dashboard.
    
    Args:
        strategy_metrics (Dict[str, float]): Strategy performance metrics
        buy_hold_metrics (Dict[str, float]): Buy-and-hold metrics
        save_path (Optional[str]): Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Returns comparison
    returns_data = {
        'Strategy': strategy_metrics.get('total_return', 0),
        'Buy & Hold': buy_hold_metrics.get('total_return', 0)
    }
    ax1.bar(returns_data.keys(), returns_data.values(), color=['skyblue', 'lightgreen'])
    ax1.set_title('Total Returns Comparison')
    ax1.set_ylabel('Return')
    
    # Plot 2: Risk metrics
    risk_metrics = {
        'Sharpe Ratio': strategy_metrics.get('sharpe_ratio', 0),
        'Sortino Ratio': strategy_metrics.get('sortino_ratio', 0),
        'Max Drawdown': strategy_metrics.get('max_drawdown', 0)
    }
    ax2.bar(risk_metrics.keys(), risk_metrics.values(), color='lightcoral')
    ax2.set_title('Risk Metrics')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Trading statistics
    trading_stats = {
        'Win Rate': strategy_metrics.get('win_rate', 0),
        'Profit Factor': strategy_metrics.get('profit_factor', 0)
    }
    ax3.bar(trading_stats.keys(), trading_stats.values(), color='gold')
    ax3.set_title('Trading Statistics')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Volatility comparison
    vol_data = {
        'Strategy': strategy_metrics.get('volatility', 0),
        'Buy & Hold': buy_hold_metrics.get('volatility', 0)
    }
    ax4.bar(vol_data.keys(), vol_data.values(), color=['purple', 'orange'])
    ax4.set_title('Volatility Comparison')
    ax4.set_ylabel('Volatility')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_optimization_convergence(study_results: List[float], 
                                title: str = "Optimization Convergence",
                                save_path: Optional[str] = None) -> None:
    """
    Plot optimization convergence over trials.
    
    Args:
        study_results (List[float]): List of optimization scores
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    if not study_results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    trials = range(1, len(study_results) + 1)
    ax.plot(trials, study_results, marker='o', linewidth=2, markersize=4)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
