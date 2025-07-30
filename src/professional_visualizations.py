"""
Professional Visualizations Module
Interactive dashboards and charts for trading strategy analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalVisualizer:
    """
    Professional visualization tools for trading strategy analysis
    """

    def __init__(self):
        """Initialize professional visualizer"""
        self.colors = {
            'strategy': '#1f77b4',
            'benchmark': '#ff7f0e',
            'excess': '#2ca02c',
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#7f7f7f'
        }

    def create_cumulative_returns_plot(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        probabilities: pd.Series,
        save_path: str = None
    ) -> go.Figure:
        """
        Create cumulative returns comparison plot

        Args:
            data: DataFrame with price data
            predictions: Model predictions
            probabilities: Model probabilities
            save_path: Path to save the plot

        Returns:
            Plotly figure object
        """
        # Calculate strategy returns
        signals = (probabilities > 0.5).astype(int)
        strategy_returns = signals * data['returns']
        buy_hold_returns = data['returns']

        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        excess_cumulative = strategy_cumulative / buy_hold_cumulative

        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative Returns', 'Excess Returns', 'Strategy Signals'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.3, 0.2]
        )

        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=strategy_cumulative,
                name='Strategy',
                line=dict(color=self.colors['strategy'], width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_cumulative,
                name='Buy & Hold',
                line=dict(color=self.colors['benchmark'], width=2)
            ),
            row=1, col=1
        )

        # Excess returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=excess_cumulative,
                name='Excess Returns',
                line=dict(color=self.colors['excess'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )

        # Add horizontal line at 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

        # Strategy signals
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signals,
                name='Long Signal',
                mode='markers',
                marker=dict(
                    color=np.where(signals == 1, self.colors['positive'], self.colors['neutral']),
                    size=8
                )
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title='Trading Strategy Performance Analysis',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Excess Return", row=2, col=1)
        fig.update_yaxes(title_text="Signal", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cumulative returns plot saved to {save_path}")

        return fig

    def create_feature_importance_plot(
        self,
        feature_importance: pd.DataFrame,
        save_path: str = None
    ) -> go.Figure:
        """
        Create feature importance visualization

        Args:
            feature_importance: DataFrame with feature importance scores
            save_path: Path to save the plot

        Returns:
            Plotly figure object
        """
        # Prepare data
        top_features = feature_importance.head(15)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Feature Importance (Combined Score)', 'Random Forest Importance'),
            vertical_spacing=0.15
        )

        # Combined score
        fig.add_trace(
            go.Bar(
                x=top_features['combined_score'],
                y=top_features['feature'],
                orientation='h',
                name='Combined Score',
                marker_color=self.colors['strategy']
            ),
            row=1, col=1
        )

        # Random Forest importance
        fig.add_trace(
            go.Bar(
                x=top_features['rf_importance'],
                y=top_features['feature'],
                orientation='h',
                name='RF Importance',
                marker_color=self.colors['benchmark']
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Feature Importance Analysis',
            height=800,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_xaxes(title_text="Importance Score", row=2, col=1)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def create_risk_return_scatter(
        self,
        returns: pd.Series,
        probabilities: pd.Series,
        save_path: str = None
    ) -> go.Figure:
        """
        Create risk-return scatter plot

        Args:
            returns: Strategy returns
            probabilities: Model probabilities
            save_path: Path to save the plot

        Returns:
            Plotly figure object
        """
        # Calculate metrics for different probability thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        results = []

        for threshold in thresholds:
            signals = (probabilities > threshold).astype(int)
            strategy_returns = signals * returns
            
            if strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std()
                win_rate = (strategy_returns > 0).mean()
                total_return = (1 + strategy_returns).prod() - 1
                
                results.append({
                    'threshold': threshold,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'volatility': strategy_returns.std(),
                    'mean_return': strategy_returns.mean()
                })

        results_df = pd.DataFrame(results)

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results_df['volatility'],
                y=results_df['mean_return'],
                mode='markers+text',
                text=results_df['threshold'].round(2),
                textposition="top center",
                marker=dict(
                    size=results_df['sharpe_ratio'] * 20,
                    color=results_df['sharpe_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Strategy Performance'
            )
        )

        # Add efficient frontier line
        sorted_df = results_df.sort_values('volatility')
        fig.add_trace(
            go.Scatter(
                x=sorted_df['volatility'],
                y=sorted_df['mean_return'],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Efficient Frontier'
            )
        )

        # Update layout
        fig.update_layout(
            title='Risk-Return Analysis by Probability Threshold',
            xaxis_title='Volatility',
            yaxis_title='Mean Return',
            height=600
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Risk-return scatter plot saved to {save_path}")

        return fig

    def create_performance_dashboard(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        probabilities: pd.Series,
        feature_importance: pd.DataFrame,
        save_path: str = None
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard

        Args:
            data: DataFrame with price data
            predictions: Model predictions
            probabilities: Model probabilities
            feature_importance: Feature importance DataFrame
            save_path: Path to save the plot

        Returns:
            Plotly figure object
        """
        # Calculate performance metrics
        signals = (probabilities > 0.5).astype(int)
        strategy_returns = signals * data['returns']
        buy_hold_returns = data['returns']

        # Cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        excess_cumulative = strategy_cumulative / buy_hold_cumulative

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Excess Returns',
                'Feature Importance', 'Returns Distribution',
                'Strategy Signals', 'Performance Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=strategy_cumulative,
                name='Strategy',
                line=dict(color=self.colors['strategy'])
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_cumulative,
                name='Buy & Hold',
                line=dict(color=self.colors['benchmark'])
            ),
            row=1, col=1
        )

        # 2. Excess returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=excess_cumulative,
                name='Excess',
                line=dict(color=self.colors['excess'])
            ),
            row=1, col=2
        )

        # 3. Feature importance
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(
                x=top_features['combined_score'],
                y=top_features['feature'],
                orientation='h',
                marker_color=self.colors['strategy']
            ),
            row=2, col=1
        )

        # 4. Returns distribution
        fig.add_trace(
            go.Histogram(
                x=strategy_returns,
                name='Strategy Returns',
                marker_color=self.colors['strategy'],
                opacity=0.7
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Histogram(
                x=buy_hold_returns,
                name='Buy & Hold Returns',
                marker_color=self.colors['benchmark'],
                opacity=0.7
            ),
            row=2, col=2
        )

        # 5. Strategy signals
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signals,
                mode='markers',
                marker=dict(
                    color=np.where(signals == 1, self.colors['positive'], self.colors['neutral']),
                    size=6
                ),
                name='Signals'
            ),
            row=3, col=1
        )

        # 6. Performance metrics table
        metrics = self._calculate_performance_metrics(strategy_returns, buy_hold_returns)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Strategy', 'Buy & Hold', 'Excess']),
                cells=dict(values=[
                    list(metrics.keys()),
                    [f"{v:.4f}" for v in metrics.values()],
                    [f"{v:.4f}" for v in metrics.values()],
                    [f"{v:.4f}" for v in metrics.values()]
                ])
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title='Comprehensive Trading Strategy Dashboard',
            height=1000,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Performance dashboard saved to {save_path}")

        return fig

    def _calculate_performance_metrics(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict:
        """
        Calculate key performance metrics

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary with performance metrics
        """
        # Basic metrics
        strategy_mean = strategy_returns.mean()
        strategy_std = strategy_returns.std()
        benchmark_mean = benchmark_returns.mean()
        benchmark_std = benchmark_returns.std()

        # Sharpe ratios
        strategy_sharpe = strategy_mean / strategy_std if strategy_std > 0 else 0
        benchmark_sharpe = benchmark_mean / benchmark_std if benchmark_std > 0 else 0

        # Maximum drawdown
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_drawdown = (strategy_cumulative / strategy_cumulative.expanding().max() - 1).min()
        
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_drawdown = (benchmark_cumulative / benchmark_cumulative.expanding().max() - 1).min()

        # Win rate
        strategy_win_rate = (strategy_returns > 0).mean()
        benchmark_win_rate = (benchmark_returns > 0).mean()

        return {
            'Total Return': (1 + strategy_returns).prod() - 1,
            'Sharpe Ratio': strategy_sharpe,
            'Max Drawdown': strategy_drawdown,
            'Win Rate': strategy_win_rate,
            'Volatility': strategy_std,
            'Excess Return': strategy_mean - benchmark_mean
        }

    def export_all_visualizations(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        probabilities: pd.Series,
        feature_importance: pd.DataFrame,
        regime_data: pd.DataFrame = None,
        save_dir: str = "exports/visualizations"
    ) -> str:
        """
        Export all visualizations

        Args:
            data: DataFrame with price data
            predictions: Model predictions
            probabilities: Model probabilities
            feature_importance: Feature importance DataFrame
            regime_data: Regime analysis data
            save_dir: Directory to save visualizations

        Returns:
            Path to saved directory
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Create all visualizations
        self.create_cumulative_returns_plot(
            data, predictions, probabilities,
            save_path=f"{save_dir}/cumulative_returns.html"
        )

        self.create_feature_importance_plot(
            feature_importance,
            save_path=f"{save_dir}/feature_importance.html"
        )

        self.create_risk_return_scatter(
            data['returns'], probabilities,
            save_path=f"{save_dir}/risk_return_scatter.html"
        )

        self.create_performance_dashboard(
            data, predictions, probabilities, feature_importance,
            save_path=f"{save_dir}/comprehensive_dashboard.html"
        )

        logger.info(f"All visualizations exported to {save_dir}")
        return save_dir

    def create_regime_analysis_plot(
        self,
        regime_data: pd.DataFrame,
        save_path: str = None
    ) -> go.Figure:
        """
        Create regime analysis visualization

        Args:
            regime_data: Regime analysis data
            save_path: Path to save the plot

        Returns:
            Plotly figure object
        """
        if regime_data is None or len(regime_data) == 0:
            logger.warning("No regime data provided")
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Regime Distribution', 'Performance by Regime',
                'Regime Transitions', 'Regime Duration'
            )
        )

        # 1. Regime distribution
        regime_counts = regime_data['regime'].value_counts()
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                name='Regime Count',
                marker_color=self.colors['strategy']
            ),
            row=1, col=1
        )

        # 2. Performance by regime
        regime_performance = regime_data.groupby('regime')['returns'].agg(['mean', 'std']).reset_index()
        fig.add_trace(
            go.Scatter(
                x=regime_performance['mean'],
                y=regime_performance['std'],
                mode='markers+text',
                text=regime_performance['regime'],
                textposition="top center",
                marker=dict(size=15),
                name='Regime Performance'
            ),
            row=1, col=2
        )

        # 3. Regime transitions (if available)
        if 'regime_change' in regime_data.columns:
            transitions = regime_data['regime_change'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=transitions.index,
                    y=transitions.values,
                    name='Transitions',
                    marker_color=self.colors['benchmark']
                ),
                row=2, col=1
            )

        # 4. Regime duration (if available)
        if 'regime_duration' in regime_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=regime_data['regime_duration'],
                    name='Duration',
                    marker_color=self.colors['excess']
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title='Regime Analysis',
            height=800,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Regime analysis plot saved to {save_path}")

        return fig


def main():
    """Test professional visualizations"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    returns = np.random.normal(0.001, 0.02, n_samples)
    probabilities = np.random.beta(2, 2, n_samples)
    predictions = (probabilities > 0.5).astype(int)
    
    data = pd.DataFrame({
        'returns': returns,
        'close': 100 * (1 + pd.Series(returns)).cumprod()
    }, index=dates)
    
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(10)],
        'combined_score': np.random.rand(10),
        'rf_importance': np.random.rand(10)
    })
    
    # Create visualizations
    visualizer = ProfessionalVisualizer()
    
    # Test individual plots
    visualizer.create_cumulative_returns_plot(
        data, predictions, probabilities,
        save_path="test_cumulative_returns.html"
    )
    
    visualizer.create_feature_importance_plot(
        feature_importance,
        save_path="test_feature_importance.html"
    )
    
    print("Professional visualizations test completed!")


if __name__ == "__main__":
    main() 