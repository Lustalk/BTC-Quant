"""
Professional Visualization Module
Python/Plotly visualizations for all analysis components
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class ProfessionalVisualizer:
    """
    Professional visualization module with Plotly
    """
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_cumulative_returns_plot(self, data, predictions, probabilities, save_path=None):
        """
        Create cumulative returns plot
        """
        # Calculate strategy returns
        positions = (probabilities > 0.52).astype(int)
        strategy_returns = positions * data['returns']
        
        # Calculate cumulative returns
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_buyhold = (1 + data['returns']).cumprod()
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=cumulative_strategy,
            mode='lines',
            name='Strategy Returns',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=cumulative_buyhold,
            mode='lines',
            name='Buy & Hold',
            line=dict(color=self.colors['secondary'], width=2)
        ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cumulative returns plot saved to: {save_path}")
        
        return fig
    
    def create_drawdown_plot(self, data, predictions, probabilities, save_path=None):
        """
        Create drawdown analysis plot
        """
        # Calculate strategy returns
        positions = (probabilities > 0.52).astype(int)
        strategy_returns = positions * data['returns']
        
        # Calculate drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown %',
            line=dict(color=self.colors['danger'], width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Strategy Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Drawdown plot saved to: {save_path}")
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance, save_path=None, top_n=15):
        """
        Create feature importance plot
        """
        # Get top features
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True
            ),
            error_x=dict(
                type='data',
                array=top_features['importance_std'],
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            height=max(600, top_n * 25),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Feature importance plot saved to: {save_path}")
        
        return fig
    
    def create_probability_distribution_plot(self, probabilities, actual, save_path=None):
        """
        Create probability distribution plot
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Probability Distribution', 'Probability by Actual',
                          'Cumulative Distribution', 'ROC Curve'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Probability distribution
        fig.add_trace(
            go.Histogram(
                x=probabilities,
                nbinsx=50,
                name='Probability Distribution',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # Probability by actual
        actual_0 = probabilities[actual == 0]
        actual_1 = probabilities[actual == 1]
        
        fig.add_trace(
            go.Box(
                y=actual_0,
                name='Actual = 0',
                marker_color=self.colors['danger']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=actual_1,
                name='Actual = 1',
                marker_color=self.colors['success']
            ),
            row=1, col=2
        )
        
        # Cumulative distribution
        sorted_probs = np.sort(probabilities)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_probs,
                y=cumulative,
                mode='lines',
                name='Cumulative Distribution',
                line=dict(color=self.colors['primary'])
            ),
            row=2, col=1
        )
        
        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(actual, probabilities)
        
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name='ROC Curve',
                line=dict(color=self.colors['primary'])
            ),
            row=2, col=2
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color=self.colors['secondary'], dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Probability Distribution Analysis',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Probability distribution plot saved to: {save_path}")
        
        return fig
    
    def create_regime_attribution_plot(self, regime_data, save_path=None):
        """
        Create regime attribution plot
        """
        if regime_data is None or len(regime_data) == 0:
            logger.warning("No regime data available for plotting")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Regime Distribution', 'Performance by Regime',
                          'Regime Timeline', 'Regime Characteristics'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Regime distribution
        regime_counts = regime_data['regime'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                name='Regime Distribution'
            ),
            row=1, col=1
        )
        
        # Performance by regime
        regime_performance = regime_data.groupby('regime').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'volatility': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=regime_performance['regime'],
                y=regime_performance['total_return'],
                name='Average Return',
                marker_color=self.colors['primary']
            ),
            row=1, col=2
        )
        
        # Regime timeline
        fig.add_trace(
            go.Scatter(
                x=regime_data['date'],
                y=regime_data['regime'],
                mode='markers',
                name='Regime Timeline',
                marker=dict(
                    size=8,
                    color=regime_data['regime'],
                    colorscale='Viridis'
                )
            ),
            row=2, col=1
        )
        
        # Regime characteristics (Sharpe vs Volatility)
        fig.add_trace(
            go.Scatter(
                x=regime_data['volatility'],
                y=regime_data['sharpe_ratio'],
                mode='markers',
                name='Sharpe vs Volatility',
                marker=dict(
                    size=8,
                    color=regime_data['regime'],
                    colorscale='Viridis'
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Regime Analysis',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Regime attribution plot saved to: {save_path}")
        
        return fig
    
    def create_risk_return_scatter(self, data, predictions, probabilities, save_path=None):
        """
        Create risk-return scatter plot
        """
        # Calculate rolling metrics
        window = 252
        rolling_metrics = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            window_probs = probabilities.iloc[i-window:i]
            
            # Strategy returns
            positions = (window_probs > 0.52).astype(int)
            strategy_returns = positions * window_data['returns']
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            
            rolling_metrics.append({
                'date': data.index[i-1],
                'return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            })
        
        rolling_df = pd.DataFrame(rolling_metrics)
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rolling_df['volatility'],
            y=rolling_df['return'],
            mode='markers',
            name='Risk-Return',
            marker=dict(
                size=8,
                color=rolling_df['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            text=rolling_df['date'].dt.strftime('%Y-%m-%d'),
            hovertemplate='<b>Date:</b> %{text}<br>' +
                         '<b>Return:</b> %{y:.2%}<br>' +
                         '<b>Volatility:</b> %{x:.2%}<br>' +
                         '<b>Sharpe:</b> %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk-Return Scatter Plot',
            xaxis_title='Volatility (Annualized)',
            yaxis_title='Return',
            template='plotly_white',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Risk-return scatter plot saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, data, predictions, probabilities, 
                                     feature_importance, regime_data=None, save_path=None):
        """
        Create comprehensive dashboard with all visualizations
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Cumulative Returns', 'Drawdown Analysis', 'Feature Importance',
                'Probability Distribution', 'Risk-Return Scatter', 'Regime Analysis',
                'Monthly Returns Heatmap', 'Volatility Analysis', 'Sharpe Ratio Timeline',
                'Win Rate Analysis', 'Correlation Matrix', 'Performance Metrics'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]]
        )
        
        # 1. Cumulative Returns
        positions = (probabilities > 0.52).astype(int)
        strategy_returns = positions * data['returns']
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_buyhold = (1 + data['returns']).cumprod()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=cumulative_strategy, mode='lines', name='Strategy'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=cumulative_buyhold, mode='lines', name='Buy & Hold'),
            row=1, col=1
        )
        
        # 2. Drawdown
        running_max = cumulative_strategy.expanding().max()
        drawdown = (cumulative_strategy - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=data.index, y=drawdown * 100, mode='lines', name='Drawdown %'),
            row=1, col=2
        )
        
        # 3. Feature Importance
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'], orientation='h'),
            row=1, col=3
        )
        
        # 4. Probability Distribution
        fig.add_trace(
            go.Histogram(x=probabilities, nbinsx=50),
            row=2, col=1
        )
        
        # 5. Risk-Return Scatter
        window = 252
        rolling_metrics = []
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            window_probs = probabilities.iloc[i-window:i]
            positions = (window_probs > 0.52).astype(int)
            strategy_returns = positions * window_data['returns']
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            rolling_metrics.append({'return': total_return, 'volatility': volatility})
        
        rolling_df = pd.DataFrame(rolling_metrics)
        fig.add_trace(
            go.Scatter(x=rolling_df['volatility'], y=rolling_df['return'], mode='markers'),
            row=2, col=2
        )
        
        # 6. Regime Analysis (if available)
        if regime_data is not None and len(regime_data) > 0:
            fig.add_trace(
                go.Scatter(x=regime_data['date'], y=regime_data['regime'], mode='markers'),
                row=2, col=3
            )
        
        # 7. Monthly Returns Heatmap
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        fig.add_trace(
            go.Heatmap(z=monthly_returns_pivot.values, x=monthly_returns_pivot.columns, y=monthly_returns_pivot.index),
            row=3, col=1
        )
        
        # 8. Volatility Analysis
        rolling_vol = strategy_returns.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=data.index, y=rolling_vol, mode='lines'),
            row=3, col=2
        )
        
        # 9. Sharpe Ratio Timeline
        rolling_sharpe = self.calculate_rolling_sharpe(strategy_returns)
        fig.add_trace(
            go.Scatter(x=data.index, y=rolling_sharpe, mode='lines'),
            row=3, col=3
        )
        
        # 10. Win Rate Analysis
        rolling_win_rate = strategy_returns.rolling(window=60).apply(lambda x: (x > 0).mean())
        fig.add_trace(
            go.Scatter(x=data.index, y=rolling_win_rate, mode='lines'),
            row=4, col=1
        )
        
        # 11. Correlation Matrix
        # Create correlation matrix for key features
        key_features = ['returns', 'volatility_20d', 'rsi_14', 'macd']
        available_features = [f for f in key_features if f in data.columns]
        if len(available_features) > 1:
            corr_matrix = data[available_features].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index),
                row=4, col=2
            )
        
        # 12. Performance Metrics Table
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_return = (1 + data['returns']).prod() - 1
        sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
        max_drawdown = self.calculate_max_drawdown(strategy_returns)
        
        metrics_table = go.Table(
            header=dict(values=['Metric', 'Strategy', 'Buy & Hold']),
            cells=dict(values=[
                ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                [f'{total_return:.2%}', f'{sharpe_ratio:.2f}', f'{max_drawdown:.2%}', f'{(strategy_returns > 0).mean():.2%}'],
                [f'{buy_hold_return:.2%}', 'N/A', 'N/A', 'N/A']
            ])
        )
        
        fig.add_trace(metrics_table, row=4, col=3)
        
        fig.update_layout(
            title='Comprehensive Trading Strategy Dashboard',
            height=1200,
            template='plotly_white',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Comprehensive dashboard saved to: {save_path}")
        
        return fig
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_rolling_sharpe(self, returns, window=252):
        """
        Calculate rolling Sharpe ratio
        """
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            sharpe = self.calculate_sharpe_ratio(window_returns)
            rolling_sharpe.append(sharpe)
        
        return pd.Series(rolling_sharpe, index=returns.index[window:])
    
    def export_all_visualizations(self, data, predictions, probabilities, 
                                 feature_importance, regime_data=None):
        """
        Export all visualizations as HTML files
        """
        export_dir = 'exports/visualizations'
        os.makedirs(export_dir, exist_ok=True)
        
        # Create all visualizations
        self.create_cumulative_returns_plot(
            data, predictions, probabilities,
            save_path=f'{export_dir}/cumulative_returns.html'
        )
        
        self.create_drawdown_plot(
            data, predictions, probabilities,
            save_path=f'{export_dir}/drawdown_analysis.html'
        )
        
        self.create_feature_importance_plot(
            feature_importance,
            save_path=f'{export_dir}/feature_importance.html'
        )
        
        self.create_probability_distribution_plot(
            probabilities, predictions,
            save_path=f'{export_dir}/probability_distribution.html'
        )
        
        self.create_risk_return_scatter(
            data, predictions, probabilities,
            save_path=f'{export_dir}/risk_return_scatter.html'
        )
        
        if regime_data is not None:
            self.create_regime_attribution_plot(
                regime_data,
                save_path=f'{export_dir}/regime_attribution.html'
            )
        
        # Create comprehensive dashboard
        self.create_comprehensive_dashboard(
            data, predictions, probabilities, feature_importance, regime_data,
            save_path=f'{export_dir}/comprehensive_dashboard.html'
        )
        
        logger.info(f"All visualizations exported to: {export_dir}")
        return export_dir 