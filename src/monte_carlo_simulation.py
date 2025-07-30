"""
Monte Carlo Simulation Module
1000+ bootstraps for statistical significance testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo simulation for statistical significance testing
    """
    
    def __init__(self, n_simulations=1000, random_state=42):
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.simulation_results = []
        self.bootstrap_results = []
        
    def bootstrap_returns(self, returns, n_bootstrap=1000):
        """
        Perform bootstrap resampling of returns
        """
        np.random.seed(self.random_state)
        
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            bootstrap_indices = np.random.choice(
                len(returns), size=len(returns), replace=True
            )
            bootstrap_sample = returns.iloc[bootstrap_indices]
            bootstrap_samples.append(bootstrap_sample)
        
        return bootstrap_samples
    
    def simulate_strategy_performance(self, returns, probabilities, threshold=0.52):
        """
        Simulate strategy performance with Monte Carlo
        """
        logger.info(f"Starting Monte Carlo simulation with {self.n_simulations} iterations...")
        
        simulation_results = []
        
        for i in range(self.n_simulations):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(
                len(returns), size=len(returns), replace=True
            )
            
            bootstrap_returns = returns.iloc[bootstrap_indices]
            bootstrap_probs = probabilities.iloc[bootstrap_indices]
            
            # Calculate strategy returns
            positions = (bootstrap_probs > threshold).astype(int)
            strategy_returns = positions * bootstrap_returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            buy_hold_return = (1 + bootstrap_returns).prod() - 1
            excess_return = total_return - buy_hold_return
            
            # Risk metrics
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self.calculate_max_drawdown(strategy_returns)
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # Win rate
            win_rate = (strategy_returns > 0).mean()
            
            result = {
                'simulation': i + 1,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': excess_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'alpha': excess_return,
                'beta': self.calculate_beta(strategy_returns, bootstrap_returns)
            }
            
            simulation_results.append(result)
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{self.n_simulations} simulations...")
        
        self.simulation_results = simulation_results
        logger.info("Monte Carlo simulation completed")
        
        return simulation_results
    
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
    
    def calculate_beta(self, strategy_returns, market_returns):
        """
        Calculate beta relative to market
        """
        if market_returns.std() == 0:
            return 0
        return np.cov(strategy_returns, market_returns)[0, 1] / market_returns.var()
    
    def statistical_significance_test(self, strategy_returns, buy_hold_returns):
        """
        Test statistical significance of excess returns
        """
        excess_returns = strategy_returns - buy_hold_returns
        
        # T-test for mean difference
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Bootstrap confidence intervals
        bootstrap_samples = self.bootstrap_returns(pd.Series(excess_returns))
        bootstrap_means = [sample.mean() for sample in bootstrap_samples]
        
        ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
        ci_99 = np.percentile(bootstrap_means, [0.5, 99.5])
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_95': p_value < 0.05,
            'significant_99': p_value < 0.01,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'ci_99_lower': ci_99[0],
            'ci_99_upper': ci_99[1],
            'mean_excess_return': excess_returns.mean(),
            'std_excess_return': excess_returns.std()
        }
    
    def rolling_metrics_analysis(self, returns, probabilities, window=252):
        """
        Calculate rolling metrics for regime analysis
        """
        logger.info("Calculating rolling metrics...")
        
        rolling_metrics = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_probs = probabilities.iloc[i-window:i]
            
            # Strategy returns
            positions = (window_probs > 0.52).astype(int)
            strategy_returns = positions * window_returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            buy_hold_return = (1 + window_returns).prod() - 1
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self.calculate_max_drawdown(strategy_returns)
            volatility = strategy_returns.std() * np.sqrt(252)
            
            rolling_metrics.append({
                'date': returns.index[i-1],
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': (strategy_returns > 0).mean()
            })
        
        return pd.DataFrame(rolling_metrics)
    
    def regime_analysis(self, returns, probabilities, n_regimes=3):
        """
        Perform regime analysis using clustering
        """
        logger.info("Performing regime analysis...")
        
        # Calculate rolling metrics
        rolling_df = self.rolling_metrics_analysis(returns, probabilities)
        
        # Features for regime detection
        features = ['sharpe_ratio', 'volatility', 'max_drawdown', 'win_rate']
        X = rolling_df[features].values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering for regime detection
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_regimes, random_state=self.random_state)
        regimes = kmeans.fit_predict(X_scaled)
        
        # Add regime information
        rolling_df['regime'] = regimes
        
        # Calculate regime statistics
        regime_stats = []
        for regime in range(n_regimes):
            regime_data = rolling_df[rolling_df['regime'] == regime]
            
            stats = {
                'regime': regime,
                'count': len(regime_data),
                'avg_sharpe': regime_data['sharpe_ratio'].mean(),
                'avg_volatility': regime_data['volatility'].mean(),
                'avg_return': regime_data['total_return'].mean(),
                'avg_excess_return': regime_data['excess_return'].mean(),
                'avg_win_rate': regime_data['win_rate'].mean(),
                'start_date': regime_data['date'].min(),
                'end_date': regime_data['date'].max()
            }
            regime_stats.append(stats)
        
        return rolling_df, pd.DataFrame(regime_stats)
    
    def export_simulation_results(self):
        """
        Export Monte Carlo simulation results
        """
        if not self.simulation_results:
            logger.warning("No simulation results available for export")
            return None
        
        # Export simulation results
        sim_df = pd.DataFrame(self.simulation_results)
        export_path = 'exports/monte_carlo_results.csv'
        sim_df.to_csv(export_path, index=False)
        logger.info(f"Monte Carlo results exported to: {export_path}")
        
        # Calculate summary statistics
        summary_stats = {
            'n_simulations': len(sim_df),
            'mean_total_return': sim_df['total_return'].mean(),
            'std_total_return': sim_df['total_return'].std(),
            'mean_excess_return': sim_df['excess_return'].mean(),
            'std_excess_return': sim_df['excess_return'].std(),
            'mean_sharpe': sim_df['sharpe_ratio'].mean(),
            'std_sharpe': sim_df['sharpe_ratio'].std(),
            'mean_max_drawdown': sim_df['max_drawdown'].mean(),
            'std_max_drawdown': sim_df['max_drawdown'].std(),
            'mean_win_rate': sim_df['win_rate'].mean(),
            'std_win_rate': sim_df['win_rate'].std(),
            'positive_excess_return_pct': (sim_df['excess_return'] > 0).mean(),
            'positive_sharpe_pct': (sim_df['sharpe_ratio'] > 0).mean(),
            'ci_95_lower_return': np.percentile(sim_df['total_return'], 2.5),
            'ci_95_upper_return': np.percentile(sim_df['total_return'], 97.5),
            'ci_99_lower_return': np.percentile(sim_df['total_return'], 0.5),
            'ci_99_upper_return': np.percentile(sim_df['total_return'], 99.5),
        }
        
        # Export summary statistics
        summary_df = pd.DataFrame([summary_stats])
        summary_path = 'exports/simulation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Simulation summary exported to: {summary_path}")
        
        return export_path, summary_path
    
    def plot_simulation_results(self, save_path=None):
        """
        Plot Monte Carlo simulation results
        """
        if not self.simulation_results:
            logger.warning("No simulation results available for plotting")
            return None
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            sim_df = pd.DataFrame(self.simulation_results)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Return Distribution', 'Sharpe Ratio Distribution',
                              'Excess Return Distribution', 'Max Drawdown Distribution',
                              'Win Rate Distribution', 'Cumulative Returns'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # Return distribution
            fig.add_trace(
                go.Histogram(
                    x=sim_df['total_return'],
                    name='Total Return',
                    nbinsx=50
                ),
                row=1, col=1
            )
            
            # Sharpe ratio distribution
            fig.add_trace(
                go.Histogram(
                    x=sim_df['sharpe_ratio'],
                    name='Sharpe Ratio',
                    nbinsx=50
                ),
                row=1, col=2
            )
            
            # Excess return distribution
            fig.add_trace(
                go.Histogram(
                    x=sim_df['excess_return'],
                    name='Excess Return',
                    nbinsx=50
                ),
                row=2, col=1
            )
            
            # Max drawdown distribution
            fig.add_trace(
                go.Histogram(
                    x=sim_df['max_drawdown'],
                    name='Max Drawdown',
                    nbinsx=50
                ),
                row=2, col=2
            )
            
            # Win rate distribution
            fig.add_trace(
                go.Histogram(
                    x=sim_df['win_rate'],
                    name='Win Rate',
                    nbinsx=50
                ),
                row=3, col=1
            )
            
            # Cumulative returns (sorted)
            sorted_returns = sim_df['total_return'].sort_values()
            cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
            
            fig.add_trace(
                go.Scatter(
                    x=sorted_returns,
                    y=cumulative_prob,
                    mode='lines',
                    name='Cumulative Probability'
                ),
                row=3, col=2
            )
            
            fig.update_layout(
                title='Monte Carlo Simulation Results',
                height=1000,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Simulation plots saved to: {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available. Using matplotlib instead.")
            
            # Fallback to matplotlib
            sim_df = pd.DataFrame(self.simulation_results)
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            # Return distribution
            axes[0, 0].hist(sim_df['total_return'], bins=50, alpha=0.7)
            axes[0, 0].set_title('Return Distribution')
            axes[0, 0].set_xlabel('Total Return')
            
            # Sharpe ratio distribution
            axes[0, 1].hist(sim_df['sharpe_ratio'], bins=50, alpha=0.7)
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            
            # Excess return distribution
            axes[1, 0].hist(sim_df['excess_return'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Excess Return Distribution')
            axes[1, 0].set_xlabel('Excess Return')
            
            # Max drawdown distribution
            axes[1, 1].hist(sim_df['max_drawdown'], bins=50, alpha=0.7)
            axes[1, 1].set_title('Max Drawdown Distribution')
            axes[1, 1].set_xlabel('Max Drawdown')
            
            # Win rate distribution
            axes[2, 0].hist(sim_df['win_rate'], bins=50, alpha=0.7)
            axes[2, 0].set_title('Win Rate Distribution')
            axes[2, 0].set_xlabel('Win Rate')
            
            # Cumulative returns
            sorted_returns = sim_df['total_return'].sort_values()
            cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
            axes[2, 1].plot(sorted_returns, cumulative_prob)
            axes[2, 1].set_title('Cumulative Return Distribution')
            axes[2, 1].set_xlabel('Total Return')
            axes[2, 1].set_ylabel('Cumulative Probability')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
                logger.info(f"Simulation plots saved to: {save_path.replace('.html', '.png')}")
            
            return fig
    
    def calculate_rolling_metrics(self, returns, probabilities, window=252):
        """
        Calculate rolling metrics for export
        """
        logger.info("Calculating rolling metrics...")
        
        rolling_metrics = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_probs = probabilities.iloc[i-window:i]
            
            # Strategy returns
            positions = (window_probs > 0.52).astype(int)
            strategy_returns = positions * window_returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            buy_hold_return = (1 + window_returns).prod() - 1
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self.calculate_max_drawdown(strategy_returns)
            volatility = strategy_returns.std() * np.sqrt(252)
            
            rolling_metrics.append({
                'date': returns.index[i-1],
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': (strategy_returns > 0).mean()
            })
        
        rolling_df = pd.DataFrame(rolling_metrics)
        
        # Export rolling metrics
        export_path = 'exports/rolling_metrics.csv'
        rolling_df.to_csv(export_path, index=False)
        logger.info(f"Rolling metrics exported to: {export_path}")
        
        return rolling_df 