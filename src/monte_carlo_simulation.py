"""
Monte Carlo Simulation Module
Statistical significance testing for trading strategy performance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

from config import ADVANCED_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for statistical significance testing
    """

    def __init__(self, n_simulations: int = None):
        """
        Initialize Monte Carlo simulator

        Args:
            n_simulations: Number of simulations to run
        """
        self.n_simulations = n_simulations or ADVANCED_CONFIG["monte_carlo"]["n_simulations"]
        self.simulation_results = []
        self.regime_results = []

    def simulate_strategy_performance(
        self,
        returns: pd.Series,
        probabilities: pd.Series,
        benchmark_returns: pd.Series = None,
        n_simulations: int = None,
    ) -> Dict:
        """
        Run Monte Carlo simulation to test statistical significance

        Args:
            returns: Strategy returns
            probabilities: Model prediction probabilities
            benchmark_returns: Benchmark returns for comparison
            n_simulations: Number of simulations

        Returns:
            Dictionary with simulation results and statistical tests
        """
        n_simulations = n_simulations or self.n_simulations
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")

        # Align returns and probabilities to same length
        min_length = min(len(returns), len(probabilities))
        returns = returns.iloc[:min_length]
        probabilities = probabilities.iloc[:min_length]
        
        logger.info(f"Aligned data: {min_length} observations")
        
        # Calculate actual strategy performance
        actual_sharpe = self._calculate_sharpe_ratio(returns)
        actual_sortino = self._calculate_sortino_ratio(returns)
        actual_max_dd = self._calculate_max_drawdown(returns)

        # Calculate actual excess returns vs benchmark
        if benchmark_returns is not None:
            actual_excess_returns = returns - benchmark_returns
            actual_excess_sharpe = self._calculate_sharpe_ratio(actual_excess_returns)
        else:
            actual_excess_returns = None
            actual_excess_sharpe = None

        # Run simulations
        simulation_sharpes = []
        simulation_sortinos = []
        simulation_max_dds = []
        simulation_excess_sharpes = []

        for i in range(n_simulations):
            # Generate random signals based on actual probability distribution
            random_signals = self._generate_random_signals(probabilities)
            
            # Calculate simulated returns
            simulated_returns = random_signals * returns
            
            # Calculate metrics for this simulation
            sim_sharpe = self._calculate_sharpe_ratio(simulated_returns)
            sim_sortino = self._calculate_sortino_ratio(simulated_returns)
            sim_max_dd = self._calculate_max_drawdown(simulated_returns)
            
            simulation_sharpes.append(sim_sharpe)
            simulation_sortinos.append(sim_sortino)
            simulation_max_dds.append(sim_max_dd)

            # Calculate excess returns if benchmark provided
            if benchmark_returns is not None:
                sim_excess_returns = simulated_returns - benchmark_returns
                sim_excess_sharpe = self._calculate_sharpe_ratio(sim_excess_returns)
                simulation_excess_sharpes.append(sim_excess_sharpe)

        # Calculate statistical significance
        sharpe_p_value = self._calculate_p_value(
            simulation_sharpes, actual_sharpe, "sharpe"
        )
        sortino_p_value = self._calculate_p_value(
            simulation_sortinos, actual_sortino, "sortino"
        )
        max_dd_p_value = self._calculate_p_value(
            simulation_max_dds, actual_max_dd, "max_drawdown"
        )

        if benchmark_returns is not None:
            excess_sharpe_p_value = self._calculate_p_value(
                simulation_excess_sharpes, actual_excess_sharpe, "excess_sharpe"
            )
        else:
            excess_sharpe_p_value = None

        # Store results
        results = {
            "actual_metrics": {
                "sharpe_ratio": actual_sharpe,
                "sortino_ratio": actual_sortino,
                "max_drawdown": actual_max_dd,
                "excess_sharpe": actual_excess_sharpe,
            },
            "simulation_metrics": {
                "sharpe_ratios": simulation_sharpes,
                "sortino_ratios": simulation_sortinos,
                "max_drawdowns": simulation_max_dds,
                "excess_sharpe_ratios": simulation_excess_sharpes,
            },
            "statistical_significance": {
                "sharpe_p_value": sharpe_p_value,
                "sortino_p_value": sortino_p_value,
                "max_dd_p_value": max_dd_p_value,
                "excess_sharpe_p_value": excess_sharpe_p_value,
            },
            "confidence_intervals": {
                "sharpe_ci": self._calculate_confidence_interval(simulation_sharpes),
                "sortino_ci": self._calculate_confidence_interval(simulation_sortinos),
                "max_dd_ci": self._calculate_confidence_interval(simulation_max_dds),
            },
            "simulation_count": n_simulations,
        }

        self.simulation_results.append(results)
        return results

    def _generate_random_signals(self, probabilities: pd.Series) -> np.ndarray:
        """
        Generate random trading signals based on probability distribution

        Args:
            probabilities: Model prediction probabilities

        Returns:
            Array of random signals (0 or 1)
        """
        # Use actual probability distribution to generate random signals
        random_signals = np.random.binomial(1, probabilities.values)
        return random_signals

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Return series

        Returns:
            Maximum drawdown as percentage
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_p_value(
        self, simulation_values: List[float], actual_value: float, metric_name: str
    ) -> float:
        """
        Calculate p-value for statistical significance

        Args:
            simulation_values: List of simulated metric values
            actual_value: Actual metric value
            metric_name: Name of the metric

        Returns:
            P-value (probability of achieving actual value by chance)
        """
        if not simulation_values:
            return 1.0
        
        # Count how many simulations achieved better or equal performance
        if metric_name in ["sharpe", "sortino", "excess_sharpe"]:
            # Higher is better for these metrics
            better_count = sum(1 for x in simulation_values if x >= actual_value)
        else:
            # Lower is better for max drawdown
            better_count = sum(1 for x in simulation_values if x <= actual_value)
        
        p_value = better_count / len(simulation_values)
        return p_value

    def _calculate_confidence_interval(
        self, values: List[float], confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval

        Args:
            values: List of values
            confidence_level: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        
        return (lower_bound, upper_bound)

    def calculate_rolling_metrics(
        self, returns: pd.Series, probabilities: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            returns: Strategy returns
            probabilities: Model probabilities
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        logger.info(f"Calculating rolling metrics with {window}-day window...")

        rolling_data = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_probabilities = probabilities.iloc[i-window:i]
            
            # Calculate metrics for this window
            sharpe = self._calculate_sharpe_ratio(window_returns)
            sortino = self._calculate_sortino_ratio(window_returns)
            max_dd = self._calculate_max_drawdown(window_returns)
            
            # Calculate signal accuracy
            signals = (window_probabilities > 0.5).astype(int)
            accuracy = signals.mean() if len(signals) > 0 else 0.0
            
            rolling_data.append({
                'date': returns.index[i],
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'signal_accuracy': accuracy,
                'window_return': window_returns.sum(),
            })

        return pd.DataFrame(rolling_data).set_index('date')

    def regime_analysis(
        self, returns: pd.Series, probabilities: pd.Series, regime_window: int = 60
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform regime analysis to identify different market conditions

        Args:
            returns: Strategy returns
            probabilities: Model probabilities
            regime_window: Window for regime detection

        Returns:
            Tuple of (regime_data, regime_stats)
        """
        logger.info(f"Performing regime analysis with {regime_window}-day window...")

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=regime_window).std() * np.sqrt(252)
        
        # Define regimes based on volatility
        volatility_median = rolling_vol.median()
        regimes = pd.cut(rolling_vol, 
                        bins=[0, volatility_median * 0.8, volatility_median * 1.2, float('inf')],
                        labels=['Low Volatility', 'Normal Volatility', 'High Volatility'])

        # Calculate performance by regime
        regime_data = pd.DataFrame({
            'returns': returns,
            'probabilities': probabilities,
            'volatility': rolling_vol,
            'regime': regimes
        })

        regime_stats = {}
        for regime in regime_data['regime'].unique():
            if pd.isna(regime):
                continue
                
            regime_returns = regime_data[regime_data['regime'] == regime]['returns']
            regime_probs = regime_data[regime_data['regime'] == regime]['probabilities']
            
            if len(regime_returns) > 0:
                regime_stats[regime] = {
                    'count': len(regime_returns),
                    'sharpe_ratio': self._calculate_sharpe_ratio(regime_returns),
                    'sortino_ratio': self._calculate_sortino_ratio(regime_returns),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns),
                    'avg_probability': regime_probs.mean(),
                    'signal_accuracy': (regime_probs > 0.5).mean(),
                    'total_return': (1 + regime_returns).prod() - 1,
                }

        return regime_data, regime_stats

    def export_simulation_results(self, filepath: str = None) -> str:
        """
        Export simulation results to CSV

        Args:
            filepath: Output file path

        Returns:
            Path to exported file
        """
        if not self.simulation_results:
            logger.warning("No simulation results to export")
            return ""

        filepath = filepath or "exports/monte_carlo_results.csv"
        
        # Flatten results for CSV export
        export_data = []
        for i, result in enumerate(self.simulation_results):
            row = {
                'simulation_id': i,
                'actual_sharpe': result['actual_metrics']['sharpe_ratio'],
                'actual_sortino': result['actual_metrics']['sortino_ratio'],
                'actual_max_dd': result['actual_metrics']['max_drawdown'],
                'sharpe_p_value': result['statistical_significance']['sharpe_p_value'],
                'sortino_p_value': result['statistical_significance']['sortino_p_value'],
                'max_dd_p_value': result['statistical_significance']['max_dd_p_value'],
                'simulation_count': result['simulation_count'],
            }
            
            # Add confidence intervals
            sharpe_ci = result['confidence_intervals']['sharpe_ci']
            row['sharpe_ci_lower'] = sharpe_ci[0]
            row['sharpe_ci_upper'] = sharpe_ci[1]
            
            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Simulation results exported to {filepath}")
        return filepath

    def print_statistical_summary(self):
        """
        Print statistical significance summary
        """
        if not self.simulation_results:
            logger.warning("No simulation results available")
            return

        logger.info("=" * 60)
        logger.info("MONTE CARLO SIMULATION - STATISTICAL SIGNIFICANCE")
        logger.info("=" * 60)

        for i, result in enumerate(self.simulation_results):
            logger.info(f"\nSimulation {i+1}:")
            logger.info(f"  Actual Sharpe Ratio: {result['actual_metrics']['sharpe_ratio']:.3f}")
            logger.info(f"  Sharpe P-Value: {result['statistical_significance']['sharpe_p_value']:.4f}")
            logger.info(f"  Sortino P-Value: {result['statistical_significance']['sortino_p_value']:.4f}")
            logger.info(f"  Max DD P-Value: {result['statistical_significance']['max_dd_p_value']:.4f}")
            
            # Interpret significance
            sharpe_significant = result['statistical_significance']['sharpe_p_value'] < 0.05
            logger.info(f"  Strategy Statistically Significant: {'YES' if sharpe_significant else 'NO'}")


def main():
    """
    Main function for testing Monte Carlo simulation
    """
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    probabilities = pd.Series(np.random.uniform(0.3, 0.7, len(dates)), index=dates)
    
    # Initialize simulator
    simulator = MonteCarloSimulator(n_simulations=100)
    
    # Run simulation
    results = simulator.simulate_strategy_performance(returns, probabilities)
    
    # Print results
    simulator.print_statistical_summary()
    
    # Export results
    simulator.export_simulation_results()


if __name__ == "__main__":
    main() 