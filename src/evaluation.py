"""
Evaluation Module
Performance analysis and visualization for technical indicator alpha project
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime

from config import EVALUATION_CONFIG, PATHS_CONFIG, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Performance evaluation and backtesting for technical indicator alpha strategy
    """

    def __init__(self):
        """Initialize performance evaluator"""
        self.results = {}
        self.strategy_returns = None
        self.benchmark_returns = None

    def calculate_strategy_returns(
        self,
        data: pd.DataFrame,
        predictions: List[int],
        probabilities: List[float],
        threshold: float = None,
    ) -> pd.Series:
        """
        Calculate strategy returns based on model predictions

        Args:
            data: DataFrame with price data
            predictions: Model predictions
            probabilities: Model probabilities
            threshold: Probability threshold for trading signal

        Returns:
            Series with strategy returns
        """
        threshold = threshold or DATA_CONFIG["threshold"]

        # Create signals based on probability threshold
        signals = (np.array(probabilities) > threshold).astype(int)

        # Align data with predictions length
        if len(signals) != len(data):
            # Use only the data that corresponds to predictions
            data_subset = data.iloc[:len(signals)]
        else:
            data_subset = data

        # Calculate forward returns (5-day horizon)
        forward_returns = (
            data_subset["close"].shift(-DATA_CONFIG["target_horizon"]) / data_subset["close"] - 1
        )

        # Strategy returns: long when signal=1, cash when signal=0
        strategy_returns = signals * forward_returns

        # Remove NaN values
        strategy_returns = strategy_returns.dropna()

        return strategy_returns

    def calculate_benchmark_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate buy-and-hold benchmark returns

        Args:
            data: DataFrame with price data

        Returns:
            Series with benchmark returns
        """
        # Buy-and-hold returns
        benchmark_returns = data["close"].pct_change()

        return benchmark_returns.dropna()

    def calculate_performance_metrics(
        self, returns: pd.Series, risk_free_rate: float = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            returns: Series with returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with performance metrics
        """
        risk_free_rate = risk_free_rate or EVALUATION_CONFIG["risk_free_rate"]

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (
            excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        )

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = (returns > 0).mean()

        # Additional metrics
        var_95 = returns.quantile(0.05)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # 95% CVaR

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "num_trades": len(returns),
            "avg_return": returns.mean(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

        return metrics

    def backtest_strategy(
        self,
        data: pd.DataFrame,
        predictions: List[int],
        probabilities: List[float],
        position_sizes: pd.Series = None,
    ) -> Dict:
        """
        Perform comprehensive backtest of the strategy
        Answers the three key questions:
        1. Technical: Does the model have predictive power?
        2. Financial: Does predictive power translate to profit?
        3. Robustness: Is the result real or just luck?

        Args:
            data: DataFrame with price data
            predictions: Model predictions
            probabilities: Model probabilities

        Returns:
            Dictionary with backtest results addressing all three questions
        """
        logger.info("Starting comprehensive strategy backtest...")
        logger.info("=" * 60)
        logger.info("ANSWERING THE THREE KEY QUESTIONS")
        logger.info("=" * 60)

        # Calculate returns
        self.strategy_returns = self.calculate_strategy_returns(
            data, predictions, probabilities
        )
        self.benchmark_returns = self.calculate_benchmark_returns(data)

        # Apply position sizing if provided
        if position_sizes is not None:
            # Align position sizes with strategy returns
            aligned_position_sizes = position_sizes[self.strategy_returns.index]
            self.strategy_returns = self.strategy_returns * aligned_position_sizes
            logger.info("Position sizing applied to strategy returns")

        # Align returns series
        common_index = self.strategy_returns.index.intersection(
            self.benchmark_returns.index
        )
        strategy_aligned = self.strategy_returns.loc[common_index]
        benchmark_aligned = self.benchmark_returns.loc[common_index]

        # QUESTION 1: TECHNICAL - Does the model have predictive power?
        logger.info("\n1. TECHNICAL QUESTION: Does the model have predictive power?")
        technical_analysis = self._analyze_predictive_power(predictions, probabilities, data)
        
        # QUESTION 2: FINANCIAL - Does predictive power translate to profit?
        logger.info("\n2. FINANCIAL QUESTION: Does predictive power translate to profit?")
        strategy_metrics = self.calculate_performance_metrics(strategy_aligned)
        benchmark_metrics = self.calculate_performance_metrics(benchmark_aligned)
        financial_analysis = self._analyze_financial_performance(
            strategy_metrics, benchmark_metrics, strategy_aligned, benchmark_aligned
        )

        # QUESTION 3: ROBUSTNESS - Is the result real or just luck?
        logger.info("\n3. ROBUSTNESS QUESTION: Is the result real or just luck?")
        robustness_analysis = self._analyze_robustness(
            strategy_aligned, benchmark_aligned, probabilities
        )

        # Calculate alpha and beta
        alpha = (
            strategy_metrics["annualized_return"]
            - benchmark_metrics["annualized_return"]
        )
        beta = np.cov(strategy_aligned, benchmark_aligned)[0, 1] / np.var(
            benchmark_aligned
        )

        # Information ratio
        tracking_error = (strategy_aligned - benchmark_aligned).std() * np.sqrt(252)
        information_ratio = (
            (strategy_aligned.mean() - benchmark_aligned.mean())
            / tracking_error
            * np.sqrt(252)
        )

        results = {
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "strategy_returns": strategy_aligned,
            "benchmark_returns": benchmark_aligned,
        }

        self.results = results

        logger.info("Backtest completed")
        logger.info(f"Strategy Sharpe: {strategy_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Benchmark Sharpe: {benchmark_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Alpha: {alpha:.3f}")

        return results

    def _analyze_predictive_power(self, predictions, probabilities, data):
        """
        Analyze predictive power of the model
        """
        logger.info("Analyzing predictive power...")
        
        # Align predictions with data
        target_subset = data['target'].iloc[:len(predictions)].values
        
        # Calculate basic metrics
        accuracy = np.mean(predictions == target_subset)
        
        # Calculate AUC if we have probabilities
        if len(probabilities) > 0:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(target_subset, probabilities)
            except:
                auc = 0.5
        else:
            auc = 0.5
        
        return {
            "accuracy": accuracy,
            "auc": auc,
            "has_predictive_power": auc > 0.5
        }

    def _analyze_financial_performance(self, strategy_metrics, benchmark_metrics, strategy_returns, benchmark_returns):
        """
        Analyze financial performance vs benchmark
        """
        logger.info("Analyzing financial performance...")
        
        # Calculate excess performance
        excess_sharpe = strategy_metrics.get("sharpe_ratio", 0) - benchmark_metrics.get("sharpe_ratio", 0)
        excess_return = strategy_metrics.get("annualized_return", 0) - benchmark_metrics.get("annualized_return", 0)
        
        # Determine if strategy outperforms
        outperforms = (
            strategy_metrics.get("sharpe_ratio", 0) > benchmark_metrics.get("sharpe_ratio", 0) and
            excess_sharpe > 0.1
        )
        
        return {
            "excess_sharpe": excess_sharpe,
            "excess_return": excess_return,
            "outperforms_benchmark": outperforms
        }

    def _analyze_robustness(self, strategy_returns, benchmark_returns, probabilities):
        """
        Analyze robustness of results
        """
        logger.info("Analyzing robustness...")
        
        # Calculate correlation with benchmark
        correlation = strategy_returns.corr(benchmark_returns)
        
        # Calculate hit rate
        positive_days = (strategy_returns > 0).sum()
        total_days = len(strategy_returns)
        hit_rate = positive_days / total_days if total_days > 0 else 0
        
        return {
            "correlation_with_benchmark": correlation,
            "hit_rate": hit_rate,
            "is_robust": hit_rate > 0.5 and correlation < 0.8
        }

    def plot_cumulative_returns(self, save_path: str = None):
        """
        Plot cumulative returns comparison

        Args:
            save_path: Path to save the plot
        """
        try:
            # import matplotlib.pyplot as plt
            # import seaborn as sns

            if self.results is None:
                logger.warning("No backtest results available")
                return

            strategy_returns = self.results["strategy_returns"]
            benchmark_returns = self.results["benchmark_returns"]

            # Calculate cumulative returns
            strategy_cumulative = (1 + strategy_returns).cumprod()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()

            # Create plot
            plt.figure(figsize=(15, 8))

            plt.plot(
                strategy_cumulative.index,
                strategy_cumulative.values,
                label="Strategy",
                linewidth=2,
                color="blue",
            )
            plt.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label="Buy & Hold",
                linewidth=2,
                color="red",
                alpha=0.7,
            )

            plt.title("Cumulative Returns: Strategy vs Buy & Hold", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Cumulative Return", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Cumulative returns plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")

    def plot_drawdown(self, save_path: str = None):
        """
        Plot drawdown analysis

        Args:
            save_path: Path to save the plot
        """
        try:
            # import matplotlib.pyplot as plt

            if self.results is None:
                logger.warning("No backtest results available")
                return

            strategy_returns = self.results["strategy_returns"]

            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max

            # Create plot
            plt.figure(figsize=(15, 8))

            plt.fill_between(
                drawdown.index,
                drawdown.values,
                0,
                alpha=0.3,
                color="red",
                label="Drawdown",
            )
            plt.plot(drawdown.index, drawdown.values, color="red", linewidth=1)

            plt.title("Strategy Drawdown Analysis", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Drawdown", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Drawdown plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")

    def create_performance_summary(self) -> pd.DataFrame:
        """
        Create performance summary table

        Returns:
            DataFrame with performance comparison
        """
        if self.results is None:
            return pd.DataFrame()

        strategy_metrics = self.results["strategy_metrics"]
        benchmark_metrics = self.results["benchmark_metrics"]

        summary_data = {
            "Metric": [
                "Total Return (%)",
                "Annualized Return (%)",
                "Volatility (%)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown (%)",
                "Calmar Ratio",
                "Win Rate (%)",
                "Alpha (%)",
                "Beta",
                "Information Ratio",
            ],
            "Strategy": [
                strategy_metrics["total_return"] * 100,
                strategy_metrics["annualized_return"] * 100,
                strategy_metrics["volatility"] * 100,
                strategy_metrics["sharpe_ratio"],
                strategy_metrics["sortino_ratio"],
                strategy_metrics["max_drawdown"] * 100,
                strategy_metrics["calmar_ratio"],
                strategy_metrics["win_rate"] * 100,
                self.results["alpha"] * 100,
                self.results["beta"],
                self.results["information_ratio"],
            ],
            "Benchmark": [
                benchmark_metrics["total_return"] * 100,
                benchmark_metrics["annualized_return"] * 100,
                benchmark_metrics["volatility"] * 100,
                benchmark_metrics["sharpe_ratio"],
                benchmark_metrics["sortino_ratio"],
                benchmark_metrics["max_drawdown"] * 100,
                benchmark_metrics["calmar_ratio"],
                benchmark_metrics["win_rate"] * 100,
                0,  # No alpha for benchmark
                1,  # Beta = 1 for benchmark
                0,  # No information ratio for benchmark
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df["Difference"] = summary_df["Strategy"] - summary_df["Benchmark"]

        return summary_df

    def save_results(self, filepath: str = None) -> str:
        """
        Save evaluation results to JSON file

        Args:
            filepath: Path to save results (default: results/model_performance.json)

        Returns:
            Path where results were saved
        """
        if filepath is None:
            filepath = os.path.join(
                PATHS_CONFIG["results_dir"], "model_performance.json"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert results to serializable format
        results_to_save = {
            "strategy_metrics": self.results["strategy_metrics"],
            "benchmark_metrics": self.results["benchmark_metrics"],
            "alpha": self.results["alpha"],
            "beta": self.results["beta"],
            "information_ratio": self.results["information_ratio"],
            "tracking_error": self.results["tracking_error"],
            "evaluation_date": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

        return filepath

    def print_summary(self):
        """
        Print performance summary to console
        """
        if self.results is None:
            logger.warning("No results to display")
            return

        strategy_metrics = self.results["strategy_metrics"]
        benchmark_metrics = self.results["benchmark_metrics"]

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"{'Metric':<25} {'Strategy':<15} {'Benchmark':<15} {'Alpha':<10}")
        print("-" * 65)

        metrics = [
            (
                "Total Return (%)",
                strategy_metrics["total_return"] * 100,
                benchmark_metrics["total_return"] * 100,
            ),
            (
                "Annualized Return (%)",
                strategy_metrics["annualized_return"] * 100,
                benchmark_metrics["annualized_return"] * 100,
            ),
            (
                "Sharpe Ratio",
                strategy_metrics["sharpe_ratio"],
                benchmark_metrics["sharpe_ratio"],
            ),
            (
                "Sortino Ratio",
                strategy_metrics["sortino_ratio"],
                benchmark_metrics["sortino_ratio"],
            ),
            (
                "Max Drawdown (%)",
                strategy_metrics["max_drawdown"] * 100,
                benchmark_metrics["max_drawdown"] * 100,
            ),
            (
                "Win Rate (%)",
                strategy_metrics["win_rate"] * 100,
                benchmark_metrics["win_rate"] * 100,
            ),
        ]

        for metric, strategy_val, benchmark_val in metrics:
            alpha = strategy_val - benchmark_val
            print(
                f"{metric:<25} {strategy_val:<15.2f} {benchmark_val:<15.2f} {alpha:<10.2f}"
            )

        print("-" * 65)
        print(f"Alpha: {self.results['alpha']:.3f}")
        print(f"Beta: {self.results['beta']:.3f}")
        print(f"Information Ratio: {self.results['information_ratio']:.3f}")
        print("=" * 60)


def main():
    """Test the performance evaluator"""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    from validation import WalkForwardValidator
    from modeling import XGBoostModel

    # Load and prepare data
    pipeline = DataPipeline()
    data = pipeline.preprocess_data()

    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_all_features(data)
    data_ready = feature_engineer.prepare_features_for_modeling(data_with_features)

    # Perform walk-forward validation
    validator = WalkForwardValidator()
    model = XGBoostModel()

    validation_results = validator.validate_model(
        data=data_ready, model=model, feature_columns=feature_engineer.feature_columns
    )

    # Evaluate performance
    evaluator = PerformanceEvaluator()
    backtest_results = evaluator.backtest_strategy(
        data=data_ready,
        predictions=validation_results["predictions"],
        probabilities=validation_results["probabilities"],
    )

    # Print and save results
    evaluator.print_summary()
    evaluator.save_results()


if __name__ == "__main__":
    main()
