"""
Parameter Optimization Module for BTC Quant Trading System.

This module uses Optuna for intelligent hyperparameter optimization of:
1. Technical indicator parameters (RSI window, SMA periods, etc.)
2. Take-profit and stop-loss levels
3. ML model hyperparameters
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from .evaluation import calculate_all_metrics
from .strategy_analysis import analyze_strategy_performance


class ParameterOptimizer:
    """Smart parameter optimizer for trading system."""

    def __init__(self, data: pd.DataFrame, n_trials: int = 100):
        """
        Initialize the parameter optimizer.

        Args:
            data: Raw OHLCV data
            n_trials: Number of optimization trials
        """
        self.data = data.copy()
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -np.inf

    def create_indicators_with_params(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Create technical indicators with given parameters.

        Args:
            params: Dictionary of indicator parameters

        Returns:
            DataFrame with indicators
        """
        df = self.data.copy()

        # Trend Indicators with variable parameters
        df[f"SMA_{params['sma_short']}"] = ta.trend.sma_indicator(
            df["Close"], window=params["sma_short"]
        )
        df[f"SMA_{params['sma_long']}"] = ta.trend.sma_indicator(
            df["Close"], window=params["sma_long"]
        )
        df[f"EMA_{params['ema_short']}"] = ta.trend.ema_indicator(
            df["Close"], window=params["ema_short"]
        )
        df[f"EMA_{params['ema_long']}"] = ta.trend.ema_indicator(
            df["Close"], window=params["ema_long"]
        )

        # MACD with variable parameters
        df["MACD"] = ta.trend.macd(
            df["Close"],
            window_fast=params["macd_fast"],
            window_slow=params["macd_slow"],
        )
        df["MACD_Signal"] = ta.trend.macd_signal(
            df["Close"],
            window_fast=params["macd_fast"],
            window_slow=params["macd_slow"],
        )
        df["MACD_Histogram"] = ta.trend.macd_diff(
            df["Close"],
            window_fast=params["macd_fast"],
            window_slow=params["macd_slow"],
        )

        # Momentum Indicators
        df[f"RSI_{params['rsi_window']}"] = ta.momentum.rsi(
            df["Close"], window=params["rsi_window"]
        )
        df["Stoch_K"] = ta.momentum.stoch(
            df["High"], df["Low"], df["Close"], window=params["stoch_window"]
        )
        df["Stoch_D"] = ta.momentum.stoch_signal(
            df["High"], df["Low"], df["Close"], window=params["stoch_window"]
        )
        df["Williams_R"] = ta.momentum.williams_r(
            df["High"], df["Low"], df["Close"], lbp=params["williams_window"]
        )

        # Volatility Indicators
        df["BB_Upper"] = ta.volatility.bollinger_hband(
            df["Close"], window=params["bb_window"]
        )
        df["BB_Lower"] = ta.volatility.bollinger_lband(
            df["Close"], window=params["bb_window"]
        )
        df["BB_Middle"] = ta.volatility.bollinger_mavg(
            df["Close"], window=params["bb_window"]
        )
        df["BB_Width"] = ta.volatility.bollinger_wband(
            df["Close"], window=params["bb_window"]
        )
        df["ATR"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=params["atr_window"]
        )

        # Volume Indicators
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            df["High"], df["Low"], df["Close"], df["Volume"]
        )

        # Price-based features
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High_Low_Ratio"] = df["High"] / df["Low"]
        df["Close_Open_Ratio"] = df["Close"] / df["Open"]

        # Lagged features with variable lags
        for lag in [params["lag_1"], params["lag_2"], params["lag_3"]]:
            df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
            df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)

        # Rolling statistics with variable windows
        for window in [
            params["roll_short"],
            params["roll_medium"],
            params["roll_long"],
        ]:
            df[f"Close_Std_{window}"] = df["Close"].rolling(window=window).std()
            df[f"Volume_Mean_{window}"] = df["Volume"].rolling(window=window).mean()
            df[f"Returns_Mean_{window}"] = df["Returns"].rolling(window=window).mean()

        return df.dropna()

    def generate_signals_with_tp_sl(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[List[int], List[float]]:
        """
        Generate trading signals with take-profit and stop-loss optimization.

        Args:
            df: DataFrame with indicators
            params: Dictionary containing TP/SL parameters

        Returns:
            Tuple of signals and entry prices
        """
        signals = []
        entry_prices = []
        position = 0
        entry_price = 0

        for i, row in df.iterrows():
            current_price = row["Close"]
            rsi = row.get(f"RSI_{params['rsi_window']}", 50)
            sma_short = row.get(f"SMA_{params['sma_short']}", current_price)
            sma_long = row.get(f"SMA_{params['sma_long']}", current_price)
            ema_short = row.get(f"EMA_{params['ema_short']}", current_price)
            ema_long = row.get(f"EMA_{params['ema_long']}", current_price)
            macd = row.get("MACD", 0)
            macd_signal = row.get("MACD_Signal", 0)
            stoch_k = row.get("Stoch_K", 50)
            stoch_d = row.get("Stoch_D", 50)

            signal = 0  # Hold

            # Entry conditions
            if position == 0:  # No position
                # Enhanced buy conditions with multiple confirmations
                rsi_oversold = rsi < params["rsi_oversold"]
                trend_up = current_price > sma_short > sma_long
                ema_trend_up = ema_short > ema_long
                macd_bullish = macd > macd_signal
                stoch_oversold = stoch_k < 20 and stoch_d < 20

                # More flexible buy conditions - any combination of 2+ conditions
                conditions_met = sum(
                    [rsi_oversold, trend_up, ema_trend_up, macd_bullish, stoch_oversold]
                )

                # Buy if at least 2 conditions are met
                if conditions_met >= 2:
                    signal = 1  # Buy
                    position = 1
                    entry_price = current_price
                # Fallback: simple RSI strategy if no complex conditions met
                elif rsi < 35:  # Very oversold
                    signal = 1  # Buy
                    position = 1
                    entry_price = current_price

            elif position == 1:  # Long position
                # Take profit
                if current_price >= entry_price * (1 + params["take_profit"]):
                    signal = -1  # Sell (take profit)
                    position = 0
                    entry_price = 0
                # Stop loss
                elif current_price <= entry_price * (1 - params["stop_loss"]):
                    signal = -1  # Sell (stop loss)
                    position = 0
                    entry_price = 0
                # RSI overbought exit
                elif rsi > params["rsi_overbought"]:
                    signal = -1  # Sell
                    position = 0
                    entry_price = 0
                # MACD bearish crossover
                elif macd < macd_signal and macd < 0:
                    signal = -1  # Sell
                    position = 0
                    entry_price = 0
                # Stochastic overbought
                elif stoch_k > 80 and stoch_d > 80:
                    signal = -1  # Sell
                    position = 0
                    entry_price = 0

            signals.append(signal)
            entry_prices.append(entry_price)

        return signals, entry_prices

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            
        Returns:
            Dict[str, Any]: Suggested parameters
        """
        return {
            # Technical Indicator Parameters
            'sma_short': trial.suggest_int('sma_short', 5, 25),
            'sma_long': trial.suggest_int('sma_long', 30, 100),
            'ema_short': trial.suggest_int('ema_short', 8, 20),
            'ema_long': trial.suggest_int('ema_long', 20, 50),
            'rsi_window': trial.suggest_int('rsi_window', 10, 21),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 35),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 80),
            'stoch_window': trial.suggest_int('stoch_window', 10, 20),
            'williams_window': trial.suggest_int('williams_window', 10, 20),
            'bb_window': trial.suggest_int('bb_window', 15, 25),
            'atr_window': trial.suggest_int('atr_window', 10, 20),
            'macd_fast': trial.suggest_int('macd_fast', 8, 15),
            'macd_slow': trial.suggest_int('macd_slow', 20, 30),
            
            # Risk Management Parameters
            'take_profit': trial.suggest_float('take_profit', 0.02, 0.15),
            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.10),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.02, 0.08),
            
            # Position Sizing Parameters
            'position_sizing_strategy': trial.suggest_categorical('position_sizing_strategy', ['fixed', 'volatility_targeted']),
            'target_volatility': trial.suggest_float('target_volatility', 0.10, 0.30),
            
            # ML Model Parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            
            # Feature Engineering Parameters
            'lag_1': trial.suggest_int('lag_1', 1, 3),
            'lag_2': trial.suggest_int('lag_2', 4, 7),
            'lag_3': trial.suggest_int('lag_3', 8, 15),
            'roll_short': trial.suggest_int('roll_short', 5, 15),
            'roll_medium': trial.suggest_int('roll_medium', 16, 30),
            'roll_long': trial.suggest_int('roll_long', 31, 60),
            
            # Transaction Cost Parameters
            'fee_type': trial.suggest_categorical('fee_type', ['percentage', 'fixed'])
        }

    def _calculate_optimization_score(self, strategy_metrics: Dict[str, float], params: Dict[str, Any]) -> float:
        """
        Calculate optimization score from strategy metrics.
        
        Args:
            strategy_metrics: Strategy performance metrics
            params: Parameters used
            
        Returns:
            float: Optimization score
        """
        # Multi-objective optimization score
        sharpe = strategy_metrics.get('sharpe_ratio', 0)
        sortino = strategy_metrics.get('sortino_ratio', 0)
        max_dd = strategy_metrics.get('max_drawdown', -1)
        win_rate = strategy_metrics.get('win_rate', 0)
        
        # Combine metrics with penalties for high risk
        score = (sharpe * 0.4 + sortino * 0.3 + win_rate * 0.2 + (1 + max_dd) * 0.1)
        
        return score

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial (optuna.Trial): Optuna trial object

        Returns:
            float: Optimization score
        """
        try:
            # Suggest parameters
            params = self._suggest_parameters(trial)

            # Create indicators with suggested parameters
            df_with_indicators = self.create_indicators_with_params(params)

            # Generate signals
            signals, entry_prices = self.generate_signals_with_tp_sl(
                df_with_indicators, params
            )

            # Calculate performance metrics
            prices = df_with_indicators["Close"].tolist()
            if len(signals) != len(prices):
                return -np.inf

            strategy_metrics = analyze_strategy_performance(prices, signals)

            # Calculate optimization score
            score = self._calculate_optimization_score(strategy_metrics, params)

            return score

        except Exception as e:
            return -np.inf

    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.

        Returns:
            Dict[str, Any]: Optimization results
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials)

        # Store best results
        self.best_params = study.best_params
        self.best_score = study.best_value

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study": study,
        }

    def get_optimized_results(self) -> Dict[str, Any]:
        """Get final results with optimized parameters."""
        if self.best_params is None:
            raise ValueError("Must run optimize() first")

        # Create indicators with best parameters
        df_optimized = self.create_indicators_with_params(self.best_params)

        # Generate signals with optimized TP/SL
        signals, entry_prices = self.generate_signals_with_tp_sl(
            df_optimized, self.best_params
        )

        # Calculate final performance
        prices = df_optimized["Close"].tolist()
        strategy_metrics = analyze_strategy_performance(prices, signals)

        # Calculate ML performance
        X, y = self.prepare_features_target(df_optimized)
        ml_score, fold_scores = self.evaluate_ml_performance(X, y, self.best_params)

        return {
            "strategy_metrics": strategy_metrics,
            "ml_score": ml_score,
            "fold_scores": fold_scores,
            "signals": signals,
            "entry_prices": entry_prices,
            "optimized_data": df_optimized,
        }
