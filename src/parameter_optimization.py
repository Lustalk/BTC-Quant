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
from .transaction_costs import TransactionCostModel


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
        df[f"SMA_{params['sma_short']}"] = ta.trend.sma_indicator(df["Close"], window=params['sma_short'])
        df[f"SMA_{params['sma_long']}"] = ta.trend.sma_indicator(df["Close"], window=params['sma_long'])
        df[f"EMA_{params['ema_short']}"] = ta.trend.ema_indicator(df["Close"], window=params['ema_short'])
        df[f"EMA_{params['ema_long']}"] = ta.trend.ema_indicator(df["Close"], window=params['ema_long'])

        # MACD with variable parameters
        df["MACD"] = ta.trend.macd(df["Close"], window_fast=params['macd_fast'],
                                   window_slow=params['macd_slow'])
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"], window_fast=params['macd_fast'],
                                                 window_slow=params['macd_slow'])
        df["MACD_Histogram"] = ta.trend.macd_diff(df["Close"], window_fast=params['macd_fast'],
                                                  window_slow=params['macd_slow'])

        # Momentum Indicators
        df[f"RSI_{params['rsi_window']}"] = ta.momentum.rsi(df["Close"], window=params['rsi_window'])
        df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=params['stoch_window'])
        df["Stoch_D"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"], window=params['stoch_window'])
        df["Williams_R"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=params['williams_window'])

        # Volatility Indicators
        df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"], window=params['bb_window'])
        df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"], window=params['bb_window'])
        df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"], window=params['bb_window'])
        df["BB_Width"] = ta.volatility.bollinger_wband(df["Close"], window=params['bb_window'])
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=params['atr_window'])

        # Volume Indicators
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])

        # Price-based features
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High_Low_Ratio"] = df["High"] / df["Low"]
        df["Close_Open_Ratio"] = df["Close"] / df["Open"]

        # Lagged features with variable lags
        for lag in [params['lag_1'], params['lag_2'], params['lag_3']]:
            df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
            df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)

        # Rolling statistics with variable windows
        for window in [params['roll_short'], params['roll_medium'], params['roll_long']]:
            df[f"Close_Std_{window}"] = df["Close"].rolling(window=window).std()
            df[f"Volume_Mean_{window}"] = df["Volume"].rolling(window=window).mean()
            df[f"Returns_Mean_{window}"] = df["Returns"].rolling(window=window).mean()

        return df.dropna()
    
    def generate_signals_with_tp_sl(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[List[int], List[float]]:
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
            current_price = row['Close']
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
                rsi_oversold = rsi < params['rsi_oversold']
                trend_up = current_price > sma_short > sma_long
                ema_trend_up = ema_short > ema_long
                macd_bullish = macd > macd_signal
                stoch_oversold = stoch_k < 20 and stoch_d < 20
                
                # More flexible buy conditions - any combination of 2+ conditions
                conditions_met = sum([
                    rsi_oversold,
                    trend_up,
                    ema_trend_up,
                    macd_bullish,
                    stoch_oversold
                ])
                
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
                if current_price >= entry_price * (1 + params['take_profit']):
                    signal = -1  # Sell (take profit)
                    position = 0
                    entry_price = 0
                # Stop loss
                elif current_price <= entry_price * (1 - params['stop_loss']):
                    signal = -1  # Sell (stop loss)
                    position = 0
                    entry_price = 0
                # RSI overbought exit
                elif rsi > params['rsi_overbought']:
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
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization with transaction costs.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Optimization score (higher is better)
        """
        # Define parameter search spaces with wider ranges
        params = {
            # Indicator parameters - wider ranges for better exploration
            'sma_short': trial.suggest_int('sma_short', 3, 50),
            'sma_long': trial.suggest_int('sma_long', 10, 200),
            'ema_short': trial.suggest_int('ema_short', 3, 30),
            'ema_long': trial.suggest_int('ema_long', 10, 100),
            'rsi_window': trial.suggest_int('rsi_window', 5, 50),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 15, 45),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 55, 85),
            'stoch_window': trial.suggest_int('stoch_window', 5, 30),
            'williams_window': trial.suggest_int('williams_window', 5, 30),
            'bb_window': trial.suggest_int('bb_window', 5, 50),
            'atr_window': trial.suggest_int('atr_window', 5, 30),
            'macd_fast': trial.suggest_int('macd_fast', 5, 25),
            'macd_slow': trial.suggest_int('macd_slow', 15, 50),
            
            # Lag parameters - wider ranges
            'lag_1': trial.suggest_int('lag_1', 1, 10),
            'lag_2': trial.suggest_int('lag_2', 2, 15),
            'lag_3': trial.suggest_int('lag_3', 3, 20),
            
            # Rolling window parameters - wider ranges
            'roll_short': trial.suggest_int('roll_short', 2, 15),
            'roll_medium': trial.suggest_int('roll_medium', 5, 30),
            'roll_long': trial.suggest_int('roll_long', 10, 100),
            
            # TP/SL parameters - wider ranges
            'take_profit': trial.suggest_float('take_profit', 0.01, 0.20),
            'stop_loss': trial.suggest_float('stop_loss', 0.005, 0.10),
            
            # Transaction cost parameters
            'fee_type': trial.suggest_categorical('fee_type', ['maker', 'taker']),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            
            # Position sizing parameters
            'position_sizing_strategy': trial.suggest_categorical('position_sizing_strategy', 
                                                                ['volatility_targeted', 'risk_based', 'hybrid']),
            'target_volatility': trial.suggest_float('target_volatility', 0.10, 0.25),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.01, 0.05),
            
            # ML model parameters - wider ranges
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'n_estimators': trial.suggest_int('n_estimators', 25, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        try:
            # Create indicators with optimized parameters
            df_with_indicators = self.create_indicators_with_params(params)
            
            # Generate signals with TP/SL
            signals, entry_prices = self.generate_signals_with_tp_sl(df_with_indicators, params)
            
            # Calculate strategy performance with transaction costs
            prices = df_with_indicators['Close'].tolist()
            volumes = df_with_indicators['Volume'].tolist()
            atr_values = df_with_indicators['ATR'].tolist()
            
            if len(signals) == len(prices):
                # Analyze strategy with transaction costs and position sizing
                strategy_metrics = analyze_strategy_performance(
                    prices, signals, 
                    volumes=volumes, 
                    atr_values=atr_values,
                    include_transaction_costs=True,
                    position_sizing_strategy=params.get('position_sizing_strategy', 'volatility_targeted')
                )
                
                # Calculate ML performance
                X, y = self.prepare_features_target(df_with_indicators)
                ml_score = self.evaluate_ml_performance(X, y, params)
                
                # Enhanced scoring with Calmar ratio as primary metric
                sortino_ratio = strategy_metrics.get('sortino_ratio', 0)
                sharpe_ratio = strategy_metrics.get('sharpe_ratio', 0)
                total_return = strategy_metrics.get('total_return', 0)
                max_drawdown = strategy_metrics.get('max_drawdown', 0.01)
                win_rate = strategy_metrics.get('win_rate', 0)
                
                # Calculate Calmar ratio (Return / Max Drawdown)
                calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
                
                # Penalize negative returns heavily
                if total_return < 0:
                    return -1000
                
                # Penalize very low win rates
                if win_rate < 0.3:
                    return -500
                
                # Penalize excessive transaction costs
                cost_impact = strategy_metrics.get('cost_impact', 0)
                if cost_impact > 0.1:  # More than 10% cost impact
                    return -200
                
                # Combined score with Calmar ratio as primary metric
                strategy_score = (0.5 * calmar_ratio +
                                0.2 * sortino_ratio +
                                0.1 * sharpe_ratio +
                                0.1 * total_return +
                                0.1 * win_rate)
                
                # Penalize high transaction costs
                cost_penalty = max(0, cost_impact * 10)
                strategy_score -= cost_penalty
                
                combined_score = (0.7 * strategy_score + 0.3 * ml_score)
                
                return combined_score
            else:
                return -1000  # Penalty for invalid signals
                
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000  # Penalty for failed trials
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML model."""
        df = df.copy()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()
        
        exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Target"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df["Target"]
        
        return X, y
    
    def evaluate_ml_performance(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> float:
        """Evaluate ML model performance with given parameters."""
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model = xgb.XGBClassifier(
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    random_state=42,
                    eval_metric="logloss"
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Use F1 score for balanced evaluation
                f1 = f1_score(y_test, y_pred, average='weighted')
                scores.append(f1)
            
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with best parameters and results
        """
        print(f"Starting optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.objective_function, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score: {self.best_score:.4f}")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }
    
    def get_optimized_results(self) -> Dict[str, Any]:
        """Get final results with optimized parameters."""
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        # Create indicators with best parameters
        df_optimized = self.create_indicators_with_params(self.best_params)
        
        # Generate signals with optimized TP/SL
        signals, entry_prices = self.generate_signals_with_tp_sl(df_optimized, self.best_params)
        
        # Calculate final performance
        prices = df_optimized['Close'].tolist()
        strategy_metrics = analyze_strategy_performance(prices, signals)
        
        # Calculate ML performance
        X, y = self.prepare_features_target(df_optimized)
        ml_score = self.evaluate_ml_performance(X, y, self.best_params)
        
        return {
            'strategy_metrics': strategy_metrics,
            'ml_score': ml_score,
            'signals': signals,
            'entry_prices': entry_prices,
            'optimized_data': df_optimized
        }