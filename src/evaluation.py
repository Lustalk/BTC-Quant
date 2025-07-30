import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def calculate_returns(prices: List[float]) -> List[float]:
    """
    Calculate simple returns from a list of prices.

    Args:
        prices (List[float]): List of prices

    Returns:
        List[float]: List of returns
    """
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns.append(ret)
    return returns


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio.

    Args:
        returns (List[float]): List of returns
        risk_free_rate (float): Annual risk-free rate (default: 0.02)

    Returns:
        float: Sharpe ratio
    """
    if not returns:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(252)  # Annualized


def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Calculate the maximum drawdown.

    Args:
        returns (List[float]): List of returns

    Returns:
        float: Maximum drawdown as a percentage
    """
    if not returns:
        return 0.0

    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return abs(max_drawdown)


def calculate_volatility(returns: List[float]) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns (List[float]): List of returns

    Returns:
        float: Annualized volatility
    """
    if not returns:
        return 0.0

    return np.std(returns) * np.sqrt(252)


def calculate_auc(actual: List[int], predicted: List[float]) -> float:
    """
    Calculate Area Under the Curve (AUC) for binary classification.

    Args:
        actual (List[int]): Actual binary labels (0 or 1)
        predicted (List[float]): Predicted probabilities

    Returns:
        float: AUC score
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have the same length")

    if len(set(actual)) != 2:
        return 0.5  # If only one class, AUC is 0.5

    # Simple AUC calculation using sklearn's approach
    from sklearn.metrics import roc_auc_score

    try:
        return roc_auc_score(actual, predicted)
    except Exception:
        # Fallback to manual calculation
        sorted_data = sorted(zip(predicted, actual), reverse=True)

        tp = fp = 0
        tn = sum(1 for x in actual if x == 0)
        fn = sum(1 for x in actual if x == 1)

        auc = 0.0
        prev_score = None

        for score, label in sorted_data:
            if prev_score != score:
                auc += trapezoid_area(fp, tp, fp, tp)
                prev_score = score

            if label == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1

        auc += trapezoid_area(fp, tp, fp, tp)

        if (tp + fn) * (fp + tn) == 0:
            return 0.5

        auc = auc / (tp + fn) / (fp + tn)

        return auc


def trapezoid_area(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate area of trapezoid for AUC calculation."""
    return (x2 - x1) * (y1 + y2) / 2


def calculate_win_rate(returns: List[float]) -> float:
    """
    Calculate the win rate (percentage of positive returns).

    Args:
        returns (List[float]): List of returns

    Returns:
        float: Win rate as a percentage
    """
    if not returns:
        return 0.0

    positive_returns = sum(1 for r in returns if r > 0)
    return positive_returns / len(returns)


def calculate_profit_factor(returns: List[float]) -> float:
    """
    Calculate the profit factor (gross profit / gross loss).

    Args:
        returns (List[float]): List of returns

    Returns:
        float: Profit factor
    """
    if not returns:
        return 0.0

    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_all_metrics(
    returns: List[float], actual: List[int] = None, predicted: List[float] = None
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        returns (List[float]): List of returns
        actual (List[int]): Actual binary labels (optional)
        predicted (List[float]): Predicted probabilities (optional)

    Returns:
        Dict[str, float]: Dictionary of all metrics
    """
    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "volatility": calculate_volatility(returns),
        "win_rate": calculate_win_rate(returns),
        "profit_factor": calculate_profit_factor(returns),
        "total_return": sum(returns),
        "avg_return": np.mean(returns) if returns else 0.0,
    }

    if actual is not None and predicted is not None:
        metrics["auc"] = calculate_auc(actual, predicted)

    return metrics
