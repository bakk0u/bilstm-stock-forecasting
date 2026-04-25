from __future__ import annotations

import numpy as np


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_1d_array(y_true)
    y_pred = _as_1d_array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate a strategy on empty arrays.")

    return y_true, y_pred


def cumulative_return_from_log_returns(log_returns: np.ndarray) -> np.ndarray:
    log_returns = _as_1d_array(log_returns)
    return np.exp(np.cumsum(log_returns)) - 1.0


def sharpe_ratio(log_returns: np.ndarray) -> float:
    log_returns = _as_1d_array(log_returns)
    if log_returns.size < 2:
        return 0.0

    vol = np.std(log_returns, ddof=1)
    if vol < 1e-12:
        return 0.0

    return float(np.mean(log_returns) / vol * np.sqrt(log_returns.size))


def max_drawdown(log_returns: np.ndarray) -> float:
    log_returns = _as_1d_array(log_returns)
    equity = np.exp(np.cumsum(log_returns))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(abs(np.min(drawdown)))


def evaluate_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transaction_cost: float = 0.0005,
    threshold: float = 0.0,
) -> dict[str, float]:
    """
    Long/flat strategy on forward log returns.

    signal_t = 1 if predicted_return_t > threshold else 0
    strategy_log_return_t = signal_t * actual_forward_log_return_t - transaction_cost_t
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    signal = (y_pred > threshold).astype(np.float64)
    position_change = np.abs(np.diff(signal, prepend=0.0))
    strategy_log_returns = signal * y_true - transaction_cost * position_change
    buy_hold_log_returns = y_true

    strategy_curve = cumulative_return_from_log_returns(strategy_log_returns)
    buy_hold_curve = cumulative_return_from_log_returns(buy_hold_log_returns)

    return {
        "total_return": float(strategy_curve[-1]),
        "buy_hold_return": float(buy_hold_curve[-1]),
        "sharpe": sharpe_ratio(strategy_log_returns),
        "max_drawdown": max_drawdown(strategy_log_returns),
        "exposure": float(np.mean(signal)),
        "num_trades": float(np.sum(position_change)),
    }


def format_strategy_metrics(metrics: dict[str, float]) -> str:
    return (
        f"Total Return={metrics['total_return']:.2%} "
        f"Buy&Hold={metrics['buy_hold_return']:.2%} "
        f"Sharpe={metrics['sharpe']:.2f} "
        f"MaxDD={metrics['max_drawdown']:.2%} "
        f"Exposure={metrics['exposure']:.2%} "
        f"Trades={metrics['num_trades']:.0f}"
    )
