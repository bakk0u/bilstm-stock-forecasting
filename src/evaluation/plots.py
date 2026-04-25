from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.trading import cumulative_return_from_log_returns


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _validate_series(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_1d_array(y_true)
    y_pred = _as_1d_array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot plot empty arrays.")

    return y_true, y_pred


def _x_axis(n_obs: int, dates=None):
    if dates is None:
        return np.arange(n_obs), "Observation Index"

    dates = np.asarray(dates)
    if len(dates) != n_obs:
        raise ValueError(f"dates length {len(dates)} does not match series length {n_obs}")

    return dates, "Date"


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates=None, output_path: str = "predictions.png") -> None:
    y_true, y_pred = _validate_series(y_true, y_pred)
    x, x_label = _x_axis(len(y_true), dates)

    plt.figure(figsize=(11, 5))
    plt.plot(x, y_true, label="Actual forward log return", linewidth=1.4)
    plt.plot(x, y_pred, label="Predicted forward log return", linewidth=1.2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Actual vs Predicted Forward Log Returns" if dates is not None else "Actual vs Predicted Forward Log Returns by Observation Index")
    plt.xlabel(x_label)
    plt.ylabel("Forward Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_errors(y_true: np.ndarray, y_pred: np.ndarray, dates=None, output_path: str = "prediction_errors.png") -> None:
    y_true, y_pred = _validate_series(y_true, y_pred)
    x, x_label = _x_axis(len(y_true), dates)
    errors = y_pred - y_true

    plt.figure(figsize=(11, 4))
    plt.plot(x, errors, label="Prediction error", linewidth=1.2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Prediction Error Over Time" if dates is not None else "Prediction Error by Observation Index")
    plt.xlabel(x_label)
    plt.ylabel("Predicted - Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_cumulative_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates=None,
    output_path: str = "cumulative_returns.png",
    transaction_cost: float = 0.0005,
    threshold: float = 0.0,
) -> None:
    y_true, y_pred = _validate_series(y_true, y_pred)
    x, x_label = _x_axis(len(y_true), dates)

    signal = (y_pred > threshold).astype(np.float64)
    position_change = np.abs(np.diff(signal, prepend=0.0))
    strategy_log_returns = signal * y_true - transaction_cost * position_change
    buy_hold_log_returns = y_true

    strategy_curve = cumulative_return_from_log_returns(strategy_log_returns)
    buy_hold_curve = cumulative_return_from_log_returns(buy_hold_log_returns)

    plt.figure(figsize=(11, 5))
    plt.plot(x, strategy_curve, label="Long/flat strategy", linewidth=1.4)
    plt.plot(x, buy_hold_curve, label="Buy and hold", linewidth=1.4)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    title = "Cumulative Return: Strategy vs Buy and Hold"
    if dates is None:
        title += " by Observation Index"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_fold_metrics(fold_metrics, output_path: str = "walk_forward_fold_metrics.png") -> None:
    required = {"fold", "model", "mae", "rmse", "directional_accuracy"}
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics is missing required columns: {missing}")

    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("directional_accuracy", "Directional Accuracy (%)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 9), sharex=True)
    for ax, (metric_col, metric_label) in zip(axes, metrics):
        for model_name, group in fold_metrics.groupby("model"):
            group = group.sort_values("fold")
            ax.plot(group["fold"], group[metric_col], marker="o", linewidth=1.2, label=model_name)
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)

    axes[0].set_title("Walk-Forward Fold Metrics")
    axes[-1].set_xlabel("Fold")
    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_classification_threshold_comparison(results, output_path: str = "classification_threshold_comparison.png") -> None:
    required = {"model", "threshold", "accuracy", "f1", "trading_sharpe"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {missing}")

    threshold_results = (
        results[results["model"] == "Classification BiLSTM"]
        .groupby("threshold", as_index=False)[["accuracy", "f1", "trading_sharpe"]]
        .mean()
        .sort_values("threshold")
    )
    if threshold_results.empty:
        raise ValueError("No Classification BiLSTM rows found for threshold comparison.")

    labels = [f"{threshold:.1f}" for threshold in threshold_results["threshold"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].bar(x, threshold_results["accuracy"])
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Classification Threshold Comparison")

    axes[1].bar(x, threshold_results["f1"])
    axes[1].set_ylabel("F1 (%)")

    axes[2].bar(x, threshold_results["trading_sharpe"])
    axes[2].set_ylabel("Sharpe")
    axes[2].set_xlabel("Probability Threshold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
