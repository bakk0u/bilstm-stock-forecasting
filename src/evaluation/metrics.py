from __future__ import annotations

import numpy as np


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _validate_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_1d_array(y_true)
    y_pred = _as_1d_array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    return y_true, y_pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_same_shape(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_same_shape(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_same_shape(y_true, y_pred)
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign) * 100.0)


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if y_true.size < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, min_abs_actual: float = 1e-4) -> float:
    """
    Mean absolute percentage error, excluding observations where the absolute
    actual return is too close to zero to make a percentage error meaningful.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)
    mask = np.abs(y_true) >= min_abs_actual

    if not np.any(mask):
        return float("nan")

    pct_error = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_error) * 100.0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": safe_mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true, y_score = _validate_same_shape(y_true, y_score)
    y_true = (y_true > 0).astype(int)
    y_pred = (y_score > threshold).astype(int)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float(np.mean(y_true == y_pred) * 100.0)
    precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp) * 100.0)
    recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn) * 100.0)
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_classification_metrics(metrics: dict[str, float]) -> str:
    return (
        f"Accuracy={metrics['accuracy']:.2f}% "
        f"Precision={metrics['precision']:.2f}% "
        f"Recall={metrics['recall']:.2f}% "
        f"F1={metrics['f1']:.2f}%"
    )


def format_metrics(metrics: dict[str, float]) -> str:
    mape = metrics["mape"]
    mape_text = "nan" if np.isnan(mape) else f"{mape:.2f}%"
    return (
        f"MAE={metrics['mae']:.6f} "
        f"RMSE={metrics['rmse']:.6f} "
        f"MAPE={mape_text} "
        f"DA={metrics['directional_accuracy']:.2f}% "
        f"Corr={metrics['correlation']:.4f}"
    )
