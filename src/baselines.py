import numpy as np


def baseline_zero(y_true: np.ndarray) -> np.ndarray:
    return np.zeros_like(y_true, dtype=np.float32)


def baseline_mean(y_true: np.ndarray, train_mean: float) -> np.ndarray:
    return np.full_like(y_true, fill_value=train_mean, dtype=np.float32)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_sign = (y_true > 0).astype(int)
    pred_sign = (y_pred > 0).astype(int)
    return float(np.mean(true_sign == pred_sign) * 100.0)


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)