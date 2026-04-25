import numpy as np


def baseline_zero(y_true: np.ndarray) -> np.ndarray:
    return np.zeros_like(y_true, dtype=np.float32)


def baseline_mean(y_true: np.ndarray, train_mean: float) -> np.ndarray:
    return np.full_like(y_true, fill_value=train_mean, dtype=np.float32)
