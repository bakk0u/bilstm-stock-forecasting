import numpy as np

def baseline_last_value(y_true: np.ndarray, last_close: np.ndarray):
    """
    Predict next close as today's close.
    y_true: actual next close
    last_close: today's close aligned with y_true
    """
    y_pred = last_close
    return y_pred

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
