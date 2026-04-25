import numpy as np
from pathlib import Path
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import Config
from data_loader import fetch_ohlcv
from features import add_indicators, make_supervised
from dataset import SequenceDataset, make_sequences
from model import BiLSTMRegressor
from baselines import baseline_zero, baseline_mean
from evaluation.metrics import compute_all_metrics, format_metrics
from evaluation.plots import plot_cumulative_returns, plot_errors, plot_predictions
from evaluation.trading import evaluate_strategy, format_strategy_metrics
from utils import ensure_dirs, device


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def load_checkpoint(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run `python src/train.py` first to train and save the model checkpoint, "
            "then rerun `python src/evaluate.py`."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def print_metric_row(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {label:<14} {format_metrics(compute_all_metrics(y_true, y_pred))}")


def print_strategy_row(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    metrics = evaluate_strategy(y_true, y_pred)
    print(f"  {label:<14} {format_strategy_metrics(metrics)}")


def evaluate(cfg: Config, ckpt_path: str):
    ensure_dirs(cfg.out_plots)
    dev = device()

    ckpt = load_checkpoint(ckpt_path)
    feature_cols = ckpt["feature_cols"]
    train_mean_target = float(ckpt["train_mean_target"])
    demean_target = bool(ckpt.get("demean_target", False))

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.horizon)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]

    _, _, test_df = time_split(df, cfg.train_ratio, cfg.val_ratio)

    def transform(split: pd.DataFrame):
        X = scaler.transform(split[feature_cols].values).astype(np.float32)
        y = split["target"].values.astype(np.float32)
        dates = pd.to_datetime(split["Date"]).values
        return X, y, dates

    X_te, y_te, dates_te = transform(test_df)

    if demean_target:
        y_te_model = y_te - train_mean_target
    else:
        y_te_model = y_te

    Xte_s, yte_s = make_sequences(X_te, y_te_model, cfg.lookback)
    _, yte_true = make_sequences(X_te, y_te, cfg.lookback)
    dates_aligned = dates_te[cfg.lookback:]

    base_zero = baseline_zero(yte_true)
    base_mean = baseline_mean(yte_true, train_mean_target)

    model = BiLSTMRegressor(
        num_features=Xte_s.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        SequenceDataset(Xte_s, yte_s),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)
            trues.append(yb.numpy().reshape(-1))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    if demean_target:
        preds = preds + train_mean_target
        trues = trues + train_mean_target

    print("Test Metrics")
    print_metric_row("Zero Baseline", yte_true, base_zero)
    print_metric_row("Mean Baseline", yte_true, base_mean)
    print_metric_row("BiLSTM", trues, preds)

    print("\nTrading Metrics")
    print_strategy_row("Zero Baseline", yte_true, base_zero)
    print_strategy_row("Mean Baseline", yte_true, base_mean)
    print_strategy_row("BiLSTM", trues, preds)

    print(f"Mean actual return: {np.mean(trues):.6f}")
    print(f"Mean predicted return: {np.mean(preds):.6f}")
    print(f"Std actual return: {np.std(trues):.6f}")
    print(f"Std predicted return: {np.std(preds):.6f}")

    predictions_plot = f"{cfg.out_plots}/{cfg.ticker}_test_predictions.png"
    errors_plot = f"{cfg.out_plots}/{cfg.ticker}_test_prediction_errors.png"
    returns_plot = f"{cfg.out_plots}/{cfg.ticker}_test_cumulative_returns.png"

    plot_predictions(trues, preds, dates=dates_aligned, output_path=predictions_plot)
    plot_errors(trues, preds, dates=dates_aligned, output_path=errors_plot)
    plot_cumulative_returns(trues, preds, dates=dates_aligned, output_path=returns_plot)

    print(f"Saved plot: {predictions_plot}")
    print(f"Saved plot: {errors_plot}")
    print(f"Saved plot: {returns_plot}")


if __name__ == "__main__":
    cfg = Config()
    ckpt_path = f"{cfg.out_models}/{cfg.ticker}_bilstm.pt"
    evaluate(cfg, ckpt_path)
