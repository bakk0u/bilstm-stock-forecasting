import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import Config
from data_loader import fetch_ohlcv
from features import add_indicators, make_supervised
from dataset import SequenceDataset, make_sequences
from model import BiLSTMRegressor
from baselines import mae, rmse, directional_accuracy, correlation, baseline_zero, baseline_mean
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
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(cfg: Config, ckpt_path: str):
    ensure_dirs(cfg.out_plots)
    dev = device()

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.horizon)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    ckpt = load_checkpoint(ckpt_path)
    feature_cols = ckpt["feature_cols"]
    train_mean_target = float(ckpt["train_mean_target"])
    demean_target = bool(ckpt.get("demean_target", False))

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
    print(
        f"  Zero Baseline  MAE={mae(yte_true, base_zero):.6f} "
        f"RMSE={rmse(yte_true, base_zero):.6f} "
        f"DA={directional_accuracy(yte_true, base_zero):.2f}% "
        f"Corr={correlation(yte_true, base_zero):.4f}"
    )
    print(
        f"  Mean Baseline  MAE={mae(yte_true, base_mean):.6f} "
        f"RMSE={rmse(yte_true, base_mean):.6f} "
        f"DA={directional_accuracy(yte_true, base_mean):.2f}% "
        f"Corr={correlation(yte_true, base_mean):.4f}"
    )
    print(
        f"  BiLSTM         MAE={mae(trues, preds):.6f} "
        f"RMSE={rmse(trues, preds):.6f} "
        f"DA={directional_accuracy(trues, preds):.2f}% "
        f"Corr={correlation(trues, preds):.4f}"
    )

    print(f"Mean actual return: {np.mean(trues):.6f}")
    print(f"Mean predicted return: {np.mean(preds):.6f}")
    print(f"Std actual return: {np.std(trues):.6f}")
    print(f"Std predicted return: {np.std(preds):.6f}")

    plt.figure(figsize=(11, 5))
    plt.plot(dates_aligned, trues, label="Actual")
    plt.plot(dates_aligned, base_zero, label="Zero baseline")
    plt.plot(dates_aligned, preds, label="BiLSTM")
    plt.title(f"{cfg.ticker} {cfg.horizon}-Day Forward Log Return Prediction (Test)")
    plt.xlabel("Date")
    plt.ylabel("Forward Log Return")
    plt.legend()
    plt.tight_layout()

    out = f"{cfg.out_plots}/{cfg.ticker}_test_predictions.png"
    plt.savefig(out, dpi=160)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    cfg = Config()
    ckpt_path = f"{cfg.out_models}/{cfg.ticker}_bilstm.pt"
    evaluate(cfg, ckpt_path)