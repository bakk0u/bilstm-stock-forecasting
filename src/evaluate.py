import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import Config
from data_loader import fetch_ohlcv
from features import add_indicators, make_supervised
from dataset import SequenceDataset, make_sequences
from model import BiLSTMRegressor
from baselines import mae, rmse, mape, baseline_last_value
from utils import ensure_dirs, device

def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def evaluate(cfg: Config, ckpt_path: str):
    ensure_dirs(cfg.out_plots)
    dev = device()

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.target_col, cfg.horizon)

    ckpt = load_checkpoint(ckpt_path)
    feature_cols = ckpt["feature_cols"]

    # rebuild scaler
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]

    train_df, val_df, test_df = time_split(df, cfg.train_ratio, cfg.val_ratio)

    def transform(split: pd.DataFrame):
        X = scaler.transform(split[feature_cols].values).astype(np.float32)
        y = split["target"].values.astype(np.float32)
        close_today = split["Close"].values.astype(np.float32)
        dates = pd.to_datetime(split["Date"]).values
        return X, y, close_today, dates

    X_te, y_te, close_te, dates_te = transform(test_df)
    Xte_s, yte_s = make_sequences(X_te, y_te, cfg.lookback)
    dates_aligned = dates_te[cfg.lookback:]
    close_aligned = close_te[cfg.lookback:]

    base = baseline_last_value(yte_s, close_aligned)

    # model
    model = BiLSTMRegressor(
        num_features=Xte_s.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(SequenceDataset(Xte_s, yte_s), batch_size=cfg.batch_size, shuffle=False)

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

    print("Test Metrics")
    print(f"  Baseline  MAE={mae(trues, base):.4f} RMSE={rmse(trues, base):.4f} MAPE={mape(trues, base):.2f}%")
    print(f"  BiLSTM    MAE={mae(trues, preds):.4f} RMSE={rmse(trues, preds):.4f} MAPE={mape(trues, preds):.2f}%")

    # Plot
    plt.figure(figsize=(11, 5))
    plt.plot(dates_aligned, trues, label="Actual")
    plt.plot(dates_aligned, base, label="Baseline (Today=Tomorrow)")
    plt.plot(dates_aligned, preds, label="BiLSTM")
    plt.title(f"{cfg.ticker} Next-Day Close Prediction (Test)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    out = f"{cfg.out_plots}/{cfg.ticker}_test_predictions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved plot: {out}")

if __name__ == "__main__":
    cfg = Config()
    ckpt_path = f"{cfg.out_models}/{cfg.ticker}_bilstm.pt"
    evaluate(cfg, ckpt_path)
