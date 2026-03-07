import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data_loader import fetch_ohlcv
from features import add_indicators, make_supervised
from dataset import SequenceDataset, make_sequences
from model import BiLSTMRegressor
from baselines import mae, rmse, mape, baseline_last_value
from utils import set_seed, ensure_dirs, device

def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test

def train_one(cfg: Config):
    set_seed(cfg.seed)
    ensure_dirs(cfg.out_models, cfg.out_plots)

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.target_col, cfg.horizon)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "target"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    train_df, val_df, test_df = time_split(df, cfg.train_ratio, cfg.val_ratio)

    bad = train_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if bad:
        raise ValueError(f"Non-numeric feature columns found: {bad}")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    def transform(split: pd.DataFrame):
        X = scaler.transform(split[feature_cols].values).astype(np.float32)
        y = split["target"].values.astype(np.float32)
        close_today = split["Close"].values.astype(np.float32)
        return X, y, close_today

    X_tr, y_tr, _ = transform(train_df)
    X_va, y_va, close_va = transform(val_df)
    X_te, y_te, close_te = transform(test_df)

    if len(X_tr) <= cfg.lookback:
        raise ValueError(f"Train split too small for lookback={cfg.lookback}")
    if len(X_va) <= cfg.lookback:
        raise ValueError(f"Validation split too small for lookback={cfg.lookback}")
    if len(X_te) <= cfg.lookback:
        raise ValueError(f"Test split too small for lookback={cfg.lookback}")

    Xtr_s, ytr_s = make_sequences(X_tr, y_tr, cfg.lookback)
    Xva_s, yva_s = make_sequences(X_va, y_va, cfg.lookback)
    Xte_s, yte_s = make_sequences(X_te, y_te, cfg.lookback)

    close_va_aligned = close_va[cfg.lookback:]
    close_te_aligned = close_te[cfg.lookback:]
    base_va = baseline_last_value(yva_s, close_va_aligned)
    base_te = baseline_last_value(yte_s, close_te_aligned)

    print("Baseline (Predict Next Close = Today Close)")
    print(f"  Val  MAE={mae(yva_s, base_va):.4f} RMSE={rmse(yva_s, base_va):.4f} MAPE={mape(yva_s, base_va):.2f}%")
    print(f"  Test MAE={mae(yte_s, base_te):.4f} RMSE={rmse(yte_s, base_te):.4f} MAPE={mape(yte_s, base_te):.2f}%")

    dev = device()
    model = BiLSTMRegressor(
        num_features=Xtr_s.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(
        SequenceDataset(Xtr_s, ytr_s),
        batch_size=cfg.batch_size,
        shuffle=True
    )
    va_loader = DataLoader(
        SequenceDataset(Xva_s, yva_s),
        batch_size=cfg.batch_size,
        shuffle=False
    )

    best_val = float("inf")
    best_path = f"{cfg.out_models}/{cfg.ticker}_bilstm.pt"
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_losses = []

        for xb, yb in tqdm(tr_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        preds = []
        trues = []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                va_losses.append(loss_fn(pred, yb).item())
                preds.append(pred.cpu().numpy().reshape(-1))
                trues.append(yb.cpu().numpy().reshape(-1))

        train_mse = float(np.mean(tr_losses))
        val_mse = float(np.mean(va_losses))
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_mae = mae(trues, preds)

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
        })

        print(f"Epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f} val_mae={val_mae:.4f}")

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "feature_cols": feature_cols,
                    "best_val_loss": best_val,
                    "cfg": vars(cfg),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping triggered.")
                break

    history_df = pd.DataFrame(history)
    history_path = f"{cfg.out_plots}/{cfg.ticker}_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    print(f"Saved best model to: {best_path}")
    return best_path

if __name__ == "__main__":
    cfg = Config()
    train_one(cfg)
