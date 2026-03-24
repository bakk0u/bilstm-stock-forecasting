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
from baselines import mae, rmse, directional_accuracy, correlation, baseline_zero, baseline_mean
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
    df = make_supervised(df, cfg.horizon)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    missing = [c for c in cfg.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing configured feature columns: {missing}")

    feature_cols = cfg.feature_cols

    train_df, val_df, test_df = time_split(df, cfg.train_ratio, cfg.val_ratio)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    def transform(split: pd.DataFrame):
        X = scaler.transform(split[feature_cols].values).astype(np.float32)
        y = split["target"].values.astype(np.float32)
        return X, y

    X_tr, y_tr = transform(train_df)
    X_va, y_va = transform(val_df)
    X_te, y_te = transform(test_df)

    train_target_mean = float(np.mean(y_tr))

    if cfg.demean_target:
        y_tr_model = y_tr - train_target_mean
        y_va_model = y_va - train_target_mean
        y_te_model = y_te - train_target_mean
    else:
        y_tr_model = y_tr
        y_va_model = y_va
        y_te_model = y_te

    Xtr_s, ytr_s = make_sequences(X_tr, y_tr_model, cfg.lookback)
    Xva_s, yva_s = make_sequences(X_va, y_va_model, cfg.lookback)
    Xte_s, yte_s = make_sequences(X_te, y_te_model, cfg.lookback)

    _, yva_true = make_sequences(X_va, y_va, cfg.lookback)
    _, yte_true = make_sequences(X_te, y_te, cfg.lookback)

    train_mean_target = train_target_mean

    base_zero_va = baseline_zero(yva_true)
    base_mean_va = baseline_mean(yva_true, train_mean_target)
    base_zero_te = baseline_zero(yte_true)
    base_mean_te = baseline_mean(yte_true, train_mean_target)

    print("Baselines")
    print(
        f"  Val  Zero  MAE={mae(yva_true, base_zero_va):.6f} "
        f"RMSE={rmse(yva_true, base_zero_va):.6f} "
        f"DA={directional_accuracy(yva_true, base_zero_va):.2f}% "
        f"Corr={correlation(yva_true, base_zero_va):.4f}"
    )
    print(
        f"  Val  Mean  MAE={mae(yva_true, base_mean_va):.6f} "
        f"RMSE={rmse(yva_true, base_mean_va):.6f} "
        f"DA={directional_accuracy(yva_true, base_mean_va):.2f}% "
        f"Corr={correlation(yva_true, base_mean_va):.4f}"
    )
    print(
        f"  Test Zero  MAE={mae(yte_true, base_zero_te):.6f} "
        f"RMSE={rmse(yte_true, base_zero_te):.6f} "
        f"DA={directional_accuracy(yte_true, base_zero_te):.2f}% "
        f"Corr={correlation(yte_true, base_zero_te):.4f}"
    )
    print(
        f"  Test Mean  MAE={mae(yte_true, base_mean_te):.6f} "
        f"RMSE={rmse(yte_true, base_mean_te):.6f} "
        f"DA={directional_accuracy(yte_true, base_mean_te):.2f}% "
        f"Corr={correlation(yte_true, base_mean_te):.4f}"
    )

    dev = device()
    model = BiLSTMRegressor(
        num_features=Xtr_s.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.loss_name.lower() == "huber":
        loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    elif cfg.loss_name.lower() == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss_name: {cfg.loss_name}")

    tr_loader = DataLoader(
        SequenceDataset(Xtr_s, ytr_s),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    va_loader = DataLoader(
        SequenceDataset(Xva_s, yva_s),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    best_val = float("inf")
    best_path = f"{cfg.out_models}/{cfg.ticker}_bilstm.pt"
    bad_epochs = 0

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

        va_loss = float(np.mean(va_losses))
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        if cfg.demean_target:
            preds_eval = preds + train_target_mean
            trues_eval = trues + train_target_mean
        else:
            preds_eval = preds
            trues_eval = trues

        print(
            f"Epoch {epoch}: "
            f"train_mse={np.mean(tr_losses):.6f} "
            f"val_mse={va_loss:.6f} "
            f"val_mae={mae(trues_eval, preds_eval):.6f} "
            f"val_da={directional_accuracy(trues_eval, preds_eval):.2f}% "
            f"val_corr={correlation(trues_eval, preds_eval):.4f}"
        )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "feature_cols": feature_cols,
                    "train_mean_target": train_target_mean,
                    "demean_target": cfg.demean_target,
                    "cfg": cfg.__dict__,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping triggered.")
                break

    print(f"Saved best model to: {best_path}")
    return best_path


if __name__ == "__main__":
    cfg = Config()
    train_one(cfg)