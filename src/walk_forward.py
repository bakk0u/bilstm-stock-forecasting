import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import fetch_ohlcv
from features import add_indicators, make_supervised
from dataset import SequenceDataset, make_sequences
from model import BiLSTMRegressor
from baselines import mae, rmse, directional_accuracy, correlation, baseline_zero, baseline_mean
from utils import set_seed, ensure_dirs, device


def build_sequences_for_block(
    full_X: np.ndarray,
    full_y: np.ndarray,
    start_idx: int,
    end_idx: int,
    lookback: int,
):
    """
    Build sequences whose targets lie in [start_idx, end_idx).
    Each sequence uses the previous `lookback` rows as history.
    """
    X_seq, y_seq = [], []
    for i in range(start_idx, end_idx):
        if i - lookback < 0:
            continue
        X_seq.append(full_X[i - lookback:i])
        y_seq.append(full_y[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def build_walk_forward_folds(
    n_obs: int,
    min_train_size: int,
    val_size: int,
    walk_step: int,
):
    folds = []
    test_start = min_train_size + val_size

    while test_start < n_obs:
        train_end = test_start - val_size
        val_end = test_start
        test_end = min(test_start + walk_step, n_obs)

        folds.append({
            "train_end": train_end,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        test_start += walk_step

    return folds


def fit_one_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    cfg: Config,
    fold: dict,
):
    dev = device()

    train_end = fold["train_end"]
    val_end = fold["val_end"]
    test_start = fold["test_start"]
    test_end = fold["test_end"]

    X_train_raw = X_all[:train_end]
    y_train_raw = y_all[:train_end]

    X_val_raw = X_all[train_end:val_end]
    y_val_raw = y_all[train_end:val_end]

    X_test_raw = X_all[:test_end]
    y_test_raw = y_all[:test_end]

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    Xtr_s, ytr_s = make_sequences(X_train, y_train_raw.astype(np.float32), cfg.lookback)
    Xva_s, yva_s = make_sequences(X_val, y_val_raw.astype(np.float32), cfg.lookback)

    # test sequences only for targets in [test_start, test_end)
    Xte_s, yte_s = build_sequences_for_block(
        full_X=X_test,
        full_y=y_test_raw.astype(np.float32),
        start_idx=test_start,
        end_idx=test_end,
        lookback=cfg.lookback,
    )

    if len(Xtr_s) == 0 or len(Xva_s) == 0 or len(Xte_s) == 0:
        return None

    model = BiLSTMRegressor(
        num_features=Xtr_s.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

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
    te_loader = DataLoader(
        SequenceDataset(Xte_s, yte_s),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is None:
        return None

    model.load_state_dict(best_state)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(dev)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)
            trues.append(yb.numpy().reshape(-1))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    train_mean_target = float(np.mean(ytr_s))
    pred_zero = baseline_zero(trues)
    pred_mean = baseline_mean(trues, train_mean_target)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_raw)

    ridge_test_block = scaler.transform(X_all[test_start:test_end]).astype(np.float32)
    ridge_pred = ridge.predict(ridge_test_block).astype(np.float32)

    return {
        "test_start": test_start,
        "test_end": test_end,
        "y_true": trues,
        "y_pred": preds,
        "y_zero": pred_zero,
        "y_mean": pred_mean,
        "y_ridge": ridge_pred,
    }

def walk_forward_evaluate(cfg: Config):
    set_seed(cfg.seed)
    ensure_dirs(cfg.out_plots)

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.horizon)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    missing = [c for c in cfg.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing configured feature columns: {missing}")

    X_all = df[cfg.feature_cols].values.astype(np.float32)
    y_all = df["target"].values.astype(np.float32)
    dates_all = pd.to_datetime(df["Date"]).values

    folds = build_walk_forward_folds(
        n_obs=len(df),
        min_train_size=cfg.min_train_size,
        val_size=cfg.val_size,
        walk_step=cfg.walk_step,
    )

    if not folds:
        raise ValueError("No walk-forward folds were created. Increase data length or reduce min_train_size/val_size.")

    all_true = []
    all_pred = []
    all_zero = []
    all_mean = []
    all_ridge = []
    all_dates = []


    print(f"Running {len(folds)} walk-forward folds...")

    for i, fold in enumerate(tqdm(folds, desc="Walk-forward folds")):
        result = fit_one_fold(X_all, y_all, cfg, fold)
        if result is None:
            continue

        test_start = result["test_start"]
        test_end = result["test_end"]

        all_true.append(result["y_true"])
        all_pred.append(result["y_pred"])
        all_zero.append(result["y_zero"])
        all_mean.append(result["y_mean"])
        all_ridge.append(result["y_ridge"])
        all_dates.append(dates_all[test_start:test_end])

        print(
            f"Fold {i+1:02d} | "
            f"BiLSTM MAE={mae(result['y_true'], result['y_pred']):.6f} | "
            f"RMSE={rmse(result['y_true'], result['y_pred']):.6f} | "
            f"DA={directional_accuracy(result['y_true'], result['y_pred']):.2f}% | "
            f"Corr={correlation(result['y_true'], result['y_pred']):.4f}"
        )

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_zero = np.concatenate(all_zero)
    y_mean = np.concatenate(all_mean)
    y_ridge = np.concatenate(all_ridge)
    dates = np.concatenate(all_dates)

    print("\nWalk-Forward Aggregated Metrics")
    print(
        f"  Zero Baseline  MAE={mae(y_true, y_zero):.6f} "
        f"RMSE={rmse(y_true, y_zero):.6f} "
        f"DA={directional_accuracy(y_true, y_zero):.2f}% "
        f"Corr={correlation(y_true, y_zero):.4f}"
    )
    print(
        f"  Mean Baseline  MAE={mae(y_true, y_mean):.6f} "
        f"RMSE={rmse(y_true, y_mean):.6f} "
        f"DA={directional_accuracy(y_true, y_mean):.2f}% "
        f"Corr={correlation(y_true, y_mean):.4f}"
    )
    print(
        f"  Ridge Baseline MAE={mae(y_true, y_ridge):.6f} "
        f"RMSE={rmse(y_true, y_ridge):.6f} "
        f"DA={directional_accuracy(y_true, y_ridge):.2f}% "
        f"Corr={correlation(y_true, y_ridge):.4f}"
    )
    print(
        f"  BiLSTM         MAE={mae(y_true, y_pred):.6f} "
        f"RMSE={rmse(y_true, y_pred):.6f} "
        f"DA={directional_accuracy(y_true, y_pred):.2f}% "
        f"Corr={correlation(y_true, y_pred):.4f}"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_zero, label="Zero baseline")
    plt.plot(dates, y_ridge, label="Ridge baseline")
    plt.plot(dates, y_pred, label="BiLSTM")
    plt.title(f"{cfg.ticker} Walk-Forward {cfg.horizon}-Day Forward Log Return Prediction")
    plt.xlabel("Date")
    plt.ylabel("Forward Log Return")
    plt.legend()
    plt.tight_layout()

    out = f"{cfg.out_plots}/{cfg.ticker}_walk_forward_predictions.png"
    plt.savefig(out, dpi=160)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    cfg = Config()
    walk_forward_evaluate(cfg)