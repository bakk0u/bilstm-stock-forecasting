import json

import numpy as np
import pandas as pd
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
from model import BiLSTMClassifier, BiLSTMRegressor
from baselines import baseline_zero, baseline_mean
from evaluation.metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    format_classification_metrics,
    format_metrics,
)
from evaluation.plots import (
    plot_classification_threshold_comparison,
    plot_cumulative_returns,
    plot_errors,
    plot_fold_metrics,
    plot_predictions,
)
from evaluation.trading import evaluate_strategy, format_strategy_metrics
from utils import set_seed, ensure_dirs, device


MODEL_NAMES = {
    "y_zero": "Zero Baseline",
    "y_mean": "Mean Baseline",
    "y_ridge": "Ridge Baseline",
    "y_pred": "BiLSTM",
}


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


def print_metric_row(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {label:<14} {format_metrics(compute_all_metrics(y_true, y_pred))}")


def print_strategy_row(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"  {label:<14} {format_strategy_metrics(evaluate_strategy(y_true, y_pred))}")


def build_fold_rows(fold_idx: int, result: dict, dates: np.ndarray) -> list[dict]:
    rows = []
    date_start = pd.Timestamp(dates[0]).date().isoformat()
    date_end = pd.Timestamp(dates[-1]).date().isoformat()

    for pred_key, model_name in MODEL_NAMES.items():
        pred_metrics = compute_all_metrics(result["y_true"], result[pred_key])
        trading_metrics = evaluate_strategy(result["y_true"], result[pred_key])

        row = {
            "fold": fold_idx,
            "model": model_name,
            "test_start_idx": result["test_start"],
            "test_end_idx": result["test_end"],
            "date_start": date_start,
            "date_end": date_end,
            "n_obs": len(result["y_true"]),
        }
        row.update(pred_metrics)
        row.update({f"trading_{key}": value for key, value in trading_metrics.items()})
        rows.append(row)

    return rows


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> dict:
    metric_cols = [
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
        "correlation",
        "trading_total_return",
        "trading_buy_hold_return",
        "trading_sharpe",
        "trading_max_drawdown",
        "trading_exposure",
        "trading_num_trades",
    ]

    summary = {}
    for model_name, group in fold_metrics.groupby("model"):
        summary[model_name] = {}
        for col in metric_cols:
            summary[model_name][f"{col}_mean"] = float(group[col].mean())
            summary[model_name][f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0

    return summary


def build_classification_fold_rows(
    fold_idx: int,
    reg_result: dict,
    clf_result: dict,
    thresholds: tuple[float, ...],
) -> list[dict]:
    rows = []
    y_true_return = reg_result["y_true"]
    y_true_direction = (y_true_return > 0).astype(np.float32)

    candidates = [
        ("Mean Baseline", reg_result["y_mean"], 0.0),
        ("Regression BiLSTM", reg_result["y_pred"], 0.0),
    ]
    candidates.extend(("Classification BiLSTM", clf_result["prob"], threshold) for threshold in thresholds)

    for model_name, score, threshold in candidates:
        class_metrics = compute_classification_metrics(y_true_direction, score, threshold=threshold)
        trading_metrics = evaluate_strategy(y_true_return, score, threshold=threshold)
        row = {
            "fold": fold_idx,
            "model": model_name,
            "threshold": threshold,
            "n_obs": len(y_true_return),
        }
        row.update(class_metrics)
        row.update({f"trading_{key}": value for key, value in trading_metrics.items()})
        rows.append(row)

    return rows


def summarize_classification_metrics(classification_folds: pd.DataFrame) -> dict:
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "trading_total_return",
        "trading_buy_hold_return",
        "trading_sharpe",
        "trading_max_drawdown",
        "trading_exposure",
        "trading_num_trades",
    ]
    summary = {}
    for (model_name, threshold), group in classification_folds.groupby(["model", "threshold"]):
        key = f"{model_name}@{threshold:.2f}"
        summary[key] = {"model": model_name, "threshold": float(threshold)}
        for col in metric_cols:
            summary[key][f"{col}_mean"] = float(group[col].mean())
            summary[key][f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0

    return summary


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

    train_mean_target = float(np.mean(y_train_raw))
    if cfg.demean_target:
        y_train_model = y_train_raw - train_mean_target
        y_val_model = y_val_raw - train_mean_target
        y_test_model = y_test_raw - train_mean_target
    else:
        y_train_model = y_train_raw
        y_val_model = y_val_raw
        y_test_model = y_test_raw

    Xtr_s, ytr_s = make_sequences(X_train, y_train_model.astype(np.float32), cfg.lookback)
    Xva_s, yva_s = make_sequences(X_val, y_val_model.astype(np.float32), cfg.lookback)

    # test sequences only for targets in [test_start, test_end)
    Xte_s, yte_s = build_sequences_for_block(
        full_X=X_test,
        full_y=y_test_model.astype(np.float32),
        start_idx=test_start,
        end_idx=test_end,
        lookback=cfg.lookback,
    )
    _, yte_true = build_sequences_for_block(
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

    with torch.no_grad():
        for xb, _ in te_loader:
            xb = xb.to(dev)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)

    preds = np.concatenate(preds)

    if cfg.demean_target:
        preds = preds + train_mean_target

    trues = yte_true
    pred_zero = baseline_zero(trues)
    pred_mean = baseline_mean(trues, train_mean_target)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_model)

    ridge_test_block = scaler.transform(X_all[test_start:test_end]).astype(np.float32)
    ridge_pred = ridge.predict(ridge_test_block).astype(np.float32)
    if cfg.demean_target:
        ridge_pred = ridge_pred + train_mean_target

    return {
        "test_start": test_start,
        "test_end": test_end,
        "y_true": trues,
        "y_pred": preds,
        "y_zero": pred_zero,
        "y_mean": pred_mean,
        "y_ridge": ridge_pred,
    }


def fit_classifier_one_fold(
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
    y_train_raw = (y_all[:train_end] > 0).astype(np.float32)
    X_val_raw = X_all[train_end:val_end]
    y_val_raw = (y_all[train_end:val_end] > 0).astype(np.float32)
    X_test_raw = X_all[:test_end]
    y_test_raw = (y_all[:test_end] > 0).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    Xtr_s, ytr_s = make_sequences(X_train, y_train_raw, cfg.lookback)
    Xva_s, yva_s = make_sequences(X_val, y_val_raw, cfg.lookback)
    Xte_s, yte_s = build_sequences_for_block(
        full_X=X_test,
        full_y=y_test_raw,
        start_idx=test_start,
        end_idx=test_end,
        lookback=cfg.lookback,
    )

    if len(Xtr_s) == 0 or len(Xva_s) == 0 or len(Xte_s) == 0:
        return None

    model = BiLSTMClassifier(
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
    loss_fn = nn.BCEWithLogitsLoss()

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

    for _ in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                val_losses.append(loss_fn(logits, yb).item())

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

    probs = []
    with torch.no_grad():
        for xb, _ in te_loader:
            xb = xb.to(dev)
            prob = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs.append(prob)

    return {
        "prob": np.concatenate(probs),
    }

def walk_forward_evaluate(cfg: Config):
    set_seed(cfg.seed)
    out_metrics = "outputs/metrics"
    ensure_dirs(cfg.out_plots, out_metrics)

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
    fold_rows = []
    classification_rows = []
    all_clf_prob = []
    regression_results = []

    print(f"Running {len(folds)} regression walk-forward folds...")

    for i, fold in enumerate(tqdm(folds, desc="Regression folds")):
        result = fit_one_fold(X_all, y_all, cfg, fold)
        if result is None:
            continue

        test_start = result["test_start"]
        test_end = result["test_end"]
        regression_results.append((i, fold, result))

        all_true.append(result["y_true"])
        all_pred.append(result["y_pred"])
        all_zero.append(result["y_zero"])
        all_mean.append(result["y_mean"])
        all_ridge.append(result["y_ridge"])
        all_dates.append(dates_all[test_start:test_end])
        fold_rows.extend(build_fold_rows(i + 1, result, dates_all[test_start:test_end]))

        print(
            f"Fold {i+1:02d} | "
            f"Reg {format_metrics(compute_all_metrics(result['y_true'], result['y_pred']))}"
        )

    print(f"\nRunning {len(regression_results)} classification walk-forward folds...")

    for i, fold, result in tqdm(regression_results, desc="Classification folds"):
        clf_result = fit_classifier_one_fold(X_all, y_all, cfg, fold)
        if clf_result is None:
            continue

        all_clf_prob.append(clf_result["prob"])
        classification_rows.extend(
            build_classification_fold_rows(i + 1, result, clf_result, cfg.classification_thresholds)
        )

        fold_class_metrics = compute_classification_metrics(
            (result["y_true"] > 0).astype(np.float32),
            clf_result["prob"],
            threshold=0.5,
        )
        print(
            f"Fold {i+1:02d} | "
            f"Cls@0.5 {format_classification_metrics(fold_class_metrics)}"
        )

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_zero = np.concatenate(all_zero)
    y_mean = np.concatenate(all_mean)
    y_ridge = np.concatenate(all_ridge)
    clf_prob = np.concatenate(all_clf_prob)
    dates = np.concatenate(all_dates)
    fold_metrics = pd.DataFrame(fold_rows)
    classification_folds = pd.DataFrame(classification_rows)

    fold_metrics_path = f"{out_metrics}/walk_forward_folds.csv"
    summary_path = f"{out_metrics}/walk_forward_summary.json"
    classification_results_path = f"{out_metrics}/classification_results.json"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    full_timeline = {}
    for pred_key, model_name in MODEL_NAMES.items():
        pred = {
            "y_zero": y_zero,
            "y_mean": y_mean,
            "y_ridge": y_ridge,
            "y_pred": y_pred,
        }[pred_key]
        full_timeline[model_name] = {
            "prediction_metrics": compute_all_metrics(y_true, pred),
            "trading_metrics": evaluate_strategy(y_true, pred),
        }

    summary = {
        "ticker": cfg.ticker,
        "start": cfg.start,
        "end": cfg.end,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "demean_target": cfg.demean_target,
        "n_folds": int(fold_metrics["fold"].nunique()),
        "n_observations": int(len(y_true)),
        "fold_mean_std": summarize_fold_metrics(fold_metrics),
        "full_timeline": full_timeline,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    classification_full_timeline = {
        "Mean Baseline@0.00": {
            "classification_metrics": compute_classification_metrics((y_true > 0).astype(np.float32), y_mean, 0.0),
            "trading_metrics": evaluate_strategy(y_true, y_mean, threshold=0.0),
        },
        "Regression BiLSTM@0.00": {
            "classification_metrics": compute_classification_metrics((y_true > 0).astype(np.float32), y_pred, 0.0),
            "trading_metrics": evaluate_strategy(y_true, y_pred, threshold=0.0),
        },
    }
    for threshold in cfg.classification_thresholds:
        classification_full_timeline[f"Classification BiLSTM@{threshold:.2f}"] = {
            "classification_metrics": compute_classification_metrics(
                (y_true > 0).astype(np.float32),
                clf_prob,
                threshold=threshold,
            ),
            "trading_metrics": evaluate_strategy(y_true, clf_prob, threshold=threshold),
        }

    classification_results = {
        "ticker": cfg.ticker,
        "start": cfg.start,
        "end": cfg.end,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "thresholds": list(cfg.classification_thresholds),
        "n_folds": int(classification_folds["fold"].nunique()),
        "n_observations": int(len(y_true)),
        "fold_mean_std": summarize_classification_metrics(classification_folds),
        "full_timeline": classification_full_timeline,
    }
    with open(classification_results_path, "w", encoding="utf-8") as f:
        json.dump(classification_results, f, indent=2)

    print("\nWalk-Forward Full-Timeline Prediction Metrics")
    print_metric_row("Zero Baseline", y_true, y_zero)
    print_metric_row("Mean Baseline", y_true, y_mean)
    print_metric_row("Ridge Baseline", y_true, y_ridge)
    print_metric_row("BiLSTM", y_true, y_pred)

    print("\nWalk-Forward Full-Timeline Trading Metrics")
    print_strategy_row("Zero Baseline", y_true, y_zero)
    print_strategy_row("Mean Baseline", y_true, y_mean)
    print_strategy_row("Ridge Baseline", y_true, y_ridge)
    print_strategy_row("BiLSTM", y_true, y_pred)

    print("\nWalk-Forward Fold Mean +/- Std")
    for model_name, values in summary["fold_mean_std"].items():
        print(
            f"  {model_name:<14} "
            f"MAE={values['mae_mean']:.6f}+/-{values['mae_std']:.6f} "
            f"RMSE={values['rmse_mean']:.6f}+/-{values['rmse_std']:.6f} "
            f"DA={values['directional_accuracy_mean']:.2f}+/-{values['directional_accuracy_std']:.2f}% "
            f"Sharpe={values['trading_sharpe_mean']:.2f}+/-{values['trading_sharpe_std']:.2f}"
        )

    print("\nClassification Decision Layer")
    for name, values in classification_full_timeline.items():
        print(
            f"  {name:<27} "
            f"{format_classification_metrics(values['classification_metrics'])} | "
            f"{format_strategy_metrics(values['trading_metrics'])}"
        )

    predictions_plot = f"{cfg.out_plots}/{cfg.ticker}_walk_forward_predictions.png"
    errors_plot = f"{cfg.out_plots}/{cfg.ticker}_walk_forward_prediction_errors.png"
    returns_plot = f"{cfg.out_plots}/{cfg.ticker}_walk_forward_cumulative_returns.png"
    fold_metrics_plot = f"{cfg.out_plots}/{cfg.ticker}_walk_forward_fold_metrics.png"
    classification_plot = f"{cfg.out_plots}/classification_threshold_comparison.png"

    plot_predictions(y_true, y_pred, dates=dates, output_path=predictions_plot)
    plot_errors(y_true, y_pred, dates=dates, output_path=errors_plot)
    plot_cumulative_returns(y_true, y_pred, dates=dates, output_path=returns_plot)
    plot_fold_metrics(fold_metrics, output_path=fold_metrics_plot)
    plot_classification_threshold_comparison(classification_folds, output_path=classification_plot)

    print(f"\nSaved metrics: {fold_metrics_path}")
    print(f"Saved metrics: {summary_path}")
    print(f"Saved metrics: {classification_results_path}")
    print(f"Saved plot: {predictions_plot}")
    print(f"Saved plot: {errors_plot}")
    print(f"Saved plot: {returns_plot}")
    print(f"Saved plot: {fold_metrics_plot}")
    print(f"Saved plot: {classification_plot}")


if __name__ == "__main__":
    cfg = Config()
    walk_forward_evaluate(cfg)
