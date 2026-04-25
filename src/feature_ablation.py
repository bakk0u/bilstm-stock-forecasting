import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import Config
from data_loader import fetch_ohlcv
from evaluation.metrics import compute_all_metrics
from evaluation.trading import evaluate_strategy
from features import add_indicators, make_supervised
from utils import ensure_dirs, set_seed
from walk_forward import build_walk_forward_folds


FEATURE_GROUPS = {
    "price_based": ["log_ret_1", "log_ret_5", "log_ret_10", "mom_10", "mom_30"],
    "trend": ["ma_gap_10", "ma_gap_30", "ema_gap_10", "ema_gap_30"],
    "volatility": ["vol_10", "vol_30", "hl_range"],
    "rsi": ["rsi_14"],
}


def feature_sets(cfg: Config) -> dict[str, list[str]]:
    full = list(cfg.feature_cols)
    experiments = {"full_model": full}

    for group_name, group_cols in FEATURE_GROUPS.items():
        remove = set(group_cols)
        experiments[f"minus_{group_name}"] = [col for col in full if col not in remove]

    return experiments


def fit_predict_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_idx: list[int],
    fold: dict,
) -> tuple[np.ndarray, np.ndarray]:
    train_end = fold["train_end"]
    test_start = fold["test_start"]
    test_end = fold["test_end"]

    X_train_raw = X_all[:train_end, :][:, feature_idx]
    y_train = y_all[:train_end]
    X_test_raw = X_all[test_start:test_end, :][:, feature_idx]
    y_test = y_all[test_start:test_end]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32)

    return y_test.astype(np.float32), pred


def evaluate_feature_set(
    X_all: np.ndarray,
    y_all: np.ndarray,
    all_feature_cols: list[str],
    selected_features: list[str],
    folds: list[dict],
) -> dict[str, float]:
    feature_idx = [all_feature_cols.index(col) for col in selected_features]
    y_true_parts = []
    y_pred_parts = []

    for fold in folds:
        y_true, y_pred = fit_predict_fold(X_all, y_all, feature_idx, fold)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    pred_metrics = compute_all_metrics(y_true, y_pred)
    trading_metrics = evaluate_strategy(y_true, y_pred)

    return {
        "n_features": len(selected_features),
        "mae": pred_metrics["mae"],
        "rmse": pred_metrics["rmse"],
        "directional_accuracy": pred_metrics["directional_accuracy"],
        "trading_sharpe": trading_metrics["sharpe"],
    }


def plot_ablation_results(results: pd.DataFrame, output_path: str) -> None:
    ordered = results.sort_values("mae")
    labels = ordered["experiment"]
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, ordered["mae"])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Feature Ablation: Walk-Forward Ridge Proxy")

    axes[1].bar(x, ordered["directional_accuracy"])
    axes[1].set_ylabel("Direction Acc. (%)")

    axes[2].bar(x, ordered["trading_sharpe"])
    axes[2].set_ylabel("Trading Sharpe")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_feature_ablation(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    out_metrics = "outputs/metrics"
    ensure_dirs(out_metrics, cfg.out_plots)

    df = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end)
    df = add_indicators(df)
    df = make_supervised(df, cfg.horizon)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    missing = [col for col in cfg.feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing configured feature columns: {missing}")

    X_all = df[cfg.feature_cols].values.astype(np.float32)
    y_all = df["target"].values.astype(np.float32)
    folds = build_walk_forward_folds(
        n_obs=len(df),
        min_train_size=cfg.min_train_size,
        val_size=cfg.val_size,
        walk_step=cfg.walk_step,
    )

    rows = []
    for experiment, selected_features in feature_sets(cfg).items():
        metrics = evaluate_feature_set(
            X_all=X_all,
            y_all=y_all,
            all_feature_cols=list(cfg.feature_cols),
            selected_features=selected_features,
            folds=folds,
        )
        rows.append({
            "experiment": experiment,
            "removed_group": "none" if experiment == "full_model" else experiment.replace("minus_", ""),
            **metrics,
        })

    results = pd.DataFrame(rows)
    results_path = f"{out_metrics}/feature_ablation.csv"
    plot_path = f"{cfg.out_plots}/feature_ablation.png"
    results.to_csv(results_path, index=False)
    plot_ablation_results(results, plot_path)

    print("Feature Ablation Results")
    print(results.sort_values("mae").to_string(index=False))
    print(f"\nSaved metrics: {results_path}")
    print(f"Saved plot: {plot_path}")

    return results


if __name__ == "__main__":
    run_feature_ablation(Config())
