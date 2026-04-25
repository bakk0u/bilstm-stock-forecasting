from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    ticker: str = "SPY"
    start: str = "2015-01-01"
    end: str = "2025-01-01"

    # Forecasting setup
    lookback: int = 60
    horizon: int = 5
    target_type: str = "forward_log_return"

    feature_cols: list[str] = field(default_factory=lambda: [
        "log_ret_1",
        "log_ret_5",
        "log_ret_10",
        "vol_10",
        "vol_30",
        "mom_10",
        "mom_30",
        "rsi_14",
        "ma_gap_10",
        "ma_gap_30",
        "ema_gap_10",
        "ema_gap_30",
        "hl_range",
        "oc_return",
        "volume_change",
        "volume_ratio_10",
    ])

    # Fixed split benchmark
    train_ratio: float = 0.75
    val_ratio: float = 0.10

    # Walk-forward evaluation
    use_walk_forward: bool = True
    walk_step: int = 60
    min_train_size: int = 500
    val_size: int = 180
    classification_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7)

    # Model
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42

    # Loss
    loss_name: str = "huber"
    huber_delta: float = 1.0

    # Target handling
    demean_target: bool = True

    # Output
    out_models: str = "outputs/models"
    out_plots: str = "outputs/plots"
