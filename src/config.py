from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    ticker: str = "SPY"
    start: str = "2015-01-01"
    end: str = "2025-01-01"

    # Feature + target setup
    lookback: int = 60
    horizon: int = 1  # predict t+horizon close
    target_col: str = "Close"

    # Train / eval split
    train_ratio: float = 0.75
    val_ratio: float = 0.10  # remaining goes to test

    # Walk-forward evaluation (more credible than random split)
    

    # Model
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42

    # Output
    out_models: str = "outputs/models"
    out_plots: str = "outputs/plots"

