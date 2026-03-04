# BiLSTM Stock Forecasting (Reproducible ML Pipeline)

End-to-end time-series forecasting project:
- download market data
- engineer technical indicators
- train a BiLSTM regressor in PyTorch
- compare against a strong baseline (predict tomorrow = today)
- evaluate on a time-based split + save plots and metrics

## Why this project is credible
Time series is evaluated chronologically (no random split).
Feature scaling is fit on train only.
A baseline model is included to validate that the deep model adds value.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
