# Deep Sequence Modeling for Short-Horizon Equity Return Forecasting

A rigorous financial time-series forecasting project that reframes naive stock price prediction into a more defensible **short-horizon forward return prediction** problem.

Instead of predicting raw future prices, this project models the **5-day forward log return of SPY** using engineered technical/statistical features and a **Bidirectional LSTM (BiLSTM)** in PyTorch. The pipeline includes leakage-aware preprocessing, chronological evaluation, baseline benchmarking, a demeaned-target experiment, and expanding-window walk-forward evaluation.

## Project goal

The goal is not to build a flashy “stock predictor,” but to test whether a sequential deep learning model can extract a useful short-term signal from historical market data under realistic evaluation.

This project asks:

- Can a BiLSTM improve on naive return baselines?
- Do fixed-split results survive more realistic walk-forward testing?
- Is the model learning genuine predictive structure, or mostly average market drift?

## Why this project is stronger than a typical beginner stock project

This repository avoids several common mistakes in financial ML:

- it predicts **forward returns**, not raw prices
- it uses **chronological splits**, not random train/test shuffling
- feature scaling is fit on **train only**
- it compares against **naive and classical baselines**
- it includes **walk-forward evaluation**, not just one lucky test split
- it tests **demeaned targets** to separate drift from predictive signal

## Problem formulation

For each date \( t \), the target is the **5-day forward log return**:

\[
y_t = \log\left(\frac{Close_{t+5}}{Close_t}\right)
\]

The input is a rolling sequence of the previous `lookback` days of engineered market features.

This makes the task a **sequence-to-one regression problem**:
predict the future short-horizon return from recent market behavior.

## Features

The model uses an explicit feature set based on normalized price and volume information:

- 1, 5, and 10-day log returns
- rolling volatility
- momentum features
- RSI
- moving-average gaps
- EMA gaps
- high-low range
- open-close return
- volume change
- volume ratio

These features are designed to be more stable and interpretable than raw price-level prediction alone.

## Models and baselines

### Main model
- **BiLSTM regressor** implemented in PyTorch

### Baselines
- **Zero-return baseline**
- **Mean-return baseline**
- **Ridge regression baseline** (walk-forward benchmark)

## Evaluation protocols

### 1. Fixed chronological split
A standard train / validation / test split with:
- train-only feature scaling
- early stopping
- Huber loss
- held-out test evaluation

### 2. Expanding-window walk-forward evaluation
A more realistic sequential evaluation where the model is repeatedly retrained on past data and tested on the next unseen block.

This is closer to how a forecasting system would be assessed in practice.

## Demeaned-target experiment

To test whether the model is learning deviations from average market drift rather than only the unconditional mean, the project includes a **demeaned-target formulation**:

\[
y_t^{\text{demeaned}} = y_t - \mu_{\text{train}}
\]

Predictions are then shifted back to the original scale for evaluation.

This helps distinguish:
- predicting the average return regime
from
- predicting excess short-horizon variation

## Main findings

### Fixed-split evaluation
The BiLSTM can produce modest improvement over the **zero baseline**, but does **not consistently beat the mean-return baseline**.

### Walk-forward evaluation
Under the more realistic expanding-window walk-forward protocol, performance weakens noticeably. Neither the BiLSTM nor the Ridge baseline consistently outperform the mean-return baseline.

### Interpretation
This is an important result rather than a failure:

- apparent signal on a single fixed split may not survive robust out-of-sample evaluation
- financial return prediction is highly noisy
- strong methodology matters more than optimistic one-off metrics

## Key results

### Fixed-split test evaluation
- Zero baseline: MAE = 0.014505, RMSE = 0.018616
- Mean baseline: MAE = 0.014054, RMSE = 0.018220
- BiLSTM: MAE = 0.014468, RMSE = 0.018407, Directional Accuracy = 58.79%, Correlation = 0.0849

### Walk-forward evaluation
- Zero baseline: MAE = 0.017968, RMSE = 0.025462
- Mean baseline: MAE = 0.017694, RMSE = 0.025391
- Ridge baseline: MAE = 0.018543, RMSE = 0.026265
- BiLSTM: MAE = 0.019907, RMSE = 0.028306, Directional Accuracy = 52.03%, Correlation = 0.0956

### Interpretation
The fixed-split evaluation suggests modest predictive structure, but the more realistic walk-forward evaluation shows that the signal is not robust enough to consistently outperform simple baselines. This highlights the importance of rigorous out-of-sample testing in financial ML.

## Current conclusion

This project demonstrates a complete and disciplined financial ML workflow:

- reframing the target correctly
- building a reproducible deep learning pipeline
- preventing leakage
- benchmarking against simple alternatives
- testing robustness with walk-forward evaluation
- interpreting results honestly

The main lesson is that **evaluation rigor is essential** in financial time-series modeling: a promising fixed-split result is not enough on its own.

## Repository structure

- `src/config.py` — experiment settings and feature list
- `src/data_loader.py` — market data loading
- `src/features.py` — feature engineering and target construction
- `src/dataset.py` — rolling sequence generation for PyTorch
- `src/model.py` — BiLSTM model definition
- `src/baselines.py` — baselines and evaluation metrics
- `src/train.py` — fixed-split training pipeline
- `src/evaluate.py` — held-out test evaluation
- `src/walk_forward.py` — expanding-window walk-forward evaluation
- `src/utils.py` — utility functions
- `outputs/models/` — saved checkpoints
- `outputs/plots/` — generated plots

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
