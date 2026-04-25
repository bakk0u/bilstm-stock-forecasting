# Final Report: Short-Horizon Equity Return Forecasting With a BiLSTM

## 1. Methodology

This project studies whether a sequence model can forecast short-horizon equity returns from historical market features. The target is the 5-day forward log return of SPY, not the future price level.

The workflow follows a conservative financial ML process:

1. Build features using only current and historical OHLCV information.
2. Construct a forward-return target.
3. Use chronological train/validation/test splits.
4. Fit scalers only on training data.
5. Compare the BiLSTM against simple baselines.
6. Re-evaluate with expanding-window walk-forward validation.
7. Add a simple long/flat trading diagnostic, while avoiding claims that it is a production backtest.

The project is designed to be explainable in an interview. Each stage has a clear purpose and avoids unnecessary complexity.

## 2. Dataset

The dataset is daily SPY OHLCV data downloaded with `yfinance`.

Configuration used in the latest run:

- Ticker: SPY
- Start date: 2015-01-01
- End date: 2025-01-01
- Forecast horizon: 5 trading days
- Sequence lookback: 60 trading days

SPY is used because it is liquid, widely followed, and suitable for demonstrating an equity-index forecasting workflow. The project does not claim that SPY is the best asset for this modeling approach.

## 3. Feature Engineering

Features are derived from price and volume history:

- 1-day, 5-day, and 10-day log returns
- rolling volatility over 10 and 30 days
- 10-day and 30-day momentum
- RSI
- moving-average gaps
- exponential moving-average gaps
- high-low range
- open-close return
- volume change
- volume ratio versus a 10-day average

The features are intentionally standard and interpretable. This makes the project easier to explain and keeps the focus on evaluation discipline rather than on opaque feature search.

## 4. Target Construction

For date `t`, the target is:

```text
y_t = log(Close[t + 5] / Close[t])
```

This target represents the forward 5-day log return. It is more defensible than predicting raw future prices because returns are closer to stationary and can be compared directly across time.

The target can optionally be demeaned:

```text
y_model = y - mean(y_train)
```

Predictions are shifted back to the original return scale for evaluation. In walk-forward evaluation, the mean is recomputed only from each fold's training window.

## 5. Model

The main model is a PyTorch BiLSTM regressor. It receives a rolling sequence of engineered features and predicts one forward return.

The architecture is deliberately modest:

- 1 recurrent layer
- hidden size 32
- dropout
- LayerNorm and feedforward regression head
- Huber loss by default
- AdamW optimizer
- early stopping on validation loss

The model is not presented as state of the art. It is used as a sequence-learning benchmark within a disciplined evaluation pipeline.

## 6. Evaluation Setup

### Fixed Split

The fixed split uses a chronological train/validation/test split. The scaler is fit on the training split only, then applied to validation and test data.

The fixed split is useful for fast iteration, but it can be sensitive to the specific market regime in the final test block.

### Walk-Forward Validation

The walk-forward setup uses expanding windows:

1. Train on historical data.
2. Validate on the next block.
3. Test on the following unseen block.
4. Move forward and repeat.

This is closer to how a forecasting model would be evaluated in practice. It also makes it harder for a single favorable test period to dominate the interpretation.

### Baselines

The project compares against:

- zero-return baseline
- training-mean return baseline
- Ridge regression baseline in walk-forward evaluation

These baselines are essential. A complex model must beat simple alternatives to justify its complexity.

### Metrics

Prediction metrics:

- MAE
- RMSE
- safe MAPE, excluding near-zero actual returns
- directional accuracy
- Pearson correlation

Trading diagnostic metrics:

- total return
- buy-and-hold return
- Sharpe ratio
- max drawdown
- exposure
- number of trades

## 7. Leakage Prevention

The pipeline includes several leakage controls:

- All splits are chronological.
- Scaling is fit only on training data.
- Walk-forward folds refit preprocessing within each fold.
- Target demeaning uses only the relevant training window.
- Validation and test periods are never used to fit scalers or target means.
- Features are based on current and historical observations, while the target is shifted forward.

One important caveat is that the target is a 5-day overlapping forward return. This is common in short-horizon return studies but must be considered when interpreting trading-style cumulative return diagnostics.

## 8. Results Interpretation

### Fixed-Split Results

Latest fixed-split test metrics:

| Model | MAE | RMSE | MAPE | Directional Accuracy | Correlation |
|---|---:|---:|---:|---:|---:|
| Zero baseline | 0.014505 | 0.018616 | 100.00% | 0.00% | 0.0000 |
| Mean baseline | 0.014054 | 0.018220 | 109.99% | 66.45% | 0.0000 |
| BiLSTM | 0.014760 | 0.018896 | 177.35% | 58.15% | 0.0027 |

The BiLSTM does not beat the mean baseline on MAE or RMSE. Its correlation is close to zero, suggesting weak linear association between predictions and realized forward returns in the test period.

### Walk-Forward Results

Latest full-timeline walk-forward metrics:

| Model | MAE | RMSE | MAPE | Directional Accuracy | Correlation |
|---|---:|---:|---:|---:|---:|
| Zero baseline | 0.017968 | 0.025462 | 100.00% | 0.06% | 0.0000 |
| Mean baseline | 0.017701 | 0.025390 | 107.58% | 61.30% | -0.0558 |
| Ridge baseline | 0.018543 | 0.026265 | 147.64% | 47.08% | -0.0142 |
| BiLSTM | 0.020567 | 0.030070 | 217.86% | 53.36% | -0.0125 |

Fold-level mean and standard deviation:

| Model | MAE Mean | MAE Std | RMSE Mean | RMSE Std | DA Mean | DA Std |
|---|---:|---:|---:|---:|---:|---:|
| Zero baseline | 0.017861 | 0.007779 | 0.022653 | 0.011196 | 0.05% | 0.30% |
| Mean baseline | 0.017664 | 0.007917 | 0.022525 | 0.011387 | 59.35% | 16.80% |
| Ridge baseline | 0.018296 | 0.008282 | 0.023048 | 0.012048 | 48.76% | 13.56% |
| BiLSTM | 0.020838 | 0.010533 | 0.026627 | 0.014138 | 51.67% | 13.90% |

The walk-forward results are the most important results in the project. They show that the BiLSTM does not produce robust improvement over simple baselines in this configuration.

## 9. Trading Diagnostic

The project includes a simple long/flat diagnostic:

```text
signal_t = 1 if predicted_return_t > 0 else 0
strategy_log_return_t = signal_t * actual_forward_log_return_t - transaction_cost_on_signal_change
cumulative_return = exp(cumsum(strategy_log_returns)) - 1
```

Default transaction cost is `0.0005` per signal change.

This diagnostic is useful for inspecting whether predictions create a stable directional signal. It is not a production backtest. Because the project predicts overlapping 5-day forward returns, compounding those returns can exaggerate economic interpretation. The diagnostic should be read as a model comparison tool, not as executable performance.

Latest walk-forward diagnostic:

| Model | Total Return | Buy-and-Hold | Sharpe | Max Drawdown | Exposure | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Zero baseline | 0.00% | 6522.91% | 0.00 | 0.00% | 0.00% | 0 |
| Mean baseline | 6519.60% | 6522.91% | 3.90 | 83.63% | 100.00% | 1 |
| Ridge baseline | 127.03% | 6522.91% | 0.91 | 74.16% | 51.36% | 358 |
| BiLSTM | 1466.98% | 6522.91% | 3.26 | 72.14% | 62.52% | 535 |

The BiLSTM is active and sometimes avoids exposure, but it does not clearly dominate the simpler mean baseline in this diagnostic.

## 10. Feature Ablation Diagnostic

The project includes a feature ablation diagnostic to better understand whether broad groups of engineered features appear helpful under walk-forward evaluation.

Feature groups:

- price-based: log returns and momentum
- trend: SMA and EMA gaps
- volatility: rolling volatility and high-low range
- RSI: `rsi_14`

The ablation uses a Ridge walk-forward proxy rather than the BiLSTM. This keeps runtime reasonable and makes the result easier to interpret. It should not be read as causal feature importance, and it may not transfer directly to the neural network.

Latest ablation results:

| Experiment | Removed Group | MAE | RMSE | Directional Accuracy | Trading Sharpe |
|---|---|---:|---:|---:|---:|
| Full model | none | 0.018543 | 0.026265 | 47.08% | 0.91 |
| Minus price-based | price-based | 0.018445 | 0.026157 | 48.36% | 1.56 |
| Minus trend | trend | 0.018030 | 0.025823 | 51.58% | 2.32 |
| Minus volatility | volatility | 0.018164 | 0.025812 | 50.58% | 2.76 |
| Minus RSI | RSI | 0.018527 | 0.026172 | 47.20% | 0.90 |

In this Ridge proxy, removing trend and volatility features improved some metrics, which suggests those indicators may be redundant or noisy for this short-horizon setup. RSI showed limited incremental value. Removing price-based features reduced performance compared with the best ablations, although it still slightly improved over the full Ridge feature set in this run.

These results are diagnostic and model-dependent. They suggest where further feature review may be useful, but they do not prove that any feature group has causal predictive value.

## 11. How To Explain This Project In Interviews

- I predicted 5-day forward log returns, not prices.
- I used chronological and walk-forward validation to avoid leakage.
- I compared against simple baselines because financial ML models often fail to beat them.
- I added trading diagnostics to connect prediction quality with decision-making.
- I extended the project with classification thresholds and feature ablation to analyze signal quality.
- The main finding is that short-horizon equity return prediction is noisy, and simple baselines remain hard to beat.

## 12. Limitations

Important limitations:

- Only one asset is modeled.
- The feature set is simple and technical-indicator based.
- The target uses overlapping 5-day forward returns.
- The trading diagnostic is not an executable portfolio backtest.
- Transaction costs are simplified.
- No slippage, borrow constraints, market impact, or realistic execution assumptions are modeled.
- Hyperparameter tuning is limited.
- The BiLSTM is compared against simple baselines, but not against a broad set of classical time-series or tree-based models.
- The feature ablation uses a Ridge proxy, so it is diagnostic rather than direct BiLSTM feature attribution.
- Results can vary across random seeds and market regimes.

These limitations are part of the honest interpretation. The project demonstrates process quality more than predictive success.

## 13. Future Improvements

Possible next steps:

- Add non-overlapping evaluation windows for cleaner trading interpretation.
- Expand to multiple ETFs or equities.
- Add macro, rates, volatility-index, or cross-asset features.
- Compare against gradient-boosted trees and regularized linear models.
- Use purged or embargoed validation for overlapping-label experiments.
- Add confidence intervals or bootstrap analysis for metric uncertainty.
- Tune thresholds for the long/flat signal using validation data only.
- Separate forecasting evaluation from a more realistic daily rebalanced backtest.
- Track experiment metadata and random seeds more formally.

## 14. Conclusion

The main result is negative but useful: in this setup, the BiLSTM does not outperform simple baselines under walk-forward evaluation. The classification and feature ablation extensions add useful diagnostic context, but they do not change the central takeaway. This project shows the importance of baselines, chronological validation, leakage controls, and conservative interpretation in financial machine learning.
