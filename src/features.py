import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)

    df["log_ret_1"] = np.log(close).diff(1)
    df["log_ret_5"] = np.log(close / close.shift(5))
    df["log_ret_10"] = np.log(close / close.shift(10))

    # Rolling volatility on 1-day log returns
    df["vol_10"] = df["log_ret_1"].rolling(10).std()
    df["vol_30"] = df["log_ret_1"].rolling(30).std()

    # Moving averages and relative gaps
    sma_10 = close.rolling(10).mean()
    sma_30 = close.rolling(30).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()

    df["ma_gap_10"] = (close - sma_10) / sma_10
    df["ma_gap_30"] = (close - sma_30) / sma_30
    df["ema_gap_10"] = (close - ema_10) / ema_10
    df["ema_gap_30"] = (close - ema_30) / ema_30

    # Momentum
    df["mom_10"] = close.pct_change(10)
    df["mom_30"] = close.pct_change(30)

    # RSI
    df["rsi_14"] = rsi(close, 14)

    # Intraday structure
    df["hl_range"] = (high - low) / close
    df["oc_return"] = (close - open_) / open_

    # Volume signals
    df["volume_change"] = volume.pct_change()
    vol_ma_10 = volume.rolling(10).mean()
    df["volume_ratio_10"] = volume / vol_ma_10

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def make_supervised(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Target = forward log return over `horizon` days:
        target_t = log(C_{t+h} / C_t)
    """
    df = df.copy()
    df["target"] = np.log(df["Close"].shift(-horizon) / df["Close"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df