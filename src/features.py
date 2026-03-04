import pandas as pd
import numpy as np

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)  # neutral RSI for early points

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1"] = df["Close"].pct_change()
    df["log_ret_1"] = (df["Close"].apply(np.log)).diff()

    # Moving averages
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_30"] = df["Close"].rolling(30).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_30"] = df["Close"].ewm(span=30, adjust=False).mean()

    # Volatility (rolling std of returns)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()

    # Momentum
    df["mom_10"] = df["Close"].pct_change(10)
    df["mom_30"] = df["Close"].pct_change(30)

    # RSI
    df["rsi_14"] = rsi(df["Close"], 14)

    # Fill
    df = df.dropna().reset_index(drop=True)
    return df

def make_supervised(df: pd.DataFrame, target_col: str, horizon: int) -> pd.DataFrame:
    """
    Adds target column: next_horizon_close
    """
    df = df.copy()
    df["target"] = df[target_col].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df
