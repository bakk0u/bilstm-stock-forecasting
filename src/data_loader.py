import pandas as pd
import yfinance as yf

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    df = df.rename_axis("Date").reset_index()
    # Standardize columns
    keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df
