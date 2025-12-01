# core/strategies.py

import pandas as pd
import ta


def apply_wyckoff_swing_rules(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["ma"] = close.rolling(50).mean()
    df["rsi"] = ta.momentum.rsi(close, window=14)

    df["ma_prev"] = df["ma"].shift(1)
    df["close_prev"] = close.shift(1)

    buy = (df["close_prev"] < df["ma_prev"]) & (close > df["ma"]) & (df["rsi"] < 70)
    sell = ((df["close_prev"] > df["ma_prev"]) & (close < df["ma"])) | (df["rsi"] > 75)

    df["signal"] = 0
    df.loc[buy, "signal"] = 1
    df.loc[sell, "signal"] = -1

    return df
