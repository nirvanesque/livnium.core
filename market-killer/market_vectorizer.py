import numpy as np

def compute_states(df, window=14, standardize=True):
    df = df.copy()

    df["return"] = df["Close"].pct_change()
    df["vol"] = df["return"].rolling(window).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(window).mean()) / df["Volume"].rolling(window).std()
    df["range"] = (df["High"] - df["Low"]) / df["Open"]
    df["ratio"] = (df["Close"] / df["Open"]) - 1

    df = df.dropna()

    X = df[["return", "vol", "volume_z", "range", "ratio"]].values

    if standardize:
        # column-wise z-score to put features on comparable scale
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std

    return X
