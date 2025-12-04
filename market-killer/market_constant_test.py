"""
Experiment script to sweep the divergence constant C and compare
geometry-driven metrics across symbols.

Metrics per (symbol, C):
- corr(tension_t, |next_return|) to test predictive power
- tension distribution summary (mean/std/90th/95th percentiles)
- 5-basin regime counts (bull/bear/neutral/panic/euphoria)

Usage:
  python market_constant_test.py            # default symbols/C values
  python market_constant_test.py AAPL MSFT  # custom symbols
"""

import sys
import numpy as np
from collections import Counter, defaultdict

from market_loader import load_market
from market_regime import classify_regime5, Regime5


def build_state_frame(df, window=14, standardize=True):
    """Replicates compute_states but also returns aligned returns."""
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["vol"] = df["return"].rolling(window).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(window).mean()) / df["Volume"].rolling(window).std()
    df["range"] = (df["High"] - df["Low"]) / df["Open"]
    df["ratio"] = (df["Close"] / df["Open"]) - 1

    df_feat = df.dropna().copy()
    cols = ["return", "vol", "volume_z", "range", "ratio"]
    X = df_feat[cols].values

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        df_feat[cols] = X

    # align returns for correlation with tension
    close_ret = df["Close"].pct_change()
    df_feat["next_abs_return"] = close_ret.reindex(df_feat.index).shift(-1).abs()

    return df_feat, X


def ema(series, alpha=0.1):
    out = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return np.stack(out)


def evaluate_symbol(df, C=0.38, warmup=20):
    df_feat, X = build_state_frame(df)
    if len(df_feat) <= warmup + 1:
        return None

    states = X
    states_ema = ema(states, alpha=0.1)

    tensions = []
    aligns = []
    next_abs = []
    regs = []

    for i in range(warmup, len(states)):
        b = states_ema[i - 1]
        h = states[i]

        b = b / (np.linalg.norm(b) + 1e-8)
        h = h / (np.linalg.norm(h) + 1e-8)

        a = np.dot(h, b)
        d = C - a
        t = abs(d)

        tensions.append(t)
        aligns.append(a)

        nxt = df_feat["next_abs_return"].iloc[i]
        if not np.isnan(nxt):
            next_abs.append(nxt)
        else:
            next_abs.append(np.nan)

        regs.append(classify_regime5(a, t))

    # correlation tension vs |next return|
    corr = np.nan
    if np.isfinite(next_abs).sum() > 2:
        corr = np.corrcoef(tensions, next_abs)[0, 1]

    tensions_arr = np.array(tensions)
    summary = {
        "mean": float(np.mean(tensions_arr)),
        "std": float(np.std(tensions_arr)),
        "p90": float(np.percentile(tensions_arr, 90)),
        "p95": float(np.percentile(tensions_arr, 95)),
    }

    counts = Counter(regs)

    return {
        "corr": corr,
        "summary": summary,
        "counts": counts,
        "n_points": len(tensions),
    }


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "TSLA", "MSFT", "SPY"]
    C_values = [0.30, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45]

    market = load_market("market")

    for C in C_values:
        print(f"\n=== Testing C = {C:.2f} ===")
        agg_corr = []
        agg_counts = defaultdict(int)
        for sym in symbols:
            if sym not in market:
                print(f"  {sym}: missing data")
                continue
            res = evaluate_symbol(market[sym], C=C)
            if res is None:
                print(f"  {sym}: insufficient data")
                continue

            agg_corr.append(res["corr"])
            for k, v in res["counts"].items():
                agg_counts[k] += v

            s = res["summary"]
            print(f"  {sym}: corr(tension,|next_ret|)={res['corr']:.4f}, "
                  f"tension mean={s['mean']:.3f}, std={s['std']:.3f}, "
                  f"p90={s['p90']:.3f}, p95={s['p95']:.3f}")

        if agg_corr:
            mean_corr = float(np.nanmean(agg_corr))
            print(f"  avg corr across symbols: {mean_corr:.4f}")
        if agg_counts:
            print("  5-basin counts (aggregated):")
            for r in Regime5:
                print(f"    {r.value:9s}: {agg_counts.get(r, 0)}")


if __name__ == "__main__":
    main()
