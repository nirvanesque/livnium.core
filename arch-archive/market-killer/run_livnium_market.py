import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from market_loader import load_market
from market_vectorizer import compute_states
from market_physics import align
from market_regime import classify_regime5, Regime5

# exponential moving average gives a smoother, more stable attractor
def ema(series, alpha=0.1):
    out = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return np.stack(out)

def main():
    market = load_market("market")
    symbol = "TSLA"
    df = market[symbol]
    states = compute_states(df)

    states_ema = ema(states, alpha=0.1)

    basins = []
    divs = []
    tens = []
    regs = []
    aligns = []

    for i in range(20, len(states)):
        b = states_ema[i - 1]  # smoothed basin up to t-1
        h = states[i]

        # normalize vectors before alignment/divergence to use pure direction
        b = b / (np.linalg.norm(b) + 1e-8)
        h = h / (np.linalg.norm(h) + 1e-8)

        # core geometry
        a = align(h, b)                  # alignment
        d = 0.38 - a                     # divergence
        t = abs(d)                       # tension

        basins.append(b)
        divs.append(d)
        tens.append(t)
        aligns.append(a)

        # 5-basin regime classification
        regs.append(classify_regime5(a, t))

    # regime counts
    counts = Counter(regs)
    print(f"5-basin regime counts for {symbol}:")
    for r in Regime5:
        print(f"  {r.value:9s}: {counts.get(r, 0)}")

    plt.figure(figsize=(10, 6))
    plt.plot(divs, label="Divergence")
    plt.plot(tens, label="Tension")
    plt.legend()
    plt.title(f"{symbol} — Livnium divergence / tension")
    plt.tight_layout()
    plt.show()

    # discrete regime trace for 5 basins
    reg_to_int = {
        Regime5.PANIC: -2,
        Regime5.BEAR: -1,
        Regime5.NEUTRAL: 0,
        Regime5.BULL: 1,
        Regime5.EUPHORIA: 2,
    }
    reg_series = [reg_to_int[r] for r in regs]
    plt.figure(figsize=(10, 3))
    plt.plot(reg_series)
    plt.yticks([-2, -1, 0, 1, 2], ["Panic", "Bear", "Neutral", "Bull", "Euphoria"])
    plt.title(f"{symbol} — Livnium 5-basin regime over time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
