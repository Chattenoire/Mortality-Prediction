import numpy as np
def brier(y, p): return float(np.mean((y - p) ** 2))
def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            acc  = y[m].mean()
            conf = p[m].mean()
            ece += np.abs(acc - conf) * m.mean()
    return float(ece)
