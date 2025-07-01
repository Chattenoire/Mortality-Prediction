import numpy as np

def brier(y, p):
    return np.mean((y - p) ** 2)

def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    acc, conf, freq = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            acc.append(y[m].mean())
            conf.append(p[m].mean())
            freq.append(m.mean())
    acc, conf, freq = map(np.asarray, (acc, conf, freq))
    return np.sum(np.abs(acc - conf) * freq)
