import numpy as np, pandas as pd, json, itertools, random
from sklearn.calibration import calibration_curve

# ---------- basic metrics ------------------------------------------------- #
def brier(y, p): return float(np.mean((y - p) ** 2))

def ece(y, p, n_bins=15):
    bin_true, bin_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    bin_counts = np.histogram(p, bins=n_bins, range=(0,1))[0]
    return float(np.sum(np.abs(bin_true - bin_pred) * bin_counts / len(y)))

# ---------- interpretability metrics -------------------------------------- #
def rule_stats(rule_list):
    lengths = [len(r.antecedent) for r in rule_list]
    cov = sum(r.support for r in rule_list) / len(rule_list)  # assumes .support filled
    return dict(n_rules=len(rule_list),
                mean_len=float(np.mean(lengths) if lengths else 0),
                coverage=float(cov))

# ---------- paired bootstrap for any metric ------------------------------- #
def paired_bootstrap(y_true, p1, p2, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        diffs.append(metric_fn(y_true[idx], p1[idx]) -
                     metric_fn(y_true[idx], p2[idx]))
    return np.median(diffs), np.percentile(diffs, [2.5, 97.5])

# ---------- DeLong two-sample AUROC p-value ------------------------------- #
from sklearn.metrics import roc_auc_score
def delong_pvalue(y, p1, p2):
    from scipy import stats
    import numpy as np
    def _variance_of_predictions(y_true, preds):
        order = np.argsort(-preds)
        y_true = y_true[order]
        distinct_value_indices = np.where(np.diff(preds[order]))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        areas = np.diff(np.r_[0, fpr])
        V10 = tpr[:-1] * areas
        V01 = (1 - tpr[:-1]) * areas
        s10 = np.sum(V10)
        s01 = np.sum(V01)
        auc = roc_auc_score(y_true, preds)
        return auc, s10, s01
    auc1, s10_1, s01_1 = _variance_of_predictions(y, p1)
    auc2, s10_2, s01_2 = _variance_of_predictions(y, p2)
    var = (s10_1 + s01_1 + s10_2 + s01_2)
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return auc1, auc2, float(p)
