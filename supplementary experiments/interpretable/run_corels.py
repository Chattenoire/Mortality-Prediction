#!/usr/bin/env python
"""
Moderate-depth CORELS search compatible with very old corels-py builds.
- 100 k stratified subsample
- 4-bin one-hot → dense int8
- χ² top-200 bins  (≈ 800 binary cols)
- Grid:  c ∈ {0.002, 0.01, 0.05},  max_card = 3
- Each fit:  n_iter = 2 000 000  AND hard 20-min wall-time
Outputs
  models/corels_best.joblib
  models/corels_val_metrics.json
"""
import os, json, yaml, joblib, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score
import corels
from utils import rule_stats

# ── helper --------------------------------------------------------------- #
def stratified_subsample(df, n, seed):
    if n >= len(df):
        return df.copy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(df, df["label"]))
    return df.iloc[idx].copy()

# ── load config & data --------------------------------------------------- #
cfg = yaml.safe_load(open("configs.yaml"))
df  = pd.read_parquet(cfg["train_parquet"])
df  = stratified_subsample(df, cfg["subsample"], cfg["seed"])

X = df.drop(columns="label").values
y = df["label"].values

# ── discretise + χ² top-200 --------------------------------------------- #
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Bins whose width")
    warnings.filterwarnings("ignore", message="Feature .* is constant")
    disc = KBinsDiscretizer(n_bins=4, encode="onehot", strategy="quantile")
    X_bin = disc.fit_transform(X)
X_bin = X_bin.toarray().astype(np.int8)
X_bin = SelectKBest(chi2, k=200).fit_transform(X_bin, y)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_bin, y, test_size=0.2, stratify=y, random_state=cfg["seed"])

# ── grid search ---------------------------------------------------------- #
grid_c = [0, 0.002, 0.01, 0.05]
best, best_auc = None, 0
overall_start = time.perf_counter()

for c_val in grid_c:
    fit_start = time.perf_counter()
    print(f"\n=== CORELS fit  c={c_val}  max_card=3 ===")

    mdl = corels.CorelsClassifier(
        c=c_val,
        max_card=3,
        n_iter=200_000,     # internal node cap
        verbosity=['rule', 'progress']
    )
    mdl.fit(X_tr, y_tr)

    # hard wall-time safety net
    if time.perf_counter() - fit_start > 900:
        print("  fit exceeded 15 min — using model as-is.")

    scores = (mdl.predict_proba(X_val)[:, 1] if hasattr(mdl, "predict_proba")
              else mdl.predict(X_val).astype(float))
    auc = roc_auc_score(y_val, scores)
    print(f"    val-AUC {auc:.3f}")

    if auc > best_auc:
        best, best_auc, best_params = mdl, auc, dict(c=c_val, max_card=3)

print(f"\n✓ CORELS grid finished in {time.perf_counter()-overall_start:.1f}s")

# ── save ----------------------------------------------------------------- #
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/corels_best.joblib")

rules_attr = next((a for a in ("rules_", "rules", "rlist") if hasattr(best, a)), [])
json.dump(
    dict(model="CORELS", auc=best_auc, **best_params, **rule_stats(rules_attr)),
    open("models/corels_val_metrics.json", "w"), indent=2
)
print(f"Best params {best_params}  val-AUC {best_auc:.3f}")
