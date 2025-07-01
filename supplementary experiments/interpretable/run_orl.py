#!/usr/bin/env python
"""
Optimal Rule List baseline (works with imodels ≤ 1.x and corels ≤ 0.0.7).

Fixes the       ValueError: The truth value of an array with more than one
element is ambiguous …  by patching corels.CorelsClassifier.fit so it
casts any ndarray 'features' back to a plain list.
"""

import os, json, yaml, joblib, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import roc_auc_score
from imodels import OptimalRuleListClassifier
import numpy as np, corels   # ← needed for the patch
from utils import rule_stats

# warning suppression
import warnings
warnings.filterwarnings(
    "ignore",
    message="Bins whose width are too small",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Feature .* is constant",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── monkey-patch corels -----------------------------------------------------
_orig_fit = corels.CorelsClassifier.fit
def _safe_fit(self, X, y, *, features=None, prediction_name='outcome'):
    if isinstance(features, np.ndarray):            # turn array → list
        features = features.tolist()
    return _orig_fit(self, X, y,
                     features=features,
                     prediction_name=prediction_name)
corels.CorelsClassifier.fit = _safe_fit
# ---------------------------------------------------------------------------

def stratified_subsample(df, n, seed):
    if n >= len(df):
        return df.copy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(df, df["label"]))
    return df.iloc[idx].copy()

# ─── load config & data -----------------------------------------------------
cfg = yaml.safe_load(open("configs.yaml"))
df  = pd.read_parquet(cfg["train_parquet"])
df  = stratified_subsample(df, cfg["subsample"], cfg["seed"])

X_all = df.drop(columns="label")
y_all = df["label"].values
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=cfg["seed"])

# ─── grid search ------------------------------------------------------------
grid = ParameterGrid({"c": [0.002, 0.01], "max_card": [2, 3]})
best, best_auc = None, 0.0

for params in grid:
    mdl = OptimalRuleListClassifier(random_state=cfg["seed"], verbosity=["progress"], **params)
    Xtr, Xval = X_tr.values, X_val.values  # pandas → ndarray
    mdl.fit(Xtr, y_tr, feature_names=X_all.columns.tolist())  # list is now safe
    auc = roc_auc_score(y_val, mdl.predict_proba(Xval)[:, 1])
    print(f"  grid {params}  val-AUC {auc:.3f}")
    if auc > best_auc:
        best, best_auc, best_params = mdl, auc, params

# ─── save -------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/orl_best.joblib")
json.dump(
    dict(model="ORL", auc=best_auc, **best_params, **rule_stats(best.rule_list_)),
    open("models/orl_val_metrics.json", "w"), indent=2
)
print(f"✓ ORL best params {best_params}  val-AUC {best_auc:.3f}")
