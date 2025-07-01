#!/usr/bin/env python
"""
Depth-3 Decision Tree baseline (interpretable).

Outputs
  models/dtree_best.joblib
  models/dtree_val_metrics.json
"""
import os, json, yaml, joblib, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from utils import rule_stats        # reuse leaf count as n_rules

def stratified_subsample(df, n, seed):
    if n >= len(df): return df.copy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(df, df["label"]))
    return df.iloc[idx].copy()

cfg = yaml.safe_load(open("configs.yaml"))
df  = pd.read_parquet(cfg["train_parquet"])
df  = stratified_subsample(df, cfg["subsample"], cfg["seed"])

X, y = df.drop(columns="label").values, df["label"].values
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=cfg["seed"])

mdl = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                             random_state=cfg["seed"])
mdl.fit(X_tr, y_tr)
auc = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])

os.makedirs("models", exist_ok=True)
joblib.dump(mdl, "models/dtree_best.joblib")
json.dump(
    dict(
        model="DT_depth3",
        auc=float(auc),                         # cast to plain float
        n_rules=int(mdl.tree_.n_leaves),        # ← cast to plain int
        mean_len=3.0
    ),
    open("models/dtree_val_metrics.json", "w"),
    indent=2
)

print(f"✓ Decision Tree depth=3  val-AUC {auc:.3f}")
