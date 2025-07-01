"""
Evaluate CORELS & FRL best models on the held-out test set and compare to FedFNN.
Produces:
  • rule_models_test.json   (metrics & rule stats)
  • comparison_stats.json   (DeLong p, bootstrap CI)
"""
import pandas as pd, numpy as np, joblib, json, yaml, os
from sklearn.metrics import roc_auc_score
from utils import brier, ece, rule_stats, paired_bootstrap, delong_pvalue

cfg = yaml.safe_load(open("configs.yaml"))
test = pd.read_parquet(cfg["test_parquet"])
X_te, y_te = test.drop(columns="label").values, test["label"].values

models = {
    "CORELS": joblib.load("models/corels_best.joblib"),
    "DT":     joblib.load("models/dtree_best.joblib")
}
results = {}
for name, mdl in models.items():
    if hasattr(mdl, "predict_proba"):
        prob = mdl.predict_proba(X_te)[:,1]
    else:
        prob = mdl.predict(X_te).astype(float)
    res  = dict(
        model=name,
        auc=float(roc_auc_score(y_te, prob)),
        brier=brier(y_te, prob),
        ece=ece(y_te, prob, n_bins=15),
        **rule_stats(mdl.rules_)
    )
    results[name] = res

json.dump(results, open("models/rule_models_test.json", "w"), indent=2)
print("Saved test metrics for CORELS & FRL")

# ---- comparison with FedFNN best ---------------------------------------- #
fedf = json.load(open("../outputs/FedFNN_best/metrics.json"))
p_fedf = np.loadtxt("../outputs/FedFNN_best/prob_test.csv") 
y_true = y_te

comp = {}
for name, mdl in models.items():
    p_rule = mdl.predict_proba(X_te)[:,1]
    _, _, p_delong = delong_pvalue(y_true, p_fedf, p_rule)
    b_diff, (b_lo, b_hi) = paired_bootstrap(y_true, p_fedf, p_rule, brier)
    e_diff, (e_lo, e_hi) = paired_bootstrap(y_true, p_fedf, p_rule, ece)
    comp[name] = dict(
        delong_p=p_delong,
        delta_brier=b_diff,  ci_brier=[b_lo, b_hi],
        delta_ece=e_diff,    ci_ece=[e_lo, e_hi]
    )

json.dump(comp, open("models/comparison_stats.json", "w"), indent=2)
print("Saved DeLong + bootstrap comparison with FedFNN")
