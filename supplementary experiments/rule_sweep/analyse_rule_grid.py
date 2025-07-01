"""
Summarises the rule-length × rule-count sweep, runs two-way ANOVA,
and emits:  (1) LaTeX-ready result tables   (2) a heat-map figure.
"""
import json, glob, pathlib, pandas as pd, numpy as np, statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

ROOT = pathlib.Path("outputs/rule_sweep")
OUT  = pathlib.Path("outputs/analysis")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1) Load every metrics.json into a tidy DataFrame
# --------------------------------------------------------------------------- #
records = [json.load(open(p)) for p in glob.glob(str(ROOT / "*/*metrics.json"))]
df = pd.DataFrame(records).sort_values(["k", "R"])
print("\nRaw summary\n------------\n", df.head())

# --------------------------------------------------------------------------- #
# 2) Save long-form CSV for the supplementary material
# --------------------------------------------------------------------------- #
df.to_csv(OUT / "rule_grid_raw.csv", index=False)

# --------------------------------------------------------------------------- #
# 3) Pivot to a k × R table for each metric
# --------------------------------------------------------------------------- #
for metric in ["auc", "brier", "ece"]:
    pivot = df.pivot(index="k", columns="R", values=metric)
    pivot.to_csv(OUT / f"{metric}_table.csv")
    pivot.to_latex(OUT / f"{metric}_table.tex",
                   float_format="%.4f", bold_rows=True,
                   caption=f"{metric.upper()} for every (k,R) pair.",
                   label=f"tab:{metric}_rule_grid")

# --------------------------------------------------------------------------- #
# 4) Figure 3 – AUROC heat-map
# --------------------------------------------------------------------------- #
auroc = df.pivot(index="k", columns="R", values="auc")
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(auroc.values, cmap="viridis", origin="lower")

# Axis labels / ticks
ax.set_xticks(range(len(auroc.columns)))
ax.set_xticklabels(auroc.columns)
ax.set_yticks(range(len(auroc.index)))
ax.set_yticklabels(auroc.index)
ax.set_xlabel("Rule count  R")
ax.set_ylabel("Max antecedent length  k")
ax.set_title("AUROC across rule-base sizes")

# Annotate cells
for i, k in enumerate(auroc.index):
    for j, R in enumerate(auroc.columns):
        val = auroc.loc[k, R]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.8 else "black")

fig.colorbar(im, ax=ax, label="AUROC")
fig.tight_layout()
fig.savefig(OUT / "figure3_rule_grid_auroc.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 5) Two-way ANOVA (type II) with η² effect sizes
# --------------------------------------------------------------------------- #
model = smf.ols("auc ~ C(k) + C(R)", data=df).fit()
anova  = sm.stats.anova_lm(model, typ=2)
anova["eta2"] = anova["sum_sq"] / (anova["sum_sq"].sum() + model.ssr)
print("\nANOVA\n-----\n", anova)

anova.to_csv(OUT / "anova_auc.csv")
anova.to_latex(OUT / "anova_auc.tex",
               float_format="%.4f",
               caption="Two-way mixed-effect ANOVA on AUROC.",
               label="tab:anova_auc")

print(f"\nAll artefacts written to   {OUT.resolve()}")
