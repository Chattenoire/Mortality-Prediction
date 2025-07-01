#!/usr/bin/env python
"""
Produce a concise textual summary of the rule-length × rule-count sweep.
Reads the CSV/LaTeX files created by `analyse_rule_grid.py`.

Outputs:
  • best configuration (k,R) w/ AUROC, Brier, ECE
  • marginal means for each k and each R
  • AUROC Δ gains (k=4 vs k=2, R=25 vs R=10, etc.)
  • ANOVA sentence (F, p, η²) ready to paste into paper
"""
import pandas as pd
import pathlib
from textwrap import indent

ROOT = pathlib.Path("outputs/analysis")

# ------------------------------------------------------------------ #
# 1) Load raw grid and ANOVA table
# ------------------------------------------------------------------ #
df     = pd.read_csv(ROOT / "rule_grid_raw.csv")
anova  = pd.read_csv(ROOT / "anova_auc.csv", index_col=0)

# ------------------------------------------------------------------ #
# 2) Best configuration
# ------------------------------------------------------------------ #
best = df.loc[df["auc"].idxmax()]
best_line = (
    f"Best AUROC = {best.auc:.3f} at k={int(best.k)}, R={int(best.R)} "
    f"(Brier={best.brier:.4f}, ECE={best.ece:.4f})"
)

# ------------------------------------------------------------------ #
# 3) Marginal means
# ------------------------------------------------------------------ #
mean_k = df.groupby("k")["auc"].mean()
mean_R = df.groupby("R")["auc"].mean()

# Δ gains relative to smallest level
delta_k = mean_k - mean_k.min()
delta_R = mean_R - mean_R.min()

# ------------------------------------------------------------------ #
# 4) ANOVA sentence (Option A model w/out interaction)
# ------------------------------------------------------------------ #
row_k = anova.loc["C(k)"]
row_R = anova.loc["C(R)"]

anova_sentence = (
    "Two-way ANOVA (no interaction) showed a significant main effect of k "
    f"(F₍3,{int(row_k['df'])}₎ = {row_k['F']:.2f}, p = {row_k['PR(>F)']:.4f}, "
    f"η² = {row_k['eta2']:.3f}) and of R "
    f"(F₍2,{int(row_R['df'])}₎ = {row_R['F']:.2f}, p = {row_R['PR(>F)']:.4f}, "
    f"η² = {row_R['eta2']:.3f})."
)

# ------------------------------------------------------------------ #
# 5) Pretty print
# ------------------------------------------------------------------ #
print("=== Rule-complexity sweep summary ===\n")
print(best_line, "\n")

print("Mean AUROC by antecedent length k:")
print(indent(mean_k.round(4).to_string(), "  "))
print("\nΔ gain vs k=min:")
print(indent(delta_k.round(4).to_string(), "  "), "\n")

print("Mean AUROC by rule count R:")
print(indent(mean_R.round(4).to_string(), "  "))
print("\nΔ gain vs R=min:")
print(indent(delta_R.round(4).to_string(), "  "), "\n")

print(anova_sentence)
print("\nLaTeX-ready heat-map figure   →  outputs/analysis/figure3_rule_grid_auroc.png")
print("LaTeX tables                  →  auc_table.tex / brier_table.tex / ece_table.tex")
