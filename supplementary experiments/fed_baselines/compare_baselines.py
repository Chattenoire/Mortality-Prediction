"""
Compare FedAvg vs FedProx vs SCAFFOLD on both rule settings.
"""

import pathlib, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = pathlib.Path("../outputs")
pairs = [("15_FedAvg", "15_FedProx"),
         ("15_FedAvg", "15_SCAFFOLD"),
         ("sw_FedAvg", "sw_FedProx"),
         ("sw_FedAvg", "sw_SCAFFOLD")]

txt = []
for base, comp in pairs:
    base_auc = np.loadtxt(ROOT/base/"auc_by_client.csv", skiprows=1, delimiter=",")
    comp_auc = np.loadtxt(ROOT/comp/"auc_by_client.csv", skiprows=1, delimiter=",")
    stat, p  = wilcoxon(base_auc, comp_auc)
    delta    = np.median(comp_auc) - np.median(base_auc)
    txt.append(f"{comp.split('_')[1]} vs {base}: Δmedian={delta:+.3f}, p={p:.4f}")

with open(ROOT/"wilcoxon_results.txt", "w") as f:
    f.write("\n".join(txt))
print("\n".join(txt))

# ---- communication efficiency plot ---- #
plt.figure(figsize=(6,4))
labels = {"15_FedAvg":"15-Rule FedAvg", "15_FedProx":"15-Rule FedProx",
          "15_SCAFFOLD":"15-Rule SCAFFOLD",
          "sw_FedAvg":"40-Rule FedAvg", "sw_FedProx":"40-Rule FedProx",
          "sw_SCAFFOLD":"40-Rule SCAFFOLD"}
for run in labels:
    data = np.loadtxt(ROOT/run/"auroc_by_round.csv", skiprows=1, delimiter=",")
    plt.plot(data[:,0], data[:,1], label=labels[run])
plt.axhline(0.75, ls="--", c="gray")
plt.xlabel("Communication round"); plt.ylabel("AUROC")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(ROOT/"figure4_comm_efficiency.png", dpi=300)
