import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ——————————————————————————————————————————————————————————————
# File path
# ——————————————————————————————————————————————————————————————
base_dir = "PATH"
pre_path  = os.path.join(r"PATH_TO_FILE\raw_results", "pred_FedFNN_noERL.npz")
post_path = os.path.join(r"PATH_TO_FILE\calibrated_results", "pred_FedFNN_noERL_calibrated.npz")

# ——————————————————————————————————————————————————————————————
# Load data
# ——————————————————————————————————————————————————————————————
def load_preds(fp):
    data = np.load(fp)
    return data["y_true"], data["y_prob"]

y_true_pre,  y_prob_pre  = load_preds(pre_path)
y_true_post, y_prob_post = load_preds(post_path)

# ——————————————————————————————————————————————————————————————
# Metrics
# ——————————————————————————————————————————————————————————————
def compute_curves(y_true, y_prob, n_bins=10):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    # Brier
    brier = brier_score_loss(y_true, y_prob)
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision": precision, "recall": recall, "pr_auc": pr_auc,
        "prob_true": prob_true, "prob_pred": prob_pred,
        "brier": brier
    }

pre = compute_curves(y_true_pre,  y_prob_pre)
post = compute_curves(y_true_post, y_prob_post)

# ——————————————————————————————————————————————————————————————
# Before vs. After
# ——————————————————————————————————————————————————————————————
for tag, stats in [("Figure 1. Pre‑calibration", pre), ("Figure 2. Post‑calibration", post)]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{tag} — FedFNN no ERL", fontsize=16)

    # (a) Calibration curve
    ax = axes[0]
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.plot(stats["prob_pred"], stats["prob_true"], marker='o', label=f"Brier={stats['brier']:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("(a) Reliability diagram")
    ax.legend(loc="lower right")

    # (b) Precision–Recall curve
    ax = axes[1]
    ax.plot(stats["recall"], stats["precision"], lw=2, label=f"AP={stats['pr_auc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision–Recall curve")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="lower left")

    # (c) ROC curve
    ax = axes[2]
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.plot(stats["fpr"], stats["tpr"], lw=2, label=f"AUC={stats['roc_auc']:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(c) ROC curve")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(base_dir, f"{tag.split()[1]}_FedFNN_noERL.png")
    plt.savefig(out_png, dpi=300)
    print(f"Saved {tag} to {out_png}")
    plt.close()
