import os
import glob
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

npz_files = glob.glob(r"PATH_TO_FILE\pred_*.npz")

out_dir = r"PATH"
os.makedirs(out_dir, exist_ok=True)

for fp in npz_files:
    name = os.path.basename(fp).replace("pred_","").replace(".npz","")
    data = np.load(fp)
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    # 2) Fit isotonic regression on the full set
    iso = IsotonicRegression(out_of_bounds='clip')
    y_cal = iso.fit_transform(y_prob, y_true)

    # 3) Compute old vs. new Brier
    old_brier = brier_score_loss(y_true, y_prob)
    new_brier = brier_score_loss(y_true, y_cal)

    print(f"{name:25s}  Brier before: {old_brier:.4f}   after: {new_brier:.4f}")

    # 4) Save calibrated probabilities
    out_fp = os.path.join(out_dir, f"pred_{name}_calibrated.npz")
    np.savez(out_fp, y_true=y_true, y_prob=y_cal)
    print(f"Saved calibrated preds to: {out_fp}\n")

print("All models calibrated.")
