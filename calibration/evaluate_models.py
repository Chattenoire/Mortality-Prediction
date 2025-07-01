# -*- coding: utf-8 -*-
"""
evaluate_models.py

從 C:\Graduation Project\pred_*.npz 讀取 y_true, y_prob
計算並儲存：
  - ROC 比較圖 (roc_comparison.png)
  - Precision-Recall 比較圖 (pr_comparison.png)
  - Calibration 比較圖 (calibration_comparison.png)
  - Metrics 表格 (model_metrics.csv)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve  # 修正自 sklearn.calibration


def main():
    out_dir = r"C:\Graduation Project\saved_prediction_results\calibrated_results"
    os.makedirs(out_dir, exist_ok=True)

    # 掃描所有 prediction 檔案
    files = sorted(glob.glob(os.path.join(out_dir, "pred_*.npz")))
    if not files:
        print(f"No prediction files found in {out_dir}")
        return

    metrics = []
    results = {}

    for fp in files:
        model_name = os.path.basename(fp).replace("pred_", "").replace(".npz", "")
        data = np.load(fp)
        y_true = data["y_true"]
        y_prob = data["y_prob"]

        # ROC 曲線 & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        # Precision-Recall & AP
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        # Brier score
        brier = brier_score_loss(y_true, y_prob)
        # Calibration 曲線
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

        # 保存至 metrics list
        metrics.append({
            'model': model_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier': brier
        })
        # 保存詳列結果
        results[model_name] = {
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall': recall, 'pr_auc': pr_auc,
            'prob_true': prob_true, 'prob_pred': prob_pred
        }

    # 1. 儲存 Metrics 表格 (CSV)
    df = pd.DataFrame(metrics).sort_values('model')
    csv_path = os.path.join(out_dir, 'model_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved model metrics to {csv_path}")

    # 2. ROC 曲線比較圖
    plt.figure(figsize=(8,6))
    for name, r in results.items():
        plt.plot(r['fpr'], r['tpr'], label=f"{name} (AUC={r['roc_auc']:.3f})")
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cal_roc_comparison.png'))
    plt.close()

    # 3. Precision-Recall 比較圖
    plt.figure(figsize=(8,6))
    for name, r in results.items():
        plt.plot(r['recall'], r['precision'], label=f"{name} (AP={r['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cal_pr_comparison.png'))
    plt.close()

    # 4. Calibration 曲線比較圖
    plt.figure(figsize=(8,6))
    for name, r in results.items():
        plt.plot(r['prob_pred'], r['prob_true'], marker='o', label=name)
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve Comparison')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cal_calibration_comparison.png'))
    plt.close()

    # 5. 列印 Brier score
    print("\n=== cal Brier Score Comparison ===")
    for entry in metrics:
        print(f"{entry['model']:<25} : {entry['brier']:.4f}")


if __name__ == '__main__':
    main()
