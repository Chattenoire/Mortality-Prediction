#!/usr/bin/env python
"""
Convert preprocessed_data_enriched.pkl → tabular_train.parquet / tabular_test.parquet
"""
import pickle, numpy as np, pandas as pd, os, argparse
from sklearn.model_selection import train_test_split

def flatten_ts(X_ts):
    """
    X_ts:  N × 48 × F  ->  pandas DataFrame with
            • mean_*
            • last_*
            • slope_*   (simple (x_t - x_0) / 48  as trend proxy)
    """
    mean  = X_ts.mean(axis=1)
    last  = X_ts[:, -1, :]
    slope = (X_ts[:, -1, :] - X_ts[:, 0, :]) / 48.0
    df = pd.DataFrame(
        np.hstack([mean, last, slope]),
        columns=[f"{stat}_ch{c}"
                 for stat in ("mean", "last", "slope")
                 for c in range(X_ts.shape[2])]
    )
    return df

def main(pkl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X_ts = np.nan_to_num(data["X_ts"], nan=0.0)
    X_sta = pd.DataFrame(data["X_static"], columns=[f"static_{i}" for i in range(data["X_static"].shape[1])])
    y = pd.Series(data["y"], name="label")

    df_ts  = flatten_ts(X_ts)
    df_all = pd.concat([df_ts, X_sta, y], axis=1)

    train, test = train_test_split(df_all, test_size=0.2, stratify=df_all["label"],
                                   random_state=42)

    train.to_parquet(os.path.join(out_dir, "tabular_train.parquet"))
    test.to_parquet(os.path.join(out_dir, "tabular_test.parquet"))
    print("✓ Wrote",
          os.path.join(out_dir, "tabular_train.parquet"),
          "and",
          os.path.join(out_dir, "tabular_test.parquet"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl",  default=r"PATH_TO_FILE\preprocessed_data_enriched.pkl")
    ap.add_argument("--out",  default=r"PATH")
    args = ap.parse_args()
    main(args.pkl, args.out)
