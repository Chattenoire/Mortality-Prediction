"""
Train ONE (k,R) model and write metrics.json + model/ directory.
"""
import os, json, random, argparse, yaml, numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils.metrics import brier, ece
from fedfnn.models import FedFNN

tf.debugging.enable_check_numerics()

# ---- deterministic ---------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

# ---- dataset loader --------------------------------------------------------
def load(path):
    import pickle, numpy as np
    with open(path, "rb") as f:
        d = pickle.load(f)
    X_ts      = np.nan_to_num(d["X_ts"],     nan=0., posinf=0., neginf=0.).astype("float32")
    X_static  = np.nan_to_num(d["X_static"], nan=0., posinf=0., neginf=0.).astype("float32")
    y = d["y"].astype("float32").reshape(-1, 1)
    assert set(np.unique(y)).issubset({0., 1.}), "Labels must be 0/1."

    return X_ts, X_static, y
# ---- CLI / main ------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--R", type=int, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config))
    cfg["run"] = {"k": args.k, "R": args.R}
    set_seed(cfg["seed"])

    Xts, Xst, y = load(cfg["dataset"]["path"])
    Xts_tr, Xts_te, Xst_tr, Xst_te, y_tr, y_te = train_test_split(
        Xts, Xst, y, test_size=0.2, stratify=y, random_state=cfg["seed"])

    model = FedFNN(Xts.shape[2], Xst.shape[1], cfg)
    opt = tf.keras.optimizers.get({"class_name": cfg["optimiser"]["name"],
                                   "config": {"learning_rate": cfg["optimiser"]["lr"]}})
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    cw = {0: cfg["training"]["neg_weight"], 1: cfg["training"]["pos_weight"]}
    cb = tf.keras.callbacks.EarlyStopping(
        monitor=cfg["training"]["early_stopping"]["monitor"],
        patience=cfg["training"]["early_stopping"]["patience"],
        min_delta=cfg["training"]["early_stopping"]["min_delta"],
        restore_best_weights=True
    )

    model.fit(
        [Xts_tr, Xst_tr], y_tr,
        validation_split=0.15,
        epochs=cfg["federated"]["rounds"],
        batch_size=cfg["training"]["batch_size"],
        class_weight=cw,
        callbacks=[cb],
        verbose=0
    )
    tf.debugging.disable_check_numerics()
    model.save_weights(os.path.join(args.out, "ckpt"))

    logit = model.predict([Xts_te, Xst_te], batch_size=256).squeeze()
    prob = tf.sigmoid(logit).numpy()
    metrics = {
        "auc":   float(roc_auc_score(y_te, prob)),
        "brier": float(brier(y_te, prob)),
        "ece":   float(ece(y_te, prob, cfg["evaluation"]["n_bins_ece"])),
        "k": args.k,
        "R": args.R
    }

    os.makedirs(args.out, exist_ok=True)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    ckpt_path = os.path.join(args.out, "ckpt")
    model.save_weights(ckpt_path)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
