"""
Train ONE aggregation algorithm on ONE rule configuration.
"""
import argparse, json, pathlib, yaml, pickle, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from aggregators import FedAvgAggregator, FedProxAggregator, ScaffoldAggregator
from federated_fedfnn_withERL import create_fuzzy_model, BEST_HPS, evolve_rules
from utils.metrics import brier, ece

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

# ------------- helpers ---------------------------------------------------- #
def load_dataset(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    Xts = np.nan_to_num(d["X_ts"], nan=0.).astype(np.float32)
    Xst = np.nan_to_num(d["X_static"], nan=0.).astype(np.float32)
    y   = d["y"].astype(np.float32)
    return Xts, Xst, y

def make_clients(Xts, Xst, y, batch, num_clients):
    size = len(y) // num_clients
    ds = []
    for i in range(num_clients):
        sl = slice(i * size, (i + 1) * size if i < num_clients-1 else len(y))
        ds.append(tf.data.Dataset
                    .from_tensor_slices(((Xts[sl], Xst[sl]), y[sl]))
                    .shuffle(512)
                    .batch(batch))
    return ds

# -------------------------------------------------------------------------- #
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", required=True)
    pa.add_argument("--out",    required=True)
    args = pa.parse_args()

    cfg = yaml.safe_load(open(args.config))
    np.random.seed(cfg["seed"]); tf.random.set_seed(cfg["seed"])

    # ---- inject rule hyper-params into BEST_HPS -------------------------- #
    BEST_HPS["num_rules"] = cfg["num_rules"]
    if "max_k" in cfg:
        BEST_HPS["max_k"] = cfg["max_k"]

    Xts, Xst, y = load_dataset(cfg["dataset"]["path"])
    Xts_tr, Xts_te, Xst_tr, Xst_te, y_tr, y_te = train_test_split(
        Xts, Xst, y, test_size=0.2, stratify=y, random_state=42)

    clients = make_clients(
        Xts_tr, Xst_tr, y_tr,
        batch=cfg["training"]["batch_size"],
        num_clients=cfg["num_clients"])

    model_fn = lambda: create_fuzzy_model(
        ts_input_dim=Xts_tr.shape[2],
        static_input_dim=Xst_tr.shape[1],
        hps=BEST_HPS)

    alg = cfg["algorithm"]
    if alg == "FedProx":
        agg = FedProxAggregator(model_fn,
                                client_lr=cfg["training"]["client_lr"],
                                mu=cfg["mu"])
    elif alg == "SCAFFOLD":
        agg = ScaffoldAggregator(model_fn,
                                 client_lr=cfg["training"]["client_lr"],
                                 server_lr=cfg["server_lr"])
    else:
        agg = FedAvgAggregator(model_fn,
                               client_lr=cfg["training"]["client_lr"])
    
    _ = agg.global_model((Xts_tr[:1], Xst_tr[:1]))
    
    # ---- re-initialise SCAFFOLD control variate after build ----
    if isinstance(agg, ScaffoldAggregator):
        agg.c = [tf.zeros_like(v) for v in agg.global_model.trainable_variables]

    # ---- training loop ---------------------------------------------------- #
    auc_by_round = []
    for rnd in range(cfg["federated"]["rounds"]):
        pkgs = [agg.client_update(ds) for ds in clients]
        agg.server_update(pkgs)
        logits = agg.global_model((Xts_te, Xst_te))
        prob   = tf.sigmoid(logits).numpy().squeeze()
        auc = float(tf.keras.metrics.AUC()(y_te, prob).numpy())
        auc_by_round.append([rnd+1, auc])
        print(f"[round {rnd+1:3d}] pooled AUROC = {auc:.3f}")

    # ---- client-level AUROC for Wilcoxon ---------------------------------- #
    client_auc = []
    for ds in clients:
        Xt, Xs, yt = [], [], []
        for (xt, xs), yb in ds: Xt.append(xt); Xs.append(xs); yt.append(yb)
        Xt = tf.concat(Xt, 0);  Xs = tf.concat(Xs, 0);  yt = tf.concat(yt, 0)
        prob_c = tf.sigmoid(agg.global_model((Xt, Xs))).numpy().squeeze()
        client_auc.append(float(tf.keras.metrics.AUC()(yt, prob_c).numpy()))

    metrics = dict(
        alg   = alg,
        rules = cfg["num_rules"],
        max_k = cfg.get("max_k", "NA"),
        auc   = auc,
        brier = brier(y_te, prob),
        ece   = ece(y_te, prob, n_bins=15)
    )

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out/"metrics.json", "w"), indent=2)
    np.savetxt(out/"auroc_by_round.csv", np.array(auc_by_round),
               delimiter=",", header="round,auc", comments="")
    np.savetxt(out/"auc_by_client.csv", np.array(client_auc),
               delimiter=",", header="auc", comments="")
    agg.global_model.save_weights(out/"ckpt")

if __name__ == "__main__":
    main()
