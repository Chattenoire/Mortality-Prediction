import json, glob, re, pathlib, pandas as pd

ROOT = pathlib.Path("../outputs")
records = []

for j in glob.glob(str(ROOT/"FedProx_mu*"/"metrics.json")):
    mu   = float(re.search(r"mu([0-9.]+)", j).group(1))
    auc  = json.load(open(j))["auc"]
    records.append(("FedProx", mu, auc))

for j in glob.glob(str(ROOT/"SCAFFOLD_lr*"/"metrics.json")):
    lr   = float(re.search(r"lr([0-9.]+)", j).group(1))
    auc  = json.load(open(j))["auc"]
    records.append(("SCAFFOLD", lr, auc))

df = pd.DataFrame(records, columns=["alg", "param", "auc"])
best = df.loc[df.groupby("alg")["auc"].idxmax()]
print(best)
best.to_csv(ROOT/"best_hparams.csv", index=False)
