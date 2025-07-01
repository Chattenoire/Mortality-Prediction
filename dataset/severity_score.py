# ---------------------------------------------------------------------
#  FedFNN-ERL – Compute SOFA- and Charlson-style severity scores
#  Copyright (c) 2024  Po-Kang Tsai & Wen-June Wang
#  Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------
#  ENV MIMIC_IV_DATA   → root of MIMIC-IV (contains /hosp, /icu, /core)
#  ENV FEDFNN_OUTDIR   → destination for derived CSVs
# ---------------------------------------------------------------------

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path(os.getenv("MIMIC_IV_DATA", "/path/to/mimic-iv")).expanduser()
OUT_DIR   = Path(os.getenv("FEDFNN_OUTDIR", "./fedfnn_output")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ICUSTAYS_FILE = DATA_ROOT / "icu" / "icustays.csv.gz"
CHARTEVENTS_FILE = DATA_ROOT / "icu" / "chartevents.csv.gz"
LABEVENTS_FILE   = DATA_ROOT / "hosp" / "labevents.csv.gz"
DIAG_FILE        = DATA_ROOT / "hosp" / "diagnoses_icd.csv.gz"

SOFA_OUT   = OUT_DIR / "sofa_scores.csv"
CHARL_OUT  = OUT_DIR / "charlson_index.csv"

# ---------------- helper utilities -----------------------------------

def _sofa_resp(paO2_FIO2: float, ventilation: bool) -> int:
    if np.isnan(paO2_FIO2):
        return 0
    if   paO2_FIO2 < 100 and ventilation: return 4
    elif paO2_FIO2 < 200 and ventilation: return 3
    elif paO2_FIO2 < 300:                 return 2
    elif paO2_FIO2 < 400:                 return 1
    return 0

def _sofa_platelets(plates: float) -> int:
    if   plates <  20: return 4
    elif plates <  50: return 3
    elif plates < 100: return 2
    elif plates < 150: return 1
    return 0

# additional SOFA subscores omitted for brevity …

# ---------------- SOFA computation -----------------------------------

print("Loading minimal ICU cohorts…")
icu = pd.read_csv(ICUSTAYS_FILE,
                  usecols=["stay_id", "hadm_id", "intime"]).astype({"stay_id": int})

# Example: load platelets during first 24 h
platelets = pd.read_csv(LABEVENTS_FILE,
                        usecols=["hadm_id", "itemid", "valuenum", "charttime"])
PLT_ITEMIDS = [51265]          # MIMIC-IV itemid for platelets
platelets   = platelets[platelets.itemid.isin(PLT_ITEMIDS)]
platelets["charttime"] = pd.to_datetime(platelets["charttime"])
platelets = (platelets
             .groupby("hadm_id")
             .valuenum
             .min()            # worst (lowest) value first 24 h
             .reset_index(name="min_platelets"))

# Merge and score
sofa = icu.merge(platelets, on="hadm_id", how="left")
sofa["resp_sub"]   = 0  # left 0 – add other subscores similarly
sofa["hema_sub"]   = sofa["min_platelets"].apply(_sofa_platelets)
sofa["SOFA_total"] = sofa[["resp_sub", "hema_sub"]].sum(axis=1)

sofa[["stay_id", "SOFA_total"]].to_csv(SOFA_OUT, index=False)
print("SOFA scores  →", SOFA_OUT)

# ---------------- Charlson computation --------------------------------

icd = pd.read_csv(DIAG_FILE, usecols=["hadm_id", "icd_code"])

# toy mapping, supply your full Charlson ICD list
CHARLSON_MAP = {
    "5859": 2,   # Chronic kidney disease, Stage 5
    "4280": 1    # CHF
}

icd["weight"] = icd.icd_code.map(lambda x: CHARLSON_MAP.get(str(x)[:4], 0))
charl = (icd.groupby("hadm_id")["weight"]
             .sum()
             .clip(upper=37)
             .reset_index(name="charlson_index"))
charl.to_csv(CHARL_OUT, index=False)
print("Charlson index →", CHARL_OUT)

# Print only aggregate info (no PHI)
print(f"Computed Charlson for {len(charl):,} admissions; "
      f"median score = {charl.charlson_index.median():.1f}")
