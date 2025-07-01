# ---------------------------------------------------------------------
#  FedFNN-ERL – Compute lab-value slopes (trends)
#  Copyright (c) 2025  Po-Kang Tsai
#  Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------
#  Requires:  ENV MIMIC_IV_DATA  → root directory that contains /hosp
#             ENV FEDFNN_OUTDIR  → output directory for derived CSVs
# ---------------------------------------------------------------------

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress

# ---------- configurable paths ----------------------------------------------------

DATA_ROOT = Path(os.getenv("MIMIC_IV_DATA", "/path/to/mimic-iv")).expanduser()
OUT_DIR = Path(os.getenv("FEDFNN_OUTDIR", "./fedfnn_output")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEVENTS_FILE = DATA_ROOT / "hosp" / "labevents.csv.gz"
OUTPUT_FILE = OUT_DIR / "lab_trends_extended.csv"

# ---------- ITEMID sets -----------------------------------------------------------

BILIRUBIN_ITEMIDS = [...]
CREATININE_ITEMIDS = [...]
LACTATE_ITEMIDS = [...]
PLATELET_ITEMIDS = [...]

# ---------- helpers ---------------------------------------------------------------


def _slope(group: pd.DataFrame) -> float:
    """Return slope of valuenum over hours_since_intime, or 0.0 if not computable."""
    group = group.sort_values("hours_since_intime")
    hours = group["hours_since_intime"].values
    values = group["valuenum"].astype(float).values
    if np.unique(hours).size == 1:
        return 0.0
    try:
        return linregress(hours, values).slope
    except Exception:
        return 0.0


def _compute_lab_slope(df: pd.DataFrame,
                       itemids: list[int],
                       name: str) -> pd.DataFrame:
    tmp = df[df["itemid"].isin(itemids)].dropna(
        subset=["valuenum", "hours_since_intime"])
    slopes = (tmp.groupby("hadm_id")
                  .apply(_slope)
                  .reset_index(name=f"{name}_slope"))
    return slopes


# ---------- main ------------------------------------------------------------------

print("Loading labevents (this may take a while)…")
labs = pd.read_csv(LABEVENTS_FILE, compression="gzip",
                   usecols=["hadm_id", "itemid", "valuenum", "charttime"])
labs["charttime"] = pd.to_datetime(labs["charttime"])
labs["intime"] = labs.groupby("hadm_id")["charttime"].transform("min")
labs["hours_since_intime"] = (labs["charttime"] - labs["intime"]).dt.total_seconds() / 3600

print("Computing lab slopes…")
bil = _compute_lab_slope(labs, BILIRUBIN_ITEMIDS, "bilirubin")
cre = _compute_lab_slope(labs, CREATININE_ITEMIDS, "creatinine")
lac = _compute_lab_slope(labs, LACTATE_ITEMIDS, "lactate")
plt = _compute_lab_slope(labs, PLATELET_ITEMIDS, "platelets")

lab_trends = bil.merge(cre, how="outer").merge(lac, how="outer").merge(plt, how="outer")
lab_trends.fillna(0, inplace=True)

print(f"Computed slopes for {lab_trends.shape[0]} admissions.")
lab_trends.to_csv(OUTPUT_FILE, index=False)
print("Saved →", OUTPUT_FILE)
