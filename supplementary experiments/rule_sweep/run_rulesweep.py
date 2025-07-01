#!/usr/bin/env python
"""
Launches the 12-config sweep (detached subprocess per job).
"""
import itertools, subprocess, yaml, os, pathlib

EXP_ROOT = pathlib.Path("outputs/rule_sweep")
EXP_ROOT.mkdir(parents=True, exist_ok=True)

GRID_K = [2, 3, 4, 5]
GRID_R = [10, 25, 40]

def run():
    for k, R in itertools.product(GRID_K, GRID_R):
        tag = f"k{k}_R{R}"
        out = EXP_ROOT / tag
        cmd = ["python", "train_server.py",
               "--k", str(k), "--R", str(R),
               "--out", str(out)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run()
