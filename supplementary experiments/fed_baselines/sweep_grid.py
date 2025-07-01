#!/usr/bin/env python
"""
Lightweight grid launcher for FedProx and SCAFFOLD.
Usage examples
  python sweep_grid.py configs/fedprox_grid.yaml
  python sweep_grid.py configs/scaffold_grid.yaml
"""
import itertools, subprocess, yaml, pathlib, sys, tempfile, copy

def merge_dicts(parent, child):
    """Shallow merge: child keys override parent."""
    out = copy.deepcopy(parent)
    out.update({k: v for k, v in child.items() if k != "defaults"})
    return out

def expand_grid(grid_cfg_path):
    child = yaml.safe_load(open(grid_cfg_path))
    # ---- optional parent merge ----------------------------------------- #
    if "defaults" in child and child["defaults"]:
        parent_path = pathlib.Path(grid_cfg_path).with_name(child["defaults"][0])
        parent = yaml.safe_load(open(parent_path))
        base = merge_dicts(parent, child)
    else:
        base = child
    # ---- Cartesian product over grid ----------------------------------- #
    grid = base.pop("grid")
    keys, vals = zip(*grid.items())
    for combo in itertools.product(*vals):
        cfg = copy.deepcopy(base)
        tag = []
        for k, v in zip(keys, combo):
            cfg[k] = v
            tag.append(f"{k}{v}")
        
        if cfg["algorithm"] == "SCAFFOLD":
            cfg["client_lr"] = cfg.pop("lr")     # remove 'lr'…
            cfg["server_lr"] = cfg["client_lr"]  # …and copy to both fields
        
        yield cfg, "_".join(tag)

def run(grid_cfg):
    for cfg, tag in expand_grid(grid_cfg):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp.flush()
            out_dir = pathlib.Path("../outputs") / f"{cfg['algorithm']}_{tag}"
            cmd = [sys.executable,
                   "run_federated.py",
                   "--config", tmp.name,
                   "--out",   str(out_dir)]        # ← cast Path to str
            print("🚀", " ".join(cmd))
            subprocess.run(cmd, check=True)
            try:
                pathlib.Path(tmp.name).unlink()
            except PermissionError:
                pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python sweep_grid.py <grid_yaml>")
        sys.exit(1)
    run(sys.argv[1])
