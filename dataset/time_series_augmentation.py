# ---------------------------------------------------------------------
#  FedFNN-ERL – Time-series augmentation utilities
#  Copyright (c) 2024  Po-Kang Tsai & Wen-June Wang
#  Licensed under the Apache License, Version 2.0
#  --------------------------------------------------------------------
#  This file contains ONLY synthetic-data helpers and references no
#  protected health-information (PHI).
# ---------------------------------------------------------------------

from __future__ import annotations
import numpy as np

# ---------- random warping helpers ------------------------------------------------


def _random_warp_time_indices(num_steps: int = 48,
                              max_warp_factor: float = 0.20) -> np.ndarray:
    """Return an array of *num_steps* resampled time indices after warping."""
    orig_t = np.arange(num_steps, dtype=float)
    n_anchors = 4
    anchor_pos = np.linspace(0, num_steps - 1, n_anchors)
    anchor_shift = (np.random.rand(n_anchors) * 2 - 1) * (max_warp_factor * num_steps)
    anchor_y = np.clip(np.sort(anchor_pos + anchor_shift), 0, num_steps - 1)
    return np.interp(orig_t, anchor_pos, anchor_y)


def apply_random_warp(ts_sequence: np.ndarray,
                      max_warp_factor: float = 0.20) -> np.ndarray:
    """Warp a (T, F) time-series array and return a new array of same shape."""
    new_t = _random_warp_time_indices(ts_sequence.shape[0], max_warp_factor)
    warped = np.empty_like(ts_sequence)
    for f in range(ts_sequence.shape[1]):
        warped[:, f] = np.interp(new_t, np.arange(ts_sequence.shape[0]),
                                 ts_sequence[:, f])
    return warped


def augment_minority_randomwarp(X_ts: np.ndarray,
                                X_static: np.ndarray,
                                y: np.ndarray,
                                max_warp_factor: float = 0.20,
                                augment_ratio: float = 1.0) -> tuple[np.ndarray, ...]:
    """
    Random-warp each minority-class sample `augment_ratio` times.
    Returns augmented (X_ts, X_static, y).
    """
    pos_idx = np.where(y == 1)[0]
    n_synth = int(len(pos_idx) * augment_ratio)
    chosen = np.random.choice(pos_idx, n_synth, replace=True)

    X_ts_syn = np.stack([apply_random_warp(X_ts[i], max_warp_factor) for i in chosen])
    X_static_syn = X_static[chosen]
    y_syn = np.ones(n_synth, dtype=y.dtype)

    return (np.concatenate([X_ts, X_ts_syn]),
            np.concatenate([X_static, X_static_syn]),
            np.concatenate([y, y_syn]))
