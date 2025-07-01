# time_series_augmentation.py

import numpy as np
import random

def random_warp_time_indices(num_steps=48, max_warp_factor=0.2):
    """
    Generate a time index mapping for random warping.
    E.g. if num_steps=48, we create an array of length 48 that
    non-linearly stretches or compresses time by up to ±max_warp_factor * 100%.
    """
    # Original time indices: 0..47
    orig_t = np.arange(num_steps, dtype=float)

    # We'll create a random warping path by sampling random offsets
    # at a few anchor points, then do a piecewise linear interpolation.
    # E.g. pick 4 anchor points for simplicity.
    n_anchors = 4
    anchor_positions = np.linspace(0, num_steps - 1, n_anchors)
    # Each anchor can shift up or down up to max_warp_factor * num_steps
    anchor_shifts = (np.random.rand(n_anchors) * 2 - 1) * (max_warp_factor * num_steps)

    # Create anchor_y by adding anchor_shifts
    anchor_y = anchor_positions + anchor_shifts

    # Ensure anchor_y is sorted, but we can allow mild crossing. 
    # We'll just do a piecewise interpolation from anchor_positions to anchor_y.
    # Sort to avoid negative or reversed time
    anchor_y = np.clip(anchor_y, 0, num_steps - 1)
    anchor_y_sorted = np.sort(anchor_y)

    # Interpolate each integer time in [0..47] to get new time indices
    new_t = np.interp(orig_t, anchor_positions, anchor_y_sorted)
    return new_t

def apply_random_warp(ts_sequence, max_warp_factor=0.2):
    """
    Given a time-series ts_sequence of shape (48, n_features),
    produce a new sequence by random time warping.
    """
    num_steps = ts_sequence.shape[0]
    new_t = random_warp_time_indices(num_steps=num_steps, max_warp_factor=max_warp_factor)
    # We re-sample each feature using linear interpolation
    warped_seq = np.zeros_like(ts_sequence)
    for f in range(ts_sequence.shape[1]):
        # Original data
        y = ts_sequence[:, f]
        # Interpolate to new_t
        warped_seq[:, f] = np.interp(new_t, np.arange(num_steps), y)
    return warped_seq

def augment_minority_randomwarp(X_ts, X_static, y, max_warp_factor=0.2, augment_ratio=1.0):
    """
    For each minority (y=1) example, create 'augment_ratio' synthetic examples
    by random warping. 
    Return X_ts_aug, X_static_aug, y_aug with new synthetic minority samples appended.
    """
    # Separate minority vs majority
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    X_ts_pos = X_ts[pos_indices]
    X_static_pos = X_static[pos_indices]

    # Number of synthetic samples to create
    n_pos = len(pos_indices)
    n_synth = int(n_pos * augment_ratio)

    # Randomly choose minority examples to warp
    # If n_synth > n_pos, we can sample with replacement
    chosen_indices = np.random.choice(pos_indices, size=n_synth, replace=True)

    # Perform random warping
    X_ts_synth = []
    X_static_synth = []
    for idx in chosen_indices:
        seq = X_ts[idx]
        warped_seq = apply_random_warp(seq, max_warp_factor=max_warp_factor)
        X_ts_synth.append(warped_seq)
        X_static_synth.append(X_static[idx])

    X_ts_synth = np.array(X_ts_synth, dtype=np.float32)
    X_static_synth = np.array(X_static_synth, dtype=np.float32)
    y_synth = np.ones((n_synth,), dtype=np.float32)

    # Combine original data + synthetic
    X_ts_aug = np.concatenate([X_ts, X_ts_synth], axis=0)
    X_static_aug = np.concatenate([X_static, X_static_synth], axis=0)
    y_aug = np.concatenate([y, y_synth], axis=0)

    return X_ts_aug, X_static_aug, y_aug
