"""
real_shots.py — the shared real-shot loader and PFF ensemble prediction
helpers, extracted out of evaluate_pff_ensemble.py so optimizer scripts
(refine_bump_center_optimizer.py, bump_center_profile_likelihood.py) don't
have to import a comparisons script as a library to get at the 14-shot list
and the ensemble-loading/prediction machinery.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.core.data_utils import normalize_apply
from src.core.pff_model import decode_v2

# The 14 real shots used throughout this project (7 shot numbers x cv/ch
# saturation-correction variants).
SHOTS = []
for shot in ["10084", "11696", "11705", "11707", "11716", "11733", "11698"]:
    for suffix in ("_cv", "_ch"):
        SHOTS.append((f"{shot}{suffix}", f"res/test_images/{shot}/{shot}_proc_vector{suffix}.csv"))


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def load_members(model_dir: str = "out/training/pff", n_members: int = 5) -> list[dict]:
    """Load up to n_members PFF ensemble models + their norm stats/bounds from model_dir."""
    members = []
    for idx in range(n_members):
        model_path = os.path.join(model_dir, f"model_pff_ensemble_{idx}.keras")
        json_path = os.path.join(model_dir, f"pff_training_results_ensemble_{idx}.json")
        if not (os.path.exists(model_path) and os.path.exists(json_path)):
            print(f"  (member {idx} not found -- skipping; ensemble will use fewer members)")
            continue
        with open(json_path) as f:
            res = json.load(f)
        model = tf.keras.models.load_model(model_path, compile=False)
        members.append({
            "idx": idx,
            "model": model,
            "mean": np.array(res["norm_mean"], dtype=np.float32),
            "std": np.array(res["norm_std"], dtype=np.float32),
            "bounds": np.array(res["param_bounds"], dtype=np.float32),
        })
    return members


def predict_ensemble(members: list, X: np.ndarray) -> tuple:
    """X: (N, 200) L1-normalised signals. Returns stacked per-member (mu, sigma, p_bump), each (M, N, ...)."""
    mus, sigmas, p_bumps = [], [], []
    for mem in members:
        x_norm = normalize_apply(X, mem["mean"], mem["std"])
        out = mem["model"].predict(x_norm, verbose=0)
        mu, sigma, p_bump = decode_v2(out, mem["bounds"])
        mus.append(mu)
        sigmas.append(sigma)
        p_bumps.append(p_bump)
    return np.stack(mus), np.stack(sigmas), np.stack(p_bumps)
