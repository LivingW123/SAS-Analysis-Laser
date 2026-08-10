"""
compare_pff_v1_v2.py — side-by-side comparison of the PFF-parameter
regressors built this session:

  v1 : model_pff_relu_uncertainty.keras       (train_pff.py)
       ReLU mean + linear-logvar-with-hard-clip, single Gaussian NLL over
       all 5 params, a4/a5 continuously weighted by normalized a3.
  v2 : model_pff_v2.keras                     (train_pff_bounded_gated.py)
       ReLU mean + tanh-bounded logvar, a3 split into an explicit bump
       classifier + magnitude-given-bump, mixed via Bernoulli-Gaussian at
       decode time.
  v3 : model_pff_v3.keras                     (train_pff_v3_realistic_noise.py)
       Same architecture as v2, but trained on generate_pff_training_data's
       now-realistic (calibrated noise + CCD saturation) generator instead
       of v1/v2's plain Poisson-only one -- isolates whether closing that
       specific domain gap changes real-shot behavior.

Runs all available versions on the same 14 real shot files (the same set
infer_cnn.py's n=50 runs used) to compare how badly each model's sigma still
blows up (or collapses to a constant) on out-of-distribution real data, plus
the synthetic calibration check (known ground truth) each training script
already does independently. Versions whose model/results files aren't found
on disk are skipped, so this also works as a v1-vs-v2-only or v2-vs-v3-only
comparison depending on what's been trained.

Usage
-----
  python compare_pff_v1_v2.py
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import (
    generate_pff_training_data,
    load_drm,
    normalize_apply,
    pff_mean_sigma,
)
from train_pff_bounded_gated import decode_v2

PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5"]

SHOTS = []
for shot in ["10084", "11696", "11705", "11707", "11716", "11733", "11698"]:
    for suffix in ("_cv", "_ch"):
        SHOTS.append((f"{shot}{suffix}", f"res/test_images/{shot}/{shot}_proc_vector{suffix}.csv"))

DRM_PATH = "res/drm/200x200.xlsx"
SYNTH_SAMPLES = 200
SYNTH_SEED = 123

# (label, model_path, results_json, is_gated) -- is_gated selects decode_v2
# (11-wide, has p_bump) vs pff_mean_sigma (10-wide, no p_bump).
VERSIONS = [
    ("v1", "model_pff_relu_uncertainty.keras", "pff_training_results_relu_uncertainty.json", False),
    ("v2", "model_pff_v2.keras", "pff_training_results_v2.json", True),
    # v3 attempts: 20k/patience=40 -> 60k/patience=80 -> 200k/patience=80,
    # each preserved under its own name to show the delta. v3 itself always
    # points at the latest attempt.
    ("v3_a1_20k", "model_pff_v3_attempt1_20k.keras", "pff_training_results_v3_attempt1_20k.json", True),
    ("v3_a2_60k", "model_pff_v3_attempt2_60k.keras", "pff_training_results_v3_attempt2_60k.json", True),
    ("v3", "model_pff_v3.keras", "pff_training_results_v3.json", True),
]


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def decode(pff_out: np.ndarray, bounds: np.ndarray, is_gated: bool):
    """pff_out is a batch, shape (N, width). Returns (mu, sigma, p_bump_or_None), each (N,5)/(N,1)."""
    if is_gated:
        mu, sigma, p_bump = decode_v2(pff_out, bounds)
        return mu, sigma, p_bump
    mu, sigma = pff_mean_sigma(pff_out, bounds)
    return mu, sigma, None


if __name__ == "__main__":
    drm = load_drm(DRM_PATH)

    loaded = []
    for label, model_path, json_path, is_gated in VERSIONS:
        if not (os.path.exists(model_path) and os.path.exists(json_path)):
            print(f"  (skip {label}: {model_path} or {json_path} not found)")
            continue
        with open(json_path) as f:
            res = json.load(f)
        model = tf.keras.models.load_model(model_path, compile=False)
        loaded.append({
            "label": label,
            "model": model,
            "mean": np.array(res["norm_mean"], dtype=np.float32),
            "std": np.array(res["norm_std"], dtype=np.float32),
            "bounds": np.array(res["param_bounds"], dtype=np.float32),
            "is_gated": is_gated,
        })
    if not loaded:
        raise SystemExit("No trained PFF models found -- nothing to compare.")
    labels = [v["label"] for v in loaded]
    print(f"Comparing: {', '.join(labels)}")

    # ---------------------------------------------------------------
    # Part 1: real shots -- how big is sigma on out-of-distribution data
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"REAL SHOTS: sigma comparison ({', '.join(labels)}), physical units")
    print("=" * 100)
    rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            print(f"  (skip {name}: {path} not found)")
            continue
        signal_l1 = l1_normalise(load_signal(path))

        row = {"shot": name}
        print(f"\n--- {name} ---")
        header = f"  {'param':>4}  " + "  ".join(
            f"{(v['label'] + ': mu +/- sigma'):>22}" for v in loaded
        )
        print(header)
        for v in loaded:
            x = normalize_apply(signal_l1.reshape(1, -1), v["mean"], v["std"])
            out = v["model"].predict(x, verbose=0)
            mu, sigma, p_bump = decode(out, v["bounds"], v["is_gated"])
            v["_mu"], v["_sigma"], v["_p_bump"] = mu[0], sigma[0], (p_bump[0, 0] if p_bump is not None else None)
            for j, n in enumerate(PARAM_NAMES):
                row[f"{n}_sigma_{v['label']}"] = v["_sigma"][j]
            if v["_p_bump"] is not None:
                row[f"p_bump_{v['label']}"] = v["_p_bump"]

        for j, n in enumerate(PARAM_NAMES):
            parts = "  ".join(
                f"{v['_mu'][j]:>8.2f} +/- {v['_sigma'][j]:>9.2f}" for v in loaded
            )
            print(f"  {n:>4}  {parts}")
        for v in loaded:
            if v["_p_bump"] is not None:
                print(f"  p(bump) [{v['label']}] = {v['_p_bump']:.3f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("Summary: mean and max sigma across all real shots, per parameter")
    print("=" * 100)
    col_hdr = "  ".join(f"{(v['label'] + ' mean/max sig'):>20}" for v in loaded)
    print(f"{'param':>4}  {col_hdr}")
    for n in PARAM_NAMES:
        parts = []
        for v in loaded:
            col = f"{n}_sigma_{v['label']}"
            parts.append(f"{df[col].mean():>8.2f}/{df[col].max():>8.2f}")
        print(f"{n:>4}  " + "  ".join(parts))
    df.to_csv("pff_v1_v2_real_shot_sigma.csv", index=False)
    print("\nSaved pff_v1_v2_real_shot_sigma.csv")

    # How often does each version's sigma vary at all across the 14 shots?
    # (v2 showed literally zero variation -- this is the direct check for
    # whether that specific failure mode recurred.)
    print("\nSigma variation across the 14 real shots (std of per-shot sigma; ~0 = collapsed/constant):")
    for v in loaded:
        parts = []
        for n in PARAM_NAMES:
            col = f"{n}_sigma_{v['label']}"
            parts.append(f"{n}={df[col].std():.4f}")
        print(f"  {v['label']}: " + "  ".join(parts))

    # ---------------------------------------------------------------
    # Part 2: synthetic data -- accuracy + calibration, in-distribution
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"SYNTHETIC DATA ({SYNTH_SAMPLES} samples, seed={SYNTH_SEED}): accuracy + calibration")
    print("=" * 100)
    rng = np.random.default_rng(SYNTH_SEED)
    X, params_true = generate_pff_training_data(drm, SYNTH_SAMPLES, rng)
    bump_mask = params_true[:, 2] > 0.0

    for v in loaded:
        x = normalize_apply(X, v["mean"], v["std"])
        out = v["model"].predict(x, verbose=0)
        mu, sigma, p_bump = decode(out, v["bounds"], v["is_gated"])
        v["_mu_all"], v["_sigma_all"], v["_p_bump_all"] = mu, sigma, p_bump

    mae_hdr = "  ".join(f"{(v['label'] + ' MAE'):>10}" for v in loaded)
    print(f"{'param':>4}  {mae_hdr}")
    for j, n in enumerate(PARAM_NAMES):
        maes = [f"{np.abs(v['_mu_all'][:, j] - params_true[:, j]).mean():>10.3f}" for v in loaded]
        print(f"{n:>4}  " + "  ".join(maes))

    print()
    for v in loaded:
        z = (v["_mu_all"] - params_true) / np.maximum(v["_sigma_all"], 1e-6)
        cov_all = np.mean(np.abs(z[:, :3]) <= 1.0)
        cov_bump = np.mean(np.abs(z[bump_mask][:, 3:]) <= 1.0) if bump_mask.any() else float("nan")
        line = f"  {v['label']}: 1-sigma coverage a1/a2/a3={cov_all*100:.0f}%  a4/a5(bump)={cov_bump*100:.0f}%  (want ~68%)"
        if v["_p_bump_all"] is not None:
            bump_pred = v["_p_bump_all"][:, 0] > 0.5
            bump_acc = float(np.mean(bump_pred == bump_mask))
            line += f"  bump_acc={bump_acc*100:.1f}%"
        print(line)
