"""
evaluate_pff_ensemble.py — the actual test of the deep-ensemble hypothesis
(train_pff_ensemble_member.py, pff_ensemble_utils.py): does cross-member
disagreement (epistemic uncertainty) come out higher on real, out-of-
distribution shots than on synthetic, in-distribution samples?

If it does, the ensemble gives a genuinely informative "trust this less"
signal that no single model this session (v1/v2/v3) could produce -- each of
those either exploded to nonsense or collapsed to a fixed constant on real
shots, regardless of how unfamiliar the specific input actually was. If
epistemic variance does NOT come out higher on real shots, the ensemble
hasn't bought anything beyond averaging, and that's worth knowing plainly
too.

Runs all N_MEMBERS models on the same 14 real shot files used throughout
this session, plus a synthetic batch drawn from the same generator training
uses (known in-distribution by construction), and reports both epistemic and
aleatoric variance separately for both sets.

Usage
-----
  python evaluate_pff_ensemble.py
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import generate_pff_training_data, load_drm, normalize_apply
from pff_ensemble_utils import decode_ensemble
from train_pff_bounded_gated import decode_v2

PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6"]
N_MEMBERS = 5
SYNTH_SAMPLES = 200
SYNTH_SEED = 123
DRM_PATH = "res/drm/200x200.xlsx"

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


def load_members():
    members = []
    for idx in range(N_MEMBERS):
        model_path = f"model_pff_ensemble_{idx}.keras"
        json_path = f"pff_training_results_ensemble_{idx}.json"
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


if __name__ == "__main__":
    members = load_members()
    if len(members) < 2:
        raise SystemExit(f"Need at least 2 trained members to form an ensemble; found {len(members)}.")
    print(f"Loaded {len(members)}/{N_MEMBERS} ensemble members")
    bounds = members[0]["bounds"]  # identical across members (same PFF_PARAM_BOUNDS)

    drm = load_drm(DRM_PATH)

    # ---------------------------------------------------------------
    # Part 1: real shots
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REAL SHOTS: ensemble mean/sigma (total = aleatoric + epistemic), physical units")
    print("=" * 100)
    real_rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            continue
        signal_l1 = l1_normalise(load_signal(path))
        mus, sigmas, p_bumps = predict_ensemble(members, signal_l1.reshape(1, -1))
        mean, sigma_total, sigma_epi, pb_mean, pb_std = decode_ensemble(mus, sigmas, p_bumps)
        mean, sigma_total, sigma_epi = mean[0], sigma_total[0], sigma_epi[0]
        pb_mean, pb_std = float(pb_mean[0, 0]), float(pb_std[0, 0])

        print(f"\n--- {name} ---  p(bump)={pb_mean:.3f}+/-{pb_std:.3f}")
        for j, n in enumerate(PARAM_NAMES):
            print(f"  {n:>4} = {mean[j]:8.2f}  sigma_total={sigma_total[j]:8.2f}  "
                  f"sigma_epistemic={sigma_epi[j]:8.2f}  ({100*sigma_epi[j]**2/max(sigma_total[j]**2,1e-9):.0f}% of variance)")

        row = {"shot": name, "is_real": True, "p_bump_mean": pb_mean, "p_bump_std": pb_std}
        for j, n in enumerate(PARAM_NAMES):
            row[f"{n}_mean"] = mean[j]
            row[f"{n}_sigma_total"] = sigma_total[j]
            row[f"{n}_sigma_epistemic"] = sigma_epi[j]
        real_rows.append(row)
    real_df = pd.DataFrame(real_rows)

    # ---------------------------------------------------------------
    # Part 2: synthetic, in-distribution by construction
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"SYNTHETIC DATA ({SYNTH_SAMPLES} samples, seed={SYNTH_SEED}, in-distribution by construction)")
    print("=" * 100)
    rng = np.random.default_rng(SYNTH_SEED)
    X_synth, params_true = generate_pff_training_data(drm, SYNTH_SAMPLES, rng)
    mus, sigmas, p_bumps = predict_ensemble(members, X_synth)
    mean, sigma_total, sigma_epi, pb_mean, pb_std = decode_ensemble(mus, sigmas, p_bumps)

    synth_rows = []
    for i in range(SYNTH_SAMPLES):
        row = {"shot": f"synth_{i}", "is_real": False,
               "p_bump_mean": float(pb_mean[i, 0]), "p_bump_std": float(pb_std[i, 0])}
        for j, n in enumerate(PARAM_NAMES):
            row[f"{n}_mean"] = mean[i, j]
            row[f"{n}_sigma_total"] = sigma_total[i, j]
            row[f"{n}_sigma_epistemic"] = sigma_epi[i, j]
        synth_rows.append(row)
    synth_df = pd.DataFrame(synth_rows)

    mae = {n: float(np.abs(mean[:, j] - params_true[:, j]).mean()) for j, n in enumerate(PARAM_NAMES)}
    z = (mean - params_true) / np.maximum(sigma_total, 1e-6)
    bump_mask = params_true[:, 2] > 0.0
    cov_all = np.mean(np.abs(z[:, :3]) <= 1.0)
    cov_bump = np.mean(np.abs(z[bump_mask][:, 3:]) <= 1.0) if bump_mask.any() else float("nan")
    print(f"Ensemble-mean MAE: " + " | ".join(f"{n}={mae[n]:.3f}" for n in PARAM_NAMES))
    print(f"1-sigma coverage (total var): a1/a2/a3={cov_all*100:.0f}%  a4/a5(bump)={cov_bump*100:.0f}%  (want ~68%)")

    # ---------------------------------------------------------------
    # THE key check: is epistemic variance higher on real (OOD) shots
    # than on synthetic (in-distribution) samples?
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("THE HYPOTHESIS CHECK: epistemic sigma, real shots vs. synthetic (in-distribution)")
    print("=" * 100)
    print(f"{'param':>4}  {'real mean':>10}  {'synth mean':>11}  {'ratio':>8}")
    for n in PARAM_NAMES:
        real_epi = real_df[f"{n}_sigma_epistemic"].mean()
        synth_epi = synth_df[f"{n}_sigma_epistemic"].mean()
        ratio = real_epi / synth_epi if synth_epi > 0 else float("inf")
        print(f"{n:>4}  {real_epi:>10.3f}  {synth_epi:>11.3f}  {ratio:>7.2f}x")

    real_bump_std = real_df["p_bump_std"].mean()
    synth_bump_std = synth_df["p_bump_std"].mean()
    print(f"\np(bump) cross-member std: real={real_bump_std:.3f}  synth={synth_bump_std:.3f}  "
          f"ratio={real_bump_std/max(synth_bump_std,1e-9):.2f}x")

    combined = pd.concat([real_df, synth_df], ignore_index=True)
    combined.to_csv("pff_ensemble_eval.csv", index=False)
    print("\nSaved pff_ensemble_eval.csv")

    # --- figure: epistemic sigma distribution, real vs synthetic ---
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 4), tight_layout=True)
    for ax, n in zip(axes, PARAM_NAMES):
        ax.hist(synth_df[f"{n}_sigma_epistemic"], bins=20, alpha=0.6, color="steelblue",
                label="synthetic (in-dist)", density=True)
        for v in real_df[f"{n}_sigma_epistemic"]:
            ax.axvline(v, color="firebrick", lw=1, alpha=0.6)
        ax.axvline(real_df[f"{n}_sigma_epistemic"].iloc[0], color="firebrick", lw=1, alpha=0.6, label="real shots")
        ax.set_xlabel(f"{n} epistemic sigma")
        ax.legend(fontsize=7)
    out_path = "pff_ensemble_epistemic_check.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
