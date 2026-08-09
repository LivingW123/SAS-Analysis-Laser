"""
pff_domain_gap_check.py — Phase 2 of the plan to fix the PFF regressor's
real-shot uncertainty problem: quantify, rather than hand-wave, how far real
shots actually sit from the synthetic training distribution.

Compares three scale-invariant signal statistics between the 14 real shots
used throughout this session and a fresh batch drawn from the same generator
train_pff_v3_realistic_noise.py trains on:

  peak_frac        : peak channel index / 200 (where the signal peaks)
  early_frac       : fraction of total L1-normalised signal in channels 0-49
                      (bremsstrahlung-tail decay-rate proxy)
  sat_frac         : fraction of channels flagged as saturation-corrected /
                      saturation-clipped (real: load_saturation_mask; synthetic:
                      apply_saturation's own returned mask)

Doesn't train anything -- pure read/compute, cheap, meant to be re-run after
any change to the training-data generator to check whether the gap it's
meant to close actually shrank.

Usage
-----
  python pff_domain_gap_check.py
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import (
    add_detector_noise,
    apply_saturation,
    load_drm,
    load_saturation_mask,
    mev_bin_centers,
    sample_pff_spectra,
)

DRM_PATH = "res/drm/200x200.xlsx"
N_SYNTH = 500
SEED = 42
EARLY_CHANNELS = 50

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


def signal_stats(signal_l1: np.ndarray, sat_mask: np.ndarray | None) -> dict:
    return {
        "peak_frac": float(signal_l1.argmax()) / 200.0,
        "early_frac": float(signal_l1[:EARLY_CHANNELS].sum()),
        "sat_frac": float(sat_mask.mean()) if sat_mask is not None else 0.0,
    }


if __name__ == "__main__":
    drm = load_drm(DRM_PATH)

    # --- real shots ---
    real_rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            print(f"  (skip {name}: not found)")
            continue
        signal_l1 = l1_normalise(load_signal(path))
        sat_mask = load_saturation_mask(path)
        stats = signal_stats(signal_l1, sat_mask)
        stats["shot"] = name
        real_rows.append(stats)
    real_df = pd.DataFrame(real_rows)

    # --- synthetic, via the same low-level calls generate_pff_training_data
    #     makes internally (unwrapped here to keep each sample's sat_mask,
    #     which the wrapper discards) ---
    rng = np.random.default_rng(SEED)
    energy_bins = mev_bin_centers(drm.shape[1])
    spectra, _ = sample_pff_spectra(N_SYNTH, energy_bins, rng, bump_fraction=0.5)
    responses = (drm @ spectra.T).T
    noisy = add_detector_noise(responses, rng)
    noisy, sat_mask_all = apply_saturation(noisy, rng)
    row_sums = noisy.sum(axis=1, keepdims=True)
    X_synth = (noisy / np.maximum(row_sums, 1e-12)).astype(np.float32)

    synth_rows = []
    for i in range(N_SYNTH):
        stats = signal_stats(X_synth[i], sat_mask_all[i])
        stats["shot"] = f"synth_{i}"
        synth_rows.append(stats)
    synth_df = pd.DataFrame(synth_rows)

    # --- report ---
    print("=" * 80)
    print(f"REAL SHOTS (n={len(real_df)}) vs SYNTHETIC (n={len(synth_df)}) signal statistics")
    print("=" * 80)
    for stat in ["peak_frac", "early_frac", "sat_frac"]:
        r = real_df[stat]
        s = synth_df[stat]
        print(f"\n{stat}:")
        print(f"  real     : mean={r.mean():.3f}  min={r.min():.3f}  max={r.max():.3f}  "
              f"median={r.median():.3f}")
        print(f"  synthetic: mean={s.mean():.3f}  min={s.min():.3f}  max={s.max():.3f}  "
              f"median={s.median():.3f}")
        # what fraction of real shots fall outside the synthetic [min, max] range?
        outside = ((r < s.min()) | (r > s.max())).mean()
        print(f"  fraction of real shots outside synthetic's [min, max] range: {outside*100:.0f}%")

    print("\nPer-shot detail:")
    print(real_df[["shot", "peak_frac", "early_frac", "sat_frac"]].to_string(index=False))

    real_df.to_csv("pff_domain_gap_real_shots.csv", index=False)
    synth_df.to_csv("pff_domain_gap_synthetic.csv", index=False)

    # --- figure: sat_frac is the most directly diagnostic given the
    # root-cause hypothesis (train_pff_v3_realistic_noise.py's docstring) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    for ax, stat in zip(axes, ["peak_frac", "early_frac", "sat_frac"]):
        ax.hist(synth_df[stat], bins=30, alpha=0.6, color="steelblue", label="synthetic", density=True)
        for _, row in real_df.iterrows():
            ax.axvline(row[stat], color="firebrick", lw=1, alpha=0.7)
        ax.axvline(real_df[stat].iloc[0], color="firebrick", lw=1, alpha=0.7, label="real shots")
        ax.set_xlabel(stat)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.set_title(stat)

    out_path = "pff_domain_gap_check.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
    print("Saved pff_domain_gap_real_shots.csv and pff_domain_gap_synthetic.csv")
