"""
bump_center_profile_likelihood.py — sanity check before trusting either the
raw ensemble's a4 (1.0-43.7 MeV, wildly scattered) or
refine_bump_center_optimizer.py's a4-constrained-to-[10,20] refit (which
pinned to the 20 MeV boundary on 10/12 bump-present shots, a sign the
optimizer may just be running into its own wall rather than finding a
genuine optimum there).

This scans a4 across a fine grid over its FULL physical range (1-49 MeV,
data_utils.PFF_PARAM_BOUNDS), and at each fixed a4 refits a1/a2/a3/a5/a6 by
bounded least-squares against the real detector signal. The result is a
profile-likelihood curve (residual RMS vs a4) for each real shot -- an
answer to "which bump centre actually explains this shot's signal best" that
doesn't depend on any prior (physical or learned) at all, only the DRM
forward model and the raw channel data. This is the deciding evidence for
whether recentring the training prior on [10,20] MeV is fixing a real
mismatch or just teaching the network a different wrong answer.

Usage
-----
  python bump_center_profile_likelihood.py

Outputs
-------
  Console: per-shot best-fit a4 (global, unconstrained by any prior) vs
           residual RMS there, plus residual RMS at a4=15 (window centre)
           for comparison.
  bump_center_profile.csv : full (shot, a4, residual_rms) grid for plotting.
  bump_center_profile.png : residual-vs-a4 curves, all bump-present shots.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize

from data_utils import PFF_PARAM_BOUNDS, load_drm, mev_bin_centers, pff_func
from evaluate_pff_ensemble import SHOTS, l1_normalise, load_members, load_signal, predict_ensemble
from pff_ensemble_utils import decode_ensemble

DRM_PATH = "res/drm/200x200.xlsx"
A4_GRID = np.arange(1.0, 49.01, 1.0)   # full physical range, 1 MeV steps
P_BUMP_THRESHOLD = 0.5


def profile_at_fixed_a4(a4: float, x0_other: np.ndarray, drm: np.ndarray,
                         energy_bins: np.ndarray, signal_l1: np.ndarray) -> float:
    """Refit a1,a2,a3,a5,a6 with a4 held fixed; return best residual RMS."""
    lo_full = PFF_PARAM_BOUNDS[:, 0].copy()
    hi_full = PFF_PARAM_BOUNDS[:, 1].copy()
    lo_full[2] = 5.0   # keep bump "on"

    other_idx = [0, 1, 2, 4, 5]
    lo = lo_full[other_idx]
    hi = hi_full[other_idx]
    x0 = np.clip(x0_other[other_idx], lo, hi)

    def residual(p_other: np.ndarray) -> np.ndarray:
        params = np.empty(6)
        params[other_idx] = p_other
        params[3] = a4
        spec = pff_func(energy_bins, params)
        resp = drm @ spec
        resp_l1 = l1_normalise(resp)
        return resp_l1 - signal_l1

    result = optimize.least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
    return float(np.sqrt(np.mean(result.fun ** 2)))


if __name__ == "__main__":
    members = load_members()
    if len(members) < 2:
        raise SystemExit(f"Need at least 2 trained ensemble members; found {len(members)}.")
    print(f"Loaded {len(members)}/5 ensemble members")

    drm = load_drm(DRM_PATH)
    energy_bins = mev_bin_centers(drm.shape[1])

    all_rows = []
    summary_rows = []
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)

    for name, path in SHOTS:
        if not os.path.exists(path):
            continue
        signal_l1 = l1_normalise(load_signal(path))
        mus, sigmas, p_bumps = predict_ensemble(members, signal_l1.reshape(1, -1))
        mean, _, _, pb_mean, _ = decode_ensemble(mus, sigmas, p_bumps)
        mean = mean[0]
        pb_mean = float(pb_mean[0, 0])
        if pb_mean <= P_BUMP_THRESHOLD:
            continue

        rms_grid = np.array([profile_at_fixed_a4(a4, mean, drm, energy_bins, signal_l1) for a4 in A4_GRID])
        best_idx = int(np.argmin(rms_grid))
        best_a4, best_rms = A4_GRID[best_idx], rms_grid[best_idx]
        rms_at_15 = rms_grid[int(np.argmin(np.abs(A4_GRID - 15.0)))]
        in_window = 10.0 <= best_a4 <= 20.0

        print(f"{name:>10}: global-best a4={best_a4:5.1f} MeV (resid={best_rms:.6f})  "
              f"resid@a4=15={rms_at_15:.6f}  {'IN [10,20]' if in_window else 'OUTSIDE [10,20]'}  "
              f"(raw NN a4={mean[3]:.1f})")

        for a4, rms in zip(A4_GRID, rms_grid):
            all_rows.append({"shot": name, "a4": a4, "resid_rms": rms})
        summary_rows.append({"shot": name, "p_bump": pb_mean, "a4_nn": mean[3],
                              "a4_global_best": best_a4, "resid_at_best": best_rms,
                              "resid_at_15": rms_at_15, "in_window_10_20": in_window})

        ax.plot(A4_GRID, rms_grid, marker="o", ms=3, lw=1, label=name)

    ax.axvspan(10, 20, color="green", alpha=0.1, label="physical prior [10,20]")
    ax.set_xlabel("a4 (MeV, fixed)")
    ax.set_ylabel("Best-fit residual RMS (L1 space)")
    ax.set_title("Profile likelihood: residual vs bump centre, real shots")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig("bump_center_profile.png", dpi=150)

    pd.DataFrame(all_rows).to_csv("bump_center_profile.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("bump_center_profile_summary.csv", index=False)
    print(f"\n{summary_df['in_window_10_20'].sum()}/{len(summary_df)} bump-present shots have their "
          f"GLOBAL best-fit (prior-free) a4 inside [10,20] MeV")
    print("Saved bump_center_profile.png, bump_center_profile.csv, bump_center_profile_summary.csv")
