"""
refine_bump_center_optimizer.py — "NN proposes, optimizer refines" hybrid for
the bump-centre (a4) accuracy problem.

evaluate_pff_ensemble.py's real-shot a4 predictions range 1.0-43.7 MeV with
sigma_total 6-13 MeV, well outside the physically expected 10-20 MeV window.
Two other scripts in this session attack this by fixing the *training*
distribution (train_pff_bumpcenter_prior.py); this one instead treats the
existing 5-member ensemble's output as a good starting guess for a1/a2/a3/
a5/a6, then runs a bounded nonlinear least-squares fit (scipy.optimize,
same tool src/core/data_utils.py already uses for the saturation-column fit)
directly against the real detector signal with a4 hard-constrained to
[10,20] MeV -- the same role PFF.m's classical fit plays, but seeded from
the network instead of a from-scratch initial guess.

Only refines shots the ensemble thinks have a bump at all (p_bump_mean >
0.5) -- forcing a bump fit onto a no-bump shot would just fit noise inside
[10,20].

Usage
-----
  python -m src.optimizers.refine_bump_center_optimizer

Outputs
-------
  Console: per-shot raw ensemble a4 vs optimizer-refined a4, plus L1-space
           residual RMS before/after refinement (does the constrained fit
           actually explain the real signal better, or is it just forcing
           a4 into range at the cost of fit quality?)
  out/optimizers/pff_bumpcenter_refined.csv
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import scipy.optimize as optimize

from src.core.data_utils import PFF_PARAM_BOUNDS, load_drm, mev_bin_centers, pff_func
from src.core.real_shots import SHOTS, l1_normalise, load_members, load_signal, predict_ensemble
from src.core.pff_ensemble_utils import decode_ensemble

OUT_DIR = "out/optimizers"
os.makedirs(OUT_DIR, exist_ok=True)

DRM_PATH = "res/drm/200x200.xlsx"
PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6"]
A4_LO, A4_HI = 10.0, 20.0   # hard physical constraint for the refinement fit
P_BUMP_THRESHOLD = 0.5


def refine_params(x0: np.ndarray, drm: np.ndarray, energy_bins: np.ndarray,
                   signal_l1: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Bounded least-squares refit of all 6 PFF params against the real
    L1-normalised signal, a4 constrained to [A4_LO, A4_HI]. Returns
    (refined_params, resid_rms_before, resid_rms_after)."""
    lo = PFF_PARAM_BOUNDS[:, 0].copy()
    hi = PFF_PARAM_BOUNDS[:, 1].copy()
    lo[2] = 5.0          # keep the bump "on" (a3 >= 5, matches bump-present sampling floor)
    lo[3], hi[3] = A4_LO, A4_HI

    x0 = np.clip(x0, lo, hi)

    def residual(params: np.ndarray) -> np.ndarray:
        spec = pff_func(energy_bins, params)
        resp = drm @ spec
        resp_l1 = l1_normalise(resp)
        return resp_l1 - signal_l1

    rms_before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    result = optimize.least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    rms_after = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return result.x, rms_before, rms_after


if __name__ == "__main__":
    members = load_members()
    if len(members) < 2:
        raise SystemExit(f"Need at least 2 trained ensemble members; found {len(members)}.")
    print(f"Loaded {len(members)}/5 ensemble members")

    drm = load_drm(DRM_PATH)
    energy_bins = mev_bin_centers(drm.shape[1])

    print(f"\n{'shot':>10}  {'p(bump)':>8}  {'a4_nn':>8}  {'a4_refined':>10}  "
          f"{'resid_before':>12}  {'resid_after':>11}")

    rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            continue
        signal_l1 = l1_normalise(load_signal(path))
        mus, sigmas, p_bumps = predict_ensemble(members, signal_l1.reshape(1, -1))
        mean, sigma_total, sigma_epi, pb_mean, pb_std = decode_ensemble(mus, sigmas, p_bumps)
        mean, sigma_total = mean[0], sigma_total[0]
        pb_mean = float(pb_mean[0, 0])

        row = {"shot": name, "p_bump": pb_mean, "a4_nn": mean[3], "a4_nn_sigma": sigma_total[3]}

        if pb_mean > P_BUMP_THRESHOLD:
            refined, rms_before, rms_after = refine_params(mean, drm, energy_bins, signal_l1)
            print(f"{name:>10}  {pb_mean:>8.3f}  {mean[3]:>8.2f}  {refined[3]:>10.2f}  "
                  f"{rms_before:>12.6f}  {rms_after:>11.6f}")
            row.update({
                "a4_refined": refined[3], "resid_rms_before": rms_before, "resid_rms_after": rms_after,
                **{f"{n}_refined": refined[j] for j, n in enumerate(PARAM_NAMES)},
            })
        else:
            print(f"{name:>10}  {pb_mean:>8.3f}  {mean[3]:>8.2f}  {'(no bump)':>10}  "
                  f"{'--':>12}  {'--':>11}")
            row.update({"a4_refined": None, "resid_rms_before": None, "resid_rms_after": None})
        rows.append(row)

    out_path = os.path.join(OUT_DIR, "pff_bumpcenter_refined.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
