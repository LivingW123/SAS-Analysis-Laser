"""
Hybrid CNN+NNLS refinement: use a CNN's predicted spectrum as a warm-start
prior for a constrained, Tikhonov-regularized least-squares refit against the
actual real-shot vector, instead of trusting the CNN's raw output (which
generalizes from synthetic training data and may not fit this specific
shot's noise realization) or an unregularized NNLS fit (which fits the
specific shot's noise as much as the signal -- noise_sweep.py's nnls_floor
is a *floor*, not a spectrum estimate anyone should trust as-is).

  minimize  ||drm_b_norm @ x - signal_l1||^2 + lam^2 * ||x - cnn_spec||^2,  x >= 0

Solved via the standard NNLS augmentation trick: stack sqrt(lam^2)*I onto A
and lam*cnn_spec onto b, then call scipy.optimize.nnls on the augmented
system. lam=0 reduces to (a numerically better-conditioned version of)
noise_sweep.nnls_floor (ignores the CNN entirely); lam -> inf keeps x ==
cnn_spec (ignores the data). The useful regime is in between: pulled toward
the DRM/vector fit but regularized toward a physically-plausible shape
instead of free to fit noise in all n_bins degrees of freedom.

drm_b_norm: the raw DRM (drm_b) holds Geant4 counts, O(1e7-1e8), while
signal_l1 and the CNN's softmax spectrum are both L1-normalised (sum to 1,
entries O(1e-2)). Fitting directly against drm_b (as noise_sweep.nnls_floor
does, relying on a post-hoc L1-normalisation of the *response* to cancel the
scale) leaves the fitted spectrum itself ~1e-9 in magnitude -- fine for a
scale-invariant residual metric, but it makes ||x - cnn_spec||^2 meaningless
(x and cnn_spec differ by ~8 orders of magnitude, so no sane lam moves the
fit). Column-normalizing the DRM (each energy bin's response shape divided
by its own sum) fixes this: for any x summing to 1, drm_b_norm @ x also
sums to 1 by construction, exactly matching signal_l1's scale with no
post-hoc renormalization, and putting x in the same units as cnn_spec. This
is an exact reparametrisation (x_norm = x_raw * col_sums), not an
approximation -- same feasible set, same objective value at corresponding
points -- so the achievable floor residual is unchanged in principle; in
practice it also converges to a *better* (lower-residual) optimum here,
because drm_b's O(1e7-1e8) dynamic range was leaving scipy.optimize.nnls
poorly conditioned.

Usage
-----
  python nnls_refine.py                    # default shots, n_bins=10, sweeps lambda
  CNN_NBINS=20 python nnls_refine.py
  CNN_MODEL_PREFIX=model_cnn_denoised CNN_RESULTS_JSON=cnn_denoised_training_results.json \
      python nnls_refine.py   # use the joint-trained model's predictions as x0
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import nnls

from data_utils import bin_drm, load_drm, mev_bin_centers, normalize_apply

DRM_PATH = "res/drm/200x200.xlsx"
N_BINS = int(os.environ.get("CNN_NBINS", 10))
MODEL_PREFIX = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
CNN_JSON = os.environ.get("CNN_RESULTS_JSON", "cnn_training_results.json")
IS_JOINT = MODEL_PREFIX == "model_cnn_denoised"  # joint model: no static norm stats, no reshape helper needed the same way

LAMBDAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

DEFAULT_SHOTS = [
    ("11696", "res/test_images/11696/11696_proc_vector.csv"),
    ("11696_ch", "res/test_images/11696/11696_proc_vector_ch.csv"),
    ("11705_ch", "res/test_images/11705/11705_proc_vector_ch.csv"),
    ("11707_ch", "res/test_images/11707/11707_proc_vector_ch.csv"),
    ("11716_ch", "res/test_images/11716/11716_proc_vector_ch.csv"),
    ("11733", "res/test_images/11733/11733_proc_vector_cv.csv"),
    ("10084", "res/test_images/10084/10084_proc_vector_cv.csv"),
]


def l1(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return (x / s).astype(np.float64) if s > 0 else x


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float64)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def normalize_drm_columns(drm_b: np.ndarray) -> np.ndarray:
    """Scale each energy-bin column to sum to 1, so a spectrum x summing to 1
    forward-projects to a response that already sums to 1 -- see module
    docstring. Exact reparametrisation of the raw DRM fit, just in units
    comparable to the CNN's L1-normalised softmax output."""
    col_sums = drm_b.sum(axis=0, keepdims=True)
    return drm_b / col_sums


def resid_pct(drm_b_norm: np.ndarray, signal_l1: np.ndarray, spec: np.ndarray) -> float:
    resp = l1(drm_b_norm @ spec)
    r = np.sqrt(np.mean((signal_l1 - resp) ** 2))
    return 100.0 * r / np.sqrt(np.mean(signal_l1 ** 2))


def refine_spectrum(drm_b_norm: np.ndarray, signal_l1: np.ndarray, x0: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov-regularized NNLS anchored to x0 (both in drm_b_norm's units,
    i.e. L1-normalised-spectrum scale). lam=0 -> plain NNLS floor."""
    if lam == 0.0:
        spec, _ = nnls(drm_b_norm, signal_l1)
        return spec
    n_bins = drm_b_norm.shape[1]
    A_aug = np.vstack([drm_b_norm, lam * np.eye(n_bins)])
    b_aug = np.concatenate([signal_l1, lam * x0])
    spec, _ = nnls(A_aug, b_aug)
    return spec


def get_cnn_spec(signal_l1: np.ndarray, drm_b: np.ndarray) -> np.ndarray:
    if IS_JOINT:
        model = tf.keras.models.load_model(f"{MODEL_PREFIX}_n{N_BINS}.keras", compile=False)
        x = signal_l1.reshape(1, 200, 1).astype(np.float32)
        _, spec_batch = model.predict(x, verbose=0)
        return spec_batch[0].astype(np.float64)

    from train_cnn import reshape_to_2d
    with open(CNN_JSON) as f:
        res = json.load(f)[str(N_BINS)]
    mean = np.array(res["norm_mean"], dtype=np.float32)
    std = np.array(res["norm_std"], dtype=np.float32)
    model = tf.keras.models.load_model(f"{MODEL_PREFIX}_n{N_BINS}.keras", compile=False)
    x_norm = reshape_to_2d(normalize_apply(signal_l1.reshape(1, -1).astype(np.float32), mean, std))
    return model.predict(x_norm, verbose=0)[0].astype(np.float64)


def main() -> None:
    drm = load_drm(DRM_PATH)
    drm_b = bin_drm(drm, N_BINS)
    drm_b_norm = normalize_drm_columns(drm_b)

    shots = DEFAULT_SHOTS if len(sys.argv) <= 1 else [
        (os.path.basename(p).split("_")[0], p) for p in sys.argv[1:]
    ]

    rows = []
    for name, path in shots:
        if not os.path.exists(path):
            continue
        signal_l1 = l1(load_signal(path))
        cnn_spec = get_cnn_spec(signal_l1, drm_b)
        cnn_resid = resid_pct(drm_b_norm, signal_l1, cnn_spec)
        floor_spec = refine_spectrum(drm_b_norm, signal_l1, cnn_spec, 0.0)
        floor_resid = resid_pct(drm_b_norm, signal_l1, floor_spec)

        print(f"\n=== {name}  (n_bins={N_BINS}, model_prefix={MODEL_PREFIX}) ===")
        print(f"  CNN alone        : resid {cnn_resid:5.1f}%")
        print(f"  NNLS floor (lam=0): resid {floor_resid:5.1f}%")

        row = {"shot": name, "cnn_resid_pct": round(cnn_resid, 1), "nnls_floor_pct": round(floor_resid, 1)}
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            spec = refine_spectrum(drm_b_norm, signal_l1, cnn_spec, lam)
            r = resid_pct(drm_b_norm, signal_l1, spec)
            # how far the refined spectrum drifted from the CNN's own prediction,
            # in L1 terms -- a proxy for "did lam keep it anchored or let it roam"
            drift = float(np.abs(spec - cnn_spec).sum())
            print(f"  lam={lam:<6g}      : resid {r:5.1f}%   |x - cnn_spec|_1 = {drift:.3f}")
            row[f"lam_{lam}_resid_pct"] = round(r, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = f"nnls_hybrid_eval_n{N_BINS}{'_denoised' if IS_JOINT else ''}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
