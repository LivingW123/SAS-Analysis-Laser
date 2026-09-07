"""
tsvd.py — Python port of `matlab/TSVD_NN.m`, the truncated-SVD unfolding
reference this whole project is the ML alternative to.

Until now the TSVD baseline only existed as MATLAB that nobody could run
inside the Python pipeline, so every "our model beats/matches the classical
method" claim was unquantified. This module makes the classical method a
first-class, callable estimator on exactly the same inputs (200-channel
detector vector) and outputs (spectrum over 0-50 MeV) as the CNN / DNN / PFF
regressors, so `src/comparisons/method_comparison.py` can score all of them
side by side.

What the MATLAB actually does (and what is reproduced here)
-----------------------------------------------------------
1. `EDRM = x200'` — same orientation as `data_utils.load_drm` (rows = the 200
   detector channels, cols = energy bins).
2. Energy bins are grouped `gp_sz` at a time by **sum** (not mean, unlike
   `data_utils.bin_drm`) into `new_EDRM`, (200, 200/gp_sz). MATLAB uses
   `gp_sz = 2` -> 100 energy bins. Sum vs mean is a constant factor per column,
   which cancels once a spectrum is L1-normalised for comparison, so the two
   binnings are interchangeable for scoring; `group_drm` keeps the sum so the
   port stays faithful to the reference at its own scale.
3. `[U,S,V] = svd(new_EDRM)`, then a truncated inverse over the first
   `num_terms = 7` singular directions:
       result = sum_i V[:,i] * (U[:,i].T @ b) / s_i
   with one wrinkle: for `i > 6` the divisor is *clamped* to `s_6` instead of
   `s_i` (`constraint = S(6,6)`). That is a hand-rolled damping of the 7th
   term — without it the 7th direction, whose singular value is far smaller,
   would dominate the reconstruction with amplified noise. `tsvd_solve`
   reproduces this via `clamp_from` (set `clamp_from=None` for a textbook TSVD).
4. `lsqcurvefit` then refits the 7 subspace coefficients against the measured
   vector *through a non-negativity projection*:
       minimize_c || new_EDRM @ max(V[:, :7] @ c, 0) - b ||^2
   starting from `c0`. Note the MATLAB computes `c0 = V(:,1:num_terms)' *
   result` on line 57 while `result` is still `zeros` (the TSVD loop that
   fills `result` runs afterwards, on line 63) — so the reference genuinely
   starts from the zero vector, not from its own TSVD estimate. `tsvd_nn_fit`
   defaults to `c0="tsvd"` (the warm start the MATLAB looks like it intended)
   but takes `c0="zeros"` to reproduce the reference exactly; on this DRM the
   two land in the same place for real shots, and the warm start converges in
   noticeably fewer function evaluations.
5. `result1 = positive_def(V[:, :7] @ c)` — the clipped spectrum is the answer.

`positive_def` in MATLAB is elementwise `max(x, 0)`, applied *inside* the
objective, which is why this is a projected fit rather than a bounded one:
the coefficients `c` are unconstrained and the clip happens on the spectrum.
That makes the residual non-smooth at the clip boundary; MATLAB's
`lsqcurvefit` tolerates it and so does `scipy.optimize.least_squares`'s
default 'trf' with a 2-point Jacobian, which is what is used here.

The single most important limitation to remember when reading the comparison
results: with `num_terms = 7` this estimator has **7 degrees of freedom**,
independent of how many energy bins the output grid has. It cannot represent
structure the first 7 right-singular vectors of the DRM do not span, and it
carries no uncertainty estimate of any kind.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# TSVD_NN.m's own settings.
GP_SZ = 2         # energy bins summed per group -> 100 bins over 0-50 MeV
NUM_TERMS = 7     # singular directions kept
CLAMP_FROM = 6    # divisors for term index > CLAMP_FROM are clamped to s[CLAMP_FROM-1]


def group_drm(drm: np.ndarray, gp_sz: int = GP_SZ) -> np.ndarray:
    """
    Sum every `gp_sz` consecutive energy-bin columns (TSVD_NN.m's `new_EDRM`).

    Parameters
    ----------
    drm   : (200, n_energy) detector-channel x energy-bin matrix, as returned
            by `data_utils.load_drm`
    gp_sz : energy bins per group; must divide the energy-bin count

    Returns
    -------
    (200, n_energy // gp_sz) grouped matrix
    """
    n_chan, n_energy = drm.shape
    assert n_energy % gp_sz == 0, f"gp_sz={gp_sz} must divide {n_energy} energy bins"
    return drm.reshape(n_chan, n_energy // gp_sz, gp_sz).sum(axis=2)


def svd_basis(drm_grouped: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin SVD of the grouped DRM, matching MATLAB's `[U,S,V] = svd(new_EDRM)`
    for the columns that matter (the first min(m,n) of each).

    Returns
    -------
    U : (200, k) left singular vectors (detector-channel space)
    s : (k,)     singular values, descending
    V : (n, k)   right singular vectors (energy space) — column j is direction j,
                 matching MATLAB's V(:,j) rather than numpy's Vh row convention
    """
    U, s, Vh = np.linalg.svd(drm_grouped, full_matrices=False)
    return U, s, Vh.T


def condition_number(s: np.ndarray) -> float:
    """s[0]/s[-1] — the number TSVD_NN.m prints at step [2/5]."""
    return float(s[0] / s[-1])


def tsvd_solve(
    b: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    num_terms: int = NUM_TERMS,
    clamp_from: int | None = CLAMP_FROM,
) -> np.ndarray:
    """
    Truncated-SVD inversion: sum over the first `num_terms` singular
    directions of V[:,i] * (U[:,i].T @ b) / divisor_i.

    `clamp_from` reproduces TSVD_NN.m's damping: for 1-based term index
    i > clamp_from, the divisor is s[clamp_from - 1] instead of s[i - 1],
    which shrinks the contribution of the noisiest retained direction.
    Pass None for a textbook (undamped) TSVD.

    Returns the raw spectrum estimate — NOT clipped to non-negative; use
    `np.maximum(x, 0)` (MATLAB's `positive_def`) if a physical spectrum is
    wanted. Left unclipped here because the negative excursions are the
    diagnostic that says the truncation level is fighting the noise.
    """
    coeffs = U[:, :num_terms].T @ b                       # (num_terms,)
    divisors = s[:num_terms].copy()
    if clamp_from is not None and num_terms > clamp_from:
        divisors[clamp_from:] = s[clamp_from - 1]
    return V[:, :num_terms] @ (coeffs / divisors)


def positive_def(x: np.ndarray) -> np.ndarray:
    """MATLAB `positive_def.m`: elementwise max(x, 0)."""
    return np.maximum(x, 0.0)


def tsvd_nn_fit(
    b: np.ndarray,
    drm_grouped: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    num_terms: int = NUM_TERMS,
    clamp_from: int | None = CLAMP_FROM,
    c0: str | np.ndarray = "tsvd",
    max_nfev: int = 100_000,
) -> dict:
    """
    TSVD_NN.m's `lsqcurvefit` stage: refit the `num_terms` subspace
    coefficients so the *non-negativity-projected* spectrum best reproduces
    the measured detector vector.

        minimize_c || drm_grouped @ max(V[:, :num_terms] @ c, 0) - b ||^2

    Parameters
    ----------
    b           : (200,) measured detector vector
    drm_grouped : (200, n_bins) grouped DRM (see `group_drm`)
    U, s, V     : output of `svd_basis(drm_grouped)`
    c0          : "tsvd" to warm-start from `tsvd_solve`'s coefficients,
                  "zeros" to reproduce the MATLAB's actual starting point,
                  or an explicit (num_terms,) array
    max_nfev    : function-evaluation budget handed to least_squares

    Returns
    -------
    dict with keys:
      spectrum   : (n_bins,) the fitted, non-negativity-clipped spectrum
      coeffs     : (num_terms,) fitted subspace coefficients
      response   : (200,) drm_grouped @ spectrum
      resnorm    : ||response - b||^2
      rel_resid  : ||response - b|| / ||b||   (TSVD_NN.m's "Relative res")
      nfev       : function evaluations used
      success    : optimizer's convergence flag
      tsvd_raw   : (n_bins,) the pre-refit truncated-SVD estimate, unclipped
    """
    Vk = V[:, :num_terms]
    tsvd_raw = tsvd_solve(b, U, s, V, num_terms, clamp_from)

    if isinstance(c0, str):
        if c0 == "tsvd":
            c_start = Vk.T @ tsvd_raw
        elif c0 == "zeros":
            c_start = np.zeros(num_terms)
        else:
            raise ValueError(f"c0 must be 'tsvd', 'zeros', or an array; got {c0!r}")
    else:
        c_start = np.asarray(c0, dtype=np.float64)

    def residual(c: np.ndarray) -> np.ndarray:
        return drm_grouped @ positive_def(Vk @ c) - b

    fit = least_squares(residual, c_start, method="trf", max_nfev=max_nfev)

    spectrum = positive_def(Vk @ fit.x)
    response = drm_grouped @ spectrum
    diff = response - b
    resnorm = float(diff @ diff)
    return {
        "spectrum": spectrum,
        "coeffs": fit.x,
        "response": response,
        "resnorm": resnorm,
        "rel_resid": float(np.sqrt(resnorm) / np.linalg.norm(b)),
        "nfev": int(fit.nfev),
        "success": bool(fit.success),
        "tsvd_raw": tsvd_raw,
    }


def unfold(
    b: np.ndarray,
    drm: np.ndarray,
    gp_sz: int = GP_SZ,
    num_terms: int = NUM_TERMS,
    clamp_from: int | None = CLAMP_FROM,
    c0: str | np.ndarray = "tsvd",
) -> dict:
    """
    One-call convenience wrapper: group the DRM, take its SVD, and run the
    projected non-negative refit. Recomputes the SVD on every call, so use
    `group_drm` + `svd_basis` + `tsvd_nn_fit` directly when unfolding many
    shots against the same DRM.
    """
    drm_g = group_drm(drm, gp_sz)
    U, s, V = svd_basis(drm_g)
    out = tsvd_nn_fit(b, drm_g, U, s, V, num_terms, clamp_from, c0)
    out["condition_number"] = condition_number(s)
    out["singular_values"] = s
    return out
