"""
data_utils.py — DRM loading, energy-bin downsampling, Poisson noise, normalization,
                and PFF-based broad-spectrum training data generation.

DRM orientation (from TSVD_NN.m: EDRM = x200'):
  xlsx rows = energy bins (200), xlsx cols = detector channels (200)
  After transpose: DRM rows = detector channels, cols = energy bins
"""

import numpy as np
import pandas as pd

# PFF form:  a1*exp(-a2*x) + a3*exp(-(x-a4)^2 / a5)
# Columns: [mean, std, lo, hi] used for bounded-normal sampling.
# a3 (bump amplitude) bounds are for the bump-present case; set to 0 for no-bump.
PFF_PARAM_SAMPLING = np.array([
    [200.0, 200.0, 0.1,  500.0],   # a1 — bremsstrahlung amplitude
    [0.25,  2.0,   0.01,   5.0],   # a2 — bremsstrahlung decay rate (1/MeV)
    [8.0,   10.0,  5.0,  100.0],   # a3 — bump amplitude (bump-present lo = 5)
    [35.0,  25.0,  1.0,   49.0],   # a4 — bump centre (MeV)
    [40.0,  25.0,  5.0,  100.0],   # a5 — bump width parameter
])

# Absolute bounds used for [0,1] normalization of parameters.
# No-bump samples have a3=0, which maps to 0.0 after normalization.
PFF_PARAM_BOUNDS = np.array([
    [0.1,  500.0],   # a1
    [0.01,   5.0],   # a2
    [0.0,  100.0],   # a3
    [1.0,   49.0],   # a4
    [5.0,  100.0],   # a5
])


def load_drm(xlsx_path: str) -> np.ndarray:
    """Load 200×200 xlsx and transpose so shape = (200 det channels, 200 energy bins)."""
    df = pd.read_excel(xlsx_path, header=None)
    drm = df.values.astype(np.float64)  # (200 energy bins, 200 det channels)
    return drm.T                         # (200 det channels, 200 energy bins)


def bin_drm(drm: np.ndarray, n: int) -> np.ndarray:
    """
    Average every (200/n) consecutive energy-bin columns.

    Parameters
    ----------
    drm : (200, 200) detector-channel × energy-bin matrix
    n   : target number of bins; must divide 200

    Returns
    -------
    (200, n) binned matrix
    """
    assert 200 % n == 0, f"n={n} must divide 200 evenly"
    cols_per_bin = 200 // n
    # reshape axis-1 from 200 → (n, cols_per_bin), then average the sub-columns
    return drm.reshape(200, n, cols_per_bin).mean(axis=2)


def mev_bin_centers(n: int) -> np.ndarray:
    """Center MeV value of each of the n bins spanning 0–50 MeV."""
    bin_width = 50.0 / n
    return (np.arange(n) + 0.5) * bin_width


def mev_bin_edges(n: int) -> np.ndarray:
    """Edge MeV values: n+1 points from 0 to 50 MeV."""
    return np.linspace(0.0, 50.0, n + 1)


def generate_synthetic_data(
    drm_binned: np.ndarray,
    samples_per_bin: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each energy-bin column of drm_binned, generate `samples_per_bin` noisy
    realizations using Poisson statistics: σ_i = √(I_i) per pixel.

    Parameters
    ----------
    drm_binned    : (200, n) binned DRM
    samples_per_bin : number of noisy samples to draw per energy bin
    rng           : numpy Generator (created with seed 42 if None)

    Returns
    -------
    X : (n * samples_per_bin, 200)  float32 — noisy detector responses
    y : (n * samples_per_bin,)       int32  — energy-bin class labels 0…n-1
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = drm_binned.shape[1]
    X_parts, y_parts = [], []

    for i in range(n):
        col = drm_binned[:, i]                          # (200,) clean response
        sigma = np.sqrt(np.maximum(col, 1e-8))          # √I noise std per pixel
        noise = rng.standard_normal((samples_per_bin, 200)) * sigma
        samples = np.clip(col + noise, 0.0, None).astype(np.float32)
        X_parts.append(samples)
        y_parts.append(np.full(samples_per_bin, i, dtype=np.int32))

    return np.vstack(X_parts), np.concatenate(y_parts)


def normalize_fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel (per detector pixel) mean and std from training data.

    Returns
    -------
    mean : (200,)
    std  : (200,)  (clipped to ≥ 1e-8 to avoid divide-by-zero)
    """
    mean = X_train.mean(axis=0)
    std  = np.maximum(X_train.std(axis=0), 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_apply(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed per-channel z-score normalization."""
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# PFF broad-spectrum generation
# ---------------------------------------------------------------------------

def pff_func(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate the PFF model: a1*exp(-a2*x) + a3*exp(-(x-a4)^2/a5)."""
    a1, a2, a3, a4, a5 = params
    return a1 * np.exp(-a2 * x) + a3 * np.exp(-(x - a4) ** 2 / a5)


def _sample_one_param(j: int, has_bump: bool, rng: np.random.Generator) -> float:
    """Scalar fallback used only by external callers; vectorised path is _sample_params_vec."""
    mu, sigma, lo, hi = PFF_PARAM_SAMPLING[j]
    if j == 2 and not has_bump:
        return 0.0
    lo_eff = 5.0 if (j == 2 and has_bump) else lo
    val = rng.normal(mu, sigma)
    while val < lo_eff or val > hi:
        val = rng.normal(mu, sigma)
    return float(val)


def _sample_params_vec(n: int, has_bump: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised bounded-normal sampling for n PFF parameter sets.

    Uses rejection sampling entirely in NumPy (no Python loop over samples).
    Returns (n, 5) array.
    """
    mu    = PFF_PARAM_SAMPLING[:, 0]   # (5,)
    sigma = PFF_PARAM_SAMPLING[:, 1]
    lo    = PFF_PARAM_SAMPLING[:, 2].copy()
    hi    = PFF_PARAM_SAMPLING[:, 3]

    if has_bump:
        lo[2] = 5.0   # visible bump: a3 >= 5

    out  = rng.normal(mu, sigma, size=(n, 5))   # (n, 5) initial draw
    mask = (out < lo) | (out > hi)              # (n, 5) invalid entries

    # Iteratively redraw invalid entries until all are in bounds.
    while mask.any():
        redraw = rng.normal(mu, sigma, size=(n, 5))
        out    = np.where(mask, redraw, out)
        mask   = (out < lo) | (out > hi)

    if not has_bump:
        out[:, 2] = 0.0   # force a3 = 0 for no-bump batch

    return out


def sample_pff_spectra(
    n_samples: int,
    energy_bins: np.ndarray,
    rng: np.random.Generator,
    bump_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample PFF spectra with a balanced bump / no-bump split.

    Vectorised — scales to millions of samples without a Python loop.

    Parameters
    ----------
    n_samples     : total number of spectra to generate
    energy_bins   : (M,) MeV values at which to evaluate the PFF
    rng           : numpy Generator
    bump_fraction : fraction of samples that have a Gaussian bump (a3 > 0)

    Returns
    -------
    spectra : (n_samples, M)  float64 — PFF spectra in energy space
    params  : (n_samples, 5)  float64 — [a1, a2, a3, a4, a5]
    """
    n_bump    = int(n_samples * bump_fraction)
    n_no_bump = n_samples - n_bump

    p_bump    = _sample_params_vec(n_bump,    True,  rng)  # (n_bump, 5)
    p_no_bump = _sample_params_vec(n_no_bump, False, rng)  # (n_no_bump, 5)
    params    = np.vstack([p_bump, p_no_bump])             # (n_samples, 5)

    # Vectorised PFF evaluation: (n, M)
    x  = energy_bins[np.newaxis, :]               # (1, M)
    a1, a2, a3, a4, a5 = (params[:, j, np.newaxis] for j in range(5))
    spectra = a1 * np.exp(-a2 * x) + a3 * np.exp(-(x - a4) ** 2 / a5)

    idx = rng.permutation(n_samples)
    return spectra[idx], params[idx]


def generate_pff_training_data(
    drm: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    bump_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (detector response, PFF params) training pairs.

    Pipeline:
      1. Sample PFF spectra in energy space
      2. drm @ spectrum  →  detector-channel response
      3. Add Poisson noise  (σ = √response per channel)

    Parameters
    ----------
    drm           : (200, 200) detector-channel × energy-bin matrix
    n_samples     : total samples to generate
    rng           : numpy Generator
    bump_fraction : fraction of samples that include a Gaussian bump

    Returns
    -------
    X      : (n_samples, 200) float32 — noisy detector responses
    params : (n_samples, 5)   float32 — PFF parameters [a1, a2, a3, a4, a5]
    """
    energy_bins = mev_bin_centers(drm.shape[1])
    spectra, params = sample_pff_spectra(n_samples, energy_bins, rng, bump_fraction)

    # DRM forward pass: (200, n) = (200, 200) @ (200, n)
    responses = (drm @ spectra.T).T              # (n_samples, 200)

    sigma = np.sqrt(np.maximum(responses, 1e-8))
    noise = rng.standard_normal(responses.shape) * sigma
    X = np.clip(responses + noise, 0.0, None)

    # L1-normalise each response so training and inference live on the same scale
    # regardless of the DRM's absolute units vs the real detector's raw units.
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)

    return X, params.astype(np.float32)


def normalize_pff_params(params: np.ndarray) -> np.ndarray:
    """Scale PFF parameters to [0, 1] using PFF_PARAM_BOUNDS."""
    lo = PFF_PARAM_BOUNDS[:, 0]
    hi = PFF_PARAM_BOUNDS[:, 1]
    return ((params - lo) / (hi - lo)).astype(np.float32)


def denormalize_pff_params(params_norm: np.ndarray) -> np.ndarray:
    """Invert normalize_pff_params back to physical units."""
    lo = PFF_PARAM_BOUNDS[:, 0]
    hi = PFF_PARAM_BOUNDS[:, 1]
    return (params_norm * (hi - lo) + lo).astype(np.float32)


def generate_spectrum_batch(
    drm_50: np.ndarray,
    n: int,
    rng: np.random.Generator,
    bump_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (detector_response, spectrum) pairs for 50-bin spectrum regression.

    Parameters
    ----------
    drm_50        : (200, 50) binned DRM — call bin_drm(drm, 50) before passing in
    n             : number of samples
    rng           : numpy Generator
    bump_fraction : fraction of samples that include a Gaussian bump

    Returns
    -------
    X : (n, 200) float32 — L1-normalised noisy detector responses
    y : (n, 50)  float32 — L1-normalised PFF spectra (targets)
    """
    energy_bins = mev_bin_centers(drm_50.shape[1])          # (50,) MeV centres
    spectra, _ = sample_pff_spectra(n, energy_bins, rng, bump_fraction)  # (n, 50)

    responses = (drm_50 @ spectra.T).T                      # (n, 200) detector space
    sigma = np.sqrt(np.maximum(responses, 1e-8))
    noise = rng.standard_normal(responses.shape) * sigma
    X = np.clip(responses + noise, 0.0, None)

    # L1-normalise detector responses to match inference scale
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)

    # L1-normalise spectra — targets are probability distributions over energy bins
    spec_sums = spectra.sum(axis=1, keepdims=True)
    y = (spectra / np.maximum(spec_sums, 1e-12)).astype(np.float32)

    return X, y
