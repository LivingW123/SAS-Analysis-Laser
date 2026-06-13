"""
data_utils.py — DRM loading, energy-bin downsampling, Poisson noise, normalization.

DRM orientation (from TSVD_NN.m: EDRM = x200'):
  xlsx rows = energy bins (200), xlsx cols = detector channels (200)
  After transpose: DRM rows = detector channels, cols = energy bins
"""

import numpy as np
import pandas as pd


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
