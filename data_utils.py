"""
data_utils.py — DRM loading, energy-bin downsampling, Poisson noise, normalization,
                and PFF-based broad-spectrum training data generation.

DRM orientation (from TSVD_NN.m: EDRM = x200'):
  xlsx rows = energy bins (200), xlsx cols = detector channels (200)
  After transpose: DRM rows = detector channels, cols = energy bins
"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize

# Six-parameter PFF form:
#   a1*exp(-a2*x) + a3*exp(-(x-a4)^2 / (a5*x + a6/x))
# Columns: [mean, std, lo, hi] used for bounded-normal sampling.
# a3 (bump amplitude) bounds are for the bump-present case; set to 0 for no-bump.
PFF_PARAM_SAMPLING = np.array([
    [150, 100, 50, 250],
    [0.3, 1, 0.01, 2],
    [20, 10, 0, 50],
    [20, 25, 1, 49],
    [0.5, 1, 1e-6, 10],
    [100, 500, 1e-6, 10000]
], dtype=np.float64)
# a2's hi was originally 5.0 /MeV, which after binning + L1-normalization
# collapses ~13% of "spectra" to a single near-delta bin (median a2 for
# collapsed samples was ~3.06 vs ~1.22 overall) — not a physical spectrum
# shape. Tightened std/hi so decay length always spans several MeV bins;
# verified this drops >90%-single-bin-mass samples to 0% at n_bins=50.

# Absolute bounds used for [0,1] normalization of parameters.
# No-bump samples have a3=0, which maps to 0.0 after normalization.
PFF_PARAM_BOUNDS = np.array([
    [0.1,  500.0],   # a1
    [0.01,   1.0],   # a2
    [0.0,  100.0],   # a3
    [1.0,   49.0],   # a4
    [1e-6,  10.0],   # a5
    [1e-6, 1.0e4],   # a6
])


def load_drm(xlsx_path: str) -> np.ndarray:
    """Load 200×200 xlsx and transpose so shape = (200 det channels, 200 energy bins)."""
    df = pd.read_excel(xlsx_path, header=None)
    drm = df.values.astype(np.float64)  # (200 energy bins, 200 det channels)
    return drm.T                         # (200 det channels, 200 energy bins)


def load_saturation_mask(corrected_csv_path: str) -> np.ndarray | None:
    """
    Identify which channels of a corrected shot CSV were altered by a
    saturation correction, by diffing against the sibling uncorrected
    *_proc_vector.csv. Handles both the legacy "*_proc_vector_corrected.csv"
    naming and the current "*_proc_vector_cv.csv" (vertical Gaussian
    correction, rescale_vector.ipynb) naming.

    Returns a (200,) bool array (True = value was imputed / corrected, so not
    a real measurement), or None if the uncorrected sibling file can't be
    found (e.g. csv_path isn't a recognised corrected-variant filename, or
    the shot doesn't have an uncorrected sibling on disk).
    """
    for suffix in ("_corrected", "_cv"):
        if suffix in corrected_csv_path:
            raw_path = corrected_csv_path.replace(suffix, "")
            break
    else:
        return None
    try:
        raw = pd.read_csv(raw_path)["signal"].values.astype(np.float32)
        corrected = pd.read_csv(corrected_csv_path)["signal"].values.astype(np.float32)
    except FileNotFoundError:
        return None
    return raw != corrected


# ---------------------------------------------------------------------------
# Saturation correction (ported from rescale_vector.ipynb)
# ---------------------------------------------------------------------------

SAT_LIMIT = 0.80   # anything over 80% of the 255 CCD range is over-saturated


def vertical_profile(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian lineout profile used to impute saturated pixels down a column."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def correct_saturation(
    shot_arr: np.ndarray,
    sat_limit: float = SAT_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace oversaturated CCD pixels with a Gaussian fit to the unsaturated
    part of their column (rescale_vector.ipynb's correction step).

    Parameters
    ----------
    shot_arr  : (H, W) processed shot image, values in [0, 255]
    sat_limit : fraction of 255 above which a pixel is considered saturated

    Returns
    -------
    corrected : (H, W) image, oversaturated columns replaced by their Gaussian fit
    saturated : (H, W) bool mask, True where the pixel was imputed (not real)
    """
    image_height, image_width = shot_arr.shape
    corrected = np.copy(shot_arr).astype(float)
    saturated = np.zeros_like(shot_arr, dtype=bool)

    threshold = sat_limit * 255
    oversaturated_cells = np.argwhere(shot_arr > threshold)
    if len(oversaturated_cells) == 0:
        return corrected, saturated

    cols_to_fix = np.unique(oversaturated_cells[:, 1])
    row_idx = np.arange(image_height)

    for c_idx in cols_to_fix:
        col = shot_arr[:, c_idx]
        good = col <= threshold
        center_guess = np.argmax(col)

        pguess, _ = optimize.curve_fit(
            vertical_profile, row_idx[good], col[good],
            p0=[255, center_guess, 5],
            bounds=([0, 0, 0], [np.inf, image_height, image_height]),
            maxfev=100000,
        )

        # the whole column is replaced by the fit (not just the over-threshold
        # cells), so every row in this column stops being a real measurement
        corrected[:, c_idx] = vertical_profile(row_idx, *pguess)
        saturated[:, c_idx] = True

    return corrected, saturated


def vectorize_shot_image(shot_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce a processed (H, W) shot image to the flat 200-channel detector
    vector used for inference, by averaging lineouts symmetric about the two
    brightest (adjacent) rows, following rescale_vector.ipynb.

    Returns
    -------
    vector       : (200,) float32
    central_rows : (2,) adjacent row indices picked as the beam center
    """
    central_rows = np.sort(np.argsort(np.average(shot_arr, axis=1))[-2:])
    assert central_rows[1] - central_rows[0] == 1, "Central rows not adjacent!"

    shot_half = np.array([
        np.average([shot_arr[central_rows[0] - i], shot_arr[central_rows[1] + i]], axis=0)
        for i in range(5)
    ])  # [s1, s2, s3, s4, s5]
    last_row = np.average(shot_half[-1].reshape(8, -1), axis=1)  # 8-averaged tail
    vector = np.concatenate((shot_half[:-1].flatten(), last_row)).astype(np.float32)
    return vector, central_rows


def vectorize_saturation_mask(saturated_2d: np.ndarray, central_rows: np.ndarray) -> np.ndarray:
    """Map a (H, W) per-pixel saturation mask through the same lineout-averaging
    used by vectorize_shot_image, so it lines up with the resulting 200-vector."""
    sat_half = np.array([
        np.any([saturated_2d[central_rows[0] - i], saturated_2d[central_rows[1] + i]], axis=0)
        for i in range(5)
    ])
    last_row = np.any(sat_half[-1].reshape(8, -1), axis=1)
    return np.concatenate((sat_half[:-1].flatten(), last_row))


def load_shot_vector(
    tif_path: str,
    correct: bool = True,
    sat_limit: float = SAT_LIMIT,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load a processed shot .tif and reduce it to the 200-channel detector
    vector, applying the Gaussian-imputation saturation correction inline.

    Returns
    -------
    vector   : (200,) float32
    sat_mask : (200,) bool, True where a channel was imputed (not a real
               measurement); None if correct=False
    """
    from PIL import Image

    img = Image.open(tif_path)
    shot_arr = np.asarray(img, dtype=float)
    img.close()

    if not correct:
        vector, _ = vectorize_shot_image(shot_arr)
        return vector, None

    corrected, saturated_2d = correct_saturation(shot_arr, sat_limit)
    vector, central_rows = vectorize_shot_image(corrected)
    sat_mask = (
        vectorize_saturation_mask(saturated_2d, central_rows)
        if saturated_2d.any() else np.zeros(200, dtype=bool)
    )
    return vector, sat_mask


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
    samples_per_bin: int = 10000,
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
# Detector saturation / clipping augmentation
# ---------------------------------------------------------------------------

# Fraction of synthetic training samples that get a saturation plateau, and the
# range of ceiling heights (as a fraction of each sample's own peak channel).
# Real CCD lineouts flat-top at the 8-bit ADC limit (~255): the brightest
# channels of bright shots (e.g. 11716, whose raw lineout peaks sit at ~231-243)
# clip to a plateau instead of following the smooth DRM response. Training on
# only smooth DRM@spectrum curves never shows the CNN that shape, so it fits
# clipped shots poorly. This adds that shape to a subset of samples.

SAT_FRACTION = 0.0
SAT_CEIL_LOW = 0.70
SAT_CEIL_HIGH = 0.98


def apply_saturation(
    X: np.ndarray,
    rng: np.random.Generator,
    sat_fraction: float = SAT_FRACTION,
    ceil_low: float = SAT_CEIL_LOW,
    ceil_high: float = SAT_CEIL_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip a random subset of samples to a per-sample plateau, emulating CCD
    saturation. For each selected sample the ceiling is drawn uniformly in
    [ceil_low, ceil_high] * (that sample's peak channel); every channel above
    the ceiling is flat-topped to it. Applied in raw detector-count space
    (before L1 normalization), so the plateau survives the later rescale.

    Parameters
    ----------
    X            : (n, C) non-negative detector responses (pre-L1-normalisation)
    rng          : numpy Generator
    sat_fraction : fraction of samples to saturate (0 disables, returns X as-is)
    ceil_low/high: ceiling range as a fraction of each sample's own peak

    Returns
    -------
    X_sat    : (n, C) responses with the selected samples clipped
    sat_mask : (n, C) bool, True where a value was clipped (imputed plateau)
    """
    n = X.shape[0]
    sat_mask = np.zeros_like(X, dtype=bool)
    if sat_fraction <= 0.0 or n == 0:
        return X, sat_mask

    X = X.copy()
    selected = rng.random(n) < sat_fraction               # (n,) bool
    peak = X.max(axis=1, keepdims=True)                    # (n, 1)
    frac = rng.uniform(ceil_low, ceil_high, size=(n, 1))   # (n, 1)
    # non-selected rows get an infinite ceiling => untouched
    ceiling = np.where(selected[:, None], frac * peak, np.inf)
    sat_mask = X > ceiling
    X = np.minimum(X, ceiling)
    return X.astype(np.float32), sat_mask


# ---------------------------------------------------------------------------
# PFF broad-spectrum generation
# ---------------------------------------------------------------------------

def pff_func(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the six-parameter PFF model:

        a1*exp(-a2*x)
        + a3*exp(-(x-a4)^2 / (a5*x + a6/x))
    """
    a1, a2, a3, a4, a5, a6 = params
    x_safe = np.maximum(np.asarray(x, dtype=np.float64), 1e-12)
    denominator = a5 * x_safe + a6 / x_safe
    denominator = np.maximum(denominator, 1e-12)
    return (
        a1 * np.exp(-a2 * x_safe)
        + a3 * np.exp(-((x_safe - a4) ** 2) / denominator)
    )


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
    Returns (n, 6) array.
    """
    mu    = PFF_PARAM_SAMPLING[:, 0]   # (6,)
    sigma = PFF_PARAM_SAMPLING[:, 1]
    lo    = PFF_PARAM_SAMPLING[:, 2].copy()
    hi    = PFF_PARAM_SAMPLING[:, 3]

    if has_bump:
        lo[2] = 5.0   # visible bump: a3 >= 5

    out  = rng.normal(mu, sigma, size=(n, 6))   # (n, 6) initial draw
    mask = (out < lo) | (out > hi)              # (n, 6) invalid entries

    # Iteratively redraw invalid entries until all are in bounds.
    while mask.any():
        redraw = rng.normal(mu, sigma, size=(n, 6))
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

    # Vectorised six-parameter PFF evaluation: (n, M)
    x = np.maximum(energy_bins[np.newaxis, :].astype(np.float64), 1e-12)
    a1, a2, a3, a4, a5, a6 = (
        params[:, j, np.newaxis] for j in range(6)
    )
    denominator = a5 * x + a6 / x
    denominator = np.maximum(denominator, 1e-12)
    spectra = (
        a1 * np.exp(-a2 * x)
        + a3 * np.exp(-((x - a4) ** 2) / denominator)
    )

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
    params : (n_samples, 6)   float32 — PFF parameters [a1, a2, a3, a4, a5, a6]
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


MAX_BUMPS = 3   # multi-bump spectrum generation — see sample_multibump_spectra


def _sample_bump_params_vec(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample bounded (a3, a4, a5, a6) values for one bump."""
    mu = PFF_PARAM_SAMPLING[2:, 0]
    sigma = PFF_PARAM_SAMPLING[2:, 1]
    lo = PFF_PARAM_SAMPLING[2:, 2].copy()
    hi = PFF_PARAM_SAMPLING[2:, 3]

    out = rng.normal(mu, sigma, size=(n, 4))
    mask = (out < lo) | (out > hi)

    while mask.any():
        redraw = rng.normal(mu, sigma, size=(n, 4))
        out = np.where(mask, redraw, out)
        mask = (out < lo) | (out > hi)

    # Preserve the file's current uniform bump-center behavior.
    out[:, 1] = rng.uniform(2.0, 48.0, size=n)

    return (
        out[:, 0],  # a3
        out[:, 1],  # a4
        out[:, 2],  # a5
        out[:, 3],  # a6
    )


def sample_multibump_spectra(
    n_samples: int,
    energy_bins: np.ndarray,
    rng: np.random.Generator,
    bump_fraction: float = 0.5,
    max_bumps: int = MAX_BUMPS,
) -> np.ndarray:
    """
    Sample PFF-family spectra generalised to a random number (0..max_bumps) of
    independent Gaussian bumps on top of the bremsstrahlung term, instead of
    always exactly 0 or 1.

    Real detector shots can show multi-humped structure that the single-bump
    PFF form (sample_pff_spectra) cannot represent — this widens the training
    distribution's shape diversity to cover that case. Used only by
    generate_spectrum_batch (histogram/softmax regression); the direct
    5-parameter regressor in train_pff.py is fixed at one bump by its output
    shape and keeps using sample_pff_spectra.

    Parameters
    ----------
    n_samples     : total number of spectra to generate
    energy_bins   : (M,) MeV values at which to evaluate the spectrum
    rng           : numpy Generator
    bump_fraction : fraction of samples that have at least one bump
    max_bumps     : upper bound on bumps per sample when bumps are present

    Returns
    -------
    spectra : (n_samples, M) float64
    """
    x = energy_bins[np.newaxis, :]   # (1, M)

    # Bremsstrahlung term (a3 forced to 0 here; bumps handled separately below).
    a1a2 = _sample_params_vec(n_samples, False, rng)   # (n_samples, 5)
    a1, a2 = a1a2[:, 0], a1a2[:, 1]
    spectra = a1[:, np.newaxis] * np.exp(-a2[:, np.newaxis] * x)   # (n_samples, M)

    has_bump = rng.random(n_samples) < bump_fraction
    n_bumps  = np.where(has_bump, rng.integers(1, max_bumps + 1, size=n_samples), 0)

    for slot in range(max_bumps):
        a3, a4, a5, a6 = _sample_bump_params_vec(n_samples, rng)
        active = slot < n_bumps
        a3 = np.where(active, a3, 0.0)

        x_safe = np.maximum(x, 1e-12)
        denominator = (
            a5[:, np.newaxis] * x_safe
            + a6[:, np.newaxis] / x_safe
        )
        denominator = np.maximum(denominator, 1e-12)

        spectra += (
            a3[:, np.newaxis]
            * np.exp(-((x_safe - a4[:, np.newaxis]) ** 2) / denominator)
        )

    return spectra


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


# ---------------------------------------------------------------------------
# Detector noise model
# ---------------------------------------------------------------------------

# The original generator used pure Poisson shot noise (sigma = sqrt(response)),
# which has no free scale and drops to ~0 in low-signal channels. Real detector
# noise usually also has a read/dark floor and can be over- or under-dispersed
# relative to sqrt(N). These knobs let a sweep find the level that best matches
# the real shots. gain=1.0/read=0.0/mult=0.0 reproduces the original pure-Poisson
# behaviour; the defaults below are the winner of NOISE_SEARCH_PLAN.md's search
# (grid -> refine -> 3-seed verification against the NNLS floor on real shots).
#
#   sigma_i = sqrt( (READ_FRAC * peak)^2  +  NOISE_GAIN * response_i
#                    + (MULT_FRAC * response_i)^2 )
#
#   NOISE_GAIN : shot-noise scale (variance = gain * signal). 1.0 = pure Poisson,
#                >1 = noisier, <1 = cleaner. This is the primary "amount of noise".
#   READ_FRAC  : additive Gaussian floor as a fraction of each sample's peak
#                channel (models read/dark noise; keeps low-signal channels noisy).
#   MULT_FRAC  : signal-proportional noise (flat-field / gain non-uniformity).
NOISE_GAIN = 0.0
READ_FRAC = 0.00
MULT_FRAC = 0.0


def add_detector_noise(
    responses: np.ndarray,
    rng: np.random.Generator,
    noise_gain: float = NOISE_GAIN,
    read_frac: float = READ_FRAC,
    mult_frac: float = MULT_FRAC,
) -> np.ndarray:
    """
    Add the parameterized detector noise model to clean responses and clip to >=0.

    Parameters
    ----------
    responses  : (n, C) non-negative clean detector responses (DRM @ spectrum)
    rng        : numpy Generator
    noise_gain : shot-noise variance scale (1.0 = pure Poisson sqrt(N))
    read_frac  : additive floor sigma as a fraction of each sample's peak channel
    mult_frac  : signal-proportional noise fraction

    Returns
    -------
    (n, C) noisy responses, clipped at 0.
    """
    peak = responses.max(axis=1, keepdims=True)                       # (n, 1)
    var = (noise_gain * np.maximum(responses, 0.0)
           + (read_frac * peak) ** 2
           + (mult_frac * responses) ** 2)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    noisy = responses + rng.standard_normal(responses.shape) * sigma
    return np.clip(noisy, 0.0, None)


def generate_spectrum_batch(
    drm_50: np.ndarray,
    n: int,
    rng: np.random.Generator,
    bump_fraction: float = 0.80,
    sat_fraction: float = 0.10,
    noise_gain: float = 5.0,
    read_frac: float = 0.02,
    mult_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:

    energy_bins = mev_bin_centers(
        drm_50.shape[1]
    )

    # Exact six-parameter family: no bump when a3=0, otherwise one bump.
    spectra, _ = sample_pff_spectra(
        n_samples=n,
        energy_bins=energy_bins,
        rng=rng,
        bump_fraction=bump_fraction,
    )

    responses = (
        drm_50 @ spectra.T
    ).T

    X = add_detector_noise(
        responses,
        rng,
        noise_gain=noise_gain,
        read_frac=read_frac,
        mult_frac=mult_frac,
    )

    X, _ = apply_saturation(
        X,
        rng,
        sat_fraction=sat_fraction,
    )

    X = X / np.maximum(
        X.sum(axis=1, keepdims=True),
        1e-12,
    )

    y = spectra / np.maximum(
        spectra.sum(axis=1, keepdims=True),
        1e-12,
    )

    return (
        X.astype(np.float32),
        y.astype(np.float32),
    )