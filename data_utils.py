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

# PFF form:  a1*exp(-a2*x) + a3*exp(-(x-a4)^2 / (a5*x + a6/x))
# 6-parameter form (was 5): the bump denominator a5 (a fixed width) became an
# energy-dependent width a5*x + a6/x -- a5 now dominates at high energy
# (width ~ a5*x), a6 dominates at low energy (width ~ a6/x), per explicit
# request. This changes what "a5" physically means (no longer a fixed width
# in the old [5,100] range) and introduces a6 with no prior calibration --
# the mean/std/lo/hi below are a placeholder chosen so that at a
# representative mid-range energy (x=20 MeV) the two terms contribute
# roughly equally and sum to the old typical width (~40): a5=1 -> a5*20=20,
# a6=400 -> a6/20=20. NOT independently validated against real shots or any
# physics reference -- flag for review if specific ranges are known.
# Columns: [mean, std, lo, hi] used for bounded-normal sampling.
# a3 (bump amplitude) bounds are for the bump-present case; set to 0 for no-bump.
PFF_PARAM_SAMPLING = np.array([
    [200.0, 200.0, 0.1,  500.0],   # a1 — bremsstrahlung amplitude
    [0.25,  0.2,   0.01,   1.0],   # a2 — bremsstrahlung decay rate (1/MeV)
    [8.0,   10.0,  5.0,  100.0],   # a3 — bump amplitude (bump-present lo = 5)
    [35.0,  25.0,  1.0,   49.0],   # a4 — bump centre (MeV)
    [1.0,   0.6,   0.1,    5.0],   # a5 — bump width, high-energy (a5*x) coefficient [PLACEHOLDER]
    [400.0, 250.0, 20.0, 1000.0],  # a6 — bump width, low-energy (a6/x) coefficient [PLACEHOLDER]
])
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
    [0.1,    5.0],   # a5 [PLACEHOLDER, see PFF_PARAM_SAMPLING]
    [20.0, 1000.0],  # a6 [PLACEHOLDER, see PFF_PARAM_SAMPLING]
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
SAT_FRACTION = 0.30
SAT_CEIL_LOW  = 0.55   # lowest ceiling = 55% of the sample's clean peak
SAT_CEIL_HIGH = 0.95   # highest ceiling = 95% of the sample's clean peak


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
    """Evaluate the PFF model: a1*exp(-a2*x) + a3*exp(-(x-a4)^2/(a5*x + a6/x))."""
    a1, a2, a3, a4, a5, a6 = params
    return a1 * np.exp(-a2 * x) + a3 * np.exp(-(x - a4) ** 2 / (a5 * x + a6 / x))


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
    n_params = PFF_PARAM_SAMPLING.shape[0]
    mu    = PFF_PARAM_SAMPLING[:, 0]   # (6,)
    sigma = PFF_PARAM_SAMPLING[:, 1]
    lo    = PFF_PARAM_SAMPLING[:, 2].copy()
    hi    = PFF_PARAM_SAMPLING[:, 3]

    if has_bump:
        lo[2] = 5.0   # visible bump: a3 >= 5

    out  = rng.normal(mu, sigma, size=(n, n_params))   # (n, 6) initial draw
    mask = (out < lo) | (out > hi)                     # (n, 6) invalid entries

    # Iteratively redraw invalid entries until all are in bounds.
    while mask.any():
        redraw = rng.normal(mu, sigma, size=(n, n_params))
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
    params  : (n_samples, 6)  float64 — [a1, a2, a3, a4, a5, a6]
    """
    n_bump    = int(n_samples * bump_fraction)
    n_no_bump = n_samples - n_bump

    p_bump    = _sample_params_vec(n_bump,    True,  rng)  # (n_bump, 6)
    p_no_bump = _sample_params_vec(n_no_bump, False, rng)  # (n_no_bump, 6)
    params    = np.vstack([p_bump, p_no_bump])             # (n_samples, 6)

    # Vectorised PFF evaluation: (n, M)
    x  = energy_bins[np.newaxis, :]               # (1, M)
    a1, a2, a3, a4, a5, a6 = (params[:, j, np.newaxis] for j in range(6))
    spectra = a1 * np.exp(-a2 * x) + a3 * np.exp(-(x - a4) ** 2 / (a5 * x + a6 / x))

    idx = rng.permutation(n_samples)
    return spectra[idx], params[idx]


MAX_BUMPS = 3   # multi-bump spectrum generation — see sample_multibump_spectra


def _sample_bump_params_vec(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised bounded-normal sampling for n (a3, a4, a5) bump triples, bump-present bounds.

    Explicitly sliced to rows 2:5 (a3, a4, a5), not a blanket [2:] -- this
    feeds sample_multibump_spectra's own simple constant-width bump term
    (a3*exp(-(x-a4)^2/a5)), a different, older 3-param bump form than
    pff_func's energy-dependent-width one, so it doesn't want a6 at all.
    """
    mu    = PFF_PARAM_SAMPLING[2:5, 0]   # (3,) a3, a4, a5 means
    sigma = PFF_PARAM_SAMPLING[2:5, 1]
    lo    = PFF_PARAM_SAMPLING[2:5, 2].copy()
    hi    = PFF_PARAM_SAMPLING[2:5, 3]
    lo[0] = 5.0   # a3 bump-present lower bound

    out  = rng.normal(mu, sigma, size=(n, 3))
    mask = (out < lo) | (out > hi)
    while mask.any():
        redraw = rng.normal(mu, sigma, size=(n, 3))
        out    = np.where(mask, redraw, out)
        mask   = (out < lo) | (out > hi)
    return out[:, 0], out[:, 1], out[:, 2]   # a3, a4, a5


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
    a1a2 = _sample_params_vec(n_samples, False, rng)   # (n_samples, 6); only cols 0,1 (a1,a2) used
    a1, a2 = a1a2[:, 0], a1a2[:, 1]
    spectra = a1[:, np.newaxis] * np.exp(-a2[:, np.newaxis] * x)   # (n_samples, M)

    has_bump = rng.random(n_samples) < bump_fraction
    n_bumps  = np.where(has_bump, rng.integers(1, max_bumps + 1, size=n_samples), 0)

    for slot in range(max_bumps):
        a3, a4, a5 = _sample_bump_params_vec(n_samples, rng)
        active = slot < n_bumps
        a3 = np.where(active, a3, 0.0)
        spectra += a3[:, np.newaxis] * np.exp(-(x - a4[:, np.newaxis]) ** 2 / a5[:, np.newaxis])

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


def pff_mean_sigma(pff_out: np.ndarray, param_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a heteroscedastic PFF-param head's raw output into physical-units
    (mean, sigma).

    pff_out is (mu_norm, logvar_norm) concatenated along the last axis, each
    width 5, with mu_norm in normalize_pff_params's [0,1]-ish space (see
    train_pff.py / train_cnn_multires.py's pff_nll_loss -- both heads use
    this same layout: ReLU mean + linear logvar). Denormalization is an
    affine map (param_bounds range * x + lo), so sigma -- unlike mu -- only
    picks up the multiplicative part; it has no lo offset to add.

    mu is clipped to param_bounds after denormalizing. ReLU only constrains
    the raw output to >=0 in normalized space, not <=1 -- it can extrapolate
    arbitrarily far past the training distribution's upper edge, which real
    (out-of-distribution) shots do: a4 (bump centre, physically must be
    within the 0-50 MeV detector range) has come back in the hundreds of MeV
    on some real shots, and a runaway a2 (decay rate) sends the reconstructed
    spectrum through dozens of orders of magnitude over 0-50 MeV -- neither
    is a physically meaningful fit. Clipping to param_bounds (the same
    bounds the training distribution itself was sampled from) keeps the
    reported/plotted fit inside the physically valid range; sigma is left
    unclipped since a mean pinned at the boundary with a wide sigma is
    itself informative (says the fit wanted to go further and is unsure).

    Works on a single sample, shape (12,) -> two (6,) arrays, or a batch,
    shape (N, 12) -> two (N, 6) arrays.
    """
    n_params = param_bounds.shape[0]
    mu_norm = pff_out[..., :n_params]
    logvar_norm = pff_out[..., n_params:]
    sigma_norm = np.exp(0.5 * np.clip(logvar_norm, -10.0, 10.0))
    mu_phys = denormalize_pff_params(mu_norm)
    mu_phys = np.clip(mu_phys, param_bounds[:, 0], param_bounds[:, 1])
    sigma_phys = sigma_norm * (param_bounds[:, 1] - param_bounds[:, 0])
    return mu_phys, sigma_phys


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
NOISE_GAIN = 5.0
READ_FRAC  = 0.10
MULT_FRAC  = 0.0


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
    bump_fraction: float = 0.5,
    sat_fraction: float = SAT_FRACTION,
    noise_gain: float = NOISE_GAIN,
    read_frac: float = READ_FRAC,
    mult_frac: float = MULT_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (detector_response, spectrum) pairs for spectrum regression.

    Spectra are drawn from sample_pff_spectra — the single bremsstrahlung
    exponential + one Gaussian bump PFF form (same form used for
    pff_infer_11733.png / train_pff.py's direct 5-parameter regressor).

    Parameters
    ----------
    drm_50        : (200, n_bins) binned DRM — call bin_drm(drm, n_bins) before passing in
    n             : number of samples
    rng           : numpy Generator
    bump_fraction : fraction of samples that include at least one Gaussian bump
    sat_fraction  : fraction of samples clipped to a saturation plateau (see
                    apply_saturation). Set 0.0 to reproduce the old smooth-only
                    behaviour.
    noise_gain / read_frac / mult_frac : detector-noise model knobs, see
                    add_detector_noise. Defaults (1.0, 0.0, 0.0) = pure Poisson.

    Returns
    -------
    X : (n, 200)     float32 — L1-normalised noisy detector responses
    y : (n, n_bins)  float32 — L1-normalised spectra (targets)
    """
    energy_bins = mev_bin_centers(drm_50.shape[1])          # (n_bins,) MeV centres
    spectra, _ = sample_pff_spectra(n, energy_bins, rng, bump_fraction)  # (n, n_bins)

    responses = (drm_50 @ spectra.T).T                      # (n, 200) detector space
    X = add_detector_noise(responses, rng, noise_gain, read_frac, mult_frac)

    # Clip a subset to a plateau, emulating CCD saturation, in raw-count space
    # (before L1 normalisation) so the model trains on near-saturated shapes.
    X, _ = apply_saturation(X, rng, sat_fraction)

    # L1-normalise detector responses to match inference scale
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)

    # L1-normalise spectra — targets are probability distributions over energy bins
    spec_sums = spectra.sum(axis=1, keepdims=True)
    y = (spectra / np.maximum(spec_sums, 1e-12)).astype(np.float32)

    return X, y


def generate_pff_training_data(
    drm: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    bump_fraction: float = 0.5,
    sat_fraction: float = SAT_FRACTION,
    noise_gain: float = NOISE_GAIN,
    read_frac: float = READ_FRAC,
    mult_frac: float = MULT_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (detector response, PFF params) training pairs.

    Pipeline:
      1. Sample PFF spectra in energy space
      2. drm @ spectrum  →  detector-channel response
      3. Add the calibrated detector noise model (add_detector_noise)
      4. Apply CCD saturation (apply_saturation)

    Uses the same noise + saturation model as generate_spectrum_batch, just
    above (the CNN spectrum classifier's generator) -- not plain
    sqrt(response) Poisson noise, as this function used to. That mismatch
    mattered: the PFF parameter regressors (train_pff.py,
    train_pff_bounded_gated.py) trained on the old plain-Poisson version
    were unfamiliar with saturated shapes every real shot actually has
    (0-92/200 channels saturation-corrected across the real shots checked
    this session), which was a large, concrete, and avoidable contributor to
    how far out-of-distribution real shots looked to those models -- see
    train_pff_v3_realistic_noise.py.

    Parameters
    ----------
    drm           : (200, 200) detector-channel × energy-bin matrix
    n_samples     : total samples to generate
    rng           : numpy Generator
    bump_fraction : fraction of samples that include a Gaussian bump
    sat_fraction  : fraction of samples clipped to a saturation plateau (see
                    apply_saturation). Set 0.0 to reproduce the old smooth-only
                    behaviour.
    noise_gain / read_frac / mult_frac : detector-noise model knobs, see
                    add_detector_noise.

    Returns
    -------
    X      : (n_samples, 200) float32 — noisy detector responses
    params : (n_samples, 6)   float32 — PFF parameters [a1, a2, a3, a4, a5, a6]
    """
    energy_bins = mev_bin_centers(drm.shape[1])
    spectra, params = sample_pff_spectra(n_samples, energy_bins, rng, bump_fraction)

    # DRM forward pass: (200, n) = (200, 200) @ (200, n)
    responses = (drm @ spectra.T).T              # (n_samples, 200)

    X = add_detector_noise(responses, rng, noise_gain, read_frac, mult_frac)
    X, _ = apply_saturation(X, rng, sat_fraction)

    # L1-normalise each response so training and inference live on the same scale
    # regardless of the DRM's absolute units vs the real detector's raw units.
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)

    return X, params.astype(np.float32)
