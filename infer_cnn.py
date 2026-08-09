"""
Run the trained CNN (5x48 input) on real shot vectors and compare against the
dense model_spectrum_n{n} baseline.

Usage
-----
  python infer_cnn.py [shot_path ...]
  CNN_NBINS=20 python infer_cnn.py [shot_path ...]
  CNN_MODEL_PREFIX=model_cnn_multibump CNN_RESULTS_JSON=cnn_training_results_multibump.json \
      CNN_NBINS=50 python infer_cnn.py [shot_path ...]

shot_path may be a raw *_proc.tif image (saturation-corrected inline) or a
pre-baked *_proc_vector[_corrected].csv (legacy path).

Defaults to the raw processed .tif images for shots 10084 and 11733; the
Gaussian-imputation saturation correction from rescale_vector.ipynb is applied
inline (see data_utils.load_shot_vector) rather than requiring pre-baked
*_proc_vector_corrected.csv files.
Env vars:
  CNN_NBINS         bin count to evaluate (default 10)
  CNN_MODEL_PREFIX  model filename prefix, loads {prefix}_n{N}.keras (default model_cnn)
  CNN_RESULTS_JSON  results file holding norm stats, keyed by str(N) (default cnn_training_results.json)
  CNN_OUT_TAG       suffix inserted into output filenames, e.g. "_multibump" (default "")
  PFF_VERSION       "v1" (default, model_pff_relu_uncertainty.keras, train_pff.py) or
                     "v2" (model_pff_v2.keras, train_pff_bounded_gated.py -- tanh-bounded
                     logvar + gated bump classifier, see that file's module docstring)
The dense model_spectrum_n{N} baseline is included only if a matching
model file and spectrum_training_results.json[str(N)] entry both exist;
otherwise the CNN is plotted alone.

Also runs a standalone PFF-parameter regressor on the same real signal and
adds it as a 4th panel: the assumed bremsstrahlung + Gaussian-bump physical
family (an assumed theory, not a measurement -- see train_pff.py's module
docstring -- so it could be wrong or incomplete for shots that don't
actually follow it), fit with a ReLU-constrained (nonnegative) mean and a
heteroscedastic-NLL uncertainty, shown as a +/-1-sigma band rather than just
a point estimate. PFF_VERSION selects which of the two regressors built this
session provides that fit (see decode_pff_output). Real shots have no
ground-truth PFF params to check the band against, so
evaluate_pff_synthetic_calibration() (run once at the end of __main__, not
per-shot) checks it against synthetic samples with known true params
instead -- predicted vs. true, with the difference.

Outputs
-------
  cnn_infer_<shot>_n<n_bins>.png : 4-panel figure per shot
      1st — real signal vs DRM-forward of CNN and dense predictions
      2nd — CNN vs dense predicted spectra (bar)
      3rd — channel-wise residuals for both models
      4th — PFF-function fit to the real signal (mean +/- 1-sigma band)
  cnn_infer_<shot>_n<n_bins>_unsat.png : 3-panel figure per shot (only when a
      saturation mask is available for the shot) — same detector-response,
      spectrum, and residual comparison, but with saturation-corrected
      (imputed) channels masked out, so the fit quality on real measurements
      is visible without the imputed-channel distortion. (The PFF panel is
      channel-mask-independent -- it's a single global parametric fit, not
      per-channel -- so it isn't duplicated here.)
  cnn_pff_synth_check.png : predicted-vs-true PFF params on synthetic data,
      from evaluate_pff_synthetic_calibration().
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import (
    bin_drm,
    generate_pff_training_data,
    load_drm,
    load_saturation_mask,
    load_shot_vector,
    mev_bin_centers,
    mev_bin_edges,
    normalize_apply,
    pff_func,
    pff_mean_sigma,
)
from train_cnn import reshape_to_2d

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DRM_PATH       = "res/drm/200x200.xlsx"
CNN_JSON       = os.environ.get("CNN_RESULTS_JSON", "cnn_training_results.json")
DENSE_JSON     = "spectrum_training_results.json"
N_BINS         = int(os.environ.get("CNN_NBINS", 10))
MODEL_PREFIX   = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
OUT_TAG        = os.environ.get("CNN_OUT_TAG", "")

# Three PFF regressors built this session, distinct from the original sigmoid
# model_pff.keras (restored from git HEAD; incompatible with all of these --
# see decode_pff_output):
#   v1 -- model_pff_relu_uncertainty.keras (train_pff.py): ReLU mean + linear
#         logvar with a hard +-10 clip, single Gaussian NLL over all 5 params.
#   v2 -- model_pff_v2.keras (train_pff_bounded_gated.py): ReLU mean +
#         tanh-bounded logvar, a3 split into an explicit bump classifier +
#         magnitude-given-bump. See that file's module docstring for why.
#   v3 -- model_pff_v3.keras (train_pff_v3_realistic_noise.py): same
#         architecture as v2, trained on generate_pff_training_data's
#         calibrated-noise + CCD-saturation generator instead of v1/v2's
#         plain-Poisson one.
PFF_VERSION = os.environ.get("PFF_VERSION", "v1")
if PFF_VERSION == "v3":
    PFF_MODEL_PATH = "model_pff_v3.keras"
    PFF_JSON       = "pff_training_results_v3.json"
elif PFF_VERSION == "v2":
    PFF_MODEL_PATH = "model_pff_v2.keras"
    PFF_JSON       = "pff_training_results_v2.json"
else:
    PFF_MODEL_PATH = "model_pff_relu_uncertainty.keras"
    PFF_JSON       = "pff_training_results_relu_uncertainty.json"
PARAM_NAMES    = ["a1", "a2", "a3", "a4", "a5"]
N_MC_DRAWS     = 500     # Monte Carlo draws for the PFF spectrum uncertainty band
MC_SEED        = 7
N_SYNTH_SAMPLES = 12     # synthetic samples for the true-vs-predicted PFF calibration check
SYNTH_SEED      = 123

DEFAULT_SHOTS = [
    "res/test_images/10084/10084_proc_vector_cv.csv",
    "res/test_images/11733/11733_proc_vector_cv.csv",
]


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def get_shot_vector(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load a shot's 200-channel vector plus its saturation mask.

    .tif inputs get the Gaussian-imputation saturation correction applied
    inline (data_utils.load_shot_vector). .csv inputs are treated as
    already-corrected vectors, with the mask recovered by diffing against
    the uncorrected sibling file (legacy path, kept for pre-baked vectors).
    """
    if path.lower().endswith((".tif", ".tiff")):
        return load_shot_vector(path)
    signal = load_signal(path)
    sat_mask = load_saturation_mask(path)
    return signal, sat_mask


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def predict(model, x_norm: np.ndarray, as_2d: bool) -> np.ndarray:
    if as_2d:
        x_norm = reshape_to_2d(x_norm)
    return model.predict(x_norm, verbose=0)[0]


def decode_pff_output(pff_out: np.ndarray, param_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Dispatch to the right decode for whichever PFF regressor PFF_VERSION
    selects. v1's output (10-wide: mu+logvar per param) and v2/v3's
    (11-wide: gated bump classifier + magnitude-given-bump -- v3 shares v2's
    architecture, just a different training-data generator) are structurally
    different -- see train_pff.py / train_pff_bounded_gated.py.

    Returns (mu_phys, sigma_phys, p_bump): p_bump is None for v1 (no
    explicit bump classifier there; bump presence is read off a3's own
    magnitude instead, same as train_pff.py's convention).
    """
    if PFF_VERSION in ("v2", "v3"):
        from train_pff_bounded_gated import decode_v2
        mu, sigma, p_bump = decode_v2(pff_out[np.newaxis, :], param_bounds)
        return mu[0], sigma[0], float(p_bump[0, 0])
    return *pff_mean_sigma(pff_out, param_bounds), None


def pff_fit_for_signal(
    signal_l1: np.ndarray, pff_model, pff_mean: np.ndarray, pff_std: np.ndarray,
    param_bounds: np.ndarray, energy_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """
    Run the PFF regressor (PFF_VERSION selects v1 or v2) on a real signal and
    Monte Carlo the +/-1-sigma spectrum band from its predicted per-parameter
    uncertainty.

    Draws N_MC_DRAWS param sets independently from each parameter's predicted
    N(mean, sigma), clipped to param_bounds (PFF params can't be negative,
    and e.g. a4 -- bump centre -- can't physically exceed the 0-50 MeV
    detector range even though sigma alone is unbounded), evaluates pff_func
    per draw, and takes pointwise 16th/84th percentiles. This propagates the
    head's uncertainty into spectrum space -- a residual alone says nothing
    about how much to trust the fit away from the measured channels.

    Returns (p_phys, p_sigma, band_lo, band_hi, p_bump): p_phys/p_sigma are
    (5,) physical-units mean/sigma; band_lo/band_hi are (len(energy_bins),)
    -- the caller recomputes the mean curve itself via
    pff_func(energy_bins, p_phys); p_bump is v2-only (see decode_pff_output).
    """
    x_norm = normalize_apply(signal_l1.reshape(1, -1), pff_mean, pff_std)
    pff_out = pff_model.predict(x_norm, verbose=0)[0]
    p_phys, p_sigma, p_bump = decode_pff_output(pff_out, param_bounds)

    mc_rng = np.random.default_rng(MC_SEED)
    draws = mc_rng.normal(p_phys, p_sigma, size=(N_MC_DRAWS, 5))
    draws = np.clip(draws, param_bounds[:, 0], param_bounds[:, 1])
    mc_spectra = np.stack([pff_func(energy_bins, draws[i]) for i in range(N_MC_DRAWS)])
    band_lo = np.percentile(mc_spectra, 16, axis=0)
    band_hi = np.percentile(mc_spectra, 84, axis=0)
    return p_phys, p_sigma, band_lo, band_hi, p_bump


def run_shot(shot_path: str, drm: np.ndarray, cnn_model, cnn_res: dict,
             dense_model=None, dense_res: dict | None = None,
             pff_model=None, pff_res: dict | None = None) -> None:
    shot_name = os.path.basename(shot_path).split("_")[0]
    signal, sat_mask = get_shot_vector(shot_path)   # sat_mask True = imputed/not a real measurement
    signal_l1 = l1_normalise(signal)
    drm_binned = bin_drm(drm, N_BINS)
    energy_bins = mev_bin_centers(N_BINS)
    channels = np.arange(1, 201)

    unsat = ~sat_mask if sat_mask is not None else np.ones(200, dtype=bool)

    models_to_run = [("CNN 5x48", cnn_model, cnn_res, True)]
    if dense_model is not None:
        models_to_run.append(("dense", dense_model, dense_res, False))

    preds = {}
    for label, model, res, as_2d in models_to_run:
        mean = np.array(res["norm_mean"], dtype=np.float32)
        std  = np.array(res["norm_std"],  dtype=np.float32)
        x_norm = normalize_apply(signal_l1.reshape(1, -1), mean, std)
        spec = predict(model, x_norm, as_2d)
        resp_l1 = l1_normalise(drm_binned @ spec)
        residual = signal_l1 - resp_l1
        rms = float(np.sqrt(np.mean(residual ** 2)))
        rms_unsat = float(np.sqrt(np.mean(residual[unsat] ** 2)))
        preds[label] = dict(spec=spec, resp=resp_l1, resid=residual, rms=rms, rms_unsat=rms_unsat)

    signal_rms = float(np.sqrt(np.mean(signal_l1 ** 2)))
    signal_rms_unsat = float(np.sqrt(np.mean(signal_l1[unsat] ** 2)))

    print(f"\n=== Shot {shot_name}  (n_bins={N_BINS}) ===")
    if sat_mask is not None:
        print(f"  {sat_mask.sum()}/200 channels are saturation-corrected (imputed, not real measurements)")
    for label, p in preds.items():
        peak = energy_bins[p["spec"].argmax()]
        print(f"  {label:9s}: peak {peak:4.0f} MeV ({p['spec'].max()*100:5.2f}%)  "
              f"resid RMS {p['rms']:.6f} ({100*p['rms']/signal_rms:.1f}% of signal, all channels)  |  "
              f"unsaturated-only {p['rms_unsat']:.6f} ({100*p['rms_unsat']/signal_rms_unsat:.1f}% of signal)")

    has_pff = pff_model is not None
    if has_pff:
        pff_energy_bins = np.linspace(0.5, 49.5, 200)   # fine grid for a smooth PFF curve
        pff_mean_stat = np.array(pff_res["norm_mean"], dtype=np.float32)
        pff_std_stat  = np.array(pff_res["norm_std"],  dtype=np.float32)
        pff_bounds    = np.array(pff_res["param_bounds"], dtype=np.float32)
        p_phys, p_sigma, pff_band_lo, pff_band_hi, p_bump = pff_fit_for_signal(
            signal_l1, pff_model, pff_mean_stat, pff_std_stat, pff_bounds, pff_energy_bins,
        )
        pff_spec = pff_func(pff_energy_bins, p_phys)
        # v2 has an explicit bump-presence classifier (p_bump); v1 doesn't, so
        # bump presence is read off a3's own magnitude instead (same convention
        # train_pff.py / infer_pff.py already use).
        bump_present = (p_bump > 0.5) if p_bump is not None else (p_phys[2] > 5.0)
        at_bound = (p_phys <= pff_bounds[:, 0] + 1e-3) | (p_phys >= pff_bounds[:, 1] - 1e-3)
        print(f"  PFF fit [{PFF_VERSION}]: " + " | ".join(
            f"{n}={p_phys[j]:.2f}+/-{p_sigma[j]:.2f}{'*' if at_bound[j] else ''}"
            for j, n in enumerate(PARAM_NAMES)
        ))
        if p_bump is not None:
            print(f"             p(bump) = {p_bump:.3f}")
        if at_bound.any():
            print(f"             * = clipped to the physical training bound "
                  f"({', '.join(PARAM_NAMES[j] for j in np.flatnonzero(at_bound))}); "
                  f"raw prediction extrapolated past it -- fit is unreliable there")
        print(f"             bump {'DETECTED' if bump_present else 'NOT detected'}"
              + (f"  (centre {p_phys[3]:.1f}+/-{p_sigma[3]:.1f} MeV)" if bump_present else ""))

    # --- figure ---
    n_panels = 4 if has_pff else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), tight_layout=True)
    fig.suptitle(f"CNN vs dense spectrum inference — shot {shot_name}  (n_bins={N_BINS})",
                 fontsize=12)

    has_dense = "dense" in preds

    ax = axes[0]
    ax.plot(channels, signal_l1, "k", lw=1.4, label="real signal (L1)")
    ax.plot(channels, preds["CNN 5x48"]["resp"], "r--", lw=1.2,
            label=f"DRM x CNN pred ({100*preds['CNN 5x48']['rms']/signal_rms:.1f}%)")
    if has_dense:
        ax.plot(channels, preds["dense"]["resp"], "b:", lw=1.2,
                label=f"DRM x dense pred ({100*preds['dense']['rms']/signal_rms:.1f}%)")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Intensity (arb.)")
    ax.legend()
    ax.set_title("Detector response: real vs reconstructed")

    ax = axes[1]
    edges = mev_bin_edges(N_BINS)
    ax.stairs(preds["CNN 5x48"]["spec"], edges, color="firebrick", lw=1.8,
              label="CNN 5x48")
    ax.plot(energy_bins, preds["CNN 5x48"]["spec"], "o", color="firebrick", ms=4)
    if has_dense:
        ax.stairs(preds["dense"]["spec"], edges, color="steelblue", lw=1.8,
                  label="dense (model_spectrum)")
        ax.plot(energy_bins, preds["dense"]["spec"], "s", color="steelblue", ms=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Intensity (arb., log)")
    ax.legend()
    ax.set_title("Predicted energy spectrum over full 0-50 MeV range")

    ax = axes[2]
    if sat_mask is not None and sat_mask.any():
        # shade contiguous saturation-corrected (imputed) channel spans
        edges_idx = np.flatnonzero(np.diff(np.r_[0, sat_mask.astype(int), 0]))
        spans = edges_idx.reshape(-1, 2)
        for k, (lo, hi) in enumerate(spans):
            ax.axvspan(channels[lo], channels[hi - 1], color="gray", alpha=0.2,
                       label="saturation-corrected" if k == 0 else None)
    ax.plot(channels, preds["CNN 5x48"]["resid"], "r", lw=1, label="CNN residual")
    if has_dense:
        ax.plot(channels, preds["dense"]["resid"], "b", lw=1, alpha=0.7, label="dense residual")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Residual (real L1 - pred L1)")
    ax.legend()
    ax.set_title("Channel residuals  (shaded = saturation-corrected, not real measurements)")

    if has_pff:
        ax = axes[3]
        ax.fill_between(pff_energy_bins, pff_band_lo, pff_band_hi, color="darkorange",
                         alpha=0.25, label="+/-1sig band (param uncertainty)")
        ax.plot(pff_energy_bins, pff_spec, color="darkorange", lw=1.8, label="PFF fit (mean)")
        if bump_present:
            ax.axvline(p_phys[3], color="darkorange", ls=":", lw=1.2,
                       label=f"bump centre {p_phys[3]:.1f}+/-{p_sigma[3]:.1f} MeV")
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Intensity (arb., log)")
        ax.legend(fontsize=8)
        param_str = "  ".join(f"{n}={p_phys[j]:.2f}±{p_sigma[j]:.2f}" for j, n in enumerate(PARAM_NAMES))
        bump_str = f"  p(bump)={p_bump:.2f}" if p_bump is not None else ""
        ax.set_title(f"PFF-function fit [{PFF_VERSION}] (assumed brems+bump physics, not a measurement)\n"
                     f"{param_str}{bump_str}", fontsize=9)

    out_path = f"cnn_infer_{shot_name}_n{N_BINS}{OUT_TAG}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")

    # --- second figure: same two panels, saturation-corrected channels masked out ---
    if sat_mask is not None and sat_mask.any():
        def masked(arr: np.ndarray) -> np.ndarray:
            out = arr.astype(np.float32).copy()
            out[sat_mask] = np.nan
            return out

        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9.5), tight_layout=True)
        fig2.suptitle(f"CNN vs dense — shot {shot_name}  (n_bins={N_BINS}, unsaturated channels only)",
                      fontsize=12)

        ax = axes2[0]
        ax.plot(channels, masked(signal_l1), "k", lw=1.4, label="real signal (L1)")
        ax.plot(channels, masked(preds["CNN 5x48"]["resp"]), "r--", lw=1.2,
                label=f"DRM x CNN pred ({100*preds['CNN 5x48']['rms_unsat']/signal_rms_unsat:.1f}%)")
        if has_dense:
            ax.plot(channels, masked(preds["dense"]["resp"]), "b:", lw=1.2,
                    label=f"DRM x dense pred ({100*preds['dense']['rms_unsat']/signal_rms_unsat:.1f}%)")
        ax.set_xlabel("Detector channel")
        ax.set_ylabel("Intensity (arb.)")
        ax.legend()
        ax.set_title("Detector response: real vs reconstructed (unsaturated channels only)")

        # Predicted spectrum is already in energy space, not channel space, so it's
        # identical to the main figure's middle panel — saturation masking doesn't apply.
        ax = axes2[1]
        ax.stairs(preds["CNN 5x48"]["spec"], edges, color="firebrick", lw=1.8,
                  label="CNN 5x48")
        ax.plot(energy_bins, preds["CNN 5x48"]["spec"], "o", color="firebrick", ms=4)
        if has_dense:
            ax.stairs(preds["dense"]["spec"], edges, color="steelblue", lw=1.8,
                      label="dense (model_spectrum)")
            ax.plot(energy_bins, preds["dense"]["spec"], "s", color="steelblue", ms=4)
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Intensity (arb., log)")
        ax.legend()
        ax.set_title("Predicted energy spectrum over full 0-50 MeV range")

        ax = axes2[2]
        ax.plot(channels, masked(preds["CNN 5x48"]["resid"]), "r", lw=1, label="CNN residual")
        if has_dense:
            ax.plot(channels, masked(preds["dense"]["resid"]), "b", lw=1, alpha=0.7, label="dense residual")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Detector channel")
        ax.set_ylabel("Residual (real L1 - pred L1)")
        ax.legend()
        ax.set_title(f"Channel residuals, unsaturated only  "
                     f"(CNN RMS={preds['CNN 5x48']['rms_unsat']:.6f}"
                     + (f", dense RMS={preds['dense']['rms_unsat']:.6f}" if has_dense else "") + ")")

        out_path_unsat = f"cnn_infer_{shot_name}_n{N_BINS}{OUT_TAG}_unsat.png"
        fig2.savefig(out_path_unsat, dpi=150)
        print(f"  Saved {out_path_unsat}")


def evaluate_pff_synthetic_calibration(drm: np.ndarray, pff_model, pff_res: dict) -> None:
    """
    Calibration check on synthetic data with known ground truth.

    Real shots (run_shot, above) have no true PFF params to compare a
    prediction against -- only self-consistency (does the reconstructed
    spectrum reproduce the measured detector response) is checkable there.
    Here, N_SYNTH_SAMPLES samples are drawn from the exact same generator
    train_pff.py trains on (data_utils.generate_pff_training_data: same PFF
    family, same Poisson-ish noise, same L1 normalization), so the true
    [a1..a5] is known and predicted vs. true can be compared directly, with
    the predicted 1-sigma band checked for whether it actually covers the
    true value at roughly the rate it should (~68%).
    """
    pff_mean_stat = np.array(pff_res["norm_mean"], dtype=np.float32)
    pff_std_stat  = np.array(pff_res["norm_std"],  dtype=np.float32)
    pff_bounds    = np.array(pff_res["param_bounds"], dtype=np.float32)

    rng = np.random.default_rng(SYNTH_SEED)
    X, params_true = generate_pff_training_data(drm, N_SYNTH_SAMPLES, rng)
    x_norm = normalize_apply(X, pff_mean_stat, pff_std_stat)
    pff_out = pff_model.predict(x_norm, verbose=0)
    if PFF_VERSION in ("v2", "v3"):
        from train_pff_bounded_gated import decode_v2
        mu_phys, sigma_phys, _p_bump = decode_v2(pff_out, pff_bounds)
    else:
        mu_phys, sigma_phys = pff_mean_sigma(pff_out, pff_bounds)

    diff = mu_phys - params_true
    z = diff / np.maximum(sigma_phys, 1e-6)

    print(f"\n=== Synthetic PFF calibration check ({N_SYNTH_SAMPLES} samples, "
          f"seed={SYNTH_SEED}) ===")
    for i in range(N_SYNTH_SAMPLES):
        row = f"  {i:>3}  "
        row += "  ".join(
            f"{PARAM_NAMES[j]}: true={params_true[i,j]:6.2f} pred={mu_phys[i,j]:6.2f}"
            f"+/-{sigma_phys[i,j]:4.2f} diff={diff[i,j]:+5.2f}"
            for j in range(5)
        )
        print(row)

    bump_mask = params_true[:, 2] > 0.0
    cov_all = np.mean(np.abs(z[:, :3]) <= 1.0)
    cov_bump = float(np.mean(np.abs(z[bump_mask][:, 3:]) <= 1.0)) if bump_mask.any() else float("nan")
    mae = {n: float(np.mean(np.abs(diff[:, j]))) for j, n in enumerate(PARAM_NAMES)}
    print(f"\n  MAE (true vs. predicted): " + " | ".join(f"{n}={mae[n]:.3f}" for n in PARAM_NAMES))
    print(f"  1-sigma coverage: a1/a2/a3={cov_all*100:.0f}% (want ~68%)  "
          f"a4/a5(bump samples only, n={int(bump_mask.sum())})={cov_bump*100:.0f}%")

    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)
    lo, hi = pff_bounds[:, 0], pff_bounds[:, 1]
    true_norm = (params_true - lo) / (hi - lo)
    mu_norm = pff_out[:, :5]
    sigma_norm = sigma_phys / (hi - lo)
    for j, n in enumerate(PARAM_NAMES):
        ax.errorbar(true_norm[:, j], mu_norm[:, j], yerr=sigma_norm[:, j],
                    fmt="o", ms=5, capsize=3, alpha=0.8, label=n)
    lims = [min(0.0, true_norm.min(), mu_norm.min()) - 0.05,
            max(1.0, true_norm.max(), mu_norm.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1, label="perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True param (normalized)")
    ax.set_ylabel("Predicted param (normalized, mean ± 1σ)")
    ax.set_title(f"PFF param calibration — {N_SYNTH_SAMPLES} synthetic samples")
    ax.legend(fontsize=8)

    out_path = f"cnn_pff_synth_check_{PFF_VERSION}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    shot_paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SHOTS

    drm = load_drm(DRM_PATH)

    with open(CNN_JSON) as f:
        cnn_res = json.load(f)[str(N_BINS)]

    cnn_model = tf.keras.models.load_model(f"{MODEL_PREFIX}_n{N_BINS}.keras", compile=False)

    dense_model, dense_res = None, None
    try:
        with open(DENSE_JSON) as f:
            dense_res = json.load(f)[str(N_BINS)]
        dense_path = f"model_spectrum_n{N_BINS}.keras"
        try:
            dense_model = tf.keras.models.load_model(dense_path, compile=False)
        except TypeError:
            # saved with newer keras: rebuild architecture, load weights only
            from train_spectrum import build_model as build_dense
            dense_model = build_dense(N_BINS)
            dense_model.load_weights(dense_path)
    except (FileNotFoundError, KeyError):
        print(f"  (no dense baseline for n_bins={N_BINS} — plotting CNN alone)")
        dense_model, dense_res = None, None

    pff_model, pff_res = None, None
    try:
        with open(PFF_JSON) as f:
            pff_res = json.load(f)
        pff_model = tf.keras.models.load_model(PFF_MODEL_PATH, compile=False)
    except FileNotFoundError:
        print("  (no PFF regressor found — skipping PFF-fit panel)")
        pff_model, pff_res = None, None

    for shot_path in shot_paths:
        run_shot(shot_path, drm, cnn_model, cnn_res, dense_model, dense_res, pff_model, pff_res)

    if pff_model is not None:
        evaluate_pff_synthetic_calibration(drm, pff_model, pff_res)
