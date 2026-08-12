"""
infer_cnn_ensemble.py — same CNN-vs-real-signal inference as infer_cnn.py,
but with the PFF-fit panel driven by the 5-member deep ensemble
(train_pff_ensemble_member.py, pff_ensemble_utils.py) instead of a single
PFF regressor.

Why a separate file rather than another infer_cnn.py PFF_VERSION branch: the
ensemble's prediction flow is structurally different (5 models to load and
run, each with its own norm stats, combined via decode_ensemble's law-of-
total-variance) rather than a drop-in decode swap the way v1/v2/v3 were --
same reasoning as why train_pff_v3_realistic_noise.py and
train_pff_bounded_gated.py are separate files rather than branches of one.

The PFF panel now shows THREE things a single model's panel couldn't:
mean +/- total sigma (as before), plus the aleatoric/epistemic split
(evaluate_pff_ensemble.py found this splits real shots' variance very
differently by parameter -- e.g. a1/a4 often >50% epistemic on real shots,
a3/a5 usually <10%, see that script's docstring), plus p(bump) mean +/- its
own cross-member std.

Saves into its own output directory (OUT_DIR, default "cnn_infer_ensemble/")
so these don't collide with infer_cnn.py's canonical single-model figures.

Usage
-----
  python infer_cnn_ensemble.py [shot_path ...]
  CNN_NBINS=50 python infer_cnn_ensemble.py
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
    load_drm,
    load_saturation_mask,
    load_shot_vector,
    mev_bin_centers,
    mev_bin_edges,
    normalize_apply,
    pff_func,
)
from pff_ensemble_utils import decode_ensemble
from train_cnn import reshape_to_2d
from train_pff_bounded_gated import decode_v2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DRM_PATH     = "res/drm/200x200.xlsx"
CNN_JSON     = os.environ.get("CNN_RESULTS_JSON", "cnn_training_results.json")
DENSE_JSON   = "spectrum_training_results.json"
N_BINS       = int(os.environ.get("CNN_NBINS", 50))
MODEL_PREFIX = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
OUT_DIR      = os.environ.get("CNN_OUT_DIR", "cnn_infer_ensemble")

N_MEMBERS    = 5
PARAM_NAMES  = ["a1", "a2", "a3", "a4", "a5"]
N_MC_DRAWS   = 500
MC_SEED      = 7

# The 14 real shots used throughout this session (6 initial + 11698, cv+ch each).
DEFAULT_SHOTS = [
    f"res/test_images/{shot}/{shot}_proc_vector{suffix}.csv"
    for shot in ["10084", "11696", "11705", "11707", "11716", "11733", "11698"]
    for suffix in ("_cv", "_ch")
]


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def get_shot_vector(path: str) -> tuple[np.ndarray, np.ndarray | None]:
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


def load_ensemble() -> list[dict]:
    members = []
    for idx in range(N_MEMBERS):
        model_path = f"model_pff_ensemble_{idx}.keras"
        json_path = f"pff_training_results_ensemble_{idx}.json"
        with open(json_path) as f:
            res = json.load(f)
        members.append({
            "model": tf.keras.models.load_model(model_path, compile=False),
            "mean": np.array(res["norm_mean"], dtype=np.float32),
            "std": np.array(res["norm_std"], dtype=np.float32),
            "bounds": np.array(res["param_bounds"], dtype=np.float32),
        })
    return members


def pff_fit_for_signal_ensemble(
    signal_l1: np.ndarray, members: list[dict], energy_bins: np.ndarray,
) -> tuple:
    """
    Run all N_MEMBERS PFF regressors on a real signal, combine via
    decode_ensemble, and Monte Carlo the +/-1-sigma spectrum band from the
    combined TOTAL (aleatoric+epistemic) sigma -- see pff_ensemble_utils.

    Returns (mean, sigma_total, sigma_epistemic, p_bump_mean, p_bump_std,
    band_lo, band_hi).
    """
    bounds = members[0]["bounds"]
    mus, sigmas, p_bumps = [], [], []
    for mem in members:
        x_norm = normalize_apply(signal_l1.reshape(1, -1), mem["mean"], mem["std"])
        out = mem["model"].predict(x_norm, verbose=0)
        mu, sigma, p_bump = decode_v2(out, bounds)
        mus.append(mu[0])
        sigmas.append(sigma[0])
        p_bumps.append(p_bump[0])
    mus, sigmas, p_bumps = np.array(mus), np.array(sigmas), np.array(p_bumps)
    mean, sigma_total, sigma_epi, pb_mean, pb_std = decode_ensemble(mus, sigmas, p_bumps)

    mc_rng = np.random.default_rng(MC_SEED)
    draws = mc_rng.normal(mean, sigma_total, size=(N_MC_DRAWS, 5))
    draws = np.clip(draws, bounds[:, 0], bounds[:, 1])
    mc_spectra = np.stack([pff_func(energy_bins, draws[i]) for i in range(N_MC_DRAWS)])
    band_lo = np.percentile(mc_spectra, 16, axis=0)
    band_hi = np.percentile(mc_spectra, 84, axis=0)
    return mean, sigma_total, sigma_epi, float(pb_mean[0]), float(pb_std[0]), band_lo, band_hi


def run_shot(shot_path: str, drm: np.ndarray, cnn_model, cnn_res: dict,
             dense_model, dense_res: dict | None, members: list[dict]) -> None:
    basename = os.path.basename(shot_path)
    shot_name = basename.split("_")[0]
    # _cv/_ch (or any other variant suffix) must be preserved in the output
    # filename -- otherwise two variants of the same shot number collide on
    # one filename and the second silently overwrites the first.
    variant_tag = ""
    for suffix in ("_cv", "_ch"):
        if suffix in basename:
            variant_tag = suffix
            break
    signal, sat_mask = get_shot_vector(shot_path)
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
        mean_stat = np.array(res["norm_mean"], dtype=np.float32)
        std_stat  = np.array(res["norm_std"],  dtype=np.float32)
        x_norm = normalize_apply(signal_l1.reshape(1, -1), mean_stat, std_stat)
        spec = predict(model, x_norm, as_2d)
        resp_l1 = l1_normalise(drm_binned @ spec)
        residual = signal_l1 - resp_l1
        rms = float(np.sqrt(np.mean(residual ** 2)))
        rms_unsat = float(np.sqrt(np.mean(residual[unsat] ** 2)))
        preds[label] = dict(spec=spec, resp=resp_l1, resid=residual, rms=rms, rms_unsat=rms_unsat)

    signal_rms = float(np.sqrt(np.mean(signal_l1 ** 2)))
    has_dense = "dense" in preds

    print(f"\n=== Shot {shot_name}{variant_tag}  (n_bins={N_BINS}) ===")
    if sat_mask is not None:
        print(f"  {sat_mask.sum()}/200 channels are saturation-corrected (imputed, not real measurements)")
    for label, p in preds.items():
        peak = energy_bins[p["spec"].argmax()]
        print(f"  {label:9s}: peak {peak:4.0f} MeV ({p['spec'].max()*100:5.2f}%)  "
              f"resid RMS {p['rms']:.6f} ({100*p['rms']/signal_rms:.1f}% of signal)")

    pff_energy_bins = np.linspace(0.5, 49.5, 200)
    mean, sigma_total, sigma_epi, pb_mean, pb_std, band_lo, band_hi = pff_fit_for_signal_ensemble(
        signal_l1, members, pff_energy_bins,
    )
    pff_spec = pff_func(pff_energy_bins, mean)
    bump_present = pb_mean > 0.5
    bounds = members[0]["bounds"]
    at_bound = (mean <= bounds[:, 0] + 1e-3) | (mean >= bounds[:, 1] - 1e-3)

    epi_frac = 100 * sigma_epi ** 2 / np.maximum(sigma_total ** 2, 1e-9)
    print(f"  PFF fit [ensemble, {len(members)} members]: " + " | ".join(
        f"{n}={mean[j]:.2f}+/-{sigma_total[j]:.2f}{'*' if at_bound[j] else ''} "
        f"(epi {epi_frac[j]:.0f}%)"
        for j, n in enumerate(PARAM_NAMES)
    ))
    print(f"             p(bump) = {pb_mean:.3f} +/- {pb_std:.3f}")
    if at_bound.any():
        print(f"             * = clipped to the physical training bound "
              f"({', '.join(PARAM_NAMES[j] for j in np.flatnonzero(at_bound))})")
    print(f"             bump {'DETECTED' if bump_present else 'NOT detected'}"
          + (f"  (centre {mean[3]:.1f}+/-{sigma_total[3]:.1f} MeV)" if bump_present else ""))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), tight_layout=True)
    fig.suptitle(f"CNN vs dense spectrum inference — shot {shot_name}  (n_bins={N_BINS}, ensemble PFF)",
                 fontsize=12)

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
    ax.stairs(preds["CNN 5x48"]["spec"], edges, color="firebrick", lw=1.8, label="CNN 5x48")
    ax.plot(energy_bins, preds["CNN 5x48"]["spec"], "o", color="firebrick", ms=4)
    if has_dense:
        ax.stairs(preds["dense"]["spec"], edges, color="steelblue", lw=1.8, label="dense (model_spectrum)")
        ax.plot(energy_bins, preds["dense"]["spec"], "s", color="steelblue", ms=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Intensity (arb., log)")
    ax.legend()
    ax.set_title("Predicted energy spectrum over full 0-50 MeV range")

    ax = axes[2]
    if sat_mask is not None and sat_mask.any():
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

    ax = axes[3]
    ax.fill_between(pff_energy_bins, band_lo, band_hi, color="darkorange", alpha=0.25,
                     label="+/-1sig band (total: aleatoric+epistemic)")
    ax.plot(pff_energy_bins, pff_spec, color="darkorange", lw=1.8, label="ensemble mean fit")
    if bump_present:
        ax.axvline(mean[3], color="darkorange", ls=":", lw=1.2,
                   label=f"bump centre {mean[3]:.1f}+/-{sigma_total[3]:.1f} MeV")
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Intensity (arb., log)")
    ax.legend(fontsize=8)
    param_str = "  ".join(f"{n}={mean[j]:.2f}±{sigma_total[j]:.2f}(epi{epi_frac[j]:.0f}%)"
                           for j, n in enumerate(PARAM_NAMES))
    ax.set_title(f"PFF fit [5-member ensemble] — p(bump)={pb_mean:.2f}±{pb_std:.2f}\n{param_str}",
                 fontsize=8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"cnn_infer_{shot_name}_n{N_BINS}{variant_tag}_ensemble.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
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
            from train_spectrum import build_model as build_dense
            dense_model = build_dense(N_BINS)
            dense_model.load_weights(dense_path)
    except (FileNotFoundError, KeyError):
        print(f"  (no dense baseline for n_bins={N_BINS} — plotting CNN alone)")

    print(f"Loading {N_MEMBERS}-member PFF ensemble...")
    members = load_ensemble()

    for shot_path in shot_paths:
        run_shot(shot_path, drm, cnn_model, cnn_res, dense_model, dense_res, members)

    print(f"\nAll figures saved to {OUT_DIR}/")
