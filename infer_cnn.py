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
The dense model_spectrum_n{N} baseline is included only if a matching
model file and spectrum_training_results.json[str(N)] entry both exist;
otherwise the CNN is plotted alone.

Outputs
-------
  cnn_infer_<shot>_n<n_bins>.png : 3-panel figure per shot
      top    — real signal vs DRM-forward of CNN and dense predictions
      middle — CNN vs dense predicted spectra (bar)
      bottom — channel-wise residuals for both models
  cnn_infer_<shot>_n<n_bins>_unsat.png : 2-panel figure per shot (only when a
      saturation mask is available for the shot) — same detector-response and
      residual comparison, but with saturation-corrected (imputed) channels
      masked out, so the fit quality on real measurements is visible without
      the imputed-channel distortion.
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
)
from train_cnn import reshape_to_2d

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DRM_PATH       = "res/drm/200x200.xlsx"
CNN_JSON       = os.environ.get("CNN_RESULTS_JSON", "cnn_training_results.json")
DENSE_JSON     = "spectrum_training_results.json"
N_BINS         = int(os.environ.get("CNN_NBINS", 10))
MODEL_PREFIX   = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
OUT_TAG        = os.environ.get("CNN_OUT_TAG", "")

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


def run_shot(shot_path: str, drm: np.ndarray, cnn_model, cnn_res: dict,
             dense_model=None, dense_res: dict | None = None) -> None:
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

    # --- figure ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), tight_layout=True)
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

    for shot_path in shot_paths:
        run_shot(shot_path, drm, cnn_model, cnn_res, dense_model, dense_res)
