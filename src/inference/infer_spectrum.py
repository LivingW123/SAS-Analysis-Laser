"""
Run the trained spectrum regressor(s) on a real shot CSV and sanity-check.

Outputs
-------
  Console : predicted spectrum stats, residual RMS (per n_bins)
  out/inference/spectrum_infer_<shot>_n<n_bins>.png : 3-panel figure per n_bins
      top    — L1-normalised real signal vs DRM-forward of predicted spectrum
      middle — predicted n_bins energy spectrum
      bottom — channel-wise residual
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.core.data_utils import (
    bin_drm,
    load_drm,
    mev_bin_centers,
    normalize_apply,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DRM_PATH  = "res/drm/200x200.xlsx"
JSON_PATH = "out/training/dnn/dnn_spectrum_training_results.json"
OUT_DIR   = "out/inference"
N_VALUES  = [10, 20, 50]


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df["signal"].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def run_inference(csv_path: str, shot_name: str, n_bins: int, signal: np.ndarray, drm: np.ndarray, results: dict) -> None:
    model_path = os.path.join("out/training/dnn", f"model_dnn_spectrum_n{n_bins}.keras")
    res = results[str(n_bins)]

    drm_binned = bin_drm(drm, n_bins)
    model = tf.keras.models.load_model(model_path, compile=False)

    mean = np.array(res["norm_mean"], dtype=np.float32)
    std  = np.array(res["norm_std"],  dtype=np.float32)

    # --- preprocess real signal ---
    signal_l1 = l1_normalise(signal)                           # (200,)
    x_norm    = normalize_apply(signal_l1.reshape(1, -1), mean, std)

    # --- predict ---
    spec_pred = model.predict(x_norm, verbose=0)[0]            # (n_bins,) L1-normed spectrum

    # --- forward model: spectrum -> detector channels ---
    response_pred = drm_binned @ spec_pred                     # (200,) absolute
    response_l1   = l1_normalise(response_pred)                # match signal_l1 scale

    residual   = signal_l1 - response_l1
    resid_rms  = float(np.sqrt(np.mean(residual ** 2)))
    signal_rms = float(np.sqrt(np.mean(signal_l1 ** 2)))

    energy_bins = mev_bin_centers(n_bins)

    print(f"\n=== Spectrum inference: {shot_name}  (n_bins={n_bins}) ===")
    print(f"Training:  n_samples={res['n_samples']:,}  mode={res['mode']}")
    print(f"           best_epoch={res['best_epoch']}  val_loss={res['best_val_loss']:.2e}")
    print(f"\nPredicted spectrum (L1-normed, {n_bins} bins):")
    print(f"  Peak bin : {energy_bins[spec_pred.argmax()]:.0f} MeV  ({spec_pred.max()*100:.2f}% of signal)")
    print(f"  Low-E fraction  (0-10 MeV)  : {spec_pred[energy_bins <= 10].sum()*100:.1f}%")
    print(f"  Mid-E fraction  (10-30 MeV) : {spec_pred[(energy_bins > 10) & (energy_bins <= 30)].sum()*100:.1f}%")
    print(f"  High-E fraction (30-50 MeV) : {spec_pred[energy_bins > 30].sum()*100:.1f}%")
    print(f"\nResidual RMS (L1 space) : {resid_rms:.6f}")
    print(f"Signal RMS  (L1 space)  : {signal_rms:.6f}")
    print(f"Relative residual       : {100*resid_rms/signal_rms:.1f}%")

    # --- figure ---
    channels = np.arange(1, 201)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), tight_layout=True)
    fig.suptitle(f"Spectrum inference — shot {shot_name}  (n_bins={n_bins})  "
                 f"(n={res['n_samples']:,}, val_loss={res['best_val_loss']:.2e})", fontsize=12)

    ax = axes[0]
    ax.plot(channels, signal_l1,   "k",   lw=1.2, label="real signal (L1-norm)")
    ax.plot(channels, response_l1, "r--", lw=1.2, label="DRM x pred spectrum (L1-norm)")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Fraction of total signal")
    ax.legend()
    ax.set_title(f"Detector response: real vs reconstructed  (residual {100*resid_rms/signal_rms:.1f}%)")

    ax = axes[1]
    ax.bar(energy_bins, spec_pred, width=0.9 * 50 / n_bins, color="steelblue", alpha=0.8, label="predicted spectrum")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Fraction of total")
    ax.legend()
    ax.set_title(f"Predicted {n_bins}-bin energy spectrum")

    ax = axes[2]
    ax.plot(channels, residual, "m", lw=1)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Residual (real L1 - pred L1)")
    ax.set_title(f"Channel residual  (RMS={resid_rms:.6f},  {100*resid_rms/signal_rms:.1f}% of signal)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"spectrum_infer_{shot_name}_n{n_bins}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "res/test_images/11733/processed/11733.csv"
    shot_name = os.path.splitext(os.path.basename(csv_path))[0]

    signal = load_signal(csv_path)
    drm    = load_drm(DRM_PATH)

    with open(JSON_PATH) as f:
        results = json.load(f)

    for n_bins in N_VALUES:
        run_inference(csv_path, shot_name, n_bins, signal, drm, results)
