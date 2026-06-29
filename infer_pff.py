"""
Run the trained PFF regressor on 11733.csv and sanity-check the result.

The real signal is L1-normalised before inference to match the training
distribution (training data was also L1-normalised in generate_pff_training_data).

Outputs
-------
  Console : predicted PFF params, normalised param positions, residual stats
  pff_infer_11733.png : 3-panel figure
      top    — L1-normalised real signal vs L1-normalised DRM-forward of predicted spectrum
      middle — predicted PFF spectrum in energy space
      bottom — channel-wise residual
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import (
    PFF_PARAM_BOUNDS,
    PFF_PARAM_SAMPLING,
    denormalize_pff_params,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    pff_func,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CSV_PATH   = "res/test_images/11733/processed/11733.csv"
JSON_PATH  = "pff_training_results.json"
MODEL_PATH = "model_pff.keras"
DRM_PATH   = "res/drm/200x200.xlsx"
PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5"]


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df["signal"].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


def l1_normalise(x: np.ndarray) -> np.ndarray:
    """Normalise a 1-D detector response to sum to 1."""
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


if __name__ == "__main__":
    # --- load ---
    signal = load_signal(CSV_PATH)
    drm    = load_drm(DRM_PATH)
    model  = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(JSON_PATH) as f:
        results = json.load(f)

    mean = np.array(results["norm_mean"], dtype=np.float32)
    std  = np.array(results["norm_std"],  dtype=np.float32)

    # L1-normalise the real signal to match training distribution
    signal_l1 = l1_normalise(signal)

    # z-score normalise (same as training)
    x_norm = normalize_apply(signal_l1.reshape(1, -1), mean, std)

    # predict
    p_norm = model.predict(x_norm, verbose=0)[0]               # (5,) in [0,1]
    p_phys = denormalize_pff_params(p_norm.reshape(1, -1))[0]  # (5,) physical units

    # --- reconstruct: spectrum -> DRM -> L1-normalise ---
    energy_bins   = mev_bin_centers(drm.shape[1])
    spec_pred     = pff_func(energy_bins, p_phys)   # (200,) energy space
    response_pred = drm @ spec_pred                  # (200,) detector space (absolute)
    response_l1   = l1_normalise(response_pred)      # same scale as signal_l1

    residual   = signal_l1 - response_l1
    resid_rms  = float(np.sqrt(np.mean(residual ** 2)))
    signal_rms = float(np.sqrt(np.mean(signal_l1 ** 2)))

    # --- where do predicted params fall in the training distribution? ---
    mu    = PFF_PARAM_SAMPLING[:, 0]
    sig_p = PFF_PARAM_SAMPLING[:, 1]

    print("\n=== PFF inference on 11733 ===")
    print(f"{'Param':>5}  {'Predicted':>10}  {'Train mean':>11}  "
          f"{'Norm [0,1]':>10}  {'z-score':>10}")
    for j, n in enumerate(PARAM_NAMES):
        norm_val = float(p_norm[j])
        phys_val = float(p_phys[j])
        z        = (phys_val - mu[j]) / sig_p[j]
        print(f"  {n:>3}  {phys_val:>10.3f}  {mu[j]:>11.3f}  "
              f"{norm_val:>10.4f}  {z:>+10.2f}")

    print(f"\nResidual RMS (L1 space) : {resid_rms:.6f}")
    print(f"Signal RMS  (L1 space)  : {signal_rms:.6f}")
    print(f"Relative residual       : {100*resid_rms/signal_rms:.1f}%")

    bump_present = float(p_phys[2]) > 5.0
    print(f"\na3 = {p_phys[2]:.2f}  ->  bump {'DETECTED' if bump_present else 'NOT detected'}")
    if bump_present:
        print(f"  Bump centre  a4 = {p_phys[3]:.1f} MeV")
        print(f"  Bump width   a5 = {p_phys[4]:.1f}")

    # --- figure ---
    channels = np.arange(1, 201)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), tight_layout=True)
    fig.suptitle("PFF inference — shot 11733", fontsize=13)

    ax = axes[0]
    ax.plot(channels, signal_l1,    "k",   lw=1.2, label="real signal (L1-norm)")
    ax.plot(channels, response_l1,  "r--", lw=1.2, label="DRM x pred spectrum (L1-norm)")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Fraction of total signal")
    ax.legend()
    ax.set_title(f"Detector response: real vs reconstructed  (residual {100*resid_rms/signal_rms:.1f}%)")

    ax = axes[1]
    brems = p_phys[0] * np.exp(-p_phys[1] * energy_bins)
    ax.plot(energy_bins, spec_pred, "b",   lw=1.5, label="full PFF spectrum")
    ax.plot(energy_bins, brems,     "g--", lw=1.2, label="bremsstrahlung only")
    if bump_present:
        ax.axvline(p_phys[3], color="r", ls=":", lw=1,
                   label=f"bump centre {p_phys[3]:.1f} MeV")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Intensity (arb.)")
    ax.legend()
    ax.set_title("Predicted PFF spectrum (energy space)")

    ax = axes[2]
    ax.plot(channels, residual, "m", lw=1)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Detector channel")
    ax.set_ylabel("Residual (real L1 - pred L1)")
    ax.set_title(f"Channel residual  (RMS={resid_rms:.6f}, "
                 f"{100*resid_rms/signal_rms:.1f}% of signal)")

    fig.savefig("pff_infer_11733.png", dpi=150)
    print("\nSaved pff_infer_11733.png")
