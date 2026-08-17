"""
Tabulate the CNN-predicted spectral peak for real shot vectors, comparing the
uncorrected *_proc_vector.csv against Matthew Price's horizontally-corrected
*_proc_vector_ch.csv.

For each shot and each CSV variant the trained 5x48 CNN (model_cnn_n{N}.keras)
is run on the L1- + z-score-normalised 200-channel lineout, and the argmax
energy bin (peak) is recorded along with its softmax weight and the L1-space
reconstruction residual.

Usage
-----
  python -m src.comparisons.peak_table
  CNN_NBINS=20 python -m src.comparisons.peak_table
  CNN_NBINS=50 python -m src.comparisons.peak_table res/test_images/11696 res/test_images/11705 ...

Positional args, if given, are shot directories (or shot numbers); otherwise the
default six shots below are used. Output goes to console and to
  out/comparisons/cnn_peak_table_n{N}.csv

The model is loaded by rebuilding src.core.cnn_model.build_model(N) and loading
weights, rather than tf.keras.load_model, so that .keras files saved with a
newer Keras (carrying a `quantization_config` Dense field that older releases
reject) still load. Falls back to load_model if the rebuild path is unavailable.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.core.data_utils import bin_drm, load_drm, mev_bin_centers, normalize_apply

DRM_PATH  = "res/drm/200x200.xlsx"
MODEL_DIR = "out/training/cnn"
OUT_DIR   = "out/comparisons"
CNN_JSON  = os.environ.get("CNN_RESULTS_JSON", os.path.join(MODEL_DIR, "cnn_training_results.json"))
N_BINS    = int(os.environ.get("CNN_NBINS", 50))
MODEL_PREFIX = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
TEST_ROOT = "res/test_images"

# 4 shots run this session + the 2 default shots (10084, 11733) from before
DEFAULT_SHOTS = ["11696", "11705", "11707", "11716", "10084", "11733"]

# CSV variants to evaluate per shot (skip _cv per request — redundant)
VARIANTS = [
    ("proc_vector",    "{s}_proc_vector.csv"),      # uncorrected
    ("proc_vector_ch", "{s}_proc_vector_ch.csv"),   # Matthew Price horizontal correction
]

# Shots to run uncorrected-only on, skipping the _ch horizontal-correction variant
NO_CH_SHOTS = {"10084", "11733"}


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)} in {csv_path}"
    return sig


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def load_cnn(n_bins: int):
    """Rebuild architecture + load weights (newer-keras-safe); fall back to load_model."""
    path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_n{n_bins}.keras")
    try:
        from src.core.cnn_model import build_model
        model = build_model(n_bins)
        model.load_weights(path)
        return model
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"  (weight-load fallback failed: {exc}; trying tf.keras.load_model)")
        return tf.keras.models.load_model(path, compile=False)


def shot_number(arg: str) -> str:
    """Accept a shot number, a shot dir, or a csv path and return the 5-digit shot id."""
    base = os.path.basename(os.path.normpath(arg))
    return base.split("_")[0]


def main() -> None:
    from src.core.cnn_model import reshape_to_2d

    os.makedirs(OUT_DIR, exist_ok=True)

    args = sys.argv[1:]
    shots = [shot_number(a) for a in args] if args else DEFAULT_SHOTS

    drm = load_drm(DRM_PATH)
    drm_binned = bin_drm(drm, N_BINS)
    energy_bins = mev_bin_centers(N_BINS)

    with open(CNN_JSON) as f:
        res = json.load(f)[str(N_BINS)]
    mean = np.array(res["norm_mean"], dtype=np.float32)
    std  = np.array(res["norm_std"],  dtype=np.float32)

    model = load_cnn(N_BINS)

    rows = []
    for s in shots:
        row = {"shot": s}
        for label, pattern in VARIANTS:
            if label == "proc_vector_ch" and s in NO_CH_SHOTS:
                row[f"{label}_peak_mev"]  = np.nan
                row[f"{label}_peak_frac"] = np.nan
                row[f"{label}_resid_pct"] = np.nan
                continue
            csv_path = os.path.join(TEST_ROOT, s, pattern.format(s=s))
            if not os.path.exists(csv_path):
                row[f"{label}_peak_mev"]  = np.nan
                row[f"{label}_peak_frac"] = np.nan
                row[f"{label}_resid_pct"] = np.nan
                continue

            signal_l1 = l1_normalise(load_signal(csv_path))
            x_norm = normalize_apply(signal_l1.reshape(1, -1), mean, std)
            spec = model.predict(reshape_to_2d(x_norm), verbose=0)[0]

            peak_mev  = float(energy_bins[spec.argmax()])
            peak_frac = float(spec.max())

            resp_l1 = l1_normalise(drm_binned @ spec)
            resid_rms  = float(np.sqrt(np.mean((signal_l1 - resp_l1) ** 2)))
            signal_rms = float(np.sqrt(np.mean(signal_l1 ** 2)))
            resid_pct  = 100.0 * resid_rms / signal_rms if signal_rms > 0 else np.nan

            row[f"{label}_peak_mev"]  = peak_mev
            row[f"{label}_peak_frac"] = round(peak_frac, 4)
            row[f"{label}_resid_pct"] = round(resid_pct, 1)
        rows.append(row)

    table = pd.DataFrame(rows)

    out_csv = os.path.join(OUT_DIR, f"cnn_peak_table_n{N_BINS}.csv")
    table.to_csv(out_csv, index=False)

    pd.set_option("display.width", 120)
    print(f"\nCNN peak table (n_bins={N_BINS})  [peak in MeV, frac = softmax weight, "
          f"resid = L1 reconstruction RMS as % of signal]\n")
    print(table.to_string(index=False))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
