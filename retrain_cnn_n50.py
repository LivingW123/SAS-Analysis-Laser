"""
retrain_cnn_n50.py — retrain the n_bins=50 CNN spectrum classifier
(model_cnn_n50.keras) now that data_utils.PFF_PARAM_SAMPLING has been
recalibrated (a4 recentred on [10,20] MeV, a5/a6 tightened -- see
data_utils.py's comment above PFF_PARAM_SAMPLING). generate_spectrum_batch
(train_cnn.py's training-data source) draws its synthetic bumps from that
same table via sample_pff_spectra, so the existing model_cnn_n50.keras was
trained on the old, poorly-calibrated bump distribution just like the PFF
regressor was.

N_SAMPLES=25_000, not train_cnn.py's file-level default of 100_000 --
matches this repo's own documented finding (README.md's CNN pipeline
section) that for n_bins=50 specifically, a smaller ~25k in-memory run
outperformed a much larger (~17M) chunked run on real-shot residuals.
Reuses train_cnn.py's train_for_n directly (only N_SAMPLES overridden) so
architecture/callbacks/logging stay identical to every other n_bins run.

Old-prior model_cnn_n50.keras + its cnn_training_results.json entry were
copied to cnn_oldprior_backup/ before this runs.

Usage
-----
  python retrain_cnn_n50.py
"""

import json
import os

import numpy as np

import train_cnn
from data_utils import load_drm

train_cnn.N_SAMPLES = 25_000

XLSX_PATH = "res/drm/200x200.xlsx"
N_BINS = 50
JSON_PATH = "cnn_training_results.json"

if __name__ == "__main__":
    rng = np.random.default_rng(train_cnn.SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM: {drm.shape}")

    results, model = train_cnn.train_for_n(drm, N_BINS, rng)

    model_path = f"model_cnn_n{N_BINS}.keras"
    model.save(model_path)

    all_results = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            all_results = json.load(f)
    all_results[str(N_BINS)] = results
    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved {model_path} and updated {JSON_PATH}")
