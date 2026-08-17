"""
CNN spectrum regression from the 2D (5x48) detector-image representation.

Input  : 5x48 single-channel image built from the 200-channel L1 +
         z-score-normalised detector response, using the same reshape as
         gen_train_data.ipynb's "make 2d samples" cell:
           rows 0-3 : first 192 channels reshaped to (4, 48)
           row  4   : last 8 channels, each tiled across 6 columns
Output : n-bin L1-normalised PFF spectrum (softmax, sums to 1)
Loss   : MSE on L1-normalised spectrum bins

Training data comes from data_utils.generate_spectrum_batch — identical to
train_dnn_spectrum.py (single bremsstrahlung exponential + one Gaussian
bump), so results are directly comparable to the dense model_dnn_spectrum_n{n}
models.

Model architecture (build_model) and the 200->5x48 reshape (reshape_to_2d)
live in src/core/cnn_model.py, shared with every other script that needs
this CNN's definition.
"""

import json
import os
import time

import numpy as np
import tensorflow as tf

from src.core.cnn_model import build_model, reshape_to_2d
from src.core.data_utils import (
    SAT_FRACTION,
    bin_drm,
    generate_spectrum_batch,
    load_drm,
    normalize_apply,
    normalize_fit,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
XLSX_PATH     = "res/drm/200x200.xlsx"
N_VALUES      = [10]         # output-node counts
N_SAMPLES     = 100_000      # training samples per n (in-memory)
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 200
BATCH_SIZE    = 256
PATIENCE      = 30
SEED          = 42
LR            = 1e-3
VAL_SAMPLES   = 5_000

OUT_DIR = "out/training/cnn"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(tf.keras.callbacks.Callback):
    """Print one line every PRINT_EVERY epochs."""
    PRINT_EVERY = 10

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.PRINT_EVERY == 0 and logs:
            print(f"  ep {epoch+1:4d} | loss {logs['loss']:.2e} | "
                  f"val_loss {logs['val_loss']:.2e} | val_mae {logs['val_mae']:.2e}",
                  flush=True)


# ---------------------------------------------------------------------------
# Train one n
# ---------------------------------------------------------------------------

def train_for_n(drm: np.ndarray, n_bins: int, rng: np.random.Generator) -> tuple[dict, tf.keras.Model]:
    print(f"\n{'='*65}")
    print(f"  CNN | n_bins = {n_bins}  |  {50/n_bins:.2f} MeV/bin  |  N_SAMPLES = {N_SAMPLES:,}")
    print(f"{'='*65}", flush=True)

    drm_binned = bin_drm(drm, n_bins)  # (200, n_bins)

    # --- z-score stats on the 200-vector (same convention as train_dnn_spectrum) ---
    print("Bootstrapping normalisation stats (10k samples)...", flush=True)
    X_boot, _ = generate_spectrum_batch(drm_binned, min(10_000, N_SAMPLES), rng, BUMP_FRACTION,
                                        sat_fraction=SAT_FRACTION)
    mean, std = normalize_fit(X_boot)
    del X_boot

    # --- fixed validation set ---
    print(f"Generating {VAL_SAMPLES:,} validation samples...", flush=True)
    val_rng = np.random.default_rng(SEED + 1)
    X_val, y_val = generate_spectrum_batch(drm_binned, VAL_SAMPLES, val_rng, BUMP_FRACTION,
                                           sat_fraction=SAT_FRACTION)
    X_val_2d = reshape_to_2d(normalize_apply(X_val, mean, std))
    del X_val

    # --- training data ---
    print(f"Generating {N_SAMPLES:,} training samples...", flush=True)
    t_gen = time.perf_counter()
    X_all, y_all = generate_spectrum_batch(drm_binned, N_SAMPLES, rng, BUMP_FRACTION,
                                           sat_fraction=SAT_FRACTION)
    X_all_2d = reshape_to_2d(normalize_apply(X_all, mean, std))
    del X_all
    print(f"  Generated in {time.perf_counter() - t_gen:.1f}s  "
          f"(X: {X_all_2d.nbytes/1e6:.0f} MB, y: {y_all.nbytes/1e6:.0f} MB)", flush=True)

    # --- model ---
    tf.random.set_seed(SEED)
    model = build_model(n_bins)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    model_path = os.path.join(OUT_DIR, f"model_cnn_n{n_bins}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss",
            save_best_only=True, mode="min", verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0,
        ),
        ProgressCallback(),
    ]

    t_train_start = time.perf_counter()
    history = model.fit(
        X_all_2d, y_all,
        validation_data=(X_val_2d, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    epochs_run = len(history.history["loss"])
    t_train = time.perf_counter() - t_train_start

    best_ep    = int(np.argmin(history.history["val_loss"]))
    best_vloss = float(history.history["val_loss"][best_ep])
    best_vmae  = float(history.history["val_mae"][best_ep])

    print(f"\n  Stopped at epoch {epochs_run}  (best: {best_ep + 1})")
    print(f"  Training wall time : {t_train:.1f}s  ({t_train / epochs_run:.2f}s/epoch)")
    print(f"  Best val_loss      : {best_vloss:.6f}  (MSE on L1-normed {n_bins}-bin spectrum)")
    print(f"  Best val_MAE       : {best_vmae:.6f}", flush=True)

    results = {
        "n_bins":         n_bins,
        "n_samples":      N_SAMPLES,
        "model":          "cnn_5x48",
        "epochs_trained": epochs_run,
        "best_epoch":     best_ep + 1,
        "best_val_loss":  best_vloss,
        "best_val_mae":   best_vmae,
        "learning_rate":  LR,
        "batch_size":     BATCH_SIZE,
        "norm_mean":      mean.tolist(),
        "norm_std":       std.tolist(),
        "val_loss":       [float(v) for v in history.history["val_loss"]],
        "train_loss":     [float(v) for v in history.history["loss"]],
        "val_mae":        [float(v) for v in history.history["val_mae"]],
    }
    return results, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    drm = load_drm(XLSX_PATH)
    print(f"DRM: {drm.shape}")

    json_path = os.path.join(OUT_DIR, "cnn_training_results.json")
    all_results: dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)

    for n_bins in N_VALUES:
        model_path = os.path.join(OUT_DIR, f"model_cnn_n{n_bins}.keras")
        if os.path.exists(model_path) and str(n_bins) in all_results:
            print(f"\n  n_bins={n_bins}: {model_path} exists — skipping training.")
            continue
        results, model = train_for_n(drm, n_bins, rng)
        all_results[str(n_bins)] = results
        model.save(model_path)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {model_path}")

    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
    print(f"Saved {json_path}")
