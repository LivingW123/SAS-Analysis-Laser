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
train_spectrum.py (single bremsstrahlung exponential + one Gaussian bump),
so results are directly comparable to the dense model_spectrum_n{n} models.
"""

import json
import os
import time

import numpy as np
import tensorflow as tf

from data_utils import (
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------------------------------------------------------
# 200-vector -> 5x48 image  (mirrors gen_train_data.ipynb "make 2d samples")
# ---------------------------------------------------------------------------

def reshape_to_2d(X: np.ndarray) -> np.ndarray:
    """
    (n, 200) -> (n, 5, 48, 1)

    Rows 0-3: channels 0-191 reshaped to (4, 48).
    Row 4   : channels 192-199 (the 8 averaged bottom values), each value
              tiled across 6 consecutive columns.
    """
    n = X.shape[0]
    out = np.empty((n, 5, 48), dtype=np.float32)
    out[:, :4, :] = X[:, :192].reshape(n, 4, 48)
    out[:, 4, :] = np.repeat(X[:, 192:], 6, axis=1)  # (n, 8) -> (n, 48)
    return out[..., np.newaxis]


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
# Model
# ---------------------------------------------------------------------------

def build_model(n_bins: int) -> tf.keras.Model:
    """
    5x48x1 -> conv stack -> dense head -> n_bins (softmax).

    The image is only 5 rows tall, so pooling happens along the width axis
    only; 3x3 convs with 'same' padding keep the row dimension intact.
    """
    inp = tf.keras.Input(shape=(5, 48, 1), name="detector_image")

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)   # 5x24

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)   # 5x12

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    out = tf.keras.layers.Dense(n_bins, activation="softmax", name="spectrum")(x)
    return tf.keras.Model(inp, out, name=f"cnn_spectrum_regressor_n{n_bins}")


# ---------------------------------------------------------------------------
# Train one n
# ---------------------------------------------------------------------------

def train_for_n(drm: np.ndarray, n_bins: int, rng: np.random.Generator) -> tuple[dict, tf.keras.Model]:
    print(f"\n{'='*65}")
    print(f"  CNN | n_bins = {n_bins}  |  {50/n_bins:.2f} MeV/bin  |  N_SAMPLES = {N_SAMPLES:,}")
    print(f"{'='*65}", flush=True)

    drm_binned = bin_drm(drm, n_bins)  # (200, n_bins)

    # --- z-score stats on the 200-vector (same convention as train_spectrum) ---
    print("Bootstrapping normalisation stats (10k samples)...", flush=True)
    X_boot, _ = generate_spectrum_batch(drm_binned, min(10_000, N_SAMPLES), rng, BUMP_FRACTION)
    mean, std = normalize_fit(X_boot)
    del X_boot

    # --- fixed validation set ---
    print(f"Generating {VAL_SAMPLES:,} validation samples...", flush=True)
    val_rng = np.random.default_rng(SEED + 1)
    X_val, y_val = generate_spectrum_batch(drm_binned, VAL_SAMPLES, val_rng, BUMP_FRACTION)
    X_val_2d = reshape_to_2d(normalize_apply(X_val, mean, std))
    del X_val

    # --- training data ---
    print(f"Generating {N_SAMPLES:,} training samples...", flush=True)
    t_gen = time.perf_counter()
    X_all, y_all = generate_spectrum_batch(drm_binned, N_SAMPLES, rng, BUMP_FRACTION)
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

    model_path = f"model_cnn_n{n_bins}.keras"
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

    drm = load_drm(XLSX_PATH)
    print(f"DRM: {drm.shape}")

    json_path = "cnn_training_results.json"
    all_results: dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)

    for n_bins in N_VALUES:
        model_path = f"model_cnn_n{n_bins}.keras"
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
