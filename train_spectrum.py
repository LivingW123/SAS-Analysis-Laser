"""
Direct 50-bin spectrum regression from 200-channel detector response.

Input  : 200-channel L1 + z-score-normalised detector response
Output : 50-bin L1-normalised PFF spectrum (softmax, sums to 1)
Loss   : MSE on L1-normalised spectrum bins

Scaling behaviour
-----------------
  N_SAMPLES <= MEMORY_LIMIT  : pre-generate all data, train on arrays in RAM
  N_SAMPLES >  MEMORY_LIMIT  : streaming generator — O(batch) RAM regardless of N
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
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
)

# ---------------------------------------------------------------------------
# Config — change N_SAMPLES to scale from 100k to 10M+
# ---------------------------------------------------------------------------
XLSX_PATH      = "res/drm/200x200.xlsx"
N_SAMPLES      = 10_000_000   # total training samples to expose the model to
N_BINS         = 50           # 1-MeV energy bins
BUMP_FRACTION  = 0.5
MAX_EPOCHS     = 200
BATCH_SIZE     = 256
PATIENCE       = 30
SEED           = 42
LR             = 1e-3
VAL_SAMPLES    = 5_000

MEMORY_LIMIT   = 1_000_000   # pre-generate below this; stream above

# In streaming mode, each "epoch" covers this many steps
STREAM_STEPS_PER_EPOCH = 2_000   # 2000 * 256 = 512k samples per epoch

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
                  f"val_loss {logs['val_loss']:.2e} | val_mae {logs['val_mae']:.2e}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    """200 -> 512 -> 256 -> 128 -> 64 -> 50 (softmax)."""
    inp = tf.keras.Input(shape=(200,), name="detector_response")
    x = tf.keras.layers.Dense(512)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(N_BINS, activation="softmax", name="spectrum")(x)
    return tf.keras.Model(inp, out, name="spectrum_regressor")


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

def _stream_gen(drm_50, mean, std, batch_size, bump_fraction, seed):
    """Infinite generator of z-score-normalised (X_batch, y_batch) arrays."""
    rng = np.random.default_rng(seed)
    while True:
        X, y = generate_spectrum_batch(drm_50, batch_size, rng, bump_fraction)
        yield normalize_apply(X, mean, std), y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    drm    = load_drm(XLSX_PATH)
    drm_50 = bin_drm(drm, N_BINS)             # (200, 50)
    print(f"DRM: {drm.shape}  ->  binned DRM: {drm_50.shape}")

    mode = "memory" if N_SAMPLES <= MEMORY_LIMIT else "stream"
    print(f"N_SAMPLES={N_SAMPLES:,}  BATCH={BATCH_SIZE}  mode={mode}")

    # --- z-score stats (bootstrap from a small sample so memory path matches stream path) ---
    print("Bootstrapping normalisation stats (10k samples)...")
    X_boot, _ = generate_spectrum_batch(drm_50, min(10_000, N_SAMPLES), rng, BUMP_FRACTION)
    mean, std = normalize_fit(X_boot)
    del X_boot

    # --- fixed validation set ---
    print(f"Generating {VAL_SAMPLES:,} validation samples...")
    val_rng = np.random.default_rng(SEED + 1)
    X_val, y_val = generate_spectrum_batch(drm_50, VAL_SAMPLES, val_rng, BUMP_FRACTION)
    X_val_n = normalize_apply(X_val, mean, std)
    del X_val

    # --- model ---
    tf.random.set_seed(SEED)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model_spectrum.keras", monitor="val_loss",
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

    if mode == "memory":
        print(f"\nGenerating {N_SAMPLES:,} training samples...")
        t_gen = time.perf_counter()
        X_all, y_all = generate_spectrum_batch(drm_50, N_SAMPLES, rng, BUMP_FRACTION)
        print(f"  Generated in {time.perf_counter() - t_gen:.1f}s  "
              f"(X: {X_all.nbytes/1e6:.0f} MB, y: {y_all.nbytes/1e6:.0f} MB)")
        X_all_n = normalize_apply(X_all, mean, std)
        del X_all

        history = model.fit(
            X_all_n, y_all,
            validation_data=(X_val_n, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )
        epochs_run = len(history.history["loss"])

    else:  # streaming
        total_steps  = N_SAMPLES // BATCH_SIZE
        spe          = STREAM_STEPS_PER_EPOCH
        n_epochs_max = max(MAX_EPOCHS, total_steps // spe + 1)
        print(f"\nStreaming: {spe} steps/epoch x {n_epochs_max} max epochs  "
              f"(= {spe * n_epochs_max * BATCH_SIZE:,} total sample exposure)")

        gen = _stream_gen(drm_50, mean, std, BATCH_SIZE, BUMP_FRACTION, SEED + 100)
        history = model.fit(
            gen,
            steps_per_epoch=spe,
            validation_data=(X_val_n, y_val),
            epochs=n_epochs_max,
            callbacks=callbacks,
            verbose=0,
        )
        epochs_run = len(history.history["loss"])

    t_train = time.perf_counter() - t_train_start

    best_ep    = int(np.argmin(history.history["val_loss"]))
    best_vloss = float(history.history["val_loss"][best_ep])
    best_vmae  = float(history.history["val_mae"][best_ep])

    print(f"\nStopped at epoch {epochs_run}  (best: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train / epochs_run:.2f}s/epoch)")
    print(f"Best val_loss      : {best_vloss:.6f}  (MSE on L1-normed 50-bin spectrum)")
    print(f"Best val_MAE       : {best_vmae:.6f}")

    results = {
        "n_samples":       N_SAMPLES,
        "n_bins":          N_BINS,
        "mode":            mode,
        "epochs_trained":  epochs_run,
        "best_epoch":      best_ep + 1,
        "best_val_loss":   best_vloss,
        "best_val_mae":    best_vmae,
        "learning_rate":   LR,
        "batch_size":      BATCH_SIZE,
        "norm_mean":       mean.tolist(),
        "norm_std":        std.tolist(),
        "val_loss":        [float(v) for v in history.history["val_loss"]],
        "train_loss":      [float(v) for v in history.history["loss"]],
        "val_mae":         [float(v) for v in history.history["val_mae"]],
    }
    with open("spectrum_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved model_spectrum.keras and spectrum_training_results.json")
    print(f"Total wall time: {time.perf_counter() - t_start:.1f}s")
