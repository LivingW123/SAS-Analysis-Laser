"""
Hybrid CNN spectrum regression from the 200-channel SAS detector response.

Input:
    200-channel L1-normalised + z-score-normalised detector response.

    Branch 1:
        channels 0-191
        -> reshape to (4, 48, 1)
        -> Conv2D feature extractor

    Branch 2:
        channels 192-199
        -> keep as 8 independent coarse detector measurements
        -> dense feature extractor

    The two branches are concatenated and mapped to an n-bin spectrum.

Output:
    n-bin L1-normalised PFF spectrum
    (softmax, sums to 1)

Loss:
    MSE on L1-normalised spectrum bins

Training data:
    data_utils.generate_spectrum_batch
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
N_VALUES      = [10]
N_SAMPLES     = 100_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 200
BATCH_SIZE    = 256
PATIENCE      = 30
SEED          = 42
LR            = 1e-3
VAL_SAMPLES   = 5_000

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------------------------------------------------------
# Split 200-vector into physically distinct detector regions
# ---------------------------------------------------------------------------

def split_detector_input(X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Split the 200 effective SAS detector measurements into:

        channels 0-191:
            four fully resolved groups of 48 pixels
            -> shape (n, 4, 48, 1)

        channels 192-199:
            eight coarse effective pixels
            -> shape (n, 8)

    No artificial repetition of the final 8 measurements is performed.
    """

    n = X.shape[0]

    fine_image = X[:, :192].reshape(n, 4, 48, 1).astype(np.float32)
    coarse_pixels = X[:, 192:200].astype(np.float32)

    return {
        "fine_image": fine_image,
        "coarse_pixels": coarse_pixels,
    }


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(tf.keras.callbacks.Callback):
    """Print one line every PRINT_EVERY epochs."""

    PRINT_EVERY = 10

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.PRINT_EVERY == 0 and logs:
            print(
                f"  ep {epoch + 1:4d} | "
                f"loss {logs['loss']:.2e} | "
                f"val_loss {logs['val_loss']:.2e} | "
                f"val_mae {logs['val_mae']:.2e}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_bins: int) -> tf.keras.Model:
    """
    Hybrid detector architecture:

        first 192 measurements
                |
             4 x 48
                |
              CNN
                |
                +----------------+
                                 |
        final 8 measurements     |
                |                |
              Dense              |
                |                |
                +---- concatenate
                         |
                       Dense
                         |
                      Softmax
                         |
                    n-bin spectrum
    """

    # ------------------------------------------------------------------
    # Branch 1: 4 x 48 spatial detector region
    # ------------------------------------------------------------------

    fine_input = tf.keras.Input(
        shape=(4, 48, 1),
        name="fine_image",
    )

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
    )(fine_input)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Pool only along the 48-pixel longitudinal direction.
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 2)
    )(x)                                          # 4 x 24 x 64

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 2)
    )(x)                                          # 4 x 12 x 128

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    fine_features = tf.keras.layers.Activation(
        "relu",
        name="fine_features",
    )(x)


    # ------------------------------------------------------------------
    # Branch 2: eight coarse effective detector pixels
    # ------------------------------------------------------------------

    coarse_input = tf.keras.Input(
        shape=(8,),
        name="coarse_pixels",
    )

    c = tf.keras.layers.Dense(32)(coarse_input)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.Activation("relu")(c)

    c = tf.keras.layers.Dense(32)(c)
    c = tf.keras.layers.BatchNormalization()(c)

    coarse_features = tf.keras.layers.Activation(
        "relu",
        name="coarse_features",
    )(c)


    # ------------------------------------------------------------------
    # Combine both detector representations
    # ------------------------------------------------------------------

    combined = tf.keras.layers.Concatenate(
        name="combined_detector_features"
    )([
        fine_features,
        coarse_features,
    ])

    x = tf.keras.layers.Dense(128)(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    output = tf.keras.layers.Dense(
        n_bins,
        activation="softmax",
        name="spectrum",
    )(x)

    return tf.keras.Model(
        inputs={
            "fine_image": fine_input,
            "coarse_pixels": coarse_input,
        },
        outputs=output,
        name=f"hybrid_cnn_spectrum_regressor_n{n_bins}",
    )


# ---------------------------------------------------------------------------
# Train one n
# ---------------------------------------------------------------------------

def train_for_n(
    drm: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
) -> tuple[dict, tf.keras.Model]:

    print(f"\n{'=' * 70}")
    print(
        f"  HYBRID CNN | n_bins = {n_bins} | "
        f"{50 / n_bins:.2f} MeV/bin | "
        f"N_SAMPLES = {N_SAMPLES:,}"
    )
    print(f"{'=' * 70}", flush=True)


    # ------------------------------------------------------------------
    # Reduce the original 200-energy-bin DRM
    #
    # Example for n_bins = 10:
    #
    #     DRM:      (200, 200)
    #                   |
    #               energy binning
    #                   |
    #               (200, 10)
    #
    # Then:
    #
    #     spectrum s:       (10,)
    #     DRM @ s:          (200,)
    #     detector response r
    # ------------------------------------------------------------------

    drm_binned = bin_drm(drm, n_bins)

    print(f"Binned DRM shape: {drm_binned.shape}")


    # ------------------------------------------------------------------
    # Normalisation statistics
    # ------------------------------------------------------------------

    print(
        "Bootstrapping normalisation stats "
        "(10k detector responses)...",
        flush=True,
    )

    X_boot, _ = generate_spectrum_batch(
        drm_binned,
        min(10_000, N_SAMPLES),
        rng,
        BUMP_FRACTION,
    )

    # Per-channel statistics over the original 200-vector.
    mean, std = normalize_fit(X_boot)

    del X_boot


    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------

    print(
        f"Generating {VAL_SAMPLES:,} validation samples...",
        flush=True,
    )

    val_rng = np.random.default_rng(SEED + 1)

    X_val, y_val = generate_spectrum_batch(
        drm_binned,
        VAL_SAMPLES,
        val_rng,
        BUMP_FRACTION,
    )

    # X_val:
    #
    #   (5000, 200)
    #       |
    #       | z-score
    #       v
    #   (5000, 200)
    #       |
    #       +--> (5000, 4, 48, 1)
    #       |
    #       +--> (5000, 8)

    X_val_norm = normalize_apply(
        X_val,
        mean,
        std,
    )

    X_val_inputs = split_detector_input(X_val_norm)

    del X_val
    del X_val_norm


    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------

    print(
        f"Generating {N_SAMPLES:,} training samples...",
        flush=True,
    )

    t_gen = time.perf_counter()

    X_all, y_all = generate_spectrum_batch(
        drm_binned,
        N_SAMPLES,
        rng,
        BUMP_FRACTION,
    )

    X_all_norm = normalize_apply(
        X_all,
        mean,
        std,
    )

    X_all_inputs = split_detector_input(X_all_norm)

    del X_all
    del X_all_norm

    fine_mb = (
        X_all_inputs["fine_image"].nbytes / 1e6
    )

    coarse_mb = (
        X_all_inputs["coarse_pixels"].nbytes / 1e6
    )

    print(
        f"  Generated in "
        f"{time.perf_counter() - t_gen:.1f}s\n"
        f"  fine-image input : {fine_mb:.0f} MB\n"
        f"  coarse input     : {coarse_mb:.1f} MB\n"
        f"  targets          : {y_all.nbytes / 1e6:.1f} MB",
        flush=True,
    )


    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    tf.random.set_seed(SEED)

    model = build_model(n_bins)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR,
            clipnorm=1.0,
        ),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()


    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    model_path = (
        f"model_hybrid_cnn_n{n_bins}.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=0,
        ),

        ProgressCallback(),
    ]


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    t_train_start = time.perf_counter()

    history = model.fit(

        # Two inputs:
        #
        # {
        #     "fine_image":    (N, 4, 48, 1),
        #     "coarse_pixels": (N, 8),
        # }

        X_all_inputs,

        # Spectrum target:
        #
        # (N, n_bins)

        y_all,

        validation_data=(
            X_val_inputs,
            y_val,
        ),

        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    epochs_run = len(
        history.history["loss"]
    )

    t_train = (
        time.perf_counter()
        - t_train_start
    )


    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    best_ep = int(
        np.argmin(
            history.history["val_loss"]
        )
    )

    best_vloss = float(
        history.history["val_loss"][best_ep]
    )

    best_vmae = float(
        history.history["val_mae"][best_ep]
    )

    print(
        f"\n"
        f"  Stopped at epoch {epochs_run} "
        f"(best: {best_ep + 1})"
    )

    print(
        f"  Training wall time : "
        f"{t_train:.1f}s "
        f"({t_train / epochs_run:.2f}s/epoch)"
    )

    print(
        f"  Best val_loss      : "
        f"{best_vloss:.6f}"
    )

    print(
        f"  Best val_MAE       : "
        f"{best_vmae:.6f}",
        flush=True,
    )


    results = {

        "n_bins": n_bins,

        "n_samples": N_SAMPLES,

        "model": "hybrid_cnn_4x48_plus_8",

        "input_fine_shape": [
            4,
            48,
            1,
        ],

        "input_coarse_shape": [
            8,
        ],

        "epochs_trained": epochs_run,

        "best_epoch": best_ep + 1,

        "best_val_loss": best_vloss,

        "best_val_mae": best_vmae,

        "learning_rate": LR,

        "batch_size": BATCH_SIZE,

        "norm_mean": mean.tolist(),

        "norm_std": std.tolist(),

        "val_loss": [
            float(v)
            for v in history.history["val_loss"]
        ],

        "train_loss": [
            float(v)
            for v in history.history["loss"]
        ],

        "val_mae": [
            float(v)
            for v in history.history["val_mae"]
        ],
    }

    return results, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    t_start = time.perf_counter()

    rng = np.random.default_rng(SEED)

    drm = load_drm(XLSX_PATH)

    print(f"Original DRM shape: {drm.shape}")


    # Separate output file so this experiment does not overwrite
    # results from the original 5x48 tiled CNN.

    json_path = (
        "hybrid_cnn_training_results.json"
    )

    all_results: dict = {}

    if os.path.exists(json_path):

        with open(json_path) as f:

            all_results = json.load(f)


    for n_bins in N_VALUES:

        model_path = (
            f"model_hybrid_cnn_n{n_bins}.keras"
        )

        if (
            os.path.exists(model_path)
            and str(n_bins) in all_results
        ):

            print(
                f"\n"
                f"n_bins={n_bins}: "
                f"{model_path} exists "
                f"— skipping training."
            )

            continue


        results, model = train_for_n(
            drm,
            n_bins,
            rng,
        )


        all_results[str(n_bins)] = results


        model.save(model_path)


        with open(json_path, "w") as f:

            json.dump(
                all_results,
                f,
                indent=2,
            )


        print(
            f"Saved {model_path}"
        )


    print(
        f"\nTotal wall time: "
        f"{time.perf_counter() - t_start:.1f}s"
    )

    print(
        f"Saved {json_path}"
    )