"""
TensorFlow FC regressor for full PFF spectral parameter estimation.

Input  : 200-channel z-score-normalized detector response
Output : 5 PFF parameters [a1, a2, a3, a4, a5] normalized to [0, 1]
Loss   : weighted MSE — a4 and a5 terms are multiplied by normalized a3,
         so bump-position/width contribute nothing to the loss when a3=0.
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_utils import (
    PFF_PARAM_BOUNDS,
    denormalize_pff_params,
    generate_pff_training_data,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
    pff_func,
)

# Config
XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 20_000
BUMP_FRACTION = 0.5      # half the data has a real Gaussian bump
MAX_EPOCHS    = 300
BATCH_SIZE    = 64
PATIENCE      = 40
SEED          = 42
LEARNING_RATE = 2e-4     # 5e-4 still oscillated; clipnorm added below

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5"]


def masked_pff_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    MSE on all 5 parameters; a4/a5 weighted by normalised a3.

    Continuous weighting (a3_norm in [0,1]) rather than a binary mask:
    - When a3=0 (no bump), a4/a5 contribute 0 gradient — correct, they're undefined.
    - When bump is small (a3_norm≈0.1), a4/a5 contribute 10% — proportional to
      how much bump position/width actually affects the spectrum, which is the right
      inductive bias.  Binary mask (0 or 1) was tried and degraded spec MSE ~2×.
    """
    sq      = tf.square(y_true - y_pred)                   # (B, 5)
    a3_w    = y_true[:, 2:3]                               # (B, 1), in [0,1]
    weights = tf.concat(
        [tf.ones_like(sq[:, :3]), tf.repeat(a3_w, 2, axis=1)],
        axis=1,
    )                                                       # (B, 5)
    return tf.reduce_mean(sq * weights)


def build_model() -> tf.keras.Model:
    """200 → 512 → 256 → 128 → 5 (sigmoid) FC network with BatchNorm + ReLU."""
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
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="pff_params")(x)
    return tf.keras.Model(inp, out, name="pff_regressor")


def _pff_batch(energy_bins: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Vectorised PFF evaluation. params shape (N, 5), returns (N, len(energy_bins))."""
    a1, a2, a3, a4, a5 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    x = energy_bins[np.newaxis, :]           # (1, M)
    brems  = a1[:, None] * np.exp(-a2[:, None] * x)
    bump   = a3[:, None] * np.exp(-(x - a4[:, None]) ** 2 / a5[:, None])
    return brems + bump                      # (N, M)


class PFFMetricsCallback(tf.keras.callbacks.Callback):
    """
    Per-parameter MAE (split bump / no-bump) and spectrum metrics each epoch.

    Metrics logged:
      mae_{param}       — MAE over all val samples
      mae_{param}_bump  — MAE restricted to bump samples (a3 > 0)
      spectrum_mse      — absolute MSE of reconstructed PFF curve
      spectrum_rel_mse  — MSE normalised by true spectrum² (catches bump-region errors)
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val_norm: np.ndarray,
        energy_bins: np.ndarray,
    ) -> None:
        super().__init__()
        self.X_val       = X_val
        self.y_val_norm  = y_val_norm
        self.energy_bins = energy_bins
        # pre-compute bump mask from true a3 (normalized a3 > 0 means bump present)
        self.bump_mask   = y_val_norm[:, 2] > 0.0
        self.history: dict[str, list] = (
            {f"mae_{n}": []      for n in PARAM_NAMES}
            | {f"mae_{n}_bump": [] for n in PARAM_NAMES}
            | {"spectrum_mse": [], "spectrum_rel_mse": []}
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        y_pred_norm = self.model.predict(self.X_val, verbose=0, batch_size=256)
        y_pred = denormalize_pff_params(y_pred_norm)
        y_true = denormalize_pff_params(self.y_val_norm)
        bmask  = self.bump_mask

        for j, name in enumerate(PARAM_NAMES):
            abs_err = np.abs(y_pred[:, j] - y_true[:, j])
            self.history[f"mae_{name}"].append(float(abs_err.mean()))
            if bmask.any():
                self.history[f"mae_{name}_bump"].append(float(abs_err[bmask].mean()))
            else:
                self.history[f"mae_{name}_bump"].append(float("nan"))

        # Vectorised spectrum reconstruction
        s_true = _pff_batch(self.energy_bins, y_true)   # (N, M)
        s_pred = _pff_batch(self.energy_bins, y_pred)
        sq_err = (s_true - s_pred) ** 2
        self.history["spectrum_mse"].append(float(sq_err.mean()))
        # Relative MSE restricted to bump-region bins (E > 5 MeV) with floor=1.0
        # avoids bremsstrahlung-tail zeros blowing up the denominator
        bump_bins = self.energy_bins > 5.0
        s_true_b  = s_true[:, bump_bins]
        sq_err_b  = sq_err[:, bump_bins]
        rel_sq    = sq_err_b / np.maximum(s_true_b, 1.0) ** 2
        self.history["spectrum_rel_mse"].append(float(rel_sq.mean()))

        # Expose spec_mse to Keras logs so EarlyStopping can monitor it
        if logs is not None:
            logs["spec_mse"] = self.history["spectrum_mse"][-1]

        if (epoch + 1) % 10 == 0 and logs:
            mae_all  = " | ".join(f"{n}={self.history[f'mae_{n}'][-1]:.3f}"      for n in PARAM_NAMES)
            mae_bump = " | ".join(f"{n}={self.history[f'mae_{n}_bump'][-1]:.3f}" for n in PARAM_NAMES)
            print(
                f"  ep {epoch+1:3d} | val_loss {logs.get('val_loss', 0):.5f} | "
                f"spec_mse {self.history['spectrum_mse'][-1]:.2f} | "
                f"rel_mse {self.history['spectrum_rel_mse'][-1]:.4f}\n"
                f"           all : {mae_all}\n"
                f"           bump: {mae_bump}"
            )


def _timed_generate(drm, n_samples, rng, bump_fraction):
    """
    generate_pff_training_data with split timing:
      - sample_time : rejection-sampling loop (Python-dominated)
      - numpy_time  : DRM matmul + noise (pure NumPy)
    """
    from data_utils import mev_bin_centers, sample_pff_spectra

    energy_bins = mev_bin_centers(drm.shape[1])

    t0 = time.perf_counter()
    spectra, params = sample_pff_spectra(n_samples, energy_bins, rng, bump_fraction)
    sample_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    responses = (drm @ spectra.T).T
    sigma     = np.sqrt(np.maximum(responses, 1e-8))
    noise     = rng.standard_normal(responses.shape) * sigma
    X         = np.clip(responses + noise, 0.0, None).astype(np.float32)
    numpy_time = time.perf_counter() - t1

    return X, params.astype(np.float32), sample_time, numpy_time


if __name__ == "__main__":
    t_start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}  min={drm.min():.3f}  max={drm.max():.3f}")

    n_bump    = int(N_SAMPLES * BUMP_FRACTION)
    n_no_bump = N_SAMPLES - n_bump
    print(f"Generating {N_SAMPLES} samples ({n_bump} with bump, {n_no_bump} without)...")

    X, y_params, t_sample, t_np = _timed_generate(drm, N_SAMPLES, rng, BUMP_FRACTION)
    y_norm = normalize_pff_params(y_params)

    print(f"  Sampling loop  : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise): {t_np:.3f}s")
    print(f"X shape: {X.shape}  y shape: {y_params.shape}")

    X_train, X_val, y_train, y_val, yp_train, yp_val = train_test_split(
        X, y_norm, y_params, test_size=0.2, random_state=SEED
    )
    mean, std  = normalize_fit(X_train)
    X_train_n  = normalize_apply(X_train, mean, std)
    X_val_n    = normalize_apply(X_val,   mean, std)

    tf.random.set_seed(SEED)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=masked_pff_loss,
    )
    model.summary()

    energy_bins = mev_bin_centers(drm.shape[1])
    metrics_cb  = PFFMetricsCallback(X_val_n, y_val, energy_bins)
    # EarlyStopping on val_loss — smooth signal for convergence detection.
    # ModelCheckpoint on spec_mse — saves weights only when spectrum reconstruction
    # improves, decoupling "when to stop" from "which weights to keep".
    spec_ckpt  = tf.keras.callbacks.ModelCheckpoint(
        "model_pff.keras", monitor="spec_mse", save_best_only=True,
        mode="min", verbose=0,
    )
    early_stop  = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=False, verbose=1
    )
    reduce_lr   = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
    )

    t_train_start = time.perf_counter()
    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[metrics_cb, spec_ckpt, early_stop, reduce_lr],
        verbose=0,
    )
    t_train = time.perf_counter() - t_train_start

    epochs_run = len(history.history["loss"])
    best_ep    = int(np.argmin(metrics_cb.history["spectrum_mse"]))  # 0-indexed

    print(f"\nStopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train/epochs_run:.2f}s/epoch)")
    print(f"Best val_loss      : {history.history['val_loss'][best_ep]:.6f}")
    print(f"Spectrum MSE       : {metrics_cb.history['spectrum_mse'][best_ep]:.4f}  (at best epoch)")
    print(f"Spectrum rel-MSE   : {metrics_cb.history['spectrum_rel_mse'][best_ep]:.6f}  (bump region, E>5 MeV)")
    print(f"{'Param':>4}  {'MAE (all)':>10}  {'MAE (bump only)':>15}  (at best epoch)")
    for n in PARAM_NAMES:
        print(f"  {n:>2}  {metrics_cb.history[f'mae_{n}'][best_ep]:>10.4f}  "
              f"{metrics_cb.history[f'mae_{n}_bump'][best_ep]:>15.4f}")

    model.save("model_pff.keras")

    results = {
        "epochs_trained":        epochs_run,
        "best_epoch":            best_ep + 1,
        "n_samples":             N_SAMPLES,
        "bump_fraction":         BUMP_FRACTION,
        "learning_rate":         LEARNING_RATE,
        "param_bounds":          PFF_PARAM_BOUNDS.tolist(),
        "train_loss":            [float(v) for v in history.history["loss"]],
        "val_loss":              [float(v) for v in history.history["val_loss"]],
        "best_val_loss":         float(history.history["val_loss"][best_ep]),
        "spectrum_mse":          metrics_cb.history["spectrum_mse"],
        "spectrum_rel_mse":      metrics_cb.history["spectrum_rel_mse"],
        "best_spectrum_mse":     metrics_cb.history["spectrum_mse"][best_ep],
        "best_spectrum_rel_mse": metrics_cb.history["spectrum_rel_mse"][best_ep],
        "norm_mean":             mean.tolist(),
        "norm_std":              std.tolist(),
        **{f"mae_{n}":           metrics_cb.history[f"mae_{n}"]      for n in PARAM_NAMES},
        **{f"mae_{n}_bump":      metrics_cb.history[f"mae_{n}_bump"] for n in PARAM_NAMES},
        **{f"best_mae_{n}":      metrics_cb.history[f"mae_{n}"][best_ep]      for n in PARAM_NAMES},
        **{f"best_mae_{n}_bump": metrics_cb.history[f"mae_{n}_bump"][best_ep] for n in PARAM_NAMES},
    }
    with open("pff_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved model_pff.keras and pff_training_results.json")

    t_total = time.perf_counter() - t_start
    print(f"\n--- Timing summary ---")
    print(f"  Sampling loop     : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise) : {t_np:.3f}s")
    print(f"  TF init + compile : (included in training time)")
    print(f"  Training          : {t_train:.1f}s  ({epochs_run} epochs, {t_train/epochs_run:.2f}s/epoch)")
    print(f"  Total wall time   : {t_total:.1f}s")
