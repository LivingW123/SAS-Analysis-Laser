"""
train_pff_v3_realistic_noise.py — same architecture as
train_pff_bounded_gated.py (v2: tanh-bounded logvar + gated bump classifier),
but trained on data_utils.generate_pff_training_data's now-realistic
generator (calibrated add_detector_noise + apply_saturation) instead of the
plain sqrt(response) Poisson noise v1/v2 both used.

Why this file exists: v1 and v2 both showed a real-shot uncertainty problem
that bounding the output activation only partially addressed -- on all 14
real shots tested, v2's sigma (and bump-presence probability) turned out to
be bit-for-bit IDENTICAL regardless of which shot was fed in, even though
the mean predictions varied wildly. That's the model saturating at a fixed
corner of its output range on anything unfamiliar, not genuine calibrated
uncertainty. The root cause traced back one level further: every real shot
checked this session has 0-92/200 channels that are saturation-corrected
(CCD flat-topping), and calibrated noise characteristics -- but the PFF
regressor's training generator never modeled either. The CNN spectrum
classifier (train_cnn.py) already uses a fully realistic, saturation-aware
generator (data_utils.generate_spectrum_batch, calibrated via a documented
staged search against real shots' NNLS floor -- see VALUE_SUMMARY.md). This
file is the isolated test of "does closing that specific, concrete gap
change how real shots look to the PFF regressor" -- architecture held fixed
at v2's, only the training-data generator changes, so any difference from
v2's results is attributable to this one change.

Deliberately reuses train_pff_bounded_gated.py's build_model / pff_loss_v2 /
decode_v2 / PFFMetricsCallbackV2 rather than copying them, since the whole
point is to hold everything except the data generator constant.

Usage
-----
  python train_pff_v3_realistic_noise.py

Outputs
-------
  model_pff_v3.keras
  pff_training_results_v3.json
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_utils import (
    PFF_PARAM_BOUNDS,
    generate_pff_training_data,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
)
from train_pff_bounded_gated import (
    LOGVAR_MAX,
    PARAM_NAMES,
    PFFMetricsCallbackV2,
    build_model,
    pff_loss_v2,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Config -- was identical to train_pff_bounded_gated.py's (20k/300/40) for a
# clean single-variable comparison against v2, but that run showed val_loss
# plateauing almost immediately (EarlyStopping's patience=40 exhausted by
# epoch 44, i.e. last improvement around epoch 4) and every accuracy/
# calibration metric landing worse than v2's -- the realistic noise +
# saturation model makes the per-sample gradient signal noisier, and 20k
# samples / patience=40 wasn't enough budget to average that out. Scaled up
# both knobs for a real second attempt rather than accepting the first
# early-stopped run as final.
XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 60_000   # 3x -- more samples per gradient step averages out
                         # the noisier per-sample loss from the realistic
                         # noise/saturation model
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 500
BATCH_SIZE    = 64
PATIENCE      = 80       # 2x -- give the noisier val_loss signal more room
                         # before declaring a plateau
SEED          = 42
LEARNING_RATE = 2e-4

MODEL_PATH   = "model_pff_v3.keras"
RESULTS_JSON = "pff_training_results_v3.json"


if __name__ == "__main__":
    t_start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}  min={drm.min():.3f}  max={drm.max():.3f}")
    print("Data generator: generate_pff_training_data (calibrated noise + saturation)")

    n_bump = int(N_SAMPLES * BUMP_FRACTION)
    n_no_bump = N_SAMPLES - n_bump
    print(f"Generating {N_SAMPLES} samples ({n_bump} with bump, {n_no_bump} without)...")

    t_gen = time.perf_counter()
    X, y_params = generate_pff_training_data(drm, N_SAMPLES, rng, BUMP_FRACTION)
    t_gen = time.perf_counter() - t_gen
    y_norm = normalize_pff_params(y_params)
    print(f"  Generated in {t_gen:.2f}s")
    print(f"X shape: {X.shape}  y shape: {y_params.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_norm, test_size=0.2, random_state=SEED
    )
    mean, std = normalize_fit(X_train)
    X_train_n = normalize_apply(X_train, mean, std)
    X_val_n = normalize_apply(X_val, mean, std)

    tf.random.set_seed(SEED)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=pff_loss_v2,
    )
    model.summary()

    energy_bins = mev_bin_centers(drm.shape[1])
    metrics_cb = PFFMetricsCallbackV2(X_val_n, y_val, energy_bins)
    spec_ckpt = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="spec_mse", save_best_only=True, mode="min", verbose=0,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=False, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
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
    best_ep = int(np.argmin(metrics_cb.history["spectrum_mse"]))

    print(f"\nStopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train/epochs_run:.2f}s/epoch)")
    print(f"Best val_loss      : {history.history['val_loss'][best_ep]:.6f}")
    print(f"Spectrum MSE       : {metrics_cb.history['spectrum_mse'][best_ep]:.4f}  (at best epoch)")
    print(f"Spectrum rel-MSE   : {metrics_cb.history['spectrum_rel_mse'][best_ep]:.6f}  (bump region, E>5 MeV)")
    print(f"Bump classifier acc: {metrics_cb.history['bump_accuracy'][best_ep]*100:.1f}%")
    print(f"{'Param':>4}  {'MAE (all)':>10}  {'MAE (bump only)':>15}  {'Max sigma':>10}  (at best epoch)")
    for n in PARAM_NAMES:
        print(f"  {n:>2}  {metrics_cb.history[f'mae_{n}'][best_ep]:>10.4f}  "
              f"{metrics_cb.history[f'mae_{n}_bump'][best_ep]:>15.4f}  "
              f"{metrics_cb.history[f'max_sigma_{n}'][best_ep]:>10.2f}")
    print(f"1-sigma coverage   : a1/a2/a3={metrics_cb.history['coverage_1sigma'][best_ep]*100:.0f}% "
          f"(want ~68%)  a4/a5(bump only)={metrics_cb.history['coverage_1sigma_bump'][best_ep]*100:.0f}%")

    model.save(MODEL_PATH)

    results = {
        "epochs_trained":        epochs_run,
        "best_epoch":            best_ep + 1,
        "n_samples":             N_SAMPLES,
        "bump_fraction":         BUMP_FRACTION,
        "learning_rate":         LEARNING_RATE,
        "logvar_max":            LOGVAR_MAX,
        "data_generator":        "generate_pff_training_data (calibrated noise + saturation)",
        "param_bounds":          PFF_PARAM_BOUNDS.tolist(),
        "param_names":           PARAM_NAMES,
        "train_loss":            [float(v) for v in history.history["loss"]],
        "val_loss":              [float(v) for v in history.history["val_loss"]],
        "best_val_loss":         float(history.history["val_loss"][best_ep]),
        "spectrum_mse":          metrics_cb.history["spectrum_mse"],
        "spectrum_rel_mse":      metrics_cb.history["spectrum_rel_mse"],
        "best_spectrum_mse":     metrics_cb.history["spectrum_mse"][best_ep],
        "best_spectrum_rel_mse": metrics_cb.history["spectrum_rel_mse"][best_ep],
        "coverage_1sigma":       metrics_cb.history["coverage_1sigma"],
        "coverage_1sigma_bump":  metrics_cb.history["coverage_1sigma_bump"],
        "best_coverage_1sigma":      metrics_cb.history["coverage_1sigma"][best_ep],
        "best_coverage_1sigma_bump": metrics_cb.history["coverage_1sigma_bump"][best_ep],
        "bump_accuracy":         metrics_cb.history["bump_accuracy"],
        "best_bump_accuracy":    metrics_cb.history["bump_accuracy"][best_ep],
        "norm_mean":             mean.tolist(),
        "norm_std":              std.tolist(),
        **{f"mae_{n}":           metrics_cb.history[f"mae_{n}"]      for n in PARAM_NAMES},
        **{f"mae_{n}_bump":      metrics_cb.history[f"mae_{n}_bump"] for n in PARAM_NAMES},
        **{f"best_mae_{n}":      metrics_cb.history[f"mae_{n}"][best_ep]      for n in PARAM_NAMES},
        **{f"best_mae_{n}_bump": metrics_cb.history[f"mae_{n}_bump"][best_ep] for n in PARAM_NAMES},
        **{f"max_sigma_{n}":     metrics_cb.history[f"max_sigma_{n}"] for n in PARAM_NAMES},
        **{f"best_max_sigma_{n}": metrics_cb.history[f"max_sigma_{n}"][best_ep] for n in PARAM_NAMES},
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {MODEL_PATH} and {RESULTS_JSON}")

    t_total = time.perf_counter() - t_start
    print(f"\n--- Timing summary ---")
    print(f"  Data generation   : {t_gen:.2f}s")
    print(f"  Training          : {t_train:.1f}s  ({epochs_run} epochs, {t_train/epochs_run:.2f}s/epoch)")
    print(f"  Total wall time   : {t_total:.1f}s")
