"""
train_pff_bounded_gated.py — PFF parameter regressor, v2: two fixes over
train_pff.py's ReLU-mean + heteroscedastic-NLL head (model_pff_relu_uncertainty.keras),
tried side by side rather than replacing it.

Problem this addresses: on real (out-of-distribution) shots, that model's
per-parameter sigma routinely saturated at its clip ceiling (logvar=+-10,
i.e. sigma_norm=exp(5)=148.4x a parameter's own physical range) -- e.g.
a2=+-146.93, a3=+-14841.32 on real shots. Both numbers are literally the same
saturation event, just scaled by each parameter's own (very different) unit
range -- a3's range (0-100) is ~100x a2's (0.01-1), so the identical
saturated head produces a far more dramatic-looking absolute sigma. Fixing
the saturation mechanism therefore fixes a2 and a3 (and a4, which saturated
identically on some shots) together, not one at a time. Two independent
changes, both tried here:

1. Bounded log-variance activation (tanh-scaled) instead of a raw linear
   layer + post-hoc clip, for every continuous head. A hard clip has zero
   gradient past the boundary -- once training pushes a unit to the ceiling,
   nothing pulls it back. tanh keeps a gradient everywhere and caps sigma at
   LOGVAR_MAX regardless of how far an input extrapolates, instead of the
   previous +-10 clip (max sigma_norm = exp(5) ~= 148.4). LOGVAR_MAX=4 caps
   it at exp(2) ~= 7.4 -- a ~20x tighter ceiling.

2. Gated bump structure for a3/a4/a5/a6. a3 (bump amplitude) is bimodal by
   construction (data_utils.sample_pff_spectra: exactly 0 for half the
   training data, ~[5,100] for the rest) -- a single Gaussian NLL head has
   no way to represent "either exactly 0, or somewhere in a wide range" and
   is forced to inflate sigma to hedge between the two regimes even
   in-distribution. This splits it into an explicit binary classifier (bump
   present or not, trained with BCE) plus a magnitude regression for
   a3/a4/a5/a6 conditioned on -- and only supervised on -- bump-present
   samples, combined at decode time via the standard Bernoulli-Gaussian
   mixture mean/variance:
     E[a3]   = p_bump * mu_given_bump
     Var[a3] = p_bump*sigma_given_bump^2 + p_bump*(1-p_bump)*mu_given_bump^2
   a4/a5/a6 (bump centre/energy-dependent-width coefficients) are undefined
   when there's no bump, so they're reported as their given-bump values
   directly rather than mixed with 0.

   [Session note: originally a4/a5 (centre/fixed width); a5 was later
   repurposed and a6 added when data_utils.pff_func's bump denominator
   changed from a constant a5 to an energy-dependent a5*x + a6/x. The gating
   logic here is unaffected by that -- it just grew from 3 to 4
   bump-conditional outputs.]

Entirely separate from train_pff.py / model_pff_relu_uncertainty.keras --
kept side by side to compare, not a replacement. Shares data_utils.py
(PFF_PARAM_BOUNDS, normalize_pff_params, generate_pff_training_data, ...)
but not train_pff.py's pff_nll_loss/build_model/PFFMetricsCallback, since
the output layout here (13-wide, gated) is structurally different from
train_pff.py's (12-wide, mu+logvar per parameter) -- see decode_v2 below
for why a shared decode helper doesn't apply either.

Model architecture (build_model), loss (pff_loss_v2), decode helper
(decode_v2), and the validation-metrics callback (PFFMetricsCallbackV2) live
in src/core/pff_model.py, shared with every other PFF trainer plus
evaluate_pff_ensemble.py and infer_cnn_ensemble.py.

Usage
-----
  python -m src.training.pff.train_pff_bounded_gated

Outputs
-------
  out/training/pff/model_pff_v2.keras
  out/training/pff/pff_training_results_v2.json
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.core.data_utils import (
    PFF_PARAM_BOUNDS,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
    sample_pff_spectra,
)
from src.core.pff_model import (
    LOGVAR_MAX,
    PARAM_NAMES,
    PFFMetricsCallbackV2,
    build_model,
    pff_loss_v2,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Config -- identical to train_pff.py's so the two are comparable apples-to-apples
XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 20_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 300
BATCH_SIZE    = 64
PATIENCE      = 40
SEED          = 42
LEARNING_RATE = 2e-4

OUT_DIR      = "out/training/pff"
MODEL_PATH   = os.path.join(OUT_DIR, "model_pff_v2.keras")
RESULTS_JSON = os.path.join(OUT_DIR, "pff_training_results_v2.json")


def _timed_generate(drm, n_samples, rng, bump_fraction):
    energy_bins = mev_bin_centers(drm.shape[1])

    t0 = time.perf_counter()
    spectra, params = sample_pff_spectra(n_samples, energy_bins, rng, bump_fraction)
    sample_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    responses = (drm @ spectra.T).T
    sigma = np.sqrt(np.maximum(responses, 1e-8))
    noise = rng.standard_normal(responses.shape) * sigma
    X = np.clip(responses + noise, 0.0, None)
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)
    numpy_time = time.perf_counter() - t1

    return X, params.astype(np.float32), sample_time, numpy_time


if __name__ == "__main__":
    t_start = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}  min={drm.min():.3f}  max={drm.max():.3f}")

    n_bump = int(N_SAMPLES * BUMP_FRACTION)
    n_no_bump = N_SAMPLES - n_bump
    print(f"Generating {N_SAMPLES} samples ({n_bump} with bump, {n_no_bump} without)...")

    X, y_params, t_sample, t_np = _timed_generate(drm, N_SAMPLES, rng, BUMP_FRACTION)
    y_norm = normalize_pff_params(y_params)

    print(f"  Sampling loop    : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise): {t_np:.3f}s")
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
    print(f"  Sampling loop     : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise) : {t_np:.3f}s")
    print(f"  Training          : {t_train:.1f}s  ({epochs_run} epochs, {t_train/epochs_run:.2f}s/epoch)")
    print(f"  Total wall time   : {t_total:.1f}s")
