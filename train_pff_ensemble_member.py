"""
train_pff_ensemble_member.py — trains ONE member of a 5-model deep ensemble,
the epistemic-uncertainty follow-up to model_pff_v3.keras.

Why this exists: every single-network PFF regressor tried this session
(v1, v2, v3/attempts 1-5) only learns *aleatoric* uncertainty -- "how noisy
is this measurement, assuming my model is basically right" -- via the
heteroscedastic-NLL logvar head. None of them have any mechanism for
*epistemic* uncertainty -- "I've never seen anything like this input, so I
genuinely don't know." That gap is exactly what caused the real-shot
failures this session traced: v1's logvar exploded unboundedly on
unfamiliar inputs, v2/v3 saturate at one fixed corner of their bounded
output range regardless of which real shot is fed in. Scaling training data
(the v3 attempts) helped some real-shot metrics but can't fix this in
principle -- a single network has no way to represent "this looks nothing
like my training distribution" no matter how much of that distribution it's
shown.

Deep ensembles are the standard fix (Lakshminarayanan et al. 2017): train N
independently-initialized copies of the same architecture (different weight
init AND different data draw here, for real diversity, not just different
random seeds feeding the same batch order). On familiar inputs the N models
tend to agree closely. On genuinely unfamiliar inputs, each one extrapolates
differently based on its own idiosyncratic learned features, so they
disagree -- and that disagreement (variance of the N means) is an honest,
directly measurable epistemic-uncertainty signal, combinable with each
member's own aleatoric variance via the standard law-of-total-variance
decomposition (see pff_ensemble_utils.decode_ensemble):
    total_var = mean_over_members(aleatoric_var) + var_over_members(mean)

Architecture/data generator are model_pff_v3.keras's (train_pff_v3_realistic_noise.py,
train_pff_bounded_gated.py's tanh-bounded/gated head): the best-validated
combination this session found, and N_SAMPLES is fixed at attempt 4's 500k
-- this session's best accuracy-per-compute-dollar point (500k matched 1M's
accuracy at 1/5.6th the training time), not attempt 5's 1M (diminishing/
negative returns) or attempt 1-2's smaller budgets (clearly worse).

Usage
-----
  python train_pff_ensemble_member.py --idx 0
  python train_pff_ensemble_member.py --idx 1
  ... --idx 2, 3, 4

Outputs (per member idx)
-------
  model_pff_ensemble_{idx}.keras
  pff_training_results_ensemble_{idx}.json
"""

import argparse
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

XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 500_000   # attempt 4's budget -- best accuracy-per-compute-dollar this session
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 200
BATCH_SIZE    = 64
# PATIENCE trimmed from 80 -> 25 (and MAX_EPOCHS 500 -> 200, reduce_lr patience
# 15 -> 10 below): every member trained this session hit its best epoch by
# epoch 1-5, then EarlyStopping(patience=80) waited out up to 80 more
# essentially-wasted epochs before stopping -- member 3 alone burned 23416s
# this way. train_pff_v4_bumpcenter.py's prototype with patience=25 landed at
# the same best-epoch-1 outcome in 2007s. Matched here so all 5 members are
# trained under identical settings (needed for a fair ensemble comparison).
PATIENCE      = 25
BASE_SEED     = 42
LEARNING_RATE = 2e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="Ensemble member index (0-based)")
    args = parser.parse_args()
    idx = args.idx
    seed = BASE_SEED + idx   # different data draw AND different weight init per member

    MODEL_PATH   = f"model_pff_ensemble_{idx}.keras"
    RESULTS_JSON = f"pff_training_results_ensemble_{idx}.json"

    t_start = time.perf_counter()
    print(f"{'='*70}\n  Ensemble member {idx}  (seed={seed})\n{'='*70}")

    rng = np.random.default_rng(seed)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}")

    t_gen = time.perf_counter()
    X, y_params = generate_pff_training_data(drm, N_SAMPLES, rng, BUMP_FRACTION)
    t_gen = time.perf_counter() - t_gen
    y_norm = normalize_pff_params(y_params)
    print(f"Generated {N_SAMPLES} samples in {t_gen:.2f}s")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_norm, test_size=0.2, random_state=seed
    )
    mean, std = normalize_fit(X_train)
    X_train_n = normalize_apply(X_train, mean, std)
    X_val_n = normalize_apply(X_val, mean, std)

    tf.random.set_seed(seed)
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
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0
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

    print(f"\nMember {idx} stopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train/epochs_run:.2f}s/epoch)")
    print(f"Spectrum MSE       : {metrics_cb.history['spectrum_mse'][best_ep]:.4f}")
    print(f"Bump classifier acc: {metrics_cb.history['bump_accuracy'][best_ep]*100:.1f}%")
    print(f"1-sigma coverage   : a1/a2/a3={metrics_cb.history['coverage_1sigma'][best_ep]*100:.0f}%  "
          f"a4/a5(bump)={metrics_cb.history['coverage_1sigma_bump'][best_ep]*100:.0f}%")
    for n in PARAM_NAMES:
        print(f"  {n}: MAE={metrics_cb.history[f'mae_{n}'][best_ep]:.4f}")

    model.save(MODEL_PATH)

    results = {
        "member_idx":                idx,
        "seed":                      seed,
        "epochs_trained":            epochs_run,
        "best_epoch":                best_ep + 1,
        "n_samples":                 N_SAMPLES,
        "logvar_max":                LOGVAR_MAX,
        "param_bounds":              PFF_PARAM_BOUNDS.tolist(),
        "param_names":               PARAM_NAMES,
        "best_spectrum_mse":         metrics_cb.history["spectrum_mse"][best_ep],
        "best_spectrum_rel_mse":     metrics_cb.history["spectrum_rel_mse"][best_ep],
        "best_coverage_1sigma":     metrics_cb.history["coverage_1sigma"][best_ep],
        "best_coverage_1sigma_bump": metrics_cb.history["coverage_1sigma_bump"][best_ep],
        "best_bump_accuracy":       metrics_cb.history["bump_accuracy"][best_ep],
        "norm_mean":                 mean.tolist(),
        "norm_std":                  std.tolist(),
        **{f"best_mae_{n}":       metrics_cb.history[f"mae_{n}"][best_ep]      for n in PARAM_NAMES},
        **{f"best_mae_{n}_bump":  metrics_cb.history[f"mae_{n}_bump"][best_ep] for n in PARAM_NAMES},
        **{f"best_max_sigma_{n}": metrics_cb.history[f"max_sigma_{n}"][best_ep] for n in PARAM_NAMES},
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {MODEL_PATH} and {RESULTS_JSON}")
    print(f"Member {idx} total wall time: {time.perf_counter() - t_start:.1f}s")
