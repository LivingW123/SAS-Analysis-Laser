"""
train_pff_ensemble_member_oldprior.py — retrains ONE member of the 5-model
PFF ensemble under the ORIGINAL, uncalibrated a4/a5/a6 priors (README's
"6-param, uncalibrated priors" generation), instead of the recentred priors
the live out/training/pff/model_pff_ensemble_{0-4}.keras were retrained
under.

Why this exists: the live ensemble's a4 (bump centre) is trained to prefer
the physically-expected [10,20] MeV window (data_utils.PFF_PARAM_SAMPLING),
so its real-shot a4 predictions are partly prior-enforced, not purely
data-derived (see CLAUDE.md's "PFF bump-centre (a4) identifiability"
section). To get an a4 reading with that prior removed, this script
monkey-patches PFF_PARAM_SAMPLING/PFF_PARAM_BOUNDS's a4/a5/a6 rows back to
the values the project used before recalibration (recovered from git history
of data_utils.py, commit "6 param" 519f1d7 — the original weights
themselves were never committed, .keras is gitignored, so this is a genuine
retrain, not a checkpoint restore):

  a4: mean=35, std=25, sampling [1,49],  clip bound [1,49]
  a5: mean=1.0, std=0.6, sampling [0.1,5],   clip bound [0.1,5]      (a5*x term)
  a6: mean=400, std=250, sampling [20,1000], clip bound [20,1000]   (a6/x term)

a1/a2/a3 are unchanged from the live ensemble (never touched by the a4/a5/a6
recalibration). Does NOT modify data_utils.py itself, and mutates the arrays
in place (row assignment, not reassignment) so every function in data_utils
that reads the module-level globals — generate_pff_training_data,
normalize_pff_params, etc. — picks up the override; the live ensemble's
files are untouched since this saves under its own "_oldprior" suffix.

Same v2 gated/bounded architecture, 500k samples, patience=25 as every
other ensemble member this session (train_pff_ensemble_member.py) — only
the prior differs, so the comparison isolates that one variable.

Usage
-----
  python -m src.training.pff.train_pff_ensemble_member_oldprior --idx 0
  ... --idx 1, 2, 3, 4

Outputs (per member idx)
-------
  out/training/pff/model_pff_ensemble_oldprior_{idx}.keras
  out/training/pff/pff_training_results_ensemble_oldprior_{idx}.json
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import src.core.data_utils as data_utils
from src.core.data_utils import (
    generate_pff_training_data,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
)
from src.core.pff_model import (
    LOGVAR_MAX,
    PARAM_NAMES,
    PFFMetricsCallbackV2,
    build_model,
    pff_loss_v2,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 500_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 200
BATCH_SIZE    = 64
PATIENCE      = 25
BASE_SEED     = 42
LEARNING_RATE = 2e-4

OUT_DIR = "out/training/pff"

# Original ("6-param, uncalibrated priors") a4/a5/a6 rows, recovered from
# data_utils.py's git history (commit 519f1d7, pre-recalibration).
OLD_SAMPLING_A4_A5_A6 = np.array([
    [35.0,  25.0,  1.0,   49.0],   # a4 — bump centre (MeV)
    [1.0,   0.6,   0.1,    5.0],   # a5 — bump width, high-energy (a5*x) coefficient
    [400.0, 250.0, 20.0, 1000.0],  # a6 — bump width, low-energy (a6/x) coefficient
])
OLD_BOUNDS_A4_A5_A6 = np.array([
    [1.0,   49.0],    # a4
    [0.1,    5.0],    # a5
    [20.0, 1000.0],   # a6
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="Ensemble member index (0-based)")
    args = parser.parse_args()
    idx = args.idx
    seed = BASE_SEED + idx

    print(f"Overriding a4/a5/a6 sampling prior to pre-recalibration values "
          f"(was {data_utils.PFF_PARAM_SAMPLING[3:6].tolist()})")
    data_utils.PFF_PARAM_SAMPLING[3:6] = OLD_SAMPLING_A4_A5_A6
    print(f"Overriding a4/a5/a6 clip bounds to pre-recalibration values "
          f"(was {data_utils.PFF_PARAM_BOUNDS[3:6].tolist()})")
    data_utils.PFF_PARAM_BOUNDS[3:6] = OLD_BOUNDS_A4_A5_A6
    PFF_PARAM_BOUNDS = data_utils.PFF_PARAM_BOUNDS

    os.makedirs(OUT_DIR, exist_ok=True)
    MODEL_PATH   = os.path.join(OUT_DIR, f"model_pff_ensemble_oldprior_{idx}.keras")
    RESULTS_JSON = os.path.join(OUT_DIR, f"pff_training_results_ensemble_oldprior_{idx}.json")

    t_start = time.perf_counter()
    print(f"{'='*70}\n  Ensemble member {idx} [OLD PRIOR]  (seed={seed})\n{'='*70}")

    rng = np.random.default_rng(seed)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}")

    t_gen = time.perf_counter()
    X, y_params = generate_pff_training_data(drm, N_SAMPLES, rng, BUMP_FRACTION)
    t_gen = time.perf_counter() - t_gen
    y_norm = normalize_pff_params(y_params)
    print(f"Generated {N_SAMPLES} samples in {t_gen:.2f}s")
    bump_mask = y_params[:, 2] > 0.0
    print(f"a4 (bump-present only) sampled range: "
          f"[{y_params[bump_mask, 3].min():.1f}, {y_params[bump_mask, 3].max():.1f}] MeV, "
          f"mean={y_params[bump_mask, 3].mean():.1f}")

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

    print(f"\nMember {idx} [OLD PRIOR] stopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
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
        "prior":                     "oldprior_uncalibrated_a4a5a6",
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
    print(f"Member {idx} [OLD PRIOR] total wall time: {time.perf_counter() - t_start:.1f}s")
