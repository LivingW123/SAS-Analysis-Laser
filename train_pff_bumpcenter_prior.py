"""
train_pff_bumpcenter_prior.py — diagnostic + prototype for the bump-centre
(a4) accuracy problem: real-shot a4 predictions from the existing 5-member
500k-sample ensemble (see evaluate_pff_ensemble.py / pff_ensemble_eval.csv)
range 1.0-43.7 MeV with sigma_total 6-13 MeV, even though a4 is physically
expected to fall between 10 and 20 MeV on real shots. The current training
prior (data_utils.PFF_PARAM_SAMPLING row a4) samples mean=35, std=25 over
[1,49] -- barely overlapping the physically expected range -- which is the
prime suspect.

This script monkey-patches PFF_PARAM_SAMPLING's a4 row to a tight prior
centred on [10,20] (mean=15, std=3, sampling bounds [7,23] to avoid a hard
wall right at the edges we care about) and trains ONE quick model (v2
gated/bounded architecture from train_pff_bounded_gated.py, same as every
ensemble member) at a reduced sample count (100k vs the ensemble's 500k) so
this finishes in minutes, not hours, as a first read on:
  1. Does the network's a4 MAE/coverage improve within [10,20] once the
     training distribution actually lives there (identifiability ceiling)?
  2. Do real-shot a4 predictions move into [10,20] with tighter sigma?

Does NOT modify data_utils.py itself -- the recentred prior lives in this
script's PFF_A4_SAMPLING override so the diagnostic run and the existing
500k-sample ensemble baseline stay independently reproducible. If this looks
like a real improvement, the override gets promoted into data_utils.py
permanently and a full-scale ensemble retrain follows.

Usage
-----
  python train_pff_bumpcenter_prior.py

Outputs
-------
  model_pff_bumpcenter_proto.keras
  pff_training_results_bumpcenter_proto.json
  Console: synthetic MAE/coverage for a4 (val set, all in [10,20] by
           construction) + real-shot a4 mean/sigma for the same 14 shots
           evaluate_pff_ensemble.py uses, for direct before/after comparison.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import data_utils
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
    decode_v2,
    pff_loss_v2,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 100_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 300
BATCH_SIZE    = 64
PATIENCE      = 40
SEED          = 42
LEARNING_RATE = 2e-4

MODEL_PATH   = "model_pff_bumpcenter_proto.keras"
RESULTS_JSON = "pff_training_results_bumpcenter_proto.json"

# a4 override: mean=15, std=3, sampling bounds [7,23] -- most mass lands in
# [10,20] (user's physical prior) with a little headroom either side so real
# shots landing just outside [10,20] aren't fought by a hard wall.
A4_MEAN, A4_STD, A4_LO, A4_HI = 15.0, 3.0, 7.0, 23.0

SHOTS = []
for shot in ["10084", "11696", "11705", "11707", "11716", "11733", "11698"]:
    for suffix in ("_cv", "_ch"):
        SHOTS.append((f"{shot}{suffix}", f"res/test_images/{shot}/{shot}_proc_vector{suffix}.csv"))


def l1_normalise(x: np.ndarray) -> np.ndarray:
    total = x.sum()
    return (x / total).astype(np.float32) if total > 0 else x


def load_signal(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    sig = df[df.columns[-1]].values.astype(np.float32)
    assert len(sig) == 200, f"Expected 200 channels, got {len(sig)}"
    return sig


if __name__ == "__main__":
    t_start = time.perf_counter()

    print(f"Overriding a4 sampling prior: mean={A4_MEAN}, std={A4_STD}, "
          f"bounds=[{A4_LO},{A4_HI}]  (was {data_utils.PFF_PARAM_SAMPLING[3].tolist()})")
    data_utils.PFF_PARAM_SAMPLING[3] = [A4_MEAN, A4_STD, A4_LO, A4_HI]

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}")

    t_gen = time.perf_counter()
    X, y_params = generate_pff_training_data(drm, N_SAMPLES, rng, BUMP_FRACTION)
    t_gen = time.perf_counter() - t_gen
    y_norm = normalize_pff_params(y_params)
    print(f"Generated {N_SAMPLES} samples in {t_gen:.2f}s")
    bump_mask_all = y_params[:, 2] > 0.0
    print(f"a4 (bump-present only) sampled range: "
          f"[{y_params[bump_mask_all, 3].min():.1f}, {y_params[bump_mask_all, 3].max():.1f}] MeV, "
          f"mean={y_params[bump_mask_all, 3].mean():.1f}")

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
    print(f"a4 MAE (bump only) : {metrics_cb.history['mae_a4_bump'][best_ep]:.3f} MeV")
    print(f"a4 max sigma       : {metrics_cb.history['max_sigma_a4'][best_ep]:.3f} MeV")
    print(f"1-sigma coverage   : a1/a2/a3={metrics_cb.history['coverage_1sigma'][best_ep]*100:.0f}% "
          f"a4/a5(bump)={metrics_cb.history['coverage_1sigma_bump'][best_ep]*100:.0f}%  (want ~68%)")

    model.save(MODEL_PATH)

    results = {
        "epochs_trained": epochs_run,
        "best_epoch": best_ep + 1,
        "n_samples": N_SAMPLES,
        "a4_prior_override": [A4_MEAN, A4_STD, A4_LO, A4_HI],
        "param_bounds": PFF_PARAM_BOUNDS.tolist(),
        "param_names": PARAM_NAMES,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        **{f"best_mae_{n}": metrics_cb.history[f"mae_{n}"][best_ep] for n in PARAM_NAMES},
        **{f"best_mae_{n}_bump": metrics_cb.history[f"mae_{n}_bump"][best_ep] for n in PARAM_NAMES},
        **{f"best_max_sigma_{n}": metrics_cb.history[f"max_sigma_{n}"][best_ep] for n in PARAM_NAMES},
        "best_coverage_1sigma": metrics_cb.history["coverage_1sigma"][best_ep],
        "best_coverage_1sigma_bump": metrics_cb.history["coverage_1sigma_bump"][best_ep],
        "best_bump_accuracy": metrics_cb.history["bump_accuracy"][best_ep],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {MODEL_PATH} and {RESULTS_JSON}")

    # --- real shots, same 14 files evaluate_pff_ensemble.py uses ---
    print(f"\n{'='*80}\nREAL SHOTS -- a4 (bump centre) prediction, recentred-prior prototype\n{'='*80}")
    param_bounds = PFF_PARAM_BOUNDS
    rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            continue
        signal_l1 = l1_normalise(load_signal(path))
        x_norm = normalize_apply(signal_l1.reshape(1, -1), mean, std)
        pff_out = model.predict(x_norm, verbose=0)[0]
        p_phys, p_sigma, p_bump = decode_v2(pff_out, param_bounds)
        print(f"  {name:>10}: p(bump)={float(p_bump[0]):.3f}  "
              f"a4={p_phys[3]:7.2f} +/- {p_sigma[3]:6.2f} MeV")
        rows.append({"shot": name, "a4_mean": p_phys[3], "a4_sigma": p_sigma[3],
                      "p_bump": float(p_bump[0])})
    pd.DataFrame(rows).to_csv("pff_bumpcenter_proto_real_shots.csv", index=False)
    print("Saved pff_bumpcenter_proto_real_shots.csv")

    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
