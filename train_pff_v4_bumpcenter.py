"""
train_pff_v4_bumpcenter.py — full-scale (500k-sample) single-model retrain
using data_utils.py's recalibrated a4/a5/a6 priors (see the comment block
above PFF_PARAM_SAMPLING there for the full derivation): a4 recentred on the
physically expected [10,20] MeV bump-centre window, a5/a6 tightened to match
matlab/PFF.m's own earlier constant-width calibration of this DRM.

Same v2 gated/bounded architecture as every model this session
(train_pff_bounded_gated.build_model/pff_loss_v2/decode_v2), same 500k
sample budget as train_pff_ensemble_member.py / train_pff_v3's attempt 4
(this session's established best accuracy-per-compute-dollar point) -- but
saved under its own name (not model_pff_ensemble_0.keras) so it doesn't
silently desync the existing 5-member ensemble, whose other 4 members were
trained under the OLD priors.

PATIENCE trimmed from the ensemble script's 80 to 25: every ensemble member
this session hit its best epoch by epoch 1-5 and then had EarlyStopping wait
out up to 80 more largely-wasted epochs (member 3 alone burned 23416s this
way) before stopping -- patience=25 keeps a comfortable margin past that
observed range without repeating that waste.

Usage
-----
  python train_pff_v4_bumpcenter.py

Outputs
-------
  model_pff_v4_bumpcenter.keras
  pff_training_results_v4_bumpcenter.json
  pff_v4_bumpcenter_real_shots.csv  (a4 mean/sigma on the same 14 real shots
                                      used throughout this session)
"""

import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_utils import (
    PFF_PARAM_BOUNDS,
    PFF_PARAM_SAMPLING,
    generate_pff_training_data,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
)
from train_pff_bounded_gated import (
    PARAM_NAMES,
    PFFMetricsCallbackV2,
    build_model,
    decode_v2,
    pff_loss_v2,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 500_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 200
BATCH_SIZE    = 64
PATIENCE      = 25
SEED          = 42
LEARNING_RATE = 2e-4

MODEL_PATH   = "model_pff_v4_bumpcenter.keras"
RESULTS_JSON = "pff_training_results_v4_bumpcenter.json"

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
    print(f"a4 prior: mean={PFF_PARAM_SAMPLING[3,0]}, std={PFF_PARAM_SAMPLING[3,1]}, "
          f"bounds=[{PFF_PARAM_SAMPLING[3,2]},{PFF_PARAM_SAMPLING[3,3]}]")
    print(f"a5 prior: mean={PFF_PARAM_SAMPLING[4,0]}, std={PFF_PARAM_SAMPLING[4,1]}, "
          f"bounds=[{PFF_PARAM_SAMPLING[4,2]},{PFF_PARAM_SAMPLING[4,3]}]")
    print(f"a6 prior: mean={PFF_PARAM_SAMPLING[5,0]}, std={PFF_PARAM_SAMPLING[5,1]}, "
          f"bounds=[{PFF_PARAM_SAMPLING[5,2]},{PFF_PARAM_SAMPLING[5,3]}]")

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}")

    t_gen = time.perf_counter()
    X, y_params = generate_pff_training_data(drm, N_SAMPLES, rng, BUMP_FRACTION)
    t_gen = time.perf_counter() - t_gen
    y_norm = normalize_pff_params(y_params)
    print(f"Generated {N_SAMPLES} samples in {t_gen:.2f}s")

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

    print(f"\nStopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train/epochs_run:.2f}s/epoch)")
    print(f"Bump classifier acc: {metrics_cb.history['bump_accuracy'][best_ep]*100:.1f}%")
    for n in PARAM_NAMES:
        print(f"  {n}: MAE={metrics_cb.history[f'mae_{n}_bump'][best_ep]:.4f}  "
              f"max_sigma={metrics_cb.history[f'max_sigma_{n}'][best_ep]:.4f}")
    print(f"1-sigma coverage   : a1/a2/a3={metrics_cb.history['coverage_1sigma'][best_ep]*100:.0f}% "
          f"a4/a5(bump)={metrics_cb.history['coverage_1sigma_bump'][best_ep]*100:.0f}%  (want ~68%)")

    model.save(MODEL_PATH)

    results = {
        "epochs_trained": epochs_run,
        "best_epoch": best_ep + 1,
        "n_samples": N_SAMPLES,
        "param_sampling": PFF_PARAM_SAMPLING.tolist(),
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

    print(f"\n{'='*80}\nREAL SHOTS -- a4 (bump centre) prediction, v4 (recalibrated priors)\n{'='*80}")
    rows = []
    for name, path in SHOTS:
        if not os.path.exists(path):
            continue
        signal_l1 = l1_normalise(load_signal(path))
        x_norm = normalize_apply(signal_l1.reshape(1, -1), mean, std)
        pff_out = model.predict(x_norm, verbose=0)[0]
        p_phys, p_sigma, p_bump = decode_v2(pff_out, PFF_PARAM_BOUNDS)
        print(f"  {name:>10}: p(bump)={float(p_bump[0]):.3f}  "
              f"a4={p_phys[3]:7.2f} +/- {p_sigma[3]:6.2f} MeV")
        rows.append({"shot": name, "a4_mean": p_phys[3], "a4_sigma": p_sigma[3],
                      "p_bump": float(p_bump[0])})
    pd.DataFrame(rows).to_csv("pff_v4_bumpcenter_real_shots.csv", index=False)
    print("Saved pff_v4_bumpcenter_real_shots.csv")

    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
