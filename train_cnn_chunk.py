"""
Resumable chunked trainer for the 5x48 CNN (see train_cnn.py).

Each invocation trains for ~TRAIN_SECONDS on freshly generated PFF data
(streaming-style), evaluates on a fixed validation set, checkpoints, and
exits. State persists in cnn_chunk_state.json so repeated invocations
continue where the last one stopped. Designed for environments with a
hard per-process wall-time cap; on a normal machine just use train_cnn.py.

Usage:  python train_cnn_chunk.py [train_seconds] [n_bins]
"""

import json
import os
import sys
import time

t0 = time.perf_counter()

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from data_utils import bin_drm, generate_spectrum_batch, load_drm, normalize_apply, normalize_fit
from train_cnn import build_model, reshape_to_2d

BUMP_FRACTION  = 0.5
BATCH_SIZE     = 256
SEED           = 42
LR_INIT        = 1e-3
VAL_SAMPLES    = 5_000
STEPS_PER_SLICE = 20            # fit() slice size between time checks
LR_PATIENCE    = 6              # chunks without improvement -> halve LR
MIN_LR         = 1e-5

TRAIN_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
N_BINS         = int(sys.argv[2]) if len(sys.argv) > 2 else 10

STATE_PATH = f"cnn_chunk_state_n{N_BINS}.json" if N_BINS != 10 else "cnn_chunk_state.json"
CKPT_PATH  = f"model_cnn_n{N_BINS}_ckpt.keras"
BEST_PATH  = f"model_cnn_n{N_BINS}.keras"
DRM_CACHE  = f"drm_binned_n{N_BINS}.npy"
XLSX_PATH  = "res/drm/200x200.xlsx"


def main() -> None:
    # --- DRM (cached as npy: xlsx load is slow) ---
    if os.path.exists(DRM_CACHE):
        drm_binned = np.load(DRM_CACHE)
    else:
        drm_binned = bin_drm(load_drm(XLSX_PATH), N_BINS)
        np.save(DRM_CACHE, drm_binned)

    # --- state ---
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        mean = np.array(state["norm_mean"], dtype=np.float32)
        std  = np.array(state["norm_std"],  dtype=np.float32)
        model = tf.keras.models.load_model(CKPT_PATH)  # includes optimizer state
    else:
        rng0 = np.random.default_rng(SEED)
        X_boot, _ = generate_spectrum_batch(drm_binned, 10_000, rng0, BUMP_FRACTION)
        mean, std = normalize_fit(X_boot)
        tf.random.set_seed(SEED)
        model = build_model(N_BINS)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR_INIT, clipnorm=1.0),
                      loss="mse", metrics=["mae"])
        state = {
            "chunks_done": 0,
            "samples_seen": 0,
            "best_val_loss": float("inf"),
            "since_improve": 0,
            "lr": LR_INIT,
            "norm_mean": mean.tolist(),
            "norm_std": std.tolist(),
            "val_loss_hist": [],
            "val_mae_hist": [],
            "samples_hist": [],
        }

    # --- fixed validation set (same seed every chunk) ---
    val_rng = np.random.default_rng(SEED + 1)
    X_val, y_val = generate_spectrum_batch(drm_binned, VAL_SAMPLES, val_rng, BUMP_FRACTION)
    X_val_2d = reshape_to_2d(normalize_apply(X_val, mean, std))
    del X_val

    # --- train on fresh data until the time budget is used ---
    chunk_rng = np.random.default_rng(SEED + 1000 + state["chunks_done"])
    model.optimizer.learning_rate.assign(state["lr"])

    slice_samples = STEPS_PER_SLICE * BATCH_SIZE
    trained = 0
    t_budget_start = time.perf_counter()
    while time.perf_counter() - t_budget_start < TRAIN_SECONDS:
        Xb, yb = generate_spectrum_batch(drm_binned, slice_samples, chunk_rng, BUMP_FRACTION)
        Xb2d = reshape_to_2d(normalize_apply(Xb, mean, std))
        model.fit(Xb2d, yb, epochs=1, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
        trained += slice_samples

    # --- validate ---
    val_loss, val_mae = model.evaluate(X_val_2d, y_val, batch_size=1024, verbose=0)

    state["chunks_done"] += 1
    state["samples_seen"] += trained
    state["val_loss_hist"].append(float(val_loss))
    state["val_mae_hist"].append(float(val_mae))
    state["samples_hist"].append(int(state["samples_seen"]))

    improved = val_loss < state["best_val_loss"]
    if improved:
        state["best_val_loss"] = float(val_loss)
        state["since_improve"] = 0
        model.save(BEST_PATH)
    else:
        state["since_improve"] += 1
        if state["since_improve"] >= LR_PATIENCE and state["lr"] > MIN_LR:
            state["lr"] = max(state["lr"] * 0.5, MIN_LR)
            state["since_improve"] = 0

    model.save(CKPT_PATH)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

    print(f"chunk {state['chunks_done']:3d} | +{trained:,} samples "
          f"(total {state['samples_seen']:,}) | val_loss {val_loss:.3e} "
          f"| best {state['best_val_loss']:.3e}{' *' if improved else ''} "
          f"| lr {state['lr']:.1e} | wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
