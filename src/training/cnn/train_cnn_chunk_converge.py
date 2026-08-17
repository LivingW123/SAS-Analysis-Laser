"""
Single-process driver for the resumable chunked CNN trainer (train_cnn_chunk.py)
that loops internally until convergence instead of requiring one external
invocation per chunk. Stops early once the model has plateaued, rather than
running to a fixed sample/time budget.

Saturation augmentation (data_utils.SAT_FRACTION) is picked up automatically
since generate_spectrum_batch defaults to it.

Early-stop rule: stop once the LR has decayed to MIN_LR *and* val_loss hasn't
improved for EARLY_STOP_PATIENCE consecutive chunks (i.e. genuinely plateaued,
not just mid-decay). A wall-clock safety cap guards against runaway loops.

Usage: python -m src.training.cnn.train_cnn_chunk_converge [n_bins] [chunk_seconds]
"""

import json
import os
import sys
import time

t0 = time.perf_counter()

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from src.core.cnn_model import build_model, reshape_to_2d
from src.core.data_utils import bin_drm, generate_spectrum_batch, load_drm, normalize_apply, normalize_fit

BUMP_FRACTION   = 0.5
BATCH_SIZE      = 256
SEED            = 42
LR_INIT         = 1e-3
VAL_SAMPLES     = 5_000
STEPS_PER_SLICE = 20
LR_PATIENCE     = 6
MIN_LR          = 1e-5

EARLY_STOP_PATIENCE = 20     # chunks without improvement, once LR has bottomed out
MAX_TOTAL_SECONDS   = 7_200  # 2h hard safety cap

N_BINS        = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CHUNK_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

OUT_DIR = "out/training/cnn"

STATE_PATH = os.path.join(OUT_DIR, f"cnn_chunk_state_n{N_BINS}.json" if N_BINS != 10 else "cnn_chunk_state.json")
CKPT_PATH  = os.path.join(OUT_DIR, f"model_cnn_n{N_BINS}_ckpt.keras")
BEST_PATH  = os.path.join(OUT_DIR, f"model_cnn_n{N_BINS}.keras")
DRM_CACHE  = os.path.join(OUT_DIR, f"drm_binned_n{N_BINS}.npy")
XLSX_PATH  = "res/drm/200x200.xlsx"
RESULTS_PATH = os.path.join(OUT_DIR, "cnn_training_results.json")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(DRM_CACHE):
        drm_binned = np.load(DRM_CACHE)
    else:
        drm_binned = bin_drm(load_drm(XLSX_PATH), N_BINS)
        np.save(DRM_CACHE, drm_binned)

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        mean = np.array(state["norm_mean"], dtype=np.float32)
        std  = np.array(state["norm_std"],  dtype=np.float32)
        model = tf.keras.models.load_model(CKPT_PATH)
        print(f"Resuming from chunk {state['chunks_done']} "
              f"({state['samples_seen']:,} samples seen, best_val_loss {state['best_val_loss']:.3e})",
              flush=True)
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
        print(f"Starting fresh run for n_bins={N_BINS} (saturation-augmented)", flush=True)

    val_rng = np.random.default_rng(SEED + 1)
    X_val, y_val = generate_spectrum_batch(drm_binned, VAL_SAMPLES, val_rng, BUMP_FRACTION)
    X_val_2d = reshape_to_2d(normalize_apply(X_val, mean, std))
    del X_val

    slice_samples = STEPS_PER_SLICE * BATCH_SIZE
    early_stop_wait = 0

    while True:
        elapsed_total = time.perf_counter() - t0
        if elapsed_total > MAX_TOTAL_SECONDS:
            print(f"\nStopping: hit {MAX_TOTAL_SECONDS}s safety cap "
                  f"(elapsed {elapsed_total:.0f}s).", flush=True)
            break

        chunk_rng = np.random.default_rng(SEED + 1000 + state["chunks_done"])
        model.optimizer.learning_rate.assign(state["lr"])

        trained = 0
        t_budget_start = time.perf_counter()
        while time.perf_counter() - t_budget_start < CHUNK_SECONDS:
            Xb, yb = generate_spectrum_batch(drm_binned, slice_samples, chunk_rng, BUMP_FRACTION)
            Xb2d = reshape_to_2d(normalize_apply(Xb, mean, std))
            model.fit(Xb2d, yb, epochs=1, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
            trained += slice_samples

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
            early_stop_wait = 0
            model.save(BEST_PATH)
        else:
            state["since_improve"] += 1
            if state["since_improve"] >= LR_PATIENCE and state["lr"] > MIN_LR:
                state["lr"] = max(state["lr"] * 0.5, MIN_LR)
                state["since_improve"] = 0
            if state["lr"] <= MIN_LR:
                early_stop_wait += 1

        model.save(CKPT_PATH)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)

        print(f"chunk {state['chunks_done']:3d} | +{trained:,} (total {state['samples_seen']:,}) "
              f"| val_loss {val_loss:.3e} | best {state['best_val_loss']:.3e}{' *' if improved else ''} "
              f"| lr {state['lr']:.1e} | wall {time.perf_counter()-t0:.0f}s", flush=True)

        if early_stop_wait >= EARLY_STOP_PATIENCE:
            print(f"\nStopping: val_loss plateaued for {EARLY_STOP_PATIENCE} chunks "
                  f"at min LR (best_val_loss {state['best_val_loss']:.3e}).", flush=True)
            break

    total_wall = time.perf_counter() - t0
    print(f"\nDone. {state['chunks_done']} chunks, {state['samples_seen']:,} samples, "
          f"best_val_loss {state['best_val_loss']:.3e}, wall {total_wall:.0f}s", flush=True)

    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    all_results[str(N_BINS)] = {
        "n_bins": N_BINS,
        "n_samples": state["samples_seen"],
        "model": "cnn_5x48_chunked_sat",
        "chunks_done": state["chunks_done"],
        "best_val_loss": state["best_val_loss"],
        "best_val_mae": min(state["val_mae_hist"]) if state["val_mae_hist"] else None,
        "learning_rate_final": state["lr"],
        "batch_size": BATCH_SIZE,
        "norm_mean": state["norm_mean"],
        "norm_std": state["norm_std"],
        "val_loss": state["val_loss_hist"],
        "val_mae": state["val_mae_hist"],
        "wall_seconds": total_wall,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Updated {RESULTS_PATH} [{N_BINS}]")


if __name__ == "__main__":
    main()
