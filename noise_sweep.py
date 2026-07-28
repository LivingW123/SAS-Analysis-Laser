"""
Search for the detector-noise level that makes the CNN generalise best to the
real shots.

Idea
----
We cannot measure spectrum accuracy directly (no ground-truth energy spectra for
real shots), but for each shot the non-negative least-squares (NNLS) fit of the
DRM to the measured vector gives the *best residual any spectrum could achieve*
— a floor set by the DRM/vector forward model, independent of the CNN. A good
noise level is one where the trained CNN's residual sits close to that floor.

For each candidate (noise_gain, read_frac, mult_frac):
  1. generate training data with that noise (data_utils.generate_spectrum_batch)
  2. fit z-score stats + train a CNN for a fixed small budget (seeded, so trials
     are comparable)
  3. score = mean over the TUNE shots of  max(CNN_resid% - NNLS_floor%, 0)
The config with the lowest tune score is reported, and its HOLDOUT score is
printed to check it didn't overfit the tune shots.

Usage
-----
  python noise_sweep.py                       # default grid
  python noise_sweep.py --mode random --trials 40
  python noise_sweep.py --samples 40000 --epochs 25 --nbins 50
Env: CNN_NBINS also honored. Results -> noise_search_results.csv
"""
import argparse, glob, itertools, json, os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import tensorflow as tf

from data_utils import bin_drm, load_drm, generate_spectrum_batch, normalize_fit, normalize_apply, SAT_FRACTION
from train_cnn import build_model, reshape_to_2d

DRM_PATH = "res/drm/200x200.xlsx"
TEST_ROOT = "res/test_images"
SEED = 42


def l1(x):
    s = x.sum()
    return x / s if s > 0 else x


def load_shots():
    """All raw *_proc_vector.csv shots, L1-normalised. (skip _ch/_cv variants)"""
    out = {}
    for p in sorted(glob.glob(f"{TEST_ROOT}/*/*_proc_vector.csv")):
        sig = pd.read_csv(p)["signal"].values.astype(float)
        if len(sig) == 200:
            out[os.path.basename(p).split("_")[0]] = l1(sig)
    return out


def nnls_floor(drm_b, shots):
    """Best achievable residual% per shot (cached)."""
    floors = {}
    for s, sig in shots.items():
        spec, _ = nnls(drm_b, sig)
        resp = l1(drm_b @ spec)
        r = np.sqrt(np.mean((sig - resp) ** 2))
        floors[s] = 100 * r / np.sqrt(np.mean(sig ** 2))
    return floors


def cnn_resid(model, mean, std, drm_b, sig):
    xn = normalize_apply(sig.reshape(1, -1), mean, std)
    spec = model.predict(reshape_to_2d(xn), verbose=0)[0]
    resp = l1(drm_b @ spec)
    r = np.sqrt(np.mean((sig - resp) ** 2))
    return 100 * r / np.sqrt(np.mean(sig ** 2))


def train_one(drm_b, nbins, cfg, samples, epochs):
    rng = np.random.default_rng(SEED)
    Xb, _ = generate_spectrum_batch(drm_b, min(10000, samples), rng, 0.5,
                                    sat_fraction=SAT_FRACTION, **cfg)
    mean, std = normalize_fit(Xb)
    X, y = generate_spectrum_batch(drm_b, samples, rng, 0.5,
                                   sat_fraction=SAT_FRACTION, **cfg)
    X2d = reshape_to_2d(normalize_apply(X, mean, std))
    tf.random.set_seed(SEED)
    model = build_model(nbins)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    model.fit(X2d, y, epochs=epochs, batch_size=256, verbose=0)
    return model, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--trials", type=int, default=30, help="random-mode trial count")
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--nbins", type=int, default=int(os.environ.get("CNN_NBINS", 50)))
    ap.add_argument("--out", default="noise_search_results.csv")
    args = ap.parse_args()

    drm_b = bin_drm(load_drm(DRM_PATH), args.nbins)
    shots = load_shots()
    floors = nnls_floor(drm_b, shots)

    # deterministic tune / holdout split (2/3 tune)
    names = sorted(shots)
    rs = np.random.default_rng(0).permutation(len(names))
    holdout = {names[i] for i in rs[: max(1, len(names) // 3)]}
    tune = [n for n in names if n not in holdout]
    hold = [n for n in names if n in holdout]
    print(f"{len(shots)} shots | tune={len(tune)} holdout={len(hold)} | nbins={args.nbins} "
          f"| samples={args.samples} epochs={args.epochs}")

    # search space
    # Stage 2 refinement: stage 1 (grid, 90 trials) picked gain=4.0/read=0.05/mult=0.0
    # as best, but read_frac=0.05 was the max tested value and dominated the top of
    # the table almost regardless of gain -> extend read upward past that boundary
    # instead of just narrowing around it. Gain narrowed to the 2-6 range that did
    # well; mult narrowed to the 0-0.03 range that won (0.06 rarely placed high).
    gains = [2.0, 3.0, 4.0, 5.0, 6.0]
    reads = [0.03, 0.05, 0.07, 0.1]
    mults = [0.0, 0.03]
    if args.mode == "grid":
        space = list(itertools.product(gains, reads, mults))
    else:
        rng = np.random.default_rng(SEED)
        space = [(float(10 ** rng.uniform(-0.7, 0.95)),
                  float(rng.choice(reads)), float(rng.choice(mults)))
                 for _ in range(args.trials)]

    rows, best = [], None
    for i, (g, rd, mu) in enumerate(space):
        cfg = dict(noise_gain=g, read_frac=rd, mult_frac=mu)
        t = time.perf_counter()
        model, mean, std = train_one(drm_b, args.nbins, cfg, args.samples, args.epochs)
        resid = {s: cnn_resid(model, mean, std, drm_b, shots[s]) for s in shots}
        excess = {s: max(resid[s] - floors[s], 0.0) for s in shots}
        tune_sc = float(np.mean([excess[s] for s in tune]))
        hold_sc = float(np.mean([excess[s] for s in hold]))
        row = dict(gain=g, read_frac=rd, mult_frac=mu,
                   tune_excess=round(tune_sc, 2), holdout_excess=round(hold_sc, 2),
                   tune_resid=round(float(np.mean([resid[s] for s in tune])), 2),
                   secs=round(time.perf_counter() - t, 1))
        rows.append(row)
        if best is None or tune_sc < best["tune_excess"]:
            best = row
        print(f"[{i+1}/{len(space)}] gain={g:<5} read={rd:<5} mult={mu:<4} "
              f"| tune_excess={tune_sc:6.2f} holdout={hold_sc:6.2f} ({row['secs']}s)", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print("\n=== BEST (by tune_excess) ===")
    print(best)
    print(f"\nRecommended data_utils.py defaults:\n"
          f"  NOISE_GAIN = {best['gain']}\n  READ_FRAC  = {best['read_frac']}\n  MULT_FRAC  = {best['mult_frac']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
