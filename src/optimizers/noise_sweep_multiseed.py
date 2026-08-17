"""
Stage 3 of the detector-noise-model search (the methodology writeup this
originally referenced, NOISE_SEARCH_PLAN.md, no longer exists in the repo --
this file and noise_sweep.py are the current source of truth): reduce metric
noise on the top ~5 configs from the stage 1+2 sweeps by training each with 3
seeds and averaging tune/holdout excess, since a single-seed score can
mislead when candidates are this close.
"""
import os
import time
import numpy as np
import pandas as pd

import src.optimizers.noise_sweep as ns

OUT_DIR = "out/optimizers"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATES = [
    dict(noise_gain=2.0, read_frac=0.10, mult_frac=0.0),
    dict(noise_gain=5.0, read_frac=0.10, mult_frac=0.0),
    dict(noise_gain=4.0, read_frac=0.05, mult_frac=0.0),
    dict(noise_gain=0.25, read_frac=0.05, mult_frac=0.03),
    dict(noise_gain=5.0, read_frac=0.10, mult_frac=0.03),
]
SEEDS = [42, 43, 44]
SAMPLES, EPOCHS, NBINS = 30000, 20, 50

drm_b = ns.bin_drm(ns.load_drm(ns.DRM_PATH), NBINS)
shots = ns.load_shots()
floors = ns.nnls_floor(drm_b, shots)
names = sorted(shots)
rs = np.random.default_rng(0).permutation(len(names))
holdout = {names[i] for i in rs[: max(1, len(names) // 3)]}
tune = [n for n in names if n not in holdout]
hold = [n for n in names if n in holdout]

rows = []
for cfg in CANDIDATES:
    tune_scores, hold_scores = [], []
    for seed in SEEDS:
        ns.SEED = seed
        t = time.perf_counter()
        model, mean, std = ns.train_one(drm_b, NBINS, cfg, SAMPLES, EPOCHS)
        resid = {s: ns.cnn_resid(model, mean, std, drm_b, shots[s]) for s in shots}
        excess = {s: max(resid[s] - floors[s], 0.0) for s in shots}
        tsc = float(np.mean([excess[s] for s in tune]))
        hsc = float(np.mean([excess[s] for s in hold]))
        tune_scores.append(tsc)
        hold_scores.append(hsc)
        print(f"cfg={cfg} seed={seed} tune={tsc:.2f} hold={hsc:.2f} ({time.perf_counter()-t:.0f}s)", flush=True)
    row = dict(**cfg,
               tune_mean=round(float(np.mean(tune_scores)), 3),
               tune_std=round(float(np.std(tune_scores)), 3),
               hold_mean=round(float(np.mean(hold_scores)), 3),
               hold_std=round(float(np.std(hold_scores)), 3))
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "noise_search_multiseed.csv"), index=False)
    print(f"  -> avg tune={row['tune_mean']}+-{row['tune_std']}  hold={row['hold_mean']}+-{row['hold_std']}", flush=True)

df = pd.DataFrame(rows).sort_values("tune_mean")
print("\n=== Multi-seed ranking (by mean tune_excess) ===")
print(df.to_string(index=False))
best = df.iloc[0]
print(f"\nRecommended data_utils.py defaults:\n"
      f"  NOISE_GAIN = {best['noise_gain']}\n  READ_FRAC  = {best['read_frac']}\n  MULT_FRAC  = {best['mult_frac']}")
