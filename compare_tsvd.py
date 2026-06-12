"""
Side-by-side comparison: TSVD-NN (MATLAB algorithm reimplemented in Python)
vs h_sparse neural network on the held-out test set.

TSVD-NN algorithm (mirrors TSVD_NN.m):
  1. SVD of DRM (200x200, full resolution — gp_sz=1 vs MATLAB's gp_sz=2)
  2. Pure TSVD: reconstruct using first 7 singular vectors
  3. TSVD-NN: refine TSVD solution by optimising 7 SVD coefficients so that
     DRM @ positive_def(V7 @ c) minimises residual vs measured b

Outputs:
  comparison_tsvd.png  — metrics table + sample spectra + forward-consistency plots
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "old_experiments"

# DRM 
DRM = np.load("DRM.npy").astype(np.float64)   # (200, 200) rows=det, cols=energy

# SVD (computed once; expensive for 200x200 but only done once) ──────────────
print("Computing SVD of DRM...")
U, S, Vt = np.linalg.svd(DRM, full_matrices=False)
V = Vt.T           # (200, 200): columns are right singular vectors
num_terms = 7
U7 = U[:, :num_terms]      # (200, 7)
S7 = S[:num_terms]         # (7,)
V7 = V[:, :num_terms]      # (200, 7)
# MATLAB: constraint = S(6,6) → 0-indexed = S[5]
# term 7 (i=6) uses this clamped denominator to avoid blow-up from tiny singular value
constraint = S[5]
print(f"  Condition number: {S[0]/S[-1]:.3e}  |  "
      f"S[0..6]: {np.array2string(S7, precision=3, separator=', ')}")

# TSVD and TSVD-NN solvers
def tsvd_solve(b):
    """Pure truncated SVD — no nonlinear refinement."""
    result = np.zeros(200)
    for i in range(num_terms):
        denom = S7[i] if i < 6 else constraint
        result += V7[:, i] * (U7[:, i] @ b / denom)
    return np.maximum(result, 0.0)

def tsvd_nn_solve(b, c0):
    """Nonlinear refinement of TSVD: optimise 7 SVD coefficients."""
    def residuals(c):
        return DRM @ np.maximum(V7 @ c, 0.0) - b
    res = least_squares(residuals, c0, method="lm", max_nfev=10_000, ftol=1e-9)
    return np.maximum(V7 @ res.x, 0.0)

# h_sparse neural network 
def build_model():
    layers, d = [], 200
    for h in [512, 512, 256]:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.05)]
        d = h
    layers += [nn.Linear(d, 200), nn.Softplus(beta=5.0)]
    return nn.Sequential(*layers)

model = build_model()
model.load_state_dict(torch.load("fc_model_best.pt", map_location="cpu"))
model.eval()

stats   = np.load("norm_best.npy")
X_mean, X_std = stats[0], stats[1]
y_scale = float(np.load("yscale_best.npy").flat[0])

def predict_nn_batch(X_raw):
    Xl = (np.log1p(X_raw) - X_mean) / X_std
    with torch.no_grad():
        out = model(torch.from_numpy(Xl.astype(np.float32)))
    return out.numpy() * y_scale

# test data 
Xt = np.load(f"{DATA_DIR}/test_X.npy").astype(np.float64)
yt = np.load(f"{DATA_DIR}/test_y.npy").astype(np.float64)
N  = len(Xt)
print(f"Test set: {N} samples")

# run h_sparse NN (batch) 
print("Running h_sparse NN...")
t0 = time.time()
nn_preds = predict_nn_batch(Xt)
print(f"  done in {time.time()-t0:.1f}s")

# run pure TSVD (fast)
print("Running TSVD...")
t0 = time.time()
tsvd_preds = np.array([tsvd_solve(Xt[i]) for i in range(N)])
print(f"  done in {time.time()-t0:.1f}s")

# run TSVD-NN (nonlinear; subset for speed if large)
N_NN = min(N, 500)     # full set if ≤500; otherwise first 500
print(f"Running TSVD-NN on {N_NN} samples (nonlinear opt)...")
t0 = time.time()
tsvd_nn_preds = np.zeros((N_NN, 200))
for i in range(N_NN):
    c0 = V7.T @ tsvd_preds[i]   # warm-start from pure TSVD
    tsvd_nn_preds[i] = tsvd_nn_solve(Xt[i], c0)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_NN}  ({time.time()-t0:.0f}s elapsed)")
print(f"  done in {time.time()-t0:.1f}s")

# metrics helper
def metrics(preds, trues, b_raw, label):
    active = trues > 0
    fwd_pred  = preds @ DRM.T                        # (N, 200) projected
    fwd_true  = b_raw                                # measured signal
    fwd_relL2 = np.linalg.norm(fwd_pred - fwd_true, axis=1) / (
                np.linalg.norm(fwd_true, axis=1) + 1e-12)
    flux_ratio = fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)
    mae_act = float(np.abs(preds[active] - trues[active]).mean()) if active.any() else float("nan")
    mae_zer = float(np.abs(preds[~active]).mean())
    recall  = float(((preds > 0.5) & active).sum() / active.sum()) if active.any() else float("nan")
    prec    = float(((preds > 0.5) & active).sum() / max((preds > 0.5).sum(), 1))
    print(f"\n[{label}] n={len(preds)}")
    print(f"  fwd relL2   : mean={fwd_relL2.mean():.4f}  med={np.median(fwd_relL2):.4f}  "
          f"p90={np.percentile(fwd_relL2, 90):.4f}")
    print(f"  flux ratio  : mean={flux_ratio.mean():.3f}  med={np.median(flux_ratio):.3f}")
    print(f"  MAE active  : {mae_act:.3f}   MAE zero: {mae_zer:.3f}")
    print(f"  Recall(>0.5): {recall:.2%}   Precision: {prec:.2%}")
    return fwd_relL2, flux_ratio

frl_nn,   fxr_nn   = metrics(nn_preds,          yt,          Xt, "h_sparse NN  (all)")
frl_tsvd, fxr_tsvd = metrics(tsvd_preds,         yt,          Xt, "TSVD         (all)")
frl_tnn,  fxr_tnn  = metrics(tsvd_nn_preds,      yt[:N_NN],   Xt[:N_NN], "TSVD-NN      (first 500)")

# figure
fig = plt.figure(figsize=(17, 12))
fig.suptitle("TSVD-NN vs h_sparse NN — held-out test set", fontsize=12)

# row 1: forward relL2 CDF
ax1 = fig.add_subplot(3, 4, 1)
for vals, lbl, col in [(frl_nn, "h_sparse NN", "C0"), (frl_tsvd, "TSVD", "C1"),
                        (frl_tnn, "TSVD-NN", "C2")]:
    x = np.sort(vals); y = np.linspace(0, 1, len(x))
    ax1.plot(x, y, label=lbl, color=col)
ax1.set_xlabel("Forward relL2"); ax1.set_ylabel("CDF"); ax1.legend(fontsize=7)
ax1.set_title("Forward relL2 CDF (lower=better)")
ax1.set_xlim(0, min(2, max(frl_tsvd.max(), frl_nn.max()) * 1.05))

# row 1: flux ratio distribution
ax2 = fig.add_subplot(3, 4, 2)
for vals, lbl, col in [(fxr_nn, "h_sparse NN", "C0"), (fxr_tsvd, "TSVD", "C1"),
                        (fxr_tnn, "TSVD-NN", "C2")]:
    ax2.hist(np.clip(vals, 0, 3), bins=40, alpha=0.5, label=lbl, color=col, density=True)
ax2.axvline(1.0, color="k", lw=1, ls="--", label="ideal")
ax2.set_xlabel("Flux ratio (DRM@pred / b)"); ax2.set_ylabel("Density")
ax2.set_title("Flux ratio (1.0 = perfect)")
ax2.legend(fontsize=7)

# rows 1-2: 6 sample spectra showing all three predictions
rng = np.random.default_rng(42)
idx6 = rng.choice(N_NN, 6, replace=False)
for col_i, idx in enumerate(idx6):
    ax = fig.add_subplot(3, 4, col_i + 3 if col_i < 2 else col_i + 5)
    ax.stem(yt[idx],             markerfmt="C0.", linefmt="C0-",  basefmt="k-", label="True")
    ax.stem(nn_preds[idx],       markerfmt="C1x", linefmt="C1--", basefmt="k-", label="h_sparse")
    ax.stem(tsvd_preds[idx],     markerfmt="C2+", linefmt="C2:",  basefmt="k-", label="TSVD")
    ax.stem(tsvd_nn_preds[idx],  markerfmt="C3^", linefmt="C3-.", basefmt="k-", label="TSVD-NN")
    frl_s  = float(np.linalg.norm(DRM @ nn_preds[idx]      - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tv = float(np.linalg.norm(DRM @ tsvd_preds[idx]    - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tn = float(np.linalg.norm(DRM @ tsvd_nn_preds[idx] - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    ax.set_title(f"Sample {idx}\nfrlL2: NN={frl_s:.3f} TSVD={frl_tv:.3f} TSVD-NN={frl_tn:.3f}",
                 fontsize=7)
    ax.set_xlabel("Energy bin", fontsize=7); ax.set_ylabel("Weight", fontsize=7)
    if col_i == 0: ax.legend(fontsize=6)

# row 3: bar chart summary of key metrics
ax_bar = fig.add_subplot(3, 1, 3)
methods  = ["h_sparse NN", "TSVD", "TSVD-NN"]
mean_frl = [frl_nn.mean(), frl_tsvd.mean(), frl_tnn.mean()]
med_frl  = [np.median(frl_nn), np.median(frl_tsvd), np.median(frl_tnn)]
x_pos = np.arange(3)
bars = ax_bar.bar(x_pos - 0.2, mean_frl, 0.35, label="mean fwd relL2", color=["C0", "C1", "C2"])
ax_bar.bar(x_pos + 0.2, med_frl, 0.35, label="median fwd relL2",
           color=["C0", "C1", "C2"], alpha=0.55)
for b_, v in zip(bars, mean_frl):
    ax_bar.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(methods)
ax_bar.set_ylabel("Forward relL2 (lower=better)")
ax_bar.set_title("Method comparison — forward relL2 (main physics metric)")
ax_bar.legend()

plt.tight_layout()
plt.savefig("comparison_tsvd.png", dpi=130)
print("\nSaved comparison_tsvd.png")
