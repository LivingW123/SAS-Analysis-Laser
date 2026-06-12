"""
Side-by-side comparison: TSVD-NN (MATLAB algorithm reimplemented in Python)
vs h_sparse neural network on the held-out test set.

TSVD-NN algorithm (mirrors TSVD_NN.m):
  1. SVD of DRM (200x200, full resolution)
  2. Pure TSVD: reconstruct using first 7 singular vectors
  3. TSVD-NN: refine TSVD by optimising 7 SVD coefficients so that
     DRM @ positive_def(V7 @ c) minimises residual vs measured b

Outputs: comparison_tsvd.png
"""

import time
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = "old_experiments"

DRM = np.load("DRM.npy").astype(np.float64)

# ── SVD (computed once) ───────────────────────────────────────────────────────
print("Computing SVD of DRM...")
U, S, Vt = np.linalg.svd(DRM, full_matrices=False)
V = Vt.T
num_terms  = 7
U7 = U[:, :num_terms]
S7 = S[:num_terms]
V7 = V[:, :num_terms]
constraint = S[5]   # mirrors MATLAB: S(6,6) clamps the 7th term denominator
print(f"  Condition number: {S[0]/S[-1]:.3e}")
print(f"  S[0..6]: {np.array2string(S7, precision=3, separator=', ')}")

# ── solvers ───────────────────────────────────────────────────────────────────
def tsvd_solve(b):
    result = np.zeros(200)
    for i in range(num_terms):
        denom = S7[i] if i < 6 else constraint
        result += V7[:, i] * (U7[:, i] @ b / denom)
    return np.maximum(result, 0.0)

def tsvd_nn_solve(b, c0):
    def residuals(c):
        return DRM @ np.maximum(V7 @ c, 0.0) - b
    res = least_squares(residuals, c0, method="lm", max_nfev=10_000, ftol=1e-9)
    return np.maximum(V7 @ res.x, 0.0)

# ── h_sparse neural network ───────────────────────────────────────────────────
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

# ── test data ─────────────────────────────────────────────────────────────────
Xt = np.load(f"{DATA_DIR}/test_X.npy").astype(np.float64)
yt = np.load(f"{DATA_DIR}/test_y.npy").astype(np.float64)
N  = len(Xt)
print(f"Test set: {N} samples")

# ── run methods ───────────────────────────────────────────────────────────────
print("Running h_sparse NN...")
t0 = time.time()
nn_preds = predict_nn_batch(Xt)
print(f"  done in {time.time()-t0:.1f}s")

print("Running TSVD...")
t0 = time.time()
tsvd_preds = np.array([tsvd_solve(Xt[i]) for i in range(N)])
print(f"  done in {time.time()-t0:.1f}s")

N_NN = min(N, 500)
print(f"Running TSVD-NN on {N_NN} samples...")
t0 = time.time()
tsvd_nn_preds = np.zeros((N_NN, 200))
for i in range(N_NN):
    c0 = V7.T @ tsvd_preds[i]
    tsvd_nn_preds[i] = tsvd_nn_solve(Xt[i], c0)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_NN}  ({time.time()-t0:.0f}s elapsed)")
print(f"  done in {time.time()-t0:.1f}s")

# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(preds, trues, b_raw, label):
    active    = trues > 0
    fwd_pred  = preds @ DRM.T
    fwd_true  = b_raw
    frl       = np.linalg.norm(fwd_pred - fwd_true, axis=1) / (
                np.linalg.norm(fwd_true, axis=1) + 1e-12)
    flux_r    = fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)
    mae_act   = float(np.abs(preds[active] - trues[active]).mean()) if active.any() else float("nan")
    mae_zer   = float(np.abs(preds[~active]).mean())
    recall    = float(((preds > 0.5) & active).sum() / active.sum()) if active.any() else float("nan")
    prec      = float(((preds > 0.5) & active).sum() / max((preds > 0.5).sum(), 1))
    print(f"\n[{label}] n={len(preds)}")
    print(f"  fwd relL2  : mean={frl.mean():.4f}  med={np.median(frl):.4f}  p90={np.percentile(frl,90):.4f}")
    print(f"  flux ratio : mean={flux_r.mean():.3f}  med={np.median(flux_r):.3f}")
    print(f"  MAE active : {mae_act:.3f}   MAE zero : {mae_zer:.3f}")
    print(f"  Recall>0.5 : {recall:.2%}   Precision : {prec:.2%}")
    return frl, flux_r

frl_nn,   fxr_nn   = metrics(nn_preds,       yt,        Xt,       "h_sparse NN  (all)")
frl_tsvd, fxr_tsvd = metrics(tsvd_preds,     yt,        Xt,       "TSVD        (all)")
frl_tnn,  fxr_tnn  = metrics(tsvd_nn_preds,  yt[:N_NN], Xt[:N_NN], f"TSVD-NN     (first {N_NN})")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 13))
fig.suptitle("TSVD-NN vs h_sparse NN — held-out test set", fontsize=12)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# row 0: relL2 CDF | flux ratio hist | sample 0 | sample 1
ax_cdf = fig.add_subplot(gs[0, 0])
for vals, lbl, col in [(frl_nn, "h_sparse NN", "C0"), (frl_tsvd, "TSVD", "C1"),
                        (frl_tnn, "TSVD-NN", "C2")]:
    x = np.sort(vals); y = np.linspace(0, 1, len(x))
    ax_cdf.plot(x, y, label=lbl, color=col, lw=1.4)
ax_cdf.set_xlabel("Forward relL2"); ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("Forward relL2 CDF (lower=better)")
ax_cdf.set_xlim(0, min(2.5, np.percentile(frl_tsvd, 99) * 1.05))
ax_cdf.legend(fontsize=7)

ax_flux = fig.add_subplot(gs[0, 1])
for vals, lbl, col in [(fxr_nn, "h_sparse NN", "C0"), (fxr_tsvd, "TSVD", "C1"),
                        (fxr_tnn, "TSVD-NN", "C2")]:
    ax_flux.hist(np.clip(vals, 0, 4), bins=40, alpha=0.5, label=lbl, color=col, density=True)
ax_flux.axvline(1.0, color="k", lw=1, ls="--", label="ideal")
ax_flux.set_xlabel("Flux ratio  DRM@pred / b"); ax_flux.set_ylabel("Density")
ax_flux.set_title("Flux ratio (1.0 = perfect)")
ax_flux.legend(fontsize=7)

# 6 sample spectra across rows 0–1
rng  = np.random.default_rng(42)
idx6 = rng.choice(N_NN, 6, replace=False)
sample_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

for (row, col), idx in zip(sample_positions, idx6):
    ax = fig.add_subplot(gs[row, col])
    ax.stem(yt[idx],            markerfmt="C0.", linefmt="C0-",  basefmt="k-", label="True")
    ax.stem(nn_preds[idx],      markerfmt="C1x", linefmt="C1--", basefmt="k-", label="h_sparse")
    ax.stem(tsvd_preds[idx],    markerfmt="C2+", linefmt="C2:",  basefmt="k-", label="TSVD")
    ax.stem(tsvd_nn_preds[idx], markerfmt="C3^", linefmt="C3-.", basefmt="k-", label="TSVD-NN")
    frl_nn_s  = float(np.linalg.norm(DRM @ nn_preds[idx]      - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tsvd_s = float(np.linalg.norm(DRM @ tsvd_preds[idx]   - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tnn_s  = float(np.linalg.norm(DRM @ tsvd_nn_preds[idx] - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    ax.set_title(f"#{idx}  NN={frl_nn_s:.3f} TSVD={frl_tsvd_s:.3f} TNN={frl_tnn_s:.3f}", fontsize=7)
    ax.set_xlabel("Energy bin", fontsize=7); ax.set_ylabel("Weight", fontsize=7)
    ax.tick_params(labelsize=6)
    if (row, col) == (0, 2):
        ax.legend(fontsize=6)

# row 2: bar chart spanning full width
ax_bar = fig.add_subplot(gs[2, :])
methods  = ["h_sparse NN", "TSVD", "TSVD-NN"]
mean_frl = [frl_nn.mean(),    frl_tsvd.mean(),    frl_tnn.mean()]
med_frl  = [np.median(frl_nn), np.median(frl_tsvd), np.median(frl_tnn)]
x_pos = np.arange(3)
bars_m = ax_bar.bar(x_pos - 0.2, mean_frl, 0.35, label="mean fwd relL2",
                    color=["C0", "C1", "C2"])
bars_d = ax_bar.bar(x_pos + 0.2, med_frl,  0.35, label="median fwd relL2",
                    color=["C0", "C1", "C2"], alpha=0.55)
for b_, v in zip(bars_m, mean_frl):
    ax_bar.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
for b_, v in zip(bars_d, med_frl):
    ax_bar.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(methods, fontsize=11)
ax_bar.set_ylabel("Forward relL2 (lower=better)")
ax_bar.set_title("Method comparison — forward relL2")
ax_bar.legend(fontsize=9)

plt.savefig("comparison_tsvd.png", dpi=130, bbox_inches="tight")
print("\nSaved comparison_tsvd.png")
