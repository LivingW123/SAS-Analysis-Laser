"""
Side-by-side comparison: TSVD-NN (MATLAB algorithm reimplemented in Python)
vs StripNet on the held-out dense-discrete test set.

TSVD-NN algorithm (mirrors TSVD_NN.m):
  1. SVD of DRM (200x200)
  2. Pure TSVD: reconstruct using first 7 singular vectors, clip to >= 0
  3. TSVD-NN: refine by optimising 7 SVD coefficients c so that
     DRM @ max(V7 @ c, 0) minimises residual vs measured b (Levenberg-Marquardt)

Outputs: figures/comparison_tsvd.png
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
import matplotlib.gridspec as gridspec

FIGS_DIR    = "figures"
N_STRIPS    = 5
STRIP_WIDTH = 40
os.makedirs(FIGS_DIR, exist_ok=True)

DRM = np.load("DRM.npy").astype(np.float64)

# ── SVD ───────────────────────────────────────────────────────────────────────
print("Computing SVD of DRM...")
U, S, Vt = np.linalg.svd(DRM, full_matrices=False)
V = Vt.T
num_terms  = 7
U7 = U[:, :num_terms]
S7 = S[:num_terms]
V7 = V[:, :num_terms]
constraint = S[5]   # mirrors MATLAB S(6,6): clamps the 7th-term denominator
print(f"  Condition number : {S[0]/S[-1]:.3e}")
print(f"  S[0..6]          : {np.array2string(S7, precision=3, separator=', ')}")

# ── TSVD solvers ──────────────────────────────────────────────────────────────
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

# ── StripNet ──────────────────────────────────────────────────────────────────
class StripNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.strip_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(STRIP_WIDTH, 64), nn.LayerNorm(64), nn.GELU())
            for _ in range(N_STRIPS)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(N_STRIPS * 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, 128),           nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 200),
            nn.Softplus(beta=5.0),
        )

    def forward(self, x):
        strips = x.reshape(x.shape[0], N_STRIPS, STRIP_WIDTH)
        feats  = torch.cat([self.strip_enc[i](strips[:, i]) for i in range(N_STRIPS)], dim=1)
        return self.decoder(feats)

model = StripNet()
model.load_state_dict(torch.load("fc_model_discrete.pt", map_location="cpu"))
model.eval()

stats   = np.load("norm_discrete.npy")
X_mean, X_std = stats[0], stats[1]
y_scale = float(np.load("yscale_discrete.npy").flat[0])

def predict_nn_batch(X_raw):
    Xl = (np.log1p(X_raw) - X_mean) / X_std
    with torch.no_grad():
        out = model(torch.from_numpy(Xl.astype(np.float32)))
    return out.numpy() * y_scale

# ── test data (dense discrete) ────────────────────────────────────────────────
Xt = np.load("test_disc_X.npy").astype(np.float64)
yt = np.load("test_disc_y.npy").astype(np.float64)
N  = len(Xt)
print(f"Test set: {N} samples  y range [{yt.min():.0f}, {yt.max():.0f}]")

# ── run methods ───────────────────────────────────────────────────────────────
print("Running StripNet...")
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
        print(f"  {i+1}/{N_NN}  ({time.time()-t0:.0f}s)")
print(f"  done in {time.time()-t0:.1f}s")

# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(preds, trues, b_raw, label):
    active   = trues > 0
    fwd_pred = preds @ DRM.T
    fwd_true = b_raw
    frl      = np.linalg.norm(fwd_pred - fwd_true, axis=1) / (
               np.linalg.norm(fwd_true, axis=1) + 1e-12)
    flux_r   = fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)
    mae_act  = float(np.abs(preds[active] - trues[active]).mean()) if active.any() else float("nan")
    mae_zer  = float(np.abs(preds[~active]).mean())
    recall   = float(((preds > 0.5) & active).sum() / active.sum()) if active.any() else float("nan")
    prec     = float(((preds > 0.5) & active).sum() / max((preds > 0.5).sum(), 1))
    print(f"\n[{label}]  n={len(preds)}")
    print(f"  fwd relL2  : mean={frl.mean():.4f}  med={np.median(frl):.4f}  p90={np.percentile(frl,90):.4f}")
    print(f"  flux ratio : mean={flux_r.mean():.3f}  med={np.median(flux_r):.3f}")
    print(f"  MAE active : {mae_act:.3f}   MAE zero : {mae_zer:.3f}")
    print(f"  Recall>0.5 : {recall:.2%}   Precision : {prec:.2%}")
    return frl, flux_r

frl_nn,   fxr_nn   = metrics(nn_preds,      yt,        Xt,        "StripNet     (all)")
frl_tsvd, fxr_tsvd = metrics(tsvd_preds,    yt,        Xt,        "TSVD         (all)")
frl_tnn,  fxr_tnn  = metrics(tsvd_nn_preds, yt[:N_NN], Xt[:N_NN], f"TSVD-NN      (first {N_NN})")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 13))
fig.suptitle("TSVD-NN vs StripNet — dense discrete test set", fontsize=12)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# row 0: relL2 CDF | flux ratio hist | sample 0 | sample 1
ax_cdf = fig.add_subplot(gs[0, 0])
for frl_vals, lbl, col in [(frl_nn, "StripNet", "C0"), (frl_tsvd, "TSVD", "C1"),
                            (frl_tnn, "TSVD-NN", "C2")]:
    xs = np.sort(frl_vals); ys = np.linspace(0, 1, len(xs))
    ax_cdf.plot(xs, ys, label=lbl, color=col, lw=1.4)
ax_cdf.set_xlabel("Forward relL2"); ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("Forward relL2 CDF (lower=better)")
ax_cdf.set_xlim(0, min(2.5, np.percentile(frl_tsvd, 99) * 1.05))
ax_cdf.legend(fontsize=7)

ax_flux = fig.add_subplot(gs[0, 1])
for fxr_vals, lbl, col in [(fxr_nn, "StripNet", "C0"), (fxr_tsvd, "TSVD", "C1"),
                             (fxr_tnn, "TSVD-NN", "C2")]:
    ax_flux.hist(np.clip(fxr_vals, 0, 4), bins=40, alpha=0.5, label=lbl, color=col, density=True)
ax_flux.axvline(1.0, color="k", lw=1, ls="--", label="ideal")
ax_flux.set_xlabel("Flux ratio  DRM@pred / b"); ax_flux.set_ylabel("Density")
ax_flux.set_title("Flux ratio (1.0 = perfect)"); ax_flux.legend(fontsize=7)

# 6 sample spectra across rows 0–1
rng  = np.random.default_rng(42)
idx6 = rng.choice(N_NN, 6, replace=False)
sample_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
energy_bins = np.arange(200)

for (row, col), idx in zip(sample_positions, idx6):
    ax = fig.add_subplot(gs[row, col])
    ax.bar(energy_bins, yt[idx],            color="C0", alpha=0.55, label="True",     width=1.0)
    ax.bar(energy_bins, nn_preds[idx],      color="C1", alpha=0.55, label="StripNet", width=1.0)
    ax.bar(energy_bins, tsvd_preds[idx],    color="C2", alpha=0.4,  label="TSVD",     width=1.0)
    ax.bar(energy_bins, tsvd_nn_preds[idx], color="C3", alpha=0.4,  label="TSVD-NN",  width=1.0)
    frl_nn_s  = float(np.linalg.norm(DRM @ nn_preds[idx]       - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tsvd_s = float(np.linalg.norm(DRM @ tsvd_preds[idx]    - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    frl_tnn_s  = float(np.linalg.norm(DRM @ tsvd_nn_preds[idx] - Xt[idx]) / (np.linalg.norm(Xt[idx]) + 1e-12))
    ax.set_title(f"#{idx}  SN={frl_nn_s:.3f} TV={frl_tsvd_s:.3f} TN={frl_tnn_s:.3f}", fontsize=7)
    ax.set_xlabel("Energy bin", fontsize=7); ax.set_ylabel("Weight", fontsize=7)
    ax.tick_params(labelsize=6)
    if (row, col) == (0, 2):
        ax.legend(fontsize=6)

# row 2: bar chart spanning full width
ax_bar = fig.add_subplot(gs[2, :])
methods  = ["StripNet", "TSVD", "TSVD-NN"]
mean_frl = [frl_nn.mean(),     frl_tsvd.mean(),     frl_tnn.mean()]
med_frl  = [np.median(frl_nn), np.median(frl_tsvd), np.median(frl_tnn)]
x_pos = np.arange(3)
bars_m = ax_bar.bar(x_pos - 0.2, mean_frl, 0.35, label="mean fwd relL2",   color=["C0", "C1", "C2"])
bars_d = ax_bar.bar(x_pos + 0.2, med_frl,  0.35, label="median fwd relL2", color=["C0", "C1", "C2"], alpha=0.55)
for b_, v in zip(bars_m, mean_frl):
    ax_bar.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
for b_, v in zip(bars_d, med_frl):
    ax_bar.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(methods, fontsize=11)
ax_bar.set_ylabel("Forward relL2 (lower=better)")
ax_bar.set_title("Method comparison — forward relL2")
ax_bar.legend(fontsize=9)

plt.savefig(f"{FIGS_DIR}/comparison_tsvd.png", dpi=130, bbox_inches="tight")
print(f"\nSaved {FIGS_DIR}/comparison_tsvd.png")
