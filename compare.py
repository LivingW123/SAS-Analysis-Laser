"""
Comparison figure: h_sparse final model vs ground truth on held-out test set,
plus real-b inference.

Outputs: comparison.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = "old_experiments"

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

stats = np.load("norm_best.npy")
X_mean, X_std = stats[0], stats[1]
y_scale = float(np.load("yscale_best.npy").flat[0])
DRM = np.load("DRM.npy").astype(np.float64)

def predict_batch(X_raw):
    Xl = (np.log1p(X_raw) - X_mean) / X_std
    with torch.no_grad():
        out = model(torch.from_numpy(Xl.astype(np.float32)))
    return out.numpy() * y_scale

# ── test-set metrics ──────────────────────────────────────────────────────────
Xt = np.load(f"{DATA_DIR}/test_X.npy").astype(np.float64)
yt = np.load(f"{DATA_DIR}/test_y.npy").astype(np.float64)
all_preds = predict_batch(Xt)

active    = yt > 0
mae_a     = float(np.abs(all_preds[active] - yt[active]).mean())
mae_z     = float(np.abs(all_preds[~active]).mean())
recall    = float(((all_preds > 0.5) & active).sum() / active.sum())
prec      = float(((all_preds > 0.5) & active).sum() / max((all_preds > 0.5).sum(), 1))
fwd_pred  = all_preds @ DRM.T
fwd_true  = yt @ DRM.T
frl       = np.linalg.norm(fwd_pred - fwd_true, axis=1) / (
            np.linalg.norm(fwd_true, axis=1) + 1e-12)
flux_r    = fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)

print(f"Test set ({len(Xt)} samples):")
print(f"  MAE active : {mae_a:.3f}   MAE zero : {mae_z:.3f}")
print(f"  Recall>0.5 : {recall:.2%}   Precision : {prec:.2%}")
print(f"  fwd relL2  : mean={frl.mean():.4f}  med={np.median(frl):.4f}  p90={np.percentile(frl,90):.4f}")
print(f"  flux ratio : mean={flux_r.mean():.3f}  med={np.median(flux_r):.3f}")

# ── sample spectra ────────────────────────────────────────────────────────────
rng   = np.random.default_rng(7)
idx6  = rng.choice(len(Xt), 6, replace=False)
pred6 = all_preds[idx6]
true6 = yt[idx6]

# ── real-b inference ──────────────────────────────────────────────────────────
b_raw     = np.load(f"{DATA_DIR}/b_real_200.npy")
b_in      = (np.flip(b_raw) - np.flip(b_raw).min()).copy()
spec_real = predict_batch(b_in[None])[0]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 13))
fig.suptitle(
    f"h_sparse  |  MAE act={mae_a:.2f}  zero={mae_z:.2f}  recall={recall:.0%}  "
    f"fwd-relL2 med={np.median(frl):.3f}  p90={np.percentile(frl,90):.3f}  "
    f"flux×{np.median(flux_r):.2f}",
    fontsize=10)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)

# row 0: relL2 CDF | flux ratio hist | sample 0 | sample 1
ax_cdf = fig.add_subplot(gs[0, 0])
x = np.sort(frl); y = np.linspace(0, 1, len(x))
ax_cdf.plot(x, y, color="C0", lw=1.5)
ax_cdf.axvline(np.median(frl), color="C0", ls="--", lw=1, label=f"med {np.median(frl):.3f}")
ax_cdf.axvline(np.percentile(frl, 90), color="C0", ls=":", lw=1, label=f"p90 {np.percentile(frl,90):.3f}")
ax_cdf.set_xlabel("Forward relL2"); ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("Forward relL2 CDF")
ax_cdf.set_xlim(left=0); ax_cdf.legend(fontsize=7)

ax_flux = fig.add_subplot(gs[0, 1])
ax_flux.hist(np.clip(flux_r, 0, 3), bins=50, color="C1", edgecolor="none", density=True)
ax_flux.axvline(1.0, color="k", lw=1, ls="--", label="ideal")
ax_flux.axvline(np.median(flux_r), color="C1", lw=1, ls="--", label=f"med {np.median(flux_r):.2f}")
ax_flux.set_xlabel("Flux ratio  DRM@pred / DRM@true"); ax_flux.set_ylabel("Density")
ax_flux.set_title("Flux ratio distribution")
ax_flux.legend(fontsize=7)

for col_i, (pred, true, raw_i) in enumerate(zip(pred6[:2], true6[:2], idx6[:2])):
    ax = fig.add_subplot(gs[0, 2 + col_i])
    ax.stem(true, markerfmt="C0.", linefmt="C0-", basefmt="k-", label="True")
    ax.stem(pred, markerfmt="C1x", linefmt="C1--", basefmt="k-", label="Pred")
    frl_s = float(np.linalg.norm(DRM @ pred - DRM @ true) / (np.linalg.norm(DRM @ true) + 1e-12))
    ax.set_title(f"Sample {raw_i}  frl={frl_s:.3f}", fontsize=8)
    ax.set_xlabel("Energy bin", fontsize=7); ax.set_ylabel("Weight", fontsize=7)
    if col_i == 0: ax.legend(fontsize=7)

# row 1: samples 2–5
for col_i, (pred, true, raw_i) in enumerate(zip(pred6[2:], true6[2:], idx6[2:])):
    ax = fig.add_subplot(gs[1, col_i])
    ax.stem(true, markerfmt="C0.", linefmt="C0-", basefmt="k-", label="True")
    ax.stem(pred, markerfmt="C1x", linefmt="C1--", basefmt="k-", label="Pred")
    frl_s = float(np.linalg.norm(DRM @ pred - DRM @ true) / (np.linalg.norm(DRM @ true) + 1e-12))
    ax.set_title(f"Sample {raw_i}  frl={frl_s:.3f}", fontsize=8)
    ax.set_xlabel("Energy bin", fontsize=7); ax.set_ylabel("Weight", fontsize=7)

# row 2: forward consistency (2 cols) | real-b pred | real-b forward check
ax_fwd = fig.add_subplot(gs[2, 0:2])
for i in range(3):
    p, t = pred6[i], true6[i]
    c = f"C{i}"
    ax_fwd.plot(DRM @ t, color=c, lw=1.2, label=f"True {i}")
    ax_fwd.plot(DRM @ p, color=c, lw=1.0, ls="--", label=f"Pred {i}")
ax_fwd.set_title("Forward consistency  DRM @ spectrum (3 samples)", fontsize=9)
ax_fwd.set_xlabel("Detector bin"); ax_fwd.set_ylabel("Signal")
ax_fwd.legend(fontsize=6, ncol=2)

ax_real = fig.add_subplot(gs[2, 2])
ax_real.stem(spec_real, markerfmt="C3.", linefmt="C3-", basefmt="k-")
ax_real.set_title("Predicted spectrum — real SAS_B\n(flipped, baseline-sub)", fontsize=8)
ax_real.set_xlabel("Energy bin", fontsize=7); ax_real.set_ylabel("Weight", fontsize=7)

ax_fwdr = fig.add_subplot(gs[2, 3])
ax_fwdr.plot(np.flip(b_raw).copy(), color="C0", lw=1.2, label="Measured b (flipped)")
ax_fwdr.plot(DRM @ spec_real, color="C3", lw=1.0, ls="--", label="DRM @ pred")
ax_fwdr.set_title("Real-b forward check", fontsize=8)
ax_fwdr.set_xlabel("Detector bin", fontsize=7); ax_fwdr.set_ylabel("Counts", fontsize=7)
ax_fwdr.legend(fontsize=7)

plt.savefig("comparison.png", dpi=130, bbox_inches="tight")
print("Saved comparison.png")
