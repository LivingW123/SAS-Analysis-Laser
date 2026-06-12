"""
Comparison figure: final model (h_sparse) vs ground truth on held-out test set,
plus forward-consistency check and real-b inference.

Outputs: comparison.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# model: h_sparse
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
DRM = np.load("DRM.npy").astype(np.float64)  # (200, 200) rows=detector cols=energy

def predict_batch(X_raw):
    Xl = (np.log1p(X_raw) - X_mean) / X_std
    with torch.no_grad():
        out = model(torch.from_numpy(Xl.astype(np.float32)))
    return out.numpy() * y_scale

# data: use held-out test set 
DATA_DIR = "old_experiments"
Xt = np.load(f"{DATA_DIR}/test_X.npy").astype(np.float64)
yt = np.load(f"{DATA_DIR}/test_y.npy").astype(np.float64)

rng = np.random.default_rng(7)
idx6 = rng.choice(len(Xt), 6, replace=False)
preds6 = predict_batch(Xt[idx6])
trues6 = yt[idx6]

# metrics over full test set (chunked to avoid OOM on large arrays)
all_preds = predict_batch(Xt)
active = yt > 0
mae_a  = float(np.abs(all_preds[active] - yt[active]).mean())
mae_z  = float(np.abs(all_preds[~active]).mean())
recall = float(((all_preds > 0.5) & active).sum() / active.sum())
prec   = float(((all_preds > 0.5) & active).sum() / max((all_preds > 0.5).sum(), 1))
fwd_pred = all_preds @ DRM.T           # (N, 200) forward-projected predictions
fwd_true = yt @ DRM.T
rel_l2   = float((np.linalg.norm(fwd_pred - fwd_true, axis=1) /
                  (np.linalg.norm(fwd_true, axis=1) + 1e-12)).mean())
flux_ratio = float((fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)).mean())

print(f"Test set ({len(Xt)} samples):")
print(f"  MAE active bins : {mae_a:.3f}   MAE zero bins: {mae_z:.3f}")
print(f"  Recall (>0.5)   : {recall:.2%}   Precision: {prec:.2%}")
print(f"  Forward rel-L2  : {rel_l2:.4f}   Flux ratio: {flux_ratio:.3f}")

# figure layout: 3 rows
fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    f"h_sparse final model  |  test MAE act={mae_a:.2f}  zero={mae_z:.2f}  "
    f"recall={recall:.0%}  fwd-relL2={rel_l2:.3f}  flux×{flux_ratio:.2f}",
    fontsize=11)

# row 1-2: predicted vs true spectra (6 samples)
for col, (pred, true) in enumerate(zip(preds6, trues6)):
    ax = fig.add_subplot(3, 3, col + 1)
    ax.stem(true, markerfmt="C0.", linefmt="C0-", basefmt="k-", label="True")
    ax.stem(pred, markerfmt="C1x", linefmt="C1--", basefmt="k-", label="Pred")
    ax.set_title(f"Test sample {idx6[col]}", fontsize=9)
    ax.set_xlabel("Energy bin", fontsize=8); ax.set_ylabel("Weight", fontsize=8)
    if col == 0:
        ax.legend(fontsize=8)

# row 3 left: forward-consistency for 3 samples
ax_fwd = fig.add_subplot(3, 3, 7)
for i, (pred, true) in enumerate(zip(preds6[:3], trues6[:3])):
    color = f"C{i}"
    ax_fwd.plot(DRM @ true, color=color, lw=1.2, label=f"True {i}")
    ax_fwd.plot(DRM @ pred, color=color, lw=1.0, ls="--", label=f"Pred {i}")
ax_fwd.set_title("Forward consistency  DRM @ spectrum", fontsize=9)
ax_fwd.set_xlabel("Detector bin", fontsize=8); ax_fwd.set_ylabel("Signal", fontsize=8)
ax_fwd.legend(fontsize=7, ncol=2)

# row 3 middle: real-b (flipped, baseline-subtracted)
B_FILE = "b_real_200.npy" if os.path.exists("b_real_200.npy") else f"{DATA_DIR}/b_real_200.npy"
b_raw = np.load(B_FILE)
b = (np.flip(b_raw) - np.flip(b_raw).min()).copy()
spec_real = predict_batch(b[None])[0]
ax_real = fig.add_subplot(3, 3, 8)
ax_real.stem(spec_real, markerfmt="C3.", linefmt="C3-", basefmt="k-")
ax_real.set_title("Predicted spectrum — real SAS_B\n(flipped, baseline-sub)", fontsize=9)
ax_real.set_xlabel("Energy bin", fontsize=8); ax_real.set_ylabel("Weight", fontsize=8)

# row 3 right: forward-projected real-b prediction vs measured b
ax_fwdr = fig.add_subplot(3, 3, 9)
ax_fwdr.plot(np.flip(b_raw).copy(), color="C0", lw=1.2, label="Measured b (flipped)")
ax_fwdr.plot(DRM @ spec_real, color="C3", lw=1.0, ls="--", label="DRM @ pred")
ax_fwdr.set_title("Real-b forward check", fontsize=9)
ax_fwdr.set_xlabel("Detector bin", fontsize=8); ax_fwdr.set_ylabel("Counts", fontsize=8)
ax_fwdr.legend(fontsize=8)

plt.tight_layout()
plt.savefig("comparison.png", dpi=130)
print("Saved comparison.png")
