"""
Verification: loss curve, val-set reconstruction quality, and inference
on the real binned detector profile.

Stops at producing the predicted spectrum — no physics comparison.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# model: h_sparse (200->512->512->256->200, LN+GELU+Dropout, Softplus)
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

stats = np.load("norm_best.npy")          # (2, 200): mean, std of log1p(X)
X_mean, X_std = stats[0], stats[1]
y_scale = float(np.load("yscale_best.npy").flat[0])

def predict(x_raw):
    x = (np.log1p(x_raw) - X_mean) / X_std
    with torch.no_grad():
        y = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    return y.numpy()[0] * y_scale   # Softplus guarantees >= 0

# Loss curve (skipped if history not present)
LOSS_FILE = "old_experiments/loss_history.npy"
if os.path.exists(LOSS_FILE):
    hist = np.load(LOSS_FILE)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist[0], label="train"); ax.plot(hist[1], label="val")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE")
    ax.legend(); ax.set_title("Training loss")
    plt.tight_layout(); plt.savefig("loss_curve.png", dpi=120)
else:
    print("loss_history.npy not found, skipping loss curve")

# Val-set reconstruction check
X_raw = np.load("X_train.npy").astype(np.float64)
y_raw = np.load("y_train.npy").astype(np.float64)
rng = np.random.default_rng(0)
val_idx = rng.choice(len(X_raw), 6, replace=False)

fig2, axes = plt.subplots(2, 3, figsize=(14, 6))
fig2.suptitle("Predictions vs ground truth (synthetic samples)")
for col, idx in enumerate(val_idx):
    y_pred = predict(X_raw[idx])
    ax = axes[col // 3, col % 3]
    ax.stem(y_raw[idx], markerfmt="C0.", linefmt="C0-", basefmt="k-", label="True")
    ax.stem(y_pred, markerfmt="C1x", linefmt="C1--", basefmt="k-", label="Pred")
    ax.set_title(f"Sample {idx}")
    if col == 0: ax.legend()
plt.tight_layout(); plt.savefig("inference_check.png", dpi=120)

idx500 = rng.choice(len(X_raw), 500, replace=False)
preds = np.array([predict(X_raw[i]) for i in idx500])
trues = y_raw[idx500]
active = trues > 0
print("Val-style metrics over 500 samples:")
print(f"  MAE on active bins : {np.abs(preds[active]-trues[active]).mean():.3f} counts (weights are 1-10)")
print(f"  MAE on zero bins   : {np.abs(preds[~active]).mean():.3f}")
hit = ((preds > 0.5) & active).sum() / active.sum()
print(f"  Active-bin recall (pred>0.5): {hit:.2%}")

# Inference on real b 
# Profile must be flipped: vertical axis runs opposite to DRM detector-channel
# ordering. Confirmed by NNLS analysis (see old_experiments/NOTES.md).
# Re-run extract_real_b.py to regenerate b_real_200.npy in root if needed.
B_FILE = "b_real_200.npy" if os.path.exists("b_real_200.npy") else "old_experiments/b_real_200.npy"
b_raw = np.load(B_FILE)
b_flipped = np.flip(b_raw).copy()

variants = {
    "flipped":           b_flipped,
    "flipped_baseline":  (b_flipped - b_flipped.min()).copy(),
}

fig3, axes = plt.subplots(len(variants), 2, figsize=(13, 4 * len(variants)))
fig3.suptitle("Real SAS_B/080425 (flipped) -> predicted spectrum  [no physics scaling]")

for row, (name, bv) in enumerate(variants.items()):
    spec = predict(bv)
    np.save(f"spectrum_pred_{name}.npy", spec)
    print(f"\n[{name}] input [{bv.min():.3e}, {bv.max():.3e}]  "
          f"-> pred: sum={spec.sum():.2f}  max={spec.max():.2f}  "
          f"nonzero(>0.5)={int((spec > 0.5).sum())} bins")
    axes[row, 0].plot(bv, color="C0")
    axes[row, 0].set_title(f"Input b ({name})")
    axes[row, 0].set_xlabel("Detector bin"); axes[row, 0].set_ylabel("Counts")
    axes[row, 1].stem(spec, markerfmt="C1.", linefmt="C1-", basefmt="k-")
    axes[row, 1].set_title(f"Predicted spectrum ({name})")
    axes[row, 1].set_xlabel("Energy bin"); axes[row, 1].set_ylabel("Weight")

plt.tight_layout()
plt.savefig("real_b_prediction.png", dpi=120)
print("\nSaved inference_check.png, real_b_prediction.png")

zb = (np.log1p(b_flipped) - X_mean) / X_std
print(f"\nIn-distribution check (flipped real b): "
      f"mean |z| = {np.abs(zb).mean():.2f}  max |z| = {np.abs(zb).max():.2f} "
      f"(training data has |z| ~ 1)")
