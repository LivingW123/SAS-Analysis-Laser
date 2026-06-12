"""
Train a strip-aware FC model on dense-discrete spectral data.

Architecture: the 200 detector channels are reshaped into 5 strips of 40
channels each (np.reshape(200) → (5, 40)), reflecting the physical detector
structure (valleys at channels 48, 96, 144, 191).  Each strip is encoded by
its own small FC, then the 5 strip embeddings are concatenated and decoded
to the 200-bin output spectrum.

Usage:
    python gen_data_v2.py 30000 0 train_disc     # generate data first
    python gen_data_v2.py 2000  999 test_disc
    python train_discrete.py [chunk_epochs]       # default 25 per call
    python train_discrete.py                      # repeat until DONE
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CHUNK      = int(sys.argv[1]) if len(sys.argv) > 1 else 25
TOTAL      = 100
BATCH_SIZE = 256
LR         = 1e-3
N_VAL      = 3000
SEED       = 0
DATA_DIR   = "."
TRAIN_PRE  = f"{DATA_DIR}/train_disc"
TEST_PRE   = f"{DATA_DIR}/test_disc"
CKPT       = "ckpt_discrete.pt"
MODEL_OUT  = "fc_model_discrete.pt"

# 5 detector strips: reshape 200 → (5, 40)
# Physical valleys at ~48, 96, 144, 191; using equal 40-channel splits
# as a clean approximation (each strip covers one bump region)
N_STRIPS    = 5
STRIP_WIDTH = 40   # 5 × 40 = 200

torch.manual_seed(SEED)
device = torch.device("cpu")

# ── data ──────────────────────────────────────────────────────────────────────
X_raw = np.load(f"{TRAIN_PRE}_X.npy").astype(np.float32)
y_raw = np.load(f"{TRAIN_PRE}_y.npy").astype(np.float32)

X_log  = np.log1p(X_raw)
X_mean = X_log.mean(axis=0, keepdims=True)
X_std  = X_log.std(axis=0, keepdims=True) + 1e-8
X_norm = (X_log - X_mean) / X_std

y_scale = float(y_raw.max())
y_norm  = y_raw / y_scale

np.save("norm_discrete.npy", np.stack([X_mean[0], X_std[0]]))
np.save("yscale_discrete.npy", np.array([y_scale]))

X_tr, y_tr = X_norm[:-N_VAL], y_norm[:-N_VAL]
X_va, y_va = X_norm[-N_VAL:], y_norm[-N_VAL:]

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
    batch_size=BATCH_SIZE, shuffle=True)
X_va_t = torch.from_numpy(X_va)
y_va_t = torch.from_numpy(y_va)

print(f"Train {len(X_tr)}  Val {len(X_va)}  "
      f"y range [{y_raw.min():.0f}, {y_raw.max():.0f}]  scale={y_scale:.1f}")

# ── model ──────────────────────────────────────────────────────────────────────
class StripNet(nn.Module):
    """
    Reshape input (200,) → (5, 40), encode each strip independently, then
    decode the concatenated strip embeddings to the output spectrum.

    Strip encoder:  40 → 64  (per strip, independent weights)
    Decoder:       320 → 256 → 128 → 200 + Softplus (y ≥ 0 by construction)
    """
    def __init__(self):
        super().__init__()
        self.strip_enc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(STRIP_WIDTH, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            ) for _ in range(N_STRIPS)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(N_STRIPS * 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, 128),           nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 200),
            nn.Softplus(beta=5.0),
        )

    def forward(self, x):
        # x: (batch, 200)
        strips = x.reshape(x.shape[0], N_STRIPS, STRIP_WIDTH)  # (batch, 5, 40)
        feats  = torch.cat([self.strip_enc[i](strips[:, i]) for i in range(N_STRIPS)], dim=1)
        return self.decoder(feats)   # (batch, 200)

model     = StripNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL)

# ── loss ───────────────────────────────────────────────────────────────────────
_drm  = np.load("DRM.npy")
DRM_t = torch.from_numpy((_drm / np.linalg.norm(_drm, 2)).astype(np.float32))

def weighted_mse(pred, target):
    w = torch.where(target > 0, 10.0, 1.0)
    return (w * (pred - target) ** 2).mean()

def total_loss(pred, target):
    loss = weighted_mse(pred, target)
    # forward consistency: DRM @ pred ≈ DRM @ target
    fp  = pred   @ DRM_t.T
    ft  = target @ DRM_t.T
    fwd = ((fp - ft) ** 2).sum(dim=1) / ((ft ** 2).sum(dim=1) + 1e-12)
    loss = loss + 1.0 * fwd.mean()
    # L1 sparsity
    loss = loss + 1e-2 * pred.abs().mean()
    # flux conservation (targets the ~1.5x null-space bias)
    pred_flux = pred.sum(dim=1)
    true_flux = target.sum(dim=1)
    flux_err  = ((pred_flux - true_flux) ** 2 / (true_flux ** 2 + 1e-12)).mean()
    loss = loss + 1.0 * flux_err
    return loss

# ── resume ────────────────────────────────────────────────────────────────────
start_epoch, best_val = 0, float("inf")
hist = {"train": [], "val": []}
if os.path.exists(CKPT):
    try:
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        start_epoch, best_val, hist = ck["epoch"], ck["best_val"], ck["hist"]
        print(f"Resumed from epoch {start_epoch}  best_val={best_val:.4e}")
    except Exception as e:
        print(f"Checkpoint unreadable ({e.__class__.__name__}); starting fresh")

# ── training loop ─────────────────────────────────────────────────────────────
end_epoch = min(start_epoch + CHUNK, TOTAL)
t0 = time.time()

for epoch in range(start_epoch + 1, end_epoch + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = total_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val = total_loss(model(X_va_t), y_va_t).item()

    hist["train"].append(running / len(X_tr))
    hist["val"].append(val)

    if val < best_val:
        best_val = val
        torch.save(model.state_dict(), MODEL_OUT)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{TOTAL}  "
              f"train={hist['train'][-1]:.4e}  val={val:.4e}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"({time.time()-t0:.0f}s)")

    if time.time() - t0 > 25:
        end_epoch = epoch
        break

# ── checkpoint (atomic write) ─────────────────────────────────────────────────
torch.save({
    "model": model.state_dict(), "opt": optimizer.state_dict(),
    "sched": scheduler.state_dict(), "epoch": end_epoch,
    "best_val": best_val, "hist": hist,
}, CKPT + ".tmp")
os.replace(CKPT + ".tmp", CKPT)

# ── test-set evaluation when training completes ────────────────────────────────
if end_epoch >= TOTAL:
    Xt = np.load(f"{TEST_PRE}_X.npy").astype(np.float32)
    yt = np.load(f"{TEST_PRE}_y.npy").astype(np.float32)
    Xtn = ((np.log1p(Xt) - X_mean) / X_std).astype(np.float32)
    ytn = yt / y_scale

    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xtn)).numpy() * y_scale
    true = yt
    act  = true > 0
    fwd_pred = pred  @ _drm
    fwd_true = true  @ _drm
    frl = np.linalg.norm(fwd_pred - fwd_true, axis=1) / (
          np.linalg.norm(fwd_true, axis=1) + 1e-12)
    flux_r = fwd_pred.sum(axis=1) / (fwd_true.sum(axis=1) + 1e-12)
    mae_a  = float(np.abs(pred[act] - true[act]).mean())
    mae_z  = float(np.abs(pred[~act]).mean())
    recall = float(((pred > 0.5) & act).sum() / act.sum())
    prec   = float(((pred > 0.5) & act).sum() / max((pred > 0.5).sum(), 1))
    print(f"\nTEST  mae_act={mae_a:.3f}  mae_zero={mae_z:.3f}  "
          f"recall={recall:.2%}  prec={prec:.2%}")
    print(f"      fwd_relL2 mean={frl.mean():.4f}  med={np.median(frl):.4f}  "
          f"p90={np.percentile(frl,90):.4f}")
    print(f"      flux_ratio mean={flux_r.mean():.3f}  med={np.median(flux_r):.3f}")
    print("DONE")
else:
    print(f"Checkpoint at epoch {end_epoch}/{TOTAL}; MORE")
