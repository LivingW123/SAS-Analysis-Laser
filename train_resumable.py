import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

CHUNK       = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TOTAL       = 200
BATCH_SIZE  = 64
LR          = 1e-3
VAL_FRAC    = 0.2
SEED        = 42
CKPT        = "checkpoint.pt"
MODEL_OUT   = "fc_model.pt"

torch.manual_seed(SEED)
device = torch.device("cpu")

X_raw = np.load("X_train.npy").astype(np.float32)
y_raw = np.load("y_train.npy").astype(np.float32)

X_log  = np.log1p(X_raw)
X_mean = X_log.mean(axis=0, keepdims=True)
X_std  = X_log.std(axis=0, keepdims=True) + 1e-8
X_norm = (X_log - X_mean) / X_std
y_norm = y_raw / y_raw.max()

np.save("X_norm_stats.npy", np.stack([X_mean[0], X_std[0]]))
np.save("y_scale.npy", np.array([y_raw.max()]))

dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y_norm))
n_val   = int(len(dataset) * VAL_FRAC)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)

class SpectrumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 200),
        )
    def forward(self, x):
        return self.net(x)

model = SpectrumNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-6)

start_epoch, best_val = 0, float("inf")
train_hist, val_hist = [], []

if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["opt"])
    scheduler.load_state_dict(ck["sched"])
    start_epoch = ck["epoch"]
    best_val    = ck["best_val"]
    train_hist  = ck["train_hist"]
    val_hist    = ck["val_hist"]
    print(f"Resumed from epoch {start_epoch} (best val {best_val:.4e})")

def weighted_mse(pred, target):
    w = torch.where(target > 0, 10.0, 1.0)
    return (w * (pred - target) ** 2).mean()

end_epoch = min(start_epoch + CHUNK, TOTAL)
for epoch in range(start_epoch + 1, end_epoch + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = weighted_mse(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)
    train_loss = running / n_train

    model.eval()
    with torch.no_grad():
        val_loss = sum(weighted_mse(model(xb), yb).item() * len(xb)
                       for xb, yb in val_loader) / n_val
    scheduler.step(val_loss)
    train_hist.append(train_loss); val_hist.append(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_OUT)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{TOTAL}  train={train_loss:.4e}  "
              f"val={val_loss:.4e}  lr={optimizer.param_groups[0]['lr']:.1e}")

torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(), "epoch": end_epoch,
            "best_val": best_val, "train_hist": train_hist,
            "val_hist": val_hist}, CKPT)
np.save("loss_history.npy", np.array([train_hist, val_hist]))
print(f"Checkpoint at epoch {end_epoch}; best val {best_val:.4e}")
print("DONE" if end_epoch >= TOTAL else "MORE")
