"""
Fine-tuning experiment harness, checkpointed for chunked runs.

Usage: python3 finetune.py <exp_name> [chunk_epochs]

Experiments are defined in EXPERIMENTS below. Each gets its own checkpoint
(ckpt_<exp>.pt) and best-model file (model_<exp>.pt). Test metrics are
appended to results.json when training completes.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS = {
    # baseline arch + 6x more data, cosine schedule
    "b_moredata": dict(train="train30k", hidden=[256, 128], dropout=0.0,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False),
    # wider/deeper net on big data
    "c_wide":     dict(train="train30k", hidden=[512, 512, 256], dropout=0.1,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False),
    # scale-invariant: each X divided by its own sum, y divided by same factor
    "d_scaleinv": dict(train="train30k", hidden=[512, 512, 256], dropout=0.1,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=True),
    # physics-informed: weighted MSE + forward-consistency (DRM@pred ~ DRM@y) + L1 sparsity
    "e_physics":  dict(train="train30k", hidden=[512, 512, 256], dropout=0.05,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False,
                       fwd_w=1.0, l1_w=1e-3),
    # forward-consistency dominant, stronger sparsity
    "f_fwdonly":  dict(train="train30k", hidden=[512, 512, 256], dropout=0.05,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False,
                       fwd_w=5.0, l1_w=3e-3),
    # non-negative output (softplus) + forward consistency: NNLS-like solution
    "g_nonneg":   dict(train="train30k", hidden=[512, 512, 256], dropout=0.05,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False,
                       fwd_w=1.0, l1_w=1e-3, nonneg=True),
    # g + 10x stronger L1 to suppress null-space flux leakage  -> FINAL MODEL
    "h_sparse":   dict(train="train30k", hidden=[512, 512, 256], dropout=0.05,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False,
                       fwd_w=1.0, l1_w=1e-2, nonneg=True),
    # h + direct spectral-flux conservation: penalise sum(pred) != sum(target)
    # targets the ~1.5x null-space flux overestimate that L1 alone can't reach
    "i_flux":     dict(train="train30k", hidden=[512, 512, 256], dropout=0.05,
                       epochs=80, lr=1e-3, batch=256, per_sample_norm=False,
                       fwd_w=1.0, l1_w=1e-2, nonneg=True, flux_w=1.0),
}

EXP      = sys.argv[1]
CHUNK    = int(sys.argv[2]) if len(sys.argv) > 2 else 25
cfg      = EXPERIMENTS[EXP]
CKPT     = f"ckpt_{EXP}.pt"
MODEL    = f"model_{EXP}.pt"
DATA_DIR = "old_experiments"   # regenerate with gen_data_v2.py to move to root

torch.manual_seed(0)
device = torch.device("cpu")

# ── data ──────────────────────────────────────────────────────────────────────
X_raw  = np.load(f"{DATA_DIR}/{cfg['train']}_X.npy").astype(np.float64)
y_raw  = np.load(f"{DATA_DIR}/{cfg['train']}_y.npy").astype(np.float64)
Xt_raw = np.load(f"{DATA_DIR}/test_X.npy").astype(np.float64)
yt_raw = np.load(f"{DATA_DIR}/test_y.npy").astype(np.float64)

REF_SUM = 1.0  # per-sample norm: scale X so sum(X)=REF_SUM*1e10 equivalent

def prepare(X, y, stats=None, per_sample=False):
    """Returns normalized X, normalized y, y_rescale (per-sample), stats."""
    if per_sample:
        s = X.sum(axis=1, keepdims=True) + 1e-30   # amplitude factor
        Xs = X / s * 1e10                          # bring into fixed range
        ys = y / s * 1e10                          # same linear factor
    else:
        Xs, ys, s = X, y, None
    Xl = np.log1p(Xs)
    if stats is None:
        m = Xl.mean(axis=0, keepdims=True)
        sd = Xl.std(axis=0, keepdims=True) + 1e-8
        stats = (m, sd)
    Xn = (Xl - stats[0]) / stats[1]
    return Xn.astype(np.float32), ys.astype(np.float32), s, stats

Xn, ys, s_train, stats = prepare(X_raw, y_raw, per_sample=cfg["per_sample_norm"])
y_scale = float(ys.max())
yn = ys / y_scale

Xtn, yts, s_test, _ = prepare(Xt_raw, yt_raw, stats=stats,
                              per_sample=cfg["per_sample_norm"])

np.save(f"norm_{EXP}.npy", np.stack([stats[0][0], stats[1][0]]))
np.save(f"yscale_{EXP}.npy", np.array([y_scale]))

n_val = 3000
X_tr, y_tr = Xn[:-n_val], yn[:-n_val]
X_va, y_va = Xn[-n_val:], yn[-n_val:]
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=cfg["batch"], shuffle=True)
X_va_t = torch.from_numpy(X_va); y_va_t = torch.from_numpy(y_va)

# ── model ─────────────────────────────────────────────────────────────────────
def build(hidden, dropout):
    layers, d = [], 200
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, 200))
    if cfg.get("nonneg"):
        layers.append(nn.Softplus(beta=5.0))   # non-negative spectrum by construction
    return nn.Sequential(*layers)

model = build(cfg["hidden"], cfg["dropout"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

start_epoch, best_val = 0, float("inf")
hist = {"train": [], "val": []}
if os.path.exists(CKPT):
    try:
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        start_epoch, best_val, hist = ck["epoch"], ck["best_val"], ck["hist"]
        print(f"[{EXP}] resumed at epoch {start_epoch} (best val {best_val:.4e})")
    except Exception as e:
        print(f"[{EXP}] checkpoint unreadable ({e.__class__.__name__}); starting fresh")

def weighted_mse(pred, target):
    w = torch.where(target > 0, 10.0, 1.0)
    return (w * (pred - target) ** 2).mean()

# physics-informed loss terms (operate in normalized-y units; DRM scaled for stability)
FWD_W  = cfg.get("fwd_w", 0.0)
L1_W   = cfg.get("l1_w", 0.0)
FLUX_W = cfg.get("flux_w", 0.0)
if FWD_W > 0:
    _drm = np.load("DRM.npy")
    DRM_t = torch.from_numpy((_drm / np.linalg.norm(_drm, 2)).astype(np.float32))

def total_loss(pred, target):
    loss = weighted_mse(pred, target)
    if FWD_W > 0:
        fp = pred @ DRM_t.T          # forward-projected prediction
        ft = target @ DRM_t.T        # forward-projected truth
        fwd = ((fp - ft) ** 2).sum(dim=1) / ((ft ** 2).sum(dim=1) + 1e-12)
        loss = loss + FWD_W * fwd.mean()
    if L1_W > 0:
        loss = loss + L1_W * pred.abs().mean()
    if FLUX_W > 0:
        # Penalise spectral-flux mismatch: null-space components are invisible
        # to the forward loss but inflate sum(pred) by ~1.5x in h_sparse.
        pred_flux = pred.sum(dim=1)
        true_flux = target.sum(dim=1)
        flux_err = ((pred_flux - true_flux) ** 2 / (true_flux ** 2 + 1e-12)).mean()
        loss = loss + FLUX_W * flux_err
    return loss

end_epoch = min(start_epoch + CHUNK, cfg["epochs"])
t0 = time.time()
for epoch in range(start_epoch + 1, end_epoch + 1):
    model.train(); running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = total_loss(model(xb), yb)
        loss.backward(); optimizer.step()
        running += loss.item() * len(xb)
    scheduler.step()
    model.eval()
    with torch.no_grad():
        val = total_loss(model(X_va_t), y_va_t).item()
    hist["train"].append(running / len(X_tr)); hist["val"].append(val)
    if val < best_val:
        best_val = val
        torch.save(model.state_dict(), MODEL)
    if epoch % 10 == 0 or epoch == 1:
        print(f"[{EXP}] {epoch:3d}/{cfg['epochs']} train={hist['train'][-1]:.4e} "
              f"val={val:.4e} ({time.time()-t0:.0f}s)")
    if time.time() - t0 > 25:   # stay well under the 45s call limit
        end_epoch = epoch
        break

torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(), "epoch": end_epoch,
            "best_val": best_val, "hist": hist}, CKPT + ".tmp")
os.replace(CKPT + ".tmp", CKPT)   # atomic — no corrupt checkpoints on kill

if end_epoch >= cfg["epochs"]:
    # ── final test-set evaluation with best model ────────────────────────────
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xtn)).clamp(min=0).numpy() * y_scale
    if cfg["per_sample_norm"]:
        pred = pred * (s_test / 1e10)   # undo per-sample scaling -> original units
    true = yt_raw
    act = true > 0
    mae_a = float(np.abs(pred[act] - true[act]).mean())
    mae_z = float(np.abs(pred[~act]).mean())
    recall = float(((pred > 0.5) & act).sum() / act.sum())
    prec   = float((act & (pred > 0.5)).sum() / max((pred > 0.5).sum(), 1))
    rel = float((np.linalg.norm(pred - true, axis=1) /
                 (np.linalg.norm(true, axis=1) + 1e-12)).mean())
    res = dict(exp=EXP, cfg={k: str(v) for k, v in cfg.items()},
               best_val=best_val, mae_active=mae_a, mae_zero=mae_z,
               recall=recall, precision=prec, rel_l2=rel)
    allres = json.load(open("results.json")) if os.path.exists("results.json") else []
    allres = [r for r in allres if r["exp"] != EXP] + [res]
    json.dump(allres, open("results.json", "w"), indent=1)
    print(f"[{EXP}] TEST  mae_act={mae_a:.3f} mae_zero={mae_z:.3f} "
          f"recall={recall:.2%} prec={prec:.2%} relL2={rel:.3f}")
    print("DONE")
else:
    print(f"checkpoint at {end_epoch}; MORE")
