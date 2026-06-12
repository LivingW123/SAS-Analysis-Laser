"""
Sparse-discrete synthetic data generator.
Usage: python gen_data_v2.py <n_samples> <seed> <out_prefix>

Each training spectrum y is sparse: 20% of the 200 energy bins (i.e. ~40 bins)
are randomly selected and assigned an integer weight drawn from {1, ..., 9};
the remaining 80% are zero.  This better reflects real gamma-ray sources, which
emit at a limited number of energies rather than across the full spectrum.

The first column of 200x200.xlsx (= the real reference measurement b used in
TSVD_NN.m) is saved to b_real.npy.

DRM column structure: 5 detector strips, each ~48 channels wide, with valleys
(inter-strip gaps) at detector channels 48, 96, 144, 191.
"""
import sys
import numpy as np
import pandas as pd

N    = int(sys.argv[1])
SEED = int(sys.argv[2])
PRE  = sys.argv[3]

ACTIVE_FRAC = 0.20   # fraction of energy bins that are nonzero per spectrum

rng = np.random.default_rng(SEED)
drm_raw = pd.read_excel("200x200.xlsx", header=None).values.astype(np.float64)
DRM = drm_raw.T  # rows=detector, cols=energy (matches MATLAB EDRM = x200')

# Real reference measurement: first column of xlsx (= b used in TSVD_NN.m)
b_real = drm_raw[:, 0]
np.save("b_real.npy", b_real.astype(np.float32))

# Sparse discrete spectra: 20% of bins active, values in {1..9}
n_active = max(1, round(200 * ACTIVE_FRAC))   # ~40 bins per spectrum
y = np.zeros((N, 200), dtype=np.float32)
for i in range(N):
    cols = rng.choice(200, size=n_active, replace=False)
    y[i, cols] = rng.integers(1, 10, size=n_active).astype(np.float32)

X = y @ DRM.T                           # (N, 200) detector signals
noise = rng.normal(0, 0.02 * np.abs(X))
X = np.maximum(X + noise, 0.0)

np.save(f"{PRE}_X.npy", X.astype(np.float32))
np.save(f"{PRE}_y.npy", y.astype(np.float32))
np.save("DRM.npy", DRM.astype(np.float32))
print(f"{PRE}: X {X.shape}  y {y.shape}")
print(f"  y  active-bins/sample={n_active}  zero-frac={(y == 0).mean():.1%}  "
      f"nonzero range=[{y[y>0].min():.0f}, {y[y>0].max():.0f}]")
print(f"  X  min={X.min():.3e}  max={X.max():.3e}  mean={X.mean():.3e}")
print(f"Saved b_real.npy  shape={b_real.shape}  "
      f"range=[{b_real.min():.3e}, {b_real.max():.3e}]")
