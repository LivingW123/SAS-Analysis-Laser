"""
Dense-discrete synthetic data generator.
Usage: python gen_data_v2.py <n_samples> <seed> <out_prefix>

Each training spectrum y is a dense random integer vector: every one of the
200 energy bins gets an independent integer weight drawn from {0, 1, ..., 9}.
This models a broad gamma-ray spectrum as a linear combination of all 200 DRM
columns, with discrete (not continuous) coefficients.

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

rng = np.random.default_rng(SEED)
drm_raw = pd.read_excel("200x200.xlsx", header=None).values.astype(np.float64)
DRM = drm_raw.T  # rows=detector, cols=energy (matches MATLAB EDRM = x200')

# Real reference measurement: first column of xlsx (= b used in TSVD_NN.m)
b_real = drm_raw[:, 0]
np.save("b_real.npy", b_real.astype(np.float32))

# Dense discrete spectra: every energy bin gets an integer in {0..9}
# Produces broadband spectra — a linear combination of all 200 DRM columns
y = rng.integers(0, 10, size=(N, 200))  # shape (N, 200), values 0–9

X = y @ DRM.T                           # (N, 200) detector signals
noise = rng.normal(0, 0.02 * np.abs(X))
X = np.maximum(X + noise, 0.0)

np.save(f"{PRE}_X.npy", X.astype(np.float32))
np.save(f"{PRE}_y.npy", y.astype(np.float32))
np.save("DRM.npy", DRM.astype(np.float32))
print(f"{PRE}: X {X.shape}  y {y.shape}")
print(f"  y  min={y.min()}  max={y.max()}  mean={y.mean():.2f}  "
      f"zero-frac={( y == 0).mean():.1%}")
print(f"  X  min={X.min():.3e}  max={X.max():.3e}  mean={X.mean():.3e}")
print(f"Saved b_real.npy  shape={b_real.shape}  "
      f"range=[{b_real.min():.3e}, {b_real.max():.3e}]")
