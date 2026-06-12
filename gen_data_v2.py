"""
Vectorized synthetic data generator (v2).
Usage: python3 gen_data_v2.py <n_samples> <seed> <out_prefix>
Same recipe as generate_training_data.py: sparse integer spectra (1-20 active
bins, weights 1-10), X = DRM^T-oriented response @ y + 2% Gaussian noise.
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

y = np.zeros((N, 200), dtype=np.float64)
for i in range(N):
    n_active = rng.integers(1, 21)
    bins = rng.choice(200, size=n_active, replace=False)
    y[i, bins] = rng.integers(1, 11, size=n_active)

X = y @ DRM.T                       # (N,200) detector signals
noise = rng.normal(0, 0.02 * np.abs(X))
X = np.maximum(X + noise, 0.0)

np.save(f"{PRE}_X.npy", X.astype(np.float32))
np.save(f"{PRE}_y.npy", y.astype(np.float32))
np.save("DRM.npy", DRM.astype(np.float32))
print(f"{PRE}: X {X.shape}, y {y.shape}")
