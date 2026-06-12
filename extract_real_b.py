"""
Step 2: Extract real detector signal b from SAS_B/080425 TIF stack.

Pipeline:
  1. Load all 52 TIFs (1040 x 1392, uint8) and sum them into one stacked image
  2. Sum along the horizontal axis -> 1040-pixel vertical profile
  3. Rebin 1040 -> 200 bins (fractional binning: 5.2 px/bin, flux-conserving)

Output: b_real_1040.npy (raw profile), b_real_200.npy (binned), b_real_preview.png
"""

import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIF_DIR = "CSU ALEPH 2025 - Liang/CSU ALEPH 2025 - Liang/SAS_B/080425"
N_BINS  = 200

# ── 1. Stack TIFs ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(f"{TIF_DIR}/*.tif"))
print(f"Found {len(files)} TIFs")

stack = None
for f in files:
    img = np.array(Image.open(f), dtype=np.float64)   # (1040, 1392)
    stack = img if stack is None else stack + img

print(f"Stacked image: {stack.shape}, total counts: {stack.sum():.3e}")

# ── 2. Sum horizontal axis -> vertical profile ───────────────────────────────
profile = stack.sum(axis=1)          # (1040,)
print(f"Vertical profile: {profile.shape}, range [{profile.min():.1f}, {profile.max():.1f}]")

# ── 3. Flux-conserving rebin 1040 -> 200 (5.2 px per bin) ────────────────────
def rebin_conserve(arr, n_out):
    """Rebin by summing, splitting fractional edge pixels proportionally."""
    n_in = len(arr)
    edges = np.linspace(0, n_in, n_out + 1)   # fractional bin edges in pixel units
    cum = np.concatenate([[0.0], np.cumsum(arr)])
    # cumulative counts at fractional positions via linear interpolation
    cum_at = np.interp(edges, np.arange(n_in + 1), cum)
    return np.diff(cum_at)

b200 = rebin_conserve(profile, N_BINS)
print(f"Binned profile: {b200.shape}, sum preserved: "
      f"{profile.sum():.6e} -> {b200.sum():.6e}")

np.save("b_real_1040.npy", profile)
np.save("b_real_200.npy", b200)
print("Saved b_real_1040.npy, b_real_200.npy")

# ── 4. Preview plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
axes[0].imshow(stack, aspect="auto", cmap="inferno")
axes[0].set_title(f"Stacked image ({len(files)} TIFs summed)")
axes[0].set_xlabel("Horizontal px"); axes[0].set_ylabel("Vertical px")

axes[1].plot(profile, lw=0.8)
axes[1].set_title("Vertical profile (sum over horizontal axis), 1040 px")
axes[1].set_xlabel("Vertical pixel"); axes[1].set_ylabel("Summed counts")

axes[2].plot(b200, lw=1.0, color="C1")
axes[2].set_title("Rebinned to 200 bins (flux-conserving, 5.2 px/bin)")
axes[2].set_xlabel("Bin"); axes[2].set_ylabel("Counts")

plt.tight_layout()
plt.savefig("b_real_preview.png", dpi=120)
print("Saved b_real_preview.png")
