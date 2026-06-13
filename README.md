# SAS MeV Energy Classification Pipeline

Neural network pipeline for identifying photon energy bins from a Geant4 Scintillator/SAS detector response matrix (DRM).

## Background

The 200×200 DRM maps incident photon energy (0–50 MeV in 200 bins of 0.25 MeV each) to detector channel responses. Given a noisy 200-channel detector reading, the goal is to classify which energy bin produced it. This is an ML alternative to the TSVD unfolding approach in `TSVD_NN.m`.

## Files

| File | Purpose |
|---|---|
| `data_utils.py` | DRM loading, energy binning, Poisson noise generation, normalization |
| `train_mev.py` | TensorFlow FC classifier training for n = 10, 20, 50, 100 bins |
| `visualize_mev.py` | All matplotlib figures |
| `200x200.xlsx` | Geant4 detector response matrix (200 energy bins × 200 detector channels) |
| `TSVD_NN.m` | MATLAB TSVD unfolding reference implementation |
| `PFF.m` | MATLAB peak-finding / fitting reference |

## Setup

```bash
pip install tensorflow numpy pandas openpyxl scikit-learn matplotlib
```

## Usage

```bash
# Train all four models (n = 10, 20, 50, 100 energy bins)
python train_mev.py

# Generate all figures (requires training_results.json from above)
python visualize_mev.py

# Pre-training figures only (DRM overview, binned DRM, noise examples)
python visualize_mev.py --pre
```

## Pipeline Overview

### 1. Data (`data_utils.py`)

- **DRM orientation**: xlsx rows = energy bins, cols = detector channels; transposed on load so `drm.shape = (200, 200)` with rows = detector channels, cols = energy bins.
- **`bin_drm(drm, n)`**: averages every `200/n` consecutive energy-bin columns → `(200, n)` matrix. Valid n values: 10, 20, 50, 100 (all divide 200).
- **Synthetic noise**: for each of the n energy-bin columns, draws 100 noisy realizations with per-pixel Gaussian noise σ = √I (Poisson statistics).
- **Normalization**: per-channel z-score computed from training split, applied to train and val sets.

### 2. Training (`train_mev.py`)

| Parameter | Value |
|---|---|
| Architecture | 200 → 512 → 256 → 128 → n, BatchNorm + ReLU, softmax |
| Loss | Sparse categorical cross-entropy |
| Optimizer | Adam (lr=1e-3), ReduceLROnPlateau ÷2 after 15 stale epochs |
| Early stopping | Patience 40 on val_accuracy, restores best weights |
| Samples | 100/bin × n bins (1k–10k total), 80/20 train/val split |

Logged per epoch: train/val loss, accuracy, macro precision, macro recall (efficiency), macro F1.

### 3. Figures (`figures/`)

| Figure | Description |
|---|---|
| `drm_overview.png` | Full DRM heatmap + integrated response vs energy |
| `binned_drm.png` | DRM heatmaps after binning for each n |
| `noise_examples.png` | Clean ± √I band vs single noisy draw for 5 energy bins |
| `noise_profile.png` | Mean √I noise level vs MeV for all n |
| `training_curves.png` | Val loss (log), accuracy, F1, efficiency vs epoch |
| `final_metrics_bar.png` | Final accuracy / F1 / precision / recall + epochs trained |
| `confusion_matrices.png` | Row-normalized confusion matrices for each n |
| `per_bin_efficiency.png` | Per-energy-bin recall with mean line |

## Outputs

Training produces the following (gitignored):

```
model_mev_n10.keras   model_mev_n20.keras   model_mev_n50.keras   model_mev_n100.keras
results_n10_confusion.npy  ...  results_n100_confusion.npy
training_results.json
```

## Notes

- The DRM is severely ill-conditioned (condition number ~10⁷; only ~3–7 singular values above the 2% noise floor). Exact 200-bin spectral recovery is impossible for any method — see `old_experiments/NOTES.md` for full analysis.
- For the classification task here, fewer bins (small n) are easier because each bin's DRM column is more distinct after averaging. Expect accuracy to drop as n increases toward 100.
- "Efficiency" in the output is macro recall: fraction of true energy-bin events correctly identified.
