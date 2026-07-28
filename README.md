# SAS MeV Energy Classification Pipeline

Neural network pipeline for identifying photon energy bins from a Geant4 Scintillator/SAS detector response matrix (DRM).

## Background

The 200×200 DRM maps incident photon energy (0–50 MeV in 200 bins of 0.25 MeV each) to detector channel responses. Given a noisy 200-channel detector reading, the goal is to classify which energy bin produced it. This is an ML alternative to the TSVD unfolding approach in `TSVD_NN.m`.

## Files

| File                 | Purpose                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `data_utils.py`    | DRM loading, energy binning, synthetic spectrum/noise/saturation generation, normalization |
| `train_mev.py`     | TensorFlow FC classifier training for n = 10, 20, 50, 100, 200 bins        |
| `visualize_mev.py` | All matplotlib figures                                                     |
| `train_cnn.py`     | CNN spectrum-regression training (in-memory), see below                    |
| `train_cnn_chunk.py` / `train_cnn_chunk_converge.py` | Resumable/chunked CNN training for large sample counts or time-limited environments |
| `infer_cnn.py`     | Per-shot diagnostic figures (predicted vs. real detector response, residuals) |
| `peak_table.py`    | Tabulates the CNN's predicted spectral peak across real shots and CSV variants |
| `noise_sweep.py`   | Grid/random search over the detector-noise model (see `NOISE_SEARCH_PLAN.md`) |
| `res/test_images/` | Real shot data, one folder per 5-digit shot number (see its own README)    |
| `200x200.xlsx`     | Geant4 detector response matrix (200 energy bins × 200 detector channels) |
| `TSVD_NN.m`        | MATLAB TSVD unfolding reference implementation                             |
| `PFF.m`            | MATLAB peak-finding / fitting reference                                    |

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

| Parameter      | Value                                                       |
| -------------- | ----------------------------------------------------------- |
| Architecture   | 200 → 512 → 256 → 128 → n, BatchNorm + ReLU, softmax    |
| Loss           | Sparse categorical cross-entropy                            |
| Optimizer      | Adam (lr=1e-3), ReduceLROnPlateau ÷2 after 15 stale epochs |
| Early stopping | Patience 40 on val_accuracy, restores best weights          |
| Samples        | 100/bin × n bins (1k–20k total), 80/20 train/val split    |

Logged per epoch: train/val loss, accuracy, macro precision, macro recall (efficiency), macro F1.

### 3. Figures (`figures/`)

| Figure                     | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `drm_overview.png`       | Full DRM heatmap + integrated response vs energy          |
| `binned_drm.png`         | DRM heatmaps after binning for each n                     |
| `noise_examples.png`     | Clean ± √I band vs single noisy draw for 5 energy bins  |
| `noise_profile.png`      | Mean √I noise level vs MeV for all n                     |
| `training_curves.png`    | Val loss (log), accuracy, F1, efficiency vs epoch         |
| `final_metrics_bar.png`  | Final accuracy / F1 / precision / recall + epochs trained |
| `confusion_matrices.png` | Row-normalized confusion matrices for each n              |
| `per_bin_efficiency.png` | Per-energy-bin recall with mean line                      |

## Outputs

Training produces the following (gitignored):

```
model_mev_n10.keras   model_mev_n20.keras   model_mev_n50.keras   model_mev_n100.keras   model_mev_n200.keras
results_n10_confusion.npy  ...  results_n200_confusion.npy
training_results.json
```

## CNN Spectrum Regression Pipeline

A second, more capable model alongside the FC classifier above: instead of classifying
a single energy bin, `train_cnn.py` regresses the full L1-normalised spectrum (softmax
over n bins) from a 5×48 image built by reshaping the 200-channel detector vector (rows
0-3 = channels 0-191 reshaped to 4×48, row 4 = the last 8 channels each tiled ×6).
Training data is synthetic PFF spectra (single bremsstrahlung exponential + optional
Gaussian bump, `sample_pff_spectra`) forward-projected through the DRM.

### Training-data augmentation (`data_utils.py`)

Two augmentations were added on top of the base Poisson-noise generator so the training
distribution better matches real shots, both applied in `generate_spectrum_batch`:

- **Saturation plateau** (`apply_saturation`, `SAT_FRACTION = 0.30`): a fraction of
  samples get their brightest channels flat-topped to a per-sample ceiling
  (`SAT_CEIL_LOW`-`SAT_CEIL_HIGH` × that sample's peak), emulating CCD flat-topping on
  bright shots (e.g. 11716). Set `sat_fraction=0.0` to disable.
- **Detector noise model** (`add_detector_noise`): generalises the original pure Poisson
  noise (`σ = √response`) to `σ = √(NOISE_GAIN·response + (READ_FRAC·peak)² +
  (MULT_FRAC·response)²)` — shot noise + a read/dark floor + signal-proportional noise.
  `NOISE_GAIN=1.0, READ_FRAC=0.0, MULT_FRAC=0.0` reproduces the old pure-Poisson behaviour.

  The current defaults (`NOISE_GAIN=5.0, READ_FRAC=0.10, MULT_FRAC=0.0`) were chosen by
  `noise_sweep.py`'s search (90-trial grid → 40-trial refinement → 3-seed robustness
  check on the top candidates), scoring each config by how close the trained CNN's
  real-shot residual gets to the NNLS-fit floor (the best any spectrum could achieve
  for that shot's DRM/vector, independent of the model). Full methodology, the scoring
  rationale, and how to rerun/re-tune the search are in `NOISE_SEARCH_PLAN.md`. This
  retrain dropped several shots' residuals substantially (e.g. shot 11707: 72.5%→29.6%)
  and tightened the spread across shots overall, at the cost of lower peak-confidence
  (softmax max) predictions — the model now outputs smoother, less sharply-peaked
  spectra, consistent with training on noisier synthetic data.

### Training scripts

- `train_cnn.py` — trains fully in memory; good for smaller sample counts (tens of
  thousands) where a single training run comfortably fits in RAM/wall-time budget.
- `train_cnn_chunk.py` — resumable: each invocation trains on freshly generated data for
  a fixed wall-time budget, checkpoints, and exits; loop it to reach large sample counts
  on machines with a hard per-process time cap.
- `train_cnn_chunk_converge.py` — like the above but loops internally with an early-stop
  rule (LR decayed to its floor + no improvement for N chunks), so one invocation runs to
  convergence instead of requiring an external loop.

Note: for the n=50 model specifically, empirically a smaller in-memory run
(`train_cnn.py`, ~25k samples) has outperformed a much larger chunked run (~17M samples)
on real-shot residuals despite worse synthetic validation loss — more training data isn't
automatically better here, so it's worth comparing both on real shots rather than assuming
scale wins.

### Verification

- `peak_table.py` — tabulates the predicted spectral peak (MeV, softmax weight, DRM
  reconstruction residual) per shot across CSV variants (`_proc_vector.csv` uncorrected,
  `_proc_vector_ch.csv` Matthew Price's horizontal correction); writes
  `cnn_peak_table_n{N}.csv`.
- `infer_cnn.py` — per-shot 3-panel diagnostic figures (detector response reconstruction,
  predicted spectrum, channel residuals), plus a saturation-masked variant when a
  correction mask is available. Writes `cnn_infer_<shot>_n<n_bins>[_unsat].png`.
