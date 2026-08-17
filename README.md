# SAS MeV Energy Classification Pipeline

Neural network pipeline for identifying photon energy bins from a Geant4 Scintillator/SAS detector response matrix (DRM).

## Background

The 200×200 DRM maps incident photon energy (0–50 MeV in 200 bins of 0.25 MeV each) to detector channel responses. Given a noisy 200-channel detector reading, the goal is to classify which energy bin produced it. This is an ML alternative to the TSVD unfolding approach in `TSVD_NN.m`.

## Layout

Code lives under `src/`, generated artifacts (models, results, figures) under `out/`, mirroring `src/`'s structure one level deep:

```
src/
  core/            data_utils.py + shared model/decode definitions (cnn_model.py, pff_model.py,
                   pff_ensemble_utils.py, real_shots.py) used across training/inference/comparisons/optimizers
  training/
    dnn/           plain dense (fully-connected) models: train_dnn_mev.py, train_dnn_spectrum.py
    cnn/           Conv2D-based spectrum regressor: train_cnn.py + chunked variants
    pff/           PFF parameter regressor generations, see its own section below
  inference/       infer_cnn_ensemble.py, infer_spectrum.py
  comparisons/     evaluate_pff_ensemble.py, peak_table.py, cnn_real_shot_examples.py,
                   pff_domain_gap_check.py, nnls_refine.py
  optimizers/      refine_bump_center_optimizer.py, bump_center_profile_likelihood.py,
                   noise_sweep.py, noise_sweep_multiseed.py
  visualization/   visualize_mev.py
  testing/         prep_tpw_data.py + test_dnn_tpw.py (the TPW real-shot dataset — extract/prep,
                   then test), test_dnn_11733.py

out/               mirrors src/ (out/training/dnn, out/training/cnn, out/training/pff, out/inference,
                   out/comparisons, out/optimizers, out/visualization, out/testing) + out/_unsorted/
                   for pre-refactor artifacts with no producing script.
```

Every script runs as a module from the repo root, e.g. `python -m src.training.dnn.train_dnn_mev` — this
keeps `cwd` at the repo root regardless of a script's nesting depth, so relative input paths like
`res/drm/200x200.xlsx` are unaffected.

Raw/reference data — `res/`, `TPW/`, `matlab/`, `sasdeconsoftware/`, `"CSU ALEPH 2025 - Liang/"`, `env/` —
and the two data-prep notebooks (`gen_train_data.ipynb`, `rescale_vector.ipynb`) stay at the repo root,
outside `src/`/`out/`. Training/run logs (`*.log`) also stay at the repo root — this project keeps them as
a permanent record (several are cited directly in `CLAUDE.md` and in this README's PFF section).

## Files

| File                 | Purpose                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `src/core/data_utils.py`    | DRM loading, energy binning, synthetic spectrum/noise/saturation generation, normalization |
| `src/training/dnn/train_dnn_mev.py`     | TensorFlow FC classifier training for n = 10, 20, 50, 100, 200 bins        |
| `src/visualization/visualize_mev.py` | All matplotlib figures                                                     |
| `src/training/cnn/train_cnn.py`     | CNN spectrum-regression training (in-memory), see below                    |
| `src/training/cnn/train_cnn_chunk.py` / `train_cnn_chunk_converge.py` | Resumable/chunked CNN training for large sample counts or time-limited environments |
| `src/inference/infer_cnn_ensemble.py` | Per-shot diagnostic figures (predicted vs. real detector response, residuals, PFF ensemble fit) |
| `src/comparisons/peak_table.py`    | Tabulates the CNN's predicted spectral peak across real shots and CSV variants |
| `src/optimizers/noise_sweep.py`   | Grid/random search over the detector-noise model |
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
python -m src.training.dnn.train_dnn_mev

# Generate all figures (requires out/training/dnn/dnn_mev_training_results.json from above)
python -m src.visualization.visualize_mev

# Pre-training figures only (DRM overview, binned DRM, noise examples)
python -m src.visualization.visualize_mev --pre
```

## Pipeline Overview

### 1. Data (`src/core/data_utils.py`)

- **DRM orientation**: xlsx rows = energy bins, cols = detector channels; transposed on load so `drm.shape = (200, 200)` with rows = detector channels, cols = energy bins.
- **`bin_drm(drm, n)`**: averages every `200/n` consecutive energy-bin columns → `(200, n)` matrix. Valid n values: 10, 20, 50, 100 (all divide 200).
- **Synthetic noise**: for each of the n energy-bin columns, draws 100 noisy realizations with per-pixel Gaussian noise σ = √I (Poisson statistics).
- **Normalization**: per-channel z-score computed from training split, applied to train and val sets.

### 2. Training (`src/training/dnn/train_dnn_mev.py`)

A plain dense (fully-connected) network — "DNN" — as opposed to the Conv2D-based CNN family below.

| Parameter      | Value                                                       |
| -------------- | ----------------------------------------------------------- |
| Architecture   | 200 → 512 → 256 → 128 → n, BatchNorm + ReLU, softmax    |
| Loss           | Sparse categorical cross-entropy                            |
| Optimizer      | Adam (lr=1e-3), ReduceLROnPlateau ÷2 after 15 stale epochs |
| Early stopping | Patience 40 on val_accuracy, restores best weights          |
| Samples        | 100/bin × n bins (1k–20k total), 80/20 train/val split    |

Logged per epoch: train/val loss, accuracy, macro precision, macro recall (efficiency), macro F1.

### 3. Figures (`out/visualization/figures/`)

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

Training produces the following, under `out/training/dnn/` (gitignored):

```
model_dnn_mev_n10.keras   model_dnn_mev_n20.keras   model_dnn_mev_n50.keras   model_dnn_mev_n100.keras   model_dnn_mev_n200.keras
results_n10_confusion.npy  ...  results_n200_confusion.npy
dnn_mev_training_results.json
```

## CNN Spectrum Regression Pipeline

A second, more capable model alongside the FC classifier above: instead of classifying
a single energy bin, `src/training/cnn/train_cnn.py` regresses the full L1-normalised spectrum (softmax
over n bins) from a 5×48 image built by reshaping the 200-channel detector vector (rows
0-3 = channels 0-191 reshaped to 4×48, row 4 = the last 8 channels each tiled ×6).
Training data is synthetic PFF spectra (single bremsstrahlung exponential + optional
Gaussian bump, `sample_pff_spectra`) forward-projected through the DRM. The model
architecture (`build_model`) and the 200→5×48 reshape (`reshape_to_2d`) live in
`src/core/cnn_model.py`, shared with every other script that needs this CNN's definition.

### Training-data augmentation (`src/core/data_utils.py`)

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
  `src/optimizers/noise_sweep.py`'s search (90-trial grid → 40-trial refinement → 3-seed robustness
  check on the top candidates), scoring each config by how close the trained CNN's
  real-shot residual gets to the NNLS-fit floor (the best any spectrum could achieve
  for that shot's DRM/vector, independent of the model). This
  retrain dropped several shots' residuals substantially (e.g. shot 11707: 72.5%→29.6%)
  and tightened the spread across shots overall, at the cost of lower peak-confidence
  (softmax max) predictions — the model now outputs smoother, less sharply-peaked
  spectra, consistent with training on noisier synthetic data. (The methodology writeup
  this was originally drawn from, `NOISE_SEARCH_PLAN.md`, no longer exists in the repo —
  the parameters above and `src/optimizers/noise_sweep.py`/`noise_sweep_multiseed.py`'s own
  code are the current source of truth for how to rerun/re-tune the search.)

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

- `src/comparisons/peak_table.py` — tabulates the predicted spectral peak (MeV, softmax weight, DRM
  reconstruction residual) per shot across CSV variants (`_proc_vector.csv` uncorrected,
  `_proc_vector_ch.csv` Matthew Price's horizontal correction); writes
  `out/comparisons/cnn_peak_table_n{N}.csv`.
- `src/inference/infer_cnn_ensemble.py` — per-shot 4-panel diagnostic figures (detector response
  reconstruction, predicted spectrum, channel residuals, PFF ensemble fit); see the PFF
  Parameter Regressor Pipeline section below for the 4th panel. Writes into
  `out/inference/cnn_infer_ensemble/`.

## PFF Parameter Regressor Pipeline

A third model family, alongside the FC classifier and CNN spectrum regressor above:
instead of a full spectrum, this directly regresses the 6 parameters of the PFF
functional form (`a1*exp(-a2*x) + a3*exp(-(x-a4)^2/(a5*x+a6/x))`, see `src/core/data_utils.py`)
with per-parameter predictive uncertainty. Only the current, final model is kept on
disk (`out/training/pff/model_pff_ensemble_0-4.keras` / `pff_training_results_ensemble_{0-4}.json`,
a 5-member deep ensemble); every earlier generation below was deleted once this
history was condensed here — see each version's own `train_pff*.log` (still at the repo
root) for full metrics, and `CLAUDE.md` for the deeper investigation notes that don't belong
in a usage README.

### Architecture history

Each generation fixed a specific real-shot failure mode the previous one exposed. All
five live under `src/training/pff/`; the shared architecture/loss/decode definitions
they build on (from v2 onward) live in `src/core/pff_model.py`:

| Version | Script | Key change | Real-shot problem it fixed (or introduced) |
| --- | --- | --- | --- |
| v1 | `train_pff.py` | ReLU mean + heteroscedastic-NLL logvar, hard clip at ±10 | baseline; sigma saturated at the clip ceiling on real (OOD) shots (e.g. a3 = ±14841) |
| v2 | `train_pff_bounded_gated.py` | tanh-bounded logvar (caps sigma_norm ~7.4 vs ~148.4) + gated bump classifier (explicit `p(bump)` head, since a3 is bimodal — exactly 0 or in [5,100]) | fixed the hard-clip saturation; but sigma/p(bump) came out bit-for-bit identical across all 14 real shots — saturating at one fixed output corner regardless of input |
| v3 | `train_pff_v3_realistic_noise.py` | same v2 architecture, training-data generator switched from plain sqrt(response) Poisson noise to the calibrated noise+saturation model (`generate_pff_training_data`) | real shots have 0-92/200 saturation-corrected channels the old generator never modeled; closing that gap is what finally made `p(bump)` vary meaningfully by shot (0.001-1.000) instead of a flat 1.000 |
| ensemble | `train_pff_ensemble_member.py` | 5 independently-initialized/seeded v3 models, combined via `pff_ensemble_utils.decode_ensemble` (law of total variance) | a single model has no way to signal "I've never seen anything like this input"; cross-member disagreement gives a genuine epistemic-uncertainty signal, confirmed ~1.7-2x higher on real shots than on in-distribution synthetic data |

### v3 sample-size sweep

`train_pff_v3_realistic_noise.py` was re-run at increasing `N_SAMPLES` (all other
config fixed) to find the accuracy/compute sweet spot:

| Samples | Wall time | Outcome |
| --- | --- | --- |
| 20k (attempt 1) | 148s | plateaued almost immediately; every metric worse than v2 |
| 60k (attempt 2) | 218s | every metric improved substantially (rel-MSE ~96x better, 1σ coverage 52/35%→75/63%); best checkpoint still epoch 4 — a better early optimum, not longer training |
| 200k (attempt 3) | 2651s | improved again (coverage 65/47%, best MAE); ~12x time for 3.3x data — likely partly run-to-run noise, not a clean scaling law |
| 500k (attempt 4) | 5928s | modest mixed improvement, coverage 65%→72% (now slightly over- rather than under-covering); time scaled ~linearly from attempt 3 — **best accuracy-per-compute-dollar**, used for every ensemble since |
| 1M (attempt 5) | 33051s | diminishing/negative returns; not used further |

### Ensemble generations

Three full 5-member ensembles have been trained on this project; only the current one
survives on disk (`out/training/pff/`):

1. **5-param** (`train_pff_ensemble_all.log`) — original constant-width bump term
   (single `a5`). `PATIENCE=80` was badly oversized: every member's true optimum landed
   at epoch 1-5 of ~83-85 run, but `EarlyStopping` then waited out up to 80 more mostly
   wasted epochs — member 3 alone burned 23416s this way.
2. **6-param, uncalibrated priors** (`train_pff_ensemble_6param_all.log`) — bump
   denominator generalised to the energy-dependent `a5*x + a6/x`. `a4` (bump centre)
   sampling prior was mean=35/std=25 over [1,49] MeV, and `a5`/`a6` bounds were an
   explicitly-flagged, unvalidated placeholder. On the 14 real shots this project
   tracks, `a4` predictions scattered 1.0-43.7 MeV with sigma 6.5-13.4 MeV.
3. **6-param, recalibrated priors (current)** — `a4` recentred to mean=15/std=3 over
   [10,20] MeV; `a5`/`a6` tightened to match `matlab/PFF.m`'s own old constant-width
   calibration (`a5 ∈ [5,15]`, i.e. σ≈1.6-2.7 MeV). `PATIENCE` also trimmed 80→25,
   matching where every member's optimum actually lands (avoids generation 1's wasted
   compute — this generation's 5 members ran 27-29 epochs each, 1735-2081s). Real-shot
   `a4` now clusters 14.3-19.8 MeV. Full derivation and the identifiability caveats that
   go with it are in `CLAUDE.md`.

### Verification

- `src/comparisons/evaluate_pff_ensemble.py` — runs all 5 members on the 14 real shots + a synthetic
  in-distribution batch, decomposes aleatoric vs. epistemic variance per parameter, and
  checks that epistemic variance is genuinely higher on real (OOD) shots. Writes
  `out/comparisons/pff_ensemble_eval.csv` and `out/comparisons/pff_ensemble_epistemic_check.png`.
- `src/inference/infer_cnn_ensemble.py` — same real-shot diagnostic figure as the CNN pipeline's
  verification script above, with a 4th panel driven by the PFF ensemble (mean ± total sigma,
  aleatoric/epistemic split, `p(bump)` ± its cross-member std). Writes to
  `out/inference/cnn_infer_ensemble/`.
