# Project notes for Claude Code

Working notes that don't belong in `README.md` (which documents usage/pipelines for
humans) — investigation findings, known caveats, and operational gotchas specific to
this repo. Read this before touching the PFF regressor or CNN spectrum classifier.

## Repo layout: src/ + out/, scripts run as modules

The repo was reorganized from ~26 flat root-level scripts into `src/core/`,
`src/training/{dnn,cnn,pff}/`, `src/inference/`, `src/comparisons/`, `src/optimizers/`,
`src/visualization/`, `src/testing/`, with generated artifacts under a mirrored `out/`
tree (see `README.md`'s Layout section for the full map). Consequences worth knowing:

- Every script runs via `python -m src.<category>.<...>.<script>` from the repo root,
  never `python src/.../script.py` directly (relative imports depend on the package
  context `-m` provides).
- Shared code that used to live inside one training script and get imported by others
  (model builders, decode helpers, the real-shot loader) was extracted into
  `src/core/cnn_model.py`, `src/core/pff_model.py`, `src/core/real_shots.py` — no script
  outside `core` imports another script across category boundaries anymore. If you add a
  new script that needs, say, the PFF `build_model`/`decode_v2`, import it from
  `src.core.pff_model`, not from a trainer file.
- "mev" was renamed to "dnn" throughout (it's a plain dense/FC network, contrasted with
  the CNN family) — `train_mev.py` → `src/training/dnn/train_dnn_mev.py`,
  `model_mev_n{n}.keras` → `model_dnn_mev_n{n}.keras`, `training_results.json` →
  `dnn_mev_training_results.json`, all now under `out/training/dnn/`.
- `run_tpw_mev.py` (which both prepped raw TPW zips/TIFs *and* ran DNN inference on them)
  was split into `src/testing/prep_tpw_data.py` (extraction/preprocessing, writes
  `out/testing/tpw_prepped_spectra.npz`) and `src/testing/test_dnn_tpw.py` (pure test —
  loads that artifact + the pretrained DNNs, writes the result CSVs). Run prep first.
- Files with no producing script left after the reorganization live in `out/_unsorted/`
  (uncategorized on purpose, not lost) rather than being guessed into a bucket.
- `.log` files, the two data-prep notebooks, and all raw/reference data directories
  (`res/`, `TPW/`, `matlab/`, `sasdeconsoftware/`, `"CSU ALEPH 2025 - Liang/"`, `env/`)
  deliberately stayed at the repo root — out of scope for this reorganization.

## Environment

- CPU-only TensorFlow, no GPU detected (`tf.config.list_physical_devices()` returns
  only `CPU:0`). Per-epoch training time on this machine has shown real run-to-run
  variance unrelated to sample count or architecture (e.g. one 500k-sample ensemble
  member ran at 20s/epoch, another at 275s/epoch under nominally identical config) —
  don't trust a single run's wall-clock as a scaling law without a second data point.
- Windows; PowerShell is the primary shell, Bash tool also available. Both are usable
  but use their own syntax — see the tool descriptions, not shell habits from Linux.

## PFF bump-centre (`a4`) identifiability — read before trusting `a4` output

The current ensemble (`out/training/pff/model_pff_ensemble_0-4.keras`) predicts real-shot
`a4` tightly clustered in 14-20 MeV, matching the physically expected [10,20] MeV window.
That result is real and validated (see `README.md`'s Ensemble generations table), **but**:

- A prior-free profile-likelihood scan (`src/optimizers/bump_center_profile_likelihood.py`,
  output in `out/optimizers/bump_center_profile_summary.csv`) found only **2 of 12**
  bump-present real shots have their genuinely unconstrained best fit (refit every other
  param at each fixed `a4` across the full 1-49 MeV range) actually inside [10,20] MeV —
  several pin at the 49 MeV grid edge. The channel-residual objective is nearly **flat**
  across most of the range for most shots (e.g. one shot: 0.001712 at its "best" a4=24 vs.
  0.001730 at a4=15 — about 1% different).
- Forcing `a4` into [10,20] and refitting the rest
  (`src/optimizers/refine_bump_center_optimizer.py`, output in
  `out/optimizers/pff_bumpcenter_refined.csv`) showed why: `a3` (bump amplitude) and `a5`
  (width) would run to their ceiling values on most shots to compensate — the "bump"
  term was being used to impersonate general spectral shape/background, not localize a
  real peak. This is a genuine degeneracy between the bump term and the bremsstrahlung
  background, not just a training-prior mismatch.
- Tightening `a5`/`a6` bounds (the fix that's now live, derived from `matlab/PFF.m`'s
  old calibrated constant-width fit) barely moved the profile-likelihood result when
  re-checked — still ~2/12 in-window, residual still close to flat for most shots. The
  degeneracy is not fully resolved, possibly a boundary/edge effect near the detector's
  49 MeV limit, possibly a more fundamental limit of what this functional form can
  distinguish given the DRM's channel resolution. Not chased further this session.

**Practical implication**: the current `a4` predictions are correct in the sense of
matching the physical prior you'd want to impose, but that tightness is substantially
**prior-enforced, not purely data-derived**. Treat the ensemble's `a4` sigma as an
optimistic lower bound, not a fully calibrated measurement uncertainty, until this
degeneracy is investigated further (e.g. why some real shots' fit wants to run all the
way to the 49 MeV edge — real signal there, or an edge artifact of the finite fit
range?).

## v1 vs v2 vs v3 — the concrete numbers behind README's architecture-history table

`compare_pff_v1_v2.py` (deleted — its model dependencies no longer exist on disk,
see File-genealogy note below) ran all three real-shot checks side by side; the raw
run is preserved in `compare_pff_v1_v2_v3.log` (repo root). The headline numbers:

- **Real-shot sigma, mean/max across all 14 shots** — v1's sigma is enormous and
  unstable (a2 mean/max sigma 98.18/146.93, a3 6361/14841, a4 509/7123 — the hard-clip
  blowup). v2 brought the max down a lot (a3 13.53, a4 6.50) but...
- **Sigma variation across the 14 shots** (std of each version's own per-shot sigma;
  ~0 = collapsed to one constant regardless of input) — **v2: a1=0.0000 a2=0.0000
  a3=0.0000 a4=0.0000 a5=0.0000**, literally zero variation on every parameter across
  every shot. v1 varied wildly but for the wrong reason (blowing up, not discriminating).
  v3 is the first version with real, bounded, per-shot-varying sigma (e.g.
  a1=19.39, a4=1.75, a5=4.03 std across shots) — this is the concrete evidence behind
  the "v2 saturates at one fixed corner" line in README's architecture table.
- **Synthetic 1-sigma coverage** (want ~68%): v1 56%/10%, v2 32%/8%, v3 (500k) 72%/46%,
  v3 (1M) 74%/60%. v2 is calibrated *worse* than v1 despite fixing the blowup — being
  stuck at a constant is its own calibration failure. v3's realistic-noise generator is
  what actually fixes this, not the tanh-bounding by itself.
- **Synthetic MAE**: a1 222.5→241.3→112.3 (v1→v2→v3), a4 22.5→26.0→14.6 — v2 is not
  uniformly better than v1 on point accuracy either; v3 is the first version that wins
  on every metric simultaneously.

## The "flat sigma" pattern — recurring, not new each time you see it

Every single-model PFF regressor this project has tried (v1, v2, v3, and the standalone
prototype that became ensemble member 0) shows the same failure mode on real/OOD data:
per-parameter sigma saturates at (or very near) one fixed value across *every* real
shot, regardless of how different the shots actually are. It's not a calibrated
uncertainty in that state — it's the tanh-bounded logvar head hitting its ceiling. The
5-member ensemble is the actual fix (cross-member disagreement gives real signal), not
anything about a single model's own reported sigma. If you retrain a single model and
its real-shot sigma looks suspiciously identical across shots, that's expected, not a
new bug — check the *ensemble's* sigma_epistemic breakdown instead
(`src/comparisons/evaluate_pff_ensemble.py` / `out/comparisons/pff_ensemble_eval.csv`).

## `EarlyStopping` patience — check the best-epoch number before raising it

Starting from v3 (`src/training/pff/train_pff_v3_realistic_noise.py`) onward, every PFF
training run logged this project has had its best checkpoint (by `spec_mse`) land at
epoch 1-13 of a run that then continues for 80+ more epochs before `EarlyStopping`
triggers — concretely, the v3 sample-size sweep's best epochs were attempt2 (60k): 4/84,
attempt3 (200k): 5/85, attempt4 (500k): 5/83 (full per-attempt metrics were in
`train_pff_v3_attempt{2,3,4}.log`, now deleted — the script's own "Config history"
comment block has the condensed numbers that came from them).
`PATIENCE=80` (the original ensemble script's default) cost real time for no benefit —
one member burned 23416s riding out a plateau after its true optimum at epoch 5.
`PATIENCE` was trimmed to 25 (and `MAX_EPOCHS` 500→200) partway through this project for
exactly this reason.

**This is NOT universal, though** — v1 (`src/training/pff/train_pff.py`,
`model_pff_relu_uncertainty.keras`) and v2 (`src/training/pff/train_pff_bounded_gated.py`,
`model_pff_v2.keras`), both trained on the older
plain-Poisson noise generator (pre-`generate_pff_training_data`), used their full
`MAX_EPOCHS=300` budget productively: v1's best epoch was 82, v2's was 299 — still
improving when the run was cut off. So the "wasted-epoch" pattern only shows up once you
switch to the realistic noise+saturation generator, not from the architecture change
(v2 and v3 share the exact same `build_model`/loss, only the generator differs) — most
likely that generator makes the val_loss signal noisier/harder in a way that lets the
model find a good-enough optimum fast and then just oscillate, though that's a plausible
read of the pattern, not something separately verified. If you add a new training
variant on the *old* generator, don't assume `best_epoch` will be early — check first.

## PFF priors and the CNN spectrum classifier are currently on DIFFERENT priors — intentional, not a bug

`generate_spectrum_batch` (the CNN spectrum classifier's training-data source, used by
`src/training/cnn/train_cnn.py`) draws its synthetic bumps from
`src.core.data_utils.PFF_PARAM_SAMPLING` via `sample_pff_spectra` — the same table the
PFF regressor ensemble uses. When the PFF priors were recalibrated (`a4` recentred to
[10,20] MeV, `a5`/`a6` tightened), the CNN model (`out/training/cnn/model_cnn_n50.keras`)
was retrained to match, evaluated, and then **explicitly reverted back to the old-prior
version** at the user's request — the new-prior CNN inference figures are preserved in
`out/inference/cnn_infer_ensemble_newprior/` for reference, but the live
`model_cnn_n50.keras` does not match the live PFF ensemble's training prior. If you
retrain the CNN for any other reason, decide deliberately whether it should pick up the
current (recalibrated) priors or stay matched to its own historical baseline — don't
assume they're already in sync.

## File-genealogy note

Older PFF model/results generations (v1 `relu_uncertainty`, v2, v3 + its `attempt1-5`
sample-size sweep, the original 5-param ensemble, and the uncalibrated-prior 6-param
ensemble) were deleted after their history was condensed into `README.md`'s PFF
Parameter Regressor Pipeline section. Their `.log` files are still on disk (not
deleted) and are the source for every number cited there and in this file. If you need
the actual old weights rather than just the metrics, they're recoverable from git
history (the commit that added `bump_center_profile_likelihood.py` etc. — search
`git log` for when `model_pff_v3.keras` etc. were last present) rather than on disk.
Note this predates the src/+out/ reorganization (see the Repo layout section above), so
git-history paths for these files will show the old flat root-level names, not the
current `src/...`/`out/...` locations.
