# Project notes for Claude Code

Working notes that don't belong in `README.md` (which documents usage/pipelines for
humans) — investigation findings, known caveats, and operational gotchas specific to
this repo. Read this before touching the PFF regressor or CNN spectrum classifier.

## Environment

- CPU-only TensorFlow, no GPU detected (`tf.config.list_physical_devices()` returns
  only `CPU:0`). Per-epoch training time on this machine has shown real run-to-run
  variance unrelated to sample count or architecture (e.g. one 500k-sample ensemble
  member ran at 20s/epoch, another at 275s/epoch under nominally identical config) —
  don't trust a single run's wall-clock as a scaling law without a second data point.
- Windows; PowerShell is the primary shell, Bash tool also available. Both are usable
  but use their own syntax — see the tool descriptions, not shell habits from Linux.

## PFF bump-centre (`a4`) identifiability — read before trusting `a4` output

The current ensemble (`model_pff_ensemble_0-4.keras`) predicts real-shot `a4` tightly
clustered in 14-20 MeV, matching the physically expected [10,20] MeV window. That
result is real and validated (see `README.md`'s Ensemble generations table), **but**:

- A prior-free profile-likelihood scan (`bump_center_profile_likelihood.py`, output in
  `bump_center_profile_summary.csv`) found only **2 of 12** bump-present real shots have
  their genuinely unconstrained best fit (refit every other param at each fixed `a4`
  across the full 1-49 MeV range) actually inside [10,20] MeV — several pin at the 49
  MeV grid edge. The channel-residual objective is nearly **flat** across most of the
  range for most shots (e.g. one shot: 0.001712 at its "best" a4=24 vs. 0.001730 at
  a4=15 — about 1% different).
- Forcing `a4` into [10,20] and refitting the rest (`refine_bump_center_optimizer.py`,
  output in `pff_bumpcenter_refined.csv`) showed why: `a3` (bump amplitude) and `a5`
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
(`evaluate_pff_ensemble.py` / `pff_ensemble_eval.csv`).

## `EarlyStopping` patience — check the best-epoch number before raising it

Across every PFF training run logged this project, the best checkpoint (by `spec_mse`)
has landed at epoch 1-13 of a run that then continues for 80+ more epochs before
`EarlyStopping` triggers. `PATIENCE=80` (the original ensemble script's default) cost
real time for no benefit — one member burned 23416s riding out a plateau after its true
optimum at epoch 5. `PATIENCE` was trimmed to 25 (and `MAX_EPOCHS` 500→200) partway
through this project for exactly this reason; if you add a new training variant, check
where `best_epoch` actually lands in your first run before assuming a larger patience is
buying you anything.

## PFF priors and the CNN spectrum classifier are currently on DIFFERENT priors — intentional, not a bug

`generate_spectrum_batch` (the CNN spectrum classifier's training-data source, used by
`train_cnn.py`) draws its synthetic bumps from `data_utils.PFF_PARAM_SAMPLING` via
`sample_pff_spectra` — the same table the PFF regressor ensemble uses. When the PFF
priors were recalibrated (`a4` recentred to [10,20] MeV, `a5`/`a6` tightened), the CNN
model (`model_cnn_n50.keras`) was retrained to match, evaluated, and then **explicitly
reverted back to the old-prior version** at the user's request — the new-prior CNN
inference figures are preserved in `cnn_infer_ensemble_newprior/` for reference, but the
live `model_cnn_n50.keras` does not match the live PFF ensemble's training prior. If you
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
