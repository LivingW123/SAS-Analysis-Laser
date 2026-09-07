"""
estimators.py — one uniform interface over every spectrum estimator this
project has: the DNN classifier, the DNN spectrum regressor, the CNN spectrum
regressor, the PFF parameter-regressor ensemble, and the classical TSVD
unfolding ported in `src/core/tsvd.py`.

New estimators belong here rather than in another one-off comparison script:
anything implementing `__call__(signal, sat_mask) -> Prediction` is
automatically scorable by `src/comparisons/method_comparison.py`.

Each estimator is a callable that takes ONE raw 200-channel detector vector
and returns a `Prediction`: an L1-normalised spectrum on a stated number of
energy bins over 0-50 MeV, plus whatever else that method happens to provide
(parameters, uncertainties, bump probability). `method_comparison.py` scores
them all through this interface so the comparison is genuinely apples-to-apples
-- same input vector, same output convention, same resampling to a common grid.

Why the estimators disagree so much is mostly visible in three fields of
`Prediction`:

  n_dof      -- how many free numbers the method actually fits. TSVD_NN has 7
                (its SVD-subspace coefficients) no matter how many energy bins
                it prints; the PFF family has 6 (its parameters); the CNN/DNN
                have n_bins outputs but were trained to produce PFF-shaped
                spectra, so their *effective* freedom is somewhere in between.
  sigma      -- per-parameter uncertainty, or None. Only the PFF ensemble has
                one at all.
  fits_data  -- whether the method's objective was "reproduce THIS vector"
                (TSVD) or "reproduce the training distribution" (every neural
                net here). This is the single biggest driver of the
                fit-vs-truth tradeoff the comparison figures show.

Grid convention
---------------
Every spectrum is reported on a uniform grid of `n_bins` bins spanning
0-50 MeV, L1-normalised (sums to 1), i.e. bin masses, not densities. All the
bin counts in play (10, 20, 50, 100, 200) divide 200, so `resample_spectrum`
converts between any two of them exactly and mass-preservingly by expanding to
the common 200-bin refinement and re-summing -- no interpolation, no leakage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from src.core.data_utils import (
    bin_drm,
    mev_bin_centers,
    normalize_apply,
    pff_func,
)

# Bin counts each family of models was trained at / is available at on disk.
COMMON_BINS = 50   # grid every method is resampled onto for scoring


@dataclass
class Prediction:
    """One estimator's answer for one detector vector."""
    method: str
    spectrum: np.ndarray            # (n_bins,) L1-normalised bin masses, 0-50 MeV
    n_bins: int
    n_dof: int                      # free numbers actually fitted (see module docstring)
    fits_data: bool                 # objective targeted THIS vector, vs a training distribution
    seconds: float = 0.0
    params: np.ndarray | None = None        # (6,) PFF params, if the method has them
    sigma: np.ndarray | None = None         # (6,) 1-sigma on those params, if any
    p_bump: float | None = None
    extra: dict = field(default_factory=dict)


def l1(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    return (x / s) if s > 0 else np.asarray(x, dtype=np.float64)


def resample_spectrum(spec: np.ndarray, n_dst: int) -> np.ndarray:
    """
    Exactly re-bin an L1-normalised spectrum from its own bin count to `n_dst`,
    preserving mass. Both counts must divide 200 (all of this project's do).

    Coarse -> fine splits each bin's mass evenly across its sub-bins; fine ->
    coarse sums. Implemented by going through the 200-bin common refinement so
    both directions are one code path.
    """
    n_src = len(spec)
    if n_src == n_dst:
        return np.asarray(spec, dtype=np.float64)
    assert 200 % n_src == 0 and 200 % n_dst == 0, f"{n_src}->{n_dst}: both must divide 200"
    fine = np.repeat(np.asarray(spec, dtype=np.float64) / (200 // n_src), 200 // n_src)
    return fine.reshape(n_dst, 200 // n_dst).sum(axis=1)


def reconstruction_residual_pct(
    signal_l1: np.ndarray, spectrum: np.ndarray, drm: np.ndarray,
    channels: np.ndarray | None = None,
) -> float:
    """
    How well a spectrum reproduces the measured vector, as a percentage:
    100 * RMS(signal_l1 - L1(drm_binned @ spectrum)) / RMS(signal_l1).

    Matches the convention already used by `nnls_refine.resid_pct` and
    `infer_cnn_ensemble` so numbers here are comparable to those already in
    `out/comparisons/`. Scale-free on both sides, so a method that only gets
    the spectrum's *shape* right is not penalised for its amplitude.

    `channels` restricts the RMS to a subset — pass `~sat_mask` to score only
    channels that are real measurements. The normalisation is applied AFTER the
    full-vector L1, so the two variants stay on the same scale and are directly
    comparable; only which residuals are averaged changes.
    """
    drm_b = bin_drm(drm, len(spectrum))
    resp = l1(drm_b @ spectrum)
    resid = signal_l1 - resp
    if channels is not None:
        resid, ref = resid[channels], signal_l1[channels]
    else:
        ref = signal_l1
    if resid.size == 0:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(ref ** 2)))


# ---------------------------------------------------------------------------
# Spectrum-comparison metrics (all on a common grid, both args L1-normalised)
# ---------------------------------------------------------------------------

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """0.5 * sum|p - q| in [0, 1]. 0 = identical, 1 = disjoint support."""
    return float(0.5 * np.abs(p - q).sum())


def wasserstein_mev(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-D Wasserstein (earth-mover) distance in MeV between two binned spectra.

    Unlike total variation this is sensitive to *how far* misplaced mass moved,
    which is the property that matters here: a method that puts the bump at
    25 MeV instead of 15 should score much worse than one that smears it over
    14-16 MeV, and TV cannot tell those apart. Computed as the L1 distance
    between the CDFs times the bin width.
    """
    n = len(p)
    bin_width = 50.0 / n
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() * bin_width)


def log_rms(p: np.ndarray, q: np.ndarray, floor: float = 1e-6) -> float:
    """
    RMS of log10(p/q) over bins where the reference q exceeds `floor`.

    These spectra span several orders of magnitude, so plain per-bin errors are
    dominated entirely by the low-energy bremsstrahlung peak and say nothing
    about the high-energy tail where the physics of interest lives. Working in
    log space weights every decade equally. Both are floored before the ratio
    so an exactly-zero predicted bin scores badly but finitely.
    """
    mask = q > floor
    if not mask.any():
        return float("nan")
    ratio = np.maximum(p[mask], floor) / np.maximum(q[mask], floor)
    return float(np.sqrt(np.mean(np.log10(ratio) ** 2)))


def high_energy_fraction(spec: np.ndarray, threshold_mev: float = 10.0) -> float:
    """Fraction of the spectrum's mass above `threshold_mev` — the single number
    that most directly says "is there real high-energy signal here"."""
    centres = mev_bin_centers(len(spec))
    return float(spec[centres > threshold_mev].sum() / max(spec.sum(), 1e-30))


def spectral_centroid(spec: np.ndarray, min_mev: float = 10.0) -> float:
    """
    Mass-weighted mean energy above `min_mev` — a method-agnostic stand-in for
    "where is the high-energy structure", computable for TSVD and the neural
    nets alike (none of which report a bump centre the way the PFF form does).
    NaN when there is no mass above the threshold.
    """
    centres = mev_bin_centers(len(spec))
    m = centres > min_mev
    w = spec[m]
    return float((centres[m] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class KerasSpectrumEstimator:
    """
    Wraps any of this project's Keras models whose output is already a softmax
    over energy bins: the CNN spectrum regressor (`model_cnn_n*.keras`, needs
    the 5x48 reshape), the dense spectrum regressor
    (`model_dnn_spectrum_n*.keras`), and the dense MeV classifier
    (`model_dnn_mev_n*.keras`).

    The MeV classifier is included on purpose even though it was trained on
    monoenergetic responses, not broad spectra: its softmax over energy bins is
    still a probability distribution over energy, so it is directly comparable,
    and seeing how it fails on broad-spectrum input is part of the answer to
    "which of these methods should we actually use". It is flagged
    `trained_monoenergetic=True` so figures can mark it.
    """

    def __init__(self, method: str, model_path: str, results_json: str,
                 n_bins: int, as_2d: bool, trained_monoenergetic: bool = False,
                 json_key: str | None = None) -> None:
        import tensorflow as tf

        self.method = method
        self.n_bins = n_bins
        self.as_2d = as_2d
        self.trained_monoenergetic = trained_monoenergetic
        with open(results_json) as f:
            res = json.load(f)[json_key or str(n_bins)]
        self.mean = np.array(res["norm_mean"], dtype=np.float32)
        self.std = np.array(res["norm_std"], dtype=np.float32)
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def __call__(self, signal_raw: np.ndarray, sat_mask=None) -> Prediction:
        import time
        t0 = time.perf_counter()
        signal_l1 = l1(signal_raw).astype(np.float32)
        x = normalize_apply(signal_l1.reshape(1, -1), self.mean, self.std)
        if self.as_2d:
            from src.core.cnn_model import reshape_to_2d
            x = reshape_to_2d(x)
        spec = self.model.predict(x, verbose=0)[0].astype(np.float64)
        return Prediction(
            method=self.method,
            spectrum=l1(spec),
            n_bins=self.n_bins,
            n_dof=self.n_bins,
            fits_data=False,
            seconds=time.perf_counter() - t0,
            extra={"trained_monoenergetic": self.trained_monoenergetic},
        )


class PFFEnsembleEstimator:
    """
    The 5-member PFF deep ensemble, reported on `n_bins` energy bins by
    evaluating `pff_func` at the bin centres and L1-normalising.

    `sigma` is the ensemble's total (aleatoric + epistemic) 1-sigma per
    parameter from `decode_ensemble`; `extra["sigma_epistemic"]` keeps the
    epistemic part alone, which is the number that actually signals "this input
    is unlike anything I trained on".
    """

    def __init__(self, model_dir: str = "out/training/pff", tag: str = "",
                 n_members: int = 5, n_bins: int = 200) -> None:
        import tensorflow as tf

        self.method = "PFF ensemble"
        self.n_bins = n_bins
        self.members = []
        for idx in range(n_members):
            mp = os.path.join(model_dir, f"model_pff_ensemble{tag}_{idx}.keras")
            jp = os.path.join(model_dir, f"pff_training_results_ensemble{tag}_{idx}.json")
            if not (os.path.exists(mp) and os.path.exists(jp)):
                continue
            with open(jp) as f:
                res = json.load(f)
            self.members.append({
                "model": tf.keras.models.load_model(mp, compile=False),
                "mean": np.array(res["norm_mean"], dtype=np.float32),
                "std": np.array(res["norm_std"], dtype=np.float32),
                "bounds": np.array(res["param_bounds"], dtype=np.float32),
            })
        if not self.members:
            raise FileNotFoundError(f"No PFF ensemble members found in {model_dir}")

    def __call__(self, signal_raw: np.ndarray, sat_mask=None) -> Prediction:
        import time

        from src.core.pff_ensemble_utils import decode_ensemble
        from src.core.pff_model import decode_v2

        t0 = time.perf_counter()
        signal_l1 = l1(signal_raw).astype(np.float32)
        bounds = self.members[0]["bounds"]
        mus, sigmas, pbs = [], [], []
        for mem in self.members:
            x = normalize_apply(signal_l1.reshape(1, -1).astype(np.float32),
                                mem["mean"], mem["std"])
            mu, sig, pb = decode_v2(mem["model"].predict(x, verbose=0), bounds)
            mus.append(mu[0]); sigmas.append(sig[0]); pbs.append(pb[0])
        mean, sigma_tot, sigma_epi, pb_mean, pb_std = decode_ensemble(
            np.array(mus), np.array(sigmas), np.array(pbs))
        spec = pff_func(mev_bin_centers(self.n_bins), mean)
        return Prediction(
            method=self.method,
            spectrum=l1(np.maximum(spec, 0.0)),
            n_bins=self.n_bins,
            n_dof=6,
            fits_data=False,
            seconds=time.perf_counter() - t0,
            params=mean,
            sigma=sigma_tot,
            p_bump=float(pb_mean[0]),
            extra={"sigma_epistemic": sigma_epi, "p_bump_std": float(pb_std[0])},
        )


class TSVDEstimator:
    """
    The classical baseline: `matlab/TSVD_NN.m` as ported in `src/core/tsvd.py`.

    The SVD of the grouped DRM is computed once at construction and reused for
    every shot, which is the only reason this is a class rather than a function.

    Two things to keep in mind when reading its numbers. It is the only method
    here whose objective is literally "reproduce THIS detector vector", so it
    should — and does — win on reconstruction residual; and it has no training
    distribution, no smoothness prior, and no uncertainty, so what it returns is
    whatever non-negative combination of 7 singular directions best fits the
    noise realisation in front of it, which in practice is a handful of spikes.
    """

    def __init__(self, drm: np.ndarray, gp_sz: int = 2, num_terms: int = 7,
                 clamp_from: int | None = 6, c0: str = "tsvd") -> None:
        from src.core.tsvd import group_drm, svd_basis

        self.method = "TSVD_NN"
        self.drm_g = group_drm(drm, gp_sz)
        self.U, self.s, self.V = svd_basis(self.drm_g)
        self.n_bins = self.drm_g.shape[1]
        self.num_terms = num_terms
        self.clamp_from = clamp_from
        self.c0 = c0

    def __call__(self, signal_raw: np.ndarray, sat_mask=None) -> Prediction:
        import time

        from src.core.tsvd import tsvd_nn_fit

        t0 = time.perf_counter()
        out = tsvd_nn_fit(np.asarray(signal_raw, dtype=np.float64), self.drm_g,
                          self.U, self.s, self.V, self.num_terms,
                          self.clamp_from, self.c0)
        spec = out["spectrum"]
        return Prediction(
            method=self.method,
            spectrum=l1(spec) if spec.sum() > 0 else spec,
            n_bins=self.n_bins,
            n_dof=self.num_terms,
            fits_data=True,
            seconds=time.perf_counter() - t0,
            extra={
                "rel_resid": out["rel_resid"],
                "nfev": out["nfev"],
                "n_nonzero_bins": int((spec > 0).sum()),
            },
        )


def build_default_estimators(drm: np.ndarray, include: set[str] | None = None) -> list:
    """
    Construct every estimator that has weights on disk right now.

    Missing models are skipped with a printed note rather than raising, so this
    still runs on a checkout where (say) only some bin counts were trained.
    `include`, if given, filters by method name.

    Only the *live* model of each family is built by default — the n=50 CNN,
    the n=20 dense spectrum regressor, the n=50 dense classifier, the current
    PFF ensemble. Other bin counts on disk are still loadable by name, e.g.

        KerasSpectrumEstimator("CNN n20", "out/training/cnn/model_cnn_n20.keras",
                               "out/training/cnn/cnn_training_results.json", 20, as_2d=True)

    but they are stale relative to the current priors/noise model and dilute the
    comparison (the n=20 CNN in particular puts essentially all of its mass
    above 10 MeV and scores a total variation of ~1.0 against truth).
    """
    ests: list = []

    def add(label: str, fn) -> None:
        if include is not None and label not in include:
            return
        try:
            ests.append(fn())
        except (FileNotFoundError, KeyError, OSError) as exc:
            print(f"  (skipping {label}: {exc})")

    add("CNN n50", lambda: KerasSpectrumEstimator(
        "CNN n50", "out/training/cnn/model_cnn_n50.keras",
        "out/training/cnn/cnn_training_results.json", 50, as_2d=True))
    add("DNN spectrum n20", lambda: KerasSpectrumEstimator(
        "DNN spectrum n20", "out/training/dnn/model_dnn_spectrum_n20.keras",
        "out/training/dnn/dnn_spectrum_training_results.json", 20, as_2d=False))
    add("DNN classifier n50", lambda: KerasSpectrumEstimator(
        "DNN classifier n50", "out/training/dnn/model_dnn_mev_n50.keras",
        "out/training/dnn/dnn_mev_training_results.json", 50, as_2d=False,
        trained_monoenergetic=True))
    add("PFF ensemble", lambda: PFFEnsembleEstimator())
    add("TSVD_NN", lambda: TSVDEstimator(drm))
    return ests
