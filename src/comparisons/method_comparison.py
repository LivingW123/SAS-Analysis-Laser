"""
method_comparison.py — score every spectrum estimator in this project against
the classical TSVD unfolding reference (`matlab/TSVD_NN.m`, ported to
`src/core/tsvd.py`) on one common footing.

Methods compared (whichever have weights on disk; see
`src.core.estimators.build_default_estimators`):

  TSVD_NN            classical truncated-SVD unfolding, 7 SVD-subspace dof,
                     fitted directly to the shot in front of it, no prior,
                     no uncertainty
  CNN n50 / n20      Conv2D spectrum regressor
  DNN spectrum n20   dense spectrum regressor
  DNN classifier n50 dense monoenergetic-bin classifier, softmax read as a
                     spectrum (out of distribution on broad spectra by
                     construction — included to quantify by how much)
  PFF ensemble       5-member deep-ensemble PFF parameter regressor, the only
                     one of the methods here carrying an uncertainty

Three benchmarks, because no one of them settles the question
--------------------------------------------------------------
1. `synthetic-inprior` — truth spectra drawn from the SAME generator the
   neural nets trained on (`sample_pff_spectra` + the calibrated noise and
   saturation model). Ground truth is known, so accuracy is measurable, but
   this is the neural nets' home turf: they have seen this distribution and
   TSVD has seen nothing. Read it as an upper bound on the nets.
2. `synthetic-offprior` — truth spectra drawn OUTSIDE the recalibrated PFF
   prior (bump centre uniform over 5-45 MeV rather than the [10,20] window
   the current ensemble was trained on, wider widths, wider amplitudes).
   Same known truth, same noise model. This is the test that separates "the
   model learned the inverse problem" from "the model learned the prior", and
   it is where a prior-free method like TSVD should claw back ground.
3. `real` — the 14 real shots this project tracks. No ground truth, so only
   self-consistency measures are available (how well each spectrum reproduces
   the measured vector, and how much the methods agree with each other).

Metrics
-------
Everything is computed on a common 50-bin 0-50 MeV grid after exact
mass-preserving resampling (`estimators.resample_spectrum`), with both spectra
L1-normalised, so methods that natively output 20, 50, 100 or 200 bins are
compared without any of them being advantaged by its grid.

  wasserstein_mev   earth-mover distance to truth, in MeV — the headline
                    accuracy number, because it is the one that punishes
                    putting the high-energy structure in the wrong place
  total_variation   0.5*sum|p-q|, shape mismatch irrespective of distance
  log_rms           RMS of log10(pred/truth) over populated bins — the
                    tail-sensitive one; these spectra span decades
  hi10_frac_err     error in the fraction of spectrum mass above 10 MeV
  resid_pct         reconstruction residual against the MEASURED vector,
                    computable with or without truth — the only accuracy-like
                    number available on real shots, and the one TSVD optimises
  resid_unsat_pct   the same, restricted to channels that are real measurements
                    rather than saturation-correction imputations (real shots
                    have 0-92 of 200 channels imputed); this is the fairer of
                    the two on real data
  n_eff_bins        1/sum(p^2), the participation ratio: how many bins the
                    answer effectively spreads over. A delta-like spike scores
                    ~1. This is how the ill-posedness shows up numerically.

Usage
-----
  python -m src.comparisons.method_comparison                 # all three suites
  python -m src.comparisons.method_comparison --suites real   # just the real shots
  MC_N_SYNTH=100 python -m src.comparisons.method_comparison  # smaller synthetic run

Writes CSVs + figures into out/comparisons/method_comparison/.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.core.data_utils import (
    add_detector_noise,
    apply_saturation,
    bin_drm,
    load_drm,
    load_saturation_mask,
    mev_bin_centers,
    pff_func,
    sample_pff_spectra,
)
from src.core.estimators import (
    COMMON_BINS,
    build_default_estimators,
    high_energy_fraction,
    l1,
    log_rms,
    reconstruction_residual_pct,
    resample_spectrum,
    spectral_centroid,
    total_variation,
    wasserstein_mev,
)
from src.core.real_shots import SHOTS, load_signal

DRM_PATH = "res/drm/200x200.xlsx"
OUT_DIR = "out/comparisons/method_comparison"
N_SYNTH = int(os.environ.get("MC_N_SYNTH", 300))
SEED = 12345

# Canonical method order — figures assign colours by this order so a method
# keeps its colour across every figure even when a suite is missing one.
METHOD_ORDER = [
    "TSVD_NN",
    "CNN n50",
    "DNN spectrum n20",
    "DNN classifier n50",
    "PFF ensemble",
    "CNN n20",
]
# Not an estimator: the TRUE spectrum scored as if it were one. On a synthetic
# suite this is the reference every other row should be read against — it is
# what a perfect method scores, and in particular its resid_pct is the residual
# left over by the noise realisation alone. Any method scoring BELOW it on
# resid_pct is fitting noise, not signal, which is the whole argument for why
# "best reconstruction residual" does not mean "best answer".
TRUTH_LABEL = "truth (reference)"
# dataviz categorical palette, light mode, assigned in fixed slot order.
# Validated (adjacent CVD dE 9.1, normal-vision 19.6); three slots sit under
# 3:1 contrast on white, so every figure carries visible labels and a CSV
# table beside it, per the relief rule.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# Secondary (non-colour) encoding so the series stay separable in CVD/print.
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]
MARKERS = ["o", "s", "^", "v", "D", "P", "X"]

GRID_KW = dict(color="#d8d7d2", lw=0.6, alpha=0.9)


def style(method: str) -> dict:
    i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER) - 1
    return dict(color=PALETTE[i % len(PALETTE)],
                ls=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)])


def tidy(ax, *, xlabel="", ylabel="", title="") -> None:
    """Recessive grid + axes, per the dataviz mark spec."""
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c8b86")
    ax.tick_params(colors="#52514e", labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    if title:
        ax.set_title(title, fontsize=10, color="#0b0b0b")


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

def sample_offprior_params(n: int, rng: np.random.Generator,
                           bump_fraction: float = 0.5) -> np.ndarray:
    """
    Truth parameters drawn deliberately OUTSIDE the current training prior.

    `PFF_PARAM_SAMPLING` currently centres the bump at 15 +/- 3 MeV over a
    [7,23] window, with tight widths — a recalibration that `CLAUDE.md` flags
    as substantially prior-enforced rather than data-derived. Any model trained
    on it will place bumps in that window whether or not the data asked for it,
    and an in-prior benchmark cannot detect that. Here a4 is uniform over
    5-45 MeV and the widths/amplitudes are widened well past the training
    bounds, so a method that is merely reproducing its prior gets caught.
    """
    a1 = 10.0 ** rng.uniform(np.log10(10.0), np.log10(500.0), n)
    a2 = rng.uniform(0.05, 0.8, n)
    a3 = rng.uniform(5.0, 100.0, n)
    a3[rng.random(n) >= bump_fraction] = 0.0
    a4 = rng.uniform(5.0, 45.0, n)
    a5 = rng.uniform(0.02, 0.5, n)
    a6 = rng.uniform(20.0, 300.0, n)
    return np.column_stack([a1, a2, a3, a4, a5, a6])


def make_synthetic_suite(drm: np.ndarray, n: int, rng: np.random.Generator,
                         off_prior: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build `n` synthetic shots: truth spectra on the DRM's own 200-bin energy
    grid, forward-projected and passed through the same noise + saturation
    model the neural nets were trained against
    (`add_detector_noise` / `apply_saturation`).

    Returns (signals_raw (n,200), truth_spectra (n,200), truth_params (n,6)).
    Signals are left in raw detector-count units, not L1-normalised: TSVD is
    scale-free but reads more naturally on counts, and every estimator
    normalises internally anyway.
    """
    energy = mev_bin_centers(drm.shape[1])
    if off_prior:
        params = sample_offprior_params(n, rng)
        spectra = np.stack([pff_func(energy, p) for p in params])
    else:
        spectra, params = sample_pff_spectra(n, energy, rng, bump_fraction=0.5)
    responses = (drm @ spectra.T).T
    noisy = add_detector_noise(responses, rng)
    noisy, _ = apply_saturation(noisy, rng)
    return noisy.astype(np.float64), spectra, params


def nnls_floor_pct(signal_l1: np.ndarray, drm: np.ndarray, n_bins: int) -> float:
    """
    Best reconstruction residual any non-negative spectrum on `n_bins` bins can
    reach for this vector — the yardstick every method's resid_pct should be
    read against. Column-normalising the DRM first (see `nnls_refine`'s
    docstring) keeps the solve well conditioned.
    """
    drm_b = bin_drm(drm, n_bins)
    drm_n = drm_b / drm_b.sum(axis=0, keepdims=True)
    spec, _ = nnls(drm_n, signal_l1)
    resp = l1(drm_n @ spec)
    return float(100.0 * np.sqrt(np.mean((signal_l1 - resp) ** 2))
                 / np.sqrt(np.mean(signal_l1 ** 2)))


def n_eff_bins(spec: np.ndarray) -> float:
    """Participation ratio 1/sum(p^2): the effective number of occupied bins.
    ~1 for a delta spike, ~len(spec) for a flat spectrum."""
    p = l1(spec)
    return float(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_suite(name: str, signals: np.ndarray, estimators: list, drm: np.ndarray,
                truth_spectra: np.ndarray | None = None,
                truth_params: np.ndarray | None = None,
                shot_names: list[str] | None = None,
                sat_masks: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Run every estimator on every signal in a suite and tabulate the metrics.

    Returns (long-format DataFrame, {(shot, method): Prediction}) — the raw
    predictions are kept so the figure code can plot spectra without re-running
    anything.
    """
    rows: list[dict] = []
    preds: dict = {}
    n_shots = len(signals)
    print(f"\n=== suite '{name}': {n_shots} shots x {len(estimators)} methods ===")

    for i in range(n_shots):
        sig = np.asarray(signals[i], dtype=np.float64)
        shot = shot_names[i] if shot_names else f"{name}_{i:04d}"
        sig_l1 = l1(sig)
        mask = sat_masks[i] if sat_masks is not None else None
        truth_c = (resample_spectrum(l1(truth_spectra[i]), COMMON_BINS)
                   if truth_spectra is not None else None)

        for est in estimators:
            pred = est(sig, mask)
            preds[(shot, pred.method)] = pred
            spec_c = resample_spectrum(pred.spectrum, COMMON_BINS)
            row = {
                "suite": name,
                "shot": shot,
                "method": pred.method,
                "n_bins": pred.n_bins,
                "n_dof": pred.n_dof,
                "fits_data": pred.fits_data,
                "seconds": pred.seconds,
                "resid_pct": reconstruction_residual_pct(sig_l1, pred.spectrum, drm),
                # Same residual over only the channels that are real
                # measurements. Real shots have 0-92 of 200 channels replaced by
                # a Gaussian fit during saturation correction, and scoring a
                # method on how well it reproduces an imputed value rewards
                # agreeing with that imputation, not with the detector.
                "resid_unsat_pct": reconstruction_residual_pct(
                    sig_l1, pred.spectrum, drm, channels=~mask if mask is not None else None),
                "n_imputed_channels": int(mask.sum()) if mask is not None else 0,
                "hi10_frac": high_energy_fraction(spec_c),
                "centroid_mev": spectral_centroid(spec_c),
                "n_eff_bins": n_eff_bins(spec_c),
            }
            if truth_c is not None:
                row.update({
                    "wasserstein_mev": wasserstein_mev(spec_c, truth_c),
                    "total_variation": total_variation(spec_c, truth_c),
                    "log_rms": log_rms(spec_c, truth_c),
                    "hi10_frac_err": abs(high_energy_fraction(spec_c)
                                         - high_energy_fraction(truth_c)),
                    "truth_hi10_frac": high_energy_fraction(truth_c),
                    "truth_n_eff_bins": n_eff_bins(truth_c),
                })
                if truth_params is not None:
                    has_bump = bool(truth_params[i, 2] > 0)
                    row["truth_a4"] = float(truth_params[i, 3])
                    row["truth_a3"] = float(truth_params[i, 2])
                    row["truth_has_bump"] = has_bump
                    # a4 is the CENTRE of a bump; on a no-bump truth there is no
                    # such thing to be right or wrong about, so those shots are
                    # left out of the a4 error and coverage statistics entirely
                    # rather than scored against a meaningless target.
                    if pred.params is not None and has_bump:
                        row["pred_a4"] = float(pred.params[3])
                        row["a4_err"] = abs(float(pred.params[3]) - float(truth_params[i, 3]))
                        if pred.sigma is not None:
                            row["pred_a4_sigma"] = float(pred.sigma[3])
                            row["a4_within_1sig"] = bool(
                                row["a4_err"] <= max(float(pred.sigma[3]), 1e-9))
            rows.append(row)

        if truth_c is not None:
            # Score the true spectrum itself, so every figure has the "perfect
            # answer" reference line -- see TRUTH_LABEL.
            rows.append({
                "suite": name, "shot": shot, "method": TRUTH_LABEL,
                "n_bins": len(truth_spectra[i]), "n_dof": 6, "fits_data": False,
                "seconds": 0.0,
                "resid_pct": reconstruction_residual_pct(sig_l1, l1(truth_spectra[i]), drm),
                "resid_unsat_pct": reconstruction_residual_pct(
                    sig_l1, l1(truth_spectra[i]), drm,
                    channels=~mask if mask is not None else None),
                "n_imputed_channels": int(mask.sum()) if mask is not None else 0,
                "hi10_frac": high_energy_fraction(truth_c),
                "centroid_mev": spectral_centroid(truth_c),
                "n_eff_bins": n_eff_bins(truth_c),
                "wasserstein_mev": 0.0, "total_variation": 0.0, "log_rms": 0.0,
                "hi10_frac_err": 0.0,
                "truth_hi10_frac": high_energy_fraction(truth_c),
                "truth_n_eff_bins": n_eff_bins(truth_c),
                "truth_a4": float(truth_params[i, 3]) if truth_params is not None else np.nan,
                "truth_has_bump": bool(truth_params[i, 2] > 0) if truth_params is not None else None,
            })
        else:
            floor = nnls_floor_pct(sig_l1, drm, COMMON_BINS)
            for r in rows:
                if r["shot"] == shot:
                    r["nnls_floor_pct"] = floor
        if (i + 1) % 25 == 0 or i == n_shots - 1:
            print(f"  {i+1}/{n_shots} shots done")

    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_accuracy_boxes(df: pd.DataFrame, suite: str, path: str) -> None:
    """Per-method distributions of the four accuracy metrics on a truth-bearing
    suite. Box plots (not bars) because the spread across shots is the point:
    several methods have similar medians and wildly different tails."""
    d = df[df.suite == suite]
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    truth_resid = d[d.method == TRUTH_LABEL].resid_pct.median()
    panels = [
        ("wasserstein_mev", "Earth-mover distance to truth (MeV)\nlower = better", False),
        ("log_rms", "RMS log10(pred/truth) over populated bins\nlower = better", False),
        ("total_variation", "Total variation from truth (0-1)\nlower = better", False),
        ("resid_pct", "Reconstruction residual vs measured vector (%)\nlower = fits the data better", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title, logy) in zip(axes.ravel(), panels):
        data = [d[d.method == m][col].dropna().values for m in methods]
        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color="#0b0b0b", lw=1.6),
                        flierprops=dict(marker=".", ms=3, alpha=0.4,
                                        markerfacecolor="#8c8b86",
                                        markeredgecolor="none"))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(style(m)["color"])
            patch.set_alpha(0.55)
            patch.set_edgecolor(style(m)["color"])
        for med, m in zip(bp["medians"], methods):
            x = med.get_xdata().mean()
            y = med.get_ydata()[0]
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5, color="#0b0b0b")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=8)
        if logy:
            ax.set_yscale("log")
        if col == "resid_pct" and np.isfinite(truth_resid):
            ax.axhline(truth_resid, color="#0b0b0b", ls="--", lw=1.6)
            ax.annotate(f"the TRUE spectrum scores {truth_resid:.0f}% —\n"
                        "below this line is fitting noise",
                        (0.98, truth_resid), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 8),
                        ha="right", fontsize=8, color="#0b0b0b",
                        bbox=dict(fc="#fcfcfb", ec="none", alpha=0.8, pad=1.5))
        tidy(ax, title=title)
    fig.suptitle(f"Accuracy against known truth — suite '{suite}'  "
                 f"({d.shot.nunique()} synthetic shots)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_fit_vs_truth(df: pd.DataFrame, suite: str, path: str) -> None:
    """
    The central figure: reconstruction residual (how well a method fits the
    measured vector) against earth-mover distance from the true spectrum.

    Facetted one panel per method rather than one scatter with six colours —
    scatter puts every pair of series on screen simultaneously, which the
    categorical palette only clears for three slots.
    """
    d = df[df.suite == suite]
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    ncol = 3
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xlim = (0, min(120, float(np.nanpercentile(d.resid_pct, 99)) * 1.1))
    ylim = (0, float(np.nanpercentile(d.wasserstein_mev, 99)) * 1.1)
    for ax, m in zip(axes, methods):
        sub = d[d.method == m]
        st = style(m)
        ax.scatter(sub.resid_pct, sub.wasserstein_mev, s=14, alpha=0.45,
                   color=st["color"], edgecolors="none")
        mx, my = sub.resid_pct.median(), sub.wasserstein_mev.median()
        ax.scatter([mx], [my], s=170, color=st["color"], marker=st["marker"],
                   edgecolors="#fcfcfb", linewidths=2, zorder=5)
        ax.annotate(f"median\n{mx:.0f}% resid\n{my:.1f} MeV err",
                    (mx, my), textcoords="offset points", xytext=(10, 8),
                    fontsize=8, color="#0b0b0b",
                    bbox=dict(fc="#fcfcfb", ec="none", alpha=0.75, pad=1.5))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        tidy(ax, xlabel="reconstruction residual (%)",
             ylabel="earth-mover error vs truth (MeV)", title=m)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle(f"Fitting the data is not the same as being right — suite '{suite}'\n"
                 "left = reproduces the measured vector well · bottom = recovers the true spectrum well",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_examples(preds: dict, signals: np.ndarray, truth: np.ndarray,
                 shot_names: list[str], drm: np.ndarray, picks: list[int],
                 path: str, suite: str) -> None:
    """Per-example spectrum overlays (left) and detector-space reconstructions
    (right), for a handful of representative synthetic shots."""
    energy_c = mev_bin_centers(COMMON_BINS)
    channels = np.arange(1, 201)
    methods = [m for m in METHOD_ORDER
               if any(k[1] == m for k in preds)]
    fig, axes = plt.subplots(len(picks), 2, figsize=(14, 3.4 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(picks):
        shot = shot_names[idx]
        t = resample_spectrum(l1(truth[idx]), COMMON_BINS)
        ax = axes[r, 0]
        ax.step(energy_c, t, where="mid", color="#0b0b0b", lw=2.4, label="truth", zorder=6)
        for m in methods:
            p = preds.get((shot, m))
            if p is None:
                continue
            st = style(m)
            ax.step(energy_c, resample_spectrum(p.spectrum, COMMON_BINS), where="mid",
                    color=st["color"], ls=st["ls"], lw=1.7, label=m, alpha=0.95)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(0, 50)
        tidy(ax, xlabel="energy (MeV)", ylabel="normalised bin mass (log)",
             title=f"{shot} — spectrum vs truth")
        if r == 0:
            ax.legend(fontsize=7.5, ncol=2, frameon=False)

        ax = axes[r, 1]
        sig_l1 = l1(signals[idx])
        ax.plot(channels, sig_l1, color="#0b0b0b", lw=2.0, label="measured (noisy)")
        for m in methods:
            p = preds.get((shot, m))
            if p is None:
                continue
            st = style(m)
            drm_b = bin_drm(drm, p.n_bins)
            ax.plot(channels, l1(drm_b @ p.spectrum), color=st["color"], ls=st["ls"],
                    lw=1.4, alpha=0.9, label=m)
        tidy(ax, xlabel="detector channel", ylabel="normalised response",
             title=f"{shot} — detector-space fit")
        if r == 0:
            ax.legend(fontsize=7.5, ncol=2, frameon=False)
    fig.suptitle(f"Representative shots — suite '{suite}'", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_real_spectra(preds: dict, shot_names: list[str], path: str) -> None:
    """All 14 real shots, every method's spectrum overlaid. No truth line here —
    that is the point of the figure: on real data the methods disagree and
    nothing in the data says which is right."""
    energy_c = mev_bin_centers(COMMON_BINS)
    methods = [m for m in METHOD_ORDER if any(k[1] == m for k in preds)]
    ncol = 3
    nrow = int(np.ceil(len(shot_names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, shot in zip(axes, shot_names):
        for m in methods:
            p = preds.get((shot, m))
            if p is None:
                continue
            st = style(m)
            ax.step(energy_c, resample_spectrum(p.spectrum, COMMON_BINS), where="mid",
                    color=st["color"], ls=st["ls"], lw=1.6, label=m, alpha=0.95)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(0, 50)
        tidy(ax, xlabel="energy (MeV)", ylabel="bin mass (log)", title=shot)
    for ax in axes[len(shot_names):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Real shots — predicted spectrum by method (no ground truth exists here)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_real_residuals(df: pd.DataFrame, path: str) -> None:
    """Grouped bars of reconstruction residual per real shot, with the NNLS
    floor drawn as the reference line each method is trying to reach."""
    d = df[df.suite == "real"]
    shots = list(dict.fromkeys(d.shot))
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    y = np.arange(len(shots))
    h = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 0.40 * len(shots) * len(methods) / 3 + 2.6))
    for k, m in enumerate(methods):
        vals = [float(d[(d.shot == s) & (d.method == m)].resid_pct.mean()) for s in shots]
        st = style(m)
        ax.barh(y + k * h - 0.4 + h / 2, vals, height=h * 0.92, color=st["color"],
                alpha=0.85, label=m, edgecolor="#fcfcfb", linewidth=0.8)
    floors = [float(d[d.shot == s].nnls_floor_pct.iloc[0]) for s in shots]
    for i, f in enumerate(floors):
        ax.plot([f, f], [i - 0.45, i + 0.45], color="#0b0b0b", lw=2.0,
                label="NNLS floor (best possible)" if i == 0 else None, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(shots, fontsize=9)
    ax.invert_yaxis()
    tidy(ax, xlabel="reconstruction residual (% of signal RMS) — lower is a better fit to the measurement")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    ax.set_title("Real shots: how well each method's spectrum reproduces the measured vector\n"
                 "(the black bar is the floor no non-negative spectrum can beat)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_tsvd_diagnostics(drm: np.ndarray, preds: dict, df: pd.DataFrame,
                         shot_names: list[str], path: str) -> None:
    """Why TSVD fits best and is still not the answer: the singular spectrum it
    truncates, and the spikiness of what comes out."""
    from src.core.tsvd import CLAMP_FROM, NUM_TERMS, group_drm, svd_basis

    g = group_drm(drm)
    _, s, _ = svd_basis(g)
    energy_c = mev_bin_centers(COMMON_BINS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(s) + 1), s, color="#2a78d6", lw=2, marker="o", ms=3)
    ax.axvline(NUM_TERMS + 0.5, color="#0b0b0b", ls="--", lw=1.4)
    ax.annotate(f"truncation at {NUM_TERMS} terms\n(everything right of here\nis discarded)",
                (NUM_TERMS + 1.5, s[0] * 0.05), fontsize=8, color="#0b0b0b")
    ax.axvline(CLAMP_FROM + 0.5, color="#eb6834", ls=":", lw=1.4)
    ax.annotate(f"divisor clamped to s{CLAMP_FROM}\nbeyond here", (CLAMP_FROM + 1.5, s[0] * 1e-4),
                fontsize=8, color="#eb6834")
    tidy(ax, xlabel="singular index", ylabel="singular value (log)",
         title=f"Grouped DRM singular spectrum\ncondition number {s[0]/s[-1]:.1e}")
    ax.set_xlim(0, 40)

    ax = axes[1]
    d = df[df.suite == "real"]
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    vals = [d[d.method == m].n_eff_bins.values for m in methods]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#0b0b0b", lw=1.6))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(style(m)["color"])
        patch.set_alpha(0.55)
        patch.set_edgecolor(style(m)["color"])
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    tidy(ax, ylabel=f"effective occupied bins (of {COMMON_BINS})",
         title="How spread out each answer is\n(1 = a single spike)")

    ax = axes[2]
    for k, shot in enumerate(shot_names[:3]):
        p = preds.get((shot, "TSVD_NN"))
        if p is None:
            continue
        ax.step(energy_c, resample_spectrum(p.spectrum, COMMON_BINS), where="mid",
                lw=1.8, label=shot, color=PALETTE[k], ls=LINESTYLES[k])
    tidy(ax, xlabel="energy (MeV)", ylabel="normalised bin mass",
         title="TSVD_NN output on real shots\n(non-negativity + 7 dof ⇒ spikes)")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("TSVD_NN diagnostics — the classical method's failure mode is spikiness, not bias",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_prior_dependence(df: pd.DataFrame, path: str) -> None:
    """
    In-prior vs off-prior accuracy, side by side. The gap between the two bars
    for one method is exactly how much of its apparent accuracy came from the
    training prior rather than from the measurement.
    """
    suites = [s for s in ("synthetic-inprior", "synthetic-offprior") if s in set(df.suite)]
    if len(suites) < 2:
        return
    d = df[df.suite.isin(suites)]
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    x = np.arange(len(methods))
    w = 0.38
    suite_color = {suites[0]: "#2a78d6", suites[1]: "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, col, lab in (
        (axes[0], "wasserstein_mev", "earth-mover error vs truth (MeV)"),
        (axes[1], "resid_pct", "reconstruction residual (%)"),
    ):
        for suite in suites:
            vals = [d[(d.method == m) & (d.suite == suite)][col].median() for m in methods]
            bars = ax.bar(x + (0 if suite == suites[0] else 1) * w - w / 2, vals,
                          width=w * 0.92, color=suite_color[suite], alpha=0.9,
                          label=suite, edgecolor="#fcfcfb", linewidth=1)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7.5, color="#0b0b0b")
        if col == "resid_pct":
            for k, suite in enumerate(suites):
                tv = d[(d.method == TRUTH_LABEL) & (d.suite == suite)].resid_pct.median()
                if np.isfinite(tv):
                    ax.axhline(tv, color=suite_color[suite], ls="--", lw=1.4, alpha=0.9)
                    ax.annotate(f"true spectrum ({suite.split('-')[-1]}): {tv:.0f}%",
                                (-0.45, tv), ha="left",
                                va="bottom" if k == 0 else "top",
                                fontsize=7.5, color="#0b0b0b",
                                bbox=dict(fc="#fcfcfb", ec="none", alpha=0.8, pad=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=8)
        ax.margins(y=0.18)          # headroom so the value labels are not clipped
        tidy(ax, ylabel=f"median {lab}")
        ax.legend(fontsize=9, frameon=False)

    # Third panel: the parameter-level version of the same question, for the
    # methods that report a bump centre at all. Bar = median |a4 error|,
    # annotation = what fraction of shots the method's own 1-sigma covered.
    ax = axes[2]
    a4_methods = [m for m in methods
                  if d[d.method == m]["a4_err"].notna().any()] if "a4_err" in d else []
    if a4_methods:
        xa = np.arange(len(a4_methods))
        for suite in suites:
            vals, covs = [], []
            for m in a4_methods:
                sub = d[(d.method == m) & (d.suite == suite)]
                vals.append(sub["a4_err"].median())
                covs.append(100 * sub["a4_within_1sig"].mean()
                            if "a4_within_1sig" in sub else np.nan)
            bars = ax.bar(xa + (0 if suite == suites[0] else 1) * w - w / 2, vals,
                          width=w * 0.92, color=suite_color[suite], alpha=0.9,
                          label=suite, edgecolor="#fcfcfb", linewidth=1)
            for b, v, c in zip(bars, vals, covs):
                txt = f"{v:.1f} MeV" + (f"\n1σ covers {c:.0f}%" if np.isfinite(c) else "")
                ax.annotate(txt, (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7.5, color="#0b0b0b")
        ax.set_xticks(xa)
        ax.set_xticklabels(a4_methods, rotation=18, ha="right", fontsize=8)
        ax.set_xlim(-0.7, len(a4_methods) - 0.3)
        ax.margins(y=0.25)
        tidy(ax, ylabel="median |a4 error| (MeV)")
        ax.legend(fontsize=9, frameon=False)
        ax.set_title("Bump centre, for the methods that report one\n"
                     "(1σ coverage should be ~68% if the uncertainty is honest)",
                     fontsize=9.5)
    else:
        ax.axis("off")

    fig.suptitle("How much of the accuracy is the prior?  in-prior truth vs off-prior truth\n"
                 "(off-prior draws the bump centre uniformly over 5-45 MeV, outside the "
                 "trained [10,20] MeV window)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig_agreement(df: pd.DataFrame, preds: dict, shot_names: list[str], path: str) -> None:
    """
    Pairwise method agreement on the real shots (median earth-mover distance
    between each pair's spectra). With no ground truth this is the only
    cross-check available: methods that agree are at least not failing
    independently.
    """
    methods = [m for m in METHOD_ORDER if any(k[1] == m for k in preds)]
    n = len(methods)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            ds = []
            for shot in shot_names:
                pa, pb = preds.get((shot, a)), preds.get((shot, b))
                if pa is None or pb is None:
                    continue
                ds.append(wasserstein_mev(resample_spectrum(pa.spectrum, COMMON_BINS),
                                          resample_spectrum(pb.spectrum, COMMON_BINS)))
            if ds:
                M[i, j] = float(np.median(ds))
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    im = ax.imshow(M, cmap="Blues", vmin=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(methods, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.annotate(f"{M[i,j]:.1f}", (j, i), ha="center", va="center",
                            fontsize=8,
                            color="#ffffff" if M[i, j] > np.nanmax(M) * 0.55 else "#0b0b0b")
    ax.set_title("Median disagreement between methods on the 14 real shots\n"
                 "(earth-mover distance in MeV; 0 = identical spectra)", fontsize=11)
    fig.colorbar(im, ax=ax, label="MeV")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Median of every metric per (suite, method) — the table the figures are
    the visual form of, saved beside them per the palette's relief rule."""
    d = df
    cols = [c for c in ("wasserstein_mev", "total_variation", "log_rms", "hi10_frac_err",
                        "resid_pct", "resid_unsat_pct", "n_eff_bins", "seconds",
                        "a4_err", "n_dof")
            if c in d.columns]
    g = d.groupby(["suite", "method"])[cols].median().reset_index()
    n = d.groupby(["suite", "method"])["shot"].nunique().reset_index(name="n_shots")
    g = g.merge(n, on=["suite", "method"], how="left")
    if "a4_within_1sig" in d.columns:
        cov = (d.groupby(["suite", "method"])["a4_within_1sig"]
                 .mean().reset_index().rename(columns={"a4_within_1sig": "a4_1sig_coverage"}))
        g = g.merge(cov, on=["suite", "method"], how="left")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*",
                    default=["synthetic-inprior", "synthetic-offprior", "real"])
    ap.add_argument("--n-synth", type=int, default=N_SYNTH)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("Loading DRM and estimators...")
    drm = load_drm(DRM_PATH)
    estimators = build_default_estimators(drm)
    print("  methods: " + ", ".join(e.method for e in estimators))

    all_df: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED)

    for suite in args.suites:
        if suite.startswith("synthetic"):
            off = suite.endswith("offprior")
            sigs, truth, params = make_synthetic_suite(drm, args.n_synth, rng, off)
            names = [f"{'off' if off else 'in'}prior_{i:04d}" for i in range(len(sigs))]
            df, preds = score_suite(suite, sigs, estimators, drm, truth, params, names)
            all_df.append(df)
            df.to_csv(os.path.join(OUT_DIR, f"metrics_{suite}.csv"), index=False)
            fig_accuracy_boxes(df, suite, os.path.join(OUT_DIR, f"accuracy_{suite}.png"))
            fig_fit_vs_truth(df, suite, os.path.join(OUT_DIR, f"fit_vs_truth_{suite}.png"))
            # pick 3 illustrative shots: no bump, bump low, bump high
            has_bump = params[:, 2] > 0
            picks = []
            if (~has_bump).any():
                picks.append(int(np.flatnonzero(~has_bump)[0]))
            if has_bump.any():
                bump_idx = np.flatnonzero(has_bump)
                order = bump_idx[np.argsort(params[bump_idx, 3])]
                picks += [int(order[0]), int(order[-1])]
            fig_examples(preds, sigs, truth, names, drm, picks,
                         os.path.join(OUT_DIR, f"examples_{suite}.png"), suite)
        else:
            names = [n for n, _ in SHOTS]
            sigs = np.stack([load_signal(p).astype(np.float64) for _, p in SHOTS])
            masks = [load_saturation_mask(p) for _, p in SHOTS]
            df, preds = score_suite("real", sigs, estimators, drm,
                                    shot_names=names, sat_masks=masks)
            all_df.append(df)
            df.to_csv(os.path.join(OUT_DIR, "metrics_real.csv"), index=False)
            fig_real_spectra(preds, names, os.path.join(OUT_DIR, "real_spectra.png"))
            fig_real_residuals(df, os.path.join(OUT_DIR, "real_residuals.png"))
            fig_tsvd_diagnostics(drm, preds, df, names,
                                 os.path.join(OUT_DIR, "tsvd_diagnostics.png"))
            fig_agreement(df, preds, names, os.path.join(OUT_DIR, "real_agreement.png"))

    # Only the per-suite tables and the summaries are written; their
    # concatenation lives in memory for the cross-suite figure and would just be
    # a megabyte of duplicated rows on disk.
    full = pd.concat(all_df, ignore_index=True)
    summary = summarise(full)
    summary.to_csv(os.path.join(OUT_DIR, "summary_medians.csv"), index=False)
    fig_prior_dependence(full, os.path.join(OUT_DIR, "prior_dependence.png"))

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n=== median metrics by suite and method ===")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote CSVs + figures to {OUT_DIR}/  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
