"""
All matplotlib figures for the MeV classification pipeline.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from data_utils import bin_drm, generate_synthetic_data, load_drm, mev_bin_centers

XLSX_PATH = "res/200x200.xlsx"
N_VALUES  = [10, 20, 50, 100, 200]
COLORS    = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]
FIGDIR    = "figures"

os.makedirs(FIGDIR, exist_ok=True)

def _save(name: str) -> None:
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.show()
    plt.close()


# DRM overview
def plot_drm_overview(drm: np.ndarray) -> None:
    """Heatmap of the full 200×200 DRM plus column-summed integrated response."""
    mev_edges = np.linspace(0, 50, 201)
    mev_full  = (mev_edges[:-1] + mev_edges[1:]) / 2   # bin centers

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Detector Response Matrix (DRM) — 200×200", fontsize=13)

    im = axes[0].imshow(
        drm, aspect="auto", cmap="hot", origin="lower",
        extent=[0, 50, 0, 200],
    )
    axes[0].set_xlabel("Energy (MeV)")
    axes[0].set_ylabel("Detector Channel")
    axes[0].set_title("Full DRM (0.25 MeV/bin)")
    plt.colorbar(im, ax=axes[0], label="Response")

    col_sum = drm.sum(axis=0)
    axes[1].plot(mev_full, col_sum, color="steelblue", lw=1.5)
    axes[1].set_xlabel("Energy (MeV)")
    axes[1].set_ylabel("Summed Detector Response")
    axes[1].set_title("Integrated Response per Energy Bin")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save("drm_overview.png")


# Binned DRM
def plot_binned_drm(drm: np.ndarray) -> None:
    """2×3 grid: binned DRM heatmaps for each n value."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Binned DRM for Each n", fontsize=13)

    flat = axes.flatten()
    for ax, n, color in zip(flat, N_VALUES, COLORS):
        binned = bin_drm(drm, n)
        im = ax.imshow(
            binned, aspect="auto", cmap="hot", origin="lower",
            extent=[0, 50, 0, 200],
        )
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Detector Channel")
        ax.set_title(f"n={n}  ({50/n:.1f} MeV/bin,  200×{n})")
        plt.colorbar(im, ax=ax, label="Response")

    flat[-1].set_visible(False)   # hide unused 6th slot
    plt.tight_layout()
    _save("binned_drm.png")


# Noise examples

def plot_noise_examples(drm: np.ndarray, n: int = 10, n_rows: int = 5) -> None:
    """Show clean ± σ band vs a single noisy realisation for n_rows energy bins."""
    rng    = np.random.default_rng(0)
    binned = bin_drm(drm, n)
    ch     = np.arange(200)
    idxs   = np.linspace(0, n - 1, n_rows, dtype=int)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.8 * n_rows))
    fig.suptitle(f"Clean vs Poisson-Noisy Detector Response  (n={n})", fontsize=13)

    for row, i in enumerate(idxs):
        col   = binned[:, i]
        sigma = np.sqrt(np.maximum(col, 1e-8))
        noisy = np.clip(col + rng.standard_normal(200) * sigma, 0, None)
        lo, hi = col - sigma, col + sigma
        mev_lo = i * 50 / n
        mev_hi = (i + 1) * 50 / n

        ax_c = axes[row, 0]
        ax_c.plot(ch, col, color="steelblue", lw=1.5, label="Clean")
        ax_c.fill_between(ch, lo, hi, alpha=0.25, color="steelblue", label="±√I band")
        ax_c.set_title(f"Bin {i}: {mev_lo:.1f}–{mev_hi:.1f} MeV — clean ± σ")
        ax_c.set_ylabel("Intensity")
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.2)

        ax_n = axes[row, 1]
        ax_n.plot(ch, noisy,  color="crimson",   lw=1,   label="Noisy sample")
        ax_n.plot(ch, col,    color="steelblue", lw=1.5, alpha=0.5, label="Clean")
        ax_n.set_title(f"One noisy realisation  (σ = √I)")
        ax_n.set_ylabel("Intensity")
        ax_n.legend(fontsize=8)
        ax_n.grid(True, alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Detector Channel")

    plt.tight_layout()
    _save("noise_examples.png")


# Training curves

def plot_training_curves(results: dict) -> None:
    """4-panel: val loss (log), val accuracy, macro F1, efficiency vs epoch."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves for All n Values", fontsize=13)

    panels = [
        ("val_loss",     "Validation Loss (log scale)", True,  axes[0, 0]),
        ("val_accuracy", "Validation Accuracy",         False, axes[0, 1]),
        ("f1",           "Macro F1 Score",              False, axes[1, 0]),
        ("efficiency",   "Efficiency (Macro Recall)",   False, axes[1, 1]),
    ]

    for key, title, log_y, ax in panels:
        for color, (n_str, res) in zip(COLORS, results.items()):
            epochs = range(1, res["epochs_trained"] + 1)
            ax.plot(epochs, res[key], color=color, lw=1.5, label=f"n={n_str}")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save("training_curves.png")


# Final metrics bar chart

def plot_final_metrics_bar(results: dict) -> None:
    """Grouped bar: accuracy / F1 / precision / recall, plus epochs trained."""
    ns     = [int(k) for k in results]
    labels = [f"n={n}" for n in ns]
    x      = np.arange(len(ns))
    w      = 0.2

    metrics = {
        "val_accuracy": "Accuracy",
        "f1":           "Macro F1",
        "precision":    "Precision",
        "recall":       "Recall / Eff",
    }
    bar_colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Final Validation Metrics by n", fontsize=13)

    offsets = np.linspace(-(len(metrics) - 1) * w / 2, (len(metrics) - 1) * w / 2, len(metrics))
    for (key, label), offset, bc in zip(metrics.items(), offsets, bar_colors):
        vals = [res[key][-1] for res in results.values()]
        bars = ax1.bar(x + offset, vals, width=w, label=label, color=bc, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy / F1 / Precision / Recall")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    epochs_trained = [res["epochs_trained"] for res in results.values()]
    bars = ax2.bar(x, epochs_trained, color=COLORS, alpha=0.85)
    for bar, ep in zip(bars, epochs_trained):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, ep + 1,
            str(ep), ha="center", va="bottom", fontsize=10
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Epochs")
    ax2.set_title("Epochs Until Early Stop")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save("final_metrics_bar.png")


# Confusion matrices

def plot_confusion_matrices() -> None:
    """2×3 grid of row-normalized confusion matrices for each n."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("Normalized Confusion Matrices (row = true bin)", fontsize=13)

    flat = axes.flatten()
    for ax, n in zip(flat, N_VALUES):
        path = f"results_n{n}_confusion.npy"
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing: {path}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"n={n}")
            continue

        cm      = np.load(path).astype(float)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n_ticks  = min(n, 10)
        tick_idx = np.linspace(0, n - 1, n_ticks, dtype=int)
        centers  = mev_bin_centers(n)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{centers[i]:.0f}" for i in tick_idx], rotation=45, fontsize=8)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{centers[i]:.0f}" for i in tick_idx], fontsize=8)
        ax.set_xlabel("Predicted MeV (bin centre)")
        ax.set_ylabel("True MeV (bin centre)")
        ax.set_title(f"n={n}  |  {50/n:.2f} MeV/bin  |  "
                     f"overall acc={cm.diagonal().sum()/cm.sum():.3f}")

    flat[-1].set_visible(False)   # hide unused 6th slot
    plt.tight_layout()
    _save("confusion_matrices.png")


# Per-bin efficiency

def plot_per_bin_efficiency() -> None:
    """Per-energy-bin recall (diagonal of confusion matrix) for each n."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Per-Bin Detection Efficiency (Recall) vs Energy", fontsize=13)

    flat = axes.flatten()
    flat[-1].set_visible(False)   # hide unused 6th slot
    for ax, n, color in zip(flat, N_VALUES, COLORS):
        path = f"results_n{n}_confusion.npy"
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing: {path}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"n={n}")
            continue

        cm      = np.load(path).astype(float)
        row_sum = cm.sum(axis=1)
        eff     = np.where(row_sum > 0, cm.diagonal() / row_sum, 0.0)
        centers = mev_bin_centers(n)

        ax.bar(centers, eff, width=50 / n * 0.85, color=color, alpha=0.75)
        ax.axhline(eff.mean(), color="k", linestyle="--", lw=1.2,
                   label=f"Mean = {eff.mean():.3f}")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Efficiency (recall)")
        ax.set_title(f"n={n}  |  {50/n:.1f} MeV/bin")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.tight_layout()
    _save("per_bin_efficiency.png")


# Figure 8: Noise sigma vs MeV (summary) ───────────────────────────────────

def plot_noise_profile(drm: np.ndarray) -> None:
    """Mean √I noise level per detector channel for each n binning."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Mean Poisson Noise Level (√I) per Energy Bin", fontsize=13)

    for n, color in zip(N_VALUES, COLORS):
        binned  = bin_drm(drm, n)
        sigma   = np.sqrt(np.maximum(binned, 1e-8))  # (200, n)
        mean_sigma = sigma.mean(axis=0)               # avg over detector channels
        centers = mev_bin_centers(n)
        ax.plot(centers, mean_sigma, marker="o", ms=3, lw=1.5,
                color=color, label=f"n={n}")

    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Mean σ = ⟨√I⟩  over detector channels")
    ax.set_title("Noise Level Grows with Signal Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("noise_profile.png")

if __name__ == "__main__":
    pre_only = "--pre" in sys.argv

    print("Loading DRM…")
    drm = load_drm(XLSX_PATH)

    print("\n[1/8] DRM overview")
    plot_drm_overview(drm)

    print("[2/8] Binned DRM")
    plot_binned_drm(drm)

    print("[3/8] Noise examples")
    plot_noise_examples(drm, n=10, n_rows=5)

    print("[4/8] Noise profile")
    plot_noise_profile(drm)

    if pre_only:
        print("\n--pre flag set: skipping post-training figures.")
        sys.exit(0)

    json_path = "training_results.json"
    if not os.path.exists(json_path):
        print(f"\n{json_path} not found — run train_mev.py first.")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    print("[5/8] Training curves")
    plot_training_curves(results)

    print("[6/8] Final metrics bar chart")
    plot_final_metrics_bar(results)

    print("[7/8] Confusion matrices")
    plot_confusion_matrices()

    print("[8/8] Per-bin efficiency")
    plot_per_bin_efficiency()

    print(f"\nAll figures saved to {FIGDIR}/")
