"""
Run the trained hybrid CNN (4x48 spatial branch + 8 coarse-pixel branch)
on real SAS shot vectors and optionally compare against the dense
model_spectrum_n{n} baseline.

Hybrid CNN input
----------------
The 200 effective detector measurements are split into:

    channels 0-191:
        -> reshape to (4, 48, 1)
        -> CNN branch

    channels 192-199:
        -> remain as 8 independent values
        -> dense branch

Usage
-----
    python infer_cnn.py [csv_path ...]

Examples
--------
    python infer_cnn.py

    CNN_NBINS=20 python infer_cnn.py

    CNN_NBINS=10 python infer_cnn.py \
        res/test_images/10084/10084_proc_vector_corrected.csv

Environment variables
---------------------
    CNN_NBINS
        Number of spectrum bins. Default: 10

    CNN_MODEL_PREFIX
        Model filename prefix.
        Default: model_hybrid_cnn

        Example:
            model_hybrid_cnn_n10.keras

    CNN_RESULTS_JSON
        JSON file containing normalization statistics.
        Default: hybrid_cnn_training_results.json

    CNN_OUT_TAG
        Optional suffix for output filenames.

Outputs
-------
    hybrid_cnn_infer_<shot>_n<n_bins>.png

    hybrid_cnn_infer_<shot>_n<n_bins>_unsat.png
        Generated when a saturation mask is available.
"""

import json
import os
import sys

# Set before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import (
    bin_drm,
    load_drm,
    load_saturation_mask,
    mev_bin_centers,
    mev_bin_edges,
    normalize_apply,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRM_PATH = "res/drm/200x200.xlsx"

CNN_JSON = os.environ.get(
    "CNN_RESULTS_JSON",
    "hybrid_cnn_training_results.json",
)

DENSE_JSON = "spectrum_training_results.json"

N_BINS = int(
    os.environ.get(
        "CNN_NBINS",
        10,
    )
)

MODEL_PREFIX = os.environ.get(
    "CNN_MODEL_PREFIX",
    "model_hybrid_cnn",
)

OUT_TAG = os.environ.get(
    "CNN_OUT_TAG",
    "",
)


DEFAULT_SHOTS = [
    "res/test_images/10084/10084_proc_vector_corrected.csv",
    "res/test_images/11733/11733_proc_vector_corrected.csv",
]


# ---------------------------------------------------------------------------
# Load detector signal
# ---------------------------------------------------------------------------

def load_signal(csv_path: str) -> np.ndarray:
    """
    Load one real SAS 200-effective-pixel detector vector.
    """

    df = pd.read_csv(csv_path)

    signal = (
        df[df.columns[-1]]
        .values
        .astype(np.float32)
    )

    if len(signal) != 200:
        raise ValueError(
            f"Expected 200 detector channels, got {len(signal)}"
        )

    return signal


# ---------------------------------------------------------------------------
# L1 normalization
# ---------------------------------------------------------------------------

def l1_normalise(
    x: np.ndarray,
) -> np.ndarray:
    """
    Normalize a vector so its elements sum to 1.
    """

    total = np.sum(x)

    if total <= 0:
        return x.astype(np.float32)

    return (
        x / total
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Prepare hybrid CNN input
# ---------------------------------------------------------------------------

def split_detector_input(
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convert a batch of 200-element detector vectors into the two inputs
    expected by the hybrid CNN.

    Input
    -----
    X:
        shape (batch_size, 200)

    Output
    ------
    {
        "fine_image":
            shape (batch_size, 4, 48, 1),

        "coarse_pixels":
            shape (batch_size, 8)
    }
    """

    if X.ndim != 2 or X.shape[1] != 200:
        raise ValueError(
            f"Expected shape (batch, 200), got {X.shape}"
        )

    batch_size = X.shape[0]

    fine_image = (
        X[:, :192]
        .reshape(
            batch_size,
            4,
            48,
            1,
        )
        .astype(np.float32)
    )

    coarse_pixels = (
        X[:, 192:200]
        .astype(np.float32)
    )

    return {
        "fine_image": fine_image,
        "coarse_pixels": coarse_pixels,
    }


# ---------------------------------------------------------------------------
# Hybrid CNN prediction
# ---------------------------------------------------------------------------

def predict_hybrid(
    model: tf.keras.Model,
    x_norm: np.ndarray,
) -> np.ndarray:
    """
    Run the hybrid CNN on one or more normalized 200-element detector vectors.
    """

    model_inputs = split_detector_input(
        x_norm
    )

    prediction = model.predict(
        model_inputs,
        verbose=0,
    )

    return prediction[0]


# ---------------------------------------------------------------------------
# Dense baseline prediction
# ---------------------------------------------------------------------------

def predict_dense(
    model: tf.keras.Model,
    x_norm: np.ndarray,
) -> np.ndarray:
    """
    Run the original dense spectrum-regression model.
    """

    prediction = model.predict(
        x_norm,
        verbose=0,
    )

    return prediction[0]


# ---------------------------------------------------------------------------
# Run inference for one shot
# ---------------------------------------------------------------------------

def run_shot(
    csv_path: str,
    drm: np.ndarray,
    cnn_model: tf.keras.Model,
    cnn_res: dict,
    dense_model: tf.keras.Model | None = None,
    dense_res: dict | None = None,
) -> None:

    # ---------------------------------------------------------------
    # Shot information
    # ---------------------------------------------------------------

    shot_name = (
        os.path.basename(csv_path)
        .split("_")[0]
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Shot {shot_name} | n_bins={N_BINS}"
    )

    print(
        f"{'=' * 70}"
    )


    # ---------------------------------------------------------------
    # Load real 200-element detector signal
    # ---------------------------------------------------------------

    signal = load_signal(
        csv_path
    )

    # Match the training convention.
    signal_l1 = l1_normalise(
        signal
    )


    # ---------------------------------------------------------------
    # Prepare DRM
    # ---------------------------------------------------------------

    drm_binned = bin_drm(
        drm,
        N_BINS,
    )

    energy_bins = mev_bin_centers(
        N_BINS
    )

    channels = np.arange(
        1,
        201,
    )


    # ---------------------------------------------------------------
    # Saturation mask
    # ---------------------------------------------------------------

    sat_mask = load_saturation_mask(
        csv_path
    )

    if sat_mask is not None:

        unsat = ~sat_mask

    else:

        unsat = np.ones(
            200,
            dtype=bool,
        )


    # ---------------------------------------------------------------
    # Run hybrid CNN
    # ---------------------------------------------------------------

    cnn_mean = np.asarray(
        cnn_res["norm_mean"],
        dtype=np.float32,
    )

    cnn_std = np.asarray(
        cnn_res["norm_std"],
        dtype=np.float32,
    )

    # Real detector vector:
    #
    #       (200,)
    #
    #           ↓
    #
    #       (1, 200)
    #
    #           ↓ z-score
    #
    #       (1, 200)
    #
    #       ↙           ↘
    #
    # (1,4,48,1)       (1,8)
    #
    #       ↘           ↙
    #
    #         Hybrid CNN
    #
    #             ↓
    #
    #         spectrum
    #         (N_BINS,)

    cnn_x_norm = normalize_apply(
        signal_l1.reshape(1, -1),
        cnn_mean,
        cnn_std,
    )

    cnn_spec = predict_hybrid(
        cnn_model,
        cnn_x_norm,
    )


    # ---------------------------------------------------------------
    # Forward-fold predicted spectrum through DRM
    # ---------------------------------------------------------------

    cnn_response = (
        drm_binned
        @ cnn_spec
    )

    cnn_response_l1 = l1_normalise(
        cnn_response
    )

    cnn_residual = (
        signal_l1
        - cnn_response_l1
    )

    cnn_rms = float(
        np.sqrt(
            np.mean(
                cnn_residual ** 2
            )
        )
    )

    cnn_rms_unsat = float(
        np.sqrt(
            np.mean(
                cnn_residual[unsat] ** 2
            )
        )
    )


    # ---------------------------------------------------------------
    # Store predictions
    # ---------------------------------------------------------------

    preds = {

        "Hybrid CNN": {

            "spec": cnn_spec,

            "resp": cnn_response_l1,

            "resid": cnn_residual,

            "rms": cnn_rms,

            "rms_unsat": cnn_rms_unsat,

        }

    }


    # ---------------------------------------------------------------
    # Optional dense baseline
    # ---------------------------------------------------------------
    dense_model = None
    dense_res = None

    print("No dense baseline model found. Running hybrid CNN only.")

    # ---------------------------------------------------------------
    # Signal RMS
    # ---------------------------------------------------------------

    signal_rms = float(
        np.sqrt(
            np.mean(
                signal_l1 ** 2
            )
        )
    )

    signal_rms_unsat = float(
        np.sqrt(
            np.mean(
                signal_l1[unsat] ** 2
            )
        )
    )


    # ---------------------------------------------------------------
    # Print numerical results
    # ---------------------------------------------------------------

    if sat_mask is not None:

        print(
            f"  {sat_mask.sum()}/200 channels are "
            f"saturation-corrected"
        )


    for label, prediction in preds.items():

        peak_index = int(
            np.argmax(
                prediction["spec"]
            )
        )

        peak_energy = (
            energy_bins[peak_index]
        )

        peak_fraction = (
            prediction["spec"][peak_index]
        )

        print(
            f"\n  {label}"
        )

        print(
            f"    Peak energy       : "
            f"{peak_energy:.2f} MeV"
        )

        print(
            f"    Peak fraction     : "
            f"{100 * peak_fraction:.2f}%"
        )

        print(
            f"    Residual RMS      : "
            f"{prediction['rms']:.6f}"
        )

        print(
            f"    Relative RMS      : "
            f"{100 * prediction['rms'] / signal_rms:.1f}% "
            f"of signal RMS"
        )

        print(
            f"    Unsaturated RMS   : "
            f"{prediction['rms_unsat']:.6f}"
        )


    # ---------------------------------------------------------------
    # Main figure
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        tight_layout=True,
    )

    fig.suptitle(
        f"Hybrid CNN spectrum inference — "
        f"shot {shot_name} "
        f"(n_bins={N_BINS})",
        fontsize=12,
    )


    # ===============================================================
    # Panel 1: detector response
    # ===============================================================

    ax = axes[0]

    ax.plot(
        channels,
        signal_l1,
        "k",
        linewidth=1.4,
        label="Real signal (L1)",
    )

    ax.plot(
        channels,
        preds["Hybrid CNN"]["resp"],
        "r--",
        linewidth=1.2,
        label=(
            "DRM × Hybrid CNN prediction "
            f"({100 * preds['Hybrid CNN']['rms'] / signal_rms:.1f}%)"
        ),
    )

    if "Dense" in preds:

        ax.plot(
            channels,
            preds["Dense"]["resp"],
            "b:",
            linewidth=1.2,
            label=(
                "DRM × Dense prediction "
                f"({100 * preds['Dense']['rms'] / signal_rms:.1f}%)"
            ),
        )

    ax.set_xlabel(
        "Effective detector pixel"
    )

    ax.set_ylabel(
        "L1-normalized response"
    )

    ax.legend()

    ax.set_title(
        "Detector response: real measurement vs forward-folded prediction"
    )


    # ===============================================================
    # Panel 2: predicted spectrum
    # ===============================================================

    ax = axes[1]

    edges = mev_bin_edges(
        N_BINS
    )

    ax.stairs(
        preds["Hybrid CNN"]["spec"],
        edges,
        linewidth=1.8,
        label="Hybrid CNN",
    )

    ax.plot(
        energy_bins,
        preds["Hybrid CNN"]["spec"],
        "o",
        markersize=4,
    )

    if "Dense" in preds:

        ax.stairs(
            preds["Dense"]["spec"],
            edges,
            linewidth=1.8,
            label="Dense",
        )

        ax.plot(
            energy_bins,
            preds["Dense"]["spec"],
            "s",
            markersize=4,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        0,
        50,
    )

    ax.set_xlabel(
        "Energy (MeV)"
    )

    ax.set_ylabel(
        "Spectrum fraction (log scale)"
    )

    ax.legend()

    ax.set_title(
        "Predicted incident gamma-ray spectrum"
    )


    # ===============================================================
    # Panel 3: residual
    # ===============================================================

    ax = axes[2]


    if (
        sat_mask is not None
        and sat_mask.any()
    ):

        edges_idx = np.flatnonzero(
            np.diff(
                np.r_[
                    0,
                    sat_mask.astype(int),
                    0,
                ]
            )
        )

        spans = edges_idx.reshape(
            -1,
            2,
        )

        for k, (lo, hi) in enumerate(
            spans
        ):

            ax.axvspan(
                channels[lo],
                channels[hi - 1],
                alpha=0.2,
                label=(
                    "Saturation-corrected"
                    if k == 0
                    else None
                ),
            )


    ax.plot(
        channels,
        preds["Hybrid CNN"]["resid"],
        linewidth=1,
        label="Hybrid CNN residual",
    )


    if "Dense" in preds:

        ax.plot(
            channels,
            preds["Dense"]["resid"],
            linewidth=1,
            alpha=0.7,
            label="Dense residual",
        )


    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )


    ax.set_xlabel(
        "Effective detector pixel"
    )

    ax.set_ylabel(
        "Real L1 - predicted L1"
    )

    ax.legend()

    ax.set_title(
        "Detector-response residuals"
    )


    # ---------------------------------------------------------------
    # Save main figure
    # ---------------------------------------------------------------

    out_path = (
        f"hybrid_cnn_infer_"
        f"{shot_name}_"
        f"n{N_BINS}"
        f"{OUT_TAG}.png"
    )

    fig.savefig(
        out_path,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"\n  Saved {out_path}"
    )


    # ---------------------------------------------------------------
    # Unsaturated-only figure
    # ---------------------------------------------------------------

    if (
        sat_mask is not None
        and sat_mask.any()
    ):

        def masked(
            array: np.ndarray,
        ) -> np.ndarray:

            output = (
                array
                .astype(np.float32)
                .copy()
            )

            output[sat_mask] = np.nan

            return output


        fig2, axes2 = plt.subplots(
            2,
            1,
            figsize=(10, 6.5),
            tight_layout=True,
        )


        fig2.suptitle(
            f"Hybrid CNN — shot {shot_name} "
            f"(n_bins={N_BINS}, "
            f"unsaturated channels only)",
            fontsize=12,
        )


        # -----------------------------------------------------------
        # Unsaturated detector response
        # -----------------------------------------------------------

        ax = axes2[0]


        ax.plot(
            channels,
            masked(signal_l1),
            "k",
            linewidth=1.4,
            label="Real signal",
        )


        ax.plot(
            channels,
            masked(
                preds["Hybrid CNN"]["resp"]
            ),
            "r--",
            linewidth=1.2,
            label=(
                "Hybrid CNN "
                f"({100 * preds['Hybrid CNN']['rms_unsat'] / signal_rms_unsat:.1f}%)"
            ),
        )


        if "Dense" in preds:

            ax.plot(
                channels,
                masked(
                    preds["Dense"]["resp"]
                ),
                "b:",
                linewidth=1.2,
                label=(
                    "Dense "
                    f"({100 * preds['Dense']['rms_unsat'] / signal_rms_unsat:.1f}%)"
                ),
            )


        ax.set_xlabel(
            "Effective detector pixel"
        )

        ax.set_ylabel(
            "L1-normalized response"
        )

        ax.legend()

        ax.set_title(
            "Detector response — unsaturated channels only"
        )


        # -----------------------------------------------------------
        # Unsaturated residual
        # -----------------------------------------------------------

        ax = axes2[1]


        ax.plot(
            channels,
            masked(
                preds["Hybrid CNN"]["resid"]
            ),
            linewidth=1,
            label="Hybrid CNN residual",
        )


        if "Dense" in preds:

            ax.plot(
                channels,
                masked(
                    preds["Dense"]["resid"]
                ),
                linewidth=1,
                alpha=0.7,
                label="Dense residual",
            )


        ax.axhline(
            0,
            linewidth=0.8,
            linestyle="--",
        )


        ax.set_xlabel(
            "Effective detector pixel"
        )

        ax.set_ylabel(
            "Real L1 - predicted L1"
        )

        ax.legend()

        ax.set_title(
            "Residuals — unsaturated channels only"
        )


        # -----------------------------------------------------------
        # Save unsaturated figure
        # -----------------------------------------------------------

        out_path_unsat = (
            f"hybrid_cnn_infer_"
            f"{shot_name}_"
            f"n{N_BINS}"
            f"{OUT_TAG}_unsat.png"
        )


        fig2.savefig(
            out_path_unsat,
            dpi=150,
        )

        plt.close(fig2)


        print(
            f"  Saved {out_path_unsat}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Determine shot files
    # ---------------------------------------------------------------

    if len(sys.argv) > 1:

        csv_paths = sys.argv[1:]

    else:

        csv_paths = DEFAULT_SHOTS


    # ---------------------------------------------------------------
    # Load original DRM
    # ---------------------------------------------------------------

    drm = load_drm(
        DRM_PATH
    )


    # ---------------------------------------------------------------
    # Load hybrid CNN results
    # ---------------------------------------------------------------

    with open(
        CNN_JSON
    ) as f:

        all_cnn_results = json.load(
            f
        )


    cnn_res = all_cnn_results[
        str(N_BINS)
    ]


    # ---------------------------------------------------------------
    # Load hybrid CNN model
    # ---------------------------------------------------------------

    cnn_model_path = (
        f"{MODEL_PREFIX}_n{N_BINS}.keras"
    )


    print(
        f"Loading hybrid CNN: "
        f"{cnn_model_path}"
    )


    cnn_model = (
        tf.keras.models.load_model(
            cnn_model_path,
            compile=False,
        )
    )


   
    # ---------------------------------------------------------------
    # Run each shot
    # ---------------------------------------------------------------
    dense_model = None
    dense_res = None

    print("No dense baseline model available. Running hybrid CNN only.")
    for csv_path in csv_paths:

        run_shot(

            csv_path=csv_path,

            drm=drm,

            cnn_model=cnn_model,

            cnn_res=cnn_res,

            dense_model=dense_model,

            dense_res=dense_res,

        )