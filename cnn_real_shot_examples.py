"""
Same layout as cnn_val_examples.py but for real shots instead of synthetic
validation data. Real shots have no ground-truth spectrum, so the right panel
compares the CNN's predicted spectrum against the NNLS floor fit's spectrum
(the best-fit shape achievable by any spectrum for that shot's measured
vector) instead of a true spectrum. Rows are labeled by shot number and
sorted by excess-over-floor (CNN resid% - NNLS floor resid%), most reasonable
fits first.

Usage: CNN_NBINS=50 python cnn_real_shot_examples.py [csv_path ...]
Positional args, if given, are explicit *_proc_vector[_ch|_cv].csv paths (so
you can pick the correction variant per shot); otherwise all shots under
res/test_images/ with a *_proc_vector.csv (uncorrected) are used.
Env: CNN_OUT_TAG suffix inserted into the output filename, e.g. "_ch".
"""
import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import nnls

from data_utils import bin_drm, load_drm, mev_bin_centers, mev_bin_edges, normalize_apply
from train_cnn import reshape_to_2d

DRM_PATH = "res/drm/200x200.xlsx"
N_BINS = int(os.environ.get("CNN_NBINS", 50))
MODEL_PREFIX = os.environ.get("CNN_MODEL_PREFIX", "model_cnn")
CNN_JSON = os.environ.get("CNN_RESULTS_JSON", "cnn_training_results.json")
OUT_TAG = os.environ.get("CNN_OUT_TAG", "")
TEST_ROOT = "res/test_images"


def l1(x):
    t = x.sum()
    return (x / t).astype(np.float32) if t > 0 else x


def main():
    drm = load_drm(DRM_PATH)
    drm_b = bin_drm(drm, N_BINS)
    energy_bins = mev_bin_centers(N_BINS)
    edges = mev_bin_edges(N_BINS)
    channels = np.arange(1, 201)

    with open(CNN_JSON) as f:
        res = json.load(f)[str(N_BINS)]
    mean = np.array(res["norm_mean"], dtype=np.float32)
    std = np.array(res["norm_std"], dtype=np.float32)
    model = tf.keras.models.load_model(f"{MODEL_PREFIX}_n{N_BINS}.keras", compile=False)

    csv_paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(f"{TEST_ROOT}/*/*_proc_vector.csv"))

    rows = []
    for p in csv_paths:
        shot = os.path.basename(p).split("_")[0]
        sig = pd.read_csv(p)["signal"].values.astype(float)
        if len(sig) != 200:
            continue
        sig_l1 = l1(sig)

        spec_nnls, _ = nnls(drm_b, sig_l1)
        resp_nnls = l1(drm_b @ spec_nnls)
        floor = 100 * np.sqrt(np.mean((sig_l1 - resp_nnls) ** 2)) / np.sqrt(np.mean(sig_l1 ** 2))
        spec_nnls_norm = l1(spec_nnls)

        xn = normalize_apply(sig_l1.reshape(1, -1).astype(np.float32), mean, std)
        spec_cnn = model.predict(reshape_to_2d(xn), verbose=0)[0]
        resp_cnn = l1(drm_b @ spec_cnn)
        resid = 100 * np.sqrt(np.mean((sig_l1 - resp_cnn) ** 2)) / np.sqrt(np.mean(sig_l1 ** 2))

        rows.append(dict(shot=shot, sig=sig_l1, resp_cnn=resp_cnn, spec_cnn=spec_cnn,
                          spec_nnls=spec_nnls_norm, resid=resid, floor=floor, excess=resid - floor))

    rows.sort(key=lambda r: r["excess"])

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), tight_layout=True)
    fig.suptitle(f"CNN real-shot examples (n_bins={N_BINS}) — sorted by excess over NNLS floor, "
                 f"most reasonable first", fontsize=12)

    for row, r in enumerate(rows):
        ax = axes[row, 0]
        ax.plot(channels, r["sig"], "k", lw=1.2, label="real response (L1)")
        ax.plot(channels, r["resp_cnn"], "r--", lw=1.1, label=f"DRM x CNN pred ({r['resid']:.1f}%)")
        ax.set_ylabel(f"{r['shot']}\nexcess {r['excess']:.1f}")
        ax.legend(fontsize=7, loc="upper right")
        if row == 0:
            ax.set_title("Detector response: real vs reconstructed")
        if row == n - 1:
            ax.set_xlabel("Detector channel")

        ax = axes[row, 1]
        ax.stairs(r["spec_nnls"], edges, color="steelblue", lw=1.4, label=f"NNLS floor ({r['floor']:.1f}%)")
        ax.stairs(r["spec_cnn"], edges, color="firebrick", lw=1.6, label="CNN predicted")
        ax.set_xlim(0, 50)
        ax.legend(fontsize=7, loc="upper right")
        if row == 0:
            ax.set_title("Energy spectrum: NNLS floor fit vs CNN predicted")
        if row == n - 1:
            ax.set_xlabel("Energy (MeV)")

    out_path = f"cnn_real_shot_examples_n{N_BINS}{OUT_TAG}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}  ({n} shots)")


if __name__ == "__main__":
    main()
