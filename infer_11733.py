"""
Run inference on sample 11733 using all trained MeV classifier models.

The CSV has signal values across multiple sweeps that together form the
200-channel detector response vector (4 × 48 + 1 × 8 = 200).
"""

import csv
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CSV_PATH    = "11733.csv"
JSON_PATH   = "training_results.json"
OUTPUT_CSV  = "inference_11733.csv"
N_VALUES    = [10, 20, 50, 100, 200]


def load_sample(csv_path: str) -> np.ndarray:
    """Read CSV and return signal values as a (200,) float32 array."""
    df = pd.read_csv(csv_path)
    signal = df["signal"].values.astype(np.float32)
    assert len(signal) == 200, f"Expected 200 signal values, got {len(signal)}"
    return signal


def mev_bin_edges(n: int) -> np.ndarray:
    return np.linspace(0.0, 50.0, n + 1)


def predict_all(signal: np.ndarray, all_results: dict) -> list[dict]:
    rows = []
    for n in N_VALUES:
        key = str(n)
        if key not in all_results:
            continue

        model_path = f"model_mev_n{n}.keras"
        if not os.path.exists(model_path):
            continue

        res  = all_results[key]
        mean = np.array(res["norm_mean"], dtype=np.float32)
        std  = np.array(res["norm_std"],  dtype=np.float32)

        x_norm = ((signal - mean) / std).reshape(1, -1)

        model = tf.keras.models.load_model(model_path, compile=False)
        probs = model.predict(x_norm, verbose=0)[0]

        edges = mev_bin_edges(n)
        top3  = np.argsort(probs)[::-1][:3]

        for rank, b in enumerate(top3, 1):
            rows.append({
                "sample":         "11733",
                "n_bins":         n,
                "mev_per_bin":    round(50.0 / n, 4),
                "rank":           rank,
                "pred_bin":       int(b),
                "energy_lo_mev":  round(float(edges[b]),     4),
                "energy_hi_mev":  round(float(edges[b + 1]), 4),
                "confidence":     round(float(probs[b]),     6),
            })

    return rows


def save_csv(rows: list[dict], path: str) -> None:
    fields = ["sample", "n_bins", "mev_per_bin", "rank",
              "pred_bin", "energy_lo_mev", "energy_hi_mev", "confidence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    signal = load_sample(CSV_PATH)
    print(f"Signal: min={signal.min():.3f}  max={signal.max():.3f}  mean={signal.mean():.3f}")

    with open(JSON_PATH) as f:
        all_results = json.load(f)

    rows = predict_all(signal, all_results)
    save_csv(rows, OUTPUT_CSV)
