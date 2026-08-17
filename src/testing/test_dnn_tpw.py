"""
test_dnn_tpw.py — Stage 2/2 of the former root-level `run_tpw_mev.py`.

Loads the prepped-spectra artifact produced by `src/testing/prep_tpw_data.py`
(out/testing/tpw_prepped_spectra.npz) and runs the trained DNN MEV energy-bin
classifiers (out/training/dnn/model_dnn_mev_n{n}.keras) against every shot in
it, purely in memory — no zip/TIF I/O and no touching of TPW/ happens here.

Outputs (unchanged in shape/columns from the original single-script version,
just relocated under out/testing/):
    out/testing/tpw_mev_results.csv      — top-3 energy predictions per shot per model
    out/testing/tpw_model_accuracy.csv   — per-model confidence summary
    out/testing/tpw_spectra_n{n}.csv     — full softmax spectrum per shot, one file per n

"TPW" is not expanded anywhere in the original script or its comments — it
is referred to here only as the TPW real-shot dataset.

Run `prep_tpw_data.py` first to produce the input artifact this script
consumes.
"""

import csv
import json
import os

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PREPPED_NPZ = "out/testing/tpw_prepped_spectra.npz"
JSON_PATH   = "out/training/dnn/dnn_mev_training_results.json"
OUT_CSV     = "out/testing/tpw_mev_results.csv"
ACC_CSV     = "out/testing/tpw_model_accuracy.csv"
N_VALUES    = [50, 200]


# ── inference ──────────────────────────────────────────────────────────────

def mev_bin_edges(n: int) -> np.ndarray:
    return np.linspace(0.0, 50.0, n + 1)


def load_models(all_results: dict) -> dict[int, tuple[tf.keras.Model, np.ndarray, np.ndarray]]:
    """Load all available MEV models; return {n -> (model, mean, std)}."""
    loaded = {}
    for n in N_VALUES:
        key = str(n)
        if key not in all_results:
            continue
        mp = os.path.join("out/training/dnn", f"model_dnn_mev_n{n}.keras")
        if not os.path.exists(mp):
            continue
        print(f"  Loading model n={n} …")
        model = tf.keras.models.load_model(mp, compile=False)
        mean  = np.array(all_results[key]["norm_mean"], dtype=np.float32)
        std   = np.array(all_results[key]["norm_std"],  dtype=np.float32)
        loaded[n] = (model, mean, std)
    return loaded


def predict(
    signal: np.ndarray,
    models: dict[int, tuple[tf.keras.Model, np.ndarray, np.ndarray]],
    top_k: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
        top_rows    — top-k rows (shot, n_bins, rank, energy range, confidence)
        spectrum_rows — one row per model with full softmax probability vector
    """
    top_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    for n, (model, mean, std) in sorted(models.items()):
        x = ((signal - mean) / std).reshape(1, -1)
        probs = model.predict(x, verbose=0)[0]
        edges = mev_bin_edges(n)
        for rank, b in enumerate(np.argsort(probs)[::-1][:top_k], 1):
            top_rows.append({
                "n_bins":        n,
                "mev_per_bin":   round(50.0 / n, 4),
                "rank":          rank,
                "pred_bin":      int(b),
                "energy_lo_mev": round(float(edges[b]),     4),
                "energy_hi_mev": round(float(edges[b + 1]), 4),
                "confidence":    round(float(probs[b]),     6),
            })
        # Full spectrum: one row, bin_0..bin_{n-1} as separate columns
        spec = {"n_bins": n, "mev_per_bin": round(50.0 / n, 4)}
        for i, p in enumerate(probs):
            spec[f"bin_{i:03d}_{edges[i]:.2f}_{edges[i+1]:.2f}mev"] = round(float(p), 8)
        spectrum_rows.append(spec)
    return top_rows, spectrum_rows


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== TPW MEV Inference (Stage 2: model inference) ===\n")

    if not os.path.exists(PREPPED_NPZ):
        raise FileNotFoundError(
            f"{PREPPED_NPZ} not found. Run `python src/testing/prep_tpw_data.py` "
            "first to extract the TPW zips and prep the 200-channel spectra."
        )

    print("Step 1 — Loading prepped spectra …")
    npz = np.load(PREPPED_NPZ)
    shot_names  = npz["shot_names"]
    source_zips = npz["source_zips"]
    spectra     = npz["spectra"]
    print(f"  {len(shot_names)} prepped shots loaded\n")

    print("Step 2 — Loading trained models …")
    with open(JSON_PATH) as f:
        all_results = json.load(f)
    models = load_models(all_results)
    print(f"  {len(models)} models loaded: n = {sorted(models)}\n")

    fields = [
        "shot", "source_zip",
        "n_bins", "mev_per_bin", "rank",
        "pred_bin", "energy_lo_mev", "energy_hi_mev", "confidence",
    ]
    output_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    errors: list[str] = []

    print("Step 3 — Running inference …")
    for name, zname, signal in zip(shot_names, source_zips, spectra):
        try:
            top_rows, spec_rows = predict(signal, models)

            for r in top_rows:
                r["shot"]       = name
                r["source_zip"] = zname
                output_rows.append(r)

            for r in spec_rows:
                r["shot"]       = name
                r["source_zip"] = zname
                spectrum_rows.append(r)

            top_conf = max(r["confidence"] for r in top_rows if r["rank"] == 1)
            top_n    = next(r["n_bins"]    for r in top_rows if r["rank"] == 1 and r["confidence"] == top_conf)
            top_lo   = next(r["energy_lo_mev"] for r in top_rows if r["rank"] == 1 and r["n_bins"] == top_n)
            top_hi   = next(r["energy_hi_mev"] for r in top_rows if r["rank"] == 1 and r["n_bins"] == top_n)
            print(f"  {name:40s}  best n={top_n:3d}  "
                  f"{top_lo:.1f}–{top_hi:.1f} MeV  conf={top_conf:.3f}")

        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"  ERROR {name}: {exc}")

    os.makedirs("out/testing", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output_rows)

    # ── full spectrum CSVs (one file per model, one row per shot) ────────────
    for n in N_VALUES:
        rows_n = [r for r in spectrum_rows if r["n_bins"] == n]
        if not rows_n:
            continue
        spec_fields = ["shot", "source_zip", "n_bins", "mev_per_bin"] + [
            k for k in rows_n[0] if k.startswith("bin_")
        ]
        spec_path = os.path.join("out/testing", f"tpw_spectra_n{n}.csv")
        with open(spec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=spec_fields)
            w.writeheader()
            w.writerows(rows_n)
        print(f"  Spectra n={n:<3}: {spec_path}  ({len(rows_n)} shots × {n} bins)")

    # ── accuracy summary ───────────────────────────────────────────────────
    rank1 = [r for r in output_rows if r["rank"] == 1]
    acc_fields = [
        "n_bins", "mev_per_bin", "shots",
        "mean_confidence", "median_confidence", "std_confidence",
        "pct_above_99", "pct_above_90", "min_confidence",
    ]
    acc_rows = []
    print("\n  Per-model confidence (rank-1, SAS shots only):")
    print(f"  {'n_bins':>6}  {'MeV/bin':>8}  {'shots':>6}  "
          f"{'mean':>7}  {'median':>7}  {'>99%':>6}  {'min':>6}")
    for n in N_VALUES:
        confs = np.array([r["confidence"] for r in rank1 if r["n_bins"] == n])
        if len(confs) == 0:
            continue
        row = {
            "n_bins":            n,
            "mev_per_bin":       round(50.0 / n, 4),
            "shots":             len(confs),
            "mean_confidence":   round(float(confs.mean()),              6),
            "median_confidence": round(float(np.median(confs)),          6),
            "std_confidence":    round(float(confs.std()),               6),
            "pct_above_99":      round(float((confs >= 0.99).mean()) * 100, 2),
            "pct_above_90":      round(float((confs >= 0.90).mean()) * 100, 2),
            "min_confidence":    round(float(confs.min()),               6),
        }
        acc_rows.append(row)
        print(f"  {n:>6}  {50/n:>8.2f}  {len(confs):>6}  "
              f"{row['mean_confidence']:>7.4f}  {row['median_confidence']:>7.4f}  "
              f"{row['pct_above_99']:>5.1f}%  {row['min_confidence']:>6.4f}")

    with open(ACC_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=acc_fields)
        w.writeheader()
        w.writerows(acc_rows)

    print(f"\n=== Done ===")
    print(f"  Results : {OUT_CSV}  ({len(output_rows)} rows)")
    print(f"  Accuracy: {ACC_CSV}")
    print(f"  Shots   : {len(shot_names)} SAS shots × {len(models)} models × 3 ranks")
    print(f"  Spectra : out/testing/tpw_spectra_n{{50,200}}.csv")
    if errors:
        print(f"  Errors  : {len(errors)}")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()
