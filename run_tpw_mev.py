"""
run_tpw_mev.py — Extract TPW zips, process SAS TIF shots, run MEV inference.

Outputs tpw_mev_results.csv with top-3 energy predictions per shot per model.
"""

import csv
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT      = Path(__file__).parent
TPW       = ROOT / "TPW"
OUTDIR    = TPW / "extracted"
JSON_PATH = ROOT / "training_results.json"
OUT_CSV   = ROOT / "tpw_mev_results.csv"
ACC_CSV   = ROOT / "tpw_model_accuracy.csv"
N_VALUES  = [10, 20, 50, 100, 200]

# Only process SAS camera image archives — not imaging-plate (ip) scanners
SAS_ZIPS = {"tpw18sas.zip", "tpw22sas.zip", "TPW_2017.zip"}


# ── file classification ────────────────────────────────────────────────────

_SKIP_NAMES = {"icfsimpic.tif", "sasdesign.tif"}

def _basename(path: str) -> str:
    return path.split("/")[-1]

def is_shot_tif(path: str) -> bool:
    base = _basename(path).lower()
    if not base.endswith((".tif", ".tiff")):
        return False
    if base in _SKIP_NAMES:
        return False
    if "dark" in base or "practice" in base:
        return False
    # must look like a real shot: starts with "shot" or is a shot-number pattern
    return bool(re.match(r"shot\b|shot_|\d{4,}_sas", base))

def is_dark_tif(path: str) -> bool:
    base = _basename(path).lower()
    return base.endswith((".tif", ".tiff")) and "dark" in base

def shot_id(path: str) -> str:
    """Canonical shot identifier: filename stem, lowercased."""
    return Path(_basename(path)).stem.lower()

def dark_path_for(shot_path: str) -> str:
    """Expected dark-frame path given a shot path."""
    for ext in (".tif", ".tiff"):
        if shot_path.lower().endswith(ext):
            return shot_path[: -len(ext)] + "_dark" + ext
    return shot_path + "_dark.tif"


# ── image processing ────────────────────────────────────────────────────────

def load_tif_array(data: bytes) -> np.ndarray:
    """Load TIF bytes to float32 2-D numpy array."""
    with Image.open(io.BytesIO(data)) as img:
        mode = img.mode
        if mode in ("I;16", "I;16B"):
            arr = np.frombuffer(img.tobytes(), dtype=">u2").reshape(
                img.height, img.width
            )
        elif mode == "I;16L":
            arr = np.frombuffer(img.tobytes(), dtype="<u2").reshape(
                img.height, img.width
            )
        elif mode == "I":
            arr = np.array(img, dtype=np.int32)
        else:
            arr = np.array(img)
    return arr.astype(np.float32)


def extract_spectrum(arr: np.ndarray, n_channels: int = 200) -> np.ndarray:
    """
    Collapse a 2-D SAS image to a 1-D n_channels vector.

    Strategy:
      1. Find the row band where signal is brightest (top 30 % of row sums).
      2. Sum those rows horizontally to produce a raw 1-D spectrum.
      3. Linearly interpolate to exactly n_channels.
    """
    h, w = arr.shape

    if h == 1:
        raw = arr[0]
    else:
        row_sums = arr.sum(axis=1)
        thresh = np.percentile(row_sums, 70)
        signal_rows = np.where(row_sums >= thresh)[0]
        if len(signal_rows) == 0:
            signal_rows = np.arange(h)
        raw = arr[signal_rows].sum(axis=0)

    raw = raw.astype(np.float32)

    if len(raw) == n_channels:
        return raw

    x_old = np.linspace(0.0, 1.0, len(raw))
    x_new = np.linspace(0.0, 1.0, n_channels)
    return interp1d(x_old, raw, kind="linear")(x_new).astype(np.float32)


# ── zip extraction ─────────────────────────────────────────────────────────

def extract_zips() -> tuple[dict[str, tuple[str, bytes]], dict[str, bytes]]:
    """
    Extract every zip in TPW/ to TPW/extracted/ (overwriting on conflict).

    Returns:
        shots : {member_path -> (zip_name, raw_bytes)}   — shot TIFs
        darks : {member_path -> raw_bytes}               — dark TIFs
    """
    OUTDIR.mkdir(exist_ok=True)
    shots: dict[str, tuple[str, bytes]] = {}
    darks: dict[str, bytes] = {}
    seen_ids: set[str] = set()

    for zname in sorted(os.listdir(TPW)):
        if not zname.endswith(".zip"):
            continue
        if zname not in SAS_ZIPS:
            print(f"  Skipping   {zname}  (not SAS camera)")
            continue
        zpath = TPW / zname
        print(f"  Extracting {zname} …")

        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                # skip Mac metadata
                if "__MACOSX" in member or _basename(member).startswith("._"):
                    continue

                dest = OUTDIR / member
                if member.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(member)
                dest.write_bytes(data)          # always overwrite

                if is_shot_tif(member):
                    sid = shot_id(member)
                    if sid not in seen_ids:     # deduplicate (TPW_2017 has mirror tree)
                        seen_ids.add(sid)
                        shots[member] = (zname, data)
                elif is_dark_tif(member):
                    darks[member] = data

    return shots, darks


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
        mp = ROOT / f"model_mev_n{n}.keras"
        if not mp.exists():
            continue
        print(f"  Loading model n={n} …")
        model = tf.keras.models.load_model(str(mp), compile=False)
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
    print("=== TPW MEV Inference Pipeline ===\n")

    print("Step 1 — Extracting zip files …")
    shots, darks = extract_zips()
    print(f"  {len(shots)} unique shot TIFs found\n")

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
    for path, (zname, data) in sorted(shots.items()):
        name = Path(_basename(path)).stem
        try:
            arr = load_tif_array(data)

            # dark subtraction
            dk = dark_path_for(path)
            if dk in darks:
                dark_arr = load_tif_array(darks[dk])
                arr = np.clip(arr - dark_arr, 0.0, None)

            signal = extract_spectrum(arr, n_channels=200)
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
        spec_path = ROOT / f"tpw_spectra_n{n}.csv"
        with open(spec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=spec_fields)
            w.writeheader()
            w.writerows(rows_n)
        print(f"  Spectra n={n:<3}: {spec_path.name}  ({len(rows_n)} shots × {n} bins)")

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
    print(f"  Shots   : {len(shots)} SAS shots × {len(models)} models × 3 ranks")
    print(f"  Spectra : tpw_spectra_n{{10,20,50,100,200}}.csv")
    if errors:
        print(f"  Errors  : {len(errors)}")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()
