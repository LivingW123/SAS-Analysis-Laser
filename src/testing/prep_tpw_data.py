"""
prep_tpw_data.py — Stage 1/2 of the former root-level `run_tpw_mev.py`.

Extracts the TPW real-shot dataset's zip archives (raw data lives in TPW/,
unchanged and out of scope for this refactor), processes each SAS-camera
shot TIF (dark-frame subtraction + collapse to a 200-channel spectrum), and
writes every prepped shot into a single hand-off artifact:

    out/testing/tpw_prepped_spectra.npz

This script does no model loading and no inference — it is pure data prep.
Its counterpart, `src/testing/test_dnn_tpw.py`, loads the artifact produced
here and runs the trained DNN MEV classifiers against it. Run this script
first; then run test_dnn_tpw.py.

"TPW" is not expanded anywhere in the original script or its comments — it
is referred to here only as the TPW real-shot dataset.
"""

import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

ROOT   = Path(__file__).resolve().parents[2]
TPW    = ROOT / "TPW"
OUTDIR = TPW / "extracted"
OUT_NPZ = "out/testing/tpw_prepped_spectra.npz"

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


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== TPW Data Prep (extract + spectrum extraction) ===\n")

    print("Step 1 — Extracting zip files …")
    shots, darks = extract_zips()
    print(f"  {len(shots)} unique shot TIFs found\n")

    print("Step 2 — Extracting 200-channel spectra …")
    shot_names: list[str] = []
    source_zips: list[str] = []
    spectra: list[np.ndarray] = []
    errors: list[str] = []

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

            shot_names.append(name)
            source_zips.append(zname)
            spectra.append(signal)
            print(f"  {name:40s}  OK  ({zname})")

        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"  ERROR {name}: {exc}")

    os.makedirs("out/testing", exist_ok=True)
    np.savez(
        OUT_NPZ,
        shot_names=np.array(shot_names, dtype=str),
        source_zips=np.array(source_zips, dtype=str),
        spectra=np.array(spectra, dtype=np.float32).reshape(-1, 200),
    )

    print(f"\n=== Done ===")
    print(f"  Prepped : {len(shot_names)} SAS shots")
    print(f"  Artifact: {OUT_NPZ}")
    if errors:
        print(f"  Errors  : {len(errors)}")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()
