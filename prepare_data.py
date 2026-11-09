"""Build the OCT sample store from oct/ and/or TrainingSet folders + TrainingAnnotation.csv.

Source layouts: oct/AHA-XXXX/{OD,OS}/AHA-XXXX_{OD|OS}_{01=pre,02=post}_OCT1/*_NNN.jpg
                TrainingSetN/<folder>/0000-XXXX{L|R}_{1=pre,2=post}NNN.jpg   (--trainingset_dir)
CSV IDs:        0000-XXXX{L|R}   (L = OS / left eye, R = OD / right eye)

The sample key AHA-XXXX always uses the CSV number XXXX. The oct/ folder numbers are a
different numbering (oct/AHA-0148 = CSV 0000-0000); --oct_map translates them. oct/ holds
only the 108 DME eyes that are also in TrainingSet1+2, so it is skipped by default.

Every slice of every scan whose eye has a CSV row becomes one sample folder:

    data/index.csv                       one row per sample (metadata + features + labels)
    data/<sample_id>/                    sample_id = AHA-XXXX_<eye>_<pre|post>_<NNN>
        image.jpg       cropped OCT B-scan
        metadata.json   patient, eye, visit, image_idx, source image, crop size
        features.json   baseline / clinical values (incl. pre-treatment flags)
        labels.json     post-treatment values (null = missing)

There is no train/val/test split here. main_finetune.py filters the samples
(eye, visit, image index, diagnosis, anti-VEGF) and splits by patient at run time.

Usage:  python prepare_data.py --trainingset_dir <TrainingSet1> <TrainingSet2> [--out data] [--overwrite]
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EYE_TO_CSV = {"OD": "R", "OS": "L"}
VISIT_CODES = {"01": "pre", "02": "post"}

FEATURES = ["gender", "age", "diagnosis", "anti-VEGF",
            "preVA", "preCST", "preIRF", "preSRF", "prePED", "preHRF"]
LABELS = ["VA", "continue injection", "CST", "IRF", "SRF", "PED", "HRF"]

# Heidelberg export: fundus on the left (x < 496), B-scan on the right, text below y = 496.
PANEL_X0, PANEL_Y1 = 496, 496
# The scale bar sits in the bottom-left corner of the B-scan panel; skip those columns
# when measuring where the image content ends.
SCALEBAR_X1 = 570


def crop_oct(img: Image.Image, thresh: int = 8, min_frac: float = 0.02) -> Image.Image:
    """Crop to the OCT B-scan and trim the black margins around it."""
    gray = np.asarray(img.convert("L"))
    panel = gray[:PANEL_Y1, PANEL_X0:]
    bright = panel > thresh

    row_frac = bright[:, SCALEBAR_X1 - PANEL_X0:].mean(axis=1)
    rows = np.where(row_frac > min_frac)[0]
    col_frac = bright[: max(1, int(rows.max() * 0.85)) if len(rows) else PANEL_Y1].mean(axis=0)
    cols = np.where(col_frac > min_frac)[0]

    if len(rows) == 0 or len(cols) == 0:  # blank panel: keep the full panel
        y0, y1, x0, x1 = 0, PANEL_Y1, 0, panel.shape[1]
    else:
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return img.crop((PANEL_X0 + x0, y0, PANEL_X0 + x1, y1))


def to_json_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.floating, float, np.integer, int)):
        v = float(v)
        return int(v) if v.is_integer() else v
    return v


def scan_oct_dir(oct_dir: Path, oct_map: Path):
    """Yield (patient_num, eye, visit, idx, path) from the oct/AHA-XXXX/... layout.

    The AHA numbers are NOT the CSV numbers (AHA-0148 is 0000-0000, not 0000-0148).
    oct_map (oct_eye -> csv_eye) was found by matching image content against the
    TrainingSets: oct/ is exactly the 108 DME eyes of TrainingSet1+2, renumbered.
    """
    mapping = pd.read_csv(oct_map).set_index("oct_eye")["csv_eye"].to_dict()
    for pdir in sorted(oct_dir.glob("AHA-*")):
        for eye in EYE_TO_CSV:
            scans = sorted((pdir / eye).glob(f"{pdir.name}_{eye}_*_OCT1"))
            if not scans:
                continue
            csv_eye = mapping.get(f"{pdir.name}_{eye}")
            if csv_eye is None:
                print(f"  skip {pdir.name}/{eye} (not in {oct_map})")
                continue
            num = csv_eye[5:9]
            for scan_dir in scans:
                visit = VISIT_CODES.get(scan_dir.name.split("_")[2])
                if visit is None:
                    print(f"  skip {scan_dir} (unknown visit code)")
                    continue
                for img_path in sorted(scan_dir.glob("*.jpg")):
                    yield num, eye, visit, int(img_path.stem.rsplit("_", 1)[1]), img_path


# TrainingSet layout: <dir>/<folder>/0000-XXXX{L|R}_{1=pre,2=post}NNN.jpg  (e.g. 0000-0251L_1003.jpg).
# A few TrainingSet1 files deviate: 0000-0063L_1.jpg, 0000-0075L_2_000003.jpg, *.tif,
# and 0000-0163R_2 (1).jpg ... (31) (1-based, so slice = N - 1).
TS_NAME = re.compile(r"^\d{4}-(\d{4})([LR])_([12])(?:_?(\d*)| \((\d+)\))$")


def scan_trainingset_dir(ts_dir: Path):
    """Yield (patient_num, eye, visit, idx, path) from a TrainingSet folder."""
    csv_to_eye = {v: k for k, v in EYE_TO_CSV.items()}
    for img_path in sorted(ts_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        m = TS_NAME.match(img_path.stem)
        if not m:
            print(f"  skip {img_path} (unrecognised name)")
            continue
        num, side, v, idx, paren = m.groups()
        idx = int(paren) - 1 if paren else int(idx or 0)
        yield num, csv_to_eye[side], {"1": "pre", "2": "post"}[v], idx, img_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oct_dir", default="",
                    help="oct/AHA-XXXX layout (default: skip; TrainingSet1+2 already contain every oct/ eye)")
    ap.add_argument("--oct_map", default="oct_to_csv_map.csv", help="oct eye -> CSV eye mapping")
    ap.add_argument("--trainingset_dir", nargs="*", default=[],
                    help="TrainingSet folder(s) with 0000-XXXX{L|R}_{1|2}NNN.jpg files")
    ap.add_argument("--csv", default="TrainingAnnotation.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--overwrite", action="store_true", help="delete --out first if it exists")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(out)

    df = pd.read_csv(args.csv, na_values=["nan"])
    df["patient ID"] = df["patient ID"].astype(str).str.strip()
    df = df.drop_duplicates("patient ID").set_index("patient ID")

    sources = []
    if args.oct_dir:
        sources.append(scan_oct_dir(Path(args.oct_dir), Path(args.oct_map)))
    sources += [scan_trainingset_dir(Path(d)) for d in args.trainingset_dir]

    out.mkdir(parents=True)
    records, skipped_eyes, duplicates = [], set(), 0
    for source in sources:
        for num, eye, visit, idx, img_path in source:
            patient = f"AHA-{num}"   # same patient key for every source, so splits stay by patient
            csv_id = f"0000-{num}{EYE_TO_CSV[eye]}"
            if csv_id not in df.index:
                skipped_eyes.add(f"{patient}/{eye} (no CSV row {csv_id})")
                continue
            sample_id = f"{patient}_{eye}_{visit}_{idx:03d}"
            sdir = out / sample_id
            if sdir.exists():  # same slice from an earlier source (TrainingSet1 contains oct/)
                duplicates += 1
                continue
            sdir.mkdir()

            row = df.loc[csv_id]
            features = {k: to_json_value(row.get(k)) for k in FEATURES}
            labels = {k: to_json_value(row.get(k)) for k in LABELS}
            crop = crop_oct(Image.open(img_path).convert("RGB"))
            crop.save(sdir / "image.jpg", quality=95)
            metadata = {
                "sample_id": sample_id,
                "patient": patient,
                "csv_id": csv_id,
                "eye": eye,
                "visit": visit,
                "image_idx": idx,
                "source_image": img_path.as_posix(),
                "crop_size": list(crop.size),
            }
            (sdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            (sdir / "features.json").write_text(json.dumps(features, indent=2))
            (sdir / "labels.json").write_text(json.dumps(labels, indent=2))
            records.append({**{k: metadata[k] for k in
                               ["sample_id", "patient", "csv_id", "eye", "visit", "image_idx"]},
                            **features, **labels})

    index = pd.DataFrame(records)
    index.to_csv(out / "index.csv", index=False)

    n_eyes = index.groupby(["patient", "eye"]).ngroups
    print(f"Wrote {len(index)} samples from {index.patient.nunique()} patients / {n_eyes} eyes to {out}/")
    print(index.groupby(["eye", "visit"]).size().rename("samples").to_string())
    if duplicates:
        print(f"Skipped {duplicates} duplicate slices (already written from an earlier source)")
    print(f"Skipped {len(skipped_eyes)} eyes:")
    for s in sorted(skipped_eyes):
        print(f"  {s}")


if __name__ == "__main__":
    main()
