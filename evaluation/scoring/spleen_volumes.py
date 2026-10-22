#!/usr/bin/env python3
"""Per-case spleen volumes, 100 external AMOS cases (input of spleen_volumetry/amos_spleen_agreement.R).

    python scoring/spleen_volumes.py --amos-labels AMOS/labelsVa \
        --ref-full masks/reference/amos_full --int8-full masks/tsportable/amos_full \
        --fp32-full masks/tsportable_fp32/amos_full --pacs masks/station \
        --out data/spleen/amos_spleen_volumes_percase.csv

Volume = voxel count x voxel volume on the input grid. Manual reference = AMOS label 1; full-mode
masks use TotalSegmentator's global id 1 (spleen). --pacs holds one folder per reading-station
configuration (int8_openvino, stock_ts_cpu, stock_ts_cuda) with <case>_spleen.nii.gz from spleen-only
mode; a binary mask whose label is not 1 is read by its single nonzero label.
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

ARMS = ("int8_openvino", "stock_ts_cpu", "stock_ts_cuda")


def vol(p, label):
    im = nib.load(str(p))
    a = np.asarray(im.dataobj)
    vx = float(np.prod(im.header.get_zooms()[:3])) / 1000.0
    return float(np.count_nonzero(a == label) * vx), np.unique(a)


ap = argparse.ArgumentParser()
for name in ("--amos-labels", "--ref-full", "--int8-full", "--fp32-full", "--pacs", "--out"):
    ap.add_argument(name, required=True)
args = ap.parse_args()

cases = sorted(p.name[:-7] for p in Path(args.amos_labels).glob("amos_*.nii.gz") if int(p.name[5:9]) < 500)
rows = []
for c in cases:
    r = {"case": c}
    r["manual_ml"], _ = vol(Path(args.amos_labels) / f"{c}.nii.gz", 1)
    r["reference_full_ml"], _ = vol(Path(args.ref_full) / f"{c}.nii.gz", 1)
    r["int8_full_ml"], _ = vol(Path(args.int8_full) / f"{c}.nii.gz", 1)
    r["fp32_full_ml"], _ = vol(Path(args.fp32_full) / f"{c}.nii.gz", 1)
    for arm in ARMS:
        p = Path(args.pacs) / arm / f"{c}_spleen.nii.gz"
        v, labs = vol(p, 1) if p.exists() else (float("nan"), [])
        if p.exists() and v == 0 and len(labs) > 1:  # binary mask with another label value
            v, labs = vol(p, int(labs[labs > 0][0]))
        r[f"pacs_{arm}_ml"] = v
        r[f"pacs_{arm}_labels"] = " ".join(str(int(x)) for x in labs[:4])
    rows.append(r)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", len(rows), "cases to", args.out)
