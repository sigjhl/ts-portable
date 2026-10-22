#!/usr/bin/env python3
"""Spleen-only mode: per-case spleen DSC and NSD@3 mm of each configuration against the manual AMOS
spleen (input of noninferiority/spleen_only_noninferiority.R).

    python scoring/score_spleen_only.py --amos-labels AMOS/labelsVa \
        --arm tsp_pacs=masks/station/int8_openvino \
        --arm ref240_pacs=masks/station/stock_ts_cpu \
        --arm ref2120_ws=masks/reference_spleen_only_2.12.0 \
        --arm ref240_ws=masks/reference_spleen_only_2.4.0 \
        --out data/spleen/amos_spleen_only_mode_metrics.csv

Each arm folder holds <case>_spleen.nii.gz or <case>.nii.gz. The spleen is label 1; a binary mask
with another single label is read by that label. An empty prediction scores 0.
"""
import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsd_core import surface_dice  # noqa: E402


def spleen(folder: Path, case: str):
    for name in (f"{case}_spleen.nii.gz", f"{case}.nii.gz"):
        if (folder / name).exists():
            a = np.asarray(nib.load(str(folder / name)).dataobj)
            if not (a == 1).any() and len(np.unique(a)) == 2:
                return a > 0
            return a == 1
    raise FileNotFoundError(f"{folder}/{case}")


ap = argparse.ArgumentParser()
ap.add_argument("--amos-labels", required=True)
ap.add_argument("--arm", action="append", required=True, help="NAME=FOLDER")
ap.add_argument("--out", required=True)
args = ap.parse_args()
arms = [a.split("=", 1) for a in args.arm]

rows = []
for p in sorted(Path(args.amos_labels).glob("amos_*.nii.gz")):
    case = p.name[:-7]
    if int(case.split("_")[1]) >= 500:
        continue
    img = nib.load(str(p))
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    manual = np.asarray(img.dataobj) == 1
    r = {"case": case}
    for name, folder in arms:
        m = spleen(Path(folder), case)
        r[f"dsc_{name}"] = 2.0 * np.logical_and(m, manual).sum() / (m.sum() + manual.sum())
        r[f"nsd3_{name}"] = surface_dice(m, manual, spacing, 3.0)
    rows.append(r)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", len(rows), "cases to", args.out)
