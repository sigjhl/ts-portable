#!/usr/bin/env python3
"""Per-instance reference volumes (mL) for the volume-stratified analysis (Supplemental Table S11).

    python scoring/reference_volumes.py --amos-labels AMOS/labelsVa --ts-root Totalsegmentator_dataset_small \
        --amos-sex data/cases/amos_sex_reviewed.csv --out data/volume_strata/reference_volumes.csv

One row per reference-mask instance: dataset (amos, ts_dev), case, structure, voxel count, voxel
volume (mL), volume (mL). AMOS uses the multilabel masks of the 100 CT scans (ids below 500); the
development set uses the per-structure masks of all 117 labels. AMOS label 15 ("prostate/uterus") is
written as prostate in men and as pelvic_label_female in women (sex from --amos-sex), so that only male
instances enter the prostate's median volume.
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

AMOS2TS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "esophagus", 6: "liver",
           7: "stomach", 8: "aorta", 9: "inferior_vena_cava", 10: "pancreas", 11: "adrenal_gland_right",
           12: "adrenal_gland_left", 13: "duodenum", 14: "urinary_bladder", 15: "prostate"}

ap = argparse.ArgumentParser()
ap.add_argument("--amos-labels", required=True, help="AMOS labelsVa folder")
ap.add_argument("--ts-root", required=True, help="TotalSegmentator dataset folder (<case>/segmentations)")
ap.add_argument("--amos-sex", required=True, help="reviewed sex assignments (data/cases/amos_sex_reviewed.csv)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
with open(args.amos_sex, newline="", encoding="utf-8") as f:
    SEX = {r["case"]: r["assigned_sex"].strip().upper()[:1] for r in csv.DictReader(f)}

with open(args.out, "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["dataset", "case", "structure", "voxels", "voxel_ml", "volume_ml"])
    for path in sorted(Path(args.amos_labels).glob("amos_*.nii.gz")):
        case = path.name[:-7]
        if int(case.split("_")[1]) >= 500:  # MRI
            continue
        img = nib.load(str(path))
        a = np.asarray(img.dataobj).astype(np.int16)
        vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        for i, n in zip(*np.unique(a, return_counts=True)):
            if i != 0:
                name = AMOS2TS[int(i)]
                if name == "prostate" and SEX[case] != "M":
                    name = "pelvic_label_female"
                w.writerow(["amos", case, name, int(n), vml, int(n) * vml])
    ts = Path(args.ts_root)
    for case in sorted(p.name for p in ts.iterdir() if (p / "ct.nii.gz").exists()):
        for f in sorted((ts / case / "segmentations").glob("*.nii.gz")):
            img = nib.load(str(f))
            vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
            n = int(np.count_nonzero(np.asarray(img.dataobj)))
            w.writerow(["ts_dev", case, f.name[:-7], n, vml, n * vml])
print("wrote", args.out)
