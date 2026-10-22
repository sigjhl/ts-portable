#!/usr/bin/env python3
"""Reference implementation version 2.4.0 versus 2.12.0 (Supplemental Table S4).

Compares TotalSegmentator 2.4.0 (PACS-station version) with 2.12.0 (workstation version) on the
100 AMOS cases, both run on the same A6000 workstation:
  * spleen-only mode, GPU: spleen volume and spleen DSC, 2.4.0 vs 2.12.0 and each vs manual
  * full mode, GPU: 2.4.0 masks vs the existing 2.12.0 masks (stock_total): per-case mean DSC over the
    15 AMOS organs vs manual, and direct DSC between versions
  * timing rows from timing.jsonl (only rows produced in the quiet window are meaningful; the
    caller decides which to report)
Writes version_summary.json.

    python spleen_volumetry/version_comparison.py --amos-labels AMOS/labelsVa \
        --amos-meta AMOS/labeled_data_meta_0000_0599.csv --amos-sex data/cases/amos_sex_reviewed.csv --masks MASKS --ref-full masks/reference/amos_full \
        --timing data/runtime/workstation_version_exp_timing.jsonl --out results/spleen_volumetry

MASKS holds ts240_spleen_gpu/, ts2120_spleen_gpu/ and ts240_full_gpu/ (<case>.nii.gz, written by
runtime/version_exp.sh); --ref-full holds the 2.12.0 full-mode masks of the main analysis.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
for name in ("--amos-labels", "--amos-meta", "--masks", "--ref-full", "--timing", "--out"):
    ap.add_argument(name, required=True)
ap.add_argument("--amos-sex", help="reviewed sex assignments (data/cases/amos_sex_reviewed.csv); override the metadata")
ARGS = ap.parse_args()
ROOT = Path(ARGS.masks)
LABELS = Path(ARGS.amos_labels)
STOCK_2120_FULL = Path(ARGS.ref_full)
AMOS2TS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "esophagus", 6: "liver",
           7: "stomach", 8: "aorta", 9: "inferior_vena_cava", 10: "pancreas", 11: "adrenal_gland_right",
           12: "adrenal_gland_left", 13: "duodenum", 14: "urinary_bladder", 15: "prostate"}
GMAP = {k: int(v) for k, v in json.loads((Path(__file__).resolve().parents[1] / "data" / "total_label_map.json").read_text()).items()}
SEX = {}  # keyed by the integer AMOS id
for r in csv.DictReader(open(ARGS.amos_meta, encoding="utf-8-sig")):
    SEX[int(r["amos_id"])] = r["Patient's Sex"].strip().upper()[:1]
if ARGS.amos_sex:
    for r in csv.DictReader(open(ARGS.amos_sex, encoding="utf-8")):
        SEX[int(r["case"].split("_")[1])] = r["assigned_sex"].strip().upper()[:1]


def arr(p):
    im = nib.load(str(p)); return np.asarray(im.dataobj), float(np.prod(im.header.get_zooms()[:3])) / 1000.0


def dsc(a, b):
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0: return None
    return 2.0 * np.logical_and(a, b).sum() / (sa + sb)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def agree(t, r):
    t, r = np.asarray(t, float), np.asarray(r, float); rel = 100 * (t - r) / ((t + r) / 2)
    return {"ccc": ccc(t, r), "rel_bias_pct": float(rel.mean()),
            "rel_loa_pct": [float(rel.mean() - 1.96 * rel.std(ddof=1)), float(rel.mean() + 1.96 * rel.std(ddof=1))],
            "median_abs_rel_pct": float(np.median(np.abs(rel))), "n_rel_gt2": int((np.abs(rel) > 2).sum())}


out = {}
cases = sorted(p.name[:-7] for p in LABELS.glob("amos_*.nii.gz") if int(p.name[5:9]) < 500)

# ---- spleen-only mode, GPU: 2.4.0 vs 2.12.0 -------------------------------------------------------
d240, d2120 = ROOT / "ts240_spleen_gpu", ROOT / "ts2120_spleen_gpu"
if d240.exists() and d2120.exists():
    v_man, v240, v2120, dsc_240_man, dsc_2120_man, dsc_240_2120 = [], [], [], [], [], []
    for c in cases:
        p240, p2120 = d240 / f"{c}.nii.gz", d2120 / f"{c}.nii.gz"
        if not (p240.exists() and p2120.exists()): continue
        gt, vx = arr(LABELS / f"{c}.nii.gz"); m = gt == 1
        a, _ = arr(p240); b, _ = arr(p2120); a, b = a == GMAP["spleen"], b == GMAP["spleen"]
        v_man.append(m.sum() * vx); v240.append(a.sum() * vx); v2120.append(b.sum() * vx)
        dsc_240_man.append(dsc(a, m)); dsc_2120_man.append(dsc(b, m)); dsc_240_2120.append(dsc(a, b))
    out["spleen_only_gpu"] = {"n": len(v_man),
        "volume_240_vs_2120": agree(v240, v2120), "volume_240_vs_manual": agree(v240, v_man), "volume_2120_vs_manual": agree(v2120, v_man),
        "dsc_vs_manual_240_mean": st.mean(dsc_240_man), "dsc_vs_manual_2120_mean": st.mean(dsc_2120_man),
        "dsc_240_vs_2120_mean": st.mean(dsc_240_2120), "dsc_240_vs_2120_min": min(dsc_240_2120),
        "identical_masks": int(sum(1 for x in dsc_240_2120 if x >= 0.99999))}

# ---- full mode, GPU: 2.4.0 vs 2.12.0 (stock_total) -----------------------------------------------
dfull = ROOT / "ts240_full_gpu"
if dfull.exists():
    per_case_240, per_case_2120, direct = [], [], {}
    for c in cases:
        p = dfull / f"{c}.nii.gz"; q = STOCK_2120_FULL / f"{c}.nii.gz"
        if not (p.exists() and q.exists()): continue
        gt, _ = arr(LABELS / f"{c}.nii.gz"); a, _ = arr(p); b, _ = arr(q)
        o240, o2120 = [], []
        for lab, name in AMOS2TS.items():
            m = gt == lab
            if m.sum() == 0: continue
            if name == "prostate" and SEX.get(int(c.split("_")[1]), "?") != "M": continue
            ga, gb = a == GMAP[name], b == GMAP[name]
            o240.append(dsc(ga, m) or 0.0); o2120.append(dsc(gb, m) or 0.0)
            dd = dsc(ga, gb)
            if dd is not None: direct.setdefault(name, []).append(dd)
        per_case_240.append(st.mean(o240)); per_case_2120.append(st.mean(o2120))
    diffs = np.array(per_case_2120) - np.array(per_case_240)
    out["full_gpu"] = {"n": len(per_case_240), "mean_dsc_vs_manual_240": st.mean(per_case_240),
        "mean_dsc_vs_manual_2120": st.mean(per_case_2120), "per_case_diff_2120_minus_240_mean": float(diffs.mean()),
        "per_case_diff_abs_max": float(np.abs(diffs).max()),
        "direct_dsc_by_organ_mean": {k: st.mean(v) for k, v in direct.items()},
        "direct_dsc_by_organ_min": {k: min(v) for k, v in direct.items()}}

# ---- timing -------------------------------------------------------------------------------------
rows = [json.loads(l) for l in Path(ARGS.timing).read_text().splitlines() if l.strip()]
tim = {}
for r in rows:
    if r.get("rc") != 0: continue
    cond, mode, dev, case = r["label"].split("|")
    tim.setdefault(f"{cond}|{mode}|{dev}", []).append(r["proc_wall_s"])
out["timing_median_wall_s"] = {k: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in tim.items()}
Path(ARGS.out).mkdir(parents=True, exist_ok=True)
(Path(ARGS.out) / "version_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
