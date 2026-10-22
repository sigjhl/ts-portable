#!/usr/bin/env python3
"""Score TS-Portable and the reference implementation against the dataset reference masks.

Writes the per-case records that the noninferiority, tail, organ, and stratum analyses read
(data/accuracy/ in this repository holds the records of the reported analysis):

    percase_<cohort>_<tier>.json                           DSC, NSD@1/2/3 mm, direct DSC, empty flags
    nsd_tolerance/nsd_tolerance_percase_<cohort>_<tier>.json   NSD@1/2 mm case means
    nsd_tolerance/raw_amos_<tier>.jsonl                    NSD@1/2 mm per organ (AMOS)
    fp32/dsc_percase_<cohort>_<tier>.json, fp32/nsd_*      fp32 ONNX arm, if --pred-fp32 is given

cohort is amos (external AMOS, 15 organs) or ts (TotalSegmentator development set, 117 labels);
tier is full (1.5-mm models) or fast (3-mm models).

Rules (Methods, "Endpoints"): every structure present in the reference mask is scored; an empty
prediction scores 0 on DSC and NSD; structure values are averaged with equal weight within a case.
In AMOS, label 15 ("prostate/uterus") is scored as the prostate in male patients only. Sex is taken from
--amos-sex (data/cases/amos_sex_reviewed.csv: sex assigned by the author from the pelvic anatomy of each
scan, as in the reported analysis), or else from the AMOS metadata CSV.
NSD is the surface-area-weighted surface Dice of the DeepMind surface-distance package, version 0.1
(scoring/nsd_core.py), computed on a crop around the union of the masks with a 6-mm margin.

Prediction folders hold one multilabel NIfTI per case, <case>.nii.gz, with TotalSegmentator's
global label ids (data/total_label_map.json), as written by both TS-Portable and the reference
implementation (`TotalSegmentator --ml`); 3-mm reference masks come from `TotalSegmentator --fast`.

Examples:

    python scoring/score_accuracy.py --cohort amos --tier full \\
        --amos-images AMOS/imagesVa --amos-labels AMOS/labelsVa \\
        --amos-meta AMOS/labeled_data_meta_0000_0599.csv --amos-sex data/cases/amos_sex_reviewed.csv \\
        --pred-int8 masks/tsportable/amos_full --pred-ref masks/reference/amos_full --out my_results

    python scoring/score_accuracy.py --cohort ts --tier fast --ts-root Totalsegmentator_dataset_small \\
        --pred-int8 masks/tsportable/ts_fast --pred-ref masks/reference/ts_fast --out my_results
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nsd_core import organ_surface_metrics  # noqa: E402

LABEL_MAP = HERE.parent / "data" / "total_label_map.json"
AMOS2TS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "inferior_vena_cava", 10: "pancreas", 11: "adrenal_gland_right",
    12: "adrenal_gland_left", 13: "duodenum", 14: "urinary_bladder", 15: "prostate",
}
TOLERANCES = (1.0, 2.0, 3.0)


def load(path: Path):
    image = nib.load(str(path))
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    return np.asarray(image.dataobj).astype(np.int16), spacing


def crop(masks, spacing, pad_mm: float = 6.0):
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    if not union.any():
        raise ValueError("reference-present structure has an empty union")
    idx = np.where(union)
    slices = []
    for axis in range(3):
        pad = int(np.ceil(pad_mm / spacing[axis])) + 1
        slices.append(slice(max(0, idx[axis].min() - pad), min(union.shape[axis], idx[axis].max() + pad + 1)))
    return tuple(slices)


def dice(pred, ref):
    denominator = int(pred.sum()) + int(ref.sum())
    if denominator == 0:
        return None
    return 2.0 * int(np.logical_and(pred, ref).sum()) / denominator


def amos_sexes(meta_csv: Path, reviewed_csv: Path | None = None) -> dict[int, str]:
    """Sex by integer AMOS id: the reviewed assignment where given, else the AMOS metadata."""
    with open(meta_csv, newline="", encoding="utf-8-sig") as f:
        sexes = {int(r["amos_id"]): r["Patient's Sex"].strip().upper()[:1] for r in csv.DictReader(f)}
    if reviewed_csv:
        with open(reviewed_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sexes[int(r["case"].split("_")[1])] = r["assigned_sex"].strip().upper()[:1]
    return sexes


def reference_structures(args, case: str, gmap: dict[str, int]):
    """Yield (structure, boolean reference mask, spacing) for every reference-present structure."""
    if args.cohort == "amos":
        gt, spacing = load(Path(args.amos_labels) / f"{case}.nii.gz")
        sex = args.sexes.get(int(case.split("_")[1]), "?")
        for amos_id, name in AMOS2TS.items():
            if name == "prostate" and sex != "M":
                continue
            manual = gt == amos_id
            if manual.any():
                yield name, manual, spacing
    else:
        for name in gmap:
            if name == "background":
                continue
            path = Path(args.ts_root) / case / "segmentations" / f"{name}.nii.gz"
            if not path.exists():
                continue
            manual, spacing = load(path)
            manual = manual > 0
            if manual.any():
                yield name, manual, spacing


def score_case(job):
    args, case, gmap = job
    int8, _ = load(Path(args.pred_int8) / f"{case}.nii.gz")
    ref, _ = load(Path(args.pred_ref) / f"{case}.nii.gz")
    fp32 = load(Path(args.pred_fp32) / f"{case}.nii.gz")[0] if args.pred_fp32 else None
    organs = []
    for name, manual, spacing in reference_structures(args, case, gmap):
        if int8.shape != manual.shape or ref.shape != manual.shape:
            raise ValueError(f"shape mismatch for {case}/{name}")
        gid = gmap[name]
        m_int8, m_ref = int8 == gid, ref == gid
        region = crop([m_int8, m_ref, manual], spacing)
        s_int8 = organ_surface_metrics(m_int8[region], manual[region], spacing, TOLERANCES)
        s_ref = organ_surface_metrics(m_ref[region], manual[region], spacing, TOLERANCES)
        rec = {
            "organ": name,
            "nsd1_int8": s_int8["nsd"][1.0], "nsd2_int8": s_int8["nsd"][2.0], "nsd3_int8": s_int8["nsd"][3.0],
            "nsd1_stock_ts": s_ref["nsd"][1.0], "nsd2_stock_ts": s_ref["nsd"][2.0], "nsd3_stock_ts": s_ref["nsd"][3.0],
            "hd95_int8": s_int8["hd95"], "hd95_stock_ts": s_ref["hd95"],
            "dsc_int8_vs_manual": dice(m_int8, manual) or 0.0,
            "dsc_stock_ts_vs_manual": dice(m_ref, manual) or 0.0,
            "dsc_int8_vs_stock_ts": dice(m_int8, m_ref),
            "int8_empty": not m_int8.any(),
            "stock_ts_empty": not m_ref.any(),
        }
        if fp32 is not None:
            m_fp = fp32 == gid
            region = crop([m_fp, m_ref, manual], spacing)
            rec["nsd3_fp32"] = organ_surface_metrics(m_fp[region], manual[region], spacing, (3.0,))["nsd"][3.0]
            rec["dsc_fp32_vs_manual"] = dice(m_fp, manual) or 0.0
            rec["fp32_empty"] = not m_fp.any()
        organs.append(rec)
    if not organs:
        raise ValueError(f"no reference-present structure in {case}")
    row = {"case": case}
    if args.cohort == "amos":
        row["sex"] = args.sexes.get(int(case.split("_")[1]), "?")
    mean = lambda key: st.mean(o[key] for o in organs)  # noqa: E731
    row.update({
        "mean_nsd3_int8": mean("nsd3_int8"), "mean_nsd3_stock_ts": mean("nsd3_stock_ts"),
        "mean_dsc_int8": mean("dsc_int8_vs_manual"), "mean_dsc_stock_ts": mean("dsc_stock_ts_vs_manual"),
        "organs": organs,
    })
    return row


def list_cases(args) -> list[str]:
    if args.cases:
        return [c.strip() for c in Path(args.cases).read_text().split() if c.strip()]
    if args.cohort == "amos":
        # The AMOS validation release; ids 500 and above are MRI and are excluded.
        cases = [p.name[:-7] for p in Path(args.amos_labels).glob("amos_*.nii.gz")]
        return sorted(c for c in cases if int(c.split("_")[1]) < 500)
    return sorted(p.name for p in Path(args.ts_root).iterdir() if (p / "segmentations").is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", choices=("amos", "ts"), required=True)
    ap.add_argument("--tier", choices=("full", "fast"), required=True)
    ap.add_argument("--pred-int8", required=True, help="TS-Portable masks, <case>.nii.gz")
    ap.add_argument("--pred-ref", required=True, help="reference implementation masks, <case>.nii.gz")
    ap.add_argument("--pred-fp32", help="optional: TS-Portable fp32 ONNX masks (Supplemental Table S5)")
    ap.add_argument("--amos-labels", help="AMOS labelsVa folder (amos_XXXX.nii.gz)")
    ap.add_argument("--amos-meta", help="AMOS labeled_data_meta_0000_0599.csv (patient sex)")
    ap.add_argument("--amos-sex", help="reviewed sex assignments (data/cases/amos_sex_reviewed.csv); override the metadata")
    ap.add_argument("--ts-root", help="TotalSegmentator dataset folder (<case>/segmentations/*.nii.gz)")
    ap.add_argument("--cases", help="optional text file of case ids (default: all)")
    ap.add_argument("--label-map", default=str(LABEL_MAP))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()
    if args.cohort == "amos" and not (args.amos_labels and args.amos_meta):
        ap.error("--cohort amos needs --amos-labels and --amos-meta")
    if args.cohort == "ts" and not args.ts_root:
        ap.error("--cohort ts needs --ts-root")
    args.sexes = (amos_sexes(Path(args.amos_meta), Path(args.amos_sex) if args.amos_sex else None)
                  if args.cohort == "amos" else {})
    gmap = {k: int(v) for k, v in json.loads(Path(args.label_map).read_text()).items()}

    cases = list_cases(args)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(score_case, [(args, c, gmap) for c in cases]))

    key = f"{args.cohort}_{args.tier}"
    out = Path(args.out)
    (out / "nsd_tolerance").mkdir(parents=True, exist_ok=True)
    (out / f"percase_{key}.json").write_text(json.dumps(rows, indent=1) + "\n")

    tol_rows = []
    for r in rows:
        organs = [{"organ": o["organ"], "nsd1_int8": o["nsd1_int8"], "nsd1_original_ts": o["nsd1_stock_ts"],
                   "nsd2_int8": o["nsd2_int8"], "nsd2_original_ts": o["nsd2_stock_ts"],
                   "int8_empty": o["int8_empty"], "original_ts_empty": o["stock_ts_empty"]} for o in r["organs"]]
        tol_rows.append({"case": r["case"],
                         **{f"mean_nsd{t}_{arm}": st.mean(o[f"nsd{t}_{arm}"] for o in organs)
                            for t in (1, 2) for arm in ("int8", "original_ts")},
                         "organs": organs})
    (out / "nsd_tolerance" / f"nsd_tolerance_percase_{key}.json").write_text(json.dumps(tol_rows, indent=1) + "\n")
    if args.cohort == "amos":
        with open(out / "nsd_tolerance" / f"raw_amos_{args.tier}.jsonl", "w") as f:
            for r in tol_rows:
                f.write(json.dumps(r) + "\n")

    if args.pred_fp32:
        (out / "fp32").mkdir(exist_ok=True)
        (out / "fp32" / f"dsc_percase_{key}.json").write_text(json.dumps(
            [{"case": r["case"], "mean_dsc_onnx_fp32": st.mean(o["dsc_fp32_vs_manual"] for o in r["organs"]),
              "mean_dsc_original_ts": r["mean_dsc_stock_ts"]} for r in rows], indent=1) + "\n")
        nsd = [{"case": r["case"], "organs": [{"organ": o["organ"], "i8": o["nsd3_int8"], "f3": o["nsd3_fp32"],
                                               "i8_empty": o["int8_empty"], "f3_empty": o["fp32_empty"]}
                                              for o in r["organs"]]} for r in rows]
        if args.cohort == "ts":
            (out / "fp32" / f"nsd_ts_percase_{args.tier}.json").write_text(json.dumps(nsd, indent=1) + "\n")
        else:  # one file for both AMOS tiers, keyed by tier
            path = out / "fp32" / "nsd_amos_percase.json"
            both = json.loads(path.read_text()) if path.exists() else {}
            both[args.tier] = nsd
            path.write_text(json.dumps(both, indent=1) + "\n")

    d = [r["mean_nsd3_stock_ts"] - r["mean_nsd3_int8"] for r in rows]
    e = [r["mean_dsc_stock_ts"] - r["mean_dsc_int8"] for r in rows]
    print(f"[{key}] n={len(rows)}  NSD@3 TS-Portable {st.mean(r['mean_nsd3_int8'] for r in rows):.4f} "
          f"reference {st.mean(r['mean_nsd3_stock_ts'] for r in rows):.4f} diff {st.mean(d):+.5f}; "
          f"DSC diff {st.mean(e):+.5f}  -> {out}")


if __name__ == "__main__":
    main()
