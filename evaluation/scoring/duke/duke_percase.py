#!/usr/bin/env python3
"""Per-case spleen metrics for the Duke CT series (input of spleen_volumetry/duke_agreement.R).
Volumes (mL) for manual, TS-Portable on the PACS station from DICOM and from NIfTI, and the reference
implementation 2.12.0 (spleen-only, workstation); DSC and NSD at 1/2/3 mm of each against manual and
of TS-Portable against the reference. All masks share the input grid (checked).

    python scoring/duke/duke_percase.py ROOT OUT_CSV

ROOT holds images/DSDS_XXXX.nii.gz and masks/DSDS_XXXX_spleen.nii.gz (convert_duke_ct.py),
station/duke_dicom/ and station/duke_nifti/ (TS-Portable spleen-only masks from the reading station,
<case>_spleen.nii.gz), and ts2120_spleen_gpu/<case>.nii.gz (reference implementation 2.12.0,
spleen-only mode, runtime/run_ts_roi.py)."""
import csv, sys
from pathlib import Path
import nibabel as nib, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsd_core import organ_surface_metrics

ROOT = Path(sys.argv[1])
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'duke_percase.csv'
SRC = {'manual': ROOT / 'masks/{c}_spleen.nii.gz', 'portable_dicom': ROOT / 'station/duke_dicom/{c}_spleen.nii.gz',
       'portable_nifti': ROOT / 'station/duke_nifti/{c}_spleen.nii.gz', 'ref2120': ROOT / 'ts2120_spleen_gpu/{c}.nii.gz'}
PAIRS = [('portable_dicom', 'manual'), ('ref2120', 'manual'), ('portable_dicom', 'ref2120')]
rows = []
for i in range(1, 30):
    c = f'DSDS_{i:04d}'
    img = nib.load(str(ROOT / f'images/{c}.nii.gz')); sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    m = {}
    for k, p in SRC.items():
        im = nib.load(str(p).format(c=c))
        assert im.shape == img.shape and np.allclose(im.affine, img.affine, atol=1e-4), (c, k)
        a = np.asarray(im.dataobj); m[k] = a > 0 if k == 'manual' else a == 1
    vox = float(np.prod(sp)) / 1000.0
    r = {'case': c, 'excluded': int(c == 'DSDS_0021')}
    for k in SRC: r[f'{k}_ml'] = round(float(m[k].sum()) * vox, 3)
    r['dicom_eq_nifti'] = int(np.array_equal(m['portable_dicom'], m['portable_nifti']))
    for a, b in PAIRS:
        s = 2.0 * np.logical_and(m[a], m[b]).sum() / (m[a].sum() + m[b].sum())
        r[f'dsc_{a}_vs_{b}'] = round(float(s), 5)
        sm = organ_surface_metrics(m[a], m[b], sp, tolerances=(1.0, 2.0, 3.0))
        for t in (1.0, 2.0, 3.0): r[f'nsd{int(t)}_{a}_vs_{b}'] = round(float(sm['nsd'][t]), 5)
    rows.append(r); print(r, flush=True)
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
