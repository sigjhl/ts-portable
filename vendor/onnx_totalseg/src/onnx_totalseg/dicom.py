"""Read a directory of DICOM files into a single volume (the largest series).

Grouping/selection is ours (pydicom headers); the DICOM->NIfTI geometry (LPS->RAS affine,
slice sorting, rescale to HU) is delegated to `dicom2nifti` so we don't hand-roll a
famously error-prone affine. One-time per study (not on the latency path).

Scope: standard single-frame CT. Compressed transfer syntaxes need pydicom pixel handlers
(pylibjpeg / python-gdcm) installed; gantry-tilt / multi-frame are not specially handled.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from uuid import uuid4

import nibabel as nib
import numpy as np
import pydicom


def _series_groups(path: str) -> dict:
    """Map SeriesInstanceUID -> [file paths] for image slices in `path`."""
    groups: dict = {}
    for entry in os.scandir(path):
        if not entry.is_file():
            continue
        try:
            ds = pydicom.dcmread(entry.path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None or getattr(ds, "Rows", None) is None:
            continue  # not an image slice
        groups.setdefault(str(uid), []).append(entry.path)
    return groups


@contextmanager
def _conversion_tempdir(path: str):
    """Create a temp dir in a writable local folder before falling back to system temp."""
    base = os.path.abspath(path)
    parent = base if os.path.isdir(base) else os.path.dirname(base)
    candidates = [os.environ.get("ONNX_TOTALSEG_TMP"), parent, os.getcwd()]
    scratch = None
    for candidate in candidates:
        if not candidate or not os.path.isdir(candidate):
            continue
        try:
            scratch = os.path.join(candidate, f"onnx_totalseg_{uuid4().hex}")
            os.makedirs(scratch)
            break
        except OSError:
            continue
    if scratch is None:
        raise OSError("could not create a temporary DICOM conversion folder")
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def read_dicom_dir(path: str) -> "nib.Nifti1Image":
    """Return the largest series in `path` as an in-memory nibabel image."""
    import dicom2nifti.convert_dicom as convert_dicom

    groups = _series_groups(path)
    if not groups:
        raise ValueError(f"no DICOM image series found in {path}")

    def voxel_count(files):
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        return int(getattr(ds, "Rows", 0)) * int(getattr(ds, "Columns", 0)) * len(files)

    uid, files = max(groups.items(), key=lambda kv: voxel_count(kv[1]))
    if len(groups) > 1:
        print(f"[dicom] {len(groups)} series found; using largest "
              f"({len(files)} slices, {voxel_count(files)} voxels)")

    datasets = [pydicom.dcmread(f, force=True) for f in files]
    with _conversion_tempdir(path) as td:
        out = os.path.join(td, "vol.nii")
        convert_dicom.dicom_array_to_nifti(datasets, out, reorient_nifti=False)
        img = nib.load(out)
        # materialize before the temp file is removed
        return nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
