"""NibabelIOWithReorient read/write, ported from nnunetv2.imageio (Apache-2.0, DKFZ).

The fast model's plans specify `NibabelIOWithReorient`, so the runtime must read/write
identically (RAS reorient, sitk-style axis order, exact inverse on write) to reproduce
stock geometry.
"""
from __future__ import annotations

import os
from typing import Tuple

import nibabel
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

try:  # ISA-L igzip: ~3x faster gzip decompress, lossless, AMD/Intel, Windows wheels
    from isal import igzip as _igzip
except Exception:  # pragma: no cover - falls back to stdlib gzip via nibabel
    _igzip = None


def _img_from_raw(raw: bytes):
    """Build a NIfTI image from uncompressed bytes (NIfTI-1 or -2, either endianness)."""
    sizes = (int.from_bytes(raw[:4], "little"), int.from_bytes(raw[:4], "big"))
    if 348 in sizes:
        return nibabel.Nifti1Image.from_bytes(raw)
    if 540 in sizes:
        return nibabel.Nifti2Image.from_bytes(raw)
    raise ValueError("unrecognized NIfTI header size")


def _load_nifti(fname: str):
    """Load a NIfTI; for .gz use ISA-L igzip (single-member standard gzip, no sidecar).
    Falls back to nibabel.load if isal is absent or anything is unexpected."""
    fname = str(fname)
    if _igzip is not None and fname.endswith(".gz"):
        try:
            with open(fname, "rb") as fh:
                raw = _igzip.decompress(fh.read())
            return _img_from_raw(raw)
        except Exception:
            pass  # any surprise -> safe stdlib path
    return nibabel.load(fname)


def _from_nib_image(nib_image) -> Tuple[np.ndarray, dict]:
    """(data[1,z,y,x] float32, props) from a nibabel image, matching NibabelIOWithReorient."""
    assert nib_image.ndim == 3, "only 3d images are supported"
    original_affine = nib_image.affine
    reoriented = nib_image.as_reoriented(io_orientation(original_affine))
    reoriented_affine = reoriented.affine
    spacing = [float(i) for i in reoriented.header.get_zooms()[::-1]]  # reversed to sitk order
    # read as float32 directly (CT HU are integer-exact in float32; downstream casts to
    # int32 anyway) — avoids the float64 materialization get_fdata() would do.
    data = np.asarray(reoriented.dataobj, dtype=np.float32).transpose((2, 1, 0))[None]
    props = {
        "nibabel_stuff": {"original_affine": original_affine, "reoriented_affine": reoriented_affine},
        "spacing": spacing,
    }
    return data, props


def read_image(fname: str) -> Tuple[np.ndarray, dict]:
    """Return (data[1,z,y,x] float32, props). Accepts a NIfTI file OR a directory of DICOM
    files (the largest series is used)."""
    fname = str(fname)
    if os.path.isdir(fname):
        from . import dicom
        return _from_nib_image(dicom.read_dicom_dir(fname))
    return _from_nib_image(_load_nifti(fname))


def lowres_affine(props: dict, target_spacing) -> np.ndarray:
    """RAS-canonical affine for a mask left at `target_spacing` (no upsample).

    Matches TS change_spacing: scale the reoriented affine's direction columns by
    target/original spacing, keep the origin. Pairs with our scipy.zoom resample so the
    coarse mask lands on the same physical grid the model saw.
    """
    aff = np.asarray(props["nibabel_stuff"]["reoriented_affine"], dtype=float).copy()
    orig_xyz = props["spacing"][::-1]  # (sx, sy, sz)
    for j in range(3):
        aff[:3, j] = aff[:3, j] * (float(target_spacing[j]) / float(orig_xyz[j]))
    return aff


def write_label_masks(seg: np.ndarray, out_dir: str, props: dict, id2name: dict,
                      restrict=None, canonical_affine=None, threads: int = 8) -> list:
    """Write one binary <name>.nii.gz per present structure (TS-style), in parallel.
    restrict: optional set of label ids to limit to (e.g. an ROI)."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(out_dir, exist_ok=True)
    present = [int(x) for x in np.unique(seg) if x != 0]
    if restrict is not None:
        present = [p for p in present if p in restrict]

    def _w(lid):
        name = id2name.get(lid)
        if name is None:
            return
        write_seg((seg == lid).astype(np.uint8), os.path.join(out_dir, f"{name}.nii.gz"),
                  props, canonical_affine=canonical_affine)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        list(ex.map(_w, present))
    return [id2name[p] for p in present if p in id2name]


def write_seg(seg: np.ndarray, output_fname: str, props: dict, canonical_affine=None) -> None:
    """Write a label map (z,y,x order) back to the original geometry.

    canonical_affine overrides the RAS-canonical affine (used for low-res output where the
    mask stays at the model spacing instead of being upsampled).
    """
    seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)
    aff = props["nibabel_stuff"]["reoriented_affine"] if canonical_affine is None else canonical_affine
    seg_nib = nibabel.Nifti1Image(seg, affine=aff)
    img_ornt = io_orientation(props["nibabel_stuff"]["original_affine"])
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    seg_reoriented = seg_nib.as_reoriented(from_canonical)
    nibabel.save(seg_reoriented, output_fname)
