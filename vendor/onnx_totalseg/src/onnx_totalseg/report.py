"""Per-organ volume report from an output mask (the edge-metric deliverable).

volume = voxel_count * voxel_volume, read from the written NIfTI's zooms (so it is correct
for full-res tiers and for the 6 mm grid of --ultrafast alike). No heavy deps; portable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

from ._nnunet_ops import change_spacing_to_shape


def compute_statistics(image_hu: np.ndarray, seg: np.ndarray, voxel_ml: float,
                       id2name: Dict[int, str], with_intensity: bool = True) -> dict:
    """Per-organ volume (+ intensity HU) from in-memory (image_hu[z,y,x], seg[z,y,x]).

    Volume is from the seg's own grid; intensity samples the ORIGINAL image within the
    mask (seg upsampled to the image grid if coarser, e.g. --ultrafast 6 mm)."""
    seg_int = seg
    if with_intensity and seg.shape != image_hu.shape:
        seg_int = change_spacing_to_shape(seg, image_hu.shape, order=0)
    counts = np.bincount(seg.reshape(-1)) if seg.size else np.zeros(1, dtype=np.int64)
    structures = {}
    for lid, name in sorted(id2name.items()):
        n = int(counts[lid]) if 0 <= lid < len(counts) else 0
        if not n:
            continue
        entry = {"label": int(lid), "volume_ml": round(n * voxel_ml, 3)}
        if with_intensity:
            region = image_hu[seg_int == lid]
            if region.size:
                entry["intensity_mean"] = round(float(region.mean()), 2)
                entry["intensity_median"] = round(float(np.median(region)), 2)
        structures[name] = entry
    return {"voxel_ml": round(voxel_ml, 6), "n_structures": len(structures), "structures": structures}


def volume_report(mask_path: str, id2name: Dict[int, str]) -> dict:
    img = nib.load(str(mask_path))
    a = np.asarray(img.dataobj).astype(np.int64, copy=False)
    vox_ml = float(np.prod([float(z) for z in img.header.get_zooms()[:3]])) / 1000.0
    flat = a.reshape(-1)
    counts = np.bincount(flat) if flat.size and flat.max() >= 0 else np.zeros(1, dtype=np.int64)
    structures = {}
    for lid, name in sorted(id2name.items()):
        n = int(counts[lid]) if 0 <= lid < len(counts) else 0
        if n:
            structures[name] = {"label": int(lid), "voxels": n, "volume_ml": round(n * vox_ml, 3)}
    return {
        "mask": str(Path(mask_path).name),
        "voxel_ml": round(vox_ml, 6),
        "n_structures": len(structures),
        "structures": structures,
    }
