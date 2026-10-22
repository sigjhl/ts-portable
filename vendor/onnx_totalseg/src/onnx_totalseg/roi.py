"""ROI subsetting: keep only requested structures, optionally via a crop-cascade.

- filter_roi: zero out non-requested labels (any tier).
- crop_cascade: TS-parity speedup for full-res tiers — a 6 mm localizer finds the requested
  organs, crop to their bbox (+margin), run the target tier ONLY on the crop, uncrop. The
  expensive model never touches the whole volume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import io_nibabel, registry


def total_label_maps() -> tuple:
    """(name->id, id->name) for the canonical 117-class total map."""
    name2id = {k: int(v) for k, v in json.loads(Path(registry.label_map()).read_text()).items()}
    id2name = {v: k for k, v in name2id.items() if k != "background"}
    return name2id, id2name


def resolve_roi(names: Sequence[str]) -> List[int]:
    name2id, _ = total_label_maps()
    ids, unknown = [], []
    for n in names:
        n = n.strip()
        if n in name2id:
            ids.append(name2id[n])
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(f"unknown ROI structure(s): {unknown}")
    return ids


def filter_roi(seg: np.ndarray, roi_ids: Sequence[int]) -> np.ndarray:
    keep = np.isin(seg, list(roi_ids))
    return np.where(keep, seg, 0).astype(seg.dtype)


def _bbox_of_labels(loc: np.ndarray, roi_ids, shape, spacing, margin_mm: float):
    mask = np.isin(loc, list(roi_ids))
    if not mask.any():
        return None
    idx = np.argwhere(mask)  # (z, y, x)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    marg = [int(round(margin_mm / float(s))) for s in spacing]  # spacing = (sz, sy, sx)
    z0 = max(0, lo[0] - marg[0]); z1 = min(shape[0], hi[0] + marg[0])
    y0 = max(0, lo[1] - marg[1]); y1 = min(shape[1], hi[1] + marg[1])
    x0 = max(0, lo[2] - marg[2]); x1 = min(shape[2], hi[2] + marg[2])
    return (z0, z1, y0, y1, x0, x1)


def crop_cascade(data: np.ndarray, props: dict, target_seg, localizer_seg,
                 roi_ids: Sequence[int], margin_mm: float = 20.0,
                 free_localizer: bool = False) -> np.ndarray:
    """Localize ROIs at 6 mm, crop, run the target tier on the crop only, uncrop, filter.
    Operates on an already-loaded (data, props) and returns the full-size label array.

    free_localizer: release the localizer's ORT session once the bbox is known, before the
    target model runs, so peak VRAM = one target model (not localizer + target). Safe only
    when the localizer is single-use (one-shot CLI); leave False when it is cached/shared
    (warm server), or later requests would rebuild it every call."""
    full_shape = data.shape[1:]

    loc, _ = localizer_seg.predict_loaded(data, props, lowres=False)  # full-res 6mm-derived
    bbox = _bbox_of_labels(loc, roi_ids, full_shape, props["spacing"], margin_mm)
    del loc
    if free_localizer:
        localizer_seg.close()

    if bbox is None:  # localizer found nothing -> fall back to whole volume
        seg, _ = target_seg.predict_loaded(data, props, restrict_ids=roi_ids)
    else:
        z0, z1, y0, y1, x0, x1 = bbox
        cropped = data[:, z0:z1, y0:y1, x0:x1]
        # predict_loaded uses only props["spacing"] + data.shape, so original props is fine.
        # restrict_ids -> the full multi-model runs ONLY the sub-model(s) owning the ROI labels.
        cseg, _ = target_seg.predict_loaded(cropped, props, restrict_ids=roi_ids)

        # truncation check: ROI touching an interior crop face means the organ likely
        # extends past the localizer's crop and got cut.
        rs = list(roi_ids)
        faces = [(cseg[0], z0 > 0), (cseg[-1], z1 < full_shape[0]),
                 (cseg[:, 0], y0 > 0), (cseg[:, -1], y1 < full_shape[1]),
                 (cseg[:, :, 0], x0 > 0), (cseg[:, :, -1], x1 < full_shape[2])]
        if any(interior and np.isin(face, rs).any() for face, interior in faces):
            print("[roi] WARNING: mask touches the crop boundary — the organ may extend "
                  "beyond the localizer crop and be truncated. Re-run the tier without "
                  "--roi for a guaranteed-complete mask.")

        seg = np.zeros(full_shape, dtype=np.uint8)
        seg[z0:z1, y0:y1, x0:x1] = cseg

    return filter_roi(seg, roi_ids)
