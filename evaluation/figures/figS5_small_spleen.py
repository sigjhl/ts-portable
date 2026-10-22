#!/usr/bin/env python3
"""Supplemental Figure S5: the smallest spleen (amos_0167) in spleen-only and full mode.

Needs the AMOS image and label of amos_0167, the reading-station spleen-only masks, TS-Portable's
full-mode mask, and two intermediate outputs of TS-Portable's spleen-only mode for this case (the 6-mm
localizer mask resampled to the input grid and the cropped CT):

    AMOS_IMAGES=... AMOS_LABELS=... STATION_MASKS=masks/station TSPORTABLE_MASKS=masks/tsportable \
    LOCALIZER=loc6mm_int8.nii.gz CROP=ct_crop.nii.gz python figures/figS5_small_spleen.py
Row 1 axial, row 2 coronal, through the manual spleen. Panel A shows the manual spleen, the 6-mm
localizer's spleen (resampled to the input grid), and the spleen-only crop box; B-D show the
reference implementation and TS-Portable in spleen-only mode and TS-Portable in full mode.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from nibabel.processing import resample_from_to

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})

CASE = "amos_0167"
ENV = os.environ
CT = Path(ENV["AMOS_IMAGES"]) / f"{CASE}.nii.gz"
GT = Path(ENV["AMOS_LABELS"]) / f"{CASE}.nii.gz"
LOC = Path(ENV["LOCALIZER"])
CROP = Path(ENV["CROP"])
REF_SO = Path(ENV["STATION_MASKS"]) / "stock_ts_cpu" / f"{CASE}_spleen.nii.gz"
TSP_SO = Path(ENV["STATION_MASKS"]) / "int8_openvino" / f"{CASE}_spleen.nii.gz"
TSP_FULL = Path(ENV["TSPORTABLE_MASKS"]) / "full" / f"{CASE}.nii.gz"
OUT = Path(ENV.get("OUT_DIR", Path(__file__).resolve().parents[1] / "results" / "figures")) / "figS5_small_spleen"

GREEN, RED, CYAN, YELLOW = "#2a9d5c", "#d1495b", "#3fc1d9", "#f2c14e"


def canonical(image, data):
    return np.asarray(nib.as_closest_canonical(nib.Nifti1Image(data.astype(np.uint8), image.affine)).dataobj) > 0


def main():
    ct_img = nib.load(str(CT))
    ct = np.asarray(nib.as_closest_canonical(ct_img).dataobj).astype(np.float32)
    voxel_ml = abs(np.linalg.det(ct_img.affine[:3, :3])) / 1000.0

    def mask(path, label=1):
        return canonical(ct_img, np.asarray(nib.load(str(path)).dataobj) == label)

    manual = mask(GT)
    loc_img = nib.load(str(LOC))
    loc_on_input = resample_from_to(nib.Nifti1Image((np.asarray(loc_img.dataobj) == 1).astype(np.uint8), loc_img.affine),
                                    (ct_img.shape, ct_img.affine), order=0)
    localizer = canonical(ct_img, np.asarray(loc_on_input.dataobj) > 0)

    # Crop box: the crop file's extent mapped to input-grid indices.
    crop_img = nib.load(str(CROP))
    corners = np.array([[i, j, k, 1] for i in (0, crop_img.shape[0] - 1) for j in (0, crop_img.shape[1] - 1)
                        for k in (0, crop_img.shape[2] - 1)])
    idx = np.rint((np.linalg.inv(ct_img.affine) @ crop_img.affine @ corners.T).T[:, :3]).astype(int)
    box = np.zeros(ct_img.shape, bool)
    lo, hi = idx.min(0), idx.max(0)
    box[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    box = canonical(ct_img, box)

    columns = [
        ("A  Manual reference", manual, GREEN, f"{manual.sum() * voxel_ml:.0f} mL"),
        ("B  Reference implementation,\nspleen-only mode", mask(REF_SO), RED, None),
        ("C  TS-Portable,\nspleen-only mode", mask(TSP_SO), RED, None),
        ("D  TS-Portable,\nfull mode", mask(TSP_FULL), RED, None),
    ]

    def dsc(a):
        return 2 * (a & manual).sum() / (a.sum() + manual.sum())

    coords = np.argwhere(manual)
    z = int(np.argmax(manual.sum(axis=(0, 1))))
    y = int(np.argmax(manual.sum(axis=(0, 2))))
    bx = np.argwhere(box)
    pad = 12
    x0, x1 = bx[:, 0].min() - pad, bx[:, 0].max() + pad
    y0, y1 = bx[:, 1].min() - pad, bx[:, 1].max() + pad
    z0, z1 = max(bx[:, 2].min() - 3, 0), min(bx[:, 2].max() + 3, ct.shape[2] - 1)
    zooms = nib.as_closest_canonical(ct_img).header.get_zooms()[:3]

    def axial(v):
        return np.fliplr(np.rot90(v[x0:x1, y0:y1, z]))

    def coronal(v):
        return np.fliplr(np.rot90(v[x0:x1, y, z0:z1]))

    views = [("Axial", axial, zooms[1] / zooms[0]), ("Coronal", coronal, zooms[2] / zooms[0])]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.4), gridspec_kw={"height_ratios": [1.0, 0.95]})
    for row, (view_name, view, aspect) in enumerate(views):
        image = view(ct)
        for col, (title, m, color, note) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=-160, vmax=240, aspect=aspect, interpolation="nearest")
            mv = view(m)
            fill = np.zeros((*mv.shape, 4))
            fill[mv] = (*matplotlib.colors.to_rgb(color), 0.45)
            ax.imshow(fill, aspect=aspect, interpolation="nearest")
            if col == 0:
                ax.contour(view(localizer), levels=[0.5], colors=CYAN, linewidths=1.4, linestyles="--")
                ax.contour(view(box), levels=[0.5], colors=YELLOW, linewidths=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
                vol = m.sum() * voxel_ml
                text = note if note else f"{vol:.1f} mL · DSC {dsc(m):.2f}" if vol < 10 else f"{vol:.0f} mL · DSC {dsc(m):.2f}"
                ax.text(0.5, -0.04, text, transform=ax.transAxes, ha="center", va="top", fontsize=11)
            if col == 0:
                ax.set_ylabel(view_name, fontsize=11.5)
    fig.legend(handles=[Patch(facecolor=GREEN, alpha=0.45, label="Manual spleen"),
                        Patch(facecolor=RED, alpha=0.45, label="Predicted spleen"),
                        Line2D([], [], color=CYAN, ls="--", lw=1.4, label="6-mm localizer spleen"),
                        Line2D([], [], color=YELLOW, lw=1.6, label="Spleen-only crop")],
               loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", OUT, "| axial z", z, "coronal y", y,
          "| volumes", {t.split()[1]: round(float(m.sum() * voxel_ml), 1) for t, m, _, _ in columns})


if __name__ == "__main__":
    main()
