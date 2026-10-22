#!/usr/bin/env python3
"""Figure 3: spleen and liver of one AMOS patient (amos_0318) at 1.5 mm, in the style of Fig. S3.

Needs the AMOS images and labels and both implementations' masks:

    AMOS_IMAGES=AMOS/imagesVa AMOS_LABELS=AMOS/labelsVa TSPORTABLE_MASKS=masks/tsportable \
    REFERENCE_MASKS=masks/reference python figures/fig3_spleen_liver_example.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
OUTLINE = [pe.withStroke(linewidth=2.2, foreground='black')]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.dpi": 100,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Data locations (environment variables). The AMOS images and labels are the public validation release;
# the masks are those written by TS-Portable and the reference implementation (runtime/run_bench.sh).
import os  # noqa: E402
EVAL = Path(__file__).resolve().parents[1]
def _need(name):
    if name not in os.environ:
        raise SystemExit(f"set {name} (see the docstring)")
    return Path(os.environ[name])
AMOS_IMAGES = _need("AMOS_IMAGES")          # imagesVa: amos_XXXX.nii.gz
AMOS_LABELS = _need("AMOS_LABELS")          # labelsVa: amos_XXXX.nii.gz
CANDIDATE = _need("TSPORTABLE_MASKS")      # full/ and fast/ subfolders, <case>.nii.gz
RESULTS = Path(os.environ.get("PERCASE_DIR", EVAL / "data" / "accuracy"))
OUT = Path(os.environ.get("OUT_DIR", EVAL / "results" / "figures"))
TS_OUTPUTS = {
    "full": _need("REFERENCE_MASKS") / "full",
    "fast": _need("REFERENCE_MASKS") / "fast",
}

GMAP = {
    key: int(value)
    for key, value in json.loads((EVAL / "data" / "total_label_map.json").read_text()).items()
}
AMOS = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "esophagus": 5,
    "liver": 6,
    "stomach": 7,
    "aorta": 8,
    "inferior_vena_cava": 9,
    "pancreas": 10,
    "adrenal_gland_right": 11,
    "adrenal_gland_left": 12,
    "duodenum": 13,
    "urinary_bladder": 14,
    "prostate": 15,
}

FIGURES = {
    "full": {
        "panel": "",
        "resolution": "1.5 mm",
        "output": "fig3_spleen_liver_example",
        "rows": [
            ("amos_0318", "spleen", "Spleen"),
            ("amos_0318", "liver", "Liver"),
        ],
    },
}

COLS = ["Reference standard", "Reference implementation", "TS-Portable"]
GREEN = "#2a9d5c"
RED = "#d1495b"


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.as_closest_canonical(nib.load(str(path)))


def bbox(mask, pad):
    ys, xs = np.where(mask)
    return (
        max(0, ys.min() - pad),
        min(mask.shape[0], ys.max() + pad),
        max(0, xs.min() - pad),
        min(mask.shape[1], xs.max() + pad),
    )


def nsd_lookup(tier):
    records = json.loads((RESULTS / f"percase_amos_{tier}.json").read_text())
    return {
        (case["case"], organ["organ"]):
        (float(organ["nsd3_stock_ts"]), float(organ["nsd3_int8"]))
        for case in records
        for organ in case["organs"]
    }


def render(tier, specification):
    rows = specification["rows"]
    scores = nsd_lookup(tier)
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.4, 2.75 * len(rows)))

    for row, (case, organ, display_name) in enumerate(rows):
        ct_image = load(AMOS_IMAGES / f"{case}.nii.gz")
        ct = np.asarray(ct_image.dataobj).astype(np.float32)
        spacing = np.sqrt((ct_image.affine[:3, :2] ** 2).sum(0))
        gt = np.asarray(load(AMOS_LABELS / f"{case}.nii.gz").dataobj)
        ts_mask = np.asarray(load(TS_OUTPUTS[tier] / f"{case}.nii.gz").dataobj)
        int8 = np.asarray(load(CANDIDATE / tier / f"{case}.nii.gz").dataobj)
        if not (ct.shape == gt.shape == ts_mask.shape == int8.shape):
            raise ValueError(f"shape mismatch for {case}/{tier}")

        amos_id = AMOS[organ]
        total_id = GMAP[organ]
        manual_mask = gt == amos_id
        z = int(np.argmax(manual_mask.sum(axis=(0, 1))))
        y0, y1, x0, x1 = bbox(manual_mask[:, :, z], 35)
        height, width = manual_mask[:, :, z].shape
        side = min(max(y1 - y0, x1 - x0), height, width)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0 = int(np.clip(cy - side // 2, 0, height - side))
        y1 = y0 + side
        x0 = int(np.clip(cx - side // 2, 0, width - side))
        x1 = x0 + side

        def axial(volume):
            # RAS-canonical arrays are flipped horizontally after rotation so
            # patient right appears on image left (radiological convention).
            return np.fliplr(np.rot90(volume[y0:y1, x0:x1, z]))

        ts_nsd, int8_nsd = scores[(case, organ)]
        panels = [
            (manual_mask, GREEN, None),
            (ts_mask == total_id, RED, f"NSD@3 mm = {ts_nsd:.4f}"),
            (int8 == total_id, RED, f"NSD@3 mm = {int8_nsd:.4f}"),
        ]
        ct_window = axial(ct)

        for column, (axis, (mask, color, note)) in enumerate(zip(axes[row], panels)):
            axis.imshow(ct_window, cmap="gray", vmin=-160, vmax=240, aspect="equal")
            mask_window = axial(mask)
            if mask_window.any():
                fill = np.zeros((*mask_window.shape, 4))
                fill[mask_window] = (*matplotlib.colors.to_rgb(color), 0.25)
                axis.imshow(fill, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#999")
                spine.set_linewidth(0.7)
            if row == 0:
                axis.set_title(COLS[column], fontsize=12, fontweight="bold", pad=6)
            if note:
                axis.text(
                    0.5,
                    -0.045,
                    note,
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=10.5,
                    color="#222",
                )
            if column == 0:
                axis.set_ylabel(display_name, fontsize=11)
                axis.text(0.035, 0.16, "R", transform=axis.transAxes, color="white", fontsize=8.5,
                          ha="left", va="bottom", fontweight="bold", path_effects=OUTLINE)
                axis.text(0.965, 0.16, "L", transform=axis.transAxes, color="white", fontsize=8.5,
                          ha="right", va="bottom", fontweight="bold", path_effects=OUTLINE)
                pixels = 20.0 / float(spacing[1])
                image_height, image_width = ct_window.shape
                axis.plot(
                    [0.06 * image_width, 0.06 * image_width + pixels],
                    [0.94 * image_height, 0.94 * image_height],
                    "w-",
                    linewidth=3,
                    solid_capstyle="butt",
                    path_effects=[pe.Stroke(linewidth=5, foreground="black"), pe.Normal()],
                )
                axis.text(
                    0.06 * image_width + pixels / 2,
                    0.905 * image_height,
                    "2 cm",
                    color="white",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    path_effects=OUTLINE,
                )

    fig.text(
        0.006, 0.995, specification["panel"], ha="left", va="top",
        fontsize=15, fontweight="bold", color="black",
    )
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, alpha=0.25, label="Reference standard"),
            Patch(facecolor=RED, alpha=0.25, label="Prediction"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.045),
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.subplots_adjust(hspace=0.22, wspace=0.06)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = OUT / f"{specification['output']}.{extension}"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {destination}")
    plt.close(fig)


for selected_tier, selected_figure in FIGURES.items():
    render(selected_tier, selected_figure)
