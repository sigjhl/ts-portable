#!/usr/bin/env python3
"""Figure 2: deployment and study design (diagram only, no data).

    python3 figures/fig2_study_design.py [OUT_DIR]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_design_helpers import box, titled_box, arrow, line, INK, BAND, WHITE  # noqa: E402

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "fig2_study_design"


def stage(ax, y, h, title):
    ax.add_patch(Rectangle((0.18, y), 0.80, h, facecolor=BAND, edgecolor=INK, linewidth=1.2, zorder=3))
    ax.text(0.58, y + h / 2, title, rotation=90, ha="center", va="center", color=INK,
            fontsize=11.0, fontweight="bold", linespacing=1.1, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(11.0, 16.2), dpi=220, facecolor=WHITE)
    ax.set_xlim(0, 10.6); ax.set_aspect("auto"); ax.axis("off")

    L, R = 1.5, 9.4          # content left/right
    LC, RC = 1.5, 5.75       # two-column left edges
    CW = 3.65                # column width
    LX, RX = LC + CW / 2, RC + CW / 2

    # Vertical layout: every inter-stage gap is G; within-stage gaps are fixed; all y's derive from the top.
    G = 0.7
    model_h, export_h = 0.7, 0.95
    deployment_h = 1.55
    cohort_h, resolution_h = 1.3, 0.55
    analysis_h, supporting_h = 2.0, 2.0
    banner_h, spleen_h = 0.6, 2.7
    top = 12.7
    model_y = top - model_h
    export_y = model_y - 0.45 - export_h
    deployment_y = export_y - G - deployment_h
    cohort_y = deployment_y - G - cohort_h
    resolution_y = cohort_y - 0.2 - resolution_h
    analysis_y = resolution_y - G - analysis_h
    supporting_y = analysis_y - 0.35 - supporting_h
    banner_y = supporting_y - G - banner_h
    spleen_y = banner_y - 0.3 - spleen_h
    ax.set_ylim(spleen_y - 0.3, top + 0.3)

    # ------------------------------------------------------------ Model preparation
    stage(ax, export_y, top - export_y, "Model\npreparation")
    box(ax, L, model_y, R - L, model_h, "TotalSegmentator models (fp32)", rounded=True, weight="bold")
    box(ax, L, export_y, 4.6, export_h, "ONNX export and\naccuracy-controlled INT8 quantization", rounded=True, weight="bold", fontsize=11.0)
    titled_box(ax, 6.7, export_y, 2.7, export_h, "CT-ORG quantization set", "Calibration 0–7 · selection 8–17\nHeld-out patch test 18–27", rounded=True, title_fs=10.5, body_fs=9.0)
    arrow(ax, (LX + 0.55, model_y), (LX + 0.55, export_y + export_h))
    arrow(ax, (6.7, export_y + export_h / 2), (6.1, export_y + export_h / 2))

    # ------------------------------------------------------------ Deployment
    stage(ax, deployment_y, deployment_h, "Deployment")
    arrow(ax, (LX + 0.55, export_y), (LX + 0.55, deployment_y + deployment_h))
    titled_box(ax, L, deployment_y, R - L, deployment_h, "TS-Portable: self-contained, on the CPU",
               "ONNX Runtime with OpenVINO; no Python, no GPU\nDICOM or NIfTI in, NIfTI mask and volume table out\nFull 117-label mode or spleen-only mode", rounded=True, title_fs=11.5, body_fs=10.2)

    # ------------------------------------------------------------ Evaluation cohorts
    stage(ax, resolution_y, cohort_y + cohort_h - resolution_y, "Evaluation\ncohorts")
    arrow(ax, (LX, deployment_y), (LX, cohort_y + cohort_h)); arrow(ax, (RX, deployment_y), (RX, cohort_y + cohort_h))
    titled_box(ax, LC, cohort_y, CW, cohort_h, "Development set", "n = 102 · 117 structures\nTraining-overlap fidelity audit", title_fs=11.5, body_fs=10.5)
    titled_box(ax, RC, cohort_y, CW, cohort_h, "External AMOS set", "n = 100 · 15 organs", title_fs=11.5, body_fs=10.5)
    box(ax, LC, resolution_y, CW, resolution_h, "1.5-mm and 3-mm models", fontsize=10.2)
    box(ax, RC, resolution_y, CW, resolution_h, "1.5-mm and 3-mm models", fontsize=10.2)
    line(ax, [LX, LX], [cohort_y, resolution_y + resolution_h]); line(ax, [RX, RX], [cohort_y, resolution_y + resolution_h])

    # ------------------------------------------------------------ Analysis
    stage(ax, supporting_y, analysis_y + analysis_h - supporting_y, "Noninferiority\nanalysis")
    arrow(ax, (LX, resolution_y), (LX, analysis_y + analysis_h)); arrow(ax, (RX, resolution_y), (RX, analysis_y + analysis_h))
    titled_box(ax, LC, analysis_y, CW, analysis_h, "Secondary analyses\n1.5 mm and 3 mm",
               "TS-Portable vs reference implementation\nEndpoints: mean DSC and NSD@3 mm\nNoninferiority margin, 0.01, per endpoint", title_fs=11.0, body_fs=9.6)
    titled_box(ax, RC, analysis_y, CW, analysis_h, "Primary analysis · 1.5 mm\nSecondary analysis · 3 mm",
               "TS-Portable vs reference implementation\nEndpoints: mean DSC and NSD@3 mm\nMargin 0.01; primary requires both", title_fs=11.0, body_fs=9.6)
    arrow(ax, (LX, analysis_y), (LX, supporting_y + supporting_h)); arrow(ax, (RX, analysis_y), (RX, supporting_y + supporting_h))
    titled_box(ax, LC, supporting_y, CW, supporting_h, "Descriptive analyses",
               "NSD at 1 and 2 mm\nPer-structure values, volume strata\nEmpty predictions", title_fs=11.0, body_fs=9.8)
    titled_box(ax, RC, supporting_y, CW, supporting_h, "Descriptive analyses",
               "NSD at 1 and 2 mm\nPer-organ values, volume strata\nEmpty predictions\nRuntime and memory, five arms", title_fs=11.0, body_fs=9.8)

    # ------------------------------------------------------------ Spleen volumetry analysis
    stage(ax, spleen_y, banner_y + banner_h - spleen_y, "Spleen\nvolumetry")
    arrow(ax, (RX, supporting_y), (RX, banner_y + banner_h))
    box(ax, L, banner_y, R - L, banner_h, "PACS reading station (six-core CPU) · spleen-only mode", rounded=True, weight="bold", fontsize=11.0)
    arrow(ax, (LX, banner_y), (LX, spleen_y + spleen_h)); arrow(ax, (RX, banner_y), (RX, spleen_y + spleen_h))
    titled_box(ax, LC, spleen_y, CW, spleen_h, "Duke set · chronic liver disease",
               "n = 28 CT series · DICOM and NIfTI input\n"
               "Volume vs manual reference and\n"
               "reference implementation (workstation)\n"
               "Lin's CCC, Bland–Altman · runtime", title_fs=11.0, body_fs=9.6)
    titled_box(ax, RC, spleen_y, CW, spleen_h, "External AMOS set",
               "n = 100 · NIfTI input\n"
               "Spleen DSC and NSD@3 mm: noninferiority,\n"
               "margin 0.01, after the primary analysis\n"
               "Volume vs manual reference and\n"
               "reference implementation\n"
               "Lin's CCC, Bland–Altman · runtime", title_fs=11.0, body_fs=9.6)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, facecolor=WHITE, bbox_inches="tight", pad_inches=0.25)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
