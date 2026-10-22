"""Figure 1: flowchart of scan selection (diagram only, no data).

    python3 figures/fig1_flowchart.py [OUT_DIR]
"""
"""AJR Figure 1: selection of scans (flowchart). Writes fig1_flowchart.png and .pdf next to this file."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.set_xlim(0, 100); ax.set_ylim(15, 66); ax.axis("off")

def box(x, y, w, h, text, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", lw=0.9, ec="#222222", fc=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.35)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222"))

W = 23.5
def column(x, top, middle, bottom, side=None):
    ax_x = x + W * 0.3                      # arrows left of centre leave room for the side note
    box(x, 54, W, 10, top)
    box(x, 38, W, 9, middle)
    box(x, 18, W, 13, bottom, fill="#f2f2f2")
    arrow(ax_x, 54, ax_x, 47); arrow(ax_x, 38, ax_x, 31)
    if side:
        ax.plot([ax_x, ax_x + 2.5], [50.5, 50.5], color="#222222", lw=0.9)
        ax.text(ax_x + 3.0, 50.5, side, va="center", fontsize=7.5, linespacing=1.3)

column(1, "AMOS validation release\n120 scans",
       "External validation\ncohort, 100 CT scans\n(100 patients)",
       "Primary (1.5 mm) and\nsecondary (3 mm)\nanalyses, 15 organs;\nspleen volumetry;\nruntime",
       "Excluded:\nMRI, 20")
column(26, "Duke Spleen Data Set\n109 series",
       "Chronic liver disease\ncohort, 28 CT series",
       "Spleen volumetry on\nreading station from\nDICOM and NIfTI;\nruntime",
       "Excluded:\nMRI, 80; mask\nnot aligned, 1")
column(51, "TotalSegmentator\ndataset, small release\n102 CT scans",
       "Development set,\n102 CT scans (98 in\nTotalSegmentator\ntraining data)",
       "Secondary analyses\n(1.5 and 3 mm),\n117 labels;\nquantization\nfidelity audit")
column(76, "CT-ORG dataset\n140 CT scans",
       "Quantization set,\nfirst 28 CT scans",
       "Quantization only:\ncalibration, 8;\nselection, 10;\npatch testing, 10\n(no accuracy evaluation)",
       "Not used:\n112")

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fig1_flowchart"
fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
print("wrote", str(out) + ".png")
