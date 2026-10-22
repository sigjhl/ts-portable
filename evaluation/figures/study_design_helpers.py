#!/usr/bin/env python3
"""Drawing helpers (boxes, arrows, colors) for figures/fig2_study_design.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle



INK = "#171717"
BAND = "#e9e9e9"          # monochrome stage bands
WHITE = "#ffffff"


def box(ax, x, y, w, h, text, *, fill=WHITE, rounded=False,
        fontsize=11.0, weight="normal", lw=1.25, zorder=3):
    """Draw a centered labeled box in data coordinates."""
    kwargs = dict(facecolor=fill, edgecolor=INK, linewidth=lw, zorder=zorder)
    if rounded:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.025,rounding_size=0.10", **kwargs,
        )
    else:
        patch = Rectangle((x, y), w, h, **kwargs)
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", color=INK,
        fontsize=fontsize, fontweight=weight, linespacing=1.3,
        zorder=zorder + 1,
    )
    return patch


def titled_box(ax, x, y, w, h, title, body, *, fill=WHITE,
               title_fs=11.5, body_fs=10.5, lw=1.25, zorder=3,
               rounded=False):
    """Box with a bold title line above a normal-weight body."""
    kwargs = dict(facecolor=fill, edgecolor=INK, linewidth=lw, zorder=zorder)
    if rounded:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.025,rounding_size=0.10", **kwargs,
        )
    else:
        patch = Rectangle((x, y), w, h, **kwargs)
    ax.add_patch(patch)
    cx = x + w / 2
    ax.text(cx, y + h * 0.72, title, ha="center", va="center", color=INK,
            fontsize=title_fs, fontweight="bold", zorder=zorder + 1)
    ax.text(cx, y + h * 0.34, body, ha="center", va="center", color=INK,
            fontsize=body_fs, linespacing=1.3, zorder=zorder + 1)


def arrow(ax, start, end, *, lw=1.4, mutation_scale=13, zorder=2):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", color=INK, linewidth=lw,
        mutation_scale=mutation_scale, connectionstyle="arc3",
        shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(a)
    return a


def line(ax, xs, ys, *, lw=1.4, zorder=1):
    ax.plot(xs, ys, color=INK, linewidth=lw, solid_capstyle="butt", zorder=zorder)


def stage_label(ax, y, h, title):
    ax.add_patch(Rectangle(
        (0.18, y), 0.80, h,
        facecolor=BAND, edgecolor=INK, linewidth=1.2, zorder=3,
    ))
    ax.text(
        0.58, y + h / 2, title,
        rotation=90, ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=INK, zorder=4,
    )
