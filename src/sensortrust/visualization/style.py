"""Figure style.

Scientific figures must remain interpretable in grayscale print and for
colour-vision deficiencies.  Every series therefore receives, in a fixed
order, a colour **and** a line style **and** a marker; every figure with two
or more series carries a legend.  The categorical palette is a validated
colour-vision-deficiency-safe ordering (adjacent-pair CVD Delta E >= 8);
sequential maps use a single blue hue from light to dark; fault periods are
drawn as a neutral grey hatched band (texture, not colour).
"""

from __future__ import annotations

import itertools

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FAULT_FACE = "#d9d8d2"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("sensortrust_blue", BLUE_RAMP)

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": GRID,
    "lines.linewidth": 1.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
    "figure.dpi": 100,
}


def apply_style() -> None:
    mpl.rcParams.update(RC)


def series_style(i: int) -> dict:
    """Colour + line style + marker of the ``i``-th series (fixed order)."""
    return {"color": PALETTE[i % len(PALETTE)], "linestyle": LINESTYLES[i % len(LINESTYLES)],
            "marker": MARKERS[i % len(MARKERS)]}


def cycle_styles():
    return (series_style(i) for i in itertools.count())
