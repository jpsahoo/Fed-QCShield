"""Shared plotting style: tab10/tab20 palette rule and selectable PDF text.

Two rules apply to every figure this project produces, in `plots.py` and in
`scripts/json_to_pdf.py` alike, so there is exactly one place either can drift:

1. **Colors come from `tab10`, or `tab20` once a plot needs more than 10
   categories.**  UNSW-NB15 has 10 classes (tab10 fits exactly); Edge-IIoTSet
   has 15 (needs tab20); the 12-model comparison bar chart also needs tab20.
   :func:`palette` makes that switch automatically so no plot is drawn with a
   fixed table.

2. **PDF text must stay selectable and searchable.**  Matplotlib's default PDF
   backend embeds Type 3 fonts, which several PDF viewers and `pdftotext`
   extract unreliably. Setting ``pdf.fonttype = 42`` embeds TrueType outlines
   instead, which every viewer treats as ordinary selectable text.
"""

from __future__ import annotations

from typing import List, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#: PGF backend rcParams: every figure is typeset by the system TeX engine
#: (see ``matplotlib.use("pgf")`` in plots.py), so glyphs in the saved PDF are
#: real embedded TeX-rendered text rather than matplotlib's own font
#: rasterization/outlining -- selectable, searchable, and font-matched to the
#: manuscript by construction, not just via a TrueType-embedding rcParam.
#: ``pgf.rcfonts: False`` stops matplotlib from fighting the TeX engine's own
#: font selection; ``pgf.texsystem`` picks the engine (xelatex, already
#: confirmed installed) that also compiles the manuscript itself.
PGF_RCPARAMS = {
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "text.usetex": False,        # pgf backend typesets text itself; usetex=True would double-compile
}

TAB10: List[str] = [matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab10").colors]
TAB20: List[str] = [matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]

# FIGSIZE_LARGE is used only by learning_curves(): panel title carries the
# template/dataset, no separate suptitle, no pinned box aspect.
FIGSIZE_LARGE = (32, 12)
DPI_MAIN = 250

#: rcParams for learning_curves() ONLY -- applied via a local rc_context in
#: plots.learning_curves(), never through apply_style()/STYLE, so changes to
#: the shared TITLE_SIZE/LABEL_SIZE/... constants (which
#: evaluation_metrics/model_comparison/etc. also use) cannot affect this plot.
#: TITLE_SIZE=26, LABEL_SIZE=24, TICK_SIZE=22, LEGEND_SIZE=22,
#: GRID_ALPHA=0.5, LINE_WIDTH=3, dashed grid, savefig.bbox="tight" (relied on
#: for this plot's exact page size; safe here since, unlike
#: evaluation_metrics, nothing in learning_curves does embed-width math
#: against the raw figsize).
LC_RCPARAMS = {
    "lines.linewidth": 3, "grid.linewidth": 1.0, "axes.linewidth": 1.5,
    "grid.alpha": 0.5, "grid.linestyle": "--", "axes.grid": True,
    "font.size": 22, "axes.titlesize": 26, "axes.labelsize": 24,
    "legend.fontsize": 22, "xtick.labelsize": 22, "ytick.labelsize": 22,
    "figure.autolayout": False, "savefig.bbox": "tight",
}

#: ``config["plot_size"]`` -- per-subplot size for the circuit gate-count bar
#: charts (plot_qcshield_distribution line 939, plot_qcshield_layer_distribution
#: line 981); actual figure width is this times the subplot count.
PLOT_SIZE = (15, 5)

#: federated client-vs-global panel: figsize=(20,6), dpi=300 -- a different
#: convention from the centralized figures.
FIGSIZE_FED = (20, 6)
DPI_FED = 300

#: Sized in the same spirit as FIGSIZE_FED.
FIGSIZE_ROUNDWISE = (26, 7.5)

#: LEGEND_SIZE is deliberately smaller than the other sizes: a uniformly
#: large legend font pushes multi-entry legends wide enough to overflow
#: their panel, while a comparatively small LEGEND_SIZE lets "Val (best 0.96
#: @ epoch 8)"-style entries fit inside a 100:60 panel without widening past
#: it.
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 13
ANNOT_SIZE = 15             # confusion-matrix cell annotation size (annot_kws={"size": 15})
GRID_ALPHA = 1.0
LINE_WIDTH = 2.2

#: Sized independently from FIGSIZE_LARGE: evaluation_metrics' confusion
#: matrix carries up to 15 rotated class-name tick labels (Edge-IIoTSet) and
#: its ROC/PR panels a matching per-class legend -- both need more width
#: than learning_curves' flattened panel to stay legible. The canvas is wide
#: so each of the 3 panels (confusion matrix, ROC, PR) prints at ~30% of the
#: manuscript's page width once embedded at 0.98\textwidth, with equal
#: width_ratios and minimal inter-panel/right margin (see
#: evaluation_metrics()) -- the panels are bound by available width there,
#: not height. EVAL_FONT_SCALE lets evaluation_metrics() scale its own
#: fonts/linewidths relative to the absolute point-sizes below (TITLE_SIZE
#: etc.) so the embedded, printed size stays correct at this canvas size.
EVAL_FIGSIZE = (30.5, 13.31)
EVAL_FONT_SCALE = EVAL_FIGSIZE[0] / 20.0
EVAL_DPI = DPI_MAIN

#: A4-conference rcParams (from pdfgn.py) plus the selectable-text setting.
STYLE = {
    **PGF_RCPARAMS,
    "pdf.fonttype": 42,          # TrueType, not Type 3 -- belt-and-braces if the pgf path is ever bypassed
    "ps.fonttype": 42,
    # Solid, light-gray grid (grid.color="0.75", linewidth=0.9, alpha=1.0)
    # instead of a darker dashed grid -- clearly visible but doesn't
    # visually compete with the plotted lines.
    "lines.linewidth": LINE_WIDTH, "grid.linewidth": 0.9, "axes.linewidth": 1.1,
    "grid.color": "0.75", "grid.alpha": GRID_ALPHA, "grid.linestyle": "-", "axes.grid": True,
    "font.size": TICK_SIZE, "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    # No "savefig.bbox": "tight" here deliberately: that option re-crops
    # every saved PDF to its content's bounding box, so the actual page
    # dimensions would never equal the `figsize` passed to plt.subplots(),
    # and how much a figure gets cropped varies unpredictably with its
    # content (box_aspect panels leaving reclaimable margin; an external
    # fig.legend() growing past the nominal canvas in the other direction).
    # Leaving this at the default (no crop) makes the saved PDF's page size
    # exactly the requested figsize, which every embed-width/native-width
    # scale-factor calculation in this module assumes.
    "figure.autolayout": False,
}


def apply_eval_style() -> None:
    """rcParams for evaluation_metrics and every bar chart -- identical to STYLE.

    Kept as a separate function from :func:`apply_style` only because callers
    already distinguish "eval style" from "line-plot style", not because the
    values actually differ.
    """
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=TAB10)


def apply_style() -> None:
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=TAB10)


def palette(n: int) -> List[str]:
    """``n`` distinct colors: tab10 while it fits, tab20 once it doesn't.

    ``n > 20`` cycles tab20 rather than repeating a single color n/20 times,
    since no plot in this project currently needs more than 20 series.
    """
    if n <= 10:
        return TAB10[:n]
    if n <= 20:
        return TAB20[:n]
    return [TAB20[i % 20] for i in range(n)]


def confusion_colormap():
    """Sequential colormap for confusion-matrix heatmaps, derived from tab10.

    tab10 is a *qualitative* palette -- its ten hues carry no inherent order,
    so using it directly as a continuous heatmap colormap would bin a smooth
    0-1 intensity into ten unordered colors and make the matrix unreadable.
    This derives a proper sequential colormap (white -> tab10's blue,
    ``TAB10[0]``) so the heatmap stays perceptually ordered while its single
    hue still comes from the tab10 palette, keeping every figure's coloring
    traceable to the same ten colors.
    """
    return LinearSegmentedColormap.from_list("tab10_blue_seq", ["#ffffff", TAB10[0]])
