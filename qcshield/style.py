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

# --------------------------------------------------------------------------
# 2026-11-15: re-derived from the ACTUAL print target, not kept at the
# mega-cell's on-screen-only values, and now computed by the same formula as
# the user-supplied reference ``CPerfVisualizer`` rather than a hand-picked
# tuple. The previous (32,12) canvas looked fine rendered standalone, but
# every embedding in the manuscript places this 3-panel image at well under
# its native width -- 0.79-0.85\textwidth (~5.5-6.3in). A vector PDF scales
# everything, text included, by (embed width / native width) when LaTeX
# resizes it: at the old 32in-wide canvas that factor is ~5.9/32 = 0.18, so a
# 26pt title prints at under 5pt -- invisible on actual A4 paper, even though
# it looks fine in an on-screen raster preview rendered at whatever pixel
# density the viewer asks for (a raster check does not catch this; only the
# embed-width/native-width ratio does).
#
# A smaller native width alone is not sufficient, though -- the first fix
# only shrank FIGSIZE_LARGE and left matplotlib's own `tight_layout()` to
# decide each panel's realized width:height ratio, which is NOT the same as
# the ratio implied by the raw figsize once title/label/legend margins are
# subtracted, and did not reliably land on a wide rectangle. The reference
# script instead derives the figure size FROM an explicit per-panel aspect
# ratio and then pins it with `ax.set_box_aspect()` in `plots.py`, which is
# what actually guarantees the panel shape regardless of layout margins:
#   PANEL_ASPECT_WH = 100/60      (width:height a panel should have)
#   FIG_W, H_PAD, MARGIN          (same names/values as the reference)
#   panel_w = FIG_W * MARGIN / 3; panel_h = panel_w / PANEL_ASPECT_WH
#   FIGSIZE_LARGE = (FIG_W, panel_h + H_PAD)
# `plots.learning_curves()` then calls `ax.set_box_aspect(1/PANEL_ASPECT_WH)`
# on every panel so the three line-plot panels there always land on the
# intended wide, short rectangle. The confusion matrix (a square grid) and
# ROC/PR curves (conventionally square axes) in `evaluation_metrics()`
# deliberately do NOT use this box aspect -- only accuracy/F1/loss are
# epoch-vs-metric line plots the "wide rectangle" convention is meant for.
# --------------------------------------------------------------------------
# 2026-11-17: the wide-rectangle derivation above (FIG_W/PANEL_ASPECT_WH/
# H_PAD/MARGIN) was reverted. The user asked for learning_curves() restored
# EXACTLY to its Revision R1 appearance (reference PDFs: panel title carries
# the template/dataset, no separate suptitle, no pinned box aspect) and
# locked there going forward, independent of whatever evaluation_metrics'
# own styling does later. FIGSIZE_LARGE is used only by learning_curves(), so
# it is restored to R1's literal (32, 12)/250 dpi rather than re-derived.
FIGSIZE_LARGE = (32, 12)
DPI_MAIN = 250

#: Locked R1 rcParams for learning_curves() ONLY -- applied via a local
#: rc_context in plots.learning_curves(), never through apply_style()/STYLE,
#: so later changes to the shared TITLE_SIZE/LABEL_SIZE/... constants (which
#: evaluation_metrics/model_comparison/etc. also use) cannot drift this plot
#: away from R1 again. Values are the exact ones git history shows R1's
#: plots.py/style.py used (commit 3ecb730, before this session's layout
#: fixes): TITLE_SIZE=26, LABEL_SIZE=24, TICK_SIZE=22, LEGEND_SIZE=22,
#: GRID_ALPHA=0.5, LINE_WIDTH=3, dashed grid, savefig.bbox="tight" (R1 relied
#: on this crop for its exact page size; safe here since, unlike
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

#: federated notebook's plot_client_global_metrics: figsize=(20,6), dpi=300 --
#: a different convention from the main notebook, preserved as-is since that
#: is what the original code did (this plotter is not shown in the pasted
#: reference above, but was confirmed against it earlier in this project).
FIGSIZE_FED = (20, 6)
DPI_FED = 300

#: roundwise_scores has no original-code equivalent (the manuscript
#: references it but the notebook never had a generator for it); sized in
#: the same spirit as FIGSIZE_FED rather than independently invented.
FIGSIZE_ROUNDWISE = (26, 7.5)

#: 2026-11-15: realigned to the exact values from the user-supplied
#: reference (``CPerfVisualizer``/``PlotConfig``), after an earlier round of
#: independently-chosen, larger values (28/26/24/22) turned out to make
#: things worse, not better, once ``savefig.bbox="tight"`` was removed
#: (see that rcParam's own history note above): a uniformly large legend
#: font is what was pushing multi-entry legends wide enough to overflow
#: their panel in the first place. The reference deliberately differentiates
#: -- a comparatively small LEGEND_SIZE is what lets "Val (best 0.96 @
#: epoch 8)"-style entries fit inside a 100:60 panel without widening past
#: it -- so matching it exactly, rather than scaling everything up
#: uniformly, is itself part of the fix.
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 13
ANNOT_SIZE = 15             # confusion-matrix cell annotation size (annot_kws={"size": 15})
GRID_ALPHA = 1.0
LINE_WIDTH = 2.2

#: Decoupled from FIGSIZE_LARGE 2026-11-15. The reference code shared one
#: figsize between the two, which was fine when both plots only ever had
#: short titles and 2-3-entry legends. evaluation_metrics' confusion matrix
#: carries up to 15 rotated class-name tick labels (Edge-IIoTSet) and its
#: ROC/PR panels a matching per-class legend -- both need more width than
#: learning_curves' flattened panel to stay legible, so this is now sized
#: independently rather than inheriting a value tuned for a different plot.
#: 2026-11-16: widened further so each of the 3 panels (confusion matrix,
#: ROC, PR) prints at ~30% of the manuscript's page width once embedded at
#: 0.98\textwidth, with equal width_ratios and minimal inter-panel/right
#: margin (see evaluation_metrics()) -- at the old (20, 7.0) canvas the same
#: layout only reached ~14% per panel, since the panels are bound by
#: available *height* there, not width. The absolute point-sizes below
#: (TITLE_SIZE etc.) were tuned for THIS function at the old 20in-wide
#: canvas; EVAL_FONT_SCALE lets evaluation_metrics() scale its own fonts/
#: linewidths back up so the embedded, printed size is unchanged.
EVAL_FIGSIZE = (30.5, 13.31)
EVAL_FONT_SCALE = EVAL_FIGSIZE[0] / 20.0
EVAL_DPI = DPI_MAIN

#: A4-conference rcParams (from pdfgn.py) plus the selectable-text setting.
STYLE = {
    **PGF_RCPARAMS,
    "pdf.fonttype": 42,          # TrueType, not Type 3 -- belt-and-braces if the pgf path is ever bypassed
    "ps.fonttype": 42,
    # Solid, light-gray grid (reference: grid.color="0.75", linewidth=0.9,
    # alpha=1.0) instead of the previous black dashed grid at alpha=0.5 --
    # the reference's grid is clearly visible but doesn't visually compete
    # with the plotted lines the way a darker dashed one does.
    "lines.linewidth": LINE_WIDTH, "grid.linewidth": 0.9, "axes.linewidth": 1.1,
    "grid.color": "0.75", "grid.alpha": GRID_ALPHA, "grid.linestyle": "-", "axes.grid": True,
    "font.size": TICK_SIZE, "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    # "savefig.bbox": "tight" (removed 2026-11-15) silently re-cropped every
    # saved PDF to its content's bounding box, so the actual page dimensions
    # never equaled the `figsize` passed to plt.subplots() -- every
    # embed-width/native-width scale factor computed this session (the whole
    # strategy behind FIGSIZE_LARGE, EVAL_FIGSIZE, PANEL_BOX_ASPECT, etc.)
    # was built on a canvas size that was not what actually got saved, and
    # how much each figure got cropped varied unpredictably with its content
    # (box_aspect panels leaving reclaimable margin; an external fig.legend()
    # growing past the nominal canvas in the other direction). Leaving this
    # at the default (no crop) makes the saved PDF's page size exactly the
    # requested figsize, which is what every scale-factor calculation in this
    # module actually assumes.
    "figure.autolayout": False,
}


def apply_eval_style() -> None:
    """rcParams for evaluation_metrics and every bar chart -- identical to STYLE.

    The reference code's ``plot_evaluation_metrics`` uses the same
    ``config["figure_size_large"]``/``config["dpi"]`` and the same font-size
    kwargs as ``plot_learning_curves`` -- kept as a separate function only
    because callers already distinguish "eval style" from "line-plot style",
    not because the values actually differ.
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
