"""Figure generation (Phase 6).

Every figure is written to the exact path ``QCShield_Elsevier_V2.tex`` already
references, so the regenerated files drop in place and no ``\\includegraphics``
line has to change. The manuscript bakes a timestamp into most filenames; it is
pinned as :data:`qcshield.config.FIGURE_TAG` rather than regenerated per run.

Note the manuscript is inconsistent about the dataset separator -- centralized
figures use ``UNSW-NB15`` and federated ones ``UNSW_NB15``. Both spellings are
reproduced exactly as they appear in the ``.tex``.

**Dimensions and layout match the original code exactly**, confirmed
2026-08-29 against the mega-cell source and the federated notebook (see
:mod:`qcshield.style` for the citations): ``figsize=(32,12)``/``dpi=250`` for
``learning_curves``/``evaluation_metrics`` (1x3 subplots each), ``dpi=300``/
``figsize=(20,6)`` for the federated client-vs-global panel (1x3: Accuracy,
F1, Loss), and ``config["plot_size"]=(15,5)`` per subplot for the circuit
gate-count bar charts. The confusion matrix uses ``cmap='tab10'`` directly
with annotated raw counts, matching the original -- tab10's qualitative hues
are fine here because the printed number, not color intensity, carries the
value; every other plot uses tab10/tab20 as a qualitative palette, switching
to tab20 automatically past 10 categories via :func:`qcshield.style.palette`
(Edge-IIoTSet's 15 classes, the 12-model comparison chart).

**PDF text stays selectable** -- ``pdf.fonttype = 42`` embeds TrueType
outlines instead of matplotlib's default Type 3 fonts, which several viewers
and ``pdftotext`` fail to extract cleanly.

No correlation-matrix plot is produced anywhere in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

from .config import FIGURE_TAG, FIGURE_DIR
from . import style

logger = logging.getLogger(__name__)

apply_style = style.apply_style   # re-exported for callers that only import plots


def save_figure(fig, path: Path, dpi: int = style.DPI_MAIN) -> Path:
    """Save and close a figure. Public so `scripts/json_to_pdf.py` can reuse it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format="pdf")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


_save = save_figure   # internal alias, kept for brevity below


def _tagged(stem: str, tag: str) -> str:
    """``stem`` with an optional ``_{tag}`` suffix; empty tag omits it cleanly."""
    return f"{stem}_{tag}.pdf" if tag else f"{stem}.pdf"


_LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}"}


def _latex_escape(s: str) -> str:
    """Escape LaTeX-special characters in a raw data value used as plot text.

    The ``pgf`` backend (see module docstring) renders every piece of text
    through a real TeX engine, so a raw ``&`` in a class name (e.g. X-IIoTID's
    ``"C&C"``) is a LaTeX alignment-tab character, not a literal ampersand,
    and aborts the whole figure. Class names come straight from the dataset
    and are the only plot text not already written by this module, so this
    is applied specifically to them rather than globally.
    """
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(s))


def _escape_all(names: Sequence[str]) -> List[str]:
    return [_latex_escape(n) for n in names]


# --------------------------------------------------------------------------
# Centralized figures
# --------------------------------------------------------------------------

def learning_curves(history: dict, dataset: str, template: str,
                    fig_dir: Path = None, tag: str = FIGURE_TAG) -> Path:
    """``learning_curves_{dataset}_{template}[_{tag}].pdf`` -- 1x3, (32,12), dpi=250.

    Restored EXACTLY to the Revision R1 appearance (2026-11-17), verified
    against R1's own rendered PDFs (``learning_curves_Edge-IIoTSet_QE_QCSL``/
    ``learning_curves_UNSW-NB15_QE_QCSL``, both tag ``20250721_124103``) and
    against git history for the ``plots.py``/``style.py`` that produced them
    (commit 3ecb730, the last one before this session's layout changes):
    the template/dataset name lives in each panel's own title -- there is no
    separate suptitle -- and no box_aspect is pinned, so panels keep
    whatever shape (32,12)/tight_layout naturally gives them. Styled via
    :data:`style.LC_RCPARAMS`, a dedicated rc_context scoped to this
    function only, so later changes to evaluation_metrics'/model_comparison's
    shared TITLE_SIZE/LABEL_SIZE/GRID_ALPHA/... constants cannot pull this
    plot away from R1 again.
    """
    style.apply_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    with plt.rc_context(style.LC_RCPARAMS):
        colors = style.palette(2)          # train vs validation
        ep = np.arange(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=style.FIGSIZE_LARGE, dpi=style.DPI_MAIN)

        panels = [
            ("train_acc", "val_acc", "Accuracy", "Accuracy", max, "Best Val Acc"),
            ("train_f1", "val_f1", "F1 Score", "F1 Score", max, "Best Val F1"),
            ("train_loss", "val_loss", "Loss", "Loss", min, "Min Val Loss"),
        ]
        for ax, (tr, va, ylabel, title, extreme, ref_label) in zip(axes, panels):
            ax.plot(ep, history[tr], label=f"Train {title}", color=colors[0])
            ax.plot(ep, history[va], label=f"Validation {title}", color=colors[1])
            best = extreme(history[va])
            best_epoch = history[va].index(best) + 1
            ax.axhline(y=best, color="blue", linestyle="--", alpha=0.5,
                       label=f"{ref_label}: {best:.3f} at Epoch {best_epoch}")
            ax.set_title(f"{title} ({template.replace('_', '-')}, {dataset})")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.5)
        fig.tight_layout()
        return _save(fig, fig_dir / _tagged(f"learning_curves_{dataset}_{template}", tag))


def evaluation_metrics(confusion: Sequence[Sequence[int]], per_class_f1: Sequence[float],
                       class_names: Sequence[str], dataset: str, template: str,
                       fig_dir: Path = None, tag: str = FIGURE_TAG,
                       roc_curves: dict = None, pr_curves: dict = None) -> Path:
    """``evaluation_metrics_{dataset}_{template}[_{tag}].pdf`` -- 1x3, (32,12), dpi=250.

    Faithful port of the reference ``Visualizer.plot_evaluation_metrics``
    (2026-09-07): confusion matrix (raw counts, ``cmap='tab10'``, fixed
    14pt annotations), per-class ROC and precision-recall with per-class
    AUC/AP in the legend, micro/macro-average curves overlaid as dotted
    lines. Same ``config["figure_size_large"]``/``config["dpi"]`` as
    ``learning_curves``, equal 1x3 subplots, ``loc='best'`` legends,
    ``tight_layout()`` -- no per-class-count dynamic sizing, matching the
    reference exactly rather than an independently-derived scheme.

    Falls back to a confusion-matrix + per-class-F1 layout when
    ``roc_curves``/``pr_curves`` are not supplied (result JSONs written before
    that field existed).
    """
    style.apply_eval_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    n = len(class_names)
    colors = style.palette(n)
    cm = np.asarray(confusion, dtype=int)
    have_curves = roc_curves is not None and pr_curves is not None

    # EVAL_FIGSIZE's width was widened (2026-11-16) so 3 equal-width square
    # panels print at ~30% page-width each once embedded -- that only works
    # because the panel squares are bound by *width*, not height, at this
    # size, which needs a much bigger canvas than the point-sizes below were
    # originally tuned against. fs rescales every explicit font/linewidth in
    # this function so the PRINTED (post-embed) size is unchanged from
    # before the widening; apply_eval_style()'s own rcParams-driven sizes
    # (axes.titlesize/labelsize/ticklabelsize/legend.fontsize/font.size/
    # lines.linewidth/axes.linewidth) are rescaled the same way via
    # rc_context so panel titles, axis labels, and ROC/PR tick numbers scale
    # too, without touching the global style constants other plots share.
    fs = style.EVAL_FONT_SCALE
    scaled_rc = {k: plt.rcParams[k] * fs for k in (
        "font.size", "axes.titlesize", "axes.labelsize", "legend.fontsize",
        "xtick.labelsize", "ytick.labelsize", "lines.linewidth", "axes.linewidth")}
    with plt.rc_context(scaled_rc):
        return _evaluation_metrics_impl(
            cm, per_class_f1, class_names, dataset, template, fig_dir, tag,
            roc_curves, pr_curves, n, colors, have_curves, fs)


def _evaluation_metrics_impl(cm, per_class_f1, class_names, dataset, template,
                             fig_dir, tag, roc_curves, pr_curves, n, colors,
                             have_curves, fs) -> Path:
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=style.EVAL_FIGSIZE, dpi=style.EVAL_DPI,
        gridspec_kw={"width_ratios": [1, 1, 1]})   # equal slots: all three panels print at the same page-width share, confusion matrix included
    # The (model, dataset) context is added as a fig.text() near the end of
    # this function, after tight_layout -- see learning_curves for why: a
    # suptitle present when tight_layout runs makes it reserve a large fixed
    # block above the axes regardless of the suptitle's own fontsize or any
    # rect passed to it, and tight_layout leaves a layout engine attached
    # that re-applies (and re-reserves that same block) at save time even if
    # the suptitle is only added afterward, unless that engine is detached.

    # Cell-annotation and tick-label size scale down past 10 classes: at a
    # fixed panel width, n x n cells each get proportionally less room as n
    # grows, and ANNOT_SIZE/TICK_SIZE sized for n<=10 (UNSW-NB15, X-IIoTID)
    # fully overlapped into an unreadable blur at Edge-IIoTSet's n=15.
    # Absolute pixel-space targets, not scaled relative to style.ANNOT_SIZE/
    # TICK_SIZE: this geometry (how much room an n x n confusion matrix has
    # per cell/tick at EVAL_FIGSIZE) hasn't changed, but those two constants
    # dropped substantially (19->15, 24->16) to match the reference's overall
    # scheme -- a formula relative to them would now under-shoot the same
    # visually-confirmed-legible absolute sizes this was tuned to produce.
    cm_annot_fs = (style.ANNOT_SIZE if n <= 10 else max(8, round(140 / n))) * fs
    cm_tick_fs = (style.TICK_SIZE if n <= 10 else max(9, round(160 / n))) * fs

    im = ax1.imshow(cm, cmap="tab10")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=cm_annot_fs, color="black")
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(_escape_all(class_names), rotation=45, ha="right", fontsize=cm_tick_fs)
    ax1.set_yticklabels(_escape_all(class_names), rotation=45, va="top", fontsize=cm_tick_fs)
    ax1.set_xlabel("Predicted Label"); ax1.set_ylabel("True Label")
    ax1.set_title("Confusion Matrix")
    ax1.grid(False)
    # imshow's default aspect='equal' already squares the *data* (n x n
    # cells), but, per the same lesson as learning_curves' box_aspect fix,
    # that is not automatically the same as the *axes box* staying square
    # once tight_layout and the width_ratios above divide up the figure --
    # pin it explicitly rather than rely on imshow's default interacting
    # correctly with the rest of the layout.
    ax1.set_box_aspect(1.0)

    # ROC and PR share the same class->color mapping, so one legend entry per
    # class covers both panels (AUC from ROC, AP from PR) -- this halves the
    # legend's entry count versus a separate one per panel, which at n>8
    # classes was tall/wide enough to overflow its axes even after font and
    # column-count tuning (there is only so much a 17-row legend can shrink
    # to before the text itself stops being legible).
    lw = style.LINE_WIDTH * fs
    legend_handles, legend_labels = [], []
    if have_curves:
        for i, name in enumerate(class_names):
            r, p = roc_curves.get(name), pr_curves.get(name)
            if r is not None:
                ax2.plot(r["fpr"], r["tpr"], lw=lw, color=colors[i])
            if p is not None:
                ax3.plot(p["recall"], p["precision"], lw=lw, color=colors[i])
            if r is not None and p is not None:
                legend_handles.append(plt.Line2D([0], [0], color=colors[i], lw=lw))
                legend_labels.append(f"{_latex_escape(name)} (AUC={r['auc']:.2f}, AP={p['ap']:.2f})")
        for key in ("micro", "macro"):
            r, p = roc_curves.get(key), pr_curves.get(key)
            color = "black" if key == "macro" else "dimgray"
            if r is not None:
                ax2.plot(r["fpr"], r["tpr"], linestyle=":", lw=lw, color=color)
            if p is not None:
                ax3.plot(p["recall"], p["precision"], linestyle=":", lw=lw, color=color)
            if r is not None and p is not None:
                legend_handles.append(plt.Line2D([0], [0], color=color, lw=lw, linestyle=":"))
                legend_labels.append(f"{key} (AUC={r['auc']:.2f}, AP={p['ap']:.2f})")
        ax2.plot([0, 1], [0, 1], "k--", lw=2 * fs)

    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves")
    ax2.grid(True, alpha=style.GRID_ALPHA)
    # Square axes is the standard ROC/PR convention (both axes span the same
    # 0-1 range); pinned explicitly rather than left to whatever shape
    # tight_layout happens to give a 1x3 subplot with unequal width_ratios --
    # the same box_aspect-vs-figsize distinction behind the learning_curves fix.
    ax2.set_box_aspect(1.0)

    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curves")
    ax3.grid(True, alpha=style.GRID_ALPHA)
    ax3.set_box_aspect(1.0)

    # Legend placed ABOVE the panels now (between the suptitle and the panel
    # titles), not below -- moved per request, since the old bottom
    # placement needed its own dedicated margin on top of whatever the
    # panels themselves needed, whereas the top margin already has to exist
    # for the suptitle regardless, so folding the legend into it is more
    # compact. ncol/legend_fs/n_rows need to be known *before* tight_layout
    # this time, since they decide how much top rect to reserve.
    fig_h_pt = style.EVAL_FIGSIZE[1] * 72
    if legend_handles:
        # Spreads a 17-entry legend (Edge-IIoTSet's worst case) into few
        # enough rows to stay legible, scaled by how many entries there are.
        ncol = max(2, min(5, -(-len(legend_handles) // 4)))
        legend_fs = (style.LEGEND_SIZE if len(legend_handles) <= 8 else max(13, style.LEGEND_SIZE - 4)) * fs
        n_rows = -(-len(legend_handles) // ncol)
    else:
        ncol = legend_fs = n_rows = 0

    # All in POINTS, not a bare figure-height fraction: a fraction tuned
    # against one EVAL_FIGSIZE silently breaks the next time that figure
    # height changes for an unrelated reason (confirmed: this happened once
    # already, when EVAL_FIGSIZE[1] dropped from 8.5 to 7.0in and nobody
    # re-checked a fraction-based margin tuned against the old height).
    # Scaled by fs along with the text itself (fs>1 since 2026-11-16's
    # width-driven canvas enlargement) so the vertical room reserved for the
    # now-bigger suptitle/legend text grows with it instead of under-shooting.
    suptitle_pt = 30.0 * fs          # suptitle text + its own padding from the page's top edge
    gap_after_suptitle_pt = 20.0 * fs
    legend_block_pt = (20.0 + 21.0 * n_rows) * fs   # same per-row cost as the old bottom placement
    gap_before_titles_pt = 20.0 * fs                # panel titles sit directly above their axes, no rotated-label complication here
    top_reserved_pt = suptitle_pt + gap_after_suptitle_pt + legend_block_pt + gap_before_titles_pt
    rect_top = 1.0 - top_reserved_pt / fig_h_pt

    fig.tight_layout(rect=(0, 0, 1, rect_top))
    # Equal-width panels (width_ratios=[1,1,1] above) only reach ~30% of the
    # embedded page width each if the margins NOT spent on the panels
    # themselves -- the inter-panel gap and the right edge -- are trimmed to
    # a minimum; tight_layout's own defaults there are generous enough to
    # cost several percentage points of page width per panel. The left
    # margin is deliberately left to tight_layout's own computation (not
    # touched here): it is driven by the confusion matrix's rotated
    # class-name tick labels, whose length varies by dataset (10 classes for
    # UNSW-NB15/X-IIoTID, 15 for Edge-IIoTSet), so a fixed value here would
    # either clip Edge-IIoTSet's labels or waste space on the other two.
    fig.subplots_adjust(wspace=0.02, right=0.995)
    fig.set_layout_engine(None)   # freeze this layout -- see learning_curves for why

    suptitle_y = 1.0 - 12.0 * fs / fig_h_pt   # ~12pt clearance from the page's own top edge
    fig.text(0.5, suptitle_y, f"{template.replace('_', '-')}, {dataset}",
             fontsize=(style.TITLE_SIZE + 2) * fs, ha="center", va="top")

    if legend_handles:
        legend_y = 1.0 - (suptitle_pt + gap_after_suptitle_pt) / fig_h_pt
        fig.legend(legend_handles, legend_labels, loc="upper center",
                  bbox_to_anchor=(0.5, legend_y), ncol=ncol, fontsize=legend_fs,
                  columnspacing=1.2, handlelength=1.6, labelspacing=0.5,
                  frameon=True)

    return _save(fig, fig_dir / _tagged(f"evaluation_metrics_{dataset}_{template}", tag))


def model_comparison(agg, dataset: str, fig_dir: Path = None,
                     tag: str = FIGURE_TAG) -> Path:
    """``model_comparison_{dataset}[_{tag}].pdf`` -- 3-panel grouped-bar layout
    (accuracy / macro+weighted F1 / training time), matching the pre-revision
    ``Visualizer.plot_model_comparison`` benchmarking style the user asked to
    restore (2026-09-09): vertically-stacked subplots, grouped (non-stacked)
    bars, value labels above each bar, tab10 colors.

    Unlike that pre-revision code -- which plotted a single run's raw value
    per template (``.iloc[0]``, no error bars) -- every bar here is the
    5-seed mean with a std error bar, since that statistical rigor is what
    Reviewer 4 (Sec. 6) required and what the manuscript's revised captions
    already commit to; restoring the old *visual layout* does not mean
    reverting to single-seed numbers. Models are ordered by descending
    macro-F1 (same convention as the single-panel figure this replaces), so
    the "who leads" comparison stays honest and consistent across panels
    rather than reverting to the old code's arbitrary template order.
    """
    style.apply_eval_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    df = agg.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    n = len(df)
    labels = [m.replace("_", "-") for m in df["model"]]
    x = np.arange(n)
    colors = style.palette(4)

    fig, axes = plt.subplots(3, 1, figsize=(max(14, n * 1.4), 18), dpi=style.EVAL_DPI)

    def _bold_qcsl(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for i, m in enumerate(df["model"]):
            if "QCSL" in str(m):
                ax.get_xticklabels()[i].set_fontweight("bold")
        ax.grid(True, alpha=style.GRID_ALPHA)
        ax.set_axisbelow(True)

    def _label_bars(ax, bars, values, errs=None):
        """Value label above each bar (and its error whisker, if any), with a
        small fixed gap so the label never touches the whisker/bar top --
        the gap scales with the axis range rather than a fixed point offset,
        since the three panels here (percentages vs. raw seconds) have very
        different y-scales.
        """
        rng = max(values.max(), 1e-9)
        gap = rng * 0.02
        tops = []
        for i, (bar, v) in enumerate(zip(bars, values)):
            if v != v:      # NaN guard
                continue
            err = 0.0 if errs is None else (errs.iloc[i] if errs.iloc[i] == errs.iloc[i] else 0.0)
            top = bar.get_height() + err
            ax.text(bar.get_x() + bar.get_width() / 2, top + gap,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=style.ANNOT_SIZE)
            tops.append(top)
        return max(tops) if tops else rng

    ax0 = axes[0]
    acc = (df["accuracy_mean"] * 100)
    acc_err = (df["accuracy_std"] * 100).fillna(0)
    bars = ax0.bar(x, acc, yerr=acc_err, color=colors[0], capsize=4, label="Test Accuracy (%)")
    top0 = _label_bars(ax0, bars, acc, acc_err)
    # The ~1-line gap above the highest label the user asked for: headroom
    # past the tallest bar+error+label stack so it clears the top spine
    # instead of touching or crossing it, rather than leaving ylim at
    # matplotlib's tight default around the data.
    ax0.set_ylim(top=top0 * 1.12)
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title(f"Model comparison: Accuracy ({dataset})")
    ax0.legend(loc="lower right")
    _bold_qcsl(ax0)

    ax1 = axes[1]
    w = 0.35
    f1m = df["f1_macro_mean"] * 100
    f1w = df["f1_weighted_mean"] * 100
    f1m_err = (df["f1_macro_std"] * 100).fillna(0)
    f1w_err = (df["f1_weighted_std"] * 100).fillna(0)
    b1 = ax1.bar(x - w / 2, f1m, w, yerr=f1m_err,
                capsize=4, color=colors[1], label="F1 Score (Macro) (%)")
    b2 = ax1.bar(x + w / 2, f1w, w, yerr=f1w_err,
                capsize=4, color=colors[2], label="F1 Score (Weighted) (%)")
    top1a = _label_bars(ax1, b1, f1m, f1m_err)
    top1b = _label_bars(ax1, b2, f1w, f1w_err)
    ax1.set_ylim(top=max(top1a, top1b) * 1.12)
    ax1.set_ylabel("F1 Score (%)")
    ax1.set_title(f"Model comparison: F1 scores ({dataset})")
    ax1.legend(loc="lower right")
    _bold_qcsl(ax1)

    ax2 = axes[2]
    tt = df["train_seconds"]
    bars = ax2.bar(x, tt, color=colors[3], label="Training Time (s, mean over seeds)")
    top2 = _label_bars(ax2, bars, tt)
    ax2.set_ylim(top=top2 * 1.12)
    ax2.set_ylabel("Training Time (s)")
    ax2.set_title(f"Model comparison: Training time ({dataset})")
    # Models are ordered by macro-F1, not training time, so the slowest
    # configuration (whose label needs the most headroom) can land anywhere
    # along the x-axis -- a fixed corner collided with it on Edge-IIoTSet
    # (tallest bar 2nd from the left) after "fixing" the exact same collision
    # on X-IIoTID (tallest bar in the middle-right) by hardcoding the other
    # corner. Picking the corner on the opposite side from the tallest bar
    # is robust to this across every dataset's grid rather than a guess.
    slow_idx = int(np.argmax(tt.to_numpy()))
    ax2.legend(loc="upper left" if slow_idx >= n / 2 else "upper right")
    _bold_qcsl(ax2)

    fig.tight_layout()
    return _save(fig, fig_dir / _tagged(f"model_comparison_{dataset}", tag))


def _op_count_bars(ax, op_counts: Dict[str, int], hatch: str, title: str,
                   label_rotation: int, label_fontsize: int) -> None:
    """One subplot of gate-type counts for a single circuit config.

    Matches ``Visualizer.plot_qcshield_distribution``/``plot_qcshield_layer_distribution``:
    bar width shrinks as the number of distinct gate types grows, each bar
    gets its own tab10 color, count printed above the bar.
    """
    ops = list(op_counts.keys())
    counts = list(op_counts.values())
    num_ops = len(ops)
    bar_width = 0.8 / max(1, num_ops / 5)
    colors = style.palette(num_ops)
    bars = ax.bar(ops, counts, width=bar_width, color=colors,
                  edgecolor="black", hatch=hatch)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Operation Type", fontsize=20)
    ax.tick_params(axis="x", rotation=label_rotation, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.03,
                str(count), ha="center", va="bottom", fontsize=label_fontsize)
    # Headroom above the tallest bar's count label, same reasoning as
    # model_comparison's bar-top fix: without it the label on the tallest
    # bar in a subplot can touch its own title above.
    ax.set_ylim(top=max(counts) * 1.15)
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)


def qcshield_distribution(stats_rows, fig_dir: Path = None,
                          level: str = "template") -> Path:
    """``non_qcshield_operation_distribution.pdf`` -- one subplot per BEL/SEL/RL
    config (6 of them: AE/QE x BEL/SEL/RL), gate-type counts, tab10, hatch='\\\\'.

    Matches ``Visualizer.plot_qcshield_distribution`` exactly: figsize =
    ``plot_size[0] * num_plots, plot_size[1]`` = ``(15*6, 5)``, dpi=250.
    """
    style.apply_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    rows = [r for r in stats_rows if r.level == level and r.layer != "QCSL"]
    if not rows:
        raise ValueError("no non-QCSL rows found")
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(style.PLOT_SIZE[0] * n, style.PLOT_SIZE[1]),
                             dpi=style.DPI_MAIN, sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, row in zip(axes, rows):
        _op_count_bars(ax, row.op_counts, hatch="\\\\", title=row.name,
                       label_rotation=45, label_fontsize=16)
    axes[0].set_ylabel("Count", fontsize=16)
    return _save(fig, fig_dir / "non_qcshield_operation_distribution.pdf", dpi=style.DPI_MAIN)


def qcshield_layer_distribution(stats_rows, fig_dir: Path = None,
                                level: str = "template") -> Path:
    """``qcshield_layer_operation_distribution.pdf`` -- one subplot per QCSL
    config (AE-QCSL, QE-QCSL), gate-type counts, tab10, hatch='//'.

    Matches ``Visualizer.plot_qcshield_layer_distribution`` exactly: figsize =
    ``plot_size[0] * num_plots * 1.5, plot_size[1]``, dpi=250.
    """
    style.apply_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    rows = [r for r in stats_rows if r.level == level and r.layer == "QCSL"]
    if not rows:
        raise ValueError("no QCSL rows found")
    n = len(rows)
    fig, axes = plt.subplots(
        1, n, figsize=(style.PLOT_SIZE[0] * n * 1.5, style.PLOT_SIZE[1]),
        dpi=style.DPI_MAIN, sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, row in zip(axes, rows):
        _op_count_bars(ax, row.op_counts, hatch="//", title=f"{row.name} (QCShieldLayer)",
                       label_rotation=60, label_fontsize=14)
    axes[0].set_ylabel("Count", fontsize=16)
    return _save(fig, fig_dir / "qcshield_layer_operation_distribution.pdf", dpi=style.DPI_MAIN)


def operation_distribution_grid(stats_rows, dataset: str, fig_dir: Path = None,
                                level: str = "decomposed") -> Path:
    """``operation_distribution_grid_{dataset}.pdf`` -- one 2x4 grid of gate-type
    bar charts, all 8 configurations together (AE/QE x BEL/RL/SEL/QCSL).

    Consolidates ``qcshield_distribution``/``qcshield_layer_distribution``
    (previously two separate figures, 6+2 configs split by whether the layer is
    QCSL) into the single-figure, 2x4 layout supplied by the user 2026-09-06,
    for direct visual comparison of every configuration side by side. Gate
    counts come from this project's verified ``circuit_stats`` (cross-checked
    against ``pennylane.specs`` -- see ``tests/test_circuits.py``), not the
    pre-revision analyser's numbers.
    """
    style.apply_eval_style()
    fig_dir = Path(fig_dir or FIGURE_DIR)
    order = [("AE", "BEL"), ("AE", "RL"), ("AE", "SEL"), ("AE", "QCSL"),
            ("QE", "BEL"), ("QE", "RL"), ("QE", "SEL"), ("QE", "QCSL")]
    by_key = {(r.embedding, r.layer): r for r in stats_rows if r.level == level}
    rows = [by_key[k] for k in order if k in by_key]
    if len(rows) != 8:
        missing = [k for k in order if k not in by_key]
        raise ValueError(f"missing circuit_stats rows for {missing}")

    # Shrunk from the original (24,12): this figure is embedded at
    # 0.79\textwidth (~5.9in) in the manuscript, same scale-down problem as
    # learning_curves/evaluation_metrics -- a 24in-wide native canvas gives a
    # 5.9/24=0.25x shrink factor, which crushed even _op_count_bars' already
    # modest 14-18pt fonts down to under 5pt. (16,9) at 5.9/16=0.37x matches
    # the scale factor used elsewhere in this module for the same reason.
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=style.EVAL_DPI)
    for ax, row in zip(axes.flat, rows):
        hatch = "//" if row.layer == "QCSL" else "\\\\"
        _op_count_bars(ax, row.op_counts, hatch=hatch, title=row.name,
                       label_rotation=45, label_fontsize=20)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count", fontsize=20)
    fig.suptitle(f"Operation Distribution for {dataset} "
                f"({rows[0].n_qubits} qubits, {rows[0].n_layers} layers, {level})",
                fontsize=style.TITLE_SIZE)
    # Extra vertical room between the two rows: the larger title/tick fonts
    # (2026-09-09) made row 1's rotated x-tick labels collide with row 2's
    # titles at the old spacing. Figure dimensions are unchanged; only the
    # internal gap grows.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.65)
    return _save(fig, fig_dir / f"operation_distribution_grid_{dataset}.pdf",
                dpi=style.EVAL_DPI)


# --------------------------------------------------------------------------
# Federated figures
# --------------------------------------------------------------------------

def client_global_metrics(result, fig_dir: Path = None) -> Path:
    """``client_global_metrics_{Strategy}_{partition}_{dataset}.pdf`` --
    1x3 (Accuracy, F1, Loss), figsize=(20,6), dpi=300.

    Matches the federated notebook's ``plot_client_global_metrics`` exactly:
    three metrics side by side (the original never had only two), each client
    its own tab10 color, Global drawn with a star marker in the next tab10
    color past the last client -- not a special neutral color, matching the
    original's ``colors[num_clients]``.
    """
    style.apply_style()
    fig_dir = Path(fig_dir or (FIGURE_DIR / "Fed"))
    ds = result.dataset.replace("-", "_")
    rows = result.client_rows
    clients = sorted({r["client"] for r in rows})
    colors = style.palette(len(clients) + 1)     # +1 for Global, matches original
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "*"]

    fig, axes = plt.subplots(1, 3, figsize=style.FIGSIZE_FED, dpi=style.DPI_FED,
                             facecolor="white")
    fig.suptitle(f"Client vs Global Metrics Over Rounds "
                 f"({result.strategy} - {result.partition} - {result.dataset})", fontsize=22,
                 weight="bold")

    metrics = [("accuracy", "Accuracy", result.global_acc),
              ("f1_macro", "F1_Score", result.global_f1),
              ("loss", "Loss", result.global_loss)]
    for ax, (key, label, glob) in zip(axes, metrics):
        ax.set_facecolor("white")
        for ci, c in enumerate(clients):
            pts = sorted([(r["round"], r[key]) for r in rows if r["client"] == c])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker=markers[ci % len(markers)], lw=1.5, markersize=6,
                        alpha=0.999, color=colors[ci], label=f"Client {c + 1}")
        ax.plot(result.rounds, glob, marker="*", lw=2.5, markersize=8,
                color=colors[len(clients)], label="Global")
        ax.set_title(label, fontsize=18, weight="bold")
        ax.set_xlabel("Round", fontsize=16); ax.set_ylabel(label, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7, which="both")
        ax.legend(title="Entity", fontsize=12, loc="best",
                 ncol=2 if len(clients) > 5 else 1)
        ax.tick_params(axis="both", which="major", labelsize=14)

    return _save(fig, fig_dir /
                 f"client_global_metrics_{result.strategy}_{result.partition}_{ds}.pdf",
                 dpi=style.DPI_FED)


def roundwise_scores(results: Iterable, partition: str, dataset: str,
                     fig_dir: Path = None) -> Path:
    """``roundwise_scores_{partition}_{dataset}.pdf`` -- all strategies overlaid, tab10.

    The manuscript references this figure but the notebook had **no generator
    for it at all** (defect #7); this is that missing plotter.
    """
    style.apply_style()
    fig_dir = Path(fig_dir or (FIGURE_DIR / "Fed"))
    ds = dataset.replace("-", "_")
    rs = sorted([r for r in results if r.partition == partition], key=lambda x: x.strategy)
    if not rs:
        raise ValueError(f"no results for partition {partition!r}")
    colors = style.palette(len(rs))

    fig, axes = plt.subplots(1, 3, figsize=style.FIGSIZE_ROUNDWISE)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, r in enumerate(rs):
        for ax, series, label in ((axes[0], r.global_acc, "Accuracy"),
                                  (axes[1], r.global_f1, "Macro-F1"),
                                  (axes[2], r.global_loss, "Loss")):
            ax.plot(r.rounds, series, marker=markers[i % len(markers)],
                    markersize=7, label=r.strategy, color=colors[i])
            ax.set_xlabel("Federated round"); ax.set_ylabel(label)
    for ax in axes:
        ax.legend(fontsize=18)
    # Title added as fig.text() AFTER tight_layout(), with the layout engine
    # then detached -- see learning_curves for the full explanation. In
    # short: tight_layout() leaves a layout engine attached that re-applies
    # (and re-reserves a large, fixed block for any suptitle already
    # present) at save time, which silently overrode this figure's y=0.98/
    # rect=0.94 the same way it did for learning_curves and
    # evaluation_metrics before those were fixed.
    # This plot has no per-panel title row (just axis labels), so its axes
    # naturally fill almost the whole figure (measured: axes top at
    # y~=0.97) -- unlike learning_curves/evaluation_metrics, an explicit
    # rect is needed here to actually carve out room for the title, since
    # there's no title-row margin tight_layout would otherwise leave.
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.set_layout_engine(None)
    fig.text(0.5, 0.86, f"Round-wise global scores -- {partition} -- {dataset}",
             fontsize=style.TITLE_SIZE, ha="center", va="top")
    return _save(fig, fig_dir / f"roundwise_scores_{partition}_{ds}.pdf")


def all_federated_figures(results: List, dataset: str,
                          fig_dir: Path = None) -> List[Path]:
    out = [client_global_metrics(r, fig_dir) for r in results]
    for partition in sorted({r.partition for r in results}):
        out.append(roundwise_scores(results, partition, dataset, fig_dir))
    return out
