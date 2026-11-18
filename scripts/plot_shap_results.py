#!/usr/bin/env python
"""Render SHAP figures from scripts/run_shap_explain.py's saved .npz output.

Separated from the (slow) SHAP computation so figure style can be iterated
on without re-running KernelExplainer. Matches this project's plotting
conventions (qcshield/style.py): tab10/tab20 palette, pgf backend for
selectable PDF text, the same +4pt font sizing applied everywhere else.

Usage:
    PYTHONPATH=. ./.venv/bin/python scripts/plot_shap_results.py --dataset Edge-IIoTSet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

from qcshield import style
from qcshield.config import FIGURE_DIR
from qcshield.plots import _latex_escape, _escape_all


def plot_summary(shap_values, feature_names, dataset_name, fig_dir: Path, top_k=12):
    style.apply_eval_style()
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    colors = style.palette(len(order))
    # The manuscript embeds this figure 3-across at 0.32\textwidth (~2.4in),
    # much narrower than the (12,8)in canvas this was originally sized for --
    # the same scale-down-crushes-the-fonts problem as learning_curves and
    # evaluation_metrics (5.9/32 and 5.9/24-style ratios), just for a figure
    # with its own, much narrower, embed target this time. (6.5,5.3) at
    # 2.4/6.5=0.37x matches the scale factor used elsewhere in this project.
    fig, ax = plt.subplots(figsize=(6.5, 5.3), dpi=style.DPI_MAIN)
    bars = ax.barh(range(len(order)), mean_abs[order][::-1], color=list(reversed(colors)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(_escape_all([feature_names[i] for i in order][::-1]))
    ax.set_xlabel("mean |SHAP value| (averaged over classes)")
    # Explicit smaller fontsize: at the default axes.titlesize (20pt, to
    # match the reference style) this title ran past the right edge of the
    # (6.5,5.3)in canvas -- previously invisible because the removed
    # savefig.bbox="tight" silently widened the saved page to fit it.
    ax.set_title(f"QE-QCSL global feature importance -- {dataset_name}", fontsize=14)
    ax.grid(True, alpha=style.GRID_ALPHA, axis="x")
    ax.set_axisbelow(True)
    # Headroom past the longest bar so it doesn't run to (or past) the right
    # spine, same reasoning as the model_comparison/circuit-distribution
    # bar-top fixes, just on the x-axis for a horizontal bar chart.
    ax.set_xlim(right=mean_abs[order].max() * 1.15)
    fig.tight_layout()
    out = fig_dir / f"shap_summary_{dataset_name}.pdf"
    fig.savefig(out, dpi=style.DPI_MAIN)
    plt.close(fig)
    return out


def plot_waterfall(shap_values, class_idx, sample_idx, feature_names, class_names,
                   dataset_name, fig_dir: Path, top_k=8):
    style.apply_eval_style()
    vals = shap_values[class_idx][sample_idx]
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    colors = ["#c44e52" if vals[i] > 0 else "#4c72b0" for i in order]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=style.DPI_MAIN)
    ax.barh(range(len(order)), vals[order][::-1], color=list(reversed(colors)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(_escape_all([feature_names[i] for i in order][::-1]))
    ax.set_xlabel("SHAP value")
    ax.set_title(f"{dataset_name}: local explanation, class '{_latex_escape(class_names[class_idx])}'")
    ax.grid(True, alpha=style.GRID_ALPHA, axis="x")
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", lw=1)
    fig.tight_layout()
    # Sanitize for use in a filename AND inside a later \includegraphics{} path --
    # strip everything but alphanumerics/underscore/hyphen so no shell- or
    # LaTeX-special character (e.g. X-IIoTID's "C&C") ever reaches a path.
    import re as _re
    safe_class = _re.sub(r"[^0-9A-Za-z_-]+", "_", str(class_names[class_idx]))
    out = fig_dir / f"shap_waterfall_{dataset_name}_{safe_class}.pdf"
    fig.savefig(out, dpi=style.DPI_MAIN)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["Edge-IIoTSet", "UNSW-NB15", "X-IIoTID"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-dir", default="results_shap")
    p.add_argument("--fig-dir", default=None)
    a = p.parse_args(argv)

    tag = a.dataset.replace("-", "_")
    z = np.load(Path(a.in_dir) / f"shap_{tag}_seed{a.seed}.npz", allow_pickle=True)
    shap_values = list(z["shap_values"])
    feature_names = list(z["feature_names"])
    class_names = list(z["class_names"])

    fig_dir = Path(a.fig_dir) if a.fig_dir else (FIGURE_DIR / "SHAP")
    fig_dir.mkdir(parents=True, exist_ok=True)

    made = [plot_summary(shap_values, feature_names, a.dataset, fig_dir)]
    for ci in range(len(class_names)):
        made.append(plot_waterfall(shap_values, ci, 0, feature_names, class_names, a.dataset, fig_dir))
    for m in made:
        print("wrote", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
