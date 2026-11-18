"""Curate a timestamped snapshot of exactly the results that back the paper
(through the Ablation Analysis section): JSON result files, CSV summaries,
and the PDF figures actually embedded in Fed-QCShield_V9.tex. Excludes
scratch/superseded runs (old buggy-pipeline X-IIoTID data, the UNSW-only
batch-size/LR-multiplier ablation stages that were kept out of the paper,
and the redundant single-seed UNSW unweighted runs superseded by the
existing 5-seed results_baseline_noclassweight/ data).

Read-only with respect to any in-progress run: only copies already-written
files, never touches results_classweight_ablation/ or any other directory
a background job might still be writing to.
"""
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]          # Code_V8/
PROJECT_ROOT = ROOT.parent.parent                     # .../Fed-QCS_Nov2026/../ (claude_research)
LATEX_DIR = ROOT.parent / "Latex_V9"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / f"paper_results_{TIMESTAMP}"


def copy_matching(src_dir: Path, patterns, dst_dir: Path, label: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in patterns:
        for f in sorted(src_dir.glob(pattern)):
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    print(f"[{label}] copied {n} files -> {dst_dir.relative_to(ROOT)}")
    return n


def write_csv_summary(results_dir: Path, dataset: str, out_csv: Path):
    from qcshield.stats import aggregate
    df = aggregate(results_dir, dataset)
    if df.empty:
        print(f"  (no data for {dataset} in {results_dir}, skipping CSV)")
        return
    cols = ["model", "n_seeds", "accuracy_mean", "accuracy_std",
            "f1_macro_mean", "f1_macro_std", "f1_weighted_mean", "f1_weighted_std",
            "wilcoxon_p_vs_headline", "headline_wins", "params", "quantum_params",
            "train_seconds"]
    cols = [c for c in cols if c in df.columns]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.relative_to(ROOT)} ({len(df)} rows)")


def main():
    print(f"Building snapshot at {OUT.relative_to(ROOT)}")

    # ---- Centralized: all three datasets, 5-seed, fixed pipeline ----------
    central = OUT / "centralized"
    copy_matching(PROJECT_ROOT / "results_30epoch_classweighted", ["UNSW_NB15_*.json"],
                  central / "UNSW-NB15", "centralized/UNSW-NB15")
    copy_matching(PROJECT_ROOT / "results_30epoch_classweighted", ["Edge_IIoTSet_*.json"],
                  central / "Edge-IIoTSet", "centralized/Edge-IIoTSet")
    copy_matching(ROOT / "results_xiiotid_cleaned_30epoch_classweighted", ["X_IIoTID_*.json"],
                  central / "X-IIoTID", "centralized/X-IIoTID (fixed pipeline)")

    print("Writing centralized CSV summaries...")
    write_csv_summary(central / "UNSW-NB15", "UNSW-NB15", central / "summary_UNSW-NB15.csv")
    write_csv_summary(central / "Edge-IIoTSet", "Edge-IIoTSet", central / "summary_Edge-IIoTSet.csv")
    write_csv_summary(central / "X-IIoTID", "X-IIoTID", central / "summary_X-IIoTID.csv")

    # ---- Federated: all three datasets ------------------------------------
    fed = OUT / "federated"
    copy_matching(PROJECT_ROOT / "results" / "fed", ["fed_*_UNSW_NB15.json"],
                  fed / "UNSW-NB15", "federated/UNSW-NB15")
    copy_matching(PROJECT_ROOT / "results" / "fed", ["fed_*_Edge_IIoTSet.json"],
                  fed / "Edge-IIoTSet", "federated/Edge-IIoTSet")
    copy_matching(ROOT / "results_xiiotid_cleaned_fed", ["fed_*_X_IIoTID.json"],
                  fed / "X-IIoTID", "federated/X-IIoTID (fixed pipeline)")

    # ---- Ablation: class-weighted vs unweighted, all three datasets -------
    abl = OUT / "ablation" / "class_weighting"
    copy_matching(PROJECT_ROOT / "results_baseline_noclassweight",
                  ["UNSW_NB15_AE_QCSL_seed*.json", "UNSW_NB15_QE_QCSL_seed*.json"],
                  abl / "UNSW-NB15_unweighted", "ablation/UNSW-NB15 unweighted (5-seed)")
    copy_matching(PROJECT_ROOT / "results_baseline_noclassweight",
                  ["Edge_IIoTSet_AE_QCSL_seed*.json", "Edge_IIoTSet_QE_QCSL_seed*.json"],
                  abl / "Edge-IIoTSet_unweighted", "ablation/Edge-IIoTSet unweighted (5-seed)")
    copy_matching(ROOT / "results_classweight_ablation",
                  ["X_IIoTID_AE_QCSL_unweighted_seed*.json", "X_IIoTID_QE_QCSL_unweighted_seed*.json"],
                  abl / "X-IIoTID_unweighted", "ablation/X-IIoTID unweighted")

    # ---- Figures: exactly what the paper embeds ----------------------------
    fig_src = LATEX_DIR / "figures" / "Plots"
    fig_dst = OUT / "figures"
    if fig_src.is_dir():
        shutil.copytree(fig_src, fig_dst, dirs_exist_ok=True)
        n = sum(1 for _ in fig_dst.rglob("*.pdf"))
        print(f"[figures] copied {n} PDFs -> {fig_dst.relative_to(ROOT)}")
    else:
        print(f"  WARNING: {fig_src} not found, skipping figures")

    # ---- Manifest ------------------------------------------------------------
    manifest = {
        "generated": TIMESTAMP,
        "covers": "Sections 6.1-6.10 (centralized, federated, ablation) of Fed-QCShield_V9.tex",
        "excludes": [
            "Stage A (batch size) and Stage B (LR multiplier) ablations -- "
            "UNSW-NB15-only, not included in the paper",
            "Old buggy-pipeline X-IIoTID results (pre data-cleaning fix)",
            "Redundant single-seed UNSW-NB15 unweighted runs superseded by "
            "the existing 5-seed results_baseline_noclassweight/ data",
            "Zero-day ablation (not yet run)",
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSnapshot complete: {OUT}")


if __name__ == "__main__":
    main()
