#!/usr/bin/env python
"""Convert Fed-QCShield result JSONs into PDF figures with selectable text.

Scans every ``*.json`` under an input directory, auto-detects whether each
file is a centralized training record (from ``qcshield.cli run``/``grid``) or a
federated simulation record (from ``qcshield.cli fed``), and renders the
corresponding plots. All plotting logic is reused from :mod:`qcshield.plots` --
this script is a thin driver, not a second implementation -- so every figure it
produces follows the same tab10/tab20 color rule and the same
``pdf.fonttype = 42`` setting that keeps PDF text selectable and searchable
(verify with ``pdftotext`` or by selecting text in a viewer). No correlation
plot is produced anywhere in this pipeline.

Usage
-----
    PYTHONPATH=. ./.venv/bin/python scripts/json_to_pdf.py \\
        --input results --output figures/from_json

    # Just the federated records, and the combined round-wise comparison:
    PYTHONPATH=. ./.venv/bin/python scripts/json_to_pdf.py \\
        --input results/fed --output figures/CH3/Plots/Fed

Each centralized JSON produces two PDFs (learning curves, evaluation metrics).
Each federated JSON produces one PDF (client-vs-global metrics). If more than
one federated record shares a (dataset, partition), a combined round-wise
comparison PDF is also produced; if more than one centralized record shares a
dataset, a combined model-comparison PDF is produced from the 5-seed
aggregate. Both combined figures can be skipped with their respective flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcshield import plots, style               # noqa: E402
from qcshield.federated import FedResult         # noqa: E402
from qcshield.stats import aggregate             # noqa: E402

logger = logging.getLogger("json_to_pdf")

_FED_FIELDS = {f.name for f in fields(FedResult)}
_FED_REQUIRED = {"strategy", "partition", "dataset", "rounds", "global_acc"}
_CENTRAL_REQUIRED = {"dataset", "model", "seed", "test", "history"}


def load_json(fp: Path) -> dict | None:
    try:
        return json.loads(fp.read_text())
    except json.JSONDecodeError:
        logger.warning("skipping unreadable JSON: %s", fp)
        return None


def is_federated_record(d: dict) -> bool:
    return _FED_REQUIRED <= d.keys()


def is_centralized_record(d: dict) -> bool:
    return _CENTRAL_REQUIRED <= d.keys()


def to_fed_result(d: dict) -> FedResult:
    return FedResult(**{k: v for k, v in d.items() if k in _FED_FIELDS})


def convert_centralized(d: dict, out_dir: Path) -> List[Path]:
    template = d["model"]              # no seed in the title/filename
    made = [plots.learning_curves(d["history"], d["dataset"], template, out_dir, tag="")]
    test = d["test"]
    made.append(plots.evaluation_metrics(
        test["confusion"], test["per_class_f1"], d["class_names"],
        d["dataset"], template, out_dir, tag="",
        roc_curves=test.get("roc_curves"), pr_curves=test.get("pr_curves")))
    return made


def convert_federated(d: dict, out_dir: Path) -> List[Path]:
    return [plots.client_global_metrics(to_fed_result(d), out_dir)]


def scan(input_dir: Path, recursive: bool, seed: int = 0
        ) -> Tuple[List[dict], List[dict], int]:
    """Return ``(centralized_records, federated_records, n_skipped)``.

    Centralized records are filtered to a single ``seed`` (default 0): plotting
    every seed makes an unfair-looking side-by-side comparison, since a model
    that happened to draw a strong seed would visually dominate the panel next
    to one that drew a weak one, even where the 5-seed mean (in
    ``model_comparison`` / ``qcshield stats``, unaffected by this filter) says
    otherwise. One fixed seed keeps every model's plotted curve directly
    comparable to every other's.
    """
    json_files = sorted(input_dir.rglob("*.json") if recursive else input_dir.glob("*.json"))
    if not json_files:
        logger.error("no .json files found under %s", input_dir)
        return [], [], 0

    centralized, federated, skipped = [], [], 0
    for fp in json_files:
        d = load_json(fp)
        if d is None:
            skipped += 1
            continue
        if is_federated_record(d):
            federated.append(d)
        elif is_centralized_record(d):
            if d.get("seed") == seed:
                centralized.append(d)
        else:
            logger.warning("skipping %s: matches neither known schema", fp.name)
            skipped += 1
    return centralized, federated, skipped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("results"),
                   help="directory to scan for *.json result files (default: results)")
    p.add_argument("--output", type=Path, default=Path("figures/from_json"),
                   help="directory to write PDFs into (default: figures/from_json)")
    p.add_argument("--no-recursive", action="store_true",
                   help="only look at *.json directly in --input, not subdirectories")
    p.add_argument("--no-model-comparison", action="store_true",
                   help="skip the combined model-comparison PDF")
    p.add_argument("--no-roundwise", action="store_true",
                   help="skip the combined round-wise federated comparison PDF")
    p.add_argument("--seed", type=int, default=0,
                   help="plot only this seed's per-run centralized records, "
                        "for a fair like-for-like comparison across models (default 0)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    style.apply_style()

    centralized, federated, skipped = scan(a.input, recursive=not a.no_recursive, seed=a.seed)
    if not centralized and not federated:
        print(f"No convertible JSON records found under {a.input}", file=sys.stderr)
        return 1

    made: List[Path] = []
    for d in centralized:
        made += convert_centralized(d, a.output)
    for d in federated:
        made += convert_federated(d, a.output)

    # -- combined figures that need the whole set --------------------------
    if federated and not a.no_roundwise:
        by_dataset: Dict[str, List[FedResult]] = {}
        for d in federated:
            by_dataset.setdefault(d["dataset"], []).append(to_fed_result(d))
        for ds, results in by_dataset.items():
            for partition in sorted({r.partition for r in results}):
                same = [r for r in results if r.partition == partition]
                if len(same) > 1:
                    made.append(plots.roundwise_scores(results, partition, ds, a.output))

    if centralized and not a.no_model_comparison:
        datasets = sorted({d["dataset"] for d in centralized})
        for ds in datasets:
            agg = aggregate(a.input, ds)
            if len(agg) > 1:
                made.append(plots.model_comparison(agg, ds, a.output, tag=""))

    print(f"\n{len(made)} PDF(s) written to {a.output}  "
         f"({len(centralized)} centralized record(s), {len(federated)} federated "
         f"record(s), {skipped} skipped)")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
