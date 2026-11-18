#!/usr/bin/env python
"""Screen for a genuinely better-performing QE-QCSL architecture (2026-09-08).

Rationale (checked directly in qcshield/models.py before writing this): every
model in the comparison -- the quantum HybridQuantumCNNBiLSTM and all four
ClassicalBaseline variants -- routes through the identical
``n_qubits * quantum_chunks`` = 12-dimensional bottleneck before the final
classifier, so the existing comparison is fair, but three real, never-tuned
design knobs exist as plain Config fields, inherited unchanged from the
original buggy code: quantum_chunks=3, n_layers=2, quantum_lr_multiplier=5.0.
None of these were ever searched. This screens one knob at a time, against
the already-completed 5-seed baseline (results/Edge_IIoTSet_QE_QCSL_seed{0,1}.json),
using 2 seeds first (full 30-epoch protocol, unchanged) to find a promising
direction before committing to the full 5-seed confirmation run.

Usage:
    PYTHONPATH=. ./.venv/bin/python scripts/search_architecture.py --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcshield.cli import get_splits, run_one          # noqa: E402
from qcshield.config import Config                     # noqa: E402

logger = logging.getLogger("search")

VARIANTS = {
    "baseline_nlayers3":  dict(n_layers=3),
    "baseline_nlayers4":  dict(n_layers=4),
    "chunks5":            dict(quantum_chunks=5),
    "chunks8":            dict(quantum_chunks=8),
    "lrmult2":            dict(quantum_lr_multiplier=2.0),
    "lrmult10":           dict(quantum_lr_multiplier=10.0),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Edge-IIoTSet")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--out", default="results_search")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        base_cfg = Config(dataset_name=a.dataset, seed=seed)
        splits = get_splits(base_cfg)
        for name, overrides in VARIANTS.items():
            fn = out_dir / f"{a.dataset.replace('-', '_')}_QE_QCSL_{name}_seed{seed}.json"
            if fn.exists():
                logger.info("skip (exists): %s", fn)
                continue
            cfg = Config(dataset_name=a.dataset, seed=seed, **overrides)
            t0 = time.perf_counter()
            record = run_one(cfg, "quantum", splits=splits, save=False)
            record["search_variant"] = name
            record["search_overrides"] = overrides
            fn.write_text(json.dumps(record, indent=2))
            logger.info("wrote %s (%.0fs) test_acc=%.4f test_f1_macro=%.4f",
                       fn, time.perf_counter() - t0,
                       record["test"]["accuracy"], record["test"]["f1_macro"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
