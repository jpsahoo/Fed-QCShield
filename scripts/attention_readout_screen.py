#!/usr/bin/env python
"""Screen the swap-test-equivalent 'attention' readout against the expval
baseline (2026-09-09), inspired by (not copied from) a user-supplied paper on
swap-test-based quantum attention for Quantum Phase Recognition.

Compares readout="attention" vs. the existing readout="expval" baseline for:
  - QE-QCSL (the quantum model)
  - CNN-BiLSTM (plain classical control -- CNN-BiLSTM-MLP dropped from this
    screen per instruction; note CNN-BiLSTM is NOT dimension-matched to the
    quantum path the way CNN-BiLSTM-MLP is, since it never reshapes fc2's
    output, so this comparison is weaker evidence than the MLP control would
    give -- acceptable for a first screen, not for the final claim)

2 seeds first (full 30-epoch protocol, unchanged), Edge-IIoTSet first. Reported
honestly regardless of outcome, same as scripts/search_architecture.py.

Usage:
    PYTHONPATH=. ./.venv/bin/python scripts/attention_readout_screen.py --seeds 0,1
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

logger = logging.getLogger("attn_screen")

MODELS = ("quantum", "CNN-BiLSTM")


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
        for model_name in MODELS:
            tag = model_name.replace("-", "_")
            fn = out_dir / f"{a.dataset.replace('-', '_')}_{tag}_attention_seed{seed}.json"
            if fn.exists():
                logger.info("skip (exists): %s", fn)
                continue
            cfg = Config(dataset_name=a.dataset, seed=seed, readout="attention")
            t0 = time.perf_counter()
            record = run_one(cfg, model_name, splits=splits, save=False)
            record["readout"] = "attention"
            fn.write_text(json.dumps(record, indent=2))
            logger.info("wrote %s (%.0fs) test_acc=%.4f test_f1_macro=%.4f",
                       fn, time.perf_counter() - t0,
                       record["test"]["accuracy"], record["test"]["f1_macro"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
