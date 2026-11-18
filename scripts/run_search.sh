#!/bin/bash
# Architecture search: 6 variants x 2 seeds = 12 runs, split across 3 parallel
# workers (2 variants each) to keep GPU contention reasonable while cutting
# wall-clock time roughly 3x versus fully sequential.
cd /home/jpsahoo/Desktop/skills-main/claude_research
LOGDIR=./logs
mkdir -p "$LOGDIR"

run_subset() {
  local tag=$1; shift
  PYTHONPATH=. ./.venv/bin/python - "$@" <<'PYEOF' > "$LOGDIR/search_${tag}.log" 2>&1
import sys, json, time, logging
from pathlib import Path
sys.path.insert(0, ".")
from qcshield.cli import get_splits, run_one
from qcshield.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search")

ALL_VARIANTS = {
    "baseline_nlayers3":  dict(n_layers=3),
    "baseline_nlayers4":  dict(n_layers=4),
    "chunks5":            dict(quantum_chunks=5),
    "chunks8":            dict(quantum_chunks=8),
    "lrmult2":            dict(quantum_lr_multiplier=2.0),
    "lrmult10":           dict(quantum_lr_multiplier=10.0),
}
names = sys.argv[1:]
out_dir = Path("results_search")
out_dir.mkdir(parents=True, exist_ok=True)

for seed in (0, 1):
    base_cfg = Config(dataset_name="Edge-IIoTSet", seed=seed)
    splits = get_splits(base_cfg)
    for name in names:
        overrides = ALL_VARIANTS[name]
        fn = out_dir / f"Edge_IIoTSet_QE_QCSL_{name}_seed{seed}.json"
        if fn.exists():
            logger.info("skip (exists): %s", fn)
            continue
        cfg = Config(dataset_name="Edge-IIoTSet", seed=seed, **overrides)
        t0 = time.perf_counter()
        record = run_one(cfg, "quantum", splits=splits, save=False)
        record["search_variant"] = name
        record["search_overrides"] = overrides
        fn.write_text(json.dumps(record, indent=2))
        logger.info("wrote %s (%.0fs) test_acc=%.4f test_f1_macro=%.4f",
                   fn, time.perf_counter() - t0,
                   record["test"]["accuracy"], record["test"]["f1_macro"])
PYEOF
}

run_subset group1 baseline_nlayers3 baseline_nlayers4 &
run_subset group2 chunks5 chunks8 &
run_subset group3 lrmult2 lrmult10 &
wait
echo "ARCHITECTURE SEARCH COMPLETE $(date)" >> "$LOGDIR/search_group1.log"
