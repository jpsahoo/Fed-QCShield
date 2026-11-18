#!/bin/bash
# X-IIoTID full grid for Revision 2 (Reviewer 4 Comment 4: dataset obsolescence).
# 12 models x 5 seeds = 60 runs, split 5-way (one worker per seed) -- the same
# concurrency pattern proven safe on UNSW-NB15/Edge-IIoTSet (see PROGRESS.md
# "Measured throughput": ~3.8x aggregate speed-up, GPU util ~97%, well under
# the 16GB card's VRAM budget at ~600MB/worker).
#
# Results land in results_xiiotid_30epoch_classweighted/, NOT results/ --
# this must never touch the existing UNSW-NB15/Edge-IIoTSet result files that
# back the published Table 3.
set -e
cd /home/jpsahoo/Desktop/skills-main/claude_research
LOGDIR=./logs
OUTDIR=./results_xiiotid_30epoch_classweighted
mkdir -p "$LOGDIR" "$OUTDIR"

run_seed() {
  local seed=$1
  PYTHONPATH=. ./.venv/bin/python - "$seed" <<'PYEOF' > "$LOGDIR/xiiotid_seed${seed}.log" 2>&1
import sys, logging
sys.path.insert(0, ".")
from qcshield.cli import get_splits, run_one
from qcshield.config import Config, SEEDS, EMBEDDINGS, LAYERS, CLASSICAL_BASELINES, all_quantum_configs
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("xiiotid_grid")

seed = int(sys.argv[1])
out_dir = Path("results_xiiotid_30epoch_classweighted")

models = ([("quantum", e, l) for e, l in all_quantum_configs()]
          + [(b, None, None) for b in CLASSICAL_BASELINES])

base = Config(dataset_name="X-IIoTID", seed=seed, epochs=30,
              class_weighted=True, results_dir=out_dir)
splits = get_splits(base)
logger.info("seed %d: %s", seed, splits.summary())

for i, (name, emb, lay) in enumerate(models, 1):
    cfg = Config(dataset_name="X-IIoTID", seed=seed, epochs=30,
                 class_weighted=True, results_dir=out_dir,
                 embedding=emb or base.embedding, layer=lay or base.layer)
    logger.info("seed %d: [%d/%d] %s %s", seed, i, len(models), name,
               f"{emb}_{lay}" if emb else "")
    run_one(cfg, name, splits=splits)

logger.info("seed %d: DONE (12/12 models)", seed)
PYEOF
}

for seed in 0 1 2 3 4; do
  run_seed "$seed" &
done
wait
echo "X-IIoTID grid complete: 60/60 runs. Results in $OUTDIR"
