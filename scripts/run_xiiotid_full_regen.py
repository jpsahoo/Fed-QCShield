"""Full X-IIoTID re-run with the fixed data pipeline (dash-placeholder fix in
qcshield/data.py). Two phases, run sequentially:

Phase 1 - centralized: 8 quantum configs (AE/QE x BEL/SEL/RL/QCSL) + 4
classical baselines (CNN, LSTM, CNN-BiLSTM, CNN-BiLSTM-MLP), x 5 seeds
(0-4), 30 epochs each, class-weighted. Mirrors cli.cmd_grid exactly, writing
to a separate results dir so the old (buggy-pipeline) results_xiiotid_
30epoch_classweighted/ is left untouched for comparison.

Phase 2 - federated: 4 strategies (FedAvg, FedProx, FedTrimmedAvg,
FaultTolerantFedAvg) x 2 partitions (iid, dirichlet), 20 rounds, seed 0,
matching the twenty-round budget used for UNSW-NB15/Edge-IIoTSet elsewhere
in the paper.
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcshield.cli import run_one, get_splits
from qcshield.config import Config, SEEDS, CLASSICAL_BASELINES, all_quantum_configs
from qcshield.data import prepare as data_prepare

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_xiiotid_full_regen")

CENTRAL_OUT = Path(__file__).resolve().parents[1] / "results_xiiotid_cleaned_30epoch_classweighted"
FED_OUT = Path(__file__).resolve().parents[1] / "results_xiiotid_cleaned_fed"

DATASET = "X-IIoTID"
EPOCHS = 30
ROUNDS = 20

def phase1_centralized():
    models = ([("quantum", e, l) for e, l in all_quantum_configs()]
              + [(b, None, None) for b in CLASSICAL_BASELINES])
    total = len(models) * len(SEEDS)
    done = 0
    t_start = time.perf_counter()
    for seed in SEEDS:
        base = Config(dataset_name=DATASET, seed=seed, epochs=EPOCHS, class_weighted=True)
        splits = get_splits(base)
        for name, emb, lay in models:
            cfg = Config(dataset_name=DATASET, seed=seed, epochs=EPOCHS,
                        embedding=emb or base.embedding, layer=lay or base.layer,
                        class_weighted=True)
            cfg.results_dir = CENTRAL_OUT
            label = f"{emb}_{lay}" if name == "quantum" else name
            fn = CENTRAL_OUT / f"{DATASET.replace('-','_')}_{label.replace('-','_')}_seed{seed}.json"
            if fn.exists():
                logger.info("PHASE1 skip (already done): %s", fn.name)
                done += 1
                continue
            run_one(cfg, name, splits=splits)
            done += 1
            elapsed = time.perf_counter() - t_start
            logger.info("PHASE1 progress %d/%d  (elapsed %.1f min)", done, total, elapsed / 60)
    logger.info("PHASE1 COMPLETE: %d/%d centralized runs", done, total)


def phase2_federated():
    from qcshield.federated import run_all
    cfg = Config(dataset_name=DATASET, seed=0, class_weighted=True)
    logger.info("PHASE2 federated: starting %s, %d rounds", DATASET, ROUNDS)
    results = run_all(cfg, FED_OUT, num_rounds=ROUNDS)
    logger.info("PHASE2 COMPLETE: %d federated runs", len(results))


if __name__ == "__main__":
    logger.info("=== Starting full X-IIoTID regeneration (fixed pipeline) ===")
    phase1_centralized()
    phase2_federated()
    logger.info("=== ALL DONE ===")
