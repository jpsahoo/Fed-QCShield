"""Fill out X-IIoTID's unweighted ablation to 5 seeds, matching the existing
5-seed unweighted sweep already available for UNSW-NB15 and Edge-IIoTSet
(results_baseline_noclassweight/). Seed 0 already exists for both AE-QCSL
and QE-QCSL (results_classweight_ablation/); this script runs seeds 1-4.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcshield.cli import run_one, get_splits
from qcshield.config import Config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_xiiotid_unweighted_remaining_seeds")

OUT_DIR = Path(__file__).resolve().parents[1] / "results_classweight_ablation"
EPOCHS = 30
QCSL_CONFIGS = [("AE", "QCSL"), ("QE", "QCSL")]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 4 * len(QCSL_CONFIGS)
    done = 0
    for seed in (1, 2, 3, 4):
        base = Config(dataset_name="X-IIoTID", seed=seed, epochs=EPOCHS, class_weighted=False)
        splits = get_splits(base)
        for emb, lay in QCSL_CONFIGS:
            label = f"{emb}_{lay}_unweighted"
            fn = OUT_DIR / f"X_IIoTID_{label}_seed{seed}.json"
            if fn.exists():
                logger.info("skip (already done): %s", fn.name)
                done += 1
                continue
            cfg = Config(dataset_name="X-IIoTID", seed=seed, epochs=EPOCHS,
                        embedding=emb, layer=lay, class_weighted=False)
            logger.info("=== X-IIoTID %s-%s unweighted, seed %d ===", emb, lay, seed)
            record = run_one(cfg, "quantum", splits=splits, save=False)
            fn.write_text(json.dumps(record, indent=2))
            logger.info("wrote %s", fn.name)
            done += 1
            logger.info("progress %d/%d", done, total)
    logger.info("=== X-IIoTID unweighted 5-seed fill COMPLETE: %d/%d ===", done, total)

if __name__ == "__main__":
    main()
