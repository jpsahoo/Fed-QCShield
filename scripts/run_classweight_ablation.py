"""Class-weighted vs. unweighted loss ablation, QCShield layer only (AE-QCSL, QE-QCSL).

Runs all three datasets with class-weighted cross-entropy disabled; the
weighted counterpart already exists in the main grid. Reuses the dataset's
standard 30-epoch, AdamW+plateau recipe, changing only the loss weighting.

Zero-day ablation lives separately in run_zeroday_ablation.py.
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
logger = logging.getLogger("run_classweight_ablation")

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 30
SEED = 0
QCSL_CONFIGS = [("AE", "QCSL"), ("QE", "QCSL")]


def _run(cfg, out_dir, label, splits):
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{cfg.dataset_name.replace('-','_')}_{label}_seed{cfg.seed}.json"
    if fn.exists():
        logger.info("skip (already done): %s", fn.name)
        return
    record = run_one(cfg, "quantum", splits=splits, save=False)
    fn.write_text(json.dumps(record, indent=2))
    logger.info("wrote %s", fn.name)


def run_class_weighting_ablation():
    out_dir = ROOT / "results_classweight_ablation"
    for ds in ("UNSW-NB15", "X-IIoTID", "Edge-IIoTSet"):
        base = Config(dataset_name=ds, seed=SEED, epochs=EPOCHS, class_weighted=False)
        splits = get_splits(base)
        for emb, lay in QCSL_CONFIGS:
            cfg = Config(dataset_name=ds, seed=SEED, epochs=EPOCHS,
                        embedding=emb, layer=lay, class_weighted=False)
            _run(cfg, out_dir, f"{emb}_{lay}_unweighted", splits)
    logger.info("CLASS-WEIGHTING ABLATION COMPLETE")


if __name__ == "__main__":
    logger.info("=== Starting class-weighting ablation ===")
    run_class_weighting_ablation()
    logger.info("=== CLASS-WEIGHTING ABLATION DONE ===")
