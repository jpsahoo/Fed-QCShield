"""Low-cost ablation suite, QCShield layer only (AE-QCSL, QE-QCSL).

Chained in one process so it can be launched once and run unattended:

  Stage A - batch size:        UNSW-NB15, batch in {128, 256, 512}, seed 0.
  Stage B - quantum LR mult.:  UNSW-NB15, multiplier in {1, 2, 10} (5 is the
                                existing main-grid result, reused as a point).
  Stage C - class weighting:   UNSW-NB15 / X-IIoTID / Edge-IIoTSet, unweighted
                                loss (weighted already exists in the main
                                grid), seed 0.

Zero-day ablation lives separately in run_zeroday_ablation.py.

Every stage reuses the dataset's standard 30-epoch, AdamW+plateau recipe;
only the one ablated knob changes per stage.
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
logger = logging.getLogger("run_ablation_suite")

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


def stage_a_batch_size():
    out_dir = ROOT / "results_unsw_batchsize_ablation"
    for bs in (128, 256, 512):
        base = Config(dataset_name="UNSW-NB15", seed=SEED, epochs=EPOCHS,
                     batch_size=bs, class_weighted=True)
        splits = get_splits(base)
        for emb, lay in QCSL_CONFIGS:
            cfg = Config(dataset_name="UNSW-NB15", seed=SEED, epochs=EPOCHS,
                        batch_size=bs, embedding=emb, layer=lay, class_weighted=True)
            _run(cfg, out_dir, f"{emb}_{lay}_bs{bs}", splits)
    logger.info("STAGE A (batch size) COMPLETE")


def stage_b_lr_multiplier():
    out_dir = ROOT / "results_lr_multiplier_ablation"
    base = Config(dataset_name="UNSW-NB15", seed=SEED, epochs=EPOCHS, class_weighted=True)
    splits = get_splits(base)
    for mult in (1.0, 2.0, 10.0):
        for emb, lay in QCSL_CONFIGS:
            cfg = Config(dataset_name="UNSW-NB15", seed=SEED, epochs=EPOCHS,
                        embedding=emb, layer=lay, class_weighted=True,
                        quantum_lr_multiplier=mult)
            _run(cfg, out_dir, f"{emb}_{lay}_lrmult{mult:g}", splits)
    logger.info("STAGE B (quantum LR multiplier) COMPLETE")


def stage_c_class_weighting():
    out_dir = ROOT / "results_classweight_ablation"
    for ds in ("UNSW-NB15", "X-IIoTID", "Edge-IIoTSet"):
        base = Config(dataset_name=ds, seed=SEED, epochs=EPOCHS, class_weighted=False)
        splits = get_splits(base)
        for emb, lay in QCSL_CONFIGS:
            cfg = Config(dataset_name=ds, seed=SEED, epochs=EPOCHS,
                        embedding=emb, layer=lay, class_weighted=False)
            _run(cfg, out_dir, f"{emb}_{lay}_unweighted", splits)
    logger.info("STAGE C (class weighting) COMPLETE")


if __name__ == "__main__":
    logger.info("=== Starting low-cost ablation suite (A/B/C) ===")
    stage_a_batch_size()
    stage_b_lr_multiplier()
    stage_c_class_weighting()
    logger.info("=== ABLATION SUITE A/B/C ALL DONE ===")
