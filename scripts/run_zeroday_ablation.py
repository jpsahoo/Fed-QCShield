"""Zero-day ablation, X-IIoTID only, QCShield layer only (AE-QCSL, QE-QCSL).

Holds crypto-ransomware entirely out of train+val (kept only in test), then
retrains AE-QCSL and QE-QCSL from scratch with no training signal for that
class at all. Reports what fraction of held-out crypto-ransomware test rows
get caught as *some* attack class (the IDS still raises an alarm, even if
mislabeled) vs. missed as Normal (the failure mode that actually matters for
a genuinely novel attack the system was never trained to recognize).

Kept as its own script (not folded into run_classweight_ablation.py) since
it needs a custom train/val split rather than a simple config-knob change.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qcshield.cli import run_one
from qcshield.config import Config
from qcshield import data as D

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_zeroday_ablation")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_zeroday_ablation"
DATASET = "X-IIoTID"
HELD_OUT_CLASS = "crypto-ransomware"
EPOCHS = 30
SEED = 0
QCSL_CONFIGS = [("AE", "QCSL"), ("QE", "QCSL")]


def build_zeroday_splits() -> D.Splits:
    base = Config(dataset_name=DATASET, seed=SEED, epochs=EPOCHS, class_weighted=True)
    full = D.prepare(base)
    target_idx = full.class_names.index(HELD_OUT_CLASS)

    train_mask = full.y_train != target_idx
    val_mask = full.y_val != target_idx
    logger.info("zero-day split: removing %d train rows, %d val rows for class %r "
               "(kept in test: %d rows)",
               int((~train_mask).sum()), int((~val_mask).sum()), HELD_OUT_CLASS,
               int((full.y_test == target_idx).sum()))

    return D.Splits(
        X_train=full.X_train[train_mask], y_train=full.y_train[train_mask],
        X_val=full.X_val[val_mask], y_val=full.y_val[val_mask],
        X_test=full.X_test, y_test=full.y_test,
        feature_names=full.feature_names, class_names=full.class_names,
        dataset_name=full.dataset_name), target_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zeroday_splits, target_idx = build_zeroday_splits()
    normal_idx = zeroday_splits.class_names.index("Normal")

    for emb, lay in QCSL_CONFIGS:
        label = f"{emb}_{lay}_zeroday_{HELD_OUT_CLASS.replace('-','')}"
        fn = OUT_DIR / f"{DATASET.replace('-','_')}_{label}_seed{SEED}.json"
        if fn.exists():
            logger.info("skip (already done): %s", fn.name)
            continue

        cfg = Config(dataset_name=DATASET, seed=SEED, epochs=EPOCHS,
                    embedding=emb, layer=lay, class_weighted=True)
        logger.info("=== zero-day %s-%s (held out: %s) ===", emb, lay, HELD_OUT_CLASS)
        record = run_one(cfg, "quantum", splits=zeroday_splits, save=False)

        confusion = np.array(record["test"]["confusion"])
        row = confusion[target_idx]
        n = int(row.sum())
        predicted_normal = int(row[normal_idx])
        record["zero_day_analysis"] = {
            "held_out_class": HELD_OUT_CLASS,
            "n_test_rows": n,
            "predicted_as_normal": predicted_normal,
            "predicted_as_some_attack": n - predicted_normal,
            "caught_rate": (n - predicted_normal) / max(n, 1),
        }
        fn.write_text(json.dumps(record, indent=2))
        logger.info("wrote %s | zero-day caught_rate=%.3f (n=%d, missed-as-Normal=%d)",
                   fn.name, record["zero_day_analysis"]["caught_rate"], n, predicted_normal)

    logger.info("=== ZERO-DAY ABLATION COMPLETE ===")


if __name__ == "__main__":
    main()
