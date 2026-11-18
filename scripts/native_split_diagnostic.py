#!/usr/bin/env python
"""Diagnostic: UNSW-NB15 using the dataset's own native train/test partition,
instead of pooling both HF splits and re-splitting ourselves (2026-09-08).

The HuggingFace mirror "Mireu-Lab/UNSW-NB15" exposes two splits, ``train``
(82,332 rows) and ``test`` (175,341 rows) -- note these sizes are the exact
reverse of the well-known official UNSW-NB15 convention (the original
authors' training-set.csv has 175,341 rows, testing-set.csv has 82,332), so
this mirror's own field *names* are swapped relative to standard usage. This
script does not trust the field names -- it uses the LARGER partition as the
training pool and the SMALLER as the held-out test set, matching how the
dataset is conventionally used in the literature, regardless of which HF
field each came from.

Everything downstream follows the exact same leak-free discipline as
qcshield/data.py's prepare(): a stratified validation slice is carved out of
the training pool only, and the ordinal encoders, scaler, and correlation
filter are all fit on the resulting training split only, never on the test
partition.

This is a diagnostic (single run, not a replacement for the shared
pipeline) -- it does not modify qcshield/data.py or results/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from qcshield.data import (OrdinalEncoderWithUnseen, Splits, _drop_rare_classes,
                           _correlated_columns, UNSW_TARGET, UNSW_DROP, EDGE_CATEGORICAL)
from qcshield.config import Config
from qcshield.cli import run_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("native_split")


def load_native_unsw_splits(cfg: Config) -> Splits:
    from datasets import load_dataset
    ds = load_dataset("Mireu-Lab/UNSW-NB15")
    a, b = ds["train"].to_pandas(), ds["test"].to_pandas()
    # Larger partition = conventional "training set"; smaller = conventional "test set".
    train_pool, test_pool = (a, b) if len(a) > len(b) else (b, a)
    logger.info("native partitions: training pool=%d rows, held-out test=%d rows",
               len(train_pool), len(test_pool))

    target, drop = UNSW_TARGET, UNSW_DROP

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=drop, errors="ignore")
        df = df.dropna(axis=0, how="any")
        return df

    train_pool, test_pool = clean(train_pool), clean(test_pool)
    # dedupe each partition on its own feature columns (leak-free: no
    # cross-partition information used to decide what to drop)
    feature_cols = [c for c in train_pool.columns if c != target]
    train_pool = train_pool.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    test_pool = test_pool.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    train_pool = _drop_rare_classes(train_pool, target)

    y_train_raw = train_pool.pop(target).astype(str)
    y_test_raw = test_pool.pop(target).astype(str)
    X_train_raw, X_test_raw = train_pool, test_pool

    # stratified validation slice carved out of the TRAINING POOL ONLY
    val_frac = cfg.val_size / (1.0 - cfg.test_size)   # same convention as prepare()
    idx = np.arange(len(X_train_raw))
    idx_train, idx_val = train_test_split(
        idx, test_size=val_frac, random_state=cfg.seed, stratify=y_train_raw.to_numpy())

    le_target = LabelEncoder().fit(y_train_raw.iloc[idx_train])
    class_names = [str(c) for c in le_target.classes_]

    X_train = X_train_raw.iloc[idx_train].copy()
    X_val = X_train_raw.iloc[idx_val].copy()
    X_test = X_test_raw.copy()
    y_train = le_target.transform(y_train_raw.iloc[idx_train])
    y_val = le_target.transform(y_train_raw.iloc[idx_val])
    # test labels: map through the same vocabulary; any class in test but not
    # in train (shouldn't happen after stratified carve-out, but guard anyway)
    unseen_mask = ~y_test_raw.isin(le_target.classes_)
    if unseen_mask.any():
        logger.warning("dropping %d test rows with a class unseen in training",
                       unseen_mask.sum())
        X_test, y_test_raw = X_test[~unseen_mask], y_test_raw[~unseen_mask]
    y_test = le_target.transform(y_test_raw)

    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        enc = OrdinalEncoderWithUnseen().fit(X_train[col])
        for part in (X_train, X_val, X_test):
            part[col] = enc.transform(part[col])

    for part_name in ("X_train", "X_val", "X_test"):
        pass
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    num_cols = X_train.columns.tolist()
    scaler = RobustScaler().fit(X_train[num_cols])
    for part in (X_train, X_val, X_test):
        part[num_cols] = scaler.transform(part[num_cols])

    to_drop = _correlated_columns(X_train[num_cols], cfg.corr_threshold)
    if to_drop:
        logger.info("dropping %d correlated columns (train-only): %s", len(to_drop), to_drop[:8])
    X_train, X_val, X_test = (p.drop(columns=to_drop, errors="ignore")
                              for p in (X_train, X_val, X_test))

    feature_names = X_train.columns.tolist()
    splits = Splits(
        X_train=X_train.to_numpy(np.float32), y_train=y_train.astype(np.int64),
        X_val=X_val.to_numpy(np.float32), y_val=y_val.astype(np.int64),
        X_test=X_test.to_numpy(np.float32), y_test=y_test.astype(np.int64),
        feature_names=feature_names, class_names=class_names, dataset_name="UNSW-NB15")
    logger.info("%s", splits.summary())
    return splits


if __name__ == "__main__":
    cfg = Config(dataset_name="UNSW-NB15", seed=0)
    splits = load_native_unsw_splits(cfg)
    record = run_one(cfg, "quantum", splits=splits, save=False)
    print("\n=== NATIVE-SPLIT DIAGNOSTIC RESULT (QE-QCSL, seed 0) ===")
    print("accuracy:", record["test"]["accuracy"])
    print("f1_macro:", record["test"]["f1_macro"])
    print("f1_weighted:", record["test"]["f1_weighted"])
    for name, f1 in zip(splits.class_names, record["test"]["per_class_f1"]):
        print(f"  {name:16s} f1={f1:.4f}")
    Path("results_search").mkdir(exist_ok=True)
    import json
    Path("results_search/UNSW_NB15_QE_QCSL_nativesplit_seed0.json").write_text(
        json.dumps(record, indent=2))
