"""Split-first, leakage-free data pipeline for Fed-QCShield.

Pipeline order, strictly enforced to avoid train/test leakage::

    load -> drop ID/leaky cols -> dedupe -> STRATIFIED split (train/val/test)
         -> fit encoders + scaler on TRAIN only -> transform val/test
         -> corr() on TRAIN only -> drop those columns from all three splits

No Gaussian noise injection anywhere in this module; ``tests/test_no_noise.py``
greps the package to keep it that way. ``OrdinalEncoderWithUnseen`` maps any
categorical level unseen at fit time to a reserved index rather than raising,
since fitting on the training split alone means val/test can contain levels
the encoder never saw. Tensors move to the device per-batch from pinned host
memory rather than being bulk-uploaded, so large datasets don't sit resident
in VRAM for the whole run.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, DATA_ROOT

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Per-dataset column specifications (from dataset_download_hint.py)
# --------------------------------------------------------------------------

#: ``label`` is the binary attack flag and is a direct leak of ``attack_cat``.
UNSW_TARGET = "attack_cat"
UNSW_DROP = ["id", "label"]

EDGE_TARGET = "Attack_type"
EDGE_DROP = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]
#: Kept as ordinal rather than one-hot to avoid a 63 -> ~1600 column blow-up
#: that would make ``.corr()`` intractable.
EDGE_CATEGORICAL = [
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
]

#: X-IIoTID (Al-Hawawreh et al.) -- a modern, industrial-protocol-specific
#: IIoT/APT benchmark. ``class2`` is the mid-granularity attack-category label
#: used throughout this project's 10-class framing of the other two datasets;
#: ``class1``/``class3`` are coarser/finer views of the exact same attack and
#: are therefore direct leaks of ``class2``, not features.
XIIOTID_TARGET = "class2"
XIIOTID_DROP = [
    "class1", "class3",                                   # other granularities of the same label -- leaks
    "Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port",  # identifiers, not generalizable features
]
XIIOTID_KAGGLE_HANDLE = "munaalhawawreh/xiiotid-iiot-intrusion-dataset"
#: Matches the 10-class framing already used for UNSW-NB15; X-IIoTID's
#: ``class2`` has more than 10 distinct values, so the comparison is kept at
#: a comparable class-count scale by keeping only the most frequent 10,
#: decided purely on label frequency (no fitted statistic), so this
#: restriction is itself leak-free and may be applied before the split.
XIIOTID_N_CLASSES = 10

#: Dispatch tables used by :func:`prepare` when it is handed an already-loaded
#: ``df`` (the multi-seed driver loads once, re-splits per seed) -- keeps the
#: target/drop lookup in one place as the dataset count grows past two.
_DATASET_TARGETS = {"UNSW-NB15": UNSW_TARGET, "Edge-IIoTSet": EDGE_TARGET,
                    "X-IIoTID": XIIOTID_TARGET}
_DATASET_DROPS = {"UNSW-NB15": UNSW_DROP, "Edge-IIoTSet": EDGE_DROP,
                  "X-IIoTID": XIIOTID_DROP}

UNSEEN = -1  # reserved ordinal index for categories absent from the train split


class OrdinalEncoderWithUnseen:
    """LabelEncoder that tolerates unseen categories instead of raising.

    ``sklearn.preprocessing.LabelEncoder`` raises ``ValueError: y contains
    previously unseen labels`` when ``transform`` meets a category missing from
    ``fit``.  Once fitting is restricted to the training split that becomes a
    real possibility, so unseen levels are mapped to :data:`UNSEEN`.
    """

    def __init__(self) -> None:
        self.mapping_: Dict[str, int] = {}

    def fit(self, values: pd.Series) -> "OrdinalEncoderWithUnseen":
        cats = pd.unique(values.astype(str))
        self.mapping_ = {c: i for i, c in enumerate(sorted(cats))}
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        return values.astype(str).map(self.mapping_).fillna(UNSEEN).to_numpy(np.int64)

    def fit_transform(self, values: pd.Series) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class Splits:
    """Train/val/test matrices plus everything needed to interpret them."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        return (f"{self.dataset_name}: {self.n_features} features, "
                f"{self.n_classes} classes, "
                f"train={len(self.y_train)} val={len(self.y_val)} test={len(self.y_test)}")


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------

def _load_unsw() -> Tuple[pd.DataFrame, str, List[str]]:
    from datasets import load_dataset

    ds = load_dataset("Mireu-Lab/UNSW-NB15")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
    return df, UNSW_TARGET, UNSW_DROP


#: Kaggle handle and the *real* in-archive path of the DNN CSV.
#:
#: A bare ``"DNN-EdgeIIoT-dataset.csv"`` 404s: the file is two directories
#: down inside the archive, and the folder names contain spaces.
EDGE_KAGGLE_HANDLE = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
EDGE_IN_ARCHIVE = "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"


def _load_edge() -> Tuple[pd.DataFrame, str, List[str]]:
    """Load Edge-IIoTSet, preferring an already-extracted local copy.

    The Kaggle archive is 1.63 GB and the CSV inside it is 1.16 GB, so once it
    has been extracted we read it directly rather than re-resolving through
    ``kagglehub`` on every run.
    """
    local = DATA_ROOT / "kaggle" / "DNN-EdgeIIoT-dataset.csv"
    if local.exists():
        logger.info("Reading cached Edge-IIoTSet CSV at %s", local)
        df = pd.read_csv(local, low_memory=False)
    else:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, EDGE_KAGGLE_HANDLE, EDGE_IN_ARCHIVE,
            pandas_kwargs={"low_memory": False},
        )
    return df, EDGE_TARGET, EDGE_DROP


def _load_xiiotid(n_classes: int = XIIOTID_N_CLASSES) -> Tuple[pd.DataFrame, str, List[str]]:
    """Load X-IIoTID, restricted to its top-``n_classes`` most frequent labels.

    Only loading and dataset-specific label selection happen here. Encoding,
    scaling, and the correlation filter are deliberately NOT duplicated in
    this function -- they flow through the same split-first :func:`prepare`
    pipeline already used for UNSW-NB15 and Edge-IIoTSet, which is what makes
    "the same 5-seed protocol" a literal, checkable claim rather than a
    similar-but-separate one.
    """
    import kagglehub

    dataset_path = kagglehub.dataset_download(XIIOTID_KAGGLE_HANDLE)
    files = glob.glob(os.path.join(dataset_path, "**", "*"), recursive=True)
    csv_files = sorted(f for f in files if os.path.isfile(f) and f.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path!r}")
    selected = csv_files[0]
    logger.info("Loading X-IIoTID file: %s", os.path.basename(selected))

    try:
        df = pd.read_csv(selected, encoding="latin-1", engine="c",
                         low_memory=False, on_bad_lines="skip")
    except Exception as e:
        logger.warning("C engine failed (%s: %s); falling back to python engine",
                       type(e).__name__, e)
        df = pd.read_csv(selected, encoding="latin-1", engine="python",
                         on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    target = XIIOTID_TARGET
    if target not in df.columns:
        cands = [c for c in df.columns if "class2" in c.lower()]
        if not cands:
            raise ValueError(f"Target column {target!r} not found in X-IIoTID "
                             f"columns: {list(df.columns)}")
        target = cands[0]

    # Restrict to the top-n_classes most frequent labels, decided on label
    # FREQUENCY alone (no fitted statistic), so this is leak-free exactly
    # like `_drop_rare_classes` below -- it only narrows which classes are
    # eligible, before any split is drawn.
    counts_full = df[target].astype(str).str.strip().value_counts()
    top_classes = counts_full.head(n_classes).index.tolist()
    df = df[df[target].astype(str).str.strip().isin(top_classes)].copy()
    logger.info("X-IIoTID restricted to top-%d classes (of %d): %s",
               n_classes, len(counts_full), top_classes)

    drop = [c for c in XIIOTID_DROP if c in df.columns]
    return df, target, drop


def load_raw(dataset_name: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """Load a dataset and return ``(dataframe, target_column, drop_columns)``."""
    if dataset_name == "UNSW-NB15":
        return _load_unsw()
    if dataset_name == "Edge-IIoTSet":
        return _load_edge()
    if dataset_name == "X-IIoTID":
        return _load_xiiotid()
    raise ValueError(f"Unsupported dataset: {dataset_name!r}")


# --------------------------------------------------------------------------
# The split-first pipeline
# --------------------------------------------------------------------------

def _drop_rare_classes(df: pd.DataFrame, target: str, min_count: int = 10
                       ) -> pd.DataFrame:
    """Remove classes too small to appear in all three stratified splits."""
    counts = df[target].value_counts()
    rare = counts[counts < min_count].index.tolist()
    if rare:
        logger.warning("Dropping %d class(es) with <%d rows: %s",
                       len(rare), min_count, rare)
        df = df[~df[target].isin(rare)]
    return df


def _correlated_columns(X: pd.DataFrame, threshold: float) -> List[str]:
    """Columns to drop, decided on the TRAIN split only.

    The original computed this on the full frame, so test-set correlation
    structure influenced which features the model was allowed to see.
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] > threshold)]


def _is_numeric_with_placeholder(col: pd.Series, min_parseable: float = 0.95) -> bool:
    """True if an ``object``-dtype column is really numeric with a missing-value
    token (``"-"``, ``"?"``, empty string, ...) rather than a genuine category.

    X-IIoTID encodes "not applicable for this flow direction" as the literal
    string ``"-"`` in roughly 40 byte-/packet-/timing-count columns (e.g.
    ``Scr_bytes``, ``total_bytes``, ``Duration``, ``Avg_system_time``); pandas
    reads the whole column as ``object`` because of that one token mixed in
    with numeric strings. The naive ``dtype == object`` check used for
    categorical detection then treats each of those columns as a nominal
    category and ``OrdinalEncoderWithUnseen`` assigns them arbitrary integer
    codes, destroying the numeric ordering (and mapping any numeric value not
    seen in the training split to the ``UNSEEN`` sentinel). UNSW-NB15 and
    Edge-IIoTSet do not have this token in their numeric columns (checked:
    0 occurrences of ``"-"``/``"?"`` in Edge-IIoTSet's raw CSV), so this is a
    latent, dataset-specific defect that only matters for X-IIoTID.

    Decided on the TRAIN split's non-null values only, so it stays leak-free
    like every other fitted decision in this module: a column is numeric if
    at least ``min_parseable`` of its non-missing-token train values parse as
    a float.
    """
    non_null = col.dropna().astype(str)
    if non_null.empty:
        return False
    candidates = non_null[~non_null.isin(["-", "?", "", "nan", "None", "null", "NULL"])]
    if candidates.empty:
        return False
    parseable = pd.to_numeric(candidates, errors="coerce").notna().mean()
    return parseable >= min_parseable


def prepare(cfg: Config, df: pd.DataFrame | None = None) -> Splits:
    """Build leakage-free train/val/test splits for ``cfg.dataset_name``.

    Pass ``df`` to reuse an already-loaded frame (the multi-seed driver loads
    once and re-splits per seed).
    """
    name = cfg.dataset_name
    if df is None:
        df, target, drop = load_raw(name)
    else:
        df = df.copy()
        target = _DATASET_TARGETS[name]
        drop = _DATASET_DROPS[name]

    # -- 1. structural cleaning: identical for every split, so leak-free ----
    df = df.drop(columns=drop, errors="ignore")
    df = df.dropna(axis=0, how="any")
    # Deduplicate on the FEATURE columns only, ignoring the label.  Deduping on
    # all columns (features + label) keeps both halves of a label-conflicting
    # pair -- two rows with an identical feature vector but different
    # ``attack_cat`` -- so the same input can still land in train and in test.
    # On UNSW-NB15 that is 9,061 rows.  Both criteria are leak-free (neither
    # uses a fitted statistic); this one is simply the stronger of the two.
    feature_cols = [c for c in df.columns if c != target]
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    df = _drop_rare_classes(df, target)

    if cfg.subsample is not None and len(df) > cfg.subsample:
        df, _ = train_test_split(df, train_size=cfg.subsample,
                                 random_state=cfg.seed, stratify=df[target])
        df = df.reset_index(drop=True)
        logger.info("Stratified subsample to %d rows", len(df))

    y_raw = df.pop(target).astype(str)
    X_raw = df

    # -- 2. SPLIT FIRST, stratified for BOTH datasets ----------------------
    # Stratify on the raw string labels so nothing has to be fitted before the
    # split -- not even the target vocabulary.
    strat = y_raw.to_numpy()
    idx = np.arange(len(X_raw))
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=strat)
    # val_size is a fraction of the ORIGINAL frame, carved out of train
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_frac, random_state=cfg.seed,
        stratify=strat[idx_train])

    # Target vocabulary fitted on TRAIN only.  Stratification guarantees every
    # class appears in train, so val/test always transform cleanly.
    le_target = LabelEncoder().fit(y_raw.iloc[idx_train])
    class_names = [str(c) for c in le_target.classes_]

    X_train, X_val, X_test = (X_raw.iloc[i].copy() for i in (idx_train, idx_val, idx_test))
    y_train, y_val, y_test = (le_target.transform(y_raw.iloc[i])
                              for i in (idx_train, idx_val, idx_test))

    # -- 3. fit encoders on TRAIN only, transform the rest ------------------
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if name == "Edge-IIoTSet":
        cat_cols = sorted(set(cat_cols) | (set(EDGE_CATEGORICAL) & set(X_train.columns)))
    elif name == "X-IIoTID":
        # Several dozen object-dtype columns are numeric byte/packet/timing
        # counts that pandas typed as `object` only because of a "-"
        # placeholder for "not applicable"; see `_is_numeric_with_placeholder`.
        # Encoding these as nominal categories (the dtype-only check's
        # default) destroys their numeric ordering, so route them to the
        # to_numeric+fillna(0) path below instead, decided on TRAIN only.
        numeric_placeholder_cols = [c for c in cat_cols
                                    if _is_numeric_with_placeholder(X_train[c])]
        if numeric_placeholder_cols:
            logger.info("X-IIoTID: treating %d dash-placeholder columns as "
                       "numeric rather than categorical: %s",
                       len(numeric_placeholder_cols), numeric_placeholder_cols)
            for col in numeric_placeholder_cols:
                for part in (X_train, X_val, X_test):
                    part[col] = part[col].replace(
                        ["-", "?", "", "nan", "None", "null", "NULL"], np.nan)
        cat_cols = [c for c in cat_cols if c not in numeric_placeholder_cols]
    for col in cat_cols:
        enc = OrdinalEncoderWithUnseen().fit(X_train[col])
        for part in (X_train, X_val, X_test):
            part[col] = enc.transform(part[col])

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    num_cols = X_train.columns.tolist()
    scaler = RobustScaler().fit(X_train[num_cols])          # TRAIN ONLY
    for part in (X_train, X_val, X_test):
        part[num_cols] = scaler.transform(part[num_cols])

    # -- 4. correlation filter decided on TRAIN only ------------------------
    to_drop = _correlated_columns(X_train[num_cols], cfg.corr_threshold)
    if to_drop:
        logger.info("Dropping %d correlated columns (|r|>%.2f, train-only): %s",
                    len(to_drop), cfg.corr_threshold, to_drop[:8])
    X_train, X_val, X_test = (p.drop(columns=to_drop, errors="ignore")
                              for p in (X_train, X_val, X_test))

    feature_names = X_train.columns.tolist()
    splits = Splits(
        X_train=X_train.to_numpy(np.float32), y_train=y_train.astype(np.int64),
        X_val=X_val.to_numpy(np.float32), y_val=y_val.astype(np.int64),
        X_test=X_test.to_numpy(np.float32), y_test=y_test.astype(np.int64),
        feature_names=feature_names, class_names=class_names, dataset_name=name)
    logger.info("%s", splits.summary())
    return splits


# --------------------------------------------------------------------------
# Loaders -- batches move to the GPU per iteration, not in bulk
# --------------------------------------------------------------------------

def make_loaders(splits: Splits, cfg: Config, shuffle_train: bool = True
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders over CPU tensors; ``train.py`` moves each batch to the device."""
    pin = torch.cuda.is_available()

    def _dl(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          pin_memory=pin, num_workers=0, drop_last=False)

    return (_dl(splits.X_train, splits.y_train, shuffle_train),
            _dl(splits.X_val, splits.y_val, False),
            _dl(splits.X_test, splits.y_test, False))


# --------------------------------------------------------------------------
# Federated partitioners -- fed from the SAME split-first pipeline, so the
# leakage fix applies to both centralized and federated experiments at once.
# --------------------------------------------------------------------------

def partition_iid(y: np.ndarray, n_clients: int, seed: int) -> List[np.ndarray]:
    """Shuffle and deal rows evenly across clients."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [np.sort(part) for part in np.array_split(idx, n_clients)]


def partition_dirichlet(y: np.ndarray, n_clients: int, seed: int,
                        alpha: float = 0.5) -> List[np.ndarray]:
    """Label-skewed partition: per class, split rows by ``Dirichlet(alpha)``.

    ``alpha=0.5`` is the setting the paper reports.
    """
    rng = np.random.default_rng(seed)
    parts: List[List[int]] = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(props)[:-1] * len(cls_idx)).astype(int)
        for c, chunk in enumerate(np.split(cls_idx, cuts)):
            parts[c].extend(chunk.tolist())
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


def partition(y: np.ndarray, n_clients: int, seed: int, scheme: str = "iid",
              alpha: float = 0.5) -> List[np.ndarray]:
    """Dispatch to a partitioner.  The paper reports ``iid`` and ``dirichlet``."""
    if scheme == "iid":
        return partition_iid(y, n_clients, seed)
    if scheme == "dirichlet":
        return partition_dirichlet(y, n_clients, seed, alpha)
    raise ValueError(f"Unknown partition scheme {scheme!r} (expected iid|dirichlet)")
