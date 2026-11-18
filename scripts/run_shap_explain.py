#!/usr/bin/env python
"""Per-sample SHAP explainability for QE-QCSL, full scale.

Trains QE-QCSL at the same recipe as every other headline result (full
dataset, seed 0, 30 epochs, class-weighted loss, AdamW with the quantum
parameter group) and explains a class-stratified sample of the test set.

KernelExplainer cost is roughly linear in background_size * nsamples per
explained instance. At background=100, nsamples=200, budget roughly 200
instances * ~18s ~= an hour per dataset.

Usage:
    PYTHONPATH=. ./.venv/bin/python scripts/run_shap_explain.py --dataset Edge-IIoTSet
    PYTHONPATH=. ./.venv/bin/python scripts/run_shap_explain.py --dataset UNSW-NB15
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcshield.config import Config, set_seed, get_device
from qcshield.cli import get_splits
from qcshield import data as D
from qcshield.models import build_model
from qcshield.train import train_model, compute_class_weights
from qcshield import style

logger = logging.getLogger("shap_explain")


def to_per_class_list(shap_values, n_classes: int) -> list:
    """Normalize whatever shape this SHAP version's KernelExplainer.shap_values()
    returns into a canonical list of length n_classes, each (n_samples, n_features).

    SHAP >=0.45's KernelExplainer returns one ndarray (n_samples, n_features,
    n_classes) for multiclass output; older versions returned a list of
    n_classes arrays each (n_samples, n_features). Handle both.
    """
    if isinstance(shap_values, (list, tuple)):
        return list(shap_values)
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] == n_classes:
        return [arr[:, :, c] for c in range(n_classes)]
    if arr.ndim == 3 and arr.shape[0] == n_classes:
        return [arr[c] for c in range(n_classes)]
    raise ValueError(f"unrecognized shap_values shape {arr.shape} for n_classes={n_classes}")


def make_predict_fn(model, device, batch_size=256):
    def predict(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
                out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)
    return predict


def stratified_explain_indices(y_test: np.ndarray, n_classes: int,
                                per_class: int, seed: int) -> np.ndarray:
    """Up to `per_class` test rows per class, capped by that class's support."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_classes):
        rows = np.flatnonzero(y_test == c)
        if len(rows) == 0:
            continue
        take = min(per_class, len(rows))
        idx.append(rng.choice(rows, size=take, replace=False))
    return np.concatenate(idx)


def train_qe_qcsl(cfg: Config, splits):
    set_seed(cfg.seed)
    device = get_device()
    train_loader, val_loader, _ = D.make_loaders(splits, cfg)
    model = build_model(cfg, splits.n_features, splits.n_classes,
                        "quantum", cfg.embedding, cfg.layer)
    class_weights = compute_class_weights(splits.y_train, splits.n_classes) if cfg.class_weighted else None
    t0 = time.perf_counter()
    model, hist = train_model(model, train_loader, val_loader, cfg, device,
                              class_weights=class_weights)
    logger.info("trained QE-QCSL on %s in %.0fs", cfg.dataset_name, time.perf_counter() - t0)
    return model, device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["Edge-IIoTSet", "UNSW-NB15", "X-IIoTID"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-background", type=int, default=100)
    p.add_argument("--per-class", type=int, default=10,
                   help="explained test samples per class (capped by class support)")
    p.add_argument("--nsamples", type=int, default=200)
    p.add_argument("--out-dir", default="results_shap")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import shap  # deferred: heavy import, only needed here

    cfg = Config(dataset_name=a.dataset, seed=a.seed)
    splits = get_splits(cfg)
    model, device = train_qe_qcsl(cfg, splits)

    rng = np.random.default_rng(a.seed)
    bg_idx = rng.choice(len(splits.X_train), size=min(a.n_background, len(splits.X_train)), replace=False)
    background = splits.X_train[bg_idx]

    ex_idx = stratified_explain_indices(splits.y_test, splits.n_classes, a.per_class, a.seed)
    X_explain = splits.X_test[ex_idx]
    y_explain = splits.y_test[ex_idx]
    logger.info("explaining %d samples (%d classes, up to %d/class) with %d background, nsamples=%d",
               len(X_explain), splits.n_classes, a.per_class, a.n_background, a.nsamples)

    predict_fn = make_predict_fn(model, device)
    explainer = shap.KernelExplainer(predict_fn, background)
    t0 = time.perf_counter()
    raw_shap_values = explainer.shap_values(X_explain, nsamples=a.nsamples, silent=False)
    logger.info("SHAP computed in %.0fs (%.1fs/instance)",
               time.perf_counter() - t0, (time.perf_counter() - t0) / len(X_explain))
    shap_values = to_per_class_list(raw_shap_values, splits.n_classes)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.dataset.replace("-", "_")

    # mean |SHAP| per feature, averaged over classes -- global ranking
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    ranking = sorted(zip(splits.feature_names, mean_abs.tolist()), key=lambda t: -t[1])

    # saved as a canonical (n_classes, n_samples, n_features) stack regardless
    # of what shape this SHAP version originally returned
    np.savez_compressed(
        out_dir / f"shap_{tag}_seed{a.seed}.npz",
        shap_values=np.stack(shap_values, axis=0), X_explain=X_explain, y_explain=y_explain,
        feature_names=np.array(splits.feature_names), class_names=np.array(splits.class_names))
    (out_dir / f"shap_{tag}_seed{a.seed}_ranking.json").write_text(
        json.dumps({"dataset": a.dataset, "feature_ranking": ranking,
                   "n_background": a.n_background, "n_explain": len(X_explain),
                   "nsamples": a.nsamples}, indent=2))
    logger.info("wrote %s", out_dir / f"shap_{tag}_seed{a.seed}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
