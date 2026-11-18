"""Training and evaluation.

Replaces ``Trainer`` from the mega-cells.  Differences that matter:

* **Batches move to the device inside the loop.**  The original uploaded the
  entire train and test matrices to the GPU in ``DataProcessor`` before training
  started; on Edge-IIoTSet that is millions of rows resident in VRAM for the
  whole run.
* **A validation split exists.**  The original selected nothing on validation
  and simply reported the final epoch; the best checkpoint is now chosen by
  validation macro-F1 and restored before the test evaluation.
* **Macro-F1 is reported alongside accuracy.**  Both datasets are heavily
  imbalanced -- UNSW-NB15 is 53% ``Normal``, Edge-IIoTSet 71% ``Normal`` -- so
  accuracy alone flatters every model. Reviewer 3 §1 asked for exactly this.
* **Runs are deterministic** given a seed (Reviewer 4 §8).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

from .config import Config, get_device

logger = logging.getLogger(__name__)


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    epoch_seconds: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    roc_auc_ovr: Optional[float]
    per_class_f1: List[float]
    confusion: List[List[int]]
    n_samples: int
    #: Per-class ROC and PR curves, resampled to a fixed point count so the
    #: JSON stays plot-ready without embedding one point per test sample
    #: (Edge-IIoTSet's 381,621-row test set would otherwise balloon the file).
    #: ``None`` for a class absent from ``y_true`` (undefined ROC/PR).
    roc_curves: Optional[Dict[str, Optional[Dict[str, object]]]] = None
    pr_curves: Optional[Dict[str, Optional[Dict[str, object]]]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"acc={self.accuracy:.4f}  f1_macro={self.f1_macro:.4f}  "
                f"f1_weighted={self.f1_weighted:.4f}")


def _run_epoch(model, loader, device, lossf, optimiser=None) -> tuple:
    """One pass. ``optimiser=None`` means evaluation."""
    training = optimiser is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds, trues = [], []

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if training:
                optimiser.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = lossf(logits, yb)
            if training:
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
            preds.append(logits.detach().argmax(1).cpu())
            trues.append(yb.detach().cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                         zero_division=0)[2]
    return total_loss / max(n, 1), acc, f1


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency class weights for ``CrossEntropyLoss``.

    Both datasets are heavily imbalanced (UNSW-NB15 53% ``Normal``,
    Edge-IIoTSet 71% ``Normal``); rare classes like ``Worms``/``Backdoor``
    otherwise contribute almost nothing to the gradient. Weights are
    ``n_samples / (n_classes * count[c])``, the standard inverse-frequency
    scheme (equivalent to sklearn's ``class_weight='balanced'``), normalised
    so the mean weight is 1 -- this keeps the loss scale comparable to the
    unweighted case, so ``learning_rate`` does not need retuning.
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)          # guard classes absent from train
    weights = len(y_train) / (n_classes * counts)
    return (weights / weights.mean()).astype(np.float32)


def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW (or Adam) with quantum parameters in their own param group.

    ``qlayer`` parameters (the PQC's rotation angles) get
    ``learning_rate * quantum_lr_multiplier`` -- see the field's docstring in
    ``config.py``. Classical-only baselines have no ``qlayer`` submodule, so
    this reduces to a single param group for them; the special-casing never
    fires, it is simply inapplicable.
    """
    quantum_params, classical_params = [], []
    for name, p in model.named_parameters():
        (quantum_params if name.startswith("qlayer") else classical_params).append(p)

    groups = [{"params": classical_params, "lr": cfg.learning_rate}]
    if quantum_params:
        groups.append({"params": quantum_params,
                       "lr": cfg.learning_rate * cfg.quantum_lr_multiplier})

    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    return opt_cls(groups, weight_decay=cfg.weight_decay)


def train_model(model: nn.Module, train_loader, val_loader, cfg: Config,
                device=None, class_weights: Optional[np.ndarray] = None
                ) -> tuple[nn.Module, History]:
    """Train, selecting the checkpoint with the best validation macro-F1.

    No early stopping: every configuration trains for the full ``cfg.epochs``
    regardless of validation trend, so no run is ever cut short before the
    others.
    """
    device = device or get_device()
    model = model.to(device)

    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=weight)
    optimiser = _build_optimizer(model, cfg)

    scheduler = None
    if cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience)

    hist = History()
    best_f1, best_state, best_epoch = -1.0, None, -1

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        tl, ta, tf = _run_epoch(model, train_loader, device, lossf, optimiser)
        vl, va, vf = _run_epoch(model, val_loader, device, lossf, None)
        dt = time.perf_counter() - t0
        if scheduler is not None:
            scheduler.step(vf)

        hist.train_loss.append(tl); hist.val_loss.append(vl)
        hist.train_acc.append(ta);  hist.val_acc.append(va)
        hist.train_f1.append(tf);   hist.val_f1.append(vf)
        hist.epoch_seconds.append(dt)

        if vf > best_f1:
            best_f1, best_epoch = vf, epoch
            best_state = copy.deepcopy(model.state_dict())

        logger.info("epoch %2d/%d  train_loss=%.4f acc=%.4f f1=%.4f | "
                    "val_loss=%.4f acc=%.4f f1=%.4f | %.1fs",
                    epoch + 1, cfg.epochs, tl, ta, tf, vl, va, vf, dt)

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best checkpoint from epoch %d (val f1=%.4f)",
                    best_epoch + 1, best_f1)
    return model, hist


def _resample_curve(x: np.ndarray, y: np.ndarray, n_points: int = 60) -> tuple:
    """Resample a curve to ``n_points`` on a fixed grid over ``x``, monotonic."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    grid = np.linspace(0.0, 1.0, n_points)
    return grid.tolist(), np.interp(grid, x, y).tolist()


def _class_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names,
                  n_points: int = 60) -> tuple:
    """Per-class ROC and PR curves, resampled and keyed by class name.

    Also adds ``"micro"`` (pool every class's true/false labels together --
    dominated by the frequent classes) and ``"macro"`` (unweighted mean of the
    per-class curves -- treats every class equally regardless of support) under
    those two reserved keys, alongside the per-class ones. Computed here, where
    the raw per-sample ``y_true``/``y_proba`` are still in memory, then
    resampled to ``n_points`` like everything else -- the raw arrays themselves
    are never stored (Edge-IIoTSet's 381,621-row test set would balloon the
    JSON otherwise).
    """
    n_classes = len(class_names)
    y_bin_all = np.eye(n_classes, dtype=int)[y_true]     # one-hot, (n_samples, n_classes)

    roc: Dict[str, Optional[Dict[str, object]]] = {}
    pr: Dict[str, Optional[Dict[str, object]]] = {}
    valid = []
    for i, name in enumerate(class_names):
        y_bin = y_bin_all[:, i]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            roc[name] = None       # ROC/AUC undefined -- class absent from y_true
            pr[name] = None
            continue
        valid.append(i)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        auc = float(roc_auc_score(y_bin, y_proba[:, i]))
        gx, gy = _resample_curve(fpr, tpr, n_points)
        roc[name] = {"fpr": gx, "tpr": gy, "auc": auc}

        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap = float(average_precision_score(y_bin, y_proba[:, i]))
        # precision_recall_curve returns recall descending; resample on recall
        gx2, gy2 = _resample_curve(rec, prec, n_points)
        pr[name] = {"recall": gx2, "precision": gy2, "ap": ap}

    if valid:
        # micro: pool every valid class's labels/scores into one binary problem
        y_bin_valid = y_bin_all[:, valid].ravel()
        y_proba_valid = y_proba[:, valid].ravel()
        fpr_mi, tpr_mi, _ = roc_curve(y_bin_valid, y_proba_valid)
        auc_mi = float(roc_auc_score(y_bin_valid, y_proba_valid))
        gx, gy = _resample_curve(fpr_mi, tpr_mi, n_points)
        roc["micro"] = {"fpr": gx, "tpr": gy, "auc": auc_mi}

        prec_mi, rec_mi, _ = precision_recall_curve(y_bin_valid, y_proba_valid)
        ap_mi = float(average_precision_score(y_bin_valid, y_proba_valid))
        gx2, gy2 = _resample_curve(rec_mi, prec_mi, n_points)
        pr["micro"] = {"recall": gx2, "precision": gy2, "ap": ap_mi}

        # macro: average the (already-resampled, common-grid) per-class curves
        roc_tprs = np.stack([roc[class_names[i]]["tpr"] for i in valid])
        roc["macro"] = {"fpr": roc[class_names[valid[0]]]["fpr"],
                        "tpr": roc_tprs.mean(axis=0).tolist(),
                        "auc": float(np.mean([roc[class_names[i]]["auc"] for i in valid]))}
        pr_precs = np.stack([pr[class_names[i]]["precision"] for i in valid])
        pr["macro"] = {"recall": pr[class_names[valid[0]]]["recall"],
                       "precision": pr_precs.mean(axis=0).tolist(),
                       "ap": float(np.mean([pr[class_names[i]]["ap"] for i in valid]))}
    else:
        roc["micro"] = roc["macro"] = pr["micro"] = pr["macro"] = None

    return roc, pr


@torch.no_grad()
def evaluate(model: nn.Module, loader, n_classes: int, device=None,
            class_names: Optional[List[str]] = None,
            with_curves: bool = True
            ) -> tuple[EvalResult, np.ndarray, np.ndarray, np.ndarray]:
    """Full metric set.  Returns ``(result, y_true, y_pred, y_proba)``.

    ``with_curves`` computes per-class ROC/PR curves (needed for the combined
    confusion-matrix/ROC/PR figure); set ``False`` to skip the extra cost.
    """
    device = device or get_device()
    model = model.to(device).eval()

    logits_all, trues = [], []
    for xb, yb in loader:
        logits_all.append(model(xb.to(device, non_blocking=True)).cpu())
        trues.append(yb)

    logits = torch.cat(logits_all)
    y_true = torch.cat(trues).numpy()
    y_proba = torch.softmax(logits, dim=1).numpy()
    y_pred = y_proba.argmax(1)

    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)[2]

    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                  average="macro", labels=list(range(n_classes))))
    except Exception as e:      # only when a class is absent from y_true
        logger.warning("ROC-AUC unavailable: %s", e)
        auc = None

    roc_curves = pr_curves = None
    if with_curves:
        names = class_names or [str(i) for i in range(n_classes)]
        roc_curves, pr_curves = _class_curves(y_true, y_proba, names)

    res = EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(pm), recall_macro=float(rm), f1_macro=float(fm),
        precision_weighted=float(pw), recall_weighted=float(rw),
        f1_weighted=float(fw), roc_auc_ovr=auc,
        per_class_f1=[float(v) for v in per_class],
        confusion=confusion_matrix(y_true, y_pred,
                                   labels=list(range(n_classes))).tolist(),
        n_samples=int(len(y_true)), roc_curves=roc_curves, pr_curves=pr_curves)
    return res, y_true, y_pred, y_proba
