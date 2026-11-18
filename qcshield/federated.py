"""Federated harness -- Flower simulation of the paper's model.

Trains :class:`HybridQuantumCNNBiLSTM` (QE-QCSL, 4 qubits) under FedAvg,
FedProx, FedTrimmedAvg, or FaultTolerantFedAvg, across IID or
Dirichlet(0.5)-partitioned clients. Key correctness properties:

* One global ``LabelEncoder`` fit before partitioning, so the same integer
  label denotes the same attack class on every client regardless of split.
* A genuine reserved test set: only the training split is partitioned across
  clients, and the global test split is never given to any client, so
  server-side evaluation reflects true held-out generalization.
* ``client_resources`` sized from a measured per-client VRAM footprint
  rather than a fixed guess, to avoid over-subscribing the GPU.

Clients run inside Ray actors, which are separate processes, so partitions are
exchanged through an on-disk cache rather than module globals.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, set_seed, get_device
from . import data as D
from .models import build_model
from .train import _build_optimizer, compute_class_weights

logger = logging.getLogger(__name__)

NUM_PARTITIONS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
PARTITIONS = ("iid", "dirichlet")
STRATEGIES = ("FedAvg", "FedProx", "FedTrimmedAvg", "FaultTolerantFedAvg")

#: Measured peak VRAM for one client (UNSW-NB15, Edge-IIoTSet). X-IIoTID's
#: larger feature count (50 vs. 32/38) and class
#: count measured substantially higher per-client VRAM in practice, so it
#: gets its own measured estimate rather than inheriting this one and
#: under-provisioning Ray's per-client GPU share.
CLIENT_VRAM_MB = 600
CLIENT_VRAM_MB_BY_DATASET = {"X-IIoTID": 1536}
#: Env var carrying the partition cache path into the Ray workers.
CACHE_ENV = "QCSHIELD_FED_CACHE"

_LOCAL_CACHE: Dict[str, dict] = {}


# --------------------------------------------------------------------------
# Partition cache -- written by the driver, read by each Ray actor
# --------------------------------------------------------------------------

def build_cache(cfg: Config, splits: D.Splits, partition: str,
                num_partitions: int, out: Path) -> Path:
    """Partition the TRAIN split only and persist everything clients need."""
    parts = D.partition(splits.y_train, num_partitions, cfg.seed, partition)
    payload = {
        "X_train": splits.X_train, "y_train": splits.y_train,
        "X_test": splits.X_test, "y_test": splits.y_test,
        "n_features": splits.n_features, "n_classes": splits.n_classes,
        "batch_size": cfg.batch_size, "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay, "seed": cfg.seed,
        "embedding": cfg.embedding, "layer": cfg.layer,
        "dataset_name": cfg.dataset_name,
    }
    for i, idx in enumerate(parts):
        payload[f"part_{i}"] = idx
    payload["num_partitions"] = num_partitions
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    logger.info("partition cache -> %s (client sizes %s)", out.name,
                [len(p) for p in parts])
    return out


def load_cache(path: Optional[str] = None) -> dict:
    path = path or os.environ.get(CACHE_ENV, "")
    if not path:
        raise RuntimeError(f"{CACHE_ENV} is not set; call build_cache first")
    if path not in _LOCAL_CACHE:
        z = np.load(path, allow_pickle=True)
        _LOCAL_CACHE[path] = {k: z[k] for k in z.files}
    return _LOCAL_CACHE[path]


def _cfg_from_cache(c: dict) -> Config:
    return Config(dataset_name=str(c["dataset_name"]),
                  seed=int(c["seed"]), batch_size=int(c["batch_size"]),
                  learning_rate=float(c["lr"]),
                  weight_decay=float(c["weight_decay"]),
                  embedding=str(c["embedding"]), layer=str(c["layer"]))


def _loader(X: np.ndarray, y: np.ndarray, bs: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                       torch.from_numpy(np.ascontiguousarray(y)))
    return DataLoader(ds, batch_size=int(bs), shuffle=shuffle)


# --------------------------------------------------------------------------
# Parameter plumbing
# --------------------------------------------------------------------------

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in net.state_dict().values()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(net.state_dict().keys())
    net.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, parameters)},
                        strict=True)


def train_local(net, loader, cfg, device, epochs=LOCAL_EPOCHS,
                proximal_mu: float = 0.0, global_ref=None,
                class_weights: Optional[np.ndarray] = None) -> None:
    """One (or more) local epochs; ``proximal_mu>0`` adds the FedProx term.

    Uses the same regime as centralized training (:func:`qcshield.train._build_optimizer`,
    :func:`qcshield.train.compute_class_weights`) -- AdamW with quantum parameters
    in their own 5x-lr param group, and inverse-frequency class-weighted loss --
    rather than a separate, weaker local implementation. Both datasets are
    heavily imbalanced, and rare classes are exactly what a poisoning/robustness
    comparison across strategies most needs represented in the gradient.
    """
    net.to(device).train()
    opt = _build_optimizer(net, cfg)
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=weight)
    ref = None
    if proximal_mu > 0 and global_ref is not None:
        ref = [torch.tensor(g, device=device, dtype=torch.float32)
               for g in global_ref]
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xb), yb)
            if ref is not None:
                prox = sum(((p - r) ** 2).sum()
                           for p, r in zip(net.parameters(), ref)
                           if p.shape == r.shape)
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            opt.step()


@torch.no_grad()
def test_model(net, loader, device,
               class_weights: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """(loss, accuracy, macro-F1).

    ``class_weights``, when given, matches the reported loss to the same
    weighted objective the model was actually trained against (as centralized
    evaluation already does in :func:`qcshield.train._run_epoch`) -- accuracy
    and macro-F1 are unaffected either way.
    """
    from sklearn.metrics import accuracy_score, f1_score
    net.to(device).eval()
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=weight)
    tot, n, preds, trues = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = net(xb)
        tot += lossf(out, yb).item() * yb.size(0); n += yb.size(0)
        preds.append(out.argmax(1).cpu()); trues.append(yb.cpu())
    yp, yt = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return (tot / max(n, 1), float(accuracy_score(yt, yp)),
            float(f1_score(yt, yp, average="macro", zero_division=0)))


# --------------------------------------------------------------------------
# Flower client
# --------------------------------------------------------------------------

def _build_net(c: dict, cfg: Config, device):
    return build_model(cfg, int(c["n_features"]), int(c["n_classes"]),
                       "quantum", cfg.embedding, cfg.layer).to(device)


def make_client_fn():
    from flwr.client import NumPyClient
    from flwr.common import Context

    class FlowerClient(NumPyClient):
        def __init__(self, pid: int):
            c = load_cache()
            self.pid = pid
            self.cfg = _cfg_from_cache(c)
            self.device = get_device()
            idx = c[f"part_{pid}"]
            # local holdout for the per-client curves; never the global test set
            n_hold = max(1, int(0.2 * len(idx)))
            hold, tr = idx[:n_hold], idx[n_hold:]
            bs = int(c["batch_size"])
            self.train_loader = _loader(c["X_train"][tr], c["y_train"][tr], bs, True)
            self.test_loader = _loader(c["X_train"][hold], c["y_train"][hold], bs, False)
            self.net = _build_net(c, self.cfg, self.device)
            self.class_weights = compute_class_weights(c["y_train"][tr], int(c["n_classes"]))

        def get_parameters(self, config):
            return get_parameters(self.net)

        def fit(self, parameters, config):
            set_parameters(self.net, parameters)
            mu = float(config.get("proximal_mu", 0.0))
            train_local(self.net, self.train_loader, self.cfg, self.device,
                        proximal_mu=mu, global_ref=parameters if mu > 0 else None,
                        class_weights=self.class_weights)
            return get_parameters(self.net), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            set_parameters(self.net, parameters)
            loss, acc, f1 = test_model(self.net, self.test_loader, self.device,
                                       class_weights=self.class_weights)
            return float(loss), len(self.test_loader.dataset), {
                "loss": float(loss), "accuracy": float(acc), "f1_macro": float(f1),
                "client_id": int(self.pid),
                "server_round": int(config.get("server_round", 0)),
                "strategy": str(config.get("strategy", "")),
                "partition_type": str(config.get("partition_type", "")),
            }

    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        return FlowerClient(pid).to_client()

    return client_fn


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class FedResult:
    strategy: str
    partition: str
    dataset: str
    rounds: List[int] = field(default_factory=list)
    global_loss: List[float] = field(default_factory=list)
    global_acc: List[float] = field(default_factory=list)
    global_f1: List[float] = field(default_factory=list)
    client_rows: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "partition": self.partition,
                "dataset": self.dataset, "rounds": self.rounds,
                "global_loss": self.global_loss, "global_acc": self.global_acc,
                "global_f1": self.global_f1, "client_rows": self.client_rows}


def run_simulation(cfg: Config, strategy_name: str, partition: str,
                   splits: Optional[D.Splits] = None,
                   num_partitions: int = NUM_PARTITIONS,
                   num_rounds: int = NUM_ROUNDS,
                   cache_dir: Path = Path("data")) -> FedResult:
    """One (strategy, partition) Flower simulation."""
    import flwr as fl
    from flwr.client import ClientApp
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.common import ndarrays_to_parameters
    from flwr.server.strategy import (FedAvg, FedProx, FedTrimmedAvg,
                                      FaultTolerantFedAvg)

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if strategy_name not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    set_seed(cfg.seed)
    device = get_device()
    splits = splits if splits is not None else D.prepare(cfg)

    cache = build_cache(cfg, splits, partition, num_partitions,
                        Path(cache_dir) / f"fedcache_{cfg.dataset_name.replace('-','_')}"
                                          f"_{partition}_seed{cfg.seed}.npz")
    os.environ[CACHE_ENV] = str(cache.resolve())

    result = FedResult(strategy=strategy_name, partition=partition,
                       dataset=cfg.dataset_name)

    # --- server-side evaluation on the RESERVED global test set -----------
    global_loader = _loader(splits.X_test, splits.y_test, cfg.batch_size, False)
    eval_net = build_model(cfg, splits.n_features, splits.n_classes,
                           "quantum", cfg.embedding, cfg.layer).to(device)
    global_class_weights = compute_class_weights(splits.y_train, splits.n_classes)

    def evaluate_fn(server_round, parameters, config):
        set_parameters(eval_net, parameters)
        loss, acc, f1 = test_model(eval_net, global_loader, device,
                                   class_weights=global_class_weights)
        if server_round > 0:
            result.rounds.append(int(server_round))
            result.global_loss.append(loss)
            result.global_acc.append(acc)
            result.global_f1.append(f1)
        logger.info("[%s/%s] round %2d/%d  GLOBAL loss=%.4f acc=%.4f f1=%.4f",
                    strategy_name, partition, server_round, num_rounds,
                    loss, acc, f1)
        return float(loss), {"accuracy": acc, "f1_macro": f1}

    def agg_eval_metrics(metrics):
        total = sum(n for n, _ in metrics) or 1
        for n, m in metrics:
            result.client_rows.append({
                "round": int(m.get("server_round", 0)),
                "client": int(m.get("client_id", -1)),
                "loss": float(m.get("loss", 0.0)),
                "accuracy": float(m.get("accuracy", 0.0)),
                "f1_macro": float(m.get("f1_macro", 0.0)), "n": int(n)})
        return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total,
                "f1_macro": sum(n * m["f1_macro"] for n, m in metrics) / total}

    def cfg_fn(server_round: int):
        d = {"server_round": server_round, "partition_type": partition,
             "strategy": strategy_name}
        if strategy_name == "FedProx":
            d["proximal_mu"] = 0.1
        return d

    init = build_model(cfg, splits.n_features, splits.n_classes,
                       "quantum", cfg.embedding, cfg.layer)
    common = dict(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=5, min_evaluate_clients=5,
        min_available_clients=num_partitions,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=cfg_fn, on_evaluate_config_fn=cfg_fn,
        evaluate_metrics_aggregation_fn=agg_eval_metrics,
        initial_parameters=ndarrays_to_parameters(get_parameters(init)))

    strategies = {
        "FedAvg": lambda: FedAvg(**common),
        "FedProx": lambda: FedProx(**common, proximal_mu=0.1),
        "FedTrimmedAvg": lambda: FedTrimmedAvg(**common, beta=0.2),
        "FaultTolerantFedAvg": lambda: FaultTolerantFedAvg(
            **common, min_completion_rate_fit=0.5,
            min_completion_rate_evaluate=0.5),
    }

    def server_fn(context):
        return ServerAppComponents(strategy=strategies[strategy_name](),
                                   config=ServerConfig(num_rounds=num_rounds))

    # client_resources from the measured footprint, not a guess
    backend: Dict[str, Dict] = {"client_resources": {"num_cpus": 2}}
    if device.type == "cuda":
        client_vram_mb = CLIENT_VRAM_MB_BY_DATASET.get(cfg.dataset_name, CLIENT_VRAM_MB)
        free_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        share = min(0.5, max(0.05, client_vram_mb / free_mb * 1.5))
        backend["client_resources"]["num_gpus"] = round(share, 3)
        logger.info("client_resources: num_gpus=%.3f (%.0f MB/client of %.0f MB)",
                    share, client_vram_mb, free_mb)

    fl.simulation.run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=make_client_fn()),
        num_supernodes=num_partitions,
        backend_config=backend)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_all(cfg: Config, out_dir: Path, num_rounds: int = NUM_ROUNDS
            ) -> List[FedResult]:
    """Every strategy x partition combination the paper reports."""
    splits = D.prepare(cfg)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for partition in PARTITIONS:
        for strategy in STRATEGIES:
            r = run_simulation(cfg, strategy, partition, splits,
                               num_rounds=num_rounds)
            results.append(r)
            fn = out_dir / (f"fed_{strategy}_{partition}_"
                            f"{cfg.dataset_name.replace('-','_')}.json")
            fn.write_text(json.dumps(r.to_dict(), indent=2))
            logger.info("wrote %s", fn.name)
    return results
