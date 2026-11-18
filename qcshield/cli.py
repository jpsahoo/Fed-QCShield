"""Command-line driver.

    python -m qcshield.cli run    --dataset UNSW-NB15 --epochs 2 --seed 0
    python -m qcshield.cli grid   --dataset UNSW-NB15 --seeds 0,1,2,3,4
    python -m qcshield.cli stats  --dataset UNSW-NB15
    python -m qcshield.cli circuits
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import (Config, SEEDS, all_quantum_configs, CLASSICAL_BASELINES,
                     set_seed, get_device, DATA_ROOT)
from . import data as D
from .models import build_model, count_parameters
from .train import train_model, evaluate, compute_class_weights

logger = logging.getLogger("qcshield")


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")


def get_splits(cfg: Config, cache: bool = True) -> D.Splits:
    """Prepared splits, cached per (dataset, seed, subsample) on disk."""
    tag = f"{cfg.dataset_name.replace('-', '_')}_seed{cfg.seed}"
    if cfg.subsample:
        tag += f"_sub{cfg.subsample}"
    path = DATA_ROOT / f"splits_{tag}.npz"

    if cache and path.exists():
        z = np.load(path, allow_pickle=True)
        logger.info("loaded cached splits %s", path.name)
        return D.Splits(
            X_train=z["X_train"], y_train=z["y_train"],
            X_val=z["X_val"], y_val=z["y_val"],
            X_test=z["X_test"], y_test=z["y_test"],
            feature_names=list(z["feature_names"]),
            class_names=list(z["class_names"]),
            dataset_name=cfg.dataset_name)

    splits = D.prepare(cfg)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, X_train=splits.X_train, y_train=splits.y_train,
            X_val=splits.X_val, y_val=splits.y_val,
            X_test=splits.X_test, y_test=splits.y_test,
            feature_names=np.array(splits.feature_names),
            class_names=np.array(splits.class_names))
        logger.info("cached splits -> %s", path.name)
    return splits


def run_one(cfg: Config, model_name: str = "quantum", splits=None,
            save: bool = True) -> dict:
    """Train and evaluate one configuration at one seed."""
    set_seed(cfg.seed)
    device = get_device()
    splits = splits if splits is not None else get_splits(cfg)
    train_loader, val_loader, test_loader = D.make_loaders(splits, cfg)

    model = build_model(cfg, splits.n_features, splits.n_classes,
                        model_name, cfg.embedding, cfg.layer)
    params = count_parameters(model)
    label = (f"{cfg.template_name}" if model_name == "quantum" else model_name)
    logger.info("=== %s | %s | seed %d | %s params (%s quantum) | device %s ===",
                cfg.dataset_name, label, cfg.seed,
                f"{params['total']:,}", f"{params['quantum']:,}", device)

    class_weights = None
    if cfg.class_weighted:
        class_weights = compute_class_weights(splits.y_train, splits.n_classes)

    t0 = time.perf_counter()
    model, hist = train_model(model, train_loader, val_loader, cfg, device,
                              class_weights=class_weights)
    train_seconds = time.perf_counter() - t0

    res, y_true, y_pred, y_proba = evaluate(model, test_loader,
                                            splits.n_classes, device,
                                            class_names=splits.class_names)
    logger.info("TEST %s | %s", label, res.summary())

    record = {
        "dataset": cfg.dataset_name, "model": label, "model_name": model_name,
        "embedding": cfg.embedding if model_name == "quantum" else None,
        "layer": cfg.layer if model_name == "quantum" else None,
        "seed": cfg.seed, "epochs": cfg.epochs,
        "n_features": splits.n_features, "n_classes": splits.n_classes,
        "class_names": splits.class_names,
        "params": params, "train_seconds": train_seconds,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20
                         if device.type == "cuda" else 0.0),
        "config": cfg.to_dict(), "test": res.to_dict(),
        "history": hist.to_dict(),
    }

    if save:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        fn = (f"{cfg.dataset_name.replace('-', '_')}_{label.replace('-', '_')}"
              f"_seed{cfg.seed}.json")
        (cfg.results_dir / fn).write_text(json.dumps(record, indent=2))
        logger.info("wrote results/%s", fn)
    return record


def cmd_run(a) -> int:
    cfg = Config(dataset_name=a.dataset, seed=a.seed, epochs=a.epochs,
                 batch_size=a.batch_size, embedding=a.embedding, layer=a.layer,
                 subsample=a.subsample, class_weighted=not a.no_class_weight)
    run_one(cfg, a.model)
    return 0


def cmd_grid(a) -> int:
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS)
    models = ([("quantum", e, l) for e, l in all_quantum_configs()]
              + [(b, None, None) for b in CLASSICAL_BASELINES])
    if a.only:
        keep = set(a.only.split(","))
        models = [m for m in models
                  if (f"{m[1]}_{m[2]}" if m[0] == "quantum" else m[0]) in keep]

    total, done = len(models) * len(seeds), 0
    logger.info("grid: %d model(s) x %d seed(s) = %d runs", len(models), len(seeds), total)
    class_weighted = not a.no_class_weight
    for seed in seeds:
        base = Config(dataset_name=a.dataset, seed=seed, epochs=a.epochs,
                      batch_size=a.batch_size, subsample=a.subsample,
                      class_weighted=class_weighted)
        splits = get_splits(base)
        for name, emb, lay in models:
            cfg = Config(dataset_name=a.dataset, seed=seed, epochs=a.epochs,
                         batch_size=a.batch_size, subsample=a.subsample,
                         embedding=emb or base.embedding, layer=lay or base.layer,
                         class_weighted=class_weighted)
            run_one(cfg, name, splits=splits)
            done += 1
            logger.info("--- grid progress %d/%d ---", done, total)
    return 0


def cmd_stats(a) -> int:
    from .stats import aggregate, write_latex_table
    df = aggregate(Path("results"), a.dataset)
    if df.empty:
        logger.error("no result JSONs found in results/ for %s", a.dataset)
        return 1
    out = Path("results") / f"summary_{a.dataset.replace('-', '_')}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    write_latex_table(df, Path("results") /
                      f"table_{a.dataset.replace('-', '_')}.tex", a.dataset)
    return 0


def cmd_circuits(a) -> int:
    from .circuit_stats import analyze_all, to_dataframe
    from .latex_circuit import write_all_circuits, compile_circuit_pdfs
    from .plots import (qcshield_distribution,
                        qcshield_layer_distribution, operation_distribution_grid)
    cfg = Config()
    rows = analyze_all(cfg)
    df = to_dataframe(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/circuit_stats_full.csv", index=False)
    print(df.to_string(index=False))
    n = write_all_circuits(cfg, Path("results") / "quantikz")
    logger.info("wrote %d quantikz fragments to results/quantikz/", n)
    m = compile_circuit_pdfs(cfg, Path("results") / "quantikz",
                             Path("figures") / "CH3" / "Plots" / "Circuits")
    logger.info("compiled %d circuit-diagram PDFs", m)
    for ds in ("UNSW-NB15", "Edge-IIoTSet"):
        pg = operation_distribution_grid(rows, ds)
        logger.info("figure %s", pg)
    p1 = qcshield_distribution(rows)
    logger.info("figure %s", p1)
    p2 = qcshield_layer_distribution(rows)
    logger.info("figure %s", p2)
    return 0


def cmd_fed(a) -> int:
    from .federated import run_all, run_simulation, PARTITIONS, STRATEGIES
    from .plots import all_federated_figures
    cfg = Config(dataset_name=a.dataset, seed=a.seed, batch_size=a.batch_size,
                 subsample=a.subsample)
    out = Path("results") / "fed"
    if a.strategy or a.partition:
        strategies = [a.strategy] if a.strategy else list(STRATEGIES)
        partitions = [a.partition] if a.partition else list(PARTITIONS)
        splits = get_splits(cfg)
        results = []
        out.mkdir(parents=True, exist_ok=True)
        for part in partitions:
            for strat in strategies:
                r = run_simulation(cfg, strat, part, splits, num_rounds=a.rounds)
                results.append(r)
                fn = out / (f"fed_{strat}_{part}_"
                            f"{cfg.dataset_name.replace('-', '_')}.json")
                fn.write_text(json.dumps(r.to_dict(), indent=2))
    else:
        results = run_all(cfg, out, num_rounds=a.rounds)
    if not a.no_figures:
        for p in all_federated_figures(results, cfg.dataset_name):
            logger.info("figure %s", p)
    return 0


def cmd_figures(a) -> int:
    """Regenerate the centralized figures from the saved result JSONs."""
    from .stats import aggregate
    from .circuit_stats import analyze_all
    from . import plots

    res_dir = Path("results")
    made = []
    for fp in sorted(res_dir.glob("*.json")):
        r = json.loads(fp.read_text())
        if r.get("dataset") != a.dataset or r.get("seed") != a.seed:
            continue
        if "QCSL" not in str(r.get("model", "")):
            continue                       # the .tex only shows the QCSL pair
        made.append(plots.learning_curves(r["history"], r["dataset"], r["model"]))
        made.append(plots.evaluation_metrics(
            r["test"]["confusion"], r["test"]["per_class_f1"],
            r["class_names"], r["dataset"], r["model"],
            roc_curves=r["test"].get("roc_curves"),
            pr_curves=r["test"].get("pr_curves")))

    agg = aggregate(res_dir, a.dataset)
    if not agg.empty:
        made.append(plots.model_comparison(agg, a.dataset))
    made.append(plots.operation_distribution_grid(analyze_all(Config()), a.dataset))
    for p in made:
        logger.info("figure %s", p)
    logger.info("%d figures written", len(made))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="qcshield")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="UNSW-NB15",
                        choices=["UNSW-NB15", "Edge-IIoTSet", "X-IIoTID"])
        sp.add_argument("--epochs", type=int, default=Config.epochs)
        sp.add_argument("--batch-size", type=int, default=Config.batch_size)
        sp.add_argument("--subsample", type=int, default=None)
        sp.add_argument("--no-class-weight", action="store_true",
                        help="disable inverse-frequency class weighting (default: on)")

    r = sub.add_parser("run", help="one model, one seed")
    common(r)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default="quantum")
    r.add_argument("--embedding", default=Config.embedding, choices=["AE", "QE"])
    r.add_argument("--layer", default=Config.layer,
                   choices=["BEL", "SEL", "RL", "QCSL"])
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("grid", help="all models x all seeds")
    common(g)
    g.add_argument("--seeds", default=None, help="comma-separated (default 0,1,2,3,4)")
    g.add_argument("--only", default=None, help="comma-separated model labels")
    g.set_defaults(func=cmd_grid)

    s = sub.add_parser("stats", help="aggregate results over seeds")
    s.add_argument("--dataset", default="UNSW-NB15")
    s.set_defaults(func=cmd_stats)

    c = sub.add_parser("circuits", help="circuit statistics + quantikz")
    c.set_defaults(func=cmd_circuits)

    f = sub.add_parser("fed", help="federated Flower simulations")
    common(f)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--rounds", type=int, default=10)
    f.add_argument("--strategy", default=None)
    f.add_argument("--partition", default=None, choices=["iid", "dirichlet"])
    f.add_argument("--no-figures", action="store_true")
    f.set_defaults(func=cmd_fed)

    fg = sub.add_parser("figures", help="regenerate centralized figures")
    fg.add_argument("--dataset", default="UNSW-NB15")
    fg.add_argument("--seed", type=int, default=0)
    fg.set_defaults(func=cmd_figures)

    a = p.parse_args(argv)
    _setup_logging()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
