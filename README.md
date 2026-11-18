# Fed-QCShield

Federated quantum-classical intrusion detection for Edge-IIoT networks. A
QAOA-inspired parameterized quantum circuit (the **QCShieldLayer**) combined
with a classical 1D-CNN + BiLSTM backbone, evaluated centrally and under
federated learning (Flower) on three network-traffic intrusion-detection
datasets: **UNSW-NB15**, **Edge-IIoTSet**, and **X-IIoTID**.

This repository accompanies the manuscript *"Fed-QCShield: A Federated
Learning Framework for Quantum-enhanced IDS"* (`Fed-QCShield_V9.tex` in the
companion `Latex_V9` repository).

## Honest headline result

Under a leakage-free, class-weighted, 5-seed statistical comparison, the
proposed quantum layer (QE-QCSL) does **not** demonstrate a detection-accuracy
advantage over a parameter-matched classical CNN-BiLSTM control on any of the
three datasets. This repository's contribution is the reproducible,
leakage-free, federated evaluation framework itself -- not a claimed quantum
accuracy edge. See the `paper_results_<timestamp>/` snapshot described below,
and the manuscript's Table 3, for the full, unedited numbers.

## X-IIoTID data-cleaning fix (important for reproducing earlier numbers)

An earlier version of this repository's results for X-IIoTID were affected by
a data-cleaning defect, now fixed in `qcshield/data.py`
(`_is_numeric_with_placeholder`): roughly 40 of X-IIoTID's raw columns are
genuinely numeric (byte counts, packet counts, durations, CPU/IO timing
stats) but pandas reads them as `object` dtype because the raw CSV uses the
literal string `"-"` as a "not applicable for this flow direction" token
mixed in with numeric values. The categorical-detection step used a
dtype-only check, so these ~40 numeric columns were silently treated as
nominal categories and ordinal-encoded -- destroying their numeric ordering
and mapping any value unseen in the training split to an arbitrary "unseen"
sentinel. UNSW-NB15 and Edge-IIoTSet's raw data contain no such placeholder
token in their numeric columns, so only X-IIoTID was affected.

**Symptom**: a severe, reproducible train/validation generalization gap
specific to X-IIoTID (train loss collapsing toward 0.02-0.03 while validation
loss oscillated between 0.5 and 1.8, a 50+ fold gap) that did not appear on
UNSW-NB15 or Edge-IIoTSet under the identical training recipe.

**Fix**: object-dtype columns are now checked (on the training split only, so
the decision stays leak-free) for whether their non-placeholder values are
>=95% numeric-parseable; if so, the `"-"` token is coerced to `NaN` and the
column is routed through the existing numeric (`to_numeric` + `fillna(0.0)`)
path instead of the categorical encoder. X-IIoTID now resolves to 40 retained
features (down from 50, since the correlation filter can finally see genuine
numeric correlations) with identical train/val/test split sizes to before.
UNSW-NB15 and Edge-IIoTSet are provably unaffected (re-verified byte-for-byte
identical split sizes and feature counts after the fix).

**Effect on results**: every X-IIoTID number in the manuscript was
regenerated after this fix. The train/validation gap disappears (now tracks
closely, matching the healthy pattern already seen on the other two
datasets), and QE-QCSL's standing relative to the classical control changes
substantially -- see `RESULTS.md`-equivalent numbers in
`paper_results_<timestamp>/centralized/summary_X-IIoTID.csv`.

## Repository layout

```
qcshield/
  config.py        # single source of truth for every hyperparameter
  data.py           # split-first, leakage-free data pipeline (3 datasets);
                     # see _is_numeric_with_placeholder for the X-IIoTID fix
  circuits.py       # QCShieldLayer and all quantum-circuit construction (PennyLane)
  circuit_stats.py  # qml.specs()-based gate/depth statistics
  models.py         # HybridQuantumCNNBiLSTM + 4 parameter-matched classical baselines
  train.py          # training loop, evaluation metrics, class-weighting
  federated.py      # Flower-based federated simulation (FedAvg/FedProx/FedTrimmedAvg/FaultTolerantFedAvg)
  stats.py          # 5-seed aggregation + Wilcoxon signed-rank testing
  plots.py          # all figure generation
  style.py          # single source of truth for plot styling (fonts, colors, DPI)
  latex_circuit.py  # quantikz LaTeX circuit-diagram emitter
  cli.py            # command-line entry points
scripts/
  run_xiiotid_full_regen.py            # full centralized (60 runs) + federated (8 runs)
                                        # X-IIoTID re-run after the data-cleaning fix
  run_ablation_suite.py                # batch-size + LR-multiplier (UNSW-NB15 only) and
                                        # class-weighting (all 3 datasets) ablations, QCSL only
  run_xiiotid_unweighted_remaining_seeds.py  # fills X-IIoTID's unweighted ablation to 5 seeds
  run_zeroday_ablation.py              # X-IIoTID zero-day holdout (crypto-ransomware excluded
                                        # from train/val, kept in test) -- not yet run
  build_paper_results_snapshot.py      # curates a timestamped, paper-relevant results snapshot
                                        # (see "Results snapshot" below)
  run_shap_explain.py, plot_shap_results.py  # per-sample SHAP explainability pipeline
  run_xiiotid_grid.sh                  # 5-way concurrent grid-launch pattern on one GPU
tests/              # AST-based and numerical regression tests
requirements.txt    # pinned dependency versions
```

## Datasets

| Dataset | Classes | Features (post-filter) | Train / Val / Test |
|---|---|---|---|
| UNSW-NB15 | 10 | 32 | 107,578 / 15,369 / 30,737 |
| Edge-IIoTSet | 15 | 38 | 1,335,671 / 190,811 / 381,621 |
| X-IIoTID | 10 (top-10 by frequency) | 40 (post data-cleaning fix) | 490,801 / 70,115 / 140,229 |

All three datasets are loaded through the **identical split-first pipeline**
(`qcshield/data.py`): deduplicate on feature columns -> stratified
train/val/test split -> fit encoders, scaler, and correlation filter on the
training split only -> apply unchanged to validation/test. Nothing is fit on
held-out data for any of the three datasets.

Each dataset downloads automatically on first use (UNSW-NB15 via
HuggingFace `datasets`, Edge-IIoTSet and X-IIoTID via `kagglehub`); no manual
download step is required beyond having valid Kaggle API credentials
configured for the latter two.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned versions and the measured hardware/driver configuration are documented
at the top of `requirements.txt`. Note: the Flower federated-learning
dependency requires a NumPy-2.0-compatible release; if `import flwr` raises
`AttributeError: np.float_ was removed`, upgrade `flwr` rather than
downgrading NumPy.

## Usage

```bash
# One model, one seed
PYTHONPATH=. python -m qcshield.cli run --dataset X-IIoTID --epochs 30 --seed 0

# Full grid: 8 quantum configurations + 4 classical baselines, 5 seeds
PYTHONPATH=. python -m qcshield.cli grid --dataset X-IIoTID --epochs 30

# Aggregate statistics + Wilcoxon signed-rank test
PYTHONPATH=. python -m qcshield.cli stats --dataset X-IIoTID

# Circuit statistics + quantikz LaTeX diagrams
PYTHONPATH=. python -m qcshield.cli circuits

# Federated simulation (Flower): 10 clients, 20 rounds, IID or Dirichlet(0.5)
PYTHONPATH=. python -m qcshield.cli fed --dataset X-IIoTID --rounds 20

# Regenerate figures from existing result JSONs
PYTHONPATH=. python -m qcshield.cli figures --dataset X-IIoTID

# Rebuild the full X-IIoTID centralized + federated result set after the data fix
python scripts/run_xiiotid_full_regen.py

# Run the low-cost ablation suite (batch size + LR multiplier on UNSW-NB15,
# class-weighting on all three datasets)
python scripts/run_ablation_suite.py

# Curate a timestamped, paper-relevant results snapshot (JSON + CSV + figures)
python scripts/build_paper_results_snapshot.py
```

`--dataset` accepts `UNSW-NB15`, `Edge-IIoTSet`, or `X-IIoTID` for every
subcommand above.

## Experimental protocol

- 5 random seeds per configuration (`0, 1, 2, 3, 4`), reported as mean $\pm$
  std, with a two-sided Wilcoxon signed-rank test of each configuration
  against the proposed QE-QCSL configuration. This protocol is identical
  across all three datasets -- see Section 6.1 of the manuscript for the
  explicit cross-dataset statement.
- 30 training epochs, batch size 512, AdamW (`lr=5e-4`, quantum parameters in
  their own parameter group at `5x` the base learning rate),
  `ReduceLROnPlateau` scheduler on validation macro-F1, no early stopping.
- Inverse-frequency class weighting applied identically to every model
  (classical and quantum alike).
- Federated learning: 10 clients, 20 communication rounds, 1 local epoch per
  round, IID and Dirichlet($\alpha=0.5$) partitioning, four aggregation
  strategies (FedAvg, FedProx, FedTrimmedAvg, FaultTolerantFedAvg).

## Ablation studies

In addition to the main 8-configuration embedding x entangling-ansatz
architecture grid (Table 3), a low-cost ablation suite checks whether three
fixed design choices are doing real work:

- **Batch size** (128 / 256 / 512, QCShieldLayer only, UNSW-NB15 only): no
  batch size beats 512 by a wide margin at this seed depth.
- **Quantum learning-rate multiplier** (1x / 2x / 5x / 10x, QCShieldLayer
  only, UNSW-NB15 only): no multiplier is clearly optimal; the fixed 5x
  choice is not doing decisive work either way.
- **Class-weighted vs. unweighted cross-entropy loss** (QCShieldLayer only,
  **all three datasets**, 5 seeds each for UNSW-NB15/Edge-IIoTSet, 5 seeds
  for X-IIoTID after `run_xiiotid_unweighted_remaining_seeds.py`): aggregate
  accuracy/macro-F1 do **not** consistently favor class weighting (it is a
  wash on UNSW-NB15 and favors *unweighted* on X-IIoTID by both metrics).
  Per-class recall tells the real story: unweighted training drives at least
  one genuine attack class to 0% recall in 5 of the 6 dataset/config pairs
  examined (UNSW-NB15: Backdoor, Worms; Edge-IIoTSet: Ransomware, and
  Fingerprinting for AE-QCSL); on the sixth pair (X-IIoTID) no class is
  zeroed, but class-weighted training still beats unweighted on the rarest
  class, crypto-ransomware. This per-class evidence, not the aggregate
  numbers, is the actual justification for class-weighted training used
  throughout this work -- accuracy and a single macro-F1 average can hide a
  model that has quietly stopped detecting entire attack categories.

The batch-size and LR-multiplier ablations are UNSW-NB15-only and kept out of
the manuscript (not enough value to justify extending them to Edge-IIoTSet,
whose QCShieldLayer runs take ~10-12x longer than the other two datasets);
the class-weighting ablation is in the manuscript (Table 12) since it spans
all three datasets. A zero-day ablation (X-IIoTID, crypto-ransomware held
out of train/val entirely) is scripted (`run_zeroday_ablation.py`) but not
yet run.

## Results snapshot

`scripts/build_paper_results_snapshot.py` curates a timestamped
`paper_results_<YYYYMMDD_HHMMSS>/` directory containing exactly the JSON
result files, CSV summaries, and PDF figures that back the manuscript
through the Ablation Analysis section -- nothing more. It explicitly
excludes: the UNSW-NB15-only batch-size/LR-multiplier ablation stages (not
in the manuscript), pre-fix X-IIoTID results, redundant single-seed runs
superseded by existing 5-seed data, and the not-yet-run zero-day ablation.
Re-run it after any further result-generating script finishes to refresh the
snapshot; it is read-only with respect to every source directory, so it is
always safe to run alongside an in-progress training job.

```
paper_results_<timestamp>/
  centralized/{UNSW-NB15,Edge-IIoTSet,X-IIoTID}/*.json   # 60 runs each
  centralized/summary_{dataset}.csv                       # aggregated, Wilcoxon-tested
  federated/{UNSW-NB15,Edge-IIoTSet,X-IIoTID}/*.json      # 8 runs each
  ablation/class_weighting/{dataset}_unweighted/*.json    # 5-seed unweighted runs
  figures/                                                 # every PDF embedded in the manuscript
  MANIFEST.json                                            # what's included/excluded and why
```

## Tests

```bash
PYTHONPATH=. python tests/test_data.py       # data-pipeline leakage checks (AST-based, not just runtime)
PYTHONPATH=. python tests/test_circuits.py   # quantum circuit correctness and gradient-flow checks
```

## Reproducibility

Every result JSON records its full configuration (dataset, model, embedding,
layer, seed, epoch count, class-weighting flag, measured training time, peak
VRAM) alongside its metrics, so any reported number can be traced back to the
exact run that produced it. Every RNG (`random`, NumPy, Torch CPU+CUDA) is
seeded via `qcshield.config.set_seed()`.

## Citation

If you use this code, please cite the accompanying manuscript (citation
details to be added upon acceptance).

## License

To be added upon acceptance / repository release.
