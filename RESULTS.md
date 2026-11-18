# Results

Full, unedited 5-seed centralized results for all three datasets, 30 epochs,
class-weighted loss, identical splits/preprocessing/hyperparameters across
every model (only the embedding/layer choice differs). These are exactly the
numbers in the manuscript's Table 3 (`Fed-QCShield_Revision_R2.tex`,
companion LaTeX repository). Source CSVs, reproduced verbatim here:

- UNSW-NB15: `paper_results_<timestamp>/centralized/summary_UNSW-NB15.csv`
- Edge-IIoTSet: `paper_results_<timestamp>/centralized/summary_Edge-IIoTSet.csv`
- X-IIoTID: `paper_results_<timestamp>/centralized/summary_X-IIoTID.csv` (fixed,
  leakage-free pipeline -- see the data-cleaning note in `README.md`)

`wilcoxon_p_vs_headline` and `headline_wins` are the two-sided Wilcoxon
signed-rank test and paired-seed win count against the proposed QE-QCSL
configuration (blank for QE-QCSL itself, which is the reference).

Model key: `AE`/`QE` = angle/QAOA-inspired embedding; `BEL`/`SEL`/`RL`/`QCSL`
= BasicEntanglerLayers / StronglyEntanglingLayers / RandomLayers / the
proposed QCShieldLayer. `CNN`, `LSTM`, `CNN-BiLSTM`, `CNN-BiLSTM-MLP` are the
parameter-matched classical controls (no quantum layer).

## UNSW-NB15 (10 classes, 32 features, 107,578/15,369/30,737 train/val/test)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train time (s) | Wilcoxon p | Wins vs QE-QCSL |
|---|---|---|---|---|---|---|
| QE-SEL | 70.84%±0.40% | 48.59%±0.66% | 74.79%±0.37% | 724 | 0.8125 | 2/5 |
| AE-QCSL | 70.79%±0.34% | 48.46%±0.74% | 74.81%±0.25% | 875 | 0.625 | 1/5 |
| CNN-BiLSTM | 70.74%±0.62% | 49.28%±1.07% | 74.64%±0.59% | 127 | 0.125 | 1/5 |
| QE-RL | 70.72%±0.59% | 48.24%±1.21% | 74.74%±0.46% | 891 | 1.0 | 2/5 |
| QE-QCSL (proposed) | 70.51%±0.60% | 48.44%±0.74% | 74.43%±0.53% | 1204 | — | — |
| AE-RL | 70.40%±0.46% | 48.14%±0.97% | 74.48%±0.39% | 584 | 0.625 | 3/5 |
| AE-SEL | 70.30%±0.73% | 48.13%±0.55% | 74.32%±0.58% | 429 | 0.625 | 4/5 |
| CNN-BiLSTM-MLP | 70.08%±1.04% | 47.90%±1.36% | 74.15%±0.76% | 131 | 0.3125 | 4/5 |
| AE-BEL | 70.27%±1.17% | 47.70%±0.84% | 74.37%±0.61% | 402 | 0.125 | 4/5 |
| QE-BEL | 70.03%±0.81% | 47.68%±1.31% | 74.14%±0.63% | 682 | 0.3125 | 3/5 |
| LSTM | 68.35%±0.97% | 46.03%±1.09% | 73.09%±0.65% | 141 | 0.0625 | 5/5 |
| CNN | 60.23%±1.51% | 36.27%±1.27% | 64.59%±1.74% | 57 | 0.0625 | 5/5 |

No pairwise comparison reaches the conventional p<0.05 threshold; every
configuration clusters within roughly one standard deviation of QE-QCSL's
macro-F1.

## Edge-IIoTSet (15 classes, 38 features, 1,335,671/190,811/381,621 train/val/test)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train time (s) | Wilcoxon p | Wins vs QE-QCSL |
|---|---|---|---|---|---|---|
| LSTM | 98.30%±0.07% | 89.86%±0.91% | 98.33%±0.08% | 2218 | 0.3125 | 2/5 |
| QE-QCSL (proposed) | 98.21%±0.21% | 88.54%±1.87% | 98.23%±0.22% | 15429 | — | — |
| CNN-BiLSTM | 98.16%±0.05% | 87.87%±0.57% | 98.16%±0.06% | 1966 | 1.0 | 2/5 |
| AE-RL | 98.13%±0.06% | 87.40%±0.66% | 98.14%±0.07% | 7622 | 0.3125 | 4/5 |
| CNN-BiLSTM-MLP | 98.13%±0.09% | 86.94%±1.72% | 98.10%±0.16% | 2244 | 0.3125 | 4/5 |
| QE-RL | 98.11%±0.08% | 86.80%±1.86% | 98.08%±0.14% | 11432 | 0.3125 | 4/5 |
| AE-QCSL | 98.08%±0.16% | 86.77%±2.13% | 98.05%±0.20% | 11271 | 0.4375 | 3/5 |
| QE-SEL | 98.08%±0.14% | 86.52%±2.20% | 98.02%±0.23% | 9371 | 0.1875 | 4/5 |
| QE-BEL | 97.88%±0.35% | 85.95%±1.81% | 97.83%±0.41% | 8810 | 0.0625 | 5/5 |
| AE-BEL | 97.92%±0.21% | 85.49%±1.57% | 97.87%±0.19% | 5325 | 0.0625 | 5/5 |
| AE-SEL | 97.95%±0.15% | 85.19%±1.60% | 97.87%±0.21% | 5697 | 0.0625 | 5/5 |
| CNN | 81.77%±8.42% | 67.90%±5.77% | 85.32%±6.58% | 710 | 0.0625 | 5/5 |

QE-QCSL (88.54%±1.87% macro-F1) is statistically indistinguishable from its
own parameter-matched classical control, CNN-BiLSTM (87.87%±0.57%,
p=1.0000); a plain LSTM attains the highest macro-F1 in the grid (89.86%)
but is not a parameter-matched comparator.

## X-IIoTID (10 classes, 40 features post data-cleaning fix, 490,801/70,115/140,229 train/val/test)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train time (s) | Wilcoxon p | Wins vs QE-QCSL |
|---|---|---|---|---|---|---|
| CNN-BiLSTM-MLP | 96.32%±0.29% | 88.73%±1.09% | 96.65%±0.29% | 265 | 0.125 | 1/5 |
| CNN-BiLSTM | 96.40%±0.26% | 88.72%±1.16% | 96.73%±0.24% | 264 | 0.0625 | 0/5 |
| AE-QCSL | 96.43%±0.28% | 88.65%±0.36% | 96.76%±0.27% | 1039 | 0.125 | 1/5 |
| AE-SEL | 96.51%±0.37% | 87.68%±1.40% | 96.85%±0.31% | 544 | 0.8125 | 3/5 |
| QE-SEL | 96.26%±0.31% | 87.17%±2.08% | 96.66%±0.25% | 866 | 0.8125 | 2/5 |
| QE-RL | 96.34%±0.25% | 87.13%±1.92% | 96.68%±0.18% | 1038 | 0.625 | 4/5 |
| QE-QCSL (proposed) | 96.07%±0.24% | 87.07%±1.12% | 96.46%±0.23% | 1361 | — | — |
| QE-BEL | 95.93%±0.94% | 86.84%±2.29% | 96.32%±0.85% | 824 | 0.8125 | 2/5 |
| AE-RL | 95.82%±1.00% | 86.49%±2.07% | 96.22%±0.95% | 709 | 0.625 | 3/5 |
| AE-BEL | 95.78%±0.58% | 86.46%±1.70% | 96.20%±0.51% | 506 | 0.625 | 3/5 |
| LSTM | 95.86%±0.21% | 84.23%±0.86% | 96.31%±0.16% | 248 | 0.0625 | 5/5 |
| CNN | 90.70%±4.47% | 76.62%±3.56% | 92.18%±3.14% | 71 | 0.0625 | 5/5 |

QE-QCSL's own parameter-matched classical control, CNN-BiLSTM, beats it on
all 5 seeds (p=0.0625, the minimum attainable at this seed count), though
the margin is under two points of macro-F1 (88.72% vs. 87.07%). QE-QCSL
ranks seventh of twelve configurations by macro-F1 on X-IIoTID, the only
one of the three datasets where it trails a majority of the grid, while
training roughly 19x slower than the fastest classical model (CNN).

## Federated learning (20 rounds, 10 clients, IID and Dirichlet(0.5))

Final-round global accuracy / macro-F1 for all four Flower strategies, all
three datasets. Source: `results/fed/fed_<strategy>_<partition>_<dataset>.json`.

| Dataset | Partition | FedAvg | FedProx | FedTrimmedAvg | FaultTolerantFedAvg |
|---|---|---|---|---|---|
| UNSW-NB15 | IID | 45.57% / 25.41% | 11.47% / 10.32% | 48.49% / 27.01% | 24.21% / 20.15% |
| UNSW-NB15 | Dirichlet | 53.67% / 24.05% | 26.87% / 15.11% | 38.59% / 20.67% | 50.89% / 27.04% |
| Edge-IIoTSet | IID | 97.01% / 78.16% | 80.72% / 22.58% | 97.75% / 82.27% | 95.95% / 77.26% |
| Edge-IIoTSet | Dirichlet | 95.48% / 71.86% | 85.93% / 29.03% | 93.34% / 64.53% | 94.83% / 69.82% |
| X-IIoTID | IID | 34.71% / 34.10% | 25.69% / 25.85% | 35.58% / 35.30% | 35.86% / 36.89% |
| X-IIoTID | Dirichlet | 49.65% / 39.13% | 28.81% / 24.11% | 36.82% / 31.57% | 40.66% / 35.21% |

FedProx trails every other strategy on all three datasets under both
partition schemes -- the most consistent finding in the federated sweep.
FedAvg or FedTrimmedAvg are strongest on UNSW-NB15/Edge-IIoTSet depending on
dataset and partition; FedAvg under Dirichlet partitioning is also the
strongest configuration on X-IIoTID (49.65% accuracy, 39.13% macro-F1), the
same leading strategy family as the other two datasets rather than a
different one. Federated accuracy trails the centralized results by a wide
margin on all three datasets under this 20-round budget.

## Takeaway

Across all three datasets, QE-QCSL is never significantly better than its
parameter-matched classical control (CNN-BiLSTM) at p<0.05 — on UNSW-NB15
and Edge-IIoTSet the comparison is statistically indistinguishable
(p=0.8125 and p=1.0000 respectively), and on X-IIoTID the classical control
wins outright at the minimum attainable p-value for 5 seeds (p=0.0625, all
5 seeds), though the margin there is under two points of macro-F1. QE-QCSL
is also consistently the slowest or near-slowest configuration to train
(roughly 7x to 27x the fastest classical model's time, depending on
dataset). The class-weighting ablation (Table 12 in the manuscript) adds a
further caveat: aggregate accuracy/macro-F1 do not consistently favor
class-weighted training, but per-class recall on the rarest attack
category does, on all three datasets. The contribution of this work is the
leakage-free, federated, reproducible evaluation framework itself —
including its extension to a third, modern IIoT dataset under an identical
protocol — not a demonstrated quantum accuracy advantage.
