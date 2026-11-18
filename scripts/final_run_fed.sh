#!/bin/bash
# Federated rerun after porting the fixed centralized regime (AdamW +
# per-quantum-group LR + class-weighted loss) into federated.py's
# train_local/test_model. Sequential across datasets: each dataset's
# 10-client Flower simulation already uses ~13GB VRAM, so running both
# datasets' simulations at once would not fit on a 16GB card.
cd /home/jpsahoo/Desktop/skills-main/claude_research
LOGDIR=./logs

echo "=== UNSW-NB15 federated rerun start $(date) ==="
PYTHONPATH=. ./.venv/bin/python -m qcshield.cli fed \
    --dataset UNSW-NB15 --seed 0 --rounds 10 --no-figures \
    > $LOGDIR/fed_unsw.log 2>&1
echo "=== UNSW-NB15 federated rerun DONE $(date) ==="

echo "=== Edge-IIoTSet federated rerun start $(date) ==="
KAGGLEHUB_CACHE=./data/kaggle PYTHONPATH=. ./.venv/bin/python -m qcshield.cli fed \
    --dataset Edge-IIoTSet --seed 0 --rounds 10 --no-figures \
    > $LOGDIR/fed_edge.log 2>&1
echo "=== Edge-IIoTSet federated rerun DONE $(date) ==="
echo "FEDERATED RERUN COMPLETE $(date)"
