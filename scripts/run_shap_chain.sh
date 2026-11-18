#!/bin/bash
set -e
cd /home/jpsahoo/Desktop/skills-main/claude_research

echo "=== SHAP: Edge-IIoTSet (per-class=7, ~105 samples) ==="
PYTHONPATH=. ./.venv/bin/python -u scripts/run_shap_explain.py --dataset Edge-IIoTSet --per-class 7
PYTHONPATH=. ./.venv/bin/python -u scripts/plot_shap_results.py --dataset Edge-IIoTSet

echo "=== SHAP: UNSW-NB15 (per-class=10, 100 samples) ==="
PYTHONPATH=. ./.venv/bin/python -u scripts/run_shap_explain.py --dataset UNSW-NB15 --per-class 10
PYTHONPATH=. ./.venv/bin/python -u scripts/plot_shap_results.py --dataset UNSW-NB15

echo "=== ALL DONE: SHAP explainability complete (matched ~100-sample settings) ==="
