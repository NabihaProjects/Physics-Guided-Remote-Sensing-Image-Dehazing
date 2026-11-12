#!/usr/bin/env bash
set -e

# Run evaluation script for remote sensing dehazing
CONFIG="${1:-configs/default.yaml}"
MODEL="${2:-rs_pgnet}"
SAMPLES="${3:-20}"

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

export PYTHONPATH="."
echo "==> Starting Evaluation: Model=$MODEL, Samples=$SAMPLES, Config=$CONFIG"
$PYTHON scripts/eval.py --config "$CONFIG" --model "$MODEL" --samples "$SAMPLES"
