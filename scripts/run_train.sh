#!/usr/bin/env bash
set -e

# Run training script for remote sensing dehazing
CONFIG="${1:-configs/default.yaml}"
MODEL="${2:-rs_pgnet}"
EPOCHS="${3:-10}"

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

export PYTHONPATH="."
echo "==> Starting Training: Model=$MODEL, Epochs=$EPOCHS, Config=$CONFIG"
$PYTHON scripts/train.py --config "$CONFIG" --model "$MODEL" --epochs "$EPOCHS"
