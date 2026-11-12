#!/usr/bin/env bash
set -e

# Run single-image / procedural demo
OUTPUT="${1:-demo_result.png}"

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

export PYTHONPATH="."
echo "==> Running Remote Sensing Dehazing Demo -> $OUTPUT"
$PYTHON scripts/demo.py --output "$OUTPUT"
