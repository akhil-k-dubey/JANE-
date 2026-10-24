#!/usr/bin/env bash
set -e

echo "=== Running Project Jane: Grid World Spatial Safety Benchmark ==="
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

$PYTHON_BIN benchmarks/eval_grid_world.py

echo "=== Grid World Benchmark Complete ==="
