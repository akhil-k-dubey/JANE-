#!/usr/bin/env bash
set -e

echo "=== Running Project Jane: Phase 3 Dynamic Action & Multi-Step Reasoning Benchmark ==="
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

$PYTHON_BIN benchmarks/eval_phase3_dynamic_execution.py

echo "=== Phase 3 Benchmark Complete ==="
