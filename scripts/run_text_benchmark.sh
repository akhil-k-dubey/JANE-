#!/usr/bin/env bash
set -e

echo "=== Running Project Jane: Text Logic & Hallucination Benchmarks ==="
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

$PYTHON_BIN benchmarks/eval_relational_memory.py
$PYTHON_BIN benchmarks/eval_hallucination_rate.py

echo "=== Text Benchmarks Complete ==="
