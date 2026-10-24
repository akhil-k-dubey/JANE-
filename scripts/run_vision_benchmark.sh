#!/usr/bin/env bash
set -e

echo "=== Running Project Jane: Phase 4 Multimodal Vision & Safety Benchmark ==="
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

$PYTHON_BIN benchmarks/eval_vision_hallucination.py

echo "=== Vision Benchmark Complete ==="
