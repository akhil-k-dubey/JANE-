#!/usr/bin/env bash
set -e

echo "=== Running Project Jane: Phase 4 Asynchronous Audio Embodiment Benchmark ==="
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

$PYTHON_BIN benchmarks/eval_async_audio_interruption.py

echo "=== Audio Benchmark Complete ==="
