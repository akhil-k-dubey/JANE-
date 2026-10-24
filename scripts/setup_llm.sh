#!/usr/bin/env bash
# Jane — One-Time Setup Script
# Installs nvcc, builds llama-cpp-python with CUDA, and downloads Qwen2.5-3B-Instruct Q4_K_M.
# Run with: bash scripts/setup_llm.sh

set -e

JANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$JANE_DIR/.venv/bin"

echo ""
echo "════════════════════════════════════════════"
echo "  Jane LLM Setup — Qwen2.5-3B + CUDA"
echo "════════════════════════════════════════════"
echo ""

# ── Step 1: Install nvcc ───────────────────────────────────────────────────
echo "[1/3] Installing CUDA compiler (nvcc)..."

# Try NVIDIA's keyring-based repository (works on Ubuntu 22.04+)
if ! command -v nvcc &>/dev/null; then
    if command -v apt-get &>/dev/null; then
        # Add NVIDIA keyring if not present
        if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
            wget -qO /tmp/cuda-keyring.deb \
                https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            dpkg -i /tmp/cuda-keyring.deb
            apt-get update -qq
        fi
        apt-get install -y cuda-nvcc-12-8 2>/dev/null || \
        apt-get install -y cuda-nvcc-12-6 2>/dev/null || \
        apt-get install -y nvidia-cuda-toolkit 2>/dev/null || \
        { echo "[WARN] apt nvcc install failed. Trying system PATH..."; }
    fi
fi

# Detect nvcc path
NVCC_PATH=$(command -v nvcc 2>/dev/null || find /usr/local/cuda*/bin -name nvcc 2>/dev/null | head -1 || echo "")
if [ -z "$NVCC_PATH" ]; then
    echo ""
    echo "[ERROR] nvcc not found. Please install it manually:"
    echo "  sudo apt-get install cuda-nvcc-12-8"
    echo "  OR: sudo apt-get install nvidia-cuda-toolkit"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
echo "  nvcc: $NVCC_PATH"
echo "  CUDA_HOME: $CUDA_HOME"

# ── Step 2: Build llama-cpp-python with CUDA ──────────────────────────────
echo ""
echo "[2/3] Building llama-cpp-python with CUDA support (this takes ~3-5 min)..."

export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
    "$VENV/pip" install llama-cpp-python \
    --upgrade \
    --force-reinstall \
    --no-cache-dir \
    --no-binary llama-cpp-python

echo ""
echo "  llama-cpp-python built with CUDA ✓"

# ── Step 3: Download Qwen2.5-3B Q4_K_M ────────────────────────────────────
echo ""
echo "[3/3] Downloading Qwen2.5-3B-Instruct Q4_K_M GGUF (~2.2 GB)..."
"$VENV/python" "$JANE_DIR/scripts/download_qwen.py"

echo ""
echo "════════════════════════════════════════════"
echo "  Setup complete. Run:  python main.py"
echo "════════════════════════════════════════════"
echo ""
