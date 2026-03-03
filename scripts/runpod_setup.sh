#!/bin/bash
# RunPod setup script for Qwen 3.5 Scope / QwenScope
# Run this ONCE after creating the pod with network volume mounted at /workspace
#
# Recommended pod: H200 SXM (141 GB VRAM), 200 GB volume, ≥16 vCPUs, ≥128 GB RAM
# Docker image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail

echo "=== Qwen 3.5 Scope — RunPod Setup ==="
SETUP_START=$SECONDS

# ─── 1. Verify GPU ───────────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Make sure you selected an H200 SXM pod."
    exit 1
fi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  Detected $NUM_GPUS GPU(s)"

# ─── 2. Parallel setup: system deps + Python deps + dirs/env ─────────────────
echo ""
echo "[2/6] Running parallel setup..."
VOLUME=/workspace
ENV_FILE="/workspace/qwenscope/.env"

# --- Background job 1: System dependencies (apt-get) ---
(
    apt-get update -qq && apt-get install -y -qq git vim tmux htop rsync > /dev/null 2>&1
    echo "  [apt] System dependencies installed"
) &
PID_APT=$!

# --- Background job 2: Python dependencies (pip) ---
(
    cd /workspace/qwenscope
    pip install --upgrade pip setuptools wheel -q
    pip install -e ".[dev]" -q
    echo "  [pip] Python dependencies installed"
) &
PID_PIP=$!

# --- Background job 3: Directories + env file (fast, but no reason to block) ---
(
    mkdir -p "$VOLUME/huggingface_cache" \
             "$VOLUME/models" \
             "$VOLUME/data/activations" \
             "$VOLUME/data/results" \
             "$VOLUME/data/contrastive_pairs" \
             "$VOLUME/data/scenarios" \
             "$VOLUME/data/saes"

    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << 'ENVEOF'
# === RunPod Environment for Qwen 3.5 Scope ===
# Edit these values, then run: source scripts/load_env.sh

HF_HOME=/workspace/huggingface_cache
QWEN35_MODEL_PATH=Qwen/Qwen3.5-27B
DEVICE=cuda
PYTHONUNBUFFERED=1

RESULTS_DIR=/workspace/qwenscope/data/results
ACTIVATIONS_DIR=/workspace/qwenscope/data/activations

# --- Set these manually ---
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
WANDB_API_KEY=REPLACE_ME
HF_TOKEN=REPLACE_ME
ENVEOF
        echo "  [env] Created .env — EDIT IT with your API keys!"
    else
        echo "  [env] .env already exists, skipping"
    fi

    cat > /workspace/qwenscope/scripts/load_env.sh << 'LOADEOF'
#!/bin/bash
# Source this to load env vars: source scripts/load_env.sh
set -a
source /workspace/qwenscope/.env
set +a
echo "Environment loaded."
LOADEOF
    echo "  [dir] Directories ready"
) &
PID_DIRS=$!

# Wait for all parallel jobs
FAIL=0
wait $PID_DIRS || FAIL=1
wait $PID_APT  || FAIL=1
wait $PID_PIP  || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "ERROR: One or more parallel setup jobs failed. Check output above."
    exit 1
fi
echo "Parallel setup complete."

# ─── 3. Upgrade PyTorch + install fast kernels for Qwen 3.5 ──────────────────
# Qwen 3.5-27B has 48 GatedDeltaNet layers + 16 attention layers.
# fla (Flash Linear Attention) provides optimized CUDA kernels for DeltaNet;
# without it the model falls back to a naive sequential recurrence (~5× slower).
# fla requires PyTorch ≥ 2.6 and Triton ≥ 3.2.
echo ""
echo "[3/6] Upgrading PyTorch and installing fast kernels..."
pip uninstall -y torch torchvision torchaudio triton -q 2>/dev/null || true
pip install "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu124 -q
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "WARN: flash-attn install failed, will use eager attention"
pip install causal-conv1d --no-build-isolation -q 2>/dev/null || echo "WARN: causal-conv1d install failed"
pip install "git+https://github.com/fla-org/flash-linear-attention.git" -q
pip install accelerate -q

# Reinstall project deps (picks up new PyTorch)
cd /workspace/qwenscope
pip install -e ".[dev]" -q

# ─── 4. HuggingFace login ────────────────────────────────────────────────────
echo ""
echo "[4/6] Logging in to HuggingFace..."
if [ -f "$ENV_FILE" ]; then
    HF_TOKEN_VAL=$(grep "^HF_TOKEN=" "$ENV_FILE" | cut -d= -f2)
    if [ "$HF_TOKEN_VAL" != "REPLACE_ME" ] && [ -n "$HF_TOKEN_VAL" ]; then
        python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN_VAL')"
        echo "  HuggingFace login OK"
    else
        echo "  SKIP: Set HF_TOKEN in .env first"
    fi
else
    echo "  SKIP: No .env file found"
fi

# ─── 5. Verify installation ──────────────────────────────────────────────────
echo ""
echo "[5/6] Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
n_gpus = torch.cuda.device_count()
print(f'  GPUs: {n_gpus}')
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f'    [{i}] {props.name} — {props.total_memory / 1e9:.1f} GB')

import triton
print(f'  Triton: {triton.__version__}')

try:
    import flash_attn
    print(f'  Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('  Flash Attention: NOT INSTALLED (will use eager)')

try:
    import causal_conv1d
    print(f'  causal-conv1d: {causal_conv1d.__version__}')
except ImportError:
    print('  causal-conv1d: NOT INSTALLED')

try:
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print(f'  FLA (Flash Linear Attention): installed — GatedDeltaRule kernels available')
except ImportError:
    print('  FLA: NOT INSTALLED (DeltaNet layers will use naive recurrence)')

import transformers
print(f'  Transformers: {transformers.__version__}')
import safetensors
print(f'  Safetensors: {safetensors.__version__}')
import pydantic
print(f'  Pydantic: {pydantic.__version__}')
"

# Quick import test
cd /workspace/qwenscope
PYTHONPATH=/workspace/qwenscope python3 -c "
from src.model.config import Qwen35Config, HOOK_POINTS
from src.sae.model import TopKSAE
from src.steering.engine import SteeringEngine
print('  All project imports OK')
"

# ─── 6. Run tests ────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Running test suite..."
cd /workspace/qwenscope
PYTHONPATH=/workspace/qwenscope python3 -m pytest tests/ -v --tb=short

ELAPSED=$(( SECONDS - SETUP_START ))
echo ""
echo "=========================================="
echo "  Setup complete! (${ELAPSED}s)"
echo ""
echo "  NEXT STEPS:"
echo "  1. Edit /workspace/qwenscope/.env with your API keys"
echo "  2. source scripts/load_env.sh"
echo "  3. Start the pipeline: python3 scripts/01_setup_model.py"
echo ""
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "  MULTI-GPU DETECTED ($NUM_GPUS GPUs):"
    echo "    SAE training will use parallel mode automatically:"
    echo "      python3 scripts/03_train_saes.py"
    echo "    Model on GPU 0, SAEs distributed across GPUs 1-$((NUM_GPUS-1))"
else
    echo "  SINGLE GPU: 03_train_saes.py still works (1 model load, all 7 layers per pass)"
fi
echo "=========================================="
