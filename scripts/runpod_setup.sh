#!/bin/bash
# RunPod setup script for Qwen 3.5 Scope / AgentGenome
# Run this ONCE after creating the pod with network volume mounted at /workspace
#
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail

echo "=== Qwen 3.5 Scope — RunPod Setup ==="
SETUP_START=$SECONDS

# ─── 1. Verify GPU ───────────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Make sure you selected an A100 80GB pod."
    exit 1
fi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  Detected $NUM_GPUS GPU(s)"

# ─── 2. Parallel setup: system deps + Python deps + dirs/env ─────────────────
echo ""
echo "[2/5] Running parallel setup..."
VOLUME=/workspace
ENV_FILE="/workspace/agentgenome/.env"

# --- Background job 1: System dependencies (apt-get) ---
(
    apt-get update -qq && apt-get install -y -qq git vim tmux htop > /dev/null 2>&1
    echo "  [apt] System dependencies installed"
) &
PID_APT=$!

# --- Background job 2: Python dependencies (pip) ---
(
    cd /workspace/agentgenome
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

RESULTS_DIR=/workspace/data/results
ACTIVATIONS_DIR=/workspace/data/activations

# --- Set these manually ---
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
WANDB_API_KEY=REPLACE_ME
HF_TOKEN=REPLACE_ME
ENVEOF
        echo "  [env] Created .env — EDIT IT with your API keys!"
    else
        echo "  [env] .env already exists, skipping"
    fi

    cat > /workspace/agentgenome/scripts/load_env.sh << 'LOADEOF'
#!/bin/bash
# Source this to load env vars: source scripts/load_env.sh
set -a
source /workspace/agentgenome/.env
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

# ─── 3. Flash Attention (after pip finishes to avoid lock contention) ─────────
echo ""
echo "[3/5] Installing flash-attn (pre-built wheel)..."
pip install flash-attn -q 2>/dev/null || echo "WARN: flash-attn install failed, will use eager attention"

# ─── 4. Verify installation ──────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
n_gpus = torch.cuda.device_count()
print(f'  GPUs: {n_gpus}')
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f'    [{i}] {props.name} — {props.total_memory / 1e9:.1f} GB')
import transformers
print(f'  Transformers: {transformers.__version__}')
import safetensors
print(f'  Safetensors: {safetensors.__version__}')
import pydantic
print(f'  Pydantic: {pydantic.__version__}')
"

# Quick import test
python3 -c "
from src.model.config import Qwen35Config, HOOK_POINTS
from src.sae.model import TopKSAE
from src.steering.engine import SteeringEngine
print('  All project imports OK')
"

# ─── 5. Run tests ────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Running test suite..."
python3 -m pytest tests/ -v --tb=short

ELAPSED=$(( SECONDS - SETUP_START ))
echo ""
echo "=========================================="
echo "  Setup complete! (${ELAPSED}s)"
echo ""
echo "  NEXT STEPS:"
echo "  1. Edit /workspace/agentgenome/.env with your API keys"
echo "  2. source scripts/load_env.sh"
echo "  3. Log in to HuggingFace: huggingface-cli login"
echo "  4. Start the pilot: python3 scripts/run_pilot.py"
echo "     Or full pipeline: python3 scripts/01_setup_model.py"
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
