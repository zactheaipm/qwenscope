#!/bin/bash
# Sync local code to RunPod pod via rsync over SSH
#
# Usage: bash scripts/sync_to_pod.sh <POD_SSH_ADDRESS>
# Example: bash scripts/sync_to_pod.sh ssh root@205.196.17.2 -p 22222
#          bash scripts/sync_to_pod.sh root@205.196.17.2 22222
#
# The SSH address and port are shown in the RunPod pod's "Connect" dialog.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/sync_to_pod.sh <USER@HOST> [PORT]"
    echo ""
    echo "Get these from RunPod > Pod > Connect > SSH"
    echo "Example: bash scripts/sync_to_pod.sh root@205.196.17.2 22222"
    exit 1
fi

SSH_HOST="$1"
SSH_PORT="${2:-22}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Ensure rsync is available on the remote (RunPod images may not have it)
echo "Ensuring rsync is installed on remote..."
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" \
    'command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1)' 2>/dev/null

echo "Syncing $PROJECT_DIR → $SSH_HOST:/workspace/agentgenome (port $SSH_PORT)"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'data/activations/' \
    --exclude 'data/results/' \
    --exclude '.env' \
    --exclude '*.egg-info' \
    --exclude '.ruff_cache' \
    -e "ssh -p $SSH_PORT" \
    "$PROJECT_DIR/" \
    "$SSH_HOST:/workspace/agentgenome/"

echo ""
echo "Sync complete. Now SSH in and run:"
echo "  ssh -p $SSH_PORT $SSH_HOST"
echo "  cd /workspace/agentgenome"
echo "  bash scripts/runpod_setup.sh"
