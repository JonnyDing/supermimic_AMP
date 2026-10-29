#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python}"
RUN_NAME="${RUN_NAME:-wmrl_scratch_14000}"
mkdir -p logs/supermimic
LAUNCH_LOG="logs/supermimic/launcher_$(date +%Y%m%d_%H%M%S).log"
"$PYTHON_BIN" -u scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4090 \
  --agent.max-iterations=14000 \
  --agent.run-name="$RUN_NAME" --agent.save-interval=1000 \
  "$@" 2>&1 | tee "$LAUNCH_LOG"
