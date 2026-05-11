#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EcoVideo — VBench Evaluation Script
# ============================================================
#
# Usage:
#   VIDEOS_PATH=outputs/my_videos PROMPT_FILE=examples/prompts.txt bash evaluation/eval_script.sh
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"

# ---------- Configuration via environment variables ----------
VIDEOS_PATH="${VIDEOS_PATH:?Set VIDEOS_PATH to the folder containing generated videos}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/examples/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"
DEVICE="${DEVICE:-cuda}"
NAME="${NAME:-ecovideo}"

# ---------- Run ----------
python "${SCRIPT_DIR}/eval_vbench.py" \
    --videos_path "$VIDEOS_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --name "$NAME"
