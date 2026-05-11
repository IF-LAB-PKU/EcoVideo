#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   WAN21_CKPT=/path/to/Wan2.1-T2V-14B EDEN_CKPT=/path/to/eden.pt bash scripts/infer_wan21.sh

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
WAN21_CKPT="${WAN21_CKPT:?Set WAN21_CKPT to your Wan2.1-T2V-14B checkpoint directory}"
EDEN_CKPT="${EDEN_CKPT:?Set EDEN_CKPT to your EDEN checkpoint path}"
RAFT_CKPT="${RAFT_CKPT:-}"
PROMPT="${PROMPT:-A boat sailing smoothly on a calm lake.}"
OUT="${OUT:-outputs/wan21_ecovideo.mp4}"

TMP_CFG="outputs/eden_infer.local.yaml"
mkdir -p outputs
python tools/make_eden_config.py --template configs/eden_infer.yaml --eden_ckpt "$EDEN_CKPT" --output "$TMP_CFG"

RAFT_ARGS=()
if [[ -n "$RAFT_CKPT" ]]; then
  RAFT_ARGS=(--raft_ckpt "$RAFT_CKPT")
fi

python scripts/run_full_pipeline.py \
  --generator wan \
  --wan_version 2.1 \
  --ckpt_dir "$WAN21_CKPT" \
  --prompt "$PROMPT" \
  --wan_task t2v-14B \
  --wan_size "1280*720" \
  --wan_frame_num 81 \
  --wan_sample_steps 50 \
  --wan_sample_solver unipc \
  --wan_guide_scale 5.0 \
  --wan_keyframe_by_entropy \
  --wan_entropy_steps 5 \
  --wan_entropy_mode ema \
  --wan_keyframe_target_fps 8 \
  --wan_use_nonkey_context \
  --eden_config "$TMP_CFG" \
  "${RAFT_ARGS[@]}" \
  --target_fps 24 \
  --keyframe_mode all \
  --output_path "$OUT"
