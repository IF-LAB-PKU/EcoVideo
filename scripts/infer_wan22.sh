#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   WAN22_CKPT=/path/to/Wan2.2-T2V-A14B EDEN_CKPT=/path/to/eden.pt bash scripts/infer_wan22.sh

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
WAN22_CKPT="${WAN22_CKPT:?Set WAN22_CKPT to your Wan2.2-T2V-A14B checkpoint directory}"
EDEN_CKPT="${EDEN_CKPT:?Set EDEN_CKPT to your EDEN checkpoint path}"
RAFT_CKPT="${RAFT_CKPT:-}"
PROMPT="${PROMPT:-A person is canoeing or kayaking.}"
OUT="${OUT:-outputs/wan22_ecovideo.mp4}"

TMP_CFG="outputs/eden_infer.local.yaml"
mkdir -p outputs
python tools/make_eden_config.py --template configs/eden_infer.yaml --eden_ckpt "$EDEN_CKPT" --output "$TMP_CFG"

RAFT_ARGS=()
if [[ -n "$RAFT_CKPT" ]]; then
  RAFT_ARGS=(--raft_ckpt "$RAFT_CKPT")
fi

python scripts/run_full_pipeline.py \
  --generator wan \
  --wan_version 2.2 \
  --ckpt_dir "$WAN22_CKPT" \
  --prompt "$PROMPT" \
  --wan_task t2v-A14B \
  --wan_size "1280*720" \
  --wan_frame_num 81 \
  --wan_sample_steps 40 \
  --wan_sample_solver unipc \
  --wan_guide_scale 4.0 \
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
