#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   COGVIDEO_CKPT=/path/to/CogVideoX-5b EDEN_CKPT=/path/to/eden.pt bash scripts/infer_cogvideo.sh

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
COGVIDEO_CKPT="${COGVIDEO_CKPT:?Set COGVIDEO_CKPT to your CogVideoX checkpoint directory}"
EDEN_CKPT="${EDEN_CKPT:?Set EDEN_CKPT to your EDEN checkpoint path}"
RAFT_CKPT="${RAFT_CKPT:-}"
PROMPT="${PROMPT:-A giraffe is walking through a green field under a blue sky.}"
OUT="${OUT:-outputs/cogvideo_ecovideo.mp4}"

TMP_CFG="outputs/eden_infer.local.yaml"
mkdir -p outputs
python tools/make_eden_config.py --template configs/eden_infer.yaml --eden_ckpt "$EDEN_CKPT" --output "$TMP_CFG"

RAFT_ARGS=()
if [[ -n "$RAFT_CKPT" ]]; then
  RAFT_ARGS=(--raft_ckpt "$RAFT_CKPT")
fi

python scripts/run_full_pipeline.py \
  --generator cogvideo \
  --ckpt_dir "$COGVIDEO_CKPT" \
  --prompt "$PROMPT" \
  --cogvideo_height 480 \
  --cogvideo_width 720 \
  --cogvideo_num_frames 49 \
  --cogvideo_num_inference_steps 50 \
  --cogvideo_keyframe_by_entropy \
  --cogvideo_entropy_steps 5 \
  --cogvideo_entropy_mode ema \
  --cogvideo_keyframe_topk 8 \
  --eden_config "$TMP_CFG" \
  "${RAFT_ARGS[@]}" \
  --target_fps 24 \
  --keyframe_mode all \
  --output_path "$OUT"
