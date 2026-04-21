# Reproduction Guide

## 1. Environment

```bash
conda create -n ecovideo python=3.10 -y
conda activate ecovideo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/base.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Run:

```bash
PYTHONPATH=src python tools/check_env.py
```

## 2. Checkpoints

Set the checkpoint paths:

```bash
export WAN21_CKPT=/path/to/Wan2.1-T2V-14B
export WAN22_CKPT=/path/to/Wan2.2-T2V-A14B
export COGVIDEO_CKPT=/path/to/CogVideoX-5b
export EDEN_CKPT=/path/to/eden.pt
export RAFT_CKPT=/path/to/raft-things.pth
```

## 3. Run examples

```bash
bash scripts/infer_wan21.sh
bash scripts/infer_wan22.sh
bash scripts/infer_cogvideo.sh
```

Each run writes a video and a metrics JSON file under `outputs/`.

## 4. Benchmark reproduction

For a formal paper release, add:

- benchmark prompt files;
- VBench installation instructions;
- a script that maps each prompt to one generated video;
- a script that aggregates latency, VBench and quality metrics.
