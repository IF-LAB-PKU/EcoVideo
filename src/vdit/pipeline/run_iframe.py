from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from vdit.interpolators.eden_local import EdenInterpolator
from vdit.metrics.fuse import IntervalScorer, ScorerWeights
from vdit.flow.raft_estimator import RaftEstimator
from vdit.scheduler.greedy_refine import greedy_refine
from vdit.pipeline.keyframes import random_keyframes, uniform_keyframes
from vdit.pipeline.video_io import read_video_tensor, write_video_tensor


@dataclass(frozen=True)
class PipelineConfig:
    eden_config: str
    raft_ckpt: Optional[str] = None
    raft_device: str = "cuda:0"
    eden_device: str = "cuda:0"
    use_split_gpu: bool = False
    target_fps: float = 24.0
    keyframe_mode: Literal["all", "uniform", "random"] = "all"
    keyframes_k: int = 8
    seed: int = 0
    topk_ratio: float = 0.1
    weights: ScorerWeights = ScorerWeights()


def run_interpolation_pipeline_from_frames(
    *,
    frames: torch.Tensor,
    fps_src: float,
    output_path: str,
    cfg: PipelineConfig,
    log_file: Optional[str] = None,
    save_keyframes_video_path: Optional[str] = None,
    return_timing: bool = False,
) -> Optional[dict]:
    """
    核心插帧 pipeline（与 video_path 解耦）：
      关键帧(均匀/随机/全帧) -> 5信息融合打分 -> greedy_refine(自适应决定每段插多少帧) -> EDEN 插帧 -> 输出视频

    frames: [T,3,H,W] float in [0,1]
    fps_src: 输入视频 fps（对 duration 与 target_len 计算很重要）

    return_timing=True 时，返回一个 timing dict（便于算加速比/拆分耗时）。
    """
    import time

    t_all0 = time.perf_counter()
    timing: dict = {}

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expect frames [T,3,H,W], got {tuple(frames.shape)}")
    if fps_src <= 0:
        raise ValueError(f"fps_src must be > 0, got {fps_src}")

    num_frames = int(frames.shape[0])
    duration = (num_frames - 1) / float(fps_src)
    target_len = int(round(duration * float(cfg.target_fps))) + 1
    if target_len < 2:
        raise ValueError("target_len < 2，输入视频太短或 target_fps 太小。")

    timing.update(
        dict(
            input_num_frames=num_frames,
            fps_src=float(fps_src),
            duration_sec=float(duration),
            target_fps=float(cfg.target_fps),
            target_len=int(target_len),
            keyframe_mode=str(cfg.keyframe_mode),
        )
    )

    # -------- 关键帧选择 --------
    t0 = time.perf_counter()
    if cfg.keyframe_mode == "all":
        init_frames = [frames[i].unsqueeze(0).cpu() for i in range(num_frames)]
    elif cfg.keyframe_mode == "uniform":
        init_frames = uniform_keyframes(frames, k=cfg.keyframes_k)
    elif cfg.keyframe_mode == "random":
        init_frames = random_keyframes(frames, k=cfg.keyframes_k, seed=cfg.seed)
    else:
        raise ValueError(f"unknown keyframe_mode: {cfg.keyframe_mode}")
    timing["keyframe_select_sec"] = float(time.perf_counter() - t0)
    timing["keyframe_count"] = int(len(init_frames))

    t0 = time.perf_counter()
    if save_keyframes_video_path is not None:
        keyframes_tensor = torch.cat(init_frames, dim=0)
        os.makedirs(os.path.dirname(save_keyframes_video_path) or ".", exist_ok=True)
        write_video_tensor(save_keyframes_video_path, keyframes_tensor, fps=float(fps_src))
    timing["save_keyframes_video_sec"] = float(time.perf_counter() - t0)

    if len(init_frames) > target_len:
        raise ValueError(
            f"initial frames ({len(init_frames)}) > target_len ({target_len}). "
            f"请先降低 keyframes_k 或调高 target_fps。"
        )

    # -------- 插帧器与打分器 --------
    t0 = time.perf_counter()
    eden = EdenInterpolator(
        config_path=cfg.eden_config,
        device=cfg.eden_device,
        use_split_gpu=cfg.use_split_gpu,
    )
    timing["eden_init_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    raft = None
    if cfg.raft_ckpt:
        raft = RaftEstimator(model_path=cfg.raft_ckpt, device=cfg.raft_device)
    timing["raft_init_sec"] = float(time.perf_counter() - t0)
    timing["raft_enabled"] = bool(cfg.raft_ckpt)

    t0 = time.perf_counter()
    scorer = IntervalScorer(raft=raft, weights=cfg.weights, topk_ratio=cfg.topk_ratio)
    timing["scorer_init_sec"] = float(time.perf_counter() - t0)
    timing["topk_ratio"] = float(cfg.topk_ratio)

    def score_fn(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(scorer(a, b))

    def interp_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return eden.interpolate(a, b)

    # -------- 自适应插帧数量 --------
    t0 = time.perf_counter()
    out_frames = greedy_refine(
        init_frames,
        target_len=target_len,
        score_fn=score_fn,
        interp_fn=interp_fn,
        verbose=True,
        log_file=log_file,
    )
    timing["greedy_refine_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    out = torch.cat(out_frames, dim=0)  # [T,3,H,W]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_video_tensor(output_path, out, fps=cfg.target_fps)
    timing["write_output_sec"] = float(time.perf_counter() - t0)
    timing["total_iframe_sec"] = float(time.perf_counter() - t_all0)
    timing["output_num_frames"] = int(out.shape[0])

    return timing if return_timing else None


def run_interpolation_pipeline(
    *,
    video_path: str,
    output_path: str,
    cfg: PipelineConfig,
    log_file: Optional[str] = None,
) -> None:
    """
    兼容原用法：从视频文件读取 -> 调用 from_frames 版本。
    """
    frames, info = read_video_tensor(video_path)
    run_interpolation_pipeline_from_frames(
        frames=frames,
        fps_src=info.fps,
        output_path=output_path,
        cfg=cfg,
        log_file=log_file,
    )
