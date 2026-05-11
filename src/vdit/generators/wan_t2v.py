# src/vdit/generators/wan_t2v.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from pathlib import Path

import math
import torch

from vdit.generators.base import register_generator
from vdit.generators.wan_pkg_loader import load_wan_package

FrameSampleMode = Literal["uniform", "random", "stratified_random"]


def auto_keyframe_topk(
    frame_num_full: int,
    fps_full: float,
    fps_key: float,
    stride_t: int,
    min_k: int = 2,
    max_k: Optional[int] = None,
) -> int:
    """
    Auto-estimate latent keyframe count K from the target keyframe fps.
    """
    if fps_key <= 0:
        raise ValueError(f"fps_key must be > 0, got {fps_key}")
    if frame_num_full <= 1:
        return max(min_k, 1)

    t_key = int(round(frame_num_full * (fps_key / float(fps_full))))
    t_key = max(2, t_key)

    k = int(round((t_key - 1) / float(stride_t))) + 1
    k = max(min_k, k)
    if max_k is not None:
        k = min(max_k, k)
    return k


@dataclass(frozen=True)
class WanGenerateConfig:
    # -------- WAN basic parameters --------
    wan_version: str = "2.1"  # "2.1" or "2.2"
    task: str = "t2v-1.3B"  # baseline: wan1.3b
    size: str = "832*480"  # 1.3B supports: 480*832 / 832*480
    frame_num: int = 81  # WAN default frame count (typically 4n+1)

    # -------- Sampling parameters (aligned with wan-main/generate.py) --------
    sample_solver: str = "unipc"  # ["unipc", "dpm++"]
    sample_steps: int = 50  # t2v default 50
    sample_shift: float = 5.0
    guide_scale: float = 5.0
    seed: int = 0
    offload_model: bool = True  # Default True for single GPU to save VRAM

    # -------- Optional: uniform/random frame sampling after generation (keep duration unchanged) --------
    out_fps: Optional[float] = None  # e.g. 8 / 12; None means no downsampling
    frame_sample: FrameSampleMode = "uniform"
    frame_sample_seed: int = 0

    # -------- Keyframe-by-entropy (true latent temporal dimension pruning) --------
    keyframe_by_entropy: bool = False
    entropy_steps: int = 5
    entropy_mode: str = "mean"  # "last" | "mean" | "ema"
    entropy_ema_alpha: float = 0.6
    entropy_block_idx: int = -1  # -1 = last block
    keyframe_topk: int = 16
    keyframe_cover: bool = True
    keyframe_select_mode: str = "entropy"  # "entropy" | "uniform" | "random" (ablation)
    keyframe_sample_seed: int = 0
    use_nonkey_context: bool = True
    debug_dir: Optional[str] = None
    save_debug_pt: bool = True
    profile_timing: bool = True
    # -------- analysis-only: trace one random latent frame across steps --------
    trace_frame_metrics: bool = False
    trace_steps: int = 50
    trace_frame_seed: int = 2026
    keyframe_out_fps: Optional[float] = None
    keyframe_target_fps: Optional[float] = 8.0

    # -------- method-2: non-key low-frequency compute (TeaCache-style) --------
    nonkey_update_mode: str = "none"
    nonkey_update_interval: int = 5
    teacache_rel_l1_thresh: float = 0.02
    teacache_max_skip: int = 10
    teacache_warmup: int = 0
    save_teacache_trace_png: bool = True

    # -------- Device settings --------
    device_id: int = 0  # Aligned with wan-main/generate.py device_id (int)
    t5_cpu: bool = False  # Set True if VRAM is tight (slower)


def _wan_video_to_vdit_frames(video: torch.Tensor) -> torch.Tensor:
    """
    WAN:   [C,T,H,W] float, typical value range ~[-1,1]
    VDiT:  [T,3,H,W] float, value range [0,1]
    """
    if video.ndim != 4:
        raise ValueError(f"Expect WAN video [C,T,H,W], got {tuple(video.shape)}")
    c, _t, _h, _w = video.shape
    if c != 3:
        raise ValueError(f"Expect C=3 RGB video, got C={c}")

    # [-1,1] -> [0,1]
    frames = (video.clamp(-1.0, 1.0) + 1.0) * 0.5
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W] -> [T,3,H,W]
    return frames


@torch.no_grad()
def generate_wan_frames(
    *,
    prompt: str,
    ckpt_dir: str,
    cfg: WanGenerateConfig,
) -> Tuple[torch.Tensor, float]:
    """
    Returns:
      frames: [T,3,H,W] float in [0,1] (CPU tensor)
      fps:    float (post-sampling fps; defaults to WAN config sample_fps if out_fps is not set)
    """
    _root = Path(__file__).resolve().parents[3]
    wan_mod = load_wan_package(_root, cfg.wan_version)

    WAN_CONFIGS = wan_mod.configs.WAN_CONFIGS  # type: ignore[attr-defined]
    SIZE_CONFIGS = wan_mod.configs.SIZE_CONFIGS  # type: ignore[attr-defined]

    if cfg.task not in WAN_CONFIGS:
        raise ValueError(
            f"[wan{cfg.wan_version}] Unknown task: {cfg.task}. "
            f"Valid: {list(WAN_CONFIGS.keys())}"
        )
    if cfg.size not in SIZE_CONFIGS:
        raise ValueError(
            f"[wan{cfg.wan_version}] Unknown size: {cfg.size}. "
            f"Valid: {list(SIZE_CONFIGS.keys())}"
        )

    wan_cfg = WAN_CONFIGS[cfg.task]
    fps_src = float(getattr(wan_cfg, "sample_fps", 24.0))
    fps_tgt = fps_src

    # ----- Build WAN pipeline (single GPU, non-distributed: rank=0) -----
    if cfg.wan_version == "2.1":
        model = wan_mod.WanT2V(  # type: ignore[attr-defined]
            config=wan_cfg,
            checkpoint_dir=ckpt_dir,
            device_id=cfg.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=cfg.t5_cpu,
        )

        keyframe_topk = cfg.keyframe_topk
        keyframe_out_fps = cfg.keyframe_out_fps
        if cfg.keyframe_by_entropy and cfg.keyframe_target_fps is not None:
            stride_t = int(
                getattr(model, "vae_stride", [4])[0]  # type: ignore[index]
            )
            max_k = (cfg.frame_num - 1) // stride_t + 1
            keyframe_topk = auto_keyframe_topk(
                frame_num_full=cfg.frame_num,
                fps_full=float(fps_src),
                fps_key=float(cfg.keyframe_target_fps),
                stride_t=stride_t,
                min_k=2,
                max_k=max_k,
            )
            if keyframe_out_fps is None:
                keyframe_out_fps = float(cfg.keyframe_target_fps)

        # Generate WAN2.1 native video tensor: [C,T,H,W]
        video = model.generate(
            prompt,
            size=SIZE_CONFIGS[cfg.size],
            frame_num=cfg.frame_num,
            shift=cfg.sample_shift,
            sample_solver=cfg.sample_solver,
            sampling_steps=cfg.sample_steps,
            guide_scale=cfg.guide_scale,
            seed=cfg.seed,
            offload_model=cfg.offload_model,
            keyframe_by_entropy=cfg.keyframe_by_entropy,
            keyframe_select_mode=cfg.keyframe_select_mode,
            keyframe_sample_seed=cfg.keyframe_sample_seed,
            entropy_steps=cfg.entropy_steps,
            entropy_mode=cfg.entropy_mode,
            entropy_ema_alpha=cfg.entropy_ema_alpha,
            entropy_block_idx=cfg.entropy_block_idx,
            keyframe_topk=keyframe_topk,
            keyframe_cover=cfg.keyframe_cover,
            use_nonkey_context=cfg.use_nonkey_context,
            debug_dir=cfg.debug_dir,
            save_debug_pt=cfg.save_debug_pt,
            profile_timing=cfg.profile_timing,
            trace_frame_metrics=cfg.trace_frame_metrics,
            trace_steps=cfg.trace_steps,
            trace_frame_seed=cfg.trace_frame_seed,
            nonkey_update_mode=cfg.nonkey_update_mode,
            nonkey_update_interval=cfg.nonkey_update_interval,
            teacache_rel_l1_thresh=cfg.teacache_rel_l1_thresh,
            teacache_max_skip=cfg.teacache_max_skip,
            teacache_warmup=cfg.teacache_warmup,
            save_teacache_trace_png=cfg.save_teacache_trace_png,
        )

        if cfg.keyframe_by_entropy:
            t_out = int(video.shape[1])
            t_full = int(cfg.frame_num)
            if t_full > 0:
                fps_tgt = fps_src * (t_out / float(t_full))
            if keyframe_out_fps is not None:
                fps_tgt = float(keyframe_out_fps)

    elif cfg.wan_version == "2.2":
        if cfg.nonkey_update_mode != "none":
            raise ValueError(
                "wan2.2 path does not support nonkey_update_mode; "
                "please set --wan_nonkey_update_mode none"
            )

        # Wan2.2 A14B pipeline (no method-2 support, entropy keyframe selection only)
        model = wan_mod.WanT2V(  # type: ignore[attr-defined]
            config=wan_cfg,
            checkpoint_dir=ckpt_dir,
            device_id=cfg.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=cfg.t5_cpu,
            convert_model_dtype=False,
        )

        target_fps = float(cfg.keyframe_target_fps or 8.0)

        video = model.generate(
            input_prompt=prompt,
            size=SIZE_CONFIGS[cfg.size],
            frame_num=cfg.frame_num,
            shift=cfg.sample_shift,
            sample_solver=cfg.sample_solver,
            sampling_steps=cfg.sample_steps,
            guide_scale=cfg.guide_scale,
            n_prompt="",
            seed=cfg.seed,
            offload_model=cfg.offload_model,
            keyframe_by_entropy=cfg.keyframe_by_entropy,
            keyframe_target_fps=target_fps,
            entropy_steps=cfg.entropy_steps,
            entropy_mode=cfg.entropy_mode,
            entropy_ema_alpha=cfg.entropy_ema_alpha,
            entropy_block_idx=cfg.entropy_block_idx,
            keyframe_cover=cfg.keyframe_cover,
            debug_dir=cfg.debug_dir,
            save_debug_pt=cfg.save_debug_pt,
            profile_timing=cfg.profile_timing,
            trace_frame_metrics=cfg.trace_frame_metrics,
            trace_steps=cfg.trace_steps,
            trace_frame_seed=cfg.trace_frame_seed,
        )

        if cfg.keyframe_by_entropy:
            fps_tgt = target_fps

    else:
        raise ValueError(f"Unsupported wan_version: {cfg.wan_version}")

    # Optional: uniform/random frame sampling by out_fps (keep duration, wan2.1 only)
    if (cfg.wan_version == "2.1" and cfg.out_fps is not None
            and not cfg.keyframe_by_entropy):
        fps_tgt = float(cfg.out_fps)
        if fps_tgt > 0 and abs(fps_tgt - fps_src) > 1e-6:
            # Lazy-import resample function from the wan package to avoid cross-version conflicts
            try:
                fs_mod = wan_mod.utils.frame_sampling  # type: ignore[attr-defined]
                resample_fn = fs_mod.resample_video_tensor
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "resample_video_tensor is not available in wan2.1 package."
                ) from exc

            video, _idx = resample_fn(
                video,
                fps_src=fps_src,
                fps_tgt=fps_tgt,
                mode=cfg.frame_sample,
                seed=cfg.frame_sample_seed,
            )

    frames = _wan_video_to_vdit_frames(video).float().cpu()

    # Proactively free WAN VRAM (critical for sequential pipelines)
    del video
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Ensure returned fps is Python float (not numpy.float64)
    return frames, float(fps_tgt)


@register_generator("wan")
class WanGenerator:
    def __init__(self, ckpt_dir: str, cfg: WanGenerateConfig):
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[torch.Tensor, float]:
        return generate_wan_frames(prompt=prompt, ckpt_dir=self.ckpt_dir, cfg=self.cfg)
