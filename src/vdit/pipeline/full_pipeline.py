# src/vdit/pipeline/full_pipeline.py

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json
import time

import torch

from vdit.generators.base import create_generator
from vdit.pipeline.run_iframe import PipelineConfig, run_interpolation_pipeline_from_frames
from vdit.pipeline.video_io import read_video_tensor, write_video_tensor


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _append_jsonl(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_csv(path: Optional[str], row: Dict[str, Any], fieldnames: list) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    file_exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in fieldnames})


@dataclass(frozen=True)
class FullPipelineConfig:
    # wan: historical field name, holds any generator cfg (WAN/CogVideo/LTX)
    wan: Any
    iframe: PipelineConfig
    generator_name: str = "wan"

    # cloud-only mode
    stop_after_wan: bool = False
    save_keyframes_video_path: Optional[str] = None

    # metrics append
    sample_id: Optional[str] = None
    cloud_metrics_jsonl_path: Optional[str] = None
    cloud_metrics_csv_path: Optional[str] = None


@torch.no_grad()
def run_full_pipeline(
    *,
    prompt: Optional[str] = None,
    wan_ckpt_dir: Optional[str] = None,
    input_video: Optional[str] = None,
    input_fps: Optional[float] = None,
    output_path: str,
    cfg: FullPipelineConfig,
    log_file: Optional[str] = None,
    save_sampled_video_path: Optional[str] = None,
    save_keyframes_video_path: Optional[str] = None,
    save_wan_video_path: Optional[str] = None,
    generate_wan_full_baseline: bool = False,
    save_wan_full_baseline_video_path: Optional[str] = None,
    metrics_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    if metrics_json_path is None:
        base = os.path.splitext(output_path)[0]
        metrics_json_path = base + ".metrics.json"

    t_pipeline0 = time.perf_counter()
    metrics: Dict[str, Any] = {
        "output_path": output_path,
        "input_video": input_video,
        "wan_ckpt_dir": wan_ckpt_dir,
        "prompt": prompt,
        "generator_name": cfg.generator_name,
        "wan_cfg": asdict(cfg.wan) if hasattr(cfg.wan, "__dataclass_fields__") else dict(vars(cfg.wan)),
        "iframe_cfg": {
            "eden_config": cfg.iframe.eden_config,
            "raft_ckpt": cfg.iframe.raft_ckpt,
            "raft_device": cfg.iframe.raft_device,
            "eden_device": cfg.iframe.eden_device,
            "use_split_gpu": cfg.iframe.use_split_gpu,
            "target_fps": cfg.iframe.target_fps,
            "keyframe_mode": cfg.iframe.keyframe_mode,
            "keyframes_k": cfg.iframe.keyframes_k,
            "seed": cfg.iframe.seed,
            "topk_ratio": cfg.iframe.topk_ratio,
        },
        "timing": {},
        "speedup": {},
    }

    # -------- 从已有视频开始：只插帧，不做 baseline --------
    if input_video:
        t0 = time.perf_counter()
        frames, info = read_video_tensor(input_video)
        fps_src = float(input_fps) if input_fps is not None else float(info.fps)
        metrics["timing"]["read_input_video_sec"] = float(time.perf_counter() - t0)
        metrics["io"] = {"fps_src": float(fps_src), "num_frames": int(frames.shape[0])}

        sampled_video_path = save_sampled_video_path or save_wan_video_path
        if sampled_video_path:
            _ensure_parent_dir(sampled_video_path)
            write_video_tensor(sampled_video_path, frames, fps=float(fps_src))

        t0 = time.perf_counter()
        iframe_timing = run_interpolation_pipeline_from_frames(
            frames=frames,
            fps_src=fps_src,
            output_path=output_path,
            cfg=cfg.iframe,
            log_file=log_file,
            save_keyframes_video_path=save_keyframes_video_path,
            return_timing=True,
        ) or {}
        metrics["timing"]["iframe_wall_sec"] = float(time.perf_counter() - t0)
        metrics["iframe_timing"] = iframe_timing

        metrics["timing"]["pipeline_total_sec"] = float(time.perf_counter() - t_pipeline0)
        _write_json(metrics_json_path, metrics)
        return metrics

    # -------- baseline WAN full generation（可选，仅 WAN）--------
    baseline_wan_timing = None
    if generate_wan_full_baseline and cfg.generator_name == "wan":
        if prompt is None or wan_ckpt_dir is None:
            raise ValueError("generate_wan_full_baseline=True requires prompt + wan_ckpt_dir")

        baseline_debug_dir = None
        if cfg.wan.debug_dir:
            baseline_debug_dir = os.path.join(str(cfg.wan.debug_dir), "baseline_full")

        from vdit.generators.wan_t2v import WanGenerateConfig
        wan_baseline_cfg = replace(
            cfg.wan,
            keyframe_by_entropy=False,
            out_fps=None,
            debug_dir=baseline_debug_dir,
        )

        t0 = time.perf_counter()
        gen_base = create_generator(cfg.generator_name, ckpt_dir=wan_ckpt_dir, cfg=wan_baseline_cfg)
        baseline_frames, baseline_fps = gen_base.generate(prompt)
        metrics["timing"]["wan_full_baseline_wall_sec"] = float(time.perf_counter() - t0)
        metrics["baseline"] = {
            "fps": float(baseline_fps),
            "num_frames": int(baseline_frames.shape[0]),
            "debug_dir": baseline_debug_dir,
        }

        if baseline_debug_dir:
            baseline_wan_timing = _read_json_if_exists(os.path.join(baseline_debug_dir, "timing.json"))
            if baseline_wan_timing is not None:
                metrics["baseline"]["wan_internal_timing"] = baseline_wan_timing

        if save_wan_full_baseline_video_path:
            _ensure_parent_dir(save_wan_full_baseline_video_path)
            write_video_tensor(save_wan_full_baseline_video_path, baseline_frames, fps=float(baseline_fps))

    # -------- main WAN generation（你的关键帧/Method2/完整等）--------
    if prompt is None or wan_ckpt_dir is None:
        raise ValueError("Must provide either (input_video) or (prompt + wan_ckpt_dir)")

    t0 = time.perf_counter()
    generator = create_generator(cfg.generator_name, ckpt_dir=wan_ckpt_dir, cfg=cfg.wan)
    frames, fps_src = generator.generate(prompt)
    metrics["timing"]["wan_main_wall_sec"] = float(time.perf_counter() - t0)
    metrics["io"] = {"fps_src": float(fps_src), "num_frames": int(frames.shape[0])}

    # -------- NEW: save keyframes video (cloud artifact) --------
    # For cloud-only mode, the WAN output is treated as "keyframes video" delivered to edge.
    keyframes_path = cfg.save_keyframes_video_path
    if keyframes_path:
        _ensure_parent_dir(keyframes_path)
        write_video_tensor(keyframes_path, frames, fps=float(fps_src))
        try:
            metrics.setdefault("io", {})
            metrics["io"]["keyframes_video_path"] = keyframes_path
            metrics["io"]["keyframes_file_bytes"] = int(os.path.getsize(keyframes_path))
        except Exception:
            pass

    # -------- NEW: stop after WAN (cloud-only) --------
    if cfg.stop_after_wan:
        # define sample_id
        if cfg.sample_id:
            sid = cfg.sample_id
        elif keyframes_path:
            sid = Path(keyframes_path).stem
        else:
            sid = f"sample_{int(time.time())}"

        cloud_record: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "role": "cloud",
            "sample_id": sid,
            "prompt": prompt,
            "wan_ckpt_dir": wan_ckpt_dir,
            "generator_name": cfg.generator_name,
            "cfg": {
                "wan_task": getattr(cfg.wan, "task", None),
                "wan_size": getattr(cfg.wan, "size", None),
                "wan_frame_num": getattr(cfg.wan, "frame_num", getattr(cfg.wan, "num_frames", None)),
                "wan_seed": getattr(cfg.wan, "seed", None),
                "keyframe_by_entropy": getattr(cfg.wan, "keyframe_by_entropy", None),
                "keyframe_topk": getattr(cfg.wan, "keyframe_topk", None),
                "keyframe_out_fps": getattr(cfg.wan, "keyframe_out_fps", None),
                "keyframe_target_fps": getattr(cfg.wan, "keyframe_target_fps", None),
                "nonkey_update_mode": getattr(cfg.wan, "nonkey_update_mode", None),
                "teacache_rel_l1_thresh": getattr(cfg.wan, "teacache_rel_l1_thresh", None),
                "teacache_max_skip": getattr(cfg.wan, "teacache_max_skip", None),
                "teacache_warmup": getattr(cfg.wan, "teacache_warmup", None),
            },
            "timing": {
                "cloud_latency_sec": float(metrics["timing"]["wan_main_wall_sec"]),
                "wan_main_wall_sec": float(metrics["timing"]["wan_main_wall_sec"]),
            },
            "io": {
                "fps_keyframes": float(fps_src),
                "num_frames_keyframes": int(frames.shape[0]),
                "keyframes_video_path": keyframes_path,
                "keyframes_file_bytes": int(metrics.get("io", {}).get("keyframes_file_bytes", 0)),
            },
        }

        _append_jsonl(cfg.cloud_metrics_jsonl_path, cloud_record)

        csv_fields = [
            "ts", "role", "sample_id",
            "cloud_latency_sec",
            "fps_keyframes", "num_frames_keyframes", "keyframes_file_bytes",
            "wan_task", "wan_size", "wan_frame_num", "wan_seed",
            "keyframe_by_entropy", "keyframe_topk", "keyframe_out_fps", "keyframe_target_fps",
            "nonkey_update_mode",
            "prompt",
            "keyframes_video_path",
        ]
        csv_row = {
            "ts": cloud_record["ts"],
            "role": "cloud",
            "sample_id": sid,
            "cloud_latency_sec": cloud_record["timing"]["cloud_latency_sec"],
            "fps_keyframes": cloud_record["io"]["fps_keyframes"],
            "num_frames_keyframes": cloud_record["io"]["num_frames_keyframes"],
            "keyframes_file_bytes": cloud_record["io"]["keyframes_file_bytes"],
            "wan_task": cloud_record["cfg"]["wan_task"],
            "wan_size": cloud_record["cfg"]["wan_size"],
            "wan_frame_num": cloud_record["cfg"]["wan_frame_num"],
            "wan_seed": cloud_record["cfg"]["wan_seed"],
            "keyframe_by_entropy": cloud_record["cfg"]["keyframe_by_entropy"],
            "keyframe_topk": cloud_record["cfg"]["keyframe_topk"],
            "keyframe_out_fps": cloud_record["cfg"]["keyframe_out_fps"],
            "keyframe_target_fps": cloud_record["cfg"]["keyframe_target_fps"],
            "nonkey_update_mode": cloud_record["cfg"]["nonkey_update_mode"],
            "prompt": cloud_record["prompt"],
            "keyframes_video_path": keyframes_path,
        }
        _append_csv(cfg.cloud_metrics_csv_path, csv_row, csv_fields)

        metrics["timing"]["pipeline_total_sec"] = float(time.perf_counter() - t_pipeline0)
        _write_json(metrics_json_path, metrics)
        return metrics

    if getattr(cfg.wan, "debug_dir", None):
        wan_internal_timing = _read_json_if_exists(os.path.join(str(getattr(cfg.wan, "debug_dir")), "timing.json"))
        if wan_internal_timing is not None:
            metrics["wan_internal_timing"] = wan_internal_timing

    sampled_video_path = save_sampled_video_path or save_wan_video_path
    if sampled_video_path:
        _ensure_parent_dir(sampled_video_path)
        write_video_tensor(sampled_video_path, frames, fps=float(fps_src))

    # -------- iframe（插帧阶段）--------
    t0 = time.perf_counter()
    iframe_timing = run_interpolation_pipeline_from_frames(
        frames=frames,
        fps_src=fps_src,
        output_path=output_path,
        cfg=cfg.iframe,
        log_file=log_file,
        save_keyframes_video_path=(None if cfg.save_keyframes_video_path else save_keyframes_video_path),
        return_timing=True,
    ) or {}
    metrics["timing"]["iframe_wall_sec"] = float(time.perf_counter() - t0)
    metrics["iframe_timing"] = iframe_timing

    metrics["timing"]["pipeline_total_sec"] = float(time.perf_counter() - t_pipeline0)

    # -------- speedup / time ratio --------
    speedup: Dict[str, Any] = {}
    wan_main = float(metrics["timing"].get("wan_main_wall_sec", 0.0))
    iframe_wall = float(metrics["timing"].get("iframe_wall_sec", 0.0))
    total = float(metrics["timing"].get("pipeline_total_sec", 0.0))
    speedup["wan_main_over_iframe"] = (wan_main / iframe_wall) if iframe_wall > 0 else None
    speedup["iframe_over_wan_main"] = (iframe_wall / wan_main) if wan_main > 0 else None
    speedup["total_over_wan_main"] = (total / wan_main) if wan_main > 0 else None

    if generate_wan_full_baseline and metrics["timing"].get("wan_full_baseline_wall_sec", 0.0) > 0:
        wan_full_wall = float(metrics["timing"]["wan_full_baseline_wall_sec"])
        speedup["wan_full_wall_over_wan_main_wall"] = (wan_full_wall / wan_main) if wan_main > 0 else None
        speedup["wan_main_wall_over_wan_full_wall"] = (wan_main / wan_full_wall) if wan_full_wall > 0 else None

    # 更公平：用 WAN 内部 timing.json 的 denoise_sec 做 speedup
    try:
        if baseline_wan_timing and ("denoise_sec" in baseline_wan_timing) and ("wan_internal_timing" in metrics):
            den_full = float(baseline_wan_timing.get("denoise_sec", 0.0))
            den_main = float(metrics["wan_internal_timing"].get("denoise_sec", 0.0))
            if den_full > 0 and den_main > 0:
                speedup["wan_full_denoise_over_wan_main_denoise"] = den_full / den_main
                speedup["wan_main_denoise_over_wan_full_denoise"] = den_main / den_full
    except Exception:
        pass

    metrics["speedup"] = speedup

    _write_json(metrics_json_path, metrics)
    return metrics
