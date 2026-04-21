from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import os
import math
import fractions

import torch
import torchvision


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    height: int
    width: int


def read_video_tensor(path: str) -> Tuple[torch.Tensor, VideoInfo]:
    """
    返回 frames: [T,3,H,W] float in [0,1]
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    try:
        frames, _, info = torchvision.io.read_video(path, pts_unit="sec")
    except Exception as e:
        raise RuntimeError(f"Failed to read video from {path}: {e}") from e
    
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    fps = float(info["video_fps"])
    t, _, h, w = frames.shape
    return frames, VideoInfo(fps=fps, num_frames=t, height=h, width=w)


def _sanitize_fps_for_pyav(fps: float) -> int | fractions.Fraction:
    """
    PyAV 接受 int 或 Fraction（有 numerator/denominator）。
    numpy.float64 会导致 AttributeError。
    """
    fps_py = float(fps)
    if not math.isfinite(fps_py) or fps_py <= 0:
        raise ValueError(f"Invalid fps: {fps_py}")

    fps_round = int(round(fps_py))
    if abs(fps_py - fps_round) < 1e-6:
        return fps_round

    return fractions.Fraction(fps_py).limit_denominator(100000)


def write_video_tensor(path: str, frames: torch.Tensor, *, fps: float) -> None:
    """
    frames: [T,3,H,W] float in [0,1]
    """
    import av

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    frames_u8 = (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    frames_u8 = frames_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    t, h, w, _ = frames_u8.shape
    fps_safe = _sanitize_fps_for_pyav(fps)
    if isinstance(fps_safe, int):
        time_base = fractions.Fraction(1, fps_safe)
    else:
        time_base = fractions.Fraction(fps_safe.denominator, fps_safe.numerator)

    with av.open(path, mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=fps_safe)
        except av.AVError:
            stream = container.add_stream("h264", rate=fps_safe)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base

        for i in range(t):
            frame = av.VideoFrame.from_ndarray(frames_u8[i], format="rgb24")
            frame = frame.reformat(width=w, height=h, format="yuv420p")
            frame.pts = i
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
