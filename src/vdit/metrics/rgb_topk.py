from __future__ import annotations

import torch


@torch.no_grad()
def topk_rgb_diff(frame0: torch.Tensor, frame1: torch.Tensor, *, topk_ratio: float = 0.1) -> float:
    """
    局部“纹理/变化”指标：取像素绝对差的 top-k 均值。
    输入 frame: [1,3,H,W] in [0,1]
    """
    delta = (frame0 - frame1).abs().mean(dim=1)  # [1,H,W]
    flat = delta.reshape(-1)
    k = max(1, int(topk_ratio * flat.numel()))
    vals, _ = torch.topk(flat, k)
    return float(vals.mean().item())


