"""
轻量 warp / grid 工具（torch-only）。

用途：
- RAFT forward-back consistency 需要把 f21 warp 到 (x + f12(x)) 上取样。
- 避免依赖 RAFT utils 里对 scipy 的硬依赖。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coords_grid(batch: int, ht: int, wd: int, device: torch.device) -> torch.Tensor:
    """返回像素坐标网格，shape [B,2,H,W]，通道为 (x,y)。"""
    yy, xx = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).float()  # [2,H,W]
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sample(
    img: torch.Tensor,
    coords_xy: torch.Tensor,
    *,
    align_corners: bool = True,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    img: [B,C,H,W]
    coords_xy: [B,H,W,2]，像素坐标系 (x,y)
    """
    b, _, h, w = img.shape
    x = coords_xy[..., 0]
    y = coords_xy[..., 1]

    # pixel -> normalized [-1,1]
    x = 2.0 * x / max(w - 1, 1) - 1.0
    y = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1)  # [B,H,W,2]
    return F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


