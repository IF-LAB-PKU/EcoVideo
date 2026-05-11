"""
Lightweight warp / grid utilities (torch-only).

Purpose:
- RAFT forward-back consistency requires warping f21 to sample at (x + f12(x)).
- Avoids hard scipy dependency in RAFT utils.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coords_grid(batch: int, ht: int, wd: int, device: torch.device) -> torch.Tensor:
    """Return pixel coordinate grid, shape [B,2,H,W], channels are (x,y)."""
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
    coords_xy: [B,H,W,2], pixel coordinates (x,y)
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


