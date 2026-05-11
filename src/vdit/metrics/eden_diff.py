from __future__ import annotations

import torch


@torch.no_grad()
def eden_diff(frame0: torch.Tensor, frame1: torch.Tensor) -> float:
    “””
    EDEN-style difference signal (mean cosine similarity).
    Lower similarity = more motion. Output is motion-like: 1 - clamp(cos_mean,0,1).
    “””
    cos = torch.cosine_similarity(frame0, frame1, dim=1)  # [1,H,W]
    cos_mean = float(cos.mean().item())
    cos_mean = max(0.0, min(1.0, cos_mean))
    return float(1.0 - cos_mean)


