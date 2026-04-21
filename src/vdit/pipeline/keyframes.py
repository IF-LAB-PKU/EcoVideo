from __future__ import annotations

import random
from typing import List

import torch


def uniform_keyframes(frames: torch.Tensor, *, k: int) -> List[torch.Tensor]:
    """
    均匀取 k 帧作为关键帧（保留首尾）。
    frames: [T,3,H,W]
    返回 list of [1,3,H,W] on CPU
    """
    t = int(frames.shape[0])
    if k <= 2 or t <= 2:
        idxs = [0, t - 1] if t > 1 else [0]
    else:
        idxs = [round(i * (t - 1) / (k - 1)) for i in range(k)]
        idxs[0] = 0
        idxs[-1] = t - 1
        # 去重保持顺序
        seen = set()
        idxs = [i for i in idxs if not (i in seen or seen.add(i))]

    return [frames[i].unsqueeze(0).cpu() for i in idxs]


def random_keyframes(frames: torch.Tensor, *, k: int, seed: int = 0) -> List[torch.Tensor]:
    """
    随机取 k 帧作为关键帧（强制包含首尾）。
    """
    t = int(frames.shape[0])
    if t <= 2:
        return [frames[i].unsqueeze(0).cpu() for i in range(t)]

    rng = random.Random(seed)
    k = max(2, min(k, t))
    mid = list(range(1, t - 1))
    rng.shuffle(mid)
    pick = sorted([0] + mid[: (k - 2)] + [t - 1])
    return [frames[i].unsqueeze(0).cpu() for i in pick]


