from __future__ import annotations

import random
from typing import List

import torch


def uniform_keyframes(frames: torch.Tensor, *, k: int) -> List[torch.Tensor]:
    """
    Uniformly sample k keyframes (always keeping first and last).
    frames: [T,3,H,W]. Returns list of [1,3,H,W] on CPU.
    """
    t = int(frames.shape[0])
    if k <= 2 or t <= 2:
        idxs = [0, t - 1] if t > 1 else [0]
    else:
        idxs = [round(i * (t - 1) / (k - 1)) for i in range(k)]
        idxs[0] = 0
        idxs[-1] = t - 1
        # Deduplicate while preserving order
        seen = set()
        idxs = [i for i in idxs if not (i in seen or seen.add(i))]

    return [frames[i].unsqueeze(0).cpu() for i in idxs]


def random_keyframes(frames: torch.Tensor, *, k: int, seed: int = 0) -> List[torch.Tensor]:
    """
    Randomly sample k keyframes (always including first and last).
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


