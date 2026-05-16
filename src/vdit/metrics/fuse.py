"""
5-signal fusion scoring: g(Ia, Ib)

Used by greedy_refine():
- score_fn(Ia, Ib) -> float, higher means the interval needs more interpolation

Fuses 5 signals by default:
1) RAFT: S_flow (motion magnitude)
2) RAFT: conf (confidence)
3) RAFT: occ_ratio (occlusion/inconsistency ratio)
4) RGB: topk_rgb (local change)
5) EDEN: diff_eden (global difference)

Notes:
- After greedy_refine inserts a new frame, it re-scores (Ia, Im) and (Im, Ib),
  so score_fn must handle EDEN-generated frames as well.
- A lightweight cache (keyed by data_ptr() pair) avoids redundant computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from vdit.flow.raft_estimator import RaftEstimator
from vdit.metrics.eden_diff import eden_diff
from vdit.metrics.rgb_topk import topk_rgb_diff


@dataclass(frozen=True)
class ScorerWeights:
    w_flow: float = 1.0
    w_conf: float = 0.5
    w_occ: float = 0.5
    w_rgb: float = 0.8
    w_eden: float = 0.8


class IntervalScorer:
    def __init__(
        self,
        *,
        raft: Optional[RaftEstimator] = None,
        weights: ScorerWeights = ScorerWeights(),
        topk_ratio: float = 0.1,
    ) -> None:
        self.raft = raft
        self.w = weights
        self.topk_ratio = float(topk_ratio)

        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache_val: Optional[float] = None

    @torch.no_grad()
    def __call__(self, frame0: torch.Tensor, frame1: torch.Tensor) -> float:
        key = (frame0.data_ptr(), frame1.data_ptr())
        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val

        # 4) RGB top-k diff
        s_rgb = topk_rgb_diff(frame0, frame1, topk_ratio=self.topk_ratio)

        # 5) EDEN diff
        s_eden = eden_diff(frame0, frame1)

        # 1-3) RAFT
        if self.raft is None:
            s_flow = 0.0
            s_conf = 0.0
            s_occ = 0.0
        else:
            m = self.raft(frame0, frame1)
            s_flow = float(m.s_flow)
            s_conf = float(m.conf)
            s_occ = float(m.occ_ratio)

        # Normalize/compress: prevent s_flow from dominating due to large range
        # Heuristic: log1p compression + simple clamping
        s_flow_n = float(torch.log1p(torch.tensor(s_flow)).item())
        s_rgb_n = min(1.0, max(0.0, s_rgb))
        s_eden_n = min(1.0, max(0.0, s_eden))

        # Higher conf = more reliable, but we want interpolation urgency: lower conf => harder/less stable => insert more
        s_conf_n = 1.0 - min(1.0, max(0.0, s_conf))
        s_occ_n = min(1.0, max(0.0, s_occ))

        score = (
            self.w.w_flow * s_flow_n
            + self.w.w_conf * s_conf_n
            + self.w.w_occ * s_occ_n
            + self.w.w_rgb * s_rgb_n
            + self.w.w_eden * s_eden_n
        )

        out = float(score)
        self._cache_key = key
        self._cache_val = out
        return out


