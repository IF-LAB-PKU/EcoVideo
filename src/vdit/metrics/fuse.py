"""
5 信息融合打分：g(Ia, Ib)

用于 greedy_refine():
- score_fn(Ia, Ib) -> float，越大表示该区间越“需要插帧”

这里默认融合 5 个量（你提到的 5 信息融合）：
1) RAFT: S_flow（运动强度）
2) RAFT: conf（置信）
3) RAFT: occ_ratio（遮挡/不一致比例）
4) RGB: topk_rgb（局部变化）
5) EDEN: diff_eden（全局差异）

说明：
- greedy_refine 每次插入新帧后，会继续对 (Ia, Im) 和 (Im, Ib) 打分；
  因此 score_fn 必须能对“EDEN生成的新帧”也计算上述指标。
- 为减少重复计算，内部提供了轻量 cache（按 data_ptr() 组合）。
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

        # 4) RGB top-k
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

        # 归一化/压缩：避免 s_flow 数值范围过大主导
        # 经验做法：log1p 压缩 + 简单截断
        s_flow_n = float(torch.log1p(torch.tensor(s_flow)).item())
        s_rgb_n = min(1.0, max(0.0, s_rgb))
        s_eden_n = min(1.0, max(0.0, s_eden))

        # conf 越大越可信，但我们要“需要插帧”的紧急度：更低 conf => 更难/更不稳定 => 更应该插
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


