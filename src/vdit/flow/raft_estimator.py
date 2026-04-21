"""
RAFT 光流封装：提供
- 前向/后向光流
- 运动强度 S_flow
- 置信度 conf（forward-back consistency）
- 遮挡比例 occ_ratio

输入输出约定：
- Ia, Ib: torch.Tensor [1,3,H,W], float, 值域 [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import os

from vdit.flow.warp import bilinear_sample, coords_grid

# third_party/raft
from third_party.raft.core.raft import RAFT  # type: ignore
from third_party.raft.core.utils_core.utils import InputPadder  # type: ignore


@dataclass(frozen=True)
class RaftMetrics:
    flow_fwd: torch.Tensor  # [1,2,H,W]
    flow_bwd: torch.Tensor  # [1,2,H,W]
    s_flow: float
    conf: float
    occ_ratio: float


class RaftEstimator:
    """
    最小可用封装：
    - 不走 demo.py 的 sys.path
    - 不强制依赖 scipy（我们不用 forward_interpolate）
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str | torch.device = "cuda:0",
        iters: int = 20,
        small: bool = False,
        mixed_precision: bool = False,
        alternate_corr: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.iters = int(iters)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RAFT 权重文件不存在：{model_path!r}\n"
                "请确认路径正确，或通过命令行参数 --raft_ckpt 指向真实的 raft-things.pth。"
            )

        args = SimpleNamespace(
            small=bool(small),
            mixed_precision=bool(mixed_precision),
            alternate_corr=bool(alternate_corr),
        )

        model = RAFT(args)
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # 兼容 DataParallel 的前缀
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model = model

        # 轻量缓存：避免 greedy_refine 对同一对帧重复算多次
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache_val: Optional[RaftMetrics] = None

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> RaftMetrics:
        key = (img1.data_ptr(), img2.data_ptr())
        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val

        f12 = self._infer_flow(img1, img2)
        f21 = self._infer_flow(img2, img1)

        s_flow = float(torch.sqrt(f12[:, 0:1] ** 2 + f12[:, 1:2] ** 2).mean().item())

        conf, occ_ratio = self._consistency_metrics(f12, f21)

        out = RaftMetrics(flow_fwd=f12, flow_bwd=f21, s_flow=s_flow, conf=conf, occ_ratio=occ_ratio)
        self._cache_key = key
        self._cache_val = out
        return out

    @torch.no_grad()
    def _infer_flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        RAFT 输入通常按 [0,255]；这里从 [0,1] 转换。
        返回 flow_up: [1,2,H,W]（unpad 后）
        """
        i1 = (img1.to(self.device) * 255.0).contiguous()
        i2 = (img2.to(self.device) * 255.0).contiguous()

        padder = InputPadder(i1.shape)
        i1p, i2p = padder.pad(i1, i2)
        _, flow_up = self.model(i1p, i2p, iters=self.iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
        return flow_up

    @torch.no_grad()
    def _consistency_metrics(self, f12: torch.Tensor, f21: torch.Tensor) -> Tuple[float, float]:
        """
        forward-back consistency（够用且稳定的实现）：
        - x2 = x + f12(x)
        - f21_warped(x) = f21(x2)
        - err = ||f12 + f21_warped||
        - occ: err > alpha*(||f12||+||f21_warped||)+beta
        - conf: mean(exp(-err/tau))
        """
        b, _, h, w = f12.shape
        device = f12.device

        coords = coords_grid(b, h, w, device)  # [B,2,H,W]
        x2 = coords + f12
        x2_hw2 = x2.permute(0, 2, 3, 1)  # [B,H,W,2]

        f21_warped = bilinear_sample(f21, x2_hw2, align_corners=True, padding_mode="zeros")

        err = torch.sqrt(((f12 + f21_warped) ** 2).sum(dim=1, keepdim=True) + 1e-6)  # [B,1,H,W]
        mag12 = torch.sqrt((f12 ** 2).sum(dim=1, keepdim=True) + 1e-6)
        mag21w = torch.sqrt((f21_warped ** 2).sum(dim=1, keepdim=True) + 1e-6)

        alpha = 0.01
        beta = 0.5
        occ = err > (alpha * (mag12 + mag21w) + beta)
        occ_ratio = float(occ.float().mean().item())

        tau = 3.0
        conf = float(torch.exp(-err / tau).mean().item())
        return conf, occ_ratio


