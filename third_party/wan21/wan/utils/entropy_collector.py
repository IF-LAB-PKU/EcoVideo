from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EntropyCollector:
    enabled: bool = False
    mode: str = "mean"  # "last" | "mean" | "ema"
    ema_alpha: float = 0.6
    block_idx: int | None = -1  # collect from which block; -1 = last

    sum_frame: torch.Tensor | None = None
    ema_frame: torch.Tensor | None = None
    last_frame: torch.Tensor | None = None
    count: int = 0

    last_token: torch.Tensor | None = None  # [B, L]
    last_grid_sizes: torch.Tensor | None = None  # [B, 3]
    last_seq_lens: torch.Tensor | None = None  # [B]

    def reset(self):
        self.sum_frame = None
        self.ema_frame = None
        self.last_frame = None
        self.last_token = None
        self.last_grid_sizes = None
        self.last_seq_lens = None
        self.count = 0

    def add(
        self,
        ent_frame: torch.Tensor,
        ent_token: torch.Tensor | None = None,
        grid_sizes: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
    ):
        if not self.enabled:
            return
        ef = ent_frame.detach().float()
        self.last_frame = ef
        if self.sum_frame is None:
            self.sum_frame = ef.clone()
        else:
            self.sum_frame += ef
        self.count += 1

        if self.ema_frame is None:
            self.ema_frame = ef.clone()
        else:
            a = float(self.ema_alpha)
            self.ema_frame = a * ef + (1 - a) * self.ema_frame

        if ent_token is not None:
            self.last_token = ent_token.detach().float().clone()
        if grid_sizes is not None:
            self.last_grid_sizes = grid_sizes.detach().clone()
        if seq_lens is not None:
            self.last_seq_lens = seq_lens.detach().clone()

    def final(self) -> torch.Tensor:
        assert self.last_frame is not None
        if self.mode == "last":
            return self.last_frame
        if self.mode == "ema":
            assert self.ema_frame is not None
            return self.ema_frame
        assert self.sum_frame is not None and self.count > 0
        return self.sum_frame / float(self.count)
