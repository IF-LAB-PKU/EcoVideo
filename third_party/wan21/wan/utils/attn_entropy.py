from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def temporal_attn_entropy_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
    eps: float = 1e-8,
    use_heads: bool = False,
) -> torch.Tensor:
    """
    Compute temporal attention entropy for each token by attending to frame-level keys.

    This avoids full token-to-token attention (O(L^2)) by pooling keys per frame.
    q, k: [B, L, H, D]
    grid_sizes: [B, 3] = (F, H, W) in patch grid
    seq_lens: [B] or None
    return: entropy_per_token [B, L]
    """
    b, l, h, d = q.shape
    scale = 1.0 / math.sqrt(d)
    ent_all = []

    for i in range(b):
        f_g, h_g, w_g = grid_sizes[i].tolist()
        valid = int(seq_lens[i].item()) if seq_lens is not None else f_g * h_g * w_g
        valid = min(valid, f_g * h_g * w_g)

        qi = q[i, :valid].float().view(f_g, h_g * w_g, h, d)
        ki = k[i, :valid].float().view(f_g, h_g * w_g, h, d)

        if use_heads:
            k_frame = ki.mean(dim=1)  # [F, H, D]
            logits = torch.einsum("fshd,khd->fshk", qi, k_frame) * scale
            attn = F.softmax(logits, dim=-1).clamp(min=eps)
            ent = -(attn * torch.log(attn)).sum(dim=-1)  # [F, S, H]
            ent = ent.mean(dim=-1)  # [F, S]
        else:
            q_mean = qi.mean(dim=2)  # [F, S, D]
            k_frame = ki.mean(dim=1).mean(dim=1)  # [F, D]
            logits = torch.einsum("fsd,kd->fsk", q_mean, k_frame) * scale
            attn = F.softmax(logits, dim=-1).clamp(min=eps)
            ent = -(attn * torch.log(attn)).sum(dim=-1)  # [F, S]

        ent_flat = ent.reshape(-1)
        ent_full = q.new_zeros(l)
        ent_full[:valid] = ent_flat
        ent_all.append(ent_full)

    return torch.stack(ent_all, dim=0)


@torch.no_grad()
def token_entropy_to_frame(
    ent_tok: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """
    ent_tok: [B, L]
    grid_sizes: [B, 3] = (F, H, W) in patch grid
    return: ent_frame [B, F]
    """
    b, _ = ent_tok.shape
    out = []
    for i in range(b):
        f_g, h_g, w_g = grid_sizes[i].tolist()
        valid = int(seq_lens[i].item())
        ent = ent_tok[i, :valid][:f_g * h_g * w_g].view(f_g, h_g * w_g).mean(dim=1)
        out.append(ent)
    return torch.stack(out, dim=0)


@torch.no_grad()
def token_entropy_to_2dmap(
    ent_tok: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """
    ent_tok: [B, L]
    return: map2d [B, F, H*W]
    """
    b, _ = ent_tok.shape
    out = []
    for i in range(b):
        f_g, h_g, w_g = grid_sizes[i].tolist()
        valid = int(seq_lens[i].item())
        ent = ent_tok[i, :valid][:f_g * h_g * w_g].view(f_g, h_g * w_g)
        out.append(ent)
    return torch.stack(out, dim=0)
