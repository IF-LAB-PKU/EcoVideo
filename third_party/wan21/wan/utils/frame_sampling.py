import math

import torch


def _compute_target_len(T_src: int, fps_src: float, fps_tgt: float) -> int:
    """Compute target frame length keeping duration unchanged."""
    if fps_tgt <= 0:
        raise ValueError("fps_tgt must be > 0")

    # 保持时长不变：T_tgt ≈ T_src * fps_tgt / fps_src
    T_tgt = int(round(T_src * float(fps_tgt) / float(fps_src)))
    T_tgt = max(1, min(T_src, T_tgt))
    return T_tgt


def sample_indices_uniform(
    T_src: int,
    T_tgt: int,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """均匀取帧，在 [0, T_src-1] 上均匀采样 T_tgt 个索引。"""
    if T_tgt <= 1:
        return torch.zeros((1,), dtype=torch.long, device=device)

    idx = torch.linspace(0, T_src - 1, steps=T_tgt, device=device)
    idx = torch.round(idx).long()

    # 极端情况下 round 可能导致重复，做一次去重并在尾部补齐
    idx = torch.unique_consecutive(idx)
    if idx.numel() < T_tgt:
        need = T_tgt - idx.numel()
        tail = torch.linspace(
            idx[-1].item(), T_src - 1, steps=need + 1, device=device
        )[1:]
        tail = torch.round(tail).long()
        idx = torch.cat([idx, tail], dim=0)

    return idx[:T_tgt]


def sample_indices_random(
    T_src: int,
    T_tgt: int,
    generator: torch.Generator,
) -> torch.LongTensor:
    """全局随机不重复取帧，再按时间排序。"""
    if T_tgt >= T_src:
        return torch.arange(T_src, dtype=torch.long)

    perm = torch.randperm(T_src, generator=generator)
    idx = perm[:T_tgt].sort().values
    return idx


def sample_indices_stratified(
    T_src: int,
    T_tgt: int,
    generator: torch.Generator,
) -> torch.LongTensor:
    """分层随机：把时间轴切成 T_tgt 个 bin，每个 bin 里随机取一帧。"""
    if T_tgt <= 1:
        return torch.zeros((1,), dtype=torch.long)

    edges = torch.linspace(0, T_src, steps=T_tgt + 1)
    idx: list[int] = []

    for i in range(T_tgt):
        lo = int(math.floor(edges[i].item()))
        hi = int(math.floor(edges[i + 1].item())) - 1

        lo = max(0, min(T_src - 1, lo))
        hi = max(0, min(T_src - 1, hi))

        if hi < lo:
            hi = lo

        if hi == lo:
            idx.append(lo)
        else:
            r = torch.randint(lo, hi + 1, (1,), generator=generator).item()
            idx.append(int(r))

    idx_tensor = torch.tensor(sorted(idx), dtype=torch.long)
    return idx_tensor


def resample_video_tensor(
    video: torch.Tensor,
    fps_src: float,
    fps_tgt: float,
    mode: str = "uniform",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """对 Wan 的视频张量做时间维下采样。

    Args:
        video: [C, T, H, W]
        fps_src: 原始 fps（例如 24）
        fps_tgt: 目标 fps
        mode: 取帧模式，"uniform" / "random" / "stratified_random"
        seed: 随机种子（用于 random / stratified_random）

    Returns:
        video_out: [C, T_tgt, H, W]
        idx: 选取的时间索引，形状 [T_tgt]
    """
    if video.dim() != 4:
        raise ValueError(f"Expect [C, T, H, W], got {tuple(video.shape)}")

    C, T_src, H, W = video.shape
    if T_src <= 0:
        raise ValueError("Video has no frames (T_src <= 0).")

    T_tgt = _compute_target_len(T_src, fps_src, fps_tgt)

    # 目标 fps 与原 fps 一样时直接返回，避免数值抖动
    if T_tgt == T_src:
        idx_identity = torch.arange(T_src, dtype=torch.long)
        return video, idx_identity

    device = video.device

    # 生成器放在 CPU，索引最后再搬到目标设备
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if mode == "uniform":
        idx = sample_indices_uniform(T_src, T_tgt, device=torch.device("cpu"))
    elif mode == "random":
        idx = sample_indices_random(T_src, T_tgt, generator=g)
    elif mode in ("stratified", "stratified_random"):
        idx = sample_indices_stratified(T_src, T_tgt, generator=g)
    else:
        raise ValueError(f"Unknown frame sampling mode: {mode}")

    # 索引到时间维（dim=1）
    video_out = video.index_select(dim=1, index=idx.to(device))
    return video_out, idx


