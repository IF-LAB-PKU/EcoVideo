from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


def save_heatmap_png(
    array2d,
    out_path: str,
    dpi: int = 300,
):
    """
    输出无装饰热力图：无坐标轴/无标题/无 colorbar，紧凑裁剪。
    array2d: 2D numpy 或 torch tensor
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if hasattr(array2d, "detach"):
        array2d = array2d.detach().float().cpu().numpy()
    elif not isinstance(array2d, np.ndarray):
        array2d = np.array(array2d, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)
    ax.imshow(array2d, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
