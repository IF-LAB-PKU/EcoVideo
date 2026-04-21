import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _blue_heavy_cmap():
    colors_and_positions = [
        (0.0, "#FF0000"),
        (0.32, "#FFC400"),
        (0.6, "#F6F2F2"),
        (1.0, "#FFFFFF"),
    ]
    return LinearSegmentedColormap.from_list("blue_heavy_custom_cmap",
                                             colors_and_positions)


def save_block_heatmap_png(data2d, out_path, dpi=300):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if hasattr(data2d, "detach"):
        data2d = data2d.detach().float().cpu().numpy()
    elif not isinstance(data2d, np.ndarray):
        data2d = np.array(data2d, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)
    ax.imshow(data2d,
              cmap=_blue_heavy_cmap(),
              interpolation="nearest",
              aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
