from vdit.utils.embedding import TimestepEmbedder, get_pos_embedding
from vdit.utils.distributions import DiagonalGaussianDistribution
import torch

class InputPadder:
    def __init__(self, img_size, divisor=32):
        self.ht, self.wd = img_size
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x):
        return torch.nn.functional.pad(x, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def preprocess_cond(x, eps=1e-8):
    x_flat = x.flatten(1)
    x_mean, x_std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
    while len(x_mean.shape) < len(x.shape):
        x_mean, x_std = x_mean.unsqueeze(-1), x_std.unsqueeze(-1)
    x_norm = (x - x_mean) / x_std
    x_mean_0, x_mean_1 = x_mean.chunk(2, dim=0)
    x_std_0, x_std_1 = x_std.chunk(2, dim=0)
    stats = ((x_mean_0 + x_mean_1) / 2, (x_std_0 + x_std_1) / 2)
    return x_norm, stats
