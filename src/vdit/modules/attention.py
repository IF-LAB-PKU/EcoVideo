from torch import nn
import torch
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention  # type: ignore
except Exception:
    memory_efficient_attention = None

_XFORMERS_FALLBACK_WARNED = False


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0., attn_type="self_attn",
                 use_xformers=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_xformers = use_xformers
        self.num_heads = num_heads
        self.scale = (dim // self.num_heads) ** -0.5
        self.attn_drop_rate = attn_drop_rate
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.qkv = None
        self.to_query = None
        self.to_key = None
        self.to_value = None
        self.attn_type = attn_type

    def attention(self, q, k, v):
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1)
        attn_map = self.attn_drop(attn_map)
        output = (attn_map @ v).transpose(1, 2)
        return output

    def attention_with_xformers(self, q, k, v):
        global _XFORMERS_FALLBACK_WARNED
        if memory_efficient_attention is None:
            if not _XFORMERS_FALLBACK_WARNED:
                print("[Attention] xformers not available -> fallback to torch SDPA.")
                _XFORMERS_FALLBACK_WARNED = True
            return self._attention_with_sdpa(q, k, v)
        try:
            return memory_efficient_attention(q, k, v, p=self.attn_drop_rate, scale=self.scale)
        except Exception as e:
            if not _XFORMERS_FALLBACK_WARNED:
                print(
                    f"[Attention] xformers attention failed ({type(e).__name__}: {e}). "
                    "Fallback to torch SDPA."
                )
                _XFORMERS_FALLBACK_WARNED = True
            return self._attention_with_sdpa(q, k, v)

    def _attention_with_sdpa(self, q, k, v):
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()
        dropout_p = float(self.attn_drop_rate) if self.training else 0.0
        try:
            out = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=self.scale
            )
        except TypeError:
            out = F.scaled_dot_product_attention(
                q_ * self.scale, k_, v_, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        return out.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def reshape_cond(x, ph, pw):
        b, n, d = x.shape
        x = x.reshape(b, ph // 2, 2, pw // 2, 2, d)
        x = torch.einsum("bhpwqc->bhwpqc", x)
        x = x.reshape(b, n // 4, 4, d)
        return x

    def forward(self, x, y=None, x0=None, x1=None, ph=None, pw=None):
        b, n, d = x.shape
        if self.attn_type == "temporal_attn":
            if x0.shape[1] != n:
                x0 = self.reshape_cond(x0, ph, pw)
                x1 = self.reshape_cond(x1, ph, pw)
                x = x.unsqueeze(2)
                x = torch.cat((x0, x, x1), dim=2).reshape(b * n, 9, d)
            else:
                x = torch.stack((x0, x, x1), dim=2).reshape(b * n, 3, d)
        # update b, n
        b1, n1 = x.shape[0:2]
        if self.attn_type == "cross_attn":
            n2 = y.shape[1]
            q = self.to_query(x).reshape(b, n, self.num_heads, d // self.num_heads)
            k = self.to_key(y).reshape(b, n2, self.num_heads, d // self.num_heads)
            v = self.to_value(y).reshape(b, n2, self.num_heads, d // self.num_heads)
        else:
            qkv = self.qkv(x).reshape(b1, n1, 3, self.num_heads, d // self.num_heads).permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0, ...], qkv[1, ...], qkv[2, ...]
        if self.use_xformers:
            x = self.attention_with_xformers(q, k, v).reshape(b1, n1, d)
        else:
            x = self.attention(q, k, v).reshape(b1, n1, d)
        if self.attn_type == "temporal_attn":
            x = x[:, n1 // 2, :]
            x = x.reshape(b, n, d)
        x = self.proj_drop(self.proj(x))
        return x


class SelfAttention(Attention):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 use_xformers=True):
        super().__init__(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, "self_attn",
                         use_xformers)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)


class CrossAttention(Attention):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 use_xformers=True):
        super().__init__(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, "cross_attn",
                         use_xformers)
        self.to_query = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_key = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_value = nn.Linear(dim, dim, bias=qkv_bias)


class TemporalAttention(Attention):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 use_xformers=True):
        super().__init__(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, "temporal_attn",
                         use_xformers)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
