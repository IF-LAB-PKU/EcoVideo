# src/vdit/generators/cogvideo_t2v.py
"""
CogVideo T2V generator with entropy-based keyframe selection (true latent-time pruning).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import os
import json
import math
import time
import inspect

import torch
import torch.nn.functional as F

from vdit.generators.base import register_generator

# ---- Diffusers / CogVideo imports (runtime dependency) ----
try:
    from diffusers import CogVideoXPipeline
    from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
    from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.cogvideox_transformer_3d import (
        CogVideoXBlock,
        CogVideoXTransformer3DModel,
    )
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
except Exception as e:  # pragma: no cover
    CogVideoXPipeline = None  # type: ignore
    CogVideoXDDIMScheduler = None  # type: ignore
    CogVideoXDPMScheduler = None  # type: ignore
    Attention = None  # type: ignore
    CogVideoXAttnProcessor2_0 = object  # type: ignore
    apply_rotary_emb = None  # type: ignore
    CogVideoXBlock = object  # type: ignore
    CogVideoXTransformer3DModel = object  # type: ignore
    retrieve_timesteps = None  # type: ignore
    _DIFFUSERS_IMPORT_ERROR = e
else:
    _DIFFUSERS_IMPORT_ERROR = None


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class CogVideoGenerateConfig:
    # ---- runtime / model ----
    precision: str = "bfloat16"  # "bfloat16" | "float16"
    device: str = "cuda"
    seed: int = 0

    # ---- generation ----
    height: int = 480
    width: int = 720
    num_frames: int = 49               # recommend 4n+1
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    use_dynamic_cfg: bool = True
    scheduler: str = "ddim"            # "ddim" | "dpm"

    # ---- keyframe by attention entropy (true latent-time prune) ----
    keyframe_by_entropy: bool = False
    entropy_steps: int = 5
    entropy_mode: str = "mean"         # "last" | "mean" | "ema"
    entropy_ema_alpha: float = 0.6
    entropy_block_idx: int = -1        # -1 = last block.attn1
    keyframe_topk: int = 8
    keyframe_cover: bool = True
    use_nonkey_context: bool = True    # only affects entropy statistic context (not denoise path)

    # fps after prune (optional)
    keyframe_out_fps: Optional[float] = None
    keyframe_target_fps: Optional[float] = 8.0

    # debug / profiling
    debug_dir: Optional[str] = None
    save_debug_pt: bool = True
    profile_timing: bool = True

    # memory options
    enable_sequential_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    vae_tiling: bool = True
    vae_slicing: bool = True


# -------------------------
# Utils
# -------------------------
def _ensure_dir(path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported precision: {s}")


def _check_num_frames_4n1(num_frames: int) -> None:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >=1, got {num_frames}")
    if num_frames % 4 != 1:
        # CogVideo often expects 4n+1 for temporal VAE alignment
        raise ValueError(
            f"CogVideo num_frames is recommended/required to be 4n+1, got {num_frames}. "
            f"Examples: 49, 81, 161"
        )


def auto_keyframe_topk_cogvideo(
    frame_num_full: int,
    fps_full: float,
    fps_key: float,
    stride_t: int = 4,
    min_k: int = 2,
    max_k: Optional[int] = None,
) -> int:
    """
    近似从目标关键帧视频 fps 估计 latent 关键帧 K。
    假设 CogVideo temporal VAE stride_t ~ 4，并近似满足:
      pixel_frames ≈ 4 * (latent_frames - 1) + 1
    """
    if fps_key <= 0:
        raise ValueError(f"fps_key must be > 0, got {fps_key}")
    if fps_full <= 0:
        raise ValueError(f"fps_full must be > 0, got {fps_full}")
    if frame_num_full <= 1:
        return max(min_k, 1)

    t_key = int(round(frame_num_full * (fps_key / float(fps_full))))
    t_key = max(2, t_key)

    k = int(round((t_key - 1) / float(stride_t))) + 1
    k = max(min_k, k)
    if max_k is not None:
        k = min(max_k, k)
    return k


def select_key_latent_frames(scores: torch.Tensor, topk: int, cover: bool = True) -> List[int]:
    """
    scores: [T_latent]
    topk:   TOTAL budget of selected latent frames (including cover frames if enabled)
    cover:  force include first and last latent frame within the budget
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be [T], got {tuple(scores.shape)}")
    T = int(scores.numel())
    if T <= 0:
        raise ValueError("Empty scores")

    k_total = max(1, min(int(topk), T))

    forced: List[int] = []
    if cover and T >= 2:
        forced = [0, T - 1]
    forced = sorted(set(forced))

    # If budget is too small, just return forced (trim if needed)
    if len(forced) >= k_total:
        return forced[:k_total]

    # Remaining slots after reserving forced frames
    k_remain = k_total - len(forced)

    # Mask forced indices so they won't be picked again
    masked = scores.detach().clone()
    for i in forced:
        masked[i] = -float("inf")

    picked = torch.topk(masked, k=k_remain, largest=True).indices.tolist()
    idx = sorted(set(forced + [int(i) for i in picked]))
    return idx


class EntropyCollector:
    """
    Collect entropy_per_frame [T_latent] for the first `entropy_steps` denoising iterations.
    """
    def __init__(self, mode: str = "mean", ema_alpha: float = 0.6):
        self.mode = str(mode)
        self.ema_alpha = float(ema_alpha)
        self.step_scores: List[torch.Tensor] = []
        self.ema: Optional[torch.Tensor] = None
        self.active: bool = True

    def add_step_frame_entropy(self, frame_entropy_t: torch.Tensor) -> None:
        x = frame_entropy_t.detach().float().cpu()
        self.step_scores.append(x)
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.ema_alpha * self.ema + (1.0 - self.ema_alpha) * x

    def get_final_scores(self) -> torch.Tensor:
        if not self.step_scores:
            raise RuntimeError("No entropy collected")
        if self.mode == "last":
            return self.step_scores[-1]
        if self.mode == "mean":
            return torch.stack(self.step_scores, dim=0).mean(dim=0)
        if self.mode == "ema":
            if self.ema is None:
                raise RuntimeError("EMA is None")
            return self.ema
        raise ValueError(f"Unknown entropy_mode: {self.mode}")


# -------------------------
# Attention processor (entropy statistic only; output path remains normal)
# -------------------------
class CogVideoXAttnProcessor2_0ForEntropy(CogVideoXAttnProcessor2_0):
    """
    A drop-in CogVideo attention processor that:
      - keeps the normal attention output path
      - additionally computes attention entropy per latent-frame (image query rows)
    Notes:
      - current_num_latent_frames MUST be set each denoise step by the caller.
      - use_nonkey_context controls entropy statistic scope only:
          True  -> image-query to all keys (text + image)
          False -> image-query to image keys only (renormalized)
    """

    def __init__(
        self,
        collector: EntropyCollector,
        use_nonkey_context: bool = True,
        cond_only: bool = False,
        # ---- memory-safe entropy statistic knobs ----
        entropy_num_heads_sample: int = 4,
        entropy_q_chunk_size: int = 8,
        entropy_tokens_per_frame_sample: int = 16,
    ):
        super().__init__()
        self.collector = collector
        self.use_nonkey_context = bool(use_nonkey_context)
        self.cond_only = bool(cond_only)
        self.enabled = True
        self.current_num_latent_frames: Optional[int] = None

        # 控制显存和速度的节流参数
        self.entropy_num_heads_sample = int(entropy_num_heads_sample)
        self.entropy_q_chunk_size = int(entropy_q_chunk_size)
        self.entropy_tokens_per_frame_sample = int(entropy_tokens_per_frame_sample)

    def _compute_frame_entropy_chunked(
        self,
        q: torch.Tensor,              # [B,H,S,D]
        k: torch.Tensor,              # [B,H,S,D]
        attention_mask: Optional[torch.Tensor],  # [B,H,Q,K] or None
        text_seq_length: int,
        image_seq_length: int,
        head_dim: int,
    ) -> Optional[torch.Tensor]:
        """
        Memory-safe entropy computation on image queries only.
        Returns frame entropy [T_latent] (cpu tensor) or None if cannot infer layout.
        """
        T_latent = self.current_num_latent_frames
        if T_latent is None or T_latent <= 0:
            return None
        if image_seq_length % T_latent != 0:
            return None

        tokens_per_frame = image_seq_length // T_latent
        if tokens_per_frame <= 0:
            return None

        # ---- optional CFG cond-only ----
        q_stat = q
        k_stat = k
        am_stat = attention_mask
        if self.cond_only and q_stat.shape[0] % 2 == 0:
            half = q_stat.shape[0] // 2
            q_stat = q_stat[half:]
            k_stat = k_stat[half:]
            if am_stat is not None and am_stat.shape[0] == q.shape[0]:
                am_stat = am_stat[half:]

        B, H, S, D = q_stat.shape
        q0 = text_seq_length
        q1 = text_seq_length + image_seq_length

        # ---- sample heads (important for memory + speed) ----
        if self.entropy_num_heads_sample > 0 and self.entropy_num_heads_sample < H:
            head_idx = torch.linspace(
                0, H - 1, steps=self.entropy_num_heads_sample, device=q_stat.device
            ).round().long().unique()
            q_stat = q_stat[:, head_idx, :, :]
            k_stat = k_stat[:, head_idx, :, :]
            if am_stat is not None and am_stat.shape[1] == H:
                am_stat = am_stat[:, head_idx, :, :]
            H_eff = q_stat.shape[1]
        else:
            H_eff = H

        # image query slice only
        q_img = q_stat[:, :, q0:q1, :]   # [B,H,Qimg,D]
        K_all = k_stat.shape[2]

        # ---- sample query tokens per frame (heuristic but effective) ----
        q_img_reshaped = q_img.reshape(B, H_eff, T_latent, tokens_per_frame, D)  # [B,H,T,Sf,D]
        sample_n = self.entropy_tokens_per_frame_sample
        if sample_n > 0 and sample_n < tokens_per_frame:
            # evenly spaced sample indices in each frame
            q_token_idx = torch.linspace(
                0, tokens_per_frame - 1,
                steps=sample_n,
                device=q_img.device
            ).round().long().unique()
            q_img_reshaped = q_img_reshaped[:, :, :, q_token_idx, :]
            tokens_used_per_frame = q_img_reshaped.shape[3]
        else:
            tokens_used_per_frame = tokens_per_frame

        # flatten sampled image queries back
        q_img_flat = q_img_reshaped.reshape(B, H_eff, T_latent * tokens_used_per_frame, D)  # [B,H,Qs,D]

        # key range for entropy context
        if self.use_nonkey_context:
            k_used = k_stat  # [B,H,K,D], all keys (text + image)
            am_used = am_stat
            # key_slice_mode = "all"
        else:
            k0 = text_seq_length
            k1 = text_seq_length + image_seq_length
            k_used = k_stat[:, :, k0:k1, :]  # image-only keys
            if am_stat is not None:
                # original attention mask is [B,H,Q,K]; for q_img queries, take q rows then key cols
                am_used = am_stat[:, :, q0:q1, k0:k1]
            else:
                am_used = None

        # Accumulate entropy per sampled query token, then map to frames
        Qs = q_img_flat.shape[2]
        ent_per_q = torch.empty((B, H_eff, Qs), device=q_img_flat.device, dtype=torch.float32)

        q_chunk = max(1, self.entropy_q_chunk_size)
        scale = (head_dim ** -0.5)

        # For sampled-row mask mapping if mask exists
        sampled_global_idx = None
        if am_used is not None:
            if tokens_used_per_frame == tokens_per_frame:
                sampled_local = torch.arange(image_seq_length, device=q_img_flat.device, dtype=torch.long)
            else:
                per_frame = torch.arange(T_latent, device=q_img_flat.device, dtype=torch.long)[:, None] * tokens_per_frame
                q_token_idx = torch.linspace(
                    0, tokens_per_frame - 1,
                    steps=tokens_used_per_frame,
                    device=q_img_flat.device
                ).round().long().unique()
                sampled_local = (per_frame + q_token_idx[None, :]).reshape(-1)
            sampled_global_idx = sampled_local

        for start in range(0, Qs, q_chunk):
            end = min(start + q_chunk, Qs)
            q_chunk_t = q_img_flat[:, :, start:end, :]  # [B,H,qc,D]

            # logits chunk: [B,H,qc,K]
            logits = torch.matmul(q_chunk_t.float(), k_used.float().transpose(-1, -2)) * scale

            if am_used is not None:
                if sampled_global_idx is None:
                    am_chunk = am_used[:, :, start:end, :]
                else:
                    rows = sampled_global_idx[start:end]
                    am_chunk = am_used[:, :, rows, :]
                logits = logits + am_chunk

            probs = torch.softmax(logits, dim=-1)
            ent_chunk = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)  # [B,H,qc]
            ent_per_q[:, :, start:end] = ent_chunk

            del logits, probs, ent_chunk, q_chunk_t

        # [B,H,Qs] -> [B,H,T,tokens_used] -> average => [T]
        ent_per_q = ent_per_q.reshape(B, H_eff, T_latent, tokens_used_per_frame)
        ent_frame = ent_per_q.mean(dim=(0, 1, 3))  # [T_latent]
        return ent_frame.detach().cpu()

    def _calculate_attention_and_entropy(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn: "Attention",
        batch_size: int,
        image_seq_length: int,
        text_seq_length: int,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query/key/value: [B, S, H*D]
        returns (hidden_states_image, hidden_states_text) after normal attention
        """
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)   # [B,H,S,D]
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        # RoPE on image token positions (standard t2v path, no reference latents)
        if image_rotary_emb is not None and apply_rotary_emb is not None:
            # apply on image token slice only
            q[:, :, text_seq_length:] = apply_rotary_emb(q[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                # in t2v self-attn without reference concatenation, key len == query len
                if k.size(2) == q.size(2):
                    k[:, :, text_seq_length:] = apply_rotary_emb(k[:, :, text_seq_length:], image_rotary_emb)
                else:
                    # fallback for unexpected variants
                    k[:, :, text_seq_length:text_seq_length + image_seq_length] = apply_rotary_emb(
                        k[:, :, text_seq_length:text_seq_length + image_seq_length], image_rotary_emb
                    )

        # ---- Entropy statistic (no-grad), only when enabled ----
        if self.enabled and self.collector is not None and self.collector.active:
            with torch.no_grad():
                ent_frame = self._compute_frame_entropy_chunked(
                    q=q,
                    k=k,
                    attention_mask=attention_mask,
                    text_seq_length=text_seq_length,
                    image_seq_length=image_seq_length,
                    head_dim=head_dim,
                )
                if ent_frame is not None:
                    self.collector.add_step_frame_entropy(ent_frame)

        # ---- Normal attention output (kept compatible) ----
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        encoder_out, hidden_out = out.split([text_seq_length, out.size(1) - text_seq_length], dim=1)
        return hidden_out, encoder_out

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_seq_length = hidden_states.size(1)
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Official CogVideo processor pattern
        batch_size, sequence_length, _ = hidden_states_cat.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states_cat)
        key = attn.to_k(hidden_states_cat)
        value = attn.to_v(hidden_states_cat)

        hidden_out, encoder_out = self._calculate_attention_and_entropy(
            query=query,
            key=key,
            value=value,
            attn=attn,
            batch_size=batch_size,
            image_seq_length=image_seq_length,
            text_seq_length=text_seq_length,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        return hidden_out, encoder_out


class OverrideCogVideoAttnProcessors:
    """
    Replace a single transformer block's attn1 processor (selected by block_idx)
    so we only pay entropy-stat cost on one block (like WAN entropy_block_idx).
    """
    def __init__(
        self,
        transformer: "CogVideoXTransformer3DModel",
        processor_factory,
        block_idx: int = -1,
    ):
        self.transformer = transformer
        self.processor_factory = processor_factory
        self.block_idx = int(block_idx)
        self.original_processors: Dict[int, Any] = {}
        self.target_block = None
        self.target_block_index = None

    def __enter__(self):
        blocks = list(self.transformer.transformer_blocks)
        if len(blocks) == 0:
            raise RuntimeError("transformer.transformer_blocks is empty")

        idx = self.block_idx if self.block_idx >= 0 else (len(blocks) + self.block_idx)
        idx = max(0, min(idx, len(blocks) - 1))

        block = cast("CogVideoXBlock", blocks[idx])
        self.target_block = block
        self.target_block_index = idx
        self.original_processors[id(block)] = block.attn1.get_processor()
        block.attn1.set_processor(self.processor_factory())
        return self

    def get_processor(self):
        if self.target_block is None:
            raise RuntimeError("Context not entered")
        return self.target_block.attn1.get_processor()

    def __exit__(self, exc_type, exc, tb):
        if self.target_block is not None:
            block = cast("CogVideoXBlock", self.target_block)
            block.attn1.set_processor(self.original_processors[id(block)])


# -------------------------
# Diffusers compatibility helpers
# -------------------------
def _pipe_prepare_latents_t2v(
    pipe: "CogVideoXPipeline",
    cfg: CogVideoGenerateConfig,
    generator: torch.Generator,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Try multiple signatures because diffusers versions differ.
    Returns latents in CogVideo latent layout [B,T,C,H,W].
    """
    if not hasattr(pipe, "prepare_latents"):
        raise RuntimeError("CogVideoXPipeline.prepare_latents not found in current diffusers version")

    fn = pipe.prepare_latents
    sig = inspect.signature(fn)
    params = sig.parameters

    # Common args candidates
    kwargs_base = dict(
        batch_size=1,
        num_frames=int(cfg.num_frames),
        height=int(cfg.height),
        width=int(cfg.width),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    # Try a few known signatures
    tries = []

    if "num_channels_latents" in params:
        num_channels_latents = getattr(pipe.transformer.config, "in_channels", None)
        if num_channels_latents is None:
            num_channels_latents = getattr(pipe.vae.config, "latent_channels", None)
        tries.append({**kwargs_base, "num_channels_latents": num_channels_latents})

    tries.append(dict(kwargs_base))

    last_err = None
    for kw in tries:
        try:
            latents = fn(**kw)  # may return tuple in some versions
            if isinstance(latents, tuple):
                latents = latents[0]
            if not torch.is_tensor(latents):
                raise TypeError(f"prepare_latents returned non-tensor: {type(latents)}")
            return latents
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to call CogVideoXPipeline.prepare_latents with known signatures. "
        f"Signature={sig}"
    ) from last_err


def _pipe_encode_prompt_t2v(
    pipe: "CogVideoXPipeline",
    prompt: str,
    do_cfg: bool,
    device: Union[str, torch.device],
):
    """
    Wrap encode_prompt because some diffusers versions change arg names/order.
    Returns (prompt_embeds, negative_prompt_embeds)
    """
    fn = pipe.encode_prompt
    sig = inspect.signature(fn)
    params = sig.parameters

    # Try kwargs path (preferred)
    tries = []

    kw = {"prompt": prompt}
    if "negative_prompt" in params:
        kw["negative_prompt"] = None
    if "do_classifier_free_guidance" in params:
        kw["do_classifier_free_guidance"] = do_cfg
    if "device" in params:
        kw["device"] = device
    tries.append(kw)

    last_err = None
    for x in tries:
        try:
            out = fn(**x)
            if isinstance(out, tuple):
                if len(out) >= 2:
                    return out[0], out[1]
                if len(out) == 1:
                    return out[0], out[0]
            if torch.is_tensor(out):
                return out, out
            raise RuntimeError(f"Unexpected encode_prompt return type: {type(out)}")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to call encode_prompt; signature={sig}") from last_err


def _pipe_postprocess_video_to_vdit_frames(pipe: "CogVideoXPipeline", latents: torch.Tensor) -> torch.Tensor:
    """
    Decode CogVideo latents and convert to VDiT frames [T,3,H,W] float [0,1] on CPU.
    """
    video = pipe.decode_latents(latents)

    # Prefer tensor output if available
    frames = None
    try:
        out = pipe.video_processor.postprocess_video(video=video, output_type="pt")
        # usually list with batch dimension
        if isinstance(out, (list, tuple)):
            v = out[0]
        else:
            v = out
        if torch.is_tensor(v):
            # common layouts: [F,C,H,W] or [B,F,C,H,W] or [C,F,H,W]
            if v.ndim == 5:
                v = v[0]
            if v.ndim == 4 and v.shape[0] == 3:
                v = v.permute(1, 0, 2, 3).contiguous()
            elif v.ndim == 4 and v.shape[1] == 3:
                pass  # [F,3,H,W]
            else:
                raise ValueError(f"Unexpected tensor video shape from postprocess_video(pt): {tuple(v.shape)}")
            frames = v.float().clamp(0.0, 1.0).cpu()
    except Exception:
        frames = None

    if frames is not None:
        return frames

    # Fallback to PIL list -> tensor
    out_pil = pipe.video_processor.postprocess_video(video=video, output_type="pil")
    pil_frames = out_pil[0] if isinstance(out_pil, (list, tuple)) else out_pil
    import numpy as np
    arrs = []
    for im in pil_frames:
        a = np.array(im, copy=False)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"Unexpected PIL frame array shape: {a.shape}")
        t = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
        arrs.append(t)
    return torch.stack(arrs, dim=0).contiguous().cpu()


# -------------------------
# Core denoise loop with entropy-based latent-time pruning
# -------------------------
@torch.no_grad()
def sample_cogvideo_t2v_with_entropy_keyframes(
    pipe: "CogVideoXPipeline",
    *,
    prompt: str,
    latents: torch.Tensor,  # [B,T,C,H,W]
    cfg: CogVideoGenerateConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Modified from official CogVideo ddim_inversion.py::sample()
    but simplified for pure T2V (no reference_latents), with true latent-time pruning.
    """
    if retrieve_timesteps is None:
        raise RuntimeError("diffusers CogVideo retrieve_timesteps is unavailable")

    t0_all = time.perf_counter()

    pipe._guidance_scale = float(cfg.guidance_scale)
    pipe._attention_kwargs = None
    pipe._interrupt = False

    device = pipe._execution_device
    do_cfg = float(cfg.guidance_scale) > 1.0

    # 1) prompt embeds
    t0 = time.perf_counter()
    prompt_embeds, negative_prompt_embeds = _pipe_encode_prompt_t2v(
        pipe=pipe,
        prompt=prompt,
        do_cfg=do_cfg,
        device=device,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    t_prompt = time.perf_counter() - t0

    # 2) timesteps
    t0 = time.perf_counter()
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, int(cfg.num_inference_steps), device)
    pipe._num_timesteps = len(timesteps)
    t_timesteps = time.perf_counter() - t0

    # 3) init latents
    t0 = time.perf_counter()
    latents = latents.to(device=device) * pipe.scheduler.init_noise_sigma
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    t_prepare_latents = time.perf_counter() - t0

    # 4) initial rotary embeds
    def _build_rotary_for_latents(_latents: torch.Tensor):
        if not pipe.transformer.config.use_rotary_positional_embeddings:
            return None
        return pipe._prepare_rotary_positional_embeddings(
            height=_latents.size(3) * pipe.vae_scale_factor_spatial,
            width=_latents.size(4) * pipe.vae_scale_factor_spatial,
            num_frames=_latents.size(1),
            device=device,
        )

    t0 = time.perf_counter()
    image_rotary_emb = _build_rotary_for_latents(latents)
    t_rotary_init = time.perf_counter() - t0

    # 5) entropy collector + processor injection
    collector = EntropyCollector(mode=cfg.entropy_mode, ema_alpha=cfg.entropy_ema_alpha)
    selected_key_idx: Optional[List[int]] = None
    selected_entropy_scores: Optional[torch.Tensor] = None
    latent_frames_before_prune = int(latents.size(1))
    latent_frames_after_prune = latent_frames_before_prune
    prune_trigger_step = None
    t_entropy_stat = 0.0
    t_transformer = 0.0
    t_scheduler = 0.0
    t_rotary_rebuild = 0.0

    if not bool(cfg.keyframe_by_entropy):
        collector.active = False

    def _proc_factory():
        return CogVideoXAttnProcessor2_0ForEntropy(
            collector=collector,
            use_nonkey_context=bool(cfg.use_nonkey_context),
            # 先只统计 cond 分支 + 采样少量 heads / tokens，避免 OOM
            cond_only=True,
            entropy_num_heads_sample=4,
            entropy_q_chunk_size=8,
            entropy_tokens_per_frame_sample=16,
        )

    with OverrideCogVideoAttnProcessors(
        transformer=cast("CogVideoXTransformer3DModel", pipe.transformer),
        processor_factory=_proc_factory,
        block_idx=int(cfg.entropy_block_idx),
    ) as attn_ctx:
        proc = attn_ctx.get_processor()

        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if getattr(pipe, "_interrupt", False):
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                timestep = t.expand(latent_model_input.shape[0])

                # Tell processor current latent-frame count (for Qimg->[T,S] reshape)
                proc.current_num_latent_frames = int(latents.size(1))
                proc.enabled = bool(cfg.keyframe_by_entropy and collector.active and (i < int(cfg.entropy_steps)))

                # transformer forward
                t_tf0 = time.perf_counter()
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()
                t_transformer += (time.perf_counter() - t_tf0)

                # dynamic cfg (same style as official sample)
                if bool(cfg.use_dynamic_cfg):
                    pipe._guidance_scale = 1 + float(cfg.guidance_scale) * (
                        (
                            1
                            - math.cos(
                                math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0
                            )
                        )
                        / 2
                    )
                else:
                    pipe._guidance_scale = float(cfg.guidance_scale)

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    gs = getattr(pipe, "guidance_scale", pipe._guidance_scale)
                    noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond)

                # scheduler step
                t_sc0 = time.perf_counter()
                latents = pipe.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                latents = latents.to(prompt_embeds.dtype)
                t_scheduler += (time.perf_counter() - t_sc0)

                # Trigger prune after entropy_steps (1-based)
                if (
                    bool(cfg.keyframe_by_entropy)
                    and collector.active
                    and (i + 1 >= int(cfg.entropy_steps))
                ):
                    t_es0 = time.perf_counter()
                    selected_entropy_scores = collector.get_final_scores()  # [T_latent]
                    # clamp topk to current latent frames
                    topk_req = int(cfg.keyframe_topk)
                    topk = min(topk_req, int(latents.size(1)))
                    selected_key_idx = select_key_latent_frames(
                        selected_entropy_scores, topk=topk, cover=bool(cfg.keyframe_cover)
                    )

                    latents = latents[:, selected_key_idx, ...].contiguous()
                    latent_frames_after_prune = int(latents.size(1))
                    prune_trigger_step = int(i + 1)

                    collector.active = False
                    proc.enabled = False
                    t_entropy_stat += (time.perf_counter() - t_es0)

                    # Rebuild rotary embeddings because num_frames changed
                    t_rb0 = time.perf_counter()
                    image_rotary_emb = _build_rotary_for_latents(latents)
                    t_rotary_rebuild += (time.perf_counter() - t_rb0)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

    # release hooks (diffusers)
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass

    t_total = time.perf_counter() - t0_all

    debug = {
        "latent_frames_before_prune": latent_frames_before_prune,
        "latent_frames_after_prune": latent_frames_after_prune,
        "selected_key_idx": selected_key_idx if selected_key_idx is not None else [],
        "entropy_scores": selected_entropy_scores,  # tensor or None (caller may save)
        "prune_trigger_step": prune_trigger_step,
        "timing": {
            "prompt_encode_sec": float(t_prompt),
            "prepare_timesteps_sec": float(t_timesteps),
            "prepare_latents_sec": float(t_prepare_latents),
            "rotary_init_sec": float(t_rotary_init),
            "transformer_total_sec": float(t_transformer),
            "scheduler_total_sec": float(t_scheduler),
            "entropy_select_sec": float(t_entropy_stat),
            "rotary_rebuild_sec": float(t_rotary_rebuild),
            "total_sec": float(t_total),
        },
    }
    return latents, debug


# -------------------------
# Main generation function
# -------------------------
@torch.no_grad()
def generate_cogvideo_frames(
    *,
    prompt: str,
    ckpt_dir: str,
    cfg: CogVideoGenerateConfig,
) -> Tuple[torch.Tensor, float]:
    """
    Returns:
      frames: [T,3,H,W] float in [0,1] (CPU)
      fps:    float (effective fps of the keyframe video)
    """
    if _DIFFUSERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "diffusers/CogVideo imports failed. Please install compatible diffusers + transformers."
        ) from _DIFFUSERS_IMPORT_ERROR

    _check_num_frames_4n1(int(cfg.num_frames))

    device = torch.device(cfg.device)
    dtype = _torch_dtype_from_str(cfg.precision)

    # ---- FPS policy (IMPORTANT) ----
    # CogVideoX: 49 frames at 8 fps ≈ 6s; previously 16 fps gave ~3s.
    fps_src = 8.0
    fps_tgt = fps_src

    _ensure_dir(cfg.debug_dir)

    # ---- init pipeline ----
    t0_init = time.perf_counter()
    pipe: CogVideoXPipeline = CogVideoXPipeline.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
    )
    # Scheduler
    if str(cfg.scheduler).lower() == "ddim":
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif str(cfg.scheduler).lower() == "dpm":
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    # Device / offload
    if bool(cfg.enable_model_cpu_offload):
        pipe.enable_model_cpu_offload()
    elif bool(cfg.enable_sequential_cpu_offload):
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if bool(cfg.vae_slicing) and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if bool(cfg.vae_tiling) and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    t_init = time.perf_counter() - t0_init

    # ---- keyframe_target_fps -> auto keyframe_topk (latent K) ----
    keyframe_topk = int(cfg.keyframe_topk)
    keyframe_out_fps = cfg.keyframe_out_fps
    if bool(cfg.keyframe_by_entropy) and (cfg.keyframe_target_fps is not None):
        # Estimate max latent frames from num_frames (4n+1 -> latent ≈ (F-1)/4 + 1)
        max_k = int((int(cfg.num_frames) - 1) // 4 + 1)
        keyframe_topk = auto_keyframe_topk_cogvideo(
            frame_num_full=int(cfg.num_frames),
            fps_full=float(fps_src),
            fps_key=float(cfg.keyframe_target_fps),
            stride_t=4,
            min_k=2,
            max_k=max_k,
        )
        if keyframe_out_fps is None:
            keyframe_out_fps = float(cfg.keyframe_target_fps)

    # clone cfg with resolved keyframe_topk
    if keyframe_topk != int(cfg.keyframe_topk) or keyframe_out_fps != cfg.keyframe_out_fps:
        cfg_eff = CogVideoGenerateConfig(**{**asdict(cfg), "keyframe_topk": keyframe_topk, "keyframe_out_fps": keyframe_out_fps})
    else:
        cfg_eff = cfg

    # ---- prepare latents ----
    t0_lat = time.perf_counter()
    g = torch.Generator(device=device).manual_seed(int(cfg_eff.seed))
    latents = _pipe_prepare_latents_t2v(
        pipe=pipe,
        cfg=cfg_eff,
        generator=g,
        device=device,
        dtype=pipe.transformer.dtype if hasattr(pipe, "transformer") else dtype,
    )
    t_prepare_latents = time.perf_counter() - t0_lat

    # ---- manual denoise loop with entropy prune ----
    t0_den = time.perf_counter()
    final_latents, debug = sample_cogvideo_t2v_with_entropy_keyframes(
        pipe=pipe,
        prompt=prompt,
        latents=latents,
        cfg=cfg_eff,
        generator=g,
    )
    t_denoise = time.perf_counter() - t0_den

    # ---- decode to frames ----
    t0_dec = time.perf_counter()
    frames = _pipe_postprocess_video_to_vdit_frames(pipe, final_latents)
    t_decode = time.perf_counter() - t0_dec

    # ---- Decide output fps ----
    # If user explicitly sets keyframe_out_fps, ALWAYS honor it (even without entropy).
    if cfg_eff.keyframe_out_fps is not None:
        fps_tgt = float(cfg_eff.keyframe_out_fps)
    else:
        # If using entropy prune and user sets keyframe_target_fps, use it as default fps
        if bool(cfg_eff.keyframe_by_entropy) and (cfg_eff.keyframe_target_fps is not None):
            fps_tgt = float(cfg_eff.keyframe_target_fps)
        else:
            fps_tgt = fps_src  # normal full video

        # If entropy prune happens, keep duration consistent with the original video:
        # duration_full = (F_full - 1)/fps_src, so fps_tgt = (F_out - 1)/duration_full
        if bool(cfg_eff.keyframe_by_entropy):
            F_full = int(cfg_eff.num_frames)
            F_out = int(frames.shape[0])
            if F_full >= 2 and F_out >= 2:
                fps_tgt = (F_out - 1) * fps_src / float(F_full - 1)

    # ---- debug save ----
    if cfg_eff.debug_dir:
        meta = {
            "prompt": prompt,
            "ckpt_dir": ckpt_dir,
            "cfg": asdict(cfg_eff),
            "timing": {
                "pipeline_init_sec": float(t_init),
                "prepare_latents_sec": float(t_prepare_latents),
                "denoise_wall_sec": float(t_denoise),
                "decode_wall_sec": float(t_decode),
                "total_wall_sec": float(t_init + t_prepare_latents + t_denoise + t_decode),
            },
            "io": {
                "fps_src_assumed": float(fps_src),
                "fps_out": float(fps_tgt),
                "frames_out": int(frames.shape[0]),
                "height_out": int(frames.shape[2]),
                "width_out": int(frames.shape[3]),
            },
            "entropy": {
                "latent_frames_before_prune": int(debug.get("latent_frames_before_prune", -1)),
                "latent_frames_after_prune": int(debug.get("latent_frames_after_prune", -1)),
                "selected_key_idx": [int(x) for x in (debug.get("selected_key_idx") or [])],
                "prune_trigger_step": debug.get("prune_trigger_step", None),
                "use_nonkey_context": bool(cfg_eff.use_nonkey_context),
                "entropy_steps": int(cfg_eff.entropy_steps),
                "entropy_mode": str(cfg_eff.entropy_mode),
                "entropy_block_idx": int(cfg_eff.entropy_block_idx),
            },
            "inner_timing": debug.get("timing", {}),
        }
        _write_json(os.path.join(cfg_eff.debug_dir, "meta.json"), meta)

        ent = debug.get("entropy_scores", None)
        if ent is not None and torch.is_tensor(ent):
            if cfg_eff.save_debug_pt:
                torch.save(ent, os.path.join(cfg_eff.debug_dir, "entropy_scores.pt"))
            # JSON too (easy to inspect)
            _write_json(
                os.path.join(cfg_eff.debug_dir, "entropy_scores.json"),
                {"scores": [float(x) for x in ent.detach().cpu().tolist()]},
            )

        # compatibility: full_pipeline.py expects timing.json (like WAN)
        _write_json(
            os.path.join(cfg_eff.debug_dir, "timing.json"),
            {
                "pipeline_init_sec": float(t_init),
                "prepare_latents_sec": float(t_prepare_latents),
                "denoise_sec": float(t_denoise),
                "decode_sec": float(t_decode),
                "total_sec": float(t_init + t_prepare_latents + t_denoise + t_decode),
                **{k: float(v) for k, v in (debug.get("timing", {}) or {}).items()},
            },
        )

    # free memory
    del final_latents, latents
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return frames.float().cpu(), float(fps_tgt)


# -------------------------
# Generator registration
# -------------------------
@register_generator("cogvideo")
class CogVideoGenerator:
    def __init__(self, ckpt_dir: str, cfg: CogVideoGenerateConfig):
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[torch.Tensor, float]:
        return generate_cogvideo_frames(prompt=prompt, ckpt_dir=self.ckpt_dir, cfg=self.cfg)
