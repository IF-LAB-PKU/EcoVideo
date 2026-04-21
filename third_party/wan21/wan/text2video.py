# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler




def _save_trace_frame_metrics_png(step_ids, entropy_vals, cos_vals, out_png, *, frame_idx, block_idx=None, dpi=220):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        logging.warning(f"[entropy-vis] matplotlib not available, skip saving {out_png}: {e}")
        return
    step_ids = np.asarray(step_ids, dtype=np.int32)
    entropy_vals = np.asarray(entropy_vals, dtype=np.float32)
    cos_vals = np.asarray(cos_vals, dtype=np.float32)
    if step_ids.ndim != 1 or step_ids.size == 0:
        return
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    line1 = ax1.plot(step_ids, entropy_vals, marker="o", linewidth=2.0, label="frame entropy")
    ax1.set_xlabel("step")
    ax1.set_ylabel("frame attention entropy")
    ax1.set_xticks(step_ids.tolist())
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    line2 = ax2.plot(step_ids, cos_vals, marker="s", linewidth=2.0, label="cosine to previous step")
    ax2.set_ylabel("cosine similarity to previous step (latent)")
    title = f"Random latent frame trace (frame={frame_idx})"
    if block_idx is not None:
        title += f" @ block {block_idx}"
    ax1.set_title(title)
    handles = line1 + line2
    ax1.legend(handles, [h.get_label() for h in handles], loc="best")
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _save_entropy_curve_png(ent_frame, out_png, *, key_idx=None, title=None, dpi=220):
    """Save a frame-level attention-entropy curve as a PNG.

    Args:
        ent_frame: 1D tensor/array-like with length F_latent.
        out_png: output png path.
        key_idx: optional 1D indices of selected keyframes (will be marked).
        title: optional title string.
        dpi: png dpi.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        logging.warning(f"[entropy-vis] matplotlib not available, skip saving {out_png}: {e}")
        return

    if isinstance(ent_frame, torch.Tensor):
        y = ent_frame.detach().float().cpu().numpy()
    else:
        y = np.asarray(ent_frame, dtype=np.float32)

    if y.ndim != 1 or y.size == 0:
        logging.warning(f"[entropy-vis] invalid ent_frame shape {getattr(y, 'shape', None)}, skip {out_png}")
        return

    x = np.arange(y.size, dtype=np.int32)

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure()
    plt.plot(x, y)

    if key_idx is not None:
        if isinstance(key_idx, torch.Tensor):
            k = key_idx.detach().cpu().numpy().astype(int).tolist()
        else:
            k = [int(v) for v in key_idx]
        k = [vv for vv in k if 0 <= vv < y.size]
        if len(k) > 0:
            plt.scatter(np.asarray(k), y[np.asarray(k)], s=18)
            for vv in k:
                plt.axvline(vv, linewidth=0.6, alpha=0.5)

    plt.xlabel("frame id (latent)")
    plt.ylabel("frame attention entropy")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 keyframe_by_entropy=False,
                 keyframe_select_mode="entropy",  # "entropy" | "uniform" | "random" (ablation)
                 keyframe_sample_seed=0,
                 entropy_steps=5,
                 entropy_mode="mean",
                 entropy_ema_alpha=0.6,
                 entropy_block_idx=-1,
                 keyframe_topk=10,
                 keyframe_cover=True,
                 use_nonkey_context=True,
                 debug_dir=None,
                 save_debug_pt=True,
                 profile_timing=True,
                 # -------- analysis-only step trace --------
                 trace_frame_metrics=False,
                 trace_steps=50,
                 trace_frame_seed=2026,
                 # ===== method-2: non-key low-frequency compute (TeaCache-style) =====
                 nonkey_update_mode="none",
                 nonkey_update_interval=5,
                 teacache_rel_l1_thresh=0.02,
                 teacache_max_skip=10,
                 teacache_warmup=0,
                 save_teacache_trace_png=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            keyframe_by_entropy (`bool`, *optional*, defaults to False):
                If True, compute attention entropy in early steps and select keyframes.
            entropy_steps (`int`, *optional*, defaults to 5):
                Number of early denoising steps used to collect entropy.
            entropy_mode (`str`, *optional*, defaults to "mean"):
                Entropy aggregation mode: "last" | "mean" | "ema".
            entropy_ema_alpha (`float`, *optional*, defaults to 0.6):
                EMA decay for entropy_mode="ema".
            entropy_block_idx (`int`, *optional*, defaults to -1):
                Which transformer block to use for entropy (-1 means last).
            keyframe_topk (`int`, *optional*, defaults to 10):
                Number of keyframes to select in latent space.
            keyframe_cover (`bool`, *optional*, defaults to True):
                Enforce start/end coverage and bucket fill when selecting keyframes.
            use_nonkey_context (`bool`, *optional*, defaults to True):
                If True, add non-keyframe summary tokens as extra cross-attn context.
            debug_dir (`str`, *optional*, defaults to None):
                Directory for debug dumps (entropy, timing, keyframe indices).
            save_debug_pt (`bool`, *optional*, defaults to True):
                Whether to save per-step entropy tensors to debug_dir.
            profile_timing (`bool`, *optional*, defaults to True):
                Whether to record timing info into debug_dir.
            nonkey_update_mode (`str`, *optional*, defaults to "none"):
                Non-key update mode: "none" | "interval" | "teacache".
            nonkey_update_interval (`int`, *optional*, defaults to 5):
                Interval mode: update non-key every N steps.
            teacache_rel_l1_thresh (`float`, *optional*, defaults to 0.02):
                TeaCache indicator threshold.
            teacache_max_skip (`int`, *optional*, defaults to 10):
                TeaCache max consecutive skips before forcing update.
            teacache_warmup (`int`, *optional*, defaults to 0):
                TeaCache warmup steps after crop (force updates).
            save_teacache_trace_png (`bool`, *optional*, defaults to True):
                Save rel_l1 trace PNG into debug_dir.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        seq_len_full = seq_len

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # keyframe entropy setup
        if keyframe_by_entropy and self.sp_size > 1:
            logging.warning(
                "Entropy-based keyframe selection is disabled under USP/context parallel."
            )
            keyframe_by_entropy = False

        entropy_steps = int(entropy_steps)
        if keyframe_by_entropy:
            entropy_steps = max(1, entropy_steps)
        else:
            entropy_steps = 0

        if debug_dir is not None and self.rank == 0:
            os.makedirs(debug_dir, exist_ok=True)

        def _write_json(path, payload):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)

        def _log_debug(msg):
            if debug_dir is None or self.rank != 0:
                return
            log_path = os.path.join(debug_dir, "log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        def calc_seq_len(f_lat):
            target_shape_local = (
                self.vae.model.z_dim,
                int(f_lat),
                size[1] // self.vae_stride[1],
                size[0] // self.vae_stride[2],
            )
            return math.ceil((target_shape_local[2] * target_shape_local[3]) /
                             (self.patch_size[1] * self.patch_size[2]) *
                             target_shape_local[1] /
                             self.sp_size) * self.sp_size

        def select_keyframes(ent_1d, topk, cover=True):
            f_lat = ent_1d.numel()
            k = min(int(topk), f_lat)
            if k <= 0:
                return torch.arange(f_lat, device=ent_1d.device)
            idx = torch.topk(ent_1d, k=k, largest=True).indices

            if cover and f_lat >= 2 and k >= 2:
                must = torch.tensor([0, f_lat - 1], device=ent_1d.device)
                idx = torch.unique(torch.cat([idx, must], dim=0))

                if idx.numel() > k:
                    mask = torch.ones(f_lat, device=ent_1d.device, dtype=torch.bool)
                    mask[must] = False
                    remain_k = max(0, k - must.numel())
                    if remain_k > 0:
                        remain = torch.topk(ent_1d[mask], k=remain_k).indices
                        remain_idx = torch.arange(f_lat, device=ent_1d.device)[mask][remain]
                        idx = torch.unique(torch.cat([must, remain_idx], dim=0))
                    else:
                        idx = must

                if idx.numel() < k:
                    need = k - idx.numel()
                    buckets = torch.linspace(
                        0, f_lat - 1, steps=need + 2,
                        device=ent_1d.device)[1:-1].round().long()
                    idx = torch.unique(torch.cat([idx, buckets], dim=0))

            return idx.sort().values

        def sample_keyframes(f_lat, topk, cover, mode, seed, device):
            """Uniform or random keyframe sampling (ablation, no entropy)."""
            k = min(int(topk), int(f_lat))
            if k <= 0:
                return torch.arange(f_lat, device=device, dtype=torch.long)
            if f_lat <= 1:
                return torch.arange(f_lat, device=device, dtype=torch.long)

            if mode == "uniform":
                idx = torch.linspace(0, f_lat - 1, steps=k, device=device)
                idx = torch.round(idx).long()
                idx = torch.unique(idx)
                if idx.numel() < k:
                    all_idx = torch.arange(f_lat, device=device)
                    mask = torch.ones(f_lat, dtype=torch.bool, device=device)
                    mask[idx] = False
                    extra = all_idx[mask][: (k - idx.numel())]
                    idx = torch.cat([idx, extra], dim=0)
                if cover and f_lat >= 2 and k >= 2:
                    must = torch.tensor([0, f_lat - 1], device=device, dtype=torch.long)
                    idx = torch.unique(torch.cat([idx, must], dim=0))
                    if idx.numel() > k:
                        idx_sorted = torch.sort(idx).values
                        if k == 2:
                            idx = must
                        else:
                            mid = idx_sorted[
                                (idx_sorted != 0) & (idx_sorted != (f_lat - 1))
                            ]
                            need = k - 2
                            idx = torch.cat(
                                [
                                    torch.tensor([0], device=device, dtype=torch.long),
                                    mid[:need],
                                    torch.tensor(
                                        [f_lat - 1], device=device, dtype=torch.long
                                    ),
                                ],
                                dim=0,
                            )
                return torch.sort(idx).values

            if mode == "random":
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed))
                if cover and f_lat >= 2 and k >= 2:
                    if f_lat == 2:
                        return torch.tensor([0, 1], device=device, dtype=torch.long)
                    mid = torch.arange(1, f_lat - 1, device=device)
                    perm = torch.randperm(mid.numel(), generator=g).to(device)
                    take = mid[perm[: (k - 2)]]
                    idx = torch.cat(
                        [
                            torch.tensor([0, f_lat - 1], device=device, dtype=torch.long),
                            take,
                        ],
                        dim=0,
                    )
                else:
                    perm = torch.randperm(f_lat, generator=g).to(device)
                    idx = perm[:k]
                return torch.sort(torch.unique(idx)).values

            raise ValueError(f"Unknown keyframe_select_mode: {mode}")

        @torch.no_grad()
        def build_nonkey_extra_context(nonkey_latent, model):
            emb = model.patch_embedding(nonkey_latent.unsqueeze(0))
            emb = emb.mean(dim=(3, 4))
            return emb.permute(0, 2, 1).contiguous()

        @torch.no_grad()
        def teacache_indicator(nonkey_latent, timestep_1d, model,
                               seq_len_local):
            """
            TeaCache-style lightweight indicator.
            nonkey_latent: [C, F, H, W]
            timestep_1d: Tensor[B], here B=1
            return: Tensor[dim] float32
            """
            x = model.patch_embedding(nonkey_latent.unsqueeze(0))
            x = x.flatten(2).transpose(1, 2)
            if x.size(1) < seq_len_local:
                pad = x.new_zeros(1, seq_len_local - x.size(1), x.size(2))
                x = torch.cat([x, pad], dim=1)

            from .modules.model import sinusoidal_embedding_1d
            with amp.autocast(dtype=torch.float32):
                e = model.time_embedding(
                    sinusoidal_embedding_1d(model.freq_dim,
                                            timestep_1d).float())
            ind = x.mean(dim=1) + e
            return ind.squeeze(0).float()

        def save_rel_l1_curve_png(trace, out_png):
            if len(trace) == 0:
                return
            try:
                import matplotlib.pyplot as plt

                xs = [d["step"] for d in trace]
                ys = [float(d["rel_l1"]) for d in trace]
                ups_x = [d["step"] for d in trace if d["updated"]]
                ups_y = [float(d["rel_l1"]) for d in trace if d["updated"]]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(xs, ys)
                if len(ups_x) > 0:
                    ax.scatter(ups_x, ups_y, marker="o")
                ax.set_xlabel("step")
                ax.set_ylabel("nonkey rel_l1")
                ax.grid(True, alpha=0.3)
                os.makedirs(os.path.dirname(out_png), exist_ok=True)
                plt.savefig(out_png, dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                _log_debug(f"[warn] failed to save rel_l1 png: {e}")

        def reset_multistep_state(scheduler):
            if hasattr(scheduler, "config") and hasattr(scheduler.config, "solver_order"):
                solver_order = int(scheduler.config.solver_order)
            elif hasattr(scheduler, "model_outputs"):
                solver_order = len(scheduler.model_outputs)
            elif hasattr(scheduler, "timestep_list"):
                solver_order = len(scheduler.timestep_list)
            else:
                solver_order = None

            if hasattr(scheduler, "model_outputs") and solver_order is not None:
                scheduler.model_outputs = [None] * solver_order
            if hasattr(scheduler, "timestep_list") and solver_order is not None:
                scheduler.timestep_list = [None] * solver_order
            if hasattr(scheduler, "lower_order_nums"):
                scheduler.lower_order_nums = 0
            if hasattr(scheduler, "last_sample"):
                scheduler.last_sample = None

        from .utils.entropy_collector import EntropyCollector

        collector = EntropyCollector(
            enabled=False,
            mode=entropy_mode,
            ema_alpha=entropy_ema_alpha,
            block_idx=entropy_block_idx)
        collector.reset()

        # analysis-only: trace one random latent frame across steps
        trace_frame_idx = None
        trace_step_ids = []
        trace_frame_entropy = []
        trace_cos_to_prev = []
        trace_prev_latent = None

        key_idx = None
        extra_context = None
        seq_len_curr = seq_len_full

        t0 = time.time()
        t_entropy_end = None

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            if entropy_steps > len(timesteps):
                entropy_steps = len(timesteps)

            # sample videos
            latents = noise

            # ===== method-2 states (non-key low-frequency compute) =====
            nonkey_latents = None
            nonkey_seq_len = None
            nonkey_scheduler = None
            nonkey_noise_cache = None
            nonkey_ind_prev = None
            nonkey_start_step = None
            nonkey_skip_streak = 0
            nonkey_update_count = 0
            nonkey_trace = []

            for step_i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                collect_entropy = (
                    keyframe_by_entropy
                    and (keyframe_select_mode == "entropy")
                    and (step_i < entropy_steps)
                )
                collect_entropy_trace = bool(trace_frame_metrics) and (step_i < trace_steps)
                collector.enabled = collect_entropy or collect_entropy_trace

                self.model.to(self.device)

                # ===== method-2: update non-key latents with low-frequency DiT =====
                if (key_idx is not None) and use_nonkey_context and (
                        nonkey_latents is not None) and (nonkey_scheduler
                                                         is not None):
                    timestep_1d = torch.stack([t])

                    rel_l1 = 0.0
                    updated = False
                    if nonkey_update_mode == "teacache":
                        ind_now = teacache_indicator(nonkey_latents,
                                                     timestep_1d, self.model,
                                                     nonkey_seq_len)
                        if nonkey_ind_prev is None:
                            rel_l1 = 1e9
                        else:
                            denom = nonkey_ind_prev.abs().mean().clamp(min=1e-6)
                            rel_l1 = (ind_now - nonkey_ind_prev).abs().mean() / denom
                            rel_l1 = float(rel_l1.detach().cpu().item())
                    else:
                        rel_l1 = 0.0

                    need_update = False
                    if nonkey_noise_cache is None:
                        need_update = True
                    elif nonkey_update_mode == "interval":
                        if nonkey_update_interval <= 0:
                            need_update = False
                        else:
                            last_up = nonkey_trace[-1]["step"] if len(
                                nonkey_trace) > 0 else (
                                    nonkey_start_step
                                    if nonkey_start_step is not None else step_i)
                            need_update = (step_i - int(last_up)
                                           ) >= int(nonkey_update_interval)
                    elif nonkey_update_mode == "teacache":
                        if nonkey_start_step is None:
                            nonkey_start_step = step_i
                        if (step_i -
                                nonkey_start_step) < int(teacache_warmup):
                            need_update = True
                        else:
                            if rel_l1 > float(teacache_rel_l1_thresh):
                                need_update = True
                            if nonkey_skip_streak >= int(teacache_max_skip):
                                need_update = True
                    else:
                        need_update = False

                    if need_update:
                        latent_model_input_nk = [nonkey_latents]
                        noise_pred_cond_nk = self.model(
                            latent_model_input_nk,
                            t=timestep_1d,
                            context=context,
                            seq_len=nonkey_seq_len,
                            entropy_collector=None,
                            extra_context=None,
                        )[0]
                        noise_pred_uncond_nk = self.model(
                            latent_model_input_nk,
                            t=timestep_1d,
                            context=context_null,
                            seq_len=nonkey_seq_len,
                            entropy_collector=None,
                            extra_context=None,
                        )[0]
                        nonkey_noise_cache = noise_pred_uncond_nk + guide_scale * (
                            noise_pred_cond_nk - noise_pred_uncond_nk)

                        nonkey_update_count += 1
                        nonkey_skip_streak = 0
                        updated = True
                        if nonkey_update_mode == "teacache":
                            nonkey_ind_prev = ind_now.detach()
                    else:
                        nonkey_skip_streak += 1

                    temp_nk = nonkey_scheduler.step(
                        nonkey_noise_cache.unsqueeze(0),
                        t,
                        nonkey_latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    nonkey_latents = temp_nk.squeeze(0)

                    extra_context = build_nonkey_extra_context(
                        nonkey_latents, self.model).detach()

                    if debug_dir is not None and self.rank == 0:
                        nonkey_trace.append(
                            dict(step=int(step_i),
                                 t=float(t),
                                 rel_l1=float(rel_l1),
                                 updated=bool(updated)))

                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=seq_len_curr,
                    entropy_collector=collector if (collect_entropy or collect_entropy_trace) else None,
                    extra_context=extra_context,
                )[0]
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    context=context_null,
                    seq_len=seq_len_curr,
                    entropy_collector=None,
                    extra_context=extra_context,
                )[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                # analysis-only: trace one random latent frame across steps
                if collect_entropy_trace and self.rank == 0:
                    if collector.last_frame is None:
                        logging.warning(
                            "[entropy-vis] trace requested but collector.last_frame is None; skip this step"
                        )
                    else:
                        if trace_frame_idx is None:
                            rng = random.Random(int(seed) + int(trace_frame_seed))
                            n_frames = latents[0].shape[1]
                            trace_frame_idx = rng.randrange(n_frames)

                        cur_frame_entropy = float(
                            collector.last_frame[0, trace_frame_idx].detach().cpu().item()
                        )
                        cur_latent_vec = latents[0][:, trace_frame_idx, :, :].reshape(-1).float().detach().cpu()
                        if trace_prev_latent is None:
                            cur_cos = float("nan")
                        else:
                            cur_cos = float(
                                torch.nn.functional.cosine_similarity(
                                    cur_latent_vec.view(1, -1),
                                    trace_prev_latent.view(1, -1),
                                    dim=1,
                                    eps=1e-8,
                                ).item()
                            )
                        trace_step_ids.append(int(step_i))
                        trace_frame_entropy.append(cur_frame_entropy)
                        trace_cos_to_prev.append(cur_cos)
                        trace_prev_latent = cur_latent_vec.clone()

                # (debug) dump per-step frame entropy; at the last entropy step, also save a "last" curve PNG
                # NOTE: keyframe selection may use an aggregated entropy (ema/mean/last) via collector.final();
                # we intentionally save the instantaneous "last" curve here to visualize per-step dynamics.
                if collect_entropy and debug_dir is not None and save_debug_pt and self.rank == 0:
                    if collector.last_frame is None:
                        logging.warning("[entropy-vis] collector.last_frame is None, skip per-step dump")
                    else:
                        ent_last = collector.last_frame[0].detach().cpu()
                        torch.save(
                            ent_last,
                            os.path.join(debug_dir, f"entropy_frame_step_{step_i:02d}.pt"))

                        # Save the instantaneous curve at the end of entropy window (a.k.a. "step 5" when entropy_steps=5)
                        if step_i == entropy_steps - 1:
                            out_png = os.path.join(debug_dir, f"entropy_curve_step_{step_i:02d}_last.png")
                            _save_entropy_curve_png(
                                ent_last,
                                out_png,
                                title=f"frame entropy (last) @ step {step_i} (block {entropy_block_idx})",
                            )

                    if (collector.last_token is not None
                            and collector.last_grid_sizes is not None
                            and collector.last_seq_lens is not None):
                        from .utils.attn_entropy import token_entropy_to_2dmap

                        map2d = token_entropy_to_2dmap(
                            collector.last_token.cpu(),
                            collector.last_grid_sizes.cpu(),
                            collector.last_seq_lens.cpu(),
                        )[0]
                        torch.save(
                            map2d,
                            os.path.join(debug_dir,
                                         f"entropy_token_map_step_{step_i:02d}.pt"))
                        from .utils.vis_entropy import save_block_heatmap_png

                        f_g, h_g, w_g = map(
                            int, collector.last_grid_sizes[0].tolist())
                        map3d = map2d.reshape(f_g, h_g, w_g)

                        imgs_dir = os.path.join(
                            debug_dir, "imgs", f"step{step_i:02d}",
                            f"block{entropy_block_idx:02d}")
                        os.makedirs(imgs_dir, exist_ok=True)
                        for fi in range(f_g):
                            out_png = os.path.join(
                                imgs_dir,
                                f"restored_entropy_f{fi:02d}_cond.png")
                            save_block_heatmap_png(map3d[fi], out_png, dpi=300)

                if keyframe_by_entropy and step_i == entropy_steps - 1:
                    f_lat = latents[0].shape[1]
                    if keyframe_select_mode == "entropy":
                        ent_final = collector.final()[0]
                        key_idx = select_keyframes(
                            ent_final, keyframe_topk, cover=keyframe_cover)
                        # Save the aggregated entropy curve (only when using entropy)
                        if debug_dir is not None and self.rank == 0:
                            try:
                                ent_used = ent_final.detach().cpu()
                                out_png = os.path.join(
                                    debug_dir,
                                    f"entropy_curve_final_{entropy_mode}_keyframes.png",
                                )
                                title = f"frame entropy (final-{entropy_mode}) @ step {step_i} (keyframes marked)"
                                if entropy_mode == "ema":
                                    title += f", alpha={entropy_ema_alpha}"
                                _save_entropy_curve_png(
                                    ent_used,
                                    out_png,
                                    key_idx=key_idx.detach().cpu(),
                                    title=title,
                                )
                            except Exception as e:
                                logging.warning(
                                    f"[entropy-vis] failed to save final curve: {e}"
                                )
                    elif keyframe_select_mode in ("uniform", "random"):
                        key_idx = sample_keyframes(
                            f_lat=f_lat,
                            topk=keyframe_topk,
                            cover=keyframe_cover,
                            mode=keyframe_select_mode,
                            seed=keyframe_sample_seed,
                            device=latents[0].device,
                        )
                    else:
                        raise ValueError(
                            f"Unknown keyframe_select_mode={keyframe_select_mode}"
                        )

                    all_idx = torch.arange(f_lat, device=key_idx.device)
                    mask = torch.ones_like(all_idx, dtype=torch.bool)
                    mask[key_idx] = False
                    nonkey_idx = all_idx[mask]

                    nonkey_lat = latents[0][:, nonkey_idx, :, :].detach()

                    # ===== method-2 init: create nonkey_scheduler + state =====
                    if use_nonkey_context and (nonkey_lat.numel() > 0) and (
                            nonkey_update_mode in ["interval", "teacache"]):
                        nonkey_latents = nonkey_lat.contiguous()
                        nonkey_seq_len = calc_seq_len(
                            nonkey_latents.shape[1])

                        if sample_solver == 'unipc':
                            nonkey_scheduler = FlowUniPCMultistepScheduler(
                                num_train_timesteps=self.num_train_timesteps,
                                shift=1,
                                use_dynamic_shifting=False)
                            nonkey_scheduler.set_timesteps(
                                sampling_steps, device=self.device, shift=shift)
                        elif sample_solver == 'dpm++':
                            nonkey_scheduler = FlowDPMSolverMultistepScheduler(
                                num_train_timesteps=self.num_train_timesteps,
                                shift=1,
                                use_dynamic_shifting=False)
                            sampling_sigmas = get_sampling_sigmas(
                                sampling_steps, shift)
                            _ts, _ = retrieve_timesteps(
                                nonkey_scheduler,
                                device=self.device,
                                sigmas=sampling_sigmas)
                        else:
                            nonkey_scheduler = None

                        if nonkey_scheduler is not None:
                            reset_multistep_state(nonkey_scheduler)

                        nonkey_noise_cache = None
                        nonkey_ind_prev = None
                        nonkey_start_step = step_i + 1
                        nonkey_skip_streak = 0
                        nonkey_update_count = 0
                        nonkey_trace = []

                        _log_debug(
                            f"[method-2] enabled nonkey_update_mode={nonkey_update_mode}, "
                            f"interval={nonkey_update_interval}, "
                            f"thr={teacache_rel_l1_thresh}, "
                            f"max_skip={teacache_max_skip}, "
                            f"warmup={teacache_warmup}")

                    if use_nonkey_context and nonkey_lat.numel() > 0:
                        extra_context = build_nonkey_extra_context(
                            nonkey_lat, self.model).detach()

                    latents = [latents[0][:, key_idx, :, :].contiguous()]
                    seq_len_curr = calc_seq_len(latents[0].shape[1])
                    reset_multistep_state(sample_scheduler)
                    t_entropy_end = time.time()

                    if debug_dir is not None and self.rank == 0:
                        torch.save(
                            key_idx.detach().cpu(),
                            os.path.join(debug_dir, "keyframes_idx_latent.pt"))
                        pixel_idx = (key_idx * self.vae_stride[0]).detach().cpu()
                        torch.save(
                            pixel_idx,
                            os.path.join(debug_dir, "keyframes_idx_pixel.pt"))
                        _log_debug(
                            f"Selected {key_idx.numel()} keyframes (latent idx, mode={keyframe_select_mode}): "
                            f"{key_idx.detach().cpu().tolist()}")
                        if keyframe_select_mode == "entropy":
                            torch.save(
                                ent_final.detach().cpu(),
                                os.path.join(
                                    debug_dir,
                                    f"entropy_frame_final_{entropy_mode}.pt"))

            x0 = latents

            # save analysis-only trace
            if trace_frame_metrics and debug_dir is not None and self.rank == 0 and len(trace_step_ids) > 0:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    out_pt = os.path.join(debug_dir, "trace_random_frame_entropy_cos.pt")
                    torch.save(
                        {
                            "frame_idx": int(trace_frame_idx),
                            "step_ids": torch.tensor(trace_step_ids, dtype=torch.long),
                            "frame_entropy": torch.tensor(trace_frame_entropy, dtype=torch.float32),
                            "cosine_to_prev": torch.tensor(trace_cos_to_prev, dtype=torch.float32),
                            "entropy_block_idx": int(entropy_block_idx),
                            "cos_definition": (
                                "cosine similarity between adjacent-step latent vectors "
                                "[C,H,W] flattened for the selected latent frame"
                            ),
                        },
                        out_pt,
                    )
                    out_png = os.path.join(debug_dir, "trace_random_frame_entropy_cos.png")
                    _save_trace_frame_metrics_png(
                        trace_step_ids, trace_frame_entropy, trace_cos_to_prev, out_png,
                        frame_idx=int(trace_frame_idx), block_idx=int(entropy_block_idx),
                    )
                except Exception as e:
                    logging.warning(f"[entropy-vis] failed to save trace: {e}")

            t_denoise_end = time.time()
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)
                t_decode_end = time.time()

                if debug_dir is not None:
                    run_cfg = dict(
                        keyframe_by_entropy=keyframe_by_entropy,
                        entropy_steps=entropy_steps,
                        entropy_mode=entropy_mode,
                        entropy_ema_alpha=entropy_ema_alpha,
                        entropy_block_idx=entropy_block_idx,
                        keyframe_topk=keyframe_topk,
                        keyframe_cover=keyframe_cover,
                        use_nonkey_context=use_nonkey_context,
                        frame_num=frame_num,
                        vae_stride=list(self.vae_stride),
                        patch_size=list(self.patch_size),
                    )
                    _write_json(os.path.join(debug_dir, "run_config.json"),
                                run_cfg)

                    if key_idx is not None:
                        meta = dict(
                            key_idx_latent=key_idx.detach().cpu().tolist(),
                            key_idx_pixel=(key_idx *
                                           self.vae_stride[0]).detach().cpu().tolist(),
                            frame_num_full=int(frame_num),
                            latent_frames_full=int(target_shape[1]),
                            stride_t=int(self.vae_stride[0]),
                        )
                        _write_json(
                            os.path.join(debug_dir, "keyframes_meta.json"),
                            meta)

                    if profile_timing:
                        timing = dict(
                            total_sec=t_decode_end - t0,
                            denoise_sec=t_denoise_end - t0,
                            decode_sec=t_decode_end - t_denoise_end,
                            entropy_steps=int(entropy_steps),
                            final_latent_frames=int(x0[0].shape[1]),
                            nonkey_update_count=int(nonkey_update_count),
                            nonkey_mode=str(nonkey_update_mode),
                        )
                        if t_entropy_end is not None:
                            timing["entropy_phase_sec"] = t_entropy_end - t0
                            timing["post_entropy_sec"] = t_denoise_end - t_entropy_end
                        _write_json(os.path.join(debug_dir, "timing.json"),
                                    timing)

                    if nonkey_scheduler is not None:
                        _write_json(
                            os.path.join(debug_dir, "nonkey_update_trace.json"),
                            nonkey_trace)
                        _write_json(
                            os.path.join(debug_dir, "nonkey_update_stats.json"),
                            dict(
                                update_count=int(nonkey_update_count),
                                total_steps=int(len(timesteps)),
                                mode=str(nonkey_update_mode),
                                interval=int(nonkey_update_interval),
                                rel_l1_thresh=float(teacache_rel_l1_thresh),
                                max_skip=int(teacache_max_skip),
                                warmup=int(teacache_warmup),
                            ),
                        )
                        if save_teacache_trace_png:
                            save_rel_l1_curve_png(
                                nonkey_trace,
                                os.path.join(debug_dir,
                                             "nonkey_rel_l1_curve.png"),
                            )

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
