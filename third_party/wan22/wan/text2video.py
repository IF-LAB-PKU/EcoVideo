# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


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


def _save_trace_frame_metrics_png(
    step_ids,
    entropy_vals,
    cos_vals,
    out_png,
    *,
    frame_idx,
    block_idx=None,
    dpi=220,
):
    """
    Save one figure for a randomly selected latent frame:
    - left y-axis: frame-level attention entropy
    - right y-axis: cosine similarity between current/previous token-entropy map
    """
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
        logging.warning(f"[entropy-vis] invalid step_ids, skip {out_png}")
        return

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))

    line1 = ax1.plot(
        step_ids,
        entropy_vals,
        marker="o",
        linewidth=2.0,
        label="frame entropy",
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("frame attention entropy")
    ax1.set_xticks(step_ids.tolist())
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        step_ids,
        cos_vals,
        marker="s",
        linewidth=2.0,
        label="cosine to previous step",
    )
    ax2.set_ylabel("cosine similarity to previous step (latent)")
    # auto-scale: let matplotlib choose y limits based on actual data

    title = f"Random latent frame trace (frame={frame_idx})"
    if block_idx is not None:
        title += f" @ block {block_idx}"
    ax1.set_title(title)

    handles = line1 + line2
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="best")

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
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
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
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

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
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

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
                 # ---- keyframe-by-entropy ----
                 keyframe_by_entropy: bool = False,
                 keyframe_target_fps: float = 8.0,
                 entropy_steps: int = 5,
                 entropy_mode: str = "mean",  # "last" | "mean" | "ema"
                 entropy_ema_alpha: float = 0.6,
                 entropy_block_idx: int = -1,  # -1 = last block
                 keyframe_cover: bool = True,
                 debug_dir: str | None = None,
                 save_debug_pt: bool = True,
                 profile_timing: bool = False,
                 # ---- analysis-only step trace ----
                 trace_frame_metrics: bool = False,
                 trace_steps: int = 50,
                 trace_frame_seed: int = 2026,
                 save_latents_pt: str | None = None,
                 save_latent_stats_json: str | None = None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        # ===================== helpers for entropy keyframes =====================

        def auto_keyframe_topk(frame_num_full: int,
                               fps_full: float,
                               fps_key: float,
                               stride_t: int,
                               min_k: int = 2,
                               max_k: int | None = None) -> int:
            """
            根据目标关键帧 fps 自动估计 latent 关键帧数量 K。
            frame_num_full: 像素帧数（例如 81）
            fps_full: 模型采样 fps（A14B 通常是 config.sample_fps=16）
            stride_t: vae 时间 stride（通常 4）
            """
            if fps_key <= 0:
                raise ValueError(f"fps_key must be > 0, got {fps_key}")
            if frame_num_full <= 1:
                return max(min_k, 1)

            # 目标像素关键帧数（近似，保持时长不变）
            t_key = int(round(frame_num_full * (fps_key / float(fps_full))))
            t_key = max(2, t_key)

            # 对齐到 VAE 约束：像素帧数 = stride_t * n + 1（stride_t=4）
            if stride_t > 1:
                n = int(round((t_key - 1) / float(stride_t)))
                t_key = n * stride_t + 1
                t_key = max(stride_t + 1, t_key)  # 至少 2 个 latent -> 至少 4+1 像素帧

            k = int(round((t_key - 1) / float(stride_t))) + 1
            k = max(min_k, k)
            if max_k is not None:
                k = min(max_k, k)
            return k

        def select_keyframes(ent_1d: torch.Tensor,
                             topk: int,
                             cover: bool = True) -> torch.Tensor:
            """
            ent_1d: [F_latent]
            return: sorted indices [K]
            """
            f = int(ent_1d.numel())
            k = min(max(1, int(topk)), f)

            # topk by entropy
            vals, idx = torch.topk(ent_1d, k=k, largest=True, sorted=False)
            _ = vals  # suppress unused warning
            idx = idx.unique()

            if cover and f >= 2:
                idx = torch.cat([idx, idx.new_tensor([0, f - 1])]).unique()

            # 如果超了 k：优先保留首尾，其余从高熵补
            if idx.numel() > k:
                keep_list = []
                if cover and f >= 2:
                    keep_list += [0, f - 1]
                keep = torch.tensor(
                    list(dict.fromkeys(keep_list)),
                    device=idx.device,
                    dtype=idx.dtype,
                )
                if keep.numel() < k:
                    order = torch.argsort(ent_1d, descending=True)
                    for j in order.tolist():
                        if j in keep.tolist():
                            continue
                        keep = torch.cat([keep, keep.new_tensor([j])])
                        if keep.numel() >= k:
                            break
                idx = keep[:k]

            # 如果不足 k：用均匀 bucket 补齐
            if idx.numel() < k and f > 1:
                need = k - idx.numel()
                if need > 0:
                    extra = torch.linspace(
                        0,
                        f - 1,
                        steps=need + 2,
                        device=idx.device,
                    )[1:-1].round().long()
                    idx = torch.cat([idx, extra]).unique()
                    if idx.numel() > k:
                        idx = idx[:k]

            return torch.sort(idx)[0]

        def reset_multistep_state(scheduler):
            # 复位 UniPC / DPM++ 的多步状态，确保裁剪后继续采样正常
            if hasattr(scheduler, "config") and hasattr(
                    scheduler.config, "solver_order"):
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

        # ========================================================================

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

        from .utils.entropy_collector import EntropyCollector

        # latent 总帧数（VAE latent time length）
        latent_frames_full = int(target_shape[1])
        fps_full = float(getattr(self.config, "sample_fps", 16))
        stride_t = int(self.vae_stride[0])

        keyframe_topk = auto_keyframe_topk(
            frame_num_full=frame_num,
            fps_full=fps_full,
            fps_key=float(keyframe_target_fps),
            stride_t=stride_t,
            min_k=2,
            max_k=latent_frames_full,
        )

        collector = EntropyCollector(
            enabled=False,
            mode=entropy_mode,
            ema_alpha=entropy_ema_alpha,
            block_idx=entropy_block_idx,
        )
        collector.reset()

        # analysis-only: trace one random latent frame across steps
        trace_frame_idx = None
        trace_step_ids = []
        trace_frame_entropy = []
        trace_cos_to_prev = []
        trace_prev_latent = None  # latent vector of selected frame from previous step

        seq_len_curr = seq_len  # may change after cropping
        key_idx = None

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

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
            if entropy_steps < 1:
                entropy_steps = 1

            if trace_frame_metrics:
                trace_steps = int(trace_steps)
                if trace_steps > len(timesteps):
                    trace_steps = len(timesteps)
                if trace_steps < 1:
                    trace_steps = 1
            else:
                trace_steps = 0

            # sample videos
            latents = noise
            for step_i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                collect_entropy_keyframe = bool(keyframe_by_entropy) and (
                    step_i < entropy_steps
                )
                collect_entropy_trace = bool(trace_frame_metrics) and (
                    step_i < trace_steps
                )
                collect_entropy = collect_entropy_keyframe or collect_entropy_trace
                collector.enabled = collect_entropy

                noise_pred_cond = model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=seq_len_curr,
                    entropy_collector=collector if collect_entropy else None,
                )[0]
                noise_pred_uncond = model(
                    latent_model_input,
                    t=timestep,
                    context=context_null,
                    seq_len=seq_len_curr,
                )[0]

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                # (debug) dump per-step frame entropy; at the last entropy step, also save a "last" curve PNG
                # NOTE: keyframe selection may use an aggregated entropy (ema/mean/last) via collector.final();
                # we intentionally save the instantaneous "last" curve here to visualize per-step dynamics.
                if collect_entropy and debug_dir is not None and save_debug_pt and self.rank == 0:
                    os.makedirs(debug_dir, exist_ok=True)
                    if collector.last_frame is None:
                        logging.warning("[entropy-vis] collector.last_frame is None, skip per-step dump")
                    else:
                        ent_last = collector.last_frame[0].detach().cpu()  # [F_latent]
                        torch.save(
                            ent_last,
                            os.path.join(debug_dir, f"entropy_frame_step_{step_i:02d}.pt"),
                        )

                        # Save the instantaneous curve at the end of entropy window (a.k.a. "step 5" when entropy_steps=5)
                        if step_i == entropy_steps - 1:
                            out_png = os.path.join(debug_dir, f"entropy_curve_step_{step_i:02d}_last.png")
                            _save_entropy_curve_png(
                                ent_last,
                                out_png,
                                title=f"frame entropy (last) @ step {step_i} (block {entropy_block_idx})",
                            )

                # analysis-only: trace one random latent frame across steps
                # - frame entropy: scalar from entropy collector
                # - cosine: between latent vectors of the selected frame at adjacent steps
                #   (latents already updated to current step above; reflects real denoising dynamics)
                if collect_entropy_trace and self.rank == 0:
                    if collector.last_frame is None:
                        logging.warning(
                            "[entropy-vis] trace requested but collector.last_frame is None; skip this step"
                        )
                    else:
                        # pick frame index once (from latent time dimension T)
                        if trace_frame_idx is None:
                            rng = random.Random(int(seed) + int(trace_frame_seed))
                            n_frames = latents[0].shape[1]  # latents shape: [C, T, H, W]
                            trace_frame_idx = rng.randrange(n_frames)

                        # frame-level entropy (scalar)
                        cur_frame_entropy = float(
                            collector.last_frame[0, trace_frame_idx].detach().cpu().item()
                        )

                        # cosine on latent: [C, H, W] flattened to 1D, then cos with previous step
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

                # -------- crop latent time dimension at the end of entropy window --------
                if keyframe_by_entropy and (step_i == entropy_steps - 1):
                    ent_final = collector.final()[0]  # [F_latent]
                    key_idx = select_keyframes(
                        ent_final,
                        topk=keyframe_topk,
                        cover=keyframe_cover,
                    )

                    # Save the aggregated entropy curve used for keyframe selection (EMA if entropy_mode=ema)
                    # with selected keyframes marked.
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
                            logging.warning(f"[entropy-vis] failed to save final curve: {e}")

                    # crop latent frames: [C, F_latent, H, W] -> keep only key frames
                    latents = [
                        latents[0][:, key_idx, :, :].contiguous(),
                    ]

                    # recompute seq_len for cropped latents
                    # seq_len ≈ ceil( (H*W)/(patch_hw) * F_latent / sp_size ) * sp_size
                    f_lat = int(latents[0].shape[1])
                    seq_len_curr = math.ceil(
                        (target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        f_lat / self.sp_size) * self.sp_size

                    reset_multistep_state(sample_scheduler)

                    # (optional) debug dump
                    if debug_dir is not None and self.rank == 0:
                        os.makedirs(debug_dir, exist_ok=True)
                        if save_debug_pt:
                            torch.save(
                                {
                                    "key_idx":
                                    key_idx.detach().cpu(),
                                    "entropy":
                                    ent_final.detach().cpu(),
                                    "keyframe_topk":
                                    int(keyframe_topk),
                                    "keyframe_target_fps":
                                    float(keyframe_target_fps),
                                    "fps_full":
                                    float(fps_full),
                                    "frame_num_full":
                                    int(frame_num),
                                    "latent_frames_full":
                                    int(latent_frames_full),
                                    "latent_frames_key":
                                    int(f_lat),
                                },
                                os.path.join(debug_dir,
                                             "entropy_keyframes.pt"),
                            )

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
                        trace_step_ids,
                        trace_frame_entropy,
                        trace_cos_to_prev,
                        out_png,
                        frame_idx=int(trace_frame_idx),
                        block_idx=int(entropy_block_idx),
                    )
                except Exception as e:
                    logging.warning(f"[entropy-vis] failed to save trace_random_frame_entropy_cos: {e}")

            x0 = latents
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                # ---- dump latent size before VAE decode (keyframes latent if entropy crop enabled) ----
                latent = x0[0]  # torch.Tensor, [C_lat, T_lat, H_lat, W_lat]
                tensor_bytes = int(latent.numel() * latent.element_size())

                packed_bytes = None
                try:
                    import io
                    buf = io.BytesIO()
                    torch.save(latent.detach().cpu(), buf)
                    packed_bytes = int(len(buf.getvalue()))
                except Exception:
                    pass

                base64_bytes = None
                if packed_bytes is not None:
                    base64_bytes = int(4 * ((packed_bytes + 2) // 3))

                stats = {
                    "latent_shape": list(latent.shape),
                    "latent_dtype": str(latent.dtype),
                    "tensor_bytes": tensor_bytes,
                    "packed_bytes": packed_bytes,
                    "base64_bytes": base64_bytes,
                }

                if (save_latent_stats_json is None) and (debug_dir is not None):
                    save_latent_stats_json = os.path.join(debug_dir, "wan_latent_size.json")

                if save_latent_stats_json is not None:
                    os.makedirs(os.path.dirname(save_latent_stats_json) or ".", exist_ok=True)
                    import json
                    with open(save_latent_stats_json, "w", encoding="utf-8") as f:
                        json.dump(stats, f, indent=2)

                if save_latents_pt is not None:
                    os.makedirs(os.path.dirname(save_latents_pt) or ".", exist_ok=True)
                    torch.save(latent.detach().cpu(), save_latents_pt)

                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
