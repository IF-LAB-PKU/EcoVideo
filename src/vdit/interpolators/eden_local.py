“””
EDEN local interpolation wrapper: extracts single midpoint interpolation from inference.py into a reusable class.

Design goals:
- Pipeline layer depends only on a stable interface: EdenInterpolator.interpolate(Ia, Ib) -> Imid
- No dependency on inference.py global args / global model variables
- Compatible with single-GPU and dual-GPU (encoder vs DiT+decoder split) modes

Input/output conventions:
- frame: torch.Tensor [1,3,H,W], float, range [0,1], can be on CPU or GPU
- returns: torch.Tensor [1,3,H,W], float, range [0,1], defaults to CPU (for caching/saving)
“””

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import yaml

from vdit.models import load_model
from vdit.transport import Sampler, create_transport
from vdit.utils import InputPadder
from vdit.utils.encode_transfer import pack_enc_out, unpack_enc_out


@dataclass(frozen=True)
class EdenDevices:
    enc: torch.device
    ditdec: torch.device


class EdenInterpolator:
    def __init__(
        self,
        *,
        config_path: str,
        device: str | torch.device = "cuda:0",
        use_split_gpu: bool = False,
        device_enc: str | torch.device = "cuda:0",
        device_ditdec: str | torch.device = "cuda:1",
    ) -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Use config dict directly instead of argparse + set_defaults as in inference.py
        self.cfg: Dict[str, Any] = dict(cfg)
        self.model_name: str = self.cfg["model_name"]
        self.pretrained_eden_path: str = self.cfg["pretrained_eden_path"]
        self.model_args: Dict[str, Any] = self.cfg["model_args"]
        self.cos_sim_mean: float = float(self.cfg.get("cos_sim_mean", 0.0))
        self.cos_sim_std: float = float(self.cfg.get("cos_sim_std", 1.0))
        self.vae_scaler: float = float(self.cfg.get("vae_scaler", 1.0))
        self.vae_shift: float = float(self.cfg.get("vae_shift", 0.0))

        self.use_split_gpu = bool(use_split_gpu)
        self.device = torch.device(device)
        self.devices = EdenDevices(enc=torch.device(device_enc), ditdec=torch.device(device_ditdec))

        ckpt = torch.load(self.pretrained_eden_path, map_location="cpu")
        eden_state = ckpt["eden"]
        del ckpt

        if self.use_split_gpu and torch.cuda.device_count() < 2:
            # Auto fallback when only one GPU is available
            self.use_split_gpu = False

        if not self.use_split_gpu:
            self.eden = load_model(self.model_name, **self.model_args)
            self.eden.load_state_dict(eden_state)
            self.eden.to(self.device).eval()

            transport = create_transport("Linear", "velocity")
            sampler = Sampler(transport)
            self.sample_fn = sampler.sample_ode(
                sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3
            )

            self.eden_enc = None
            self.eden_ditdec = None
            self.sample_fn_ditdec = None
        else:
            # encoder (cuda:0)
            self.eden_enc = load_model(self.model_name, **self.model_args)
            self.eden_enc.load_state_dict(eden_state)
            self.eden_enc.to(self.devices.enc).eval()

            # DiT+decoder (cuda:1)
            self.eden_ditdec = load_model(self.model_name, **self.model_args)
            self.eden_ditdec.load_state_dict(eden_state)
            self.eden_ditdec.to(self.devices.ditdec).eval()

            transport_ditdec = create_transport("Linear", "velocity")
            sampler_ditdec = Sampler(transport_ditdec)
            self.sample_fn_ditdec = sampler_ditdec.sample_ode(
                sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3
            )

            self.eden = None
            self.sample_fn = None

    @torch.no_grad()
    def interpolate(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        """Interpolate one midpoint frame."""
        h, w = frame0.shape[2:]
        padder = InputPadder([h, w])

        if not self.use_split_gpu:
            assert self.eden is not None and self.sample_fn is not None
            f0 = frame0.to(self.device)
            f1 = frame1.to(self.device)

            difference = (
                (torch.mean(torch.cosine_similarity(f0, f1), dim=[1, 2]) - self.cos_sim_mean)
                / (self.cos_sim_std + 1e-8)
            ).unsqueeze(1).to(self.device)

            cond_frames = padder.pad(torch.cat((f0, f1), dim=0))
            enc_out = self.eden.encode(cond_frames)

            new_h, new_w = cond_frames.shape[2:]
            noise = torch.randn(
                [1, new_h // 32 * new_w // 32, self.model_args["latent_dim"]],
                device=self.device,
            )

            def denoise_wrapper(query_latents: torch.Tensor, t: Any) -> torch.Tensor:
                if isinstance(t, torch.Tensor):
                    denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t[0:1]
                else:
                    denoise_timestep = torch.tensor([t], device=query_latents.device, dtype=torch.float32)
                if denoise_timestep.dim() == 0:
                    denoise_timestep = denoise_timestep.unsqueeze(0)
                return self.eden.denoise_from_tokens(query_latents, denoise_timestep, enc_out, difference)

            samples = self.sample_fn(noise, denoise_wrapper)[-1]
            denoise_latents = samples / self.vae_scaler + self.vae_shift
            generated = self.eden.decode(denoise_latents).clamp(0.0, 1.0)
            generated = padder.unpad(generated).cpu()
            return generated

        # split-gpu
        assert self.eden_enc is not None and self.eden_ditdec is not None and self.sample_fn_ditdec is not None

        f0_enc = frame0.to(self.devices.enc)
        f1_enc = frame1.to(self.devices.enc)
        cond_frames_enc = padder.pad(torch.cat((f0_enc, f1_enc), dim=0))

        difference = (
            (torch.mean(torch.cosine_similarity(f0_enc, f1_enc), dim=[1, 2]) - self.cos_sim_mean)
            / (self.cos_sim_std + 1e-8)
        ).unsqueeze(1)

        enc_out = self.eden_enc.encode(cond_frames_enc)
        blob = pack_enc_out(enc_out)  # -> CPU bytes-like

        enc_out_dit = unpack_enc_out(blob, self.devices.ditdec)
        difference_dit = difference.to(self.devices.ditdec)

        new_h, new_w = cond_frames_enc.shape[2:]
        noise = torch.randn(
            [1, new_h // 32 * new_w // 32, self.model_args["latent_dim"]],
            device=self.devices.ditdec,
        )

        def denoise_wrapper(query_latents: torch.Tensor, t: Any) -> torch.Tensor:
            if isinstance(t, torch.Tensor):
                denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t[0:1]
            else:
                denoise_timestep = torch.tensor([t], device=query_latents.device, dtype=torch.float32)
            if denoise_timestep.dim() == 0:
                denoise_timestep = denoise_timestep.unsqueeze(0)
            return self.eden_ditdec.denoise_from_tokens(query_latents, denoise_timestep, enc_out_dit, difference_dit)

        samples = self.sample_fn_ditdec(noise, denoise_wrapper)[-1]
        denoise_latents = samples / self.vae_scaler + self.vae_shift
        generated = self.eden_ditdec.decode(denoise_latents).clamp(0.0, 1.0)
        generated = padder.unpad(generated.cpu())
        return generated


