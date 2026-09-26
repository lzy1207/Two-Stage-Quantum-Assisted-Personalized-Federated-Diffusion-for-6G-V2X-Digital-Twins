"""Quantum-prototype-conditioned REM diffusion, PDF equations (43)--(58).

Maps use BCHW layout and the training normalization (normally [-1, 1]).
Coordinates use [0, 1] with x=column and y=row. Timesteps are zero-indexed
internally: t=0 corresponds to paper t=1. The native PyTorch U-Net widths and
two residual blocks per scale follow the supplied coordinate-DDPM notebook;
the paper's prototype projection and FiLM extend that reference architecture.
No model weights or external downloads are needed by this module.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return math.gcd(channels, min(8, max(1, channels // 4)))


def _time_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    scale = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    angles = t.float()[:, None] * scale[None, :]
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


class _FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = nn.Linear(embedding_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, time: Tensor, prototype: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time)[:, :, None, None]
        h = self.norm2(h)
        gamma, beta = self.film(prototype).chunk(2, dim=-1)
        h = (1 + gamma[:, :, None, None]) * h + beta[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class _SpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        # PyTorch dispatches memory-efficient CUDA attention when supported.
        q, k, v = (item.transpose(1, 2).unsqueeze(1) for item in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2)
        return x + self.proj(out.reshape(b, c, h, w))


class _Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, attention: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            _FiLMResidualBlock(in_channels, out_channels, embedding_dim),
            _FiLMResidualBlock(out_channels, out_channels, embedding_dim),
        ])
        self.attention = _SpatialAttention(out_channels) if attention else nn.Identity()

    def forward(self, x: Tensor, time: Tensor, prototype: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, time, prototype)
        return self.attention(x)


class ConditionalUNet(nn.Module):
    """Predict epsilon from noisy map, spatial conditions and QNN moments.

    Spatial channels are coarse REM, Gaussian transmitter heatmap, then
    optional semantic channels. Every residual block has independent FiLM
    projections of the global quantum prototype embedding (PDF Eq. 47).
    The architecture is reconstructed, not checkpoint-compatible with the
    notebook's coordinate-only diffusers.UNet2DModel.
    """

    def __init__(self, semantic_channels: int, prototype_dim: int, base_channels: int = 64,
                 channel_mults: Sequence[int] = (1, 2, 4, 4), embedding_dim: int = 64):
        super().__init__()
        if semantic_channels < 0 or prototype_dim < 1 or base_channels < 1 or embedding_dim < 4:
            raise ValueError("Invalid U-Net channel or embedding dimensions")
        if not channel_mults or any(int(m) < 1 for m in channel_mults):
            raise ValueError("channel_mults must be nonempty positive integers")
        self.semantic_channels = int(semantic_channels)
        self.prototype_dim = int(prototype_dim)
        self.embedding_dim = int(embedding_dim)
        self.config = dict(semantic_channels=semantic_channels, prototype_dim=prototype_dim,
                           base_channels=base_channels, channel_mults=list(channel_mults),
                           embedding_dim=embedding_dim)
        channels = [base_channels * int(m) for m in channel_mults]
        self.time_projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 4), nn.SiLU(),
                                             nn.Linear(embedding_dim * 4, embedding_dim))
        self.prototype_projection = nn.Sequential(nn.Linear(prototype_dim, embedding_dim), nn.SiLU())
        self.input_conv = nn.Conv2d(3 + semantic_channels, channels[0], 3, padding=1)
        attention_level = max(0, len(channels) - 2)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        previous = channels[0]
        for level, c in enumerate(channels):
            self.down.append(_Stage(previous, c, embedding_dim, level == attention_level))
            if level < len(channels) - 1:
                self.downsample.append(nn.Conv2d(c, channels[level + 1], 3, stride=2, padding=1))
                previous = channels[level + 1]
        self.middle = _Stage(channels[-1], channels[-1], embedding_dim, True)
        self.up = nn.ModuleList()
        previous = channels[-1]
        for level in reversed(range(len(channels))):
            c = channels[level]
            self.up.append(_Stage(previous + c, c, embedding_dim, level == attention_level))
            previous = c
        self.output_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.output_conv = nn.Conv2d(channels[0], 1, 3, padding=1)

    def forward(self, noisy: Tensor, t: Tensor, spatial: Tensor, prototype: Tensor) -> Tensor:
        if noisy.ndim != 4 or noisy.shape[1] != 1:
            raise ValueError("noisy must have shape [B,1,H,W]")
        expected = (noisy.shape[0], 2 + self.semantic_channels, *noisy.shape[-2:])
        if tuple(spatial.shape) != expected:
            raise ValueError(f"spatial must have shape {expected}; got {tuple(spatial.shape)}")
        if tuple(prototype.shape) != (noisy.shape[0], self.prototype_dim):
            raise ValueError(f"prototype must have shape [B,{self.prototype_dim}]")
        t = torch.as_tensor(t, device=noisy.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(noisy.shape[0])
        if t.numel() != noisy.shape[0]:
            raise ValueError("t must be scalar or contain one timestep per batch item")
        time = self.time_projection(_time_embedding(t, self.embedding_dim).to(noisy.dtype))
        quantum = self.prototype_projection(prototype.to(noisy.dtype))
        h = self.input_conv(torch.cat((noisy, spatial), dim=1))
        skips = []
        for i, stage in enumerate(self.down):
            h = stage(h, time, quantum)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self.middle(h, time, quantum)
        for stage, skip in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = stage(torch.cat((h, skip), dim=1), time, quantum)
        return self.output_conv(F.silu(self.output_norm(h)))


def make_spatial_condition(coarse: Tensor, tx_xy: Tensor, semantics: Tensor | None,
                           sigma: float = .04) -> Tensor:
    """Build Eq. (45), using a [0,1] heatmap without per-image rescaling.

    A coarse map has shape [B,1,H,W], tx_xy is [B,2] with normalized x,y,
    and semantics is [B,C,H,W] or None. No target map is accepted.
    """
    if coarse.ndim != 4 or coarse.shape[1] != 1:
        raise ValueError("coarse must have shape [B,1,H,W]")
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    xy = torch.as_tensor(tx_xy, device=coarse.device, dtype=coarse.dtype)
    if xy.shape != (coarse.shape[0], 2) or not torch.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
        raise ValueError("tx_xy must have shape [B,2] with finite normalized coordinates in [0,1]")
    h, w = coarse.shape[-2:]
    y = torch.linspace(0, 1, h, device=coarse.device, dtype=coarse.dtype)[None, :, None]
    x = torch.linspace(0, 1, w, device=coarse.device, dtype=coarse.dtype)[None, None, :]
    squared_distance = (x - xy[:, 0, None, None]).square() + (y - xy[:, 1, None, None]).square()
    position = torch.exp(-squared_distance / (2 * sigma ** 2))[:, None]
    parts = [coarse, position]
    if semantics is not None:
        if semantics.ndim != 4 or semantics.shape[0] != coarse.shape[0] or semantics.shape[-2:] != (h, w):
            raise ValueError("semantics must have shape [B,C,H,W] matching coarse")
        parts.append(semantics.to(device=coarse.device, dtype=coarse.dtype))
    return torch.cat(parts, dim=1)


def ssim_map(x: Tensor, y: Tensor, data_range: float = 2.0) -> Tensor:
    """Differentiable local SSIM, Gaussian window <=11 and population moments.

    Inputs are BCHW. Window sigma=1.5, C1=(.01*range)^2, C2=(.03*range)^2.
    Replicate padding keeps constant images constant at the map boundary.
    The returned tensor has the input shape; reductions are caller-defined.
    """
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 1:
        raise ValueError("SSIM inputs must have identical nonempty BCHW shapes")
    if data_range <= 0 or not math.isfinite(float(data_range)):
        raise ValueError("data_range must be finite and positive")
    window = min(11, x.shape[-2], x.shape[-1])
    window -= 1 - window % 2
    axis = torch.arange(window, device=x.device, dtype=x.dtype) - (window - 1) / 2
    gaussian = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None].expand(x.shape[1], 1, -1, -1)
    padding = window // 2

    def average(z: Tensor) -> Tensor:
        return F.conv2d(F.pad(z, (padding,) * 4, mode="replicate"), kernel, groups=x.shape[1])

    mux, muy = average(x), average(y)
    vx = (average(x.square()) - mux.square()).clamp_min(0)
    vy = (average(y.square()) - muy.square()).clamp_min(0)
    covariance = average(x * y) - mux * muy
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    return ((2 * mux * muy + c1) * (2 * covariance + c2) /
            ((mux.square() + muy.square() + c1) * (vx + vy + c2)))


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


class GaussianDiffusion(nn.Module):
    """Linear-schedule epsilon DDPM with the paper's composite objective.

    Full sampling follows Eq. (53) and posterior variance without clipping.
    Setting steps<T explicitly selects deterministic eta=0 DDIM acceleration;
    it is useful for smoke tests and is not the paper's full reverse chain.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = .02):
        super().__init__()
        if not isinstance(timesteps, int) or timesteps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Require positive integer timesteps and 0<beta_start<=beta_end<1")
        self.timesteps = timesteps
        self.config = dict(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1 - betas
        alpha_bars = alphas.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bars[:-1]))
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())
        self.register_buffer("posterior_variance", (betas * (1 - previous) / (1 - alpha_bars)).float())

    def _extract(self, array: Tensor, t: Tensor, x: Tensor) -> Tensor:
        t = torch.as_tensor(t, device=x.device, dtype=torch.long).reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        if t.numel() != x.shape[0] or (t < 0).any() or (t >= self.timesteps).any():
            raise ValueError("Invalid batch timesteps")
        return array.to(device=x.device, dtype=x.dtype)[t].reshape(-1, *([1] * (x.ndim - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        if noise.shape != x0.shape:
            raise ValueError("noise and x0 must have identical shapes")
        abar = self._extract(self.alpha_bars, t, x0)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def training_loss(self, model: nn.Module, x0: Tensor, spatial: Tensor, prototype: Tensor,
                      weights: Sequence[float] = (1, .1, .1), mask: Tensor | None = None,
                      data_range: float = 2.0) -> dict[str, Tensor]:
        """Return differentiable loss/noise/rec/grad/ssim scalars.

        weights=(lambda_rec,lambda_grad,lambda_ssim). Default loss follows the
        paper on the full image. Optional mask is a documented valid-pixel
        extension; a gradient edge is included only when both endpoints are
        valid, and SSIM centers are restricted by mask. Unclipped x0 estimates
        retain the published loss; all loss arithmetic runs in float32.
        data_range is the fixed span in the target's normalization: 2 for
        [-1,1] maps, or (training_max_db-training_min_db)/std for z-scores.
        """
        if len(weights) != 3 or any(not math.isfinite(float(v)) or v < 0 for v in weights):
            raise ValueError("weights must contain three finite nonnegative coefficients")
        if not math.isfinite(float(data_range)) or data_range <= 0:
            raise ValueError("data_range must be finite and positive")
        if mask is not None:
            mask = torch.as_tensor(mask, device=x0.device, dtype=torch.float32)
            if mask.ndim == 3:
                mask = mask[:, None]
            if mask.shape != x0.shape or not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any() or mask.sum() <= 0:
                raise ValueError("mask must match x0, lie in [0,1], and contain valid pixels")
        t = torch.randint(self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        predicted_noise = model(xt, t, spatial, prototype)
        if predicted_noise.shape != x0.shape:
            raise ValueError("The denoising model must return the x0 shape")
        # Explicitly disable outer AMP for clean estimates and structural loss.
        with torch.autocast(device_type=x0.device.type, enabled=False):
            target, eps, prediction = x0.float(), noise.float(), predicted_noise.float()
            abar = self._extract(self.alpha_bars, t, target)
            clean = (xt.float() - (1 - abar).sqrt() * prediction) / abar.sqrt()
            noise_loss = _masked_mean((prediction - eps).square(), mask)
            rec_loss = _masked_mean((clean - target).square(), mask)
            residual = clean - target
            dx = (residual[..., 1:] - residual[..., :-1]).abs()
            dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs()
            if mask is None:
                grad_loss = (dx.sum() + dy.sum()) / target.numel()
            else:
                mx = mask[..., 1:] * mask[..., :-1]
                my = mask[..., 1:, :] * mask[..., :-1, :]
                grad_loss = ((dx * mx).sum() + (dy * my).sum()) / mask.sum()
            structural = _masked_mean(1 - ssim_map(clean, target, data_range=data_range), mask)
            total = noise_loss + weights[0] * rec_loss + weights[1] * grad_loss + weights[2] * structural
        return dict(loss=total, noise=noise_loss, rec=rec_loss, grad=grad_loss, ssim=structural)

    @torch.no_grad()
    def sample(self, model: nn.Module, spatial: Tensor, prototype: Tensor, steps: int | None = None,
               generator: torch.Generator | None = None) -> Tensor:
        """Generate maps without accepting or accessing any target image.

        generator must target the same device as spatial. A specified steps<T
        selects a deterministic DDIM trajectory with stochastic initial noise.
        Output is not clipped; apply physical output constraints explicitly in
        the caller if the intended experiment requires them.
        """
        steps = self.timesteps if steps is None else steps
        if not isinstance(steps, int) or not 1 <= steps <= self.timesteps:
            raise ValueError("steps must be an integer between 1 and timesteps")
        if spatial.ndim != 4 or spatial.shape[1] < 2 or prototype.shape[0] != spatial.shape[0]:
            raise ValueError("Invalid spatial/prototype batch shapes")
        shape = (spatial.shape[0], 1, *spatial.shape[-2:])
        x = torch.randn(shape, device=spatial.device, dtype=spatial.dtype, generator=generator)
        was_training = model.training
        model.eval()
        try:
            if steps == self.timesteps:
                for index in range(self.timesteps - 1, -1, -1):
                    t = torch.full((shape[0],), index, device=x.device, dtype=torch.long)
                    eps = model(x, t, spatial, prototype)
                    beta = self._extract(self.betas, t, x)
                    alpha = self._extract(self.alphas, t, x)
                    abar = self._extract(self.alpha_bars, t, x)
                    x = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
                    if index > 0:
                        noise = torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
                        x = x + self._extract(self.posterior_variance, t, x).sqrt() * noise
            else:
                indices = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
                for position, index in enumerate(indices):
                    t = torch.full((shape[0],), index, device=x.device, dtype=torch.long)
                    eps = model(x, t, spatial, prototype)
                    abar = self._extract(self.alpha_bars, t, x)
                    clean = (x - (1 - abar).sqrt() * eps) / abar.sqrt()
                    if position + 1 == len(indices):
                        x = clean
                    else:
                        next_t = torch.full_like(t, indices[position + 1])
                        next_abar = self._extract(self.alpha_bars, next_t, x)
                        x = next_abar.sqrt() * clean + (1 - next_abar).sqrt() * eps
            return x
        finally:
            model.train(was_training)
