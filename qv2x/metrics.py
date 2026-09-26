"""Held-out REM metrics, PDF equations (62)--(65), with explicit units.

RMSE is pooled over every valid pixel (not averaged per-image RMSE). SSIM is
averaged over full physical-unit maps as Eq. (63), with Gaussian-window
population moments and a fixed, caller-supplied physical data_range. Neither
prediction nor target is independently rescaled or shifted for SSIM.

Radial bins are equal-width from zero to the farthest grid point of each
transmitter. Empty bins are omitted from radial MAE; trend violation averages
positive outward jumps over adjacent, populated bins only. This explicit
empty-bin convention is necessary for tiny test maps and masked buildings.
mean_power_db is the pooled arithmetic mean of predicted dB values (not
linear power averaging or the paper's normalized Mean Power). The pipeline
reports a separate mean_power using full training-map valid-pixel min/max.
No prediction/seed is selected using ground truth.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from .diffusion import ssim_map


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _maps(value, name: str) -> np.ndarray:
    value = _numpy(value)
    if value.ndim == 2:
        value = value[None]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{name} must be [H,W], [N,H,W], or [N,1,H,W]")
    return value


def evaluate_maps(pred_db, true_db, valid, tx_xy, data_range: float, radial_bins: int = 16) -> dict[str, float | None]:
    """Return rmse, ssim, radial_mae, trend_violation and mean_power_db.

    Prediction/target units are dB. valid uses True for evaluated pixels.
    tx_xy uses normalized x,y coordinates in [0,1]. data_range is the fixed
    physical normalization span (for example max_db-min_db), never a range
    independently inferred from predictions. NumPy arrays or tensors work.
    trend_violation is None if no adjacent populated radial bins are present.
    """
    prediction = _maps(pred_db, "pred_db").astype(np.float64)
    truth = _maps(true_db, "true_db").astype(np.float64)
    mask = _maps(valid, "valid").astype(bool)
    if prediction.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("Prediction, target and mask shapes must match")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Prediction and target must be finite, including building pixels")
    if not np.all(mask.reshape(mask.shape[0], -1).any(axis=1)):
        raise ValueError("Every map must have at least one valid pixel")
    if not math.isfinite(float(data_range)) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if not isinstance(radial_bins, int) or radial_bins < 1:
        raise ValueError("radial_bins must be a positive integer")
    xy = _numpy(tx_xy).astype(float)
    if xy.shape == (2,) and len(prediction) == 1:
        xy = xy[None]
    if xy.shape != (len(prediction), 2) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
        raise ValueError("tx_xy must have shape [N,2] and normalized x,y coordinates")
    errors = prediction[mask] - truth[mask]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    # Use float64 because physical dB baselines otherwise lose precision in
    # local variance subtraction for nearly constant maps.
    pt = torch.from_numpy(prediction)[:, None]
    gt = torch.from_numpy(truth)[:, None]
    with torch.no_grad():
        ssim = float(ssim_map(pt, gt, data_range).mean().item())
    n, h, w = prediction.shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    radial_errors, trend_errors = [], []
    for i in range(n):
        radii = np.hypot(xx - xy[i, 0], yy - xy[i, 1])
        maximum = max(float(radii.max()), np.finfo(float).eps)
        bins = np.minimum((radii / maximum * radial_bins).astype(np.int64), radial_bins - 1)
        pred_means = np.full(radial_bins, np.nan)
        for k in range(radial_bins):
            inside = mask[i] & (bins == k)
            if inside.any():
                pred_means[k] = prediction[i][inside].mean()
                radial_errors.append(abs(pred_means[k] - truth[i][inside].mean()))
        for left, right in zip(pred_means[:-1], pred_means[1:]):
            if np.isfinite(left) and np.isfinite(right):
                trend_errors.append(max(float(right - left), 0.0))
    return dict(rmse=rmse, ssim=ssim, radial_mae=float(np.mean(radial_errors)),
                trend_violation=float(np.mean(trend_errors)) if trend_errors else None,
                mean_power_db=float(prediction[mask].mean()))
