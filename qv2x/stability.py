"""Prespecified-seed stochastic REM evaluation without ground-truth ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def summarize_samples(predictions, truth, valid, cdf_points: int = 256) -> dict:
    """Summarize every independent sample; no best-of-N or target selection.

    predictions=[S,H,W], truth/valid=[H,W], in dB. Standard deviations use
    ddof=1 for S>1 and are zero for a singleton. CDFs use only valid pixels
    and a common grid spanning all observed predicted/true values. RMSE is
    computed separately per sample; mean_map_rmse is separately named because
    averaging stochastic images before evaluation is a different estimator.
    """
    prediction = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if prediction.ndim != 3 or len(prediction) < 1 or prediction.shape[1:] != truth.shape or mask.shape != truth.shape:
        raise ValueError("Expected predictions [S,H,W] and matching truth/valid [H,W]")
    if not mask.any() or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Need finite maps and at least one valid pixel")
    if not isinstance(cdf_points, int) or cdf_points < 2:
        raise ValueError("cdf_points must be an integer >=2")
    values = prediction[:, mask]
    target = truth[mask]
    rmse = np.sqrt(np.mean((values - target[None]) ** 2, axis=1))
    lower, upper = min(float(values.min()), float(target.min())), max(float(values.max()), float(target.max()))
    # A repeated threshold still defines the exact CDF for constant fields.
    grid = np.linspace(lower, upper, cdf_points)
    cdfs = np.stack([np.searchsorted(np.sort(row), grid, side="right") / row.size for row in values])
    target_cdf = np.searchsorted(np.sort(target), grid, side="right") / target.size
    ddof = 1 if len(prediction) > 1 else 0
    mean = prediction.mean(axis=0)
    return {
        "mean_db": mean.astype(np.float32),
        "std_db": prediction.std(axis=0, ddof=ddof).astype(np.float32),
        "per_sample_rmse": rmse,
        "cdf_x": grid,
        "cdf_mean": cdfs.mean(axis=0),
        "cdf_std": cdfs.std(axis=0, ddof=ddof),
        "cdf_truth": target_cdf,
        "rmse_mean_db": float(rmse.mean()),
        "rmse_std_db": float(rmse.std(ddof=ddof)),
        "mean_map_rmse_db": float(np.sqrt(np.mean((mean[mask] - target) ** 2))),
    }


def _plot_stability(output: Path, statistics: dict, truth: np.ndarray, valid: np.ndarray, tx_xy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x, mean, std = (statistics[name] for name in ("cdf_x", "cdf_mean", "cdf_std"))
    ax.plot(x, statistics["cdf_truth"], color="#111827", linewidth=1.8, label="Ground truth")
    ax.plot(x, mean, color="#2563eb", linewidth=1.8, label="All samples: mean CDF")
    ax.fill_between(x, np.maximum(0, mean-std), np.minimum(1, mean+std),
                    color="#2563eb", alpha=.2, label="Across samples: ±1 SD")
    ax.set(xlabel="Received power (dB)", ylabel="Empirical CDF", ylim=(0, 1),
           title="Fixed transmitter: stochastic distribution stability")
    ax.grid(alpha=.2)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(output / "cdf_stability.png", dpi=180)
    plt.close(fig)

    h, w = truth.shape
    column = int(round(float(tx_xy[0]) * (w-1)))
    row = int(round(float(tx_xy[1]) * (h-1)))
    mean, std = statistics["mean_db"], statistics["std_db"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    slices = ((np.linspace(0, 1, w), truth[row, :], mean[row, :], std[row, :], valid[row, :],
               "Receiver x (normalized)", f"Horizontal profile, row {row}"),
              (np.linspace(0, 1, h), truth[:, column], mean[:, column], std[:, column], valid[:, column],
               "Receiver y (normalized)", f"Vertical profile, column {column}"))
    for ax, (coordinate, target, center, spread, mask, xlabel, title) in zip(axes, slices):
        target = np.where(mask, target, np.nan)
        center = np.where(mask, center, np.nan)
        spread = np.where(mask, spread, np.nan)
        ax.plot(coordinate, target, color="#111827", linewidth=1.6, label="Ground truth")
        ax.plot(coordinate, center, color="#2563eb", linewidth=1.6, label="All samples: mean")
        ax.fill_between(coordinate, center-spread, center+spread, color="#2563eb", alpha=.2, label="±1 SD")
        ax.set(xlabel=xlabel, ylabel="Received power (dB)", title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Profiles through the transmitter; invalid pixels excluded")
    fig.savefig(output / "profile_stability.png", dpi=180)
    plt.close(fig)


def evaluate_stability(run_dir, realization_id=None, samples: int = 100,
                       output_dir=None, seed: int = 7000) -> dict:
    """Generate independent maps for one held-out transmitter, frozen conditions.

    realization_id is the zero-based dataset index recorded in split.json,
    defaults to the first test index, and must belong to the test split.
    Every seed is retained in aggregate statistics. The checkpoint is already
    selected on validation data; neither samples nor seeds are ranked by truth.
    """
    from .data import load_dataset
    from .pipeline import (read_config, load_stage1, build_diffusion, generate_indices,
                           write_json, verify_conditions)

    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    root = Path(run_dir)
    config = read_config(root / "config.json")
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    test_ids = [int(value) for value in split["test"]]
    if not test_ids:
        raise ValueError("The saved test split is empty")
    index = test_ids[0] if realization_id is None else int(realization_id)
    if index not in test_ids:
        raise ValueError(f"realization_id={index} does not belong to the saved held-out test split")
    synthetic = config["data"]["mode"] == "synthetic"
    data_path = root / "dataset.npz" if synthetic else Path(config["data"]["path"])
    if not data_path.exists() and not data_path.is_absolute():
        data_path = Path(__file__).resolve().parents[1] / data_path
    dataset = load_dataset(data_path)
    if not 0 <= index < dataset.count:
        raise ValueError("The saved split does not match the dataset")
    signature = verify_conditions(config, root)
    digest = hashlib.sha256()
    for array in (dataset.maps_db, dataset.tx_xy, dataset.valid_mask, dataset.semantics, dataset.features):
        if array is not None:
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    if digest.hexdigest() != signature["data_sha256"]:
        raise ValueError("Loaded dataset differs from the condition cache's source data")
    with np.load(root / "conditions.npz", allow_pickle=False) as archive:
        conditions = {key: archive[key] for key in ("coarse_db", "prototype")}
    if len(conditions["coarse_db"]) != dataset.count or len(conditions["prototype"]) != dataset.count:
        raise ValueError("Condition cache and dataset have different realization counts")
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The saved run requests CUDA, which is unavailable")
    torch.set_num_threads(config.get("cpu_threads", 4))
    state = load_stage1(config, root, device)
    saved = torch.load(root / "stage2_best.pt", map_location=device, weights_only=False)
    if saved.get("conditions_signature") != signature:
        raise ValueError("Stage-II checkpoint and condition cache have different provenance")
    model, diffusion = build_diffusion(config, saved["semantic_channels"], saved["prototype_dim"], device)
    model.load_state_dict(saved["model"], strict=True)
    maps = []
    seeds = [int(seed) + i for i in range(samples)]
    for number, sample_seed in enumerate(seeds, start=1):
        prediction = generate_indices(config, dataset, conditions, [index], state.scaler,
                                      model, diffusion, device, sample_seed)[0]
        maps.append(prediction)
        if number == 1 or number % 10 == 0 or number == samples:
            print(f"Stability transmitter {index}: sample {number}/{samples}", flush=True)
    stats = summarize_samples(np.stack(maps), dataset.maps_db[index], dataset.valid_mask[index])
    destination = Path(output_dir) if output_dir is not None else root / f"stability_tx_{index}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "sample_stats.npz",
                        **{key: value for key, value in stats.items() if isinstance(value, np.ndarray)},
                        seeds=np.asarray(seeds), tx_xy=dataset.tx_xy[index],
                        truth_db=dataset.maps_db[index], valid_mask=dataset.valid_mask[index])
    _plot_stability(destination, stats, dataset.maps_db[index], dataset.valid_mask[index], dataset.tx_xy[index])
    sampler = "DDPM" if config["diffusion"].get("sample_steps") in (None, diffusion.timesteps) else "DDIM"
    result = {
        "realization_id": index, "samples": samples, "seeds": seeds,
        "rmse_mean_db": stats["rmse_mean_db"], "rmse_std_db": stats["rmse_std_db"],
        "mean_map_rmse_db": stats["mean_map_rmse_db"],
        "synthetic": synthetic, "sampler": sampler, "best_iteration": saved["iteration"],
        "standard_deviation_ddof": 1 if samples > 1 else 0,
        "condition_policy": "One cached frozen Stage-I condition for every sample; no target ranking",
        "conditions_signature": signature,
        "output_dir": str(destination.resolve()),
        "note": "Measured stochastic-run statistics, not original manuscript results.",
    }
    write_json(destination / "stability_summary.json", result)
    return result
