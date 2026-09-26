"""Sequential train/condition/refine/evaluate workflow with explicit artifacts."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import (MapDataset, load_dataset, save_dataset, make_synthetic_dataset,
                   split_realizations, get_features, partition_clients)
from .quantum import make_backbone
from .federated import train_federated, build_conditions, save_federated, load_federated
from .diffusion import ConditionalUNet, GaussianDiffusion, make_spatial_condition
from .metrics import evaluate_maps


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    with open(path, encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema_version")
    return config


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_signature(config, output):
    data = json.loads((output/"data_manifest.json").read_text(encoding="utf-8"))
    return {"stage1_sha256": file_digest(output/"stage1.pt"), "data_sha256": data["sha256"],
            "split_sha256": file_digest(output/"split.json"), "quantum": config["quantum"],
            "tau": config["federated"].get("confidence_tau", .05),
            "eps": config["federated"].get("confidence_eps", 1e-6),
            "confidence": config.get("ablation", {}).get("confidence", True)}


def verify_conditions(config, output):
    path = output/"conditions_manifest.json"
    if not path.exists():
        raise ValueError("Missing conditions provenance; run the conditions stage")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["signature"] != condition_signature(config, output) or manifest["cache_sha256"] != file_digest(output/"conditions.npz"):
        raise ValueError("Condition cache is stale or modified; rerun conditions, then retrain Stage II")
    return manifest["signature"]


def setup(config, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set device to cpu or auto")
    return output, device


def record_run(config, output, device):
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {"torch": str(torch.__version__), "numpy": np.__version__,
               "device": device, "seed": config["seed"], "synthetic": config["data"]["mode"] == "synthetic",
               "status": "reconstructed implementation; results are not original manuscript measurements"})


def prepare_data(config, output):
    dc = config["data"]
    if dc["mode"] == "synthetic":
        dataset = make_synthetic_dataset(count=dc.get("count", 12), size=dc.get("size", 16), seed=dc.get("seed", config.get("split_seed", 2025)))
    elif dc["mode"] == "npz":
        dataset = load_dataset(dc["path"])
    else:
        raise ValueError("data.mode must be synthetic or npz")
    split = split_realizations(dataset.count, seed=config.get("split_seed", 2025), ratios=(.7, .1, .2))
    fingerprint = hashlib.sha256()
    for array in (dataset.maps_db, dataset.tx_xy, dataset.valid_mask, dataset.semantics, dataset.features):
        if array is not None:
            fingerprint.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    manifest = {"sha256": fingerprint.hexdigest(), "count": dataset.count,
               "height": dataset.height, "width": dataset.width, "metadata": dataset.metadata,
               "partition_level": "transmitter-conditioned realization", "split_seed": config.get("split_seed", 2025)}
    manifest_path = output/"data_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous["sha256"], previous["split_seed"]) != (manifest["sha256"], manifest["split_seed"]):
            raise ValueError("Existing run has different data/split; choose a new output directory")
    write_json(manifest_path, manifest)
    write_json(output/"split.json", {key: np.asarray(value).tolist() for key, value in split.items()})
    if dc["mode"] == "synthetic":
        save_dataset(dataset, output/"dataset.npz")
    return dataset, split


def stage1(config, output, dataset, split, device):
    seed_all(config["seed"])
    dc = config["data"]
    clients = partition_clients(dataset, split["train"], split["val"],
            grid=tuple(dc.get("client_grid", [10, 9])),
            samples_per_client=dc.get("samples_per_client", 128),
            neighbor_ratio=dc.get("neighbor_ratio", .1), seed=config["seed"])
    backbone = make_backbone(config["quantum"])
    print(f"Stage I: {len(clients)} clients; {sum(p.numel() for p in backbone.parameters())} shared parameters", flush=True)
    state = train_federated(backbone, clients, {**config["federated"], "seed": config["seed"]}, output, device)
    save_federated(state, output/"stage1.pt", config)
    return state


def load_stage1(config, output, device):
    return load_federated(output/"stage1.pt", make_backbone(config["quantum"]), device)


def construct_conditions(config, output, dataset, state, device):
    """Only geometry, semantics and frozen model reach build_conditions()."""
    coarse, prototypes = [], []
    for index in range(dataset.count):
        features = get_features(dataset, [index])[0]
        x, p = build_conditions(state, features, device=device,
              tau=config["federated"].get("confidence_tau", .05),
              eps=config["federated"].get("confidence_eps", 1e-6),
              batch_size=config.get("query_batch_size", 1024),
              confidence=config.get("ablation", {}).get("confidence", True))
        coarse.append(x)
        prototypes.append(p)
        print(f"Conditions: {index+1}/{dataset.count}", flush=True)
    result = {"coarse_db": np.stack(coarse), "prototype": np.stack(prototypes)}
    np.savez_compressed(output/"conditions.npz", **result)
    write_json(output/"conditions_manifest.json", {"signature": condition_signature(config, output),
               "cache_sha256": file_digest(output/"conditions.npz")})
    return result


def build_diffusion(config, semantic_channels, prototype_dim, device):
    dc = config["diffusion"]
    model = ConditionalUNet(semantic_channels, prototype_dim,
             base_channels=dc.get("base_channels", 64),
             channel_mults=tuple(dc.get("channel_mults", [1, 2, 4, 4])),
             embedding_dim=dc.get("embedding_dim", 64)).to(device)
    diffusion = GaussianDiffusion(dc.get("timesteps", 1000), dc.get("beta_start", 1e-4), dc.get("beta_end", .02))
    return model, diffusion


def condition_batch(config, dataset, conditions, indices, scaler, device):
    """Build model inputs without reading maps_db or valid_mask values."""
    ids = np.asarray(indices, dtype=int)
    coarse = torch.as_tensor(scaler.encode(conditions["coarse_db"][ids]), device=device).float().unsqueeze(1)
    tx = torch.as_tensor(dataset.tx_xy[ids], device=device).float()
    semantics = None if dataset.semantics is None else torch.as_tensor(dataset.semantics[ids], device=device).float()
    proto = torch.as_tensor(conditions["prototype"][ids], device=device).float()
    spatial = make_spatial_condition(coarse, tx, semantics, sigma=config["diffusion"].get("sigma_pos", .04))
    ablation = config.get("ablation", {})
    if not ablation.get("coarse", True):
        spatial[:, 0] = 0
    if not ablation.get("heatmap", True):
        spatial[:, 1] = 0
    if not ablation.get("prototype", True):
        proto = torch.zeros_like(proto)
    if not ablation.get("semantics", True):
        spatial[:, 2:] = 0
    return spatial, proto


def make_batch(config, dataset, conditions, indices, scaler, device):
    spatial, proto = condition_batch(config, dataset, conditions, indices, scaler, device)
    ids = np.asarray(indices, dtype=int)
    x0 = torch.as_tensor(scaler.encode(dataset.maps_db[ids]), device=device).float().unsqueeze(1)
    mask = torch.as_tensor(dataset.valid_mask[ids], device=device).bool().unsqueeze(1)
    return x0, spatial, proto, mask


@torch.no_grad()
def generate_indices(config, dataset, conditions, indices, scaler, model, diffusion, device, seed):
    model.eval()
    predictions = []
    # No target-based rejection, ranking, or best-of-N selection.
    generator = torch.Generator(device=device).manual_seed(seed)
    for index in indices:
        spatial, proto = condition_batch(config, dataset, conditions, [index], scaler, device)
        prediction = diffusion.sample(model, spatial, proto,
                     steps=config["diffusion"].get("sample_steps"), generator=generator)
        predictions.append(scaler.decode(prediction[:, 0].cpu().numpy())[0])
    return np.stack(predictions)


def metrics_for(dataset, ids, pred, data_range, radial_bins, normalization_min=0., normalization_range=1.):
    result = evaluate_maps(pred, dataset.maps_db[ids], dataset.valid_mask[ids], dataset.tx_xy[ids],
                         data_range=data_range, radial_bins=radial_bins)
    result["mean_power"] = float(((pred-normalization_min)/normalization_range)[dataset.valid_mask[ids]].mean())
    return result


def evaluation_scale(config, dataset, split):
    values = dataset.maps_db[split["train"]][dataset.valid_mask[split["train"]]]
    minimum, spread = float(values.min()), max(1e-6, float(np.ptp(values)))
    data_range = float(config.get("evaluation", {}).get("data_range_db", spread))
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("evaluation.data_range_db must be finite and positive")
    return minimum, spread, data_range


def stage2(config, output, dataset, split, conditions, state, device, resume=False):
    seed_all(config["seed"]+1000)
    dc = config["diffusion"]
    signature = verify_conditions(config, output)
    semantic_channels = 0 if dataset.semantics is None else dataset.semantics.shape[1]
    model, diffusion = build_diffusion(config, semantic_channels, conditions["prototype"].shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=dc.get("lr", 2e-4), weight_decay=dc.get("weight_decay", 0.0))
    start, best = 0, float("inf")
    rng = np.random.default_rng(config["seed"]+1000)
    history = []
    minimum, spread, data_range = evaluation_scale(config, dataset, split)
    radial_bins = config.get("evaluation", {}).get("radial_bins", 16)
    if resume:
        saved = torch.load(output/"stage2_last.pt", map_location=device, weights_only=False)
        if saved.get("conditions_signature") != signature:
            raise ValueError("Cannot resume Stage II with changed Stage-I model or conditions")
        if dc["iterations"] < saved["iteration"]:
            raise ValueError("Requested iterations precede the saved checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, history = saved["iteration"], saved["best_rmse"], saved["history"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    weights = tuple(dc.get("loss_weights", [1, .1, .1]))
    if not config.get("ablation", {}).get("structural_losses", True):
        weights = (0., 0., 0.)
    for iteration in range(start+1, dc["iterations"]+1):
        model.train()
        ids = rng.choice(split["train"], size=dc.get("batch_size", 16), replace=True)
        x0, spatial, proto, mask = make_batch(config, dataset, conditions, ids, state.scaler, device)
        optimizer.zero_grad(set_to_none=True)
        losses = diffusion.training_loss(model, x0, spatial, proto, weights=weights,
                   mask=mask if dc.get("mask_training_loss", False) else None,
                   data_range=data_range/state.scaler.std)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Nonfinite diffusion loss at iteration {iteration}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), dc.get("grad_clip", 1.))
        optimizer.step()
        row = {"iteration": iteration, **{key: float(value.detach()) for key, value in losses.items()}}
        history.append(row)
        if iteration == 1 or iteration % dc.get("log_every", 100) == 0:
            print(f"Stage II {iteration}/{dc['iterations']}: loss={row['loss']:.6f}", flush=True)
        if iteration % dc.get("validate_every", 1000) == 0 or iteration == dc["iterations"]:
            pred = generate_indices(config, dataset, conditions, split["val"], state.scaler, model, diffusion, device, config["seed"]+2000)
            metric = metrics_for(dataset, split["val"], pred, data_range, radial_bins, minimum, spread)
            row["val_rmse"] = metric["rmse"]
            improved = metric["rmse"] < best
            best = min(best, metric["rmse"])
            saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration,
                     "best_rmse": best, "history": history, "config": config, "scaler": vars(state.scaler),
                     "conditions_signature": signature,
                     "semantic_channels": semantic_channels, "prototype_dim": conditions["prototype"].shape[1],
                     "evaluation_scale": {"minimum_db": minimum, "range_db": spread, "ssim_range_db": data_range},
                     "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
            torch.save(saved, output/"stage2_last.pt")
            if improved:
                torch.save(saved, output/"stage2_best.pt")
            write_json(output/"diffusion_history.json", history)
            print(f"Validation RMSE={metric['rmse']:.4f} dB; best={best:.4f}", flush=True)
    return model, diffusion


def evaluate(config, output, dataset, split, conditions, state, device):
    saved = torch.load(output/"stage2_best.pt", map_location=device, weights_only=False)
    if saved.get("conditions_signature") != verify_conditions(config, output):
        raise ValueError("Stage-II checkpoint was trained on different Stage-I conditions")
    model, diffusion = build_diffusion(config, saved["semantic_channels"], saved["prototype_dim"], device)
    model.load_state_dict(saved["model"])
    ids = split["test"]
    pred = generate_indices(config, dataset, conditions, ids, state.scaler, model, diffusion, device, config["seed"]+3000)
    minimum, spread, data_range = evaluation_scale(config, dataset, split)
    radial_bins = config.get("evaluation", {}).get("radial_bins", 16)
    result = {"stage1": metrics_for(dataset, ids, conditions["coarse_db"][ids], data_range, radial_bins, minimum, spread),
              "stage2": metrics_for(dataset, ids, pred, data_range, radial_bins, minimum, spread),
              "test_realization_ids": np.asarray(ids).tolist(), "data_range_db": data_range,
              "synthetic": config["data"]["mode"] == "synthetic", "best_iteration": saved["iteration"],
              "sampler": "DDPM" if config["diffusion"].get("sample_steps") in (None, config["diffusion"]["timesteps"]) else "DDIM",
              "power_normalization": {"minimum_db": minimum, "range_db": spread, "clip": False},
              "note": "Measured outputs of this run, not original paper values."}
    write_json(output/"metrics.json", result)
    np.savez_compressed(output/"predictions.npz", realization_ids=ids, prediction_db=pred,
                        coarse_db=conditions["coarse_db"][ids], truth_db=dataset.maps_db[ids],
                        valid_mask=dataset.valid_mask[ids], tx_xy=dataset.tx_xy[ids])
    plot_results(output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    from .plots import plot_federated_history
    plot_federated_history(output)
    with np.load(output/"predictions.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    truth, coarse, pred = [arrays[k][0] for k in ("truth_db", "coarse_db", "prediction_db")]
    vmin, vmax = float(truth.min()), float(truth.max())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, title, data in zip(axes, ["Ground truth", "Stage I", "Stage II", "Absolute error"], [truth, coarse, pred, np.abs(pred-truth)]):
        im = ax.imshow(data, cmap="viridis" if title != "Absolute error" else "magma", vmin=vmin if title != "Absolute error" else 0, vmax=vmax if title != "Absolute error" else None)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=.75, label="dB")
    fig.suptitle("Reconstruction run - see metrics.json for dataset and settings")
    fig.savefig(output/"rem_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for key, name in [("truth_db", "Ground truth"), ("coarse_db", "Stage I"), ("prediction_db", "Stage II")]:
        values = np.sort(arrays[key][arrays["valid_mask"].astype(bool)])
        ax.plot(values, np.arange(1, len(values)+1)/len(values), label=name)
    ax.set(xlabel="Received power (dB)", ylabel="Empirical CDF")
    ax.legend()
    fig.savefig(output/"cdf.png", dpi=160)
    plt.close(fig)
    if (output/"diffusion_history.json").exists():
        history = json.loads((output/"diffusion_history.json").read_text())
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot([r["iteration"] for r in history], [r["loss"] for r in history])
        ax.set(xlabel="Iteration", ylabel="Training loss", yscale="log")
        fig.savefig(output/"diffusion_loss.png", dpi=160)
        plt.close(fig)


def run(config, output_dir, stage="all", resume=False):
    previous_path = Path(output_dir)/"config.json"
    if previous_path.exists() and stage not in ("all", "stage1"):
        previous = read_config(previous_path)
        for key in ("quantum", "data", "seed", "split_seed", "ablation"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Cannot change {key} while reusing existing stage artifacts")
        if previous["federated"] != config["federated"]:
            raise ValueError("Cannot change federated settings while reusing stage artifacts")
        allowed_changes = {"iterations", "validate_every", "log_every"}
        for key in set(previous["diffusion"]) | set(config["diffusion"]):
            if key not in allowed_changes and previous["diffusion"].get(key) != config["diffusion"].get(key):
                raise ValueError(f"Cannot change diffusion.{key} while reusing artifacts")
    output, device = setup(config, output_dir)
    dataset, split = prepare_data(config, output)
    record_run(config, output, device)
    if stage in ("all", "stage1"):
        state = stage1(config, output, dataset, split, device)
        if stage == "stage1":
            return
    else:
        state = load_stage1(config, output, device)
    if stage in ("all", "conditions"):
        conditions = construct_conditions(config, output, dataset, state, device)
        if stage == "conditions":
            return
    else:
        verify_conditions(config, output)
        with np.load(output/"conditions.npz", allow_pickle=False) as archive:
            conditions = {k: archive[k] for k in archive.files}
    if stage in ("all", "stage2"):
        stage2(config, output, dataset, split, conditions, state, device, resume=resume)
        if stage == "stage2":
            return
    return evaluate(config, output, dataset, split, conditions, state, device)


def infer(run_dir, query_path, destination, seed=42):
    """Query NPZ contains tx_xy, features and/or semantics; no radio targets."""
    output = Path(run_dir)
    config = read_config(output/"config.json")
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(config.get("cpu_threads", 4))
    seed_all(seed)
    with np.load(query_path, allow_pickle=False) as q:
        tx = q["tx_xy"]
        features = q["features"] if "features" in q else None
        semantics = q["semantics"] if "semantics" in q else None
        shape = features.shape[-2:] if features is not None else semantics.shape[-2:] if semantics is not None else tuple(q["grid_shape"])
        # Placeholder is solely a shape carrier; never supplied to build_conditions.
        dataset = MapDataset(np.zeros((len(tx), *shape), dtype=np.float32), tx, semantics=semantics, features=features)
    state = load_stage1(config, output, device)
    coarse, prototype = [], []
    for i in range(dataset.count):
        x, p = build_conditions(state, get_features(dataset, [i])[0], device=device,
               tau=config["federated"].get("confidence_tau", .05), eps=config["federated"].get("confidence_eps", 1e-6),
               batch_size=config.get("query_batch_size", 1024), confidence=config.get("ablation", {}).get("confidence", True))
        coarse.append(x)
        prototype.append(p)
    conditions = {"coarse_db": np.stack(coarse), "prototype": np.stack(prototype)}
    saved = torch.load(output/"stage2_best.pt", map_location=device, weights_only=False)
    if saved.get("conditions_signature", {}).get("stage1_sha256") != file_digest(output/"stage1.pt"):
        raise ValueError("Stage-II checkpoint does not match current Stage-I checkpoint")
    model, diffusion = build_diffusion(config, saved["semantic_channels"], saved["prototype_dim"], device)
    actual_channels = 0 if semantics is None else semantics.shape[1]
    if actual_channels != saved["semantic_channels"]:
        raise ValueError("Query semantic channels must match trained model")
    model.load_state_dict(saved["model"])
    pred = generate_indices(config, dataset, conditions, range(dataset.count), state.scaler, model, diffusion, device, seed)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, prediction_db=pred, tx_xy=tx, **conditions)


def ablation_config(config, variant):
    result = copy.deepcopy(config)
    result.setdefault("ablation", {})
    if variant == "full":
        pass
    elif variant == "classical":
        result["quantum"]["kind"] = "classical"
    elif variant == "no_entanglement":
        result["quantum"]["entanglement"] = False
    elif variant == "no_personalization":
        result["federated"]["personalized"] = False
    elif variant == "fedavg_classical":
        result["quantum"]["kind"] = "classical"
        result["federated"].update(personalized=False, topk_ratio=1., quant_bits=32)
    elif variant in ("no_confidence", "no_coarse", "no_heatmap", "no_prototype", "no_structural_losses"):
        result["ablation"][variant[3:]] = False
    elif variant == "no_compression":
        result["federated"].update(topk_ratio=1., quant_bits=32)
    elif variant == "quantization_only":
        result["federated"].update(topk_ratio=1., quant_bits=8)
    elif variant == "topk_only":
        result["federated"].update(topk_ratio=.5, quant_bits=32)
    elif variant == "coordinate_ddpm":
        result["ablation"].update(coarse=False, prototype=False, semantics=False)
    else:
        raise ValueError(f"Unknown variant {variant}")
    return result


def run_sweep(config, output_dir, variants, seeds):
    rows = []
    for variant in variants:
        for seed in seeds:
            cfg = ablation_config(config, variant)
            cfg["seed"] = seed
            metrics = run(cfg, Path(output_dir)/variant/f"seed_{seed}")
            for stage_name in ("stage1", "stage2"):
                rows.append({"variant": variant, "seed": seed, "stage": stage_name, **metrics[stage_name]})
    output = Path(output_dir)
    with (output/"runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for variant in variants:
        for stage_name in ("stage1", "stage2"):
            subset = [r for r in rows if r["variant"] == variant and r["stage"] == stage_name]
            entry = {"variant": variant, "stage": stage_name, "runs": len(subset)}
            for metric in ("rmse", "ssim", "radial_mae", "trend_violation"):
                values = [r[metric] for r in subset if r[metric] is not None]
                entry[metric+"_mean"] = float(np.mean(values)) if values else None
                entry[metric+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0. if values else None
            summary.append(entry)
    write_json(output/"summary.json", summary)
