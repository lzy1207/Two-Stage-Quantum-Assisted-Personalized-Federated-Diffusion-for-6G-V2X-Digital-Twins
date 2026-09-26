"""Personalized federated Stage I and target-label-free grid conditioning.

Clients are simulated sequentially on one device. Saved private heads are local
simulation artifacts, not transmitted uplink content or a privacy guarantee.
The PDF (Eq. 25-42) is authoritative over the older supplied LaTeX draft.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .compression import compress_update
from .data import ClientData, TargetScaler


class PersonalizedHead(nn.Module):
    """Paper Table I: two affine layers, normally 32 -> 16 -> 1."""
    def __init__(self, input_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, values):
        return self.net(values)


@dataclass
class FederatedState:
    backbone: nn.Module
    heads: list[nn.Module]
    counts: list[int]
    rx_xy: list[torch.Tensor]
    validation_mse: list[float]
    scaler: TargetScaler
    history: list[dict[str, Any]]
    client_ids: list[int] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    error_buffers: list[torch.Tensor] = field(default_factory=list)
    torch_rng_state: torch.Tensor | None = None
    cuda_rng_states: list[torch.Tensor] = field(default_factory=list)


def weighted_average(updates: list[torch.Tensor], counts: list[int]) -> torch.Tensor:
    """Eq. 24: normalize sample counts over participating clients only."""
    if not updates or len(updates) != len(counts) or min(counts) <= 0:
        raise ValueError("Need updates and positive corresponding sample counts")
    total = sum(counts)
    out = torch.zeros_like(updates[0])
    for update, count in zip(updates, counts):
        if update.shape != out.shape:
            raise ValueError("Mismatched client update shapes")
        out.add_(update, alpha=count / total)
    return out


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_federated(state: FederatedState, path: str | Path, config=None) -> None:
    """Atomic checkpoint containing all state needed for faithful inference/resume.

    Only tensors and primitive Python values are saved, allowing weights_only
    loading. Local optimizers are intentionally restarted each federated round.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1, "backbone": _cpu_state(state.backbone),
        "heads": [_cpu_state(head) for head in state.heads],
        "head_input_dim": int(state.heads[0].input_dim),
        "head_hidden_dim": int(state.heads[0].hidden_dim),
        "counts": list(state.counts), "client_ids": list(state.client_ids),
        "rx_xy": [x.detach().cpu() for x in state.rx_xy],
        "validation_mse": list(state.validation_mse),
        "scaler": {"mean": state.scaler.mean, "std": state.scaler.std},
        "history": state.history, "config": dict(state.config),
        "run_config": dict(state.config if config is None else config),
        "error_buffers": [x.detach().cpu() for x in state.error_buffers],
        "torch_rng_state": state.torch_rng_state,
        "cuda_rng_states": state.cuda_rng_states,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_federated(path: str | Path, backbone: nn.Module, device="cpu") -> FederatedState:
    """Restore with a caller-constructed compatible backbone, without refitting."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Stage-I checkpoint format")
    backbone = backbone.to(device)
    backbone.load_state_dict(checkpoint["backbone"], strict=True)
    heads = []
    for state in checkpoint["heads"]:
        head = PersonalizedHead(checkpoint["head_input_dim"], checkpoint["head_hidden_dim"]).to(device)
        head.load_state_dict(state, strict=True)
        head.eval()
        heads.append(head)
    if not heads or len(heads) != len(checkpoint["counts"]):
        raise ValueError("Invalid checkpoint client state")
    backbone.eval()
    return FederatedState(backbone, heads, checkpoint["counts"], checkpoint["rx_xy"],
                          checkpoint["validation_mse"], TargetScaler(**checkpoint["scaler"]),
                          checkpoint["history"], checkpoint["client_ids"], checkpoint["config"],
                          checkpoint.get("error_buffers", []), checkpoint.get("torch_rng_state"),
                          checkpoint.get("cuda_rng_states", []))


@torch.no_grad()
def _validation(state: FederatedState, clients: list[ClientData], device, batch_size):
    state.backbone.eval()
    total_sse, total_sae, total_count = 0., 0., 0
    mses = []
    for client, head in zip(clients, state.heads):
        head.eval()
        sse, sae, count = 0., 0., 0
        for start in range(0, len(client.val_x), batch_size):
            x = client.val_x[start:start + batch_size].to(device)
            target = client.val_y[start:start + batch_size].to(device).reshape(-1, 1)
            prediction = state.scaler.decode(head(state.backbone(x)))
            difference = prediction - target
            sse += difference.double().square().sum().item()
            sae += difference.double().abs().sum().item()
            count += difference.numel()
        if count == 0:
            raise ValueError(f"Client {client.client_id} has no validation examples")
        mses.append(sse / count)
        total_sse, total_sae, total_count = total_sse + sse, total_sae + sae, total_count + count
    state.validation_mse = mses
    return math.sqrt(total_sse / total_count), total_sae / total_count


def _write_history(history, output):
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=False)
    if history:
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def train_federated(backbone: nn.Module, clients: list[ClientData], config: dict,
                    output_dir: str | Path, device="cpu") -> FederatedState:
    """Train shared QNN and local heads, logging actual measured payloads.

    Regression is in training-only standardized target units; reported metrics
    and reliability errors are in dB. ``rounds`` is the total target round count,
    including completed rounds when ``resume_from`` is supplied. No test labels
    are accepted. Validation realizes a separate transmitter-level split.
    """
    cfg = dict(config)
    rounds, epochs = int(cfg.get("rounds", 400)), int(cfg.get("local_epochs", 2))
    batch_size, lr = int(cfg.get("batch_size", 64)), float(cfg.get("lr", 1e-3))
    participation, ratio = float(cfg.get("participation", 1.)), float(cfg.get("topk_ratio", .5))
    bits, seed = int(cfg.get("quant_bits", 8)), int(cfg.get("seed", 0))
    personalized, use_ef = bool(cfg.get("personalized", True)), bool(cfg.get("error_feedback", False))
    if not clients or rounds < 1 or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Need nonempty clients, positive rounds/epochs/batch size/learning rate")
    if not 0 < participation <= 1 or not 0 < ratio <= 1 or bits not in (8, 32):
        raise ValueError("Invalid participation, topk_ratio, or quant_bits")
    for client in clients:
        if client.count < 1 or len(client.val_x) < 1:
            raise ValueError("Each client requires nonempty train and validation subsets")
        for values in (client.train_x, client.train_y, client.val_x, client.val_y, client.rx_xy):
            if not torch.isfinite(values).all():
                raise ValueError("Client data must be finite")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backbone = backbone.to(device)
    resume = cfg.get("resume_from")
    if resume:
        state = load_federated(resume, backbone, device)
        if state.client_ids != [c.client_id for c in clients] or state.counts != [c.count for c in clients]:
            raise ValueError("Resume clients must match checkpoint identities and counts")
        if bool(state.config.get("personalized", True)) != personalized:
            raise ValueError("Cannot change personalization during resume")
        for saved, client in zip(state.rx_xy, clients):
            if not torch.equal(saved.cpu(), client.rx_xy.cpu()):
                raise ValueError("Resume requires the same historical receiver samples")
        if state.torch_rng_state is not None:
            torch.set_rng_state(state.torch_rng_state)
        if device.type == "cuda" and state.cuda_rng_states:
            torch.cuda.set_rng_state_all(state.cuda_rng_states)
        state.config = cfg
    else:
        head = PersonalizedHead(int(backbone.hidden_dim), int(cfg.get("head_hidden_dim", 16))).to(device)
        state = FederatedState(backbone, [deepcopy(head) for _ in clients], [c.count for c in clients],
                               [c.rx_xy.detach().cpu().clone() for c in clients], [1.] * len(clients),
                               TargetScaler.fit(torch.cat([c.train_y.reshape(-1) for c in clients])), [],
                               [c.client_id for c in clients], cfg)
    n_shared = sum(p.numel() for p in backbone.parameters())
    n_upload = n_shared + (0 if personalized else sum(p.numel() for p in state.heads[0].parameters()))
    if not state.error_buffers:
        state.error_buffers = [torch.zeros(n_upload) for _ in clients]
    if any(buffer.numel() != n_upload for buffer in state.error_buffers):
        raise ValueError("Checkpoint error-feedback dimension mismatch")
    completed = int(state.history[-1]["round"]) if state.history else 0
    if rounds < completed:
        raise ValueError("Requested rounds precede the checkpoint")
    cumulative_paper = int(state.history[-1]["cumulative_paper_bits"]) if state.history else 0
    cumulative_wire = int(state.history[-1]["cumulative_wire_bits"]) if state.history else 0
    with (output / "federated_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    for round_number in range(completed + 1, rounds + 1):
        selected = np.random.default_rng(seed + round_number).choice(
            len(clients), max(1, int(math.ceil(participation * len(clients)))), replace=False)
        global_vector = parameters_to_vector(backbone.parameters()).detach().clone()
        if not personalized:
            global_vector = torch.cat((global_vector, parameters_to_vector(state.heads[0].parameters()).detach()))
        updates, selected_counts = [], []
        paper_bits, wire_bits = 0, 0
        train_loss_sum, train_batches = 0., 0
        for index in selected:
            index = int(index)
            client = clients[index]
            local = deepcopy(backbone).to(device)
            head = deepcopy(state.heads[index]).to(device)
            local.train()
            head.train()
            parameters = list(local.parameters()) + list(head.parameters())
            optimizer = torch.optim.Adam(parameters, lr=lr)
            generator = torch.Generator().manual_seed(seed + round_number * 100003 + client.client_id)
            for _ in range(epochs):
                order = torch.randperm(client.count, generator=generator)
                for start in range(0, client.count, batch_size):
                    ids = order[start:start + batch_size]
                    x = client.train_x[ids].to(device)
                    target = state.scaler.encode(client.train_y[ids].to(device).reshape(-1, 1))
                    prediction = head(local(x))
                    mse = (prediction - target).square().mean()
                    q_regularizer = local.regularization()
                    head_regularizer = sum(p.square().sum() for p in head.parameters())
                    loss = mse + float(cfg.get("lambda_q", 1e-4)) * q_regularizer
                    loss = loss + float(cfg.get("lambda_v", 1e-4)) * head_regularizer
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite local loss at round {round_number}, client {client.client_id}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if float(cfg.get("clip_grad", 1.)) > 0:
                        nn.utils.clip_grad_norm_(parameters, float(cfg.get("clip_grad", 1.)))
                    optimizer.step()
                    train_loss_sum += mse.detach().item()
                    train_batches += 1
            state.heads[index] = head.eval()
            local_vector = parameters_to_vector(local.parameters()).detach()
            if not personalized:
                local_vector = torch.cat((local_vector, parameters_to_vector(head.parameters()).detach()))
            delta = (local_vector - global_vector).cpu()
            compensated = delta + state.error_buffers[index] if use_ef else delta
            packet = compress_update(compensated, ratio, bits)
            decoded = packet.decompress(device)
            # Include both sparsification and quantization error in the residual.
            state.error_buffers[index] = (compensated - decoded.cpu()) if use_ef else torch.zeros_like(delta)
            updates.append(decoded)
            selected_counts.append(client.count)
            paper_bits += int(packet.paper_bits)
            wire_bits += int(packet.wire_bits)
        aggregate = global_vector + weighted_average(updates, selected_counts)
        vector_to_parameters(aggregate[:n_shared], backbone.parameters())
        if not personalized:
            for head in state.heads:
                vector_to_parameters(aggregate[n_shared:].clone(), head.parameters())
        rmse, mae = _validation(state, clients, device, batch_size)
        cumulative_paper += paper_bits
        cumulative_wire += wire_bits
        state.history.append({
            "round": round_number, "train_mse_normalized": train_loss_sum / max(train_batches, 1),
            "val_rmse_db": rmse, "val_mae_db": mae, "participating_clients": len(selected),
            "shared_parameters": n_shared, "uploaded_parameters": n_upload,
            "paper_bits": paper_bits, "wire_bits": wire_bits,
            "cumulative_paper_bits": cumulative_paper, "cumulative_wire_bits": cumulative_wire,
            "cumulative_uplink_mb": cumulative_paper / 8e6,
            "cumulative_wire_mb": cumulative_wire / 8e6,
        })
        state.torch_rng_state = torch.get_rng_state()
        state.cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        _write_history(state.history, output)
        save_federated(state, output / "latest.pt", cfg)
        print(f"[Stage I {round_number}/{rounds}] val RMSE={rmse:.4f} dB; "
              f"uplink={paper_bits / 8e6:.6f} MB; clients={len(selected)}", flush=True)
    backbone.eval()
    save_federated(state, output / "federated.pt", cfg)
    return state


@torch.no_grad()
def confidence_log_weights(rx_xy, query_xy, validation_mse, tau=.05, eps=1e-6, batch_size=1024):
    """PDF Eq. 25-26 in log space, using historical observations only."""
    if tau <= 0 or eps <= 0 or batch_size < 1:
        raise ValueError("tau, eps, and batch_size must be positive")
    if len(rx_xy) == 0 or validation_mse < 0 or not math.isfinite(validation_mse):
        raise ValueError("Need historical locations and a finite nonnegative validation MSE")
    historical = torch.as_tensor(rx_xy, dtype=query_xy.dtype, device=query_xy.device)
    chunks = []
    for start in range(0, len(query_xy), batch_size):
        squared = torch.cdist(query_xy[start:start + batch_size], historical).square().min(dim=1).values
        chunks.append(-squared / (2 * tau * tau) - math.log(validation_mse + eps))
    return torch.cat(chunks)


def prototype_from_moments(features: torch.Tensor, log_confidence: torch.Tensor,
                           counts: list[int]) -> torch.Tensor:
    """PDF Eq. 35-41: within-client moments plus total-variance correction.

    features=[pixels,d_q], log_confidence=[clients,pixels]. Historical validation
    reliability cancels within each client's spatial normalization, as in Eq.35.
    """
    if features.ndim != 2 or log_confidence.shape != (len(counts), len(features)):
        raise ValueError("Invalid quantum-feature or confidence dimensions")
    if not counts or min(counts) <= 0:
        raise ValueError("Prototype requires positive historical counts")
    alpha = torch.softmax(log_confidence, dim=1)
    means = alpha @ features
    variances = (alpha @ features.square() - means.square()).clamp_min(0)
    rho = torch.as_tensor(counts, dtype=features.dtype, device=features.device)
    rho /= rho.sum()
    mean = (rho[:, None] * means).sum(dim=0)
    variance = (rho[:, None] * (variances + (means - mean).square())).sum(dim=0)
    return torch.cat((mean, variance.clamp_min(0)))


@torch.no_grad()
def build_conditions(state: FederatedState, query_features, device="cpu", tau=.05,
                     eps=1e-6, batch_size=1024, confidence=True):
    """Frozen grid-query coarse map and quantum prototype without target labels.

    Input [H,W,10] contains the target Tx coordinate, Rx grid, and known
    environmental descriptors only. Historical Stage-I confidence works for an
    unseen transmitter. Returns coarse dB map [H,W], prototype [2*d_q].
    """
    query = torch.as_tensor(query_features, dtype=torch.float32, device=device)
    if query.ndim != 3 or query.shape[-1] != 10 or not torch.isfinite(query).all():
        raise ValueError("query_features must be finite [H,W,10]")
    if batch_size < 1 or not state.heads or len(state.heads) != len(state.counts):
        raise ValueError("Invalid batch size or federated client state")
    h, w = query.shape[:2]
    flat = query.reshape(-1, 10)
    backbone = state.backbone.to(device).eval()
    hidden_chunks, quantum_chunks = [], []
    for start in range(0, len(flat), batch_size):
        x = flat[start:start + batch_size]
        # Both supplied backbones expose this projection, so a single shot draw
        # supports prediction and prototype for an identical grid query.
        q = backbone.quantum_features(x)
        if hasattr(backbone, "output_projection"):
            hidden = torch.relu(backbone.output_projection(q))
        else:
            hidden = backbone(x)
        quantum_chunks.append(q)
        hidden_chunks.append(hidden)
    hidden, quantum = torch.cat(hidden_chunks), torch.cat(quantum_chunks)
    log_confidence, predictions = [], []
    for head, historical, reliability in zip(state.heads, state.rx_xy, state.validation_mse):
        head = head.to(device).eval()
        prediction = state.scaler.decode(head(hidden)).reshape(-1)
        predictions.append(prediction)
        log_confidence.append(confidence_log_weights(historical, flat[:, 2:4], reliability,
                                                     tau, eps, batch_size))
    log_confidence = torch.stack(log_confidence)
    weights = torch.softmax(log_confidence, dim=0) if confidence else torch.ones_like(log_confidence) / len(state.heads)
    coarse = (weights * torch.stack(predictions)).sum(dim=0).reshape(h, w)
    prototype = prototype_from_moments(quantum, log_confidence, state.counts)
    return coarse.cpu().numpy(), prototype.cpu().numpy()
