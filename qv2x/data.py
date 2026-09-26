"""Numeric radio-map I/O, leakage-free features, and spatial client partitions.

Coordinates are normalized (x, y), with x along columns and y along rows.
The six environmental channels are not enumerated in the paper. Supplied
``features`` take precedence; the documented fallback below uses only known
geometry and semantic maps, never received-power targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class MapDataset:
    maps_db: np.ndarray
    tx_xy: np.ndarray
    semantics: np.ndarray | None = None
    features: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.maps_db = np.asarray(self.maps_db, dtype=np.float32)
        self.tx_xy = np.asarray(self.tx_xy, dtype=np.float32)
        if self.maps_db.ndim != 3 or min(self.maps_db.shape) < 1:
            raise ValueError("maps_db must be nonempty [M,H,W]")
        m, h, w = self.maps_db.shape
        if self.tx_xy.shape != (m, 2) or not np.isfinite(self.tx_xy).all():
            raise ValueError("tx_xy must be finite [M,2]")
        if np.any(self.tx_xy < 0) or np.any(self.tx_xy > 1):
            raise ValueError("tx_xy coordinates must be normalized into [0,1]")
        if self.valid_mask is None:
            self.valid_mask = np.isfinite(self.maps_db)
        else:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.valid_mask.shape != (m, h, w):
            raise ValueError("valid_mask must have shape [M,H,W]")
        if not self.valid_mask.reshape(m, -1).any(axis=1).all():
            raise ValueError("Every realization needs at least one valid pixel")
        if not np.isfinite(self.maps_db[self.valid_mask]).all():
            raise ValueError("Valid target pixels must be finite")
        # Invalid pixels remain masked, but finite storage also makes diffusion safe.
        self.maps_db = np.where(np.isfinite(self.maps_db), self.maps_db, 0).astype(np.float32)
        if self.semantics is not None:
            self.semantics = np.asarray(self.semantics, dtype=np.float32)
            if self.semantics.ndim != 4 or self.semantics.shape[0] != m or self.semantics.shape[2:] != (h, w):
                raise ValueError("semantics must have shape [M,C,H,W]")
            if not np.isfinite(self.semantics).all():
                raise ValueError("semantics must be finite")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float32)
            if self.features.shape != (m, 6, h, w) or not np.isfinite(self.features).all():
                raise ValueError("features must be finite [M,6,H,W]")

    @property
    def count(self) -> int:
        return int(self.maps_db.shape[0])

    @property
    def height(self) -> int:
        return int(self.maps_db.shape[1])

    @property
    def width(self) -> int:
        return int(self.maps_db.shape[2])

    def subset(self, indices: np.ndarray | list[int]) -> "MapDataset":
        ids = np.asarray(indices, dtype=np.int64)
        return MapDataset(
            self.maps_db[ids], self.tx_xy[ids],
            None if self.semantics is None else self.semantics[ids],
            None if self.features is None else self.features[ids],
            self.valid_mask[ids], dict(self.metadata),
        )


def save_dataset(dataset: MapDataset, path: str | Path) -> None:
    """Write only numeric arrays and UTF-8 JSON metadata; no pickles."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "maps_db": dataset.maps_db,
        "tx_xy": dataset.tx_xy,
        "valid_mask": dataset.valid_mask,
        "metadata_json": np.asarray(json.dumps(dataset.metadata, ensure_ascii=False)),
    }
    if dataset.semantics is not None:
        arrays["semantics"] = dataset.semantics
    if dataset.features is not None:
        arrays["features"] = dataset.features
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_dataset(path: str | Path) -> MapDataset:
    """Load a canonical NPZ, rejecting pickle/object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        if "maps_db" not in archive or "tx_xy" not in archive:
            raise ValueError("Dataset requires maps_db and tx_xy arrays")
        optional = {key: archive[key] if key in archive else None
                    for key in ("semantics", "features", "valid_mask")}
        metadata = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
        return MapDataset(archive["maps_db"], archive["tx_xy"], metadata=metadata, **optional)


def coordinate_grid(height: int, width: int) -> np.ndarray:
    """Return [H,W,2] receiver coordinates with x before y."""
    yy, xx = np.meshgrid(np.linspace(0, 1, height, dtype=np.float32),
                         np.linspace(0, 1, width, dtype=np.float32), indexing="ij")
    return np.stack((xx, yy), axis=-1)


def get_features(dataset: MapDataset, indices=None) -> np.ndarray:
    """Return [M,H,W,10] = [tx_x,tx_y,rx_x,rx_y,six descriptors].

    Fallback descriptors: semantic channel 0 (or 0), normalized Tx distance,
    |Tx-Rx x|, |Tx-Rx y|, sin(pi*Rx x), sin(pi*Rx y). These are explicit
    reconstruction assumptions, not claimed recovered original features.
    This function never reads maps_db values.
    """
    ids = np.arange(dataset.count) if indices is None else np.atleast_1d(indices).astype(np.int64)
    n, h, w = len(ids), dataset.height, dataset.width
    rx = np.broadcast_to(coordinate_grid(h, w), (n, h, w, 2))
    tx = np.broadcast_to(dataset.tx_xy[ids, None, None, :], (n, h, w, 2))
    if dataset.features is not None:
        environmental = dataset.features[ids].transpose(0, 2, 3, 1)
    else:
        occupancy = np.zeros((n, h, w), dtype=np.float32)
        if dataset.semantics is not None and dataset.semantics.shape[1] > 0:
            occupancy = dataset.semantics[ids, 0]
        d = np.abs(tx - rx)
        environmental = np.stack((occupancy, np.linalg.norm(d, axis=-1) / np.sqrt(2),
                                  d[..., 0], d[..., 1], np.sin(np.pi * rx[..., 0]),
                                  np.sin(np.pi * rx[..., 1])), axis=-1)
    return np.concatenate((tx, rx, environmental), axis=-1).astype(np.float32)


def split_realizations(count: int, seed: int = 0, ratios=(.7, .1, .2)) -> dict[str, np.ndarray]:
    """Disjoint realization-level splits, each nonempty for count >= 3.

    The paper splits by transmitter-conditioned realization. Users wanting
    stronger unseen-scene evaluation must supply external scene-disjoint IDs.
    """
    ratios = np.asarray(ratios, dtype=float)
    if count < 3 or ratios.shape != (3,) or (ratios <= 0).any() or not np.isfinite(ratios).all():
        raise ValueError("Need at least 3 realizations and three positive ratios")
    ratios /= ratios.sum()
    sizes = np.maximum(1, np.floor(count * ratios).astype(int))
    while sizes.sum() > count:
        eligible = np.where(sizes > 1, sizes - count * ratios, -np.inf)
        sizes[int(np.argmax(eligible))] -= 1
    while sizes.sum() < count:
        sizes[int(np.argmax(count * ratios - sizes))] += 1
    ids = np.random.default_rng(seed).permutation(count)
    return {key: value for key, value in zip(("train", "val", "test"),
            np.split(ids, np.cumsum(sizes)[:-1]))}


def make_synthetic_dataset(count: int = 12, size: int = 16, seed: int = 0) -> MapDataset:
    """Seeded synthetic urban-style maps for software verification, not paper scores.

    Independent scenes have rectangular obstacles, log-distance attenuation,
    obstacle-related shadowing, smooth variation, and small measurement noise.
    The generator is intentionally lightweight and is not a ray tracer.
    """
    if count < 1 or size < 4:
        raise ValueError("Synthetic data need count >= 1 and size >= 4")
    rng = np.random.default_rng(seed)
    xy = coordinate_grid(size, size)
    txs = rng.uniform(.08, .92, (count, 2)).astype(np.float32)
    maps, semantics = [], []
    for tx in txs:
        buildings = np.zeros((size, size), dtype=np.float32)
        for _ in range(max(2, size // 5)):
            cx, cy = rng.uniform(.1, .9, 2)
            sx, sy = rng.uniform(.04, .16, 2)
            buildings[(np.abs(xy[..., 0] - cx) < sx) & (np.abs(xy[..., 1] - cy) < sy)] = 1
        distance_m = np.linalg.norm(xy - tx, axis=-1) * 250 + 3
        # Ray occupancy is a generator-internal physical proxy, never a feature target.
        shadow = np.zeros((size, size), dtype=np.float32)
        for fraction in np.linspace(.05, .95, 16):
            ray = tx + fraction * (xy - tx)
            xx = np.clip(np.rint(ray[..., 0] * (size - 1)).astype(int), 0, size - 1)
            yy = np.clip(np.rint(ray[..., 1] * (size - 1)).astype(int), 0, size - 1)
            shadow += buildings[yy, xx] / 16
        smooth = 1.5 * np.sin(2 * np.pi * xy[..., 0] + rng.uniform(0, 6)) * np.cos(
            2 * np.pi * xy[..., 1] + rng.uniform(0, 6))
        signal = -35 - 22 * np.log10(distance_m) - 16 * shadow - 6 * buildings + smooth
        signal += rng.normal(0, .35, signal.shape)
        maps.append(np.clip(signal, -127, -35))
        semantics.append(buildings[None])
    return MapDataset(np.asarray(maps), txs, np.asarray(semantics), metadata={
        "source": "synthetic_demo_not_RadioMapSeer", "seed": seed,
        "coordinate_system": "normalized_xy", "signal_unit": "dB",
        "scene_ids": list(range(count)), "generator": "log_distance_obstacle_proxy_v1",
    })


@dataclass
class ClientData:
    client_id: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    rx_xy: torch.Tensor

    @property
    def count(self) -> int:
        return int(len(self.train_x))


def partition_clients(dataset: MapDataset, train_ids, val_ids, grid=(10, 9),
                      samples_per_client: int = 128, neighbor_ratio: float = .1,
                      seed: int = 0) -> list[ClientData]:
    """Sample disjoint spatial client pools within disjoint realization splits.

    A measurement first belongs to its receiver's grid cell; ``neighbor_ratio``
    of assignments migrate to a random adjacent cell. Each client then samples
    without replacement, up to samples_per_client training records. A uniform
    random-priority reservoir processes one realization at a time, so extra
    feature memory is O(H*W*10 + clients*samples_per_client*10), not O(M*H*W*10).
    Empty cells are omitted explicitly. No synthetic duplicates fill sparse
    clients. Loading the canonical NPZ itself still loads its map arrays.
    """
    nx, ny = map(int, grid)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)
    if nx < 1 or ny < 1 or samples_per_client < 1 or not 0 <= neighbor_ratio <= 1:
        raise ValueError("Invalid grid, samples_per_client, or neighbor_ratio")
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Both training and validation realization sets must be nonempty")
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("Training and validation realizations must be disjoint")
    for ids in (train_ids, val_ids):
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= dataset.count):
            raise ValueError("Invalid or duplicate realization indices")
    rng = np.random.default_rng(seed)

    def sampled_records(ids, limit):
        reservoirs = {cid: (np.empty((0, 10), np.float32),
                            np.empty((0, 1), np.float32), np.empty(0))
                      for cid in range(nx * ny)}
        for realization_id in ids:
            # One realization avoids allocating the complete M*H*W*10 cube.
            features = get_features(dataset, [realization_id])[0].reshape(-1, 10)
            targets = dataset.maps_db[realization_id].reshape(-1, 1)
            mask = dataset.valid_mask[realization_id].reshape(-1)
            features, targets = features[mask], targets[mask]
            cells_x = np.minimum((features[:, 2] * nx).astype(int), nx - 1)
            cells_y = np.minimum((features[:, 3] * ny).astype(int), ny - 1)
            assignments = cells_y * nx + cells_x
            move = np.flatnonzero(rng.random(len(assignments)) < neighbor_ratio)
            if len(move) and nx * ny > 1:
                cx, cy, owner = cells_x[move], cells_y[move], assignments[move]
                candidates = np.stack((np.where(cx > 0, owner - 1, -1),
                                       np.where(cx < nx - 1, owner + 1, -1),
                                       np.where(cy > 0, owner - nx, -1),
                                       np.where(cy < ny - 1, owner + nx, -1)), axis=1)
                valid_count = (candidates >= 0).sum(axis=1)
                # Sort valid IDs before sentinel values; sample each neighbor equally.
                candidates = np.sort(np.where(candidates < 0, nx * ny, candidates), axis=1)
                choice = (rng.random(len(move)) * valid_count).astype(int)
                assignments[move] = candidates[np.arange(len(move)), choice]
            priorities = rng.random(len(features))
            for cid in range(nx * ny):
                selected = np.flatnonzero(assignments == cid)
                if len(selected) > limit:
                    selected = selected[np.argpartition(priorities[selected], -limit)[-limit:]]
                previous_x, previous_y, previous_priority = reservoirs[cid]
                merged_x = np.concatenate((previous_x, features[selected]))
                merged_y = np.concatenate((previous_y, targets[selected]))
                merged_priority = np.concatenate((previous_priority, priorities[selected]))
                if len(merged_priority) > limit:
                    retain = np.argpartition(merged_priority, -limit)[-limit:]
                    merged_x, merged_y, merged_priority = merged_x[retain], merged_y[retain], merged_priority[retain]
                reservoirs[cid] = merged_x, merged_y, merged_priority
        return reservoirs

    training = sampled_records(train_ids, samples_per_client)
    validation = sampled_records(val_ids, max(1, samples_per_client // 4))
    clients = []
    for client_id in range(nx * ny):
        train_x, train_y, _ = training[client_id]
        val_x, val_y, _ = validation[client_id]
        if len(train_x) == 0 or len(val_x) == 0:
            # Confidence requires an honest held-out validation MSE.
            continue
        clients.append(ClientData(client_id, torch.from_numpy(train_x.copy()),
                                  torch.from_numpy(train_y.copy()),
                                  torch.from_numpy(val_x.copy()),
                                  torch.from_numpy(val_y.copy()),
                                  torch.from_numpy(train_x[:, 2:4].copy())))
    if not clients:
        raise ValueError("No client has both training and validation samples; use a coarser grid")
    return clients


@dataclass
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values) -> "TargetScaler":
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("Scaler needs nonempty finite training targets")
        return cls(float(array.mean()), max(float(array.std()), 1e-6))

    def encode(self, values):
        return (values - self.mean) / self.std

    def decode(self, values):
        return values * self.std + self.mean
