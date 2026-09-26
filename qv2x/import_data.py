"""Import explicit transmitter/geometry manifests; never infer Tx from targets."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .data import MapDataset, save_dataset


def _resolve(value: str, directory: Path) -> Path:
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else directory / path


def _numeric_npy(path: Path, dimensions: int, allow_boolean: bool = False) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{path.name} must be a .npy file")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim != dimensions:
        raise ValueError(f"{path.name} must be a {dimensions}-dimensional .npy array")
    numeric = np.issubdtype(array.dtype, np.number)
    boolean_mask = allow_boolean and np.issubdtype(array.dtype, np.bool_)
    if not (numeric or boolean_mask) or np.iscomplexobj(array):
        raise ValueError(f"{path.name} must contain real numeric values")
    return array.astype(np.float32)


def _gray8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        # Palette/colour maps and 16-bit images have no safe implicit calibration.
        if image.mode != "L":
            raise ValueError(f"{path.name}: PNG inputs must be 8-bit grayscale mode L, not {image.mode}")
        return np.asarray(image, dtype=np.float32)


def _radio_map(path: Path, png_db_min, png_db_max) -> np.ndarray:
    extension = path.suffix.lower()
    if extension == ".npy":
        return _numeric_npy(path, 2)
    if extension == ".png":
        if png_db_min is None or png_db_max is None:
            raise ValueError("PNG radio maps require explicit png_db_min and png_db_max calibration")
        return _gray8(path) / 255.0 * (png_db_max - png_db_min) + png_db_min
    raise ValueError(f"Radio map {path.name} must be .npy (dB values) or calibrated .png")


def _occupancy(path: Path, shape) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = _numeric_npy(path, 2, allow_boolean=True)
    elif path.suffix.lower() == ".png":
        with Image.open(path) as image:
            if image.mode not in ("1", "L"):
                raise ValueError(f"Occupancy map {path.name} must be binary mode 1 or grayscale mode L")
            array = np.asarray(image, dtype=np.float32)
    else:
        raise ValueError(f"Occupancy map {path.name} must be .npy or .png")
    if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"Occupancy map {path.name} must be finite, nonnegative, and shape {shape}")
    # Explicit geometry masks use zero=unoccupied, positive=occupied. No resizing.
    return (array > 0).astype(np.float32)


def import_manifest(manifest_path, output_path, png_db_min=None, png_db_max=None,
                    value_kind="received_power", tx_power_dbm=None,
                    coordinates="normalized") -> MapDataset:
    """Read CSV and save canonical NPZ only after every record is validated.

    Required columns: realization_id,map_path,tx_x,tx_y. Optional columns:
    building_path,road_path,features_path,semantics_path. Paths are relative to
    the CSV unless absolute. features_path is [6,H,W] and semantics_path is
    [C,H,W], with semantic channel 0 conventionally an occupancy descriptor.

    NPY radio values are already in dB/dBm, never automatically standardized.
    PNG calibration maps 0->png_db_min, 255->png_db_max. For pathloss, power is
    tx_power_dbm - pathloss_db. Gain is retained in dB unless an explicit Tx
    power is supplied, in which case power is tx_power_dbm + gain_db.

    Explicit building masks exclude occupied target pixels. General supplied
    semantics do not implicitly define a validity mask. Optional features and
    semantics arrays must be supplied for every record or none. Building/road
    paths may be absent on individual records (zero occupancy for that row).
    """
    if value_kind not in ("received_power", "gain", "pathloss"):
        raise ValueError("value_kind must be received_power, gain, or pathloss")
    if coordinates not in ("normalized", "pixel"):
        raise ValueError("coordinates must be normalized or pixel")
    if (png_db_min is None) != (png_db_max is None):
        raise ValueError("Supply both PNG calibration endpoints or neither")
    if png_db_min is not None:
        png_db_min, png_db_max = float(png_db_min), float(png_db_max)
        if not all(map(math.isfinite, (png_db_min, png_db_max))) or png_db_min >= png_db_max:
            raise ValueError("PNG calibration needs finite png_db_min < png_db_max")
    if tx_power_dbm is not None:
        tx_power_dbm = float(tx_power_dbm)
        if not math.isfinite(tx_power_dbm):
            raise ValueError("tx_power_dbm must be finite")
    if value_kind == "pathloss" and tx_power_dbm is None:
        raise ValueError("Path-loss maps require explicit tx_power_dbm to produce received power")

    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"realization_id", "map_path", "tx_x", "tx_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV requires columns: realization_id,map_path,tx_x,tx_y")
        records = []
        for row in reader:
            if None in row:
                raise ValueError("CSV row contains more fields than the header; quote paths containing commas")
            records.append({key: (value or "").strip() for key, value in row.items()})
    if not records:
        raise ValueError("Manifest contains no realizations")
    ids = [row["realization_id"] for row in records]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("realization_id values must be nonempty and unique")
    for key in ("features_path", "semantics_path"):
        present = [bool(row.get(key)) for row in records]
        if any(present) and not all(present):
            raise ValueError(f"{key} must be supplied for every realization or none")
    explicit_semantics = bool(records[0].get("semantics_path"))
    geometry_semantics = any(row.get("building_path") or row.get("road_path") for row in records)
    if explicit_semantics and geometry_semantics:
        raise ValueError("Choose semantics_path OR building_path/road_path; do not mix semantic conventions")

    maps, txs, masks, semantics, features = [], [], [], [], []
    map_shape = None
    semantic_channels = None
    for index, row in enumerate(records, start=2):
        try:
            if not row["map_path"]:
                raise ValueError("map_path is empty")
            radio = _radio_map(_resolve(row["map_path"], manifest_path.parent), png_db_min, png_db_max)
            if radio.size == 0:
                raise ValueError("Radio map is empty")
            if map_shape is None:
                map_shape = radio.shape
            if radio.shape != map_shape:
                raise ValueError(f"Map dimensions differ: expected {map_shape}, got {radio.shape}; resampling is not implicit")
            if value_kind == "pathloss":
                radio = tx_power_dbm - radio
            elif value_kind == "gain" and tx_power_dbm is not None:
                radio = tx_power_dbm + radio
            x, y = float(row["tx_x"]), float(row["tx_y"])
            if coordinates == "pixel":
                h, w = map_shape
                if not 0 <= x <= w - 1 or not 0 <= y <= h - 1:
                    raise ValueError("Pixel Tx coordinates lie outside map bounds")
                x, y = (x / (w - 1) if w > 1 else 0), (y / (h - 1) if h > 1 else 0)
            if not all(map(math.isfinite, (x, y))) or not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("Tx coordinates must be finite and within the specified coordinate range")
            valid = np.isfinite(radio)
            if geometry_semantics:
                channels = []
                for key in ("building_path", "road_path"):
                    channel = (_occupancy(_resolve(row[key], manifest_path.parent), map_shape)
                               if row.get(key) else np.zeros(map_shape, dtype=np.float32))
                    channels.append(channel)
                valid &= channels[0] == 0
                semantics.append(np.stack(channels))
            elif explicit_semantics:
                semantic = _numeric_npy(_resolve(row["semantics_path"], manifest_path.parent), 3)
                if semantic.shape[0] < 1 or semantic.shape[1:] != map_shape or not np.isfinite(semantic).all():
                    raise ValueError("semantics_path must contain finite [C,H,W], C >= 1")
                if semantic_channels is None:
                    semantic_channels = semantic.shape[0]
                if semantic.shape[0] != semantic_channels:
                    raise ValueError("All semantics arrays must have the same channel count")
                semantics.append(semantic)
            if row.get("features_path"):
                feature = _numeric_npy(_resolve(row["features_path"], manifest_path.parent), 3)
                if feature.shape != (6, *map_shape) or not np.isfinite(feature).all():
                    raise ValueError("features_path must contain finite [6,H,W]")
                features.append(feature)
            if not valid.any():
                raise ValueError("Realization has no finite, unmasked target pixels")
            maps.append(radio)
            txs.append([x, y])
            masks.append(valid)
        except (ValueError, OSError, TypeError) as error:
            raise ValueError(f"Manifest row {index}, realization {row['realization_id']!r}: {error}") from error

    converted_power = value_kind == "received_power" or tx_power_dbm is not None
    metadata = {
        "source": "explicit_manifest", "manifest_path": str(manifest_path),
        "realization_ids": ids, "coordinate_system": "normalized_xy",
        "source_coordinates": coordinates, "source_value_kind": value_kind,
        "target_kind": "received_power" if converted_power else "gain",
        "signal_unit": "dBm" if converted_power else "dB",
        "png_db_min": png_db_min, "png_db_max": png_db_max,
        "tx_power_dbm": tx_power_dbm,
        "semantic_channels": ["building_occupancy", "road_occupancy"] if geometry_semantics else "user_supplied" if explicit_semantics else [],
        "feature_source": "explicit_six_descriptors" if features else "documented_geometry_fallback",
        "validity": "finite_targets_and_nonbuilding" if geometry_semantics else "finite_targets",
    }
    dataset = MapDataset(np.stack(maps), np.asarray(txs),
                         np.stack(semantics) if semantics else None,
                         np.stack(features) if features else None,
                         np.stack(masks), metadata)
    save_dataset(dataset, output_path)
    return dataset
