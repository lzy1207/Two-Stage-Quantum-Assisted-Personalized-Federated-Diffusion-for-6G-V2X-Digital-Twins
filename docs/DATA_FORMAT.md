# Data Format and Real-Data Import

Real experiments enter the project through a CSV manifest and are converted to a canonical NPZ. Every row explicitly identifies a radio map, transmitter coordinates, and optional environmental information. The importer never derives the transmitter from the brightest target pixel and never derives buildings from received power.

Consult the [RadioMapSeer project](https://radiomapseer.github.io/) and the [RadioUNet repository](https://github.com/RonLevie/RadioUNet) for the original dataset. This project neither bundles nor downloads the full dataset. Gain, path loss, received-power, coordinate, and grayscale conventions differ across releases; verify the convention for the exact version being used.

## CSV manifest

The CSV must be UTF-8, optionally with a BOM. Relative paths are resolved from the manifest's directory. Standard CSV quoting is required for paths containing commas.

```csv
realization_id,map_path,tx_x,tx_y,building_path,road_path,features_path,semantics_path
scene_000_tx_00,maps/000_00.npy,0.42,0.63,buildings/000.png,roads/000.png,,
```

| Field | Meaning |
|---|---|
| `realization_id` | Required unique nonempty identifier; metadata only, never a model input |
| `map_path` | Required 2D numeric `.npy`, or calibrated 8-bit grayscale `.png` |
| `tx_x`, `tx_y` | Required Tx coordinates; x follows columns and y follows rows; normalized by default |
| `building_path` | Optional independent occupancy map; zero is free space and positive is building |
| `road_path` | Optional independent road occupancy map |
| `features_path` | Optional `.npy` environmental descriptors with shape `[6,H,W]` |
| `semantics_path` | Optional custom semantics with shape `[C,H,W]`; mutually exclusive with building/road paths |

All radio maps must have the same `(H,W)`. Environmental arrays must be pixel-aligned. The importer never crops, resizes, interpolates, or changes coordinate systems. If `features_path` or `semantics_path` is used, every row must provide it, and semantic channel counts must agree. Building and road paths may be empty on individual rows, producing zero occupancy.

With `--coordinates pixel`, pixel coordinates are normalized as `x/(W-1)` and `y/(H-1)`. Boundary coordinates are valid and out-of-range values are rejected.

## Radio values and calibration

### Numeric NPY

An NPY radio map must be a two-dimensional real numeric array. Values are assumed to already use the declared dB/dBm convention. Normalized `[0,1]` arrays and `0..255` arrays are not silently converted.

```bash
python -m qv2x import-data \
  --manifest data/manifest.csv \
  --output data/radiomapseer.npz \
  --value-kind received_power
```

| `--value-kind` | Input | Stored `maps_db` |
|---|---|---|
| `received_power` | Received power in dBm | Unchanged |
| `gain` | Channel gain in dB | Unchanged, or `tx_power_dbm + gain_db` when Tx power is given |
| `pathloss` | Positive path loss in dB | Requires Tx power; stores `tx_power_dbm - pathloss_db` |

Example for path-loss data and 23 dBm transmit power:

```bash
python -m qv2x import-data --manifest data/manifest.csv --output data/radiomapseer.npz --value-kind pathloss --tx-power-dbm 23
```

This conversion contains only the supplied scalar Tx power. Apply antenna gains, reference levels, and other calibration required by the source dataset before import.

### Explicit PNG calibration

Radio PNGs must be Pillow mode `L` 8-bit grayscale. RGB, palette, pseudocolor, and 16-bit images are rejected because they have no reliable universal power mapping. Both endpoints are mandatory:

```text
radio_db = pixel / 255 * (png_db_max - png_db_min) + png_db_min
```

```bash
python -m qv2x import-data --manifest data/manifest.csv --output data/radiomapseer.npz --png-db-min <documented-min> --png-db-max <documented-max>
```

The project intentionally provides no default RadioMapSeer limits. Reverse grayscale, nonlinear encodings, clipping, and special invalid-value codes must be decoded to a numeric NPY first. A zero PNG pixel is not automatically considered a building or invalid target.

## Geometry, descriptors, and valid pixels

Building and road maps must be independent geometry. Boolean NPY masks and binary/grayscale PNG masks are accepted; `value>0` becomes occupied. When both paths are used, semantic channels are ordered as building then road.

Building pixels are excluded from valid target pixels. Roads do not alter validity. Without a building path, target values are never used to guess buildings. Nonfinite radio NPY values are marked invalid and replaced by finite storage placeholders. Every realization requires at least one valid pixel.

General `semantics_path` arrays are preserved and do not implicitly define an evaluation mask. For a custom mask, construct the canonical NPZ directly.

Explicit six-channel features must be available at inference and must not contain target power, target-map statistics, errors, realization IDs, or test labels. Without them, `get_features` uses this documented reconstruction fallback:

1. semantic channel 0, or zero;
2. normalized Tx/Rx Euclidean distance divided by `sqrt(2)`;
3. absolute x distance;
4. absolute y distance;
5. `sin(pi*rx_x)`;
6. `sin(pi*rx_y)`.

The final point input is `[tx_x,tx_y,rx_x,rx_y,six descriptors]` with dimension 10. The fallback was not recovered from the lost original source.

## Canonical NPZ

| Array | Shape and meaning |
|---|---|
| `maps_db` | float32 `[M,H,W]`, declared dB/dBm values |
| `tx_xy` | float32 `[M,2]`, normalized Tx coordinates |
| `valid_mask` | bool `[M,H,W]`, valid target/evaluation pixels |
| `semantics` | optional float32 `[M,C,H,W]` known environmental channels |
| `features` | optional float32 `[M,6,H,W]` known descriptors |
| `metadata_json` | scalar UTF-8 JSON containing IDs, source convention, calibration, and units |

Files are loaded with `allow_pickle=False`. The current loader materializes arrays in RAM; compressed NPZ does not provide true memory mapping. Budget approximately `M*H*W*channels*4` bytes plus masks, conditions, and intermediate copies. The client sampler processes one realization at a time, but the base maps and cached conditions remain resident. For larger datasets, add a sharded or memory-mapped backend.

Training, validation, and testing are split by complete transmitter-conditioned realizations, never by randomly splitting pixels from one map. Stronger unseen-city evaluation should additionally group by scene. Query feature construction and Stage-I condition generation may read coordinates, known semantics, historical support, and frozen models, but never the held-out target map.

These checks enforce the interface and declared units. They do not establish that a third-party dataset was interpreted correctly or guarantee the paper's reported values.
