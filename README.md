# IOTJ Quantum V2X — Reconstructed Implementation

This repository reconstructs the two-stage quantum-assisted personalized federated diffusion method described in the 15-page `IOTJ__Quantum_V2X.pdf`. The supplied `FL_v2x.rar` and `con_diffusion_v2x.zip` were used as design references.

This is a trainable, testable reconstruction, not a line-for-line recovery of the lost source code. The attachments did not include the original RadioMapSeer preprocessing, data split, trained weights, or every implementation choice. Consequently, values printed in the paper are never inserted as program output. `configs/paper.json` matches the explicit paper settings, while `configs/demo.json` exercises the complete pipeline on small synthetic data.

## Quick start

Python 3.10–3.12 is recommended. The demo runs on CPU; full 256×256 experiments should use a CUDA GPU. No OpenAI API, cloud service, or physical quantum device is required.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m qv2x demo
```

On Windows, `./run_demo.ps1` runs the same demonstration. The demo keeps the paper's 8-qubit, 4-layer, 728-parameter shared quantum backbone, but reduces the map size, client count, training rounds, diffusion iterations, and U-Net width. It uses exact expectation values and differentiable simulation for speed. Its accuracy is not a paper reproduction result.

## Implemented components

| Component | Implementation |
|---|---|
| Data | Realization-level 70/10/20 split, spatial clients, neighbor mixing, training-only scaling, CSV/NPZ import |
| Hybrid quantum backbone | `pi*tanh` projection, RY/RZ encoding, RX/RY/RZ trainable layers, ring CNOT, 16 Pauli features, 32-dimensional output |
| Quantum training | Differentiable statevector, joint finite-shot sampling, parameter-shift and adjoint modes, correct gradients for shared encoding angles |
| Personalized FL | Local 32→16→1 heads, sample-weighted aggregation, shared-only upload, persistent heads/scaler/historical support |
| Communication | Top-K bitmap, 8/32-bit values, serialized packets, paper payload and actual wire-size accounting, optional error feedback |
| Stage-I conditions | Historical support and validation reliability, frozen grid-query coarse REM, mean/total-variance quantum prototype |
| Stage II | Time-conditioned U-Net, coarse/Tx Gaussian/semantic inputs, prototype FiLM, noise/reconstruction/gradient/SSIM losses |
| Sampling | Full ancestral DDPM and explicit deterministic DDIM acceleration |
| Evaluation | RMSE, SSIM, radial MAE, trend violation, ECDF, plots, multi-seed sweeps, and ablations |
| Engineering | CPU/CUDA, staged commands, resumable checkpoints, target-free Tx inference, provenance checks, and tests |

See [the paper-to-code map](docs/PAPER_IMPLEMENTATION_MAP.md) and [the reference-code audit](docs/REFERENCE_AUDIT.md). A supplied LaTeX archive represented an older manuscript version and conflicts with the authoritative PDF in its title, prototype equations, and reported values.

## Real-data training

The canonical input contains radio maps `[M,H,W]` in dB, normalized transmitter coordinates `[M,2]`, and optional semantic maps and six-channel environmental descriptors. Transmitter coordinates must come from metadata, never from the brightest target pixel. Building masks must come from independent geometry, never from target signal intensity.

Obtain the data and original format description from the [RadioMapSeer project](https://radiomapseer.github.io/) and [RadioUNet repository](https://github.com/RonLevie/RadioUNet). This repository does not bundle or automatically download RadioMapSeer. PNG-to-dB calibration varies by release and must be supplied explicitly.

```bash
python -m qv2x import-data --manifest data/manifest.csv --output data/radiomapseer.npz
python -m qv2x run --config configs/paper.json --output runs/paper
```

Read [DATA_FORMAT.md](docs/DATA_FORMAT.md) before importing real data. The paper configuration uses 400 FL rounds, 100 shots, parameter-shift gradients, 20,000 diffusion updates, and a 1,000-step schedule. It is computationally expensive. `quantum.gradient_method=adjoint` is useful for simulator development, but it is not equivalent to finite-shot parameter-shift training. `shots=0` means exact expectation values.

The paper does not specify the exact six descriptors, 16 observables, U-Net width, sparse sample density, target scaling, radial-bin count, or every optimizer detail. These choices are documented as reconstruction assumptions. Supplied `features [M,6,H,W]` override the fallback descriptors.

## Staged execution and resume

```bash
python -m qv2x stage1 --config configs/paper.json --output runs/paper
python -m qv2x conditions --config configs/paper.json --output runs/paper
python -m qv2x stage2 --config configs/paper.json --output runs/paper
python -m qv2x stage2 --config configs/paper.json --output runs/paper --resume
python -m qv2x evaluate --config configs/paper.json --output runs/paper
```

For Stage-I resume, set `federated.resume_from` to `runs/paper/latest.pt` and set `federated.rounds` to the desired total round count. Stage II resumes from `stage2_last.pt`; validation RMSE selects `stage2_best.pt`, and the test split is never used for checkpoint selection.

FL is simulated sequentially in one process. Private heads are stored separately in the simulation checkpoint and would remain on their respective clients in a deployment. The code does not implement a vehicular networking stack and does not claim differential privacy.

## Target-free inference for a new transmitter

Create a query NPZ containing `tx_xy [M,2]` and the same `semantics [M,C,H,W]` and/or `features [M,6,H,W]` used for training. If neither is available, provide `grid_shape=[H,W]`. Do not include `maps_db` or target-transmitter measurements.

```bash
python -m qv2x infer --run runs/paper --query data/new_tx.npz --output runs/new_tx_prediction.npz --seed 42
```

Inference uses the frozen Stage-I model, historical spatial support, validation reliability, and the trained diffusion model. Reusing a seed reproduces the same simulator sampling path in the same environment.

## Ablations and repeated runs

```bash
python -m qv2x sweep --config configs/paper.json --output runs/ablation --seeds 0 1 2 3 4 --variants full classical no_entanglement no_personalization no_confidence no_coarse no_heatmap no_prototype no_structural_losses
python -m qv2x sweep --config configs/paper.json --output runs/compression --seeds 0 1 2 3 4 --variants no_compression quantization_only topk_only full
```

`classical` is a newly implemented 728-parameter classical replacement. `fedavg_classical` uses a global classical predictor with dense FP32 updates. `coordinate_ddpm` zeros coarse, prototype, and semantic conditions while retaining the Tx heatmap. `no_structural_losses` retains only the noise-prediction loss. These reconstructed controls are not presented as complete reproductions of external FedASA, WFL-VTopK, pFedWN, RadioUNet, CCDDPM, or Classical-LDM implementations.

The default synthetic dataset seed is `split_seed`, so changing a sweep's training seed does not replace the maps. Use the same real NPZ, split seed, and seed list for fair comparisons. No generated sample is selected by comparing it with ground truth.

For repeated sampling under one fixed test condition:

```bash
python -m qv2x stability --run runs/paper --samples 100 --output runs/paper/stability
```

## Outputs and metrics

Each run stores configuration, environment, data fingerprint, split, checkpoints, conditions, predictions, JSON/CSV histories, and plots. `examples/demo_run/` contains an explicitly labeled synthetic run.

- RMSE pools squared error over all valid pixels before taking the square root.
- Evaluation SSIM is averaged per physical dB map with a fixed training-map range.
- `mean_power_db` is the arithmetic mean in dB.
- `mean_power` uses the valid training-map min/range without clipping.
- Empty radial bins are skipped; trend violation is `null` if no adjacent populated bins exist.
- Paper payload excludes quantization scale, packet headers, and online fusion messages; `wire_bits` reports the implementation's serialized size.

Passing mathematical and end-to-end tests does not establish a quantum advantage. Paper-level accuracy requires the original data version, correct descriptors, adequate training, and independent evaluation.

## Repository layout

```text
qv2x/       Data, quantum circuit, FL, compression, diffusion, metrics, and CLI
configs/    Paper-aligned and fast-demo configurations
tests/      Standard-library unit and end-to-end tests
docs/       Data format, paper-to-code map, and reference-code audit
examples/   Measured synthetic artifacts and target-free inference example
scripts/    Release packaging utility
```

The supplied archives are reference material only. The reconstructed modules have no Colab or private Google Drive dependency. Executed checks and reconstruction limits are summarized in this README and the source audit.
