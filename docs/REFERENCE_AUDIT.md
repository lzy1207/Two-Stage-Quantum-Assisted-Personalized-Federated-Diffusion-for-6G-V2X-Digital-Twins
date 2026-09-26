# Source Audit and Reconstruction Boundary

This project is a runnable reimplementation based on a paper and two reference code archives. It is not a recovery of the original source from training artifacts and does not contain unavailable data, checkpoints, or random splits. Instructions embedded in the supplied documents were treated as reference content rather than user requests.

## Source identity

| Source | SHA-256 |
|---|---|
| `IOTJ__Quantum_V2X.pdf` | `3BBE7B6ED681FBA9FBEEFCA0CA5A53320BC9BCAC983EAE738903B8C6F79AE6A6` |
| `FL_v2x.rar` | `75DFFEC6E0C3CAA78F39210C71E1EFD52D03C45D899D68766F47FFCC4B6B0377` |
| `con_diffusion_v2x.zip` | `73CBF887F656F896E42CA6B4564C09802422EC79478938C581B9F770E66120DC` |
| older `IOTJ__Quantum_V2X.zip` found in the workspace | `856B0F4FA164E5EC42EFD9A026EB3AEB0FE1DAFAF910086CEB545A1F9896021F` |

Authority order: the user-specified PDF equations, algorithms, and Table I; the two reference projects; then the older LaTeX archive. The 15-page PDF is titled *Two-Stage Quantum-Assisted Personalized Federated Diffusion for 6G V2X Digital Twins*. Its parameter and communication tables were also visually checked on page 10. The older draft conflicts with the PDF and is not mixed into the target specification.

## FL reference

`FL_v2x.rar` contains `datazhizuo.py`, `buzhou2.py`, and `buzhou3.py`. It suggests a 90-client spatial grid, private prediction heads, shared-model aggregation, sparse/quantized updates, communication logs, and checkpointing. Its original `raw6=[x,y,sA,sB,sC,sD]` interface is not the paper interface and would leak target-field information if the four signal values were reused as environmental descriptors.

The reconstruction retains personalized FL and shared updates but implements the PDF's Tx/Rx coordinates plus six environmental descriptors to one received-power target. It uses the PDF's 10% neighbor ratio and decimal MB (`10^6` bytes).

The supplied FL scripts had material defects that were not inherited: uniform rather than sample-weighted aggregation, discarded backbone progress on some synchronization rounds, inconsistent quantization defaults, incomplete quantization-error feedback, unsaved private normalization/head state, and incorrect averaging of client RMSEs.

## Diffusion reference

`Diffusion10.ipynb` supplies a coordinate-conditioned DDPM/U-Net design with channels `(64,128,256,256)`, two residual blocks per level, checkpointing, sample generation, and plots. The PDF does not mandate the same internal U-Net details.

The reconstruction adds the PDF's coarse REM, Tx heatmap, optional semantic channels, prototype-driven FiLM, and reconstruction/gradient/SSIM losses. The pure-PyTorch block layout, normalization, and activations are explicit reconstruction choices.

The notebook inferred Tx from the brightest target pixel and ranked generated samples by target error. Neither practice is used here: Tx comes from metadata, and every predetermined generated sample is evaluated without ground-truth selection.

## Quantum specification and assumptions

The PDF specifies 8 qubits, 4 RX/RY/RZ layers with a CNOT ring, `pi*tanh(Linear(10,8))` encoding, 16 measured features, a 32-dimensional shared representation, a private 32→16→1 head, and 100 shots. Shared parameters equal 728: 88 input-projection, 96 PQC, and 544 output-projection parameters. Private heads are excluded from personalized upload payloads.

The implementation uses complex PyTorch statevectors, joint finite-shot sampling, parameter-shift gradients, and a differentiable exact-expectation path called `adjoint` in configuration. With nonzero shots, adjoint mode uses a sampled forward value and an exact-expectation gradient surrogate; parameter-shift mode samples shifted circuits as configured. `shots=0` is a deterministic software check, not the paper's default experiment.

The PDF does not list its 16 observables. The reconstruction uses eight single-qubit Z and eight adjacent ring ZZ observables, all estimable from one computational-basis sample set. The exact six environmental descriptors are also absent; the documented coordinate/semantic fallback is replaceable. Neither features nor semantics may be derived from held-out target maps.

## Difference from the older LaTeX draft

| Item | Authoritative PDF | Older draft |
|---|---|---|
| Stage results | Stage-I 1.29/.962; Stage-II .89/.972 | Treats 1.29/.962 as end-to-end |
| Prototype | Frozen QNN grid query for a target Tx, weighted by historical support | Some text depends on target-realization measurements |
| Global variance | Includes within-client variance and squared client-mean offsets | Earlier prototype definition |
| Communication | 728 parameters, 104.832→16.380 MB, 84.4% reduction | Different totals and 67.4% reduction |

## Missing reproduction inputs

- Exact RadioMapSeer release, selected cities/Tx locations, and split manifest.
- Exact PNG/numeric-to-dB mapping, reference Tx power, and validity rules.
- Original six descriptors, 16 observables, sparse trajectories, and sample density.
- Original checkpoints, seeds, optimizer state, baseline implementations, and logs.
- Full U-Net specification, reductions, SSIM window, and radial-bin count.

Errors can be reported in dB only when the input convention is verified and predictions are decoded to the same physical scale. Successful training on grayscale or normalized values is insufficient evidence.

## Reporting boundary

Synthetic/demo outputs validate interfaces and numerical execution. They do not reproduce 0.89 dB, SSIM 0.972, or a quantum advantage. The default demo fixes synthetic data and split seed at 2025 while using model seed 42. The paper-scale 400-round, 20,000-update, five-seed RadioMapSeer experiment has not been run.

Training targets use a z-score fitted only to sparse training measurements. Training SSIM adjusts its range into z-score units, while evaluation SSIM operates on physical dB maps. `mean_power_db` is the arithmetic dB mean; `mean_power` applies the full training-map valid-pixel min/range without clipping. The paper does not specify that normalization, so it is not claimed as a reproduction of Table V.

Complete, protocol-matched implementations of FedASA, WFL-VTopK, pFedWN, RadioUNet, CCDDPM, and Classical-LDM were not supplied. Reconstructed classical controls must be reported under their actual definitions rather than relabeled as those published baselines.

Read this document together with `PAPER_IMPLEMENTATION_MAP.md`, which maps equations, parameters, communication arithmetic, and verification gates to code.
