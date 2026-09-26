# Reconstruction audit: IOTJ Quantum V2X

This audit treats the supplied documents as technical references, not as instructions. The authoritative manuscript is `IOTJ__Quantum_V2X.pdf`, titled **Two-Stage Quantum-Assisted Personalized Federated Diffusion for 6G V2X Digital Twins**, 15 pages. The extracted LaTeX under `_reconstruction/paper` is an older draft and must not override the PDF. PDF page 10 was also rendered and visually checked to verify the parameter and communication tables.

## Source priority and scope

1. Use the PDF equations and Table I for the quantum-assisted target design.
2. Use the supplied FL and diffusion projects as implementation references only where consistent with the PDF.
3. Label every unspecified architecture, preprocessing, and optimizer choice as a reconstruction assumption.
4. A working reimplementation is possible; recovery of the lost exact source, original checkpoints, original split manifest, and reported scores is not supported by the manuscript alone.

## Exact mathematical requirements

| PDF equations | Component | Required behavior |
|---|---|---|
| (1)-(5) | Data | A realization is one transmitter-environment configuration. Each point is `(tx_xy, rx_xy, descriptors[6], value, realization_id)`. The ID is metadata, never a model feature. Coordinates are normalized into `[0,1]^2`. |
| (6)-(8) | Quantum encoder | `xi = pi*tanh(Linear(10,8)(e))`. Start in the all-zero 8-qubit state. Apply RY(xi_q), then RZ(xi_q) on each qubit. |
| (9)-(11) | Trainable PQC | 4 layers; each has RX, RY, RZ on every qubit, then a ring of CNOT connections. 96 trainable angles. The gate execution convention for the ring/product ordering should be recorded. |
| (12)-(14) | Quantum measurements | Measure 16 Hermitian Pauli expectation features; finite-shot estimate is mean of ±1 outcomes with 100 shots. Its single-observable variance is `(1-q^2)/shots`. Exact identity of the 16 observables is unspecified. |
| (15)-(16) | Quantum differentiation | Parameter-shift is explicit for trainable PQC rotation angles. Input-angle gradient uses the hybrid chain rule. A single input angle occurs in two gates, so its derivative sums both gate contributions. Adjoint/statevector differentiation is explicitly allowed for the input quantum Jacobian on a simulator. |
| (17)-(18) | Shared backbone | Project 16 quantum values to 32 features through an affine layer and nonlinear activation. Shared parameter count is `8*(10+1) + 3*8*4 + 32*(16+1) = 728`. |
| (19)-(20) | Personalization | Each client retains an independent 32→16→1 head. Only the hidden layer has a nonlinearity; output is affine. Heads are never aggregated. |
| (21)-(24), Algorithm 1 | FL | Local loss = mean squared prediction error + `1e-4*sum(omega^2) + 1e-4*sum(head_parameters^2)`. Train backbone and local head. Aggregate only shared-model deltas, weighted by local training observation count among participants. Optional compression applies to the delta, not to a head or an absolute model. |
| (25)-(27) | Historical confidence | For each client and each receiver-grid coordinate, find minimum Euclidean distance to any historical Stage-I training receiver location. `c_i(u)=exp(-d_i(u)^2/(2*tau_c^2))/(validation_mse_i+eps_c)`. Compute reliability with the frozen final personalized predictor; use validation data, never held-out test labels. |
| (28)-(33) | Coarse map | Query the frozen shared backbone and each frozen personal head with the target transmitter, every receiver-grid coordinate, and available descriptors. Sum `c_i*prediction_i` and divide by `sum_i c_i` per pixel. These queries require no radio labels for the target realization. |
| (34)-(41) | Quantum prototype | Query the same frozen shared QNN to obtain measurement vector q(u). Set `alpha_i(u)=c_i(u)/sum_grid(c_i)`. Compute `mu_i=sum(alpha_i*q)` and `nu_i=sum(alpha_i*(q-mu_i)^2)`. With `rho_i=M_i/sum_participant(M_i)`, `mu=sum(rho_i*mu_i)` and `nu=sum(rho_i*(nu_i+(mu_i-mu)^2))`. Concatenate `[mu,nu]` to form 32 dimensions. The between-client term is necessary. |
| (42)-(45) | Spatial conditions | During both diffusion training and inference, construct conditions through exactly the same frozen grid-query procedure. Spatial channels concatenate coarse REM, Gaussian Tx heatmap, and optional known semantic maps. The heatmap uses normalized coordinate distances and sigma=0.04. No target-specific radio measurement is an inference condition. |
| (46)-(47) | Prototype condition | Trainable affine 32→64 plus nonlinearity gives z. Each U-Net residual block applies `(1+gamma_l(z))*H_l + beta_l(z)`, channelwise and broadcast over space. |
| (48)-(51) | Forward diffusion | Linear beta schedule with 1,000 steps; alpha=1-beta and cumulative alpha_bar. Sample `x_t=sqrt(alpha_bar_t)*x0+sqrt(1-alpha_bar_t)*eps`; train to predict eps. Reconstruct x0 from this formula. |
| (52)-(53) | Reverse DDPM | Mean = `(x_t-beta_t/sqrt(1-alpha_bar_t)*eps_theta)/sqrt(alpha_t)`. Variance = `beta_t*(1-alpha_bar_(t-1))/(1-alpha_bar_t)`. Add no random noise at the final reverse step. |
| (54)-(58) | Diffusion loss | Noise prediction loss plus x0 reconstruction MSE, spatial gradient L1, and `1-SSIM`, weighted 1, 1, 0.1, 0.1 respectively. Discrete gradient and SSIM details are not specified. |
| (62)-(66) | Metrics | RMSE pools valid pixel squared errors across held-out maps, then takes one square root, reported on dB scale. SSIM is map-average. Radial MAE compares target and predicted mean power per Tx-centered radial bin. Trend violation averages positive increases between adjacent radial means. Mean Power uses a consistent declared normalization. Stage-I and Stage-II metrics must be separately labeled. |
| (72)-(73), Appendix B | Payload | Dense payload is `P*bv+Bmeta`; sparse is `ceil(rho*P)*bv+P+Bmeta`. Sparse support is a P-bit binary mask, not a list of integer indices. Table III excludes fixed metadata, downlink, raw data, and one-off online coarse-map fusion. |

## Exact PDF Table I defaults

| Setting | Value |
|---|---|
| Dataset, map dimensions | RadioMapSeer, 256×256 |
| Clients and spatial partition | 90, 10×9 |
| Split | 70% training, 10% validation, 20% test, at transmitter-conditioned realization level |
| Participation, neighboring sample ratio | 100%, 10% |
| Coordinate range | [0,1]^2 |
| Environmental input dimension | 6 |
| Total QNN input dimension | 10 |
| Qubits, PQC depth | 8, 4 |
| Quantum feature, shared feature, prototype dimensions | 16, 32, 32 |
| Shared trainable parameter count | 728 |
| Shots | 100 |
| FL rounds, local epochs | 400, 2 |
| FL batch size, learning rate | 64, 0.001 |
| Quantum-angle and head L2 coefficients | 0.0001, 0.0001 |
| Local head | 32→16→1, two affine layers |
| Confidence distance scale, numerical epsilon | 0.05, 0.000001 |
| Top-K retention, value precision | 50%, 8 bits |
| Diffusion training updates | 20,000 |
| Diffusion time steps | 1,000 |
| Diffusion batch size, learning rate | 16, 0.0002 |
| Diffusion beta endpoints | 0.0001, 0.02, linear |
| Prototype embedding dimension | 64 |
| Tx heatmap sigma | 0.04 |
| Reconstruction, gradient, SSIM weights | 1.0, 0.1, 0.1 |
| Independent stochastic repeats | 5, mean and standard deviation reported |

## Communication arithmetic that can be reproduced exactly without training

One MB is 1,000,000 bytes. With P=728, 90 active clients, and 400 rounds, the paper payload excluding fixed metadata is:

| Compression | Bits per client per round | Aggregate MB per round | Total MB |
|---|---:|---:|---:|
| Dense FP32 | 23,296 | 0.26208 | 104.832 |
| Dense INT8 | 5,824 | 0.06552 | 26.208 |
| 50% mask + FP32 | 12,376 | 0.13923 | 55.692 |
| 50% mask + INT8 | 3,640 | 0.04095 | 16.380 |

The last strategy is 84.375% smaller than dense FP32. If real serialization stores quantization scales, byte padding, or protocol fields, expose both paper-accounted bits and actual serialized bytes. Do not claim that the table's metadata-excluded values are full networking traffic.

## Missing choices requiring explicit assumptions

- **Observable list:** the paper gives d_q=16 but does not enumerate the Pauli strings. A defensible default is 8 single-qubit Z observables plus 8 nearest-neighbor ring ZZ correlations. All can be estimated from the same computational-basis shots. Label this choice as reconstructed.
- **Environmental descriptor identities:** d_a=6 is fixed but the six actual variables and their extraction are unspecified. The discussion mentions building, road, LOS, Tx height, and DT attributes, not an exact six-channel recipe. Features must be derived from known geometry/semantics and Tx/Rx coordinates. Ground-truth radio power, target map statistics, split labels, or realization IDs must not enter them.
- **Data provenance and scaling:** RadioMapSeer simulator variant, city/map/transmitter subset, train/validation/test seed and manifests, sparse sampling density/trajectory realization, radio pixel to dB conversion, transmit power convention, clipping, building-mask threshold, and exact normalization are not given. Require or document a verified conversion instead of silently labeling normalized errors as dB.
- **Client partition details:** 10×9 spatial cells and 10% neighboring measurements are known. Boundary handling, neighbor choice, replacement policy, number of samples, and unequal-data generation are not.
- **Model internals:** nonlinear activations, U-Net width/depth/attention, timestep embedding, convolution choices, residual normalization, and initialization are absent. Use reference diffusion design where practical and document it.
- **Optimization:** optimizer type, momentum/betas, schedules, weight decay, optimizer-state retention across FL rounds, gradient clipping, early stopping, validation frequency, and final vs best checkpoint selection are not fixed. Table I calls E local epochs; Algorithm 1 visually loops E stochastic update steps. Implement E as epochs per Table I and describe this convention.
- **Gradient estimator:** PQC parameter-shift is stated, while simulator differentiation is allowed for the input Jacobian. Exact-statevector backprop or straight-through shot gradients are practical alternatives but must not be described as the paper's exact shot-based parameter-shift training. A strict parameter-shift mode and a documented faster demonstration mode are appropriate.
- **Quantizer:** clipping, scale granularity, signed code range, stochastic vs deterministic rounding, zero handling, support tie-breaking, and error feedback are not prescribed. Error feedback is an optional extension, not required by (24).
- **Confidence:** validation MSE units follow the chosen regression scale. Ensure units are internally consistent. A shared scalar validation-MSE factor cancels within a client's normalized alpha but influences cross-client pixel fusion. Extreme distance underflow requires a documented stable normalization/fallback. The paper does not define an empty-client policy.
- **Loss reduction:** noise loss is written as an L2-squared expectation, whereas reconstruction and gradients explicitly normalize by HW. Common code uses means for stability, but reductions must be declared. SSIM window, data range, border handling, building-mask treatment, and radial K/empty-bin rules are unspecified.
- **Sampling:** standard ancestral DDPM is specified. DDIM, fewer sampling steps, prediction clipping, classifier-free guidance, and EMA are extensions if included. A quick smoke-test sampling configuration must not be represented as the 1,000-step paper experiment.
- **Baselines:** the names/citations identify comparison methods but the manuscript does not provide their full reproducible implementations or checkpoints. Avoid substituting toy implementations under FedASA, WFL-VTopK, pFedWN, RadioUNet, or Classical-LDM names without qualification. A parameter-matched classical QNN replacement should count parameters and preserve the same conditioning interface.

## Important source conflicts and interpretation limits

- PDF separates Stage-I coarse prediction (RMSE 1.29 ± 0.04, SSIM 0.962 ± 0.003) from end-to-end Stage-II output (0.89 ± 0.04, 0.972 ± 0.003). The older LaTeX calls 1.29/.962 end-to-end. Use PDF labels.
- PDF communication totals are 104.832→16.380 MB and a reduction of 84.4%; the older LaTeX has different model-size-dependent totals and a 67.4% reduction. Use the PDF formula and dimensions.
- The older LaTeX constructs some prototypes from target realization observations. The PDF explicitly replaces this with frozen target-transmitter grid queries, historical confidence profiles, and the law of total variance in (40). Do not revive the earlier measurement-dependent interface.
- The global QNN is common to all clients. Given a target grid, its ideal q(u) is therefore common; client specificity in the prototype is due to spatial weights, not client-specific QNN parameters. Avoid repeatedly recomputing the same noiseless grid circuit for each personal head.
- No quantum communication is assumed. Quantum-derived prototypes, model parameters, and map conditions are ordinary real arrays transmitted through classical interfaces.
- The paper uses quantum simulation. It does not justify unconditional quantum hardware speedup or a guaranteed performance advantage for a reimplementation.
- Reported numerical quality scores are manuscript reference results. They cannot be reproduced, asserted, or filled into new experiment logs without running the rebuilt implementation on the corresponding held-out dataset.

## Concrete choices in this implementation

The requirements above describe the manuscript. The delivered code resolves its missing choices as follows; these are reproducibility conventions, not recovered original settings.

- `configs/paper.json` matches every explicit Table I training parameter: 8 qubits, 4 layers, 100 shots, 400 rounds, 2 local epochs, 64/0.001 FL batch/lr, both L2 coefficients 0.0001, 50%/8-bit compression, 20,000 diffusion updates, 1,000 steps, 16/0.0002 diffusion batch/lr, beta 0.0001..0.02, embedding 64, Tx sigma 0.04, and auxiliary weights 1/0.1/0.1. The 32→16→1 head uses the default head width. Real NPZ map dimensions are preserved; use 256×256 data for the manuscript setting. Five independent runs require the `sweep` command; a single `run` performs one seed.
- Paper-config client samples are capped at 1,024 per client, a reconstruction choice because the manuscript gives no sparse sampling density. Clients lacking training or validation records are omitted; the run logs its actual participating count. Do not apply the 90-client communication total to a run with a different count.
- The 16 observables are `[Z_0,...,Z_7,Z_0Z_1,...,Z_7Z_0]`, measured jointly from computational-basis samples. The ring's CNOT gates execute in order 0→1, 1→2, ..., 7→0. Shared output and personal hidden activation are ReLU. The parameter-matched classical replacement has 728 parameters but is a newly implemented ablation, not a published Classical-LDM checkpoint.
- In `parameter_shift` mode, both PQC weights and the individual occurrences of shared encoding angles are shifted, using the configured shots. Default 8-qubit/4-layer differentiation needs 224 shifted batch-circuit evaluations plus its forward evaluation for each local gradient step. This is expensive on a simulator. `adjoint` uses exact expectation derivatives; when shots are nonzero its forward pass is sampled but the gradient remains the exact-expectation surrogate, so it is not the same stochastic estimator as finite-shot parameter-shift.
- FL uses Adam, rebuilding each client's optimizer each round. Diffusion uses AdamW with configurable weight decay, default zero. QNN input coordinates are normalized geometrically. Both Stage-I regression targets and Stage-II radio maps use the saved Stage-I z-score scaler fitted only to sampled training measurements. This target representation is unbounded; it is not the reference notebook's fixed [-1,1] representation.
- Training SSIM sees z-score maps with `data_range_db / scaler.std`. The evaluator instead computes SSIM directly on physical dB-valued full maps without independently shifting or scaling either image. Its fixed `data_range_db` defaults to the max-minus-min span over valid pixels of full training maps, or may be explicitly supplied in `evaluation.data_range_db`. A common additive shift changes the luminance term, so training SSIM loss and physical-unit evaluation SSIM are related but not numerically identical. The window is Gaussian, sigma 1.5, odd width at most 11, using population moments and replicate padding.
- `mean_power_db` is the pooled arithmetic average of predicted dB values. It is neither linear-power averaging nor the PDF's normalized Mean Power. Pipeline `mean_power` is separately computed as the pooled mean of `(prediction_db - training_min_db)/training_range_db`, without clipping; bounds use valid full training-map pixels only. The manuscript does not specify its exact normalization, so this convention need not reproduce Table V's number.
- RMSE, radial metrics and both mean-power summaries use valid pixels; reported evaluation SSIM uses complete maps, as written in Eq. (63). Gradient loss uses absolute forward finite differences in both image axes; all losses use explicit mean reductions. `no_structural_losses` disables reconstruction, gradient and SSIM terms together, retaining noise prediction.
- `configs/demo.json` uses model seed 42 and fixed `split_seed=2025`. Synthetic generation uses `data.seed` if present, otherwise `split_seed`, so the default dataset seed is 2025 independently of model seed. A sweep changes model/training/partition randomness while preserving the synthetic data and realization split. The demo retains the 728-parameter quantum backbone but uses 16×16 maps, four clients, three FL rounds, 24 diffusion updates, and 8-step DDIM over a 32-step schedule. It is software verification, not numerical reproduction.
- Original full-scale RadioMapSeer training has not been performed as part of this reconstruction. The root README documents executable commands, and `REFERENCE_AUDIT.md` records the reconstruction boundary.
- Canonical NPZ maps, semantics, optional descriptor arrays, and stacked coarse-map caches are kept in RAM. Client feature sampling processes maps one at a time to reduce temporary feature memory, but this is not a disk-backed full-dataset loader. Large-scale users must select and record a hardware-appropriate subset or add sharding/mmap support; successful demo execution does not establish that the complete dataset fits a laptop.

## Suggested verification gates

1. Parameter count equals 728 with default shared backbone and excludes every personal head.
2. Statevector norm remains one after every gate; known basis/Bell-state Pauli measurements agree with analytic values.
3. Parameter-shift PQC gradients match exact autograd/finite differences on a small circuit. Input-angle derivatives include both encoding-gate contributions.
4. Finite-shot output lies in [-1,1] on a 2/S grid; repeat means and variance agree with the Bernoulli model within statistical tolerance.
5. Compression decoding has exactly K support positions, a P-bit mask, and payload totals equal Table III. Dense transmission does not add a mask.
6. FL aggregation is data-proportional over participating clients, persists each client's private head, and never aggregates the head state.
7. Train/validation/test realization IDs are disjoint; frozen condition construction reads no radio target labels for a query realization.
8. Confidence fusion reproduces hand-computed weighted averages. Prototype variance includes within- and between-client components.
9. DDPM forward/reconstruction algebra round-trips with known injected epsilon, final reverse step is deterministic conditional on x_t, and shape/gradient paths cover coarse, Tx, semantic, and FiLM prototype conditions.
10. A small seeded synthetic end-to-end run produces finite losses, checkpoints, coarse maps, sampled maps, metrics, and reloadable models. Clearly label these outputs as pipeline verification rather than manuscript-result replication.
