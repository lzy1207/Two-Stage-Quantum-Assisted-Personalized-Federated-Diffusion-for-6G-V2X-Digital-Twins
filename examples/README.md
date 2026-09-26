# Synthetic Run Example

`demo_run/` was produced by:

```bash
python -m qv2x demo --output examples/demo_run
```

It contains measured artifacts from 12 synthetic 16×16 maps, four clients, three FL rounds, 24 diffusion updates, and 8-step DDIM sampling. The quantum module retains 8 qubits, 4 layers, and 728 shared parameters, using exact expectation values and differentiable simulation.

This example verifies software integration only. It is deliberately undertrained, cannot replace a RadioMapSeer experiment, and does not guarantee that Stage II improves over Stage I. The recorded metrics are in `demo_run/metrics.json`; no paper table values were inserted.

`new_tx_query.npz` contains one transmitter coordinate and an empty building-semantic channel. It contains no target radio map. `new_tx_prediction.npz` was created with:

```bash
python -m qv2x infer --run examples/demo_run --query examples/new_tx_query.npz --output examples/new_tx_prediction.npz --seed 42
```

The example checkpoints match the small demo configuration and must not be loaded into the paper-scale model.
