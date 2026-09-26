from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from qv2x.data import TargetScaler, get_features, make_synthetic_dataset, partition_clients, split_realizations
from qv2x.federated import (FederatedState, PersonalizedHead, build_conditions,
                            confidence_log_weights, load_federated,
                            prototype_from_moments, save_federated,
                            train_federated, weighted_average)


class TinyBackbone(nn.Module):
    """Deterministic test double: tests federation independently of PQC physics."""
    def __init__(self):
        super().__init__()
        self.hidden_dim, self.feature_dim = 4, 2
        self.input_projection = nn.Linear(10, 2)
        self.output_projection = nn.Linear(2, 4)

    def quantum_features(self, x):
        return torch.tanh(self.input_projection(x))

    def forward(self, x):
        return torch.relu(self.output_projection(self.quantum_features(x)))

    def regularization(self):
        return self.input_projection.weight.square().sum()


class FederatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        data = make_synthetic_dataset(count=6, size=8, seed=3)
        split = split_realizations(6)
        clients = partition_clients(data, split["train"], split["val"], (2, 1), 12, 0, 1)
        config = dict(rounds=1, local_epochs=1, batch_size=6, lr=.01, seed=9,
                      topk_ratio=.5, quant_bits=8, error_feedback=True,
                      lambda_q=1e-4, lambda_v=1e-4)
        return data, split, clients, config

    def test_weighted_aggregation_is_not_uniform(self):
        result = weighted_average([torch.tensor([1., 2.]), torch.tensor([9., 6.])], [1, 3])
        torch.testing.assert_close(result, torch.tensor([7., 5.]))
        with self.assertRaises(ValueError):
            weighted_average([torch.ones(2)], [0])

    def test_total_variance_includes_between_client_means(self):
        features = torch.tensor([[-1.], [1.]])
        log_confidence = torch.tensor([[0., -1000.], [-1000., 0.]])
        result = prototype_from_moments(features, log_confidence, [1, 3])
        # Local variances approach zero. Global variance is still .75.
        torch.testing.assert_close(result, torch.tensor([.5, .75]), atol=1e-6, rtol=1e-6)

    def test_confidence_is_finite_far_from_observations(self):
        query = torch.tensor([[0., 0.], [1., 1.]])
        logs = torch.stack([confidence_log_weights(torch.tensor([[0., 0.]]), query, 1., tau=1e-5),
                            confidence_log_weights(torch.tensor([[1., 1.]]), query, 4., tau=1e-5)])
        weights = torch.softmax(logs, dim=0)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.sum(0), torch.ones(2))
        self.assertEqual(weights[0, 0].item(), 1.)

    def test_training_checkpoint_and_target_free_conditions(self):
        data, split, clients, config = self.fixture()
        # Extreme validation labels cannot contaminate training normalization.
        expected_mean = torch.cat([c.train_y for c in clients]).mean().item()
        clients[0].val_y = clients[0].val_y + 100
        with tempfile.TemporaryDirectory() as directory:
            state = train_federated(TinyBackbone(), clients, config, directory)
            self.assertAlmostEqual(state.scaler.mean, expected_mean, places=4)
            query = get_features(data, [split["test"][0]])[0]
            coarse, proto = build_conditions(state, query, batch_size=11)
            self.assertEqual(coarse.shape, (8, 8))
            self.assertEqual(proto.shape, (4,))
            self.assertTrue(np.isfinite(coarse).all())
            self.assertTrue(np.all(proto[2:] >= 0))
            restored = load_federated(Path(directory) / "federated.pt", TinyBackbone())
            coarse2, proto2 = build_conditions(restored, query, batch_size=11)
            np.testing.assert_allclose(coarse, coarse2)
            np.testing.assert_allclose(proto, proto2)
            self.assertEqual(restored.history, state.history)
            self.assertEqual(restored.validation_mse, state.validation_mse)
            shared = sum(p.numel() for p in state.backbone.parameters())
            self.assertEqual(state.history[0]["uploaded_parameters"], shared)
            # Per-round paper bitmap + 8-bit Top-K values, private heads absent.
            expected_bits = len(clients) * (shared + 8 * int(np.ceil(shared * .5)))
            self.assertEqual(state.history[0]["paper_bits"], expected_bits)
            self.assertGreater(state.history[0]["wire_bits"], expected_bits)
            global_config = {"federated": config, "backbone": {"hidden_dim": 4}, "seed": 5}
            save_federated(state, Path(directory) / "with_run_config.pt", global_config)
            restored = load_federated(Path(directory) / "with_run_config.pt", TinyBackbone())
            self.assertEqual(restored.config, config)
            checkpoint = torch.load(Path(directory) / "with_run_config.pt", weights_only=True)
            self.assertEqual(checkpoint["run_config"], global_config)

    def test_resume_matches_continuous_training(self):
        _, _, clients, config = self.fixture()
        torch.manual_seed(12)
        initial = TinyBackbone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            full = train_federated(deepcopy(initial), clients, {**config, "rounds": 2}, path / "full")
            train_federated(deepcopy(initial), clients, config, path / "first")
            resumed = train_federated(deepcopy(initial), clients,
                {**config, "rounds": 2, "resume_from": str(path / "first" / "latest.pt")}, path / "resumed")
            for key, value in full.backbone.state_dict().items():
                torch.testing.assert_close(value, resumed.backbone.state_dict()[key], rtol=0, atol=0)
            for head, restored in zip(full.heads, resumed.heads):
                for key, value in head.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(full.history, resumed.history)

    def test_nonpersonalized_heads_aggregate_and_count_in_payload(self):
        _, _, clients, config = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            state = train_federated(TinyBackbone(), clients,
                {**config, "personalized": False, "topk_ratio": 1., "quant_bits": 32}, directory)
            for key, value in state.heads[0].state_dict().items():
                torch.testing.assert_close(value, state.heads[1].state_dict()[key])
            shared = sum(p.numel() for p in state.backbone.parameters())
            private = sum(p.numel() for p in state.heads[0].parameters())
            self.assertEqual(state.history[0]["paper_bits"], len(clients) * (shared + private) * 32)


if __name__ == "__main__":
    unittest.main()
