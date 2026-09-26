"""End-to-end contracts: real stage artifacts, no-label inference and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from qv2x.pipeline import run, infer, ablation_config, condition_batch, verify_conditions


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).resolve().parents[1]/"configs/demo.json").read_text())
        cls.config["data"].update(count=6, size=8, client_grid=[2, 1], samples_per_client=8)
        cls.config["quantum"].update(n_qubits=2, n_layers=1, hidden_dim=8)
        cls.config["federated"].update(rounds=1, batch_size=4)
        cls.config["diffusion"].update(iterations=2, timesteps=4, batch_size=1,
                                      base_channels=4, channel_mults=[1, 2],
                                      embedding_dim=16, sample_steps=2, validate_every=1)
        cls.config["evaluation"]["radial_bins"] = 4

    def test_complete_run_saved_config_target_free_infer_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run(copy.deepcopy(self.config), root)
            saved_config = json.loads((root/"config.json").read_text())
            self.assertEqual(saved_config["schema_version"], 1)
            self.assertTrue(np.isfinite(result["stage2"]["rmse"]))
            for name in ("stage1.pt", "stage2_best.pt", "predictions.npz", "fl_rmse_vs_round.png"):
                self.assertTrue((root/name).exists(), name)
            query = root/"query.npz"
            # The file intentionally has no maps_db or target observations.
            np.savez(query, tx_xy=np.array([[.4, .6]], np.float32), semantics=np.zeros((1, 1, 8, 8), np.float32))
            infer(root, query, root/"out1.npz", seed=123)
            infer(root, query, root/"out2.npz", seed=123)
            with np.load(root/"out1.npz") as first, np.load(root/"out2.npz") as second:
                np.testing.assert_array_equal(first["prediction_db"], second["prediction_db"])
                self.assertEqual(first["prediction_db"].shape, (1, 8, 8))
            config = copy.deepcopy(self.config)
            config["diffusion"]["iterations"] = 3
            run(config, root, stage="stage2", resume=True)
            resumed = torch.load(root/"stage2_last.pt", weights_only=False)
            self.assertEqual(resumed["iteration"], 3)
            self.assertEqual(len(resumed["history"]), 3)
            run(config, root, stage="evaluate")
            # A changed cache cannot silently be paired with the trained model.
            with np.load(root/"conditions.npz") as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["coarse_db"][0, 0, 0] += 1.
            np.savez_compressed(root/"conditions.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "stale or modified"):
                verify_conditions(config, root)

    def test_staged_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = copy.deepcopy(self.config)
            c["diffusion"]["iterations"] = 3
            run(c, root/"full")
            run(copy.deepcopy(self.config), root/"split")
            run(c, root/"split", stage="stage2", resume=True)
            a = torch.load(root/"full/stage2_last.pt", weights_only=False)
            b = torch.load(root/"split/stage2_last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_ablation_configs_are_distinct_and_do_not_mutate_original(self):
        original = copy.deepcopy(self.config)
        self.assertEqual(ablation_config(original, "classical")["quantum"]["kind"], "classical")
        self.assertFalse(ablation_config(original, "no_personalization")["federated"]["personalized"])
        for name in ("confidence", "coarse", "heatmap", "prototype", "structural_losses"):
            self.assertFalse(ablation_config(original, "no_"+name)["ablation"][name])
        self.assertEqual(original, self.config)


if __name__ == "__main__":
    unittest.main()
