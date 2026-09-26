import unittest
import numpy as np
import torch

from qv2x.metrics import evaluate_maps


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        yy, xx = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16), indexing="ij")
        self.radii = np.hypot(xx - .5, yy - .5)
        self.truth = (-60 - 30 * self.radii)[None]
        self.valid = np.ones_like(self.truth, dtype=bool)

    def test_identity_and_monotone_trend(self):
        result = evaluate_maps(self.truth, self.truth, self.valid, [[.5, .5]], 100, radial_bins=5)
        self.assertEqual(result["rmse"], 0)
        self.assertAlmostEqual(result["ssim"], 1, places=10)
        self.assertEqual(result["radial_mae"], 0)
        self.assertEqual(result["trend_violation"], 0)

    def test_offset_errors_and_mean_power(self):
        result = evaluate_maps(self.truth + 3, self.truth, self.valid, [[.5, .5]], 100, radial_bins=5)
        self.assertAlmostEqual(result["rmse"], 3)
        self.assertAlmostEqual(result["radial_mae"], 3)
        self.assertAlmostEqual(result["mean_power_db"], (self.truth + 3).mean())

    def test_pooled_rmse_and_mask(self):
        truth = np.zeros((2, 4, 4))
        pred = truth.copy()
        pred[0] = 1
        pred[1] = 3
        valid = np.ones_like(truth, dtype=bool)
        valid[1, :2] = False
        result = evaluate_maps(pred, truth, valid, [[.5, .5], [.5, .5]], 10, radial_bins=2)
        self.assertAlmostEqual(result["rmse"], np.sqrt((16 + 8 * 9) / 24))

    def test_outward_increase_is_violation(self):
        pred = (-100 + 30 * self.radii)[None]
        result = evaluate_maps(pred, self.truth, self.valid, [[.5, .5]], 100, radial_bins=5)
        self.assertGreater(result["trend_violation"], 0)

    def test_empty_valid_map_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_maps(self.truth, self.truth, np.zeros_like(self.valid), [[.5, .5]], 100)

    def test_missing_radial_pairs_are_json_null(self):
        import json
        result = evaluate_maps(self.truth, self.truth, self.valid, [[.5, .5]], 100, radial_bins=1)
        self.assertIsNone(result["trend_violation"])
        self.assertIn('"trend_violation": null', json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
