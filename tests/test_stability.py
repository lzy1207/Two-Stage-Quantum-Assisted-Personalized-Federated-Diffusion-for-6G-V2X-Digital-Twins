import json
import unittest

import numpy as np

from qv2x.stability import summarize_samples


class StabilityTests(unittest.TestCase):
    def test_identical_samples_have_zero_variance_and_exact_cdf(self):
        truth = np.array([[-90., -80.], [-70., -60.]])
        result = summarize_samples(np.stack([truth, truth]), truth, np.ones_like(truth, dtype=bool), 4)
        np.testing.assert_array_equal(result["mean_db"], truth)
        np.testing.assert_array_equal(result["std_db"], np.zeros_like(truth))
        np.testing.assert_array_equal(result["cdf_mean"], [.25, .5, .75, 1])
        np.testing.assert_array_equal(result["cdf_std"], [0, 0, 0, 0])
        self.assertEqual(result["rmse_mean_db"], 0)

    def test_all_seeds_count_and_ensemble_metric_is_distinct(self):
        truth = np.full((3, 3), -80.)
        predictions = np.stack([truth - 2, truth + 2])
        result = summarize_samples(predictions, truth, np.ones_like(truth, dtype=bool))
        self.assertEqual(result["rmse_mean_db"], 2)
        self.assertEqual(result["rmse_std_db"], 0)
        self.assertEqual(result["mean_map_rmse_db"], 0)
        np.testing.assert_allclose(result["std_db"], 2 * np.sqrt(2))

    def test_masked_pixels_do_not_affect_rmse_or_cdf(self):
        truth = np.full((2, 2), -80.)
        pred = truth.copy()
        pred[0, 0] = 1000
        valid = np.ones_like(truth, dtype=bool)
        valid[0, 0] = False
        result = summarize_samples(pred[None], truth, valid)
        self.assertEqual(result["rmse_mean_db"], 0)
        np.testing.assert_array_equal(result["cdf_mean"], np.ones(256))
        np.testing.assert_array_equal(result["cdf_x"], np.full(256, -80))
        scalars = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        json.dumps(scalars, allow_nan=False)

    def test_invalid_samples_fail(self):
        truth = np.zeros((2, 2))
        for predictions, valid in ((np.zeros((0, 2, 2)), np.ones((2, 2))),
                                   (np.zeros((2, 3, 2)), np.ones((2, 2))),
                                   (truth[None], np.zeros((2, 2))),
                                   (np.full((1, 2, 2), np.nan), np.ones((2, 2)))):
            with self.assertRaises(ValueError):
                summarize_samples(predictions, truth, valid)


if __name__ == "__main__":
    unittest.main()
