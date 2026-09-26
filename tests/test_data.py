import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from qv2x.data import (MapDataset, TargetScaler, get_features, load_dataset,
                       make_synthetic_dataset, partition_clients, save_dataset,
                       split_realizations)


class DataTests(unittest.TestCase):
    def test_npz_roundtrip_and_features_are_target_independent(self):
        data = make_synthetic_dataset(count=6, size=8, seed=3)
        expected = get_features(data)
        changed = MapDataset(data.maps_db + 1000, data.tx_xy, data.semantics)
        np.testing.assert_array_equal(expected, get_features(changed))
        self.assertEqual(expected.shape, (6, 8, 8, 10))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.npz"
            save_dataset(data, path)
            loaded = load_dataset(path)
            np.testing.assert_array_equal(loaded.maps_db, data.maps_db)
            np.testing.assert_array_equal(get_features(loaded), expected)
            self.assertEqual(loaded.metadata, data.metadata)

    def test_supplied_descriptors_take_precedence(self):
        data = make_synthetic_dataset(count=3, size=4)
        descriptor = np.full((3, 6, 4, 4), .123, dtype=np.float32)
        data = MapDataset(data.maps_db, data.tx_xy, features=descriptor)
        np.testing.assert_array_equal(get_features(data)[..., 4:], np.moveaxis(descriptor, 1, -1))

    def test_realization_splits_are_disjoint_exhaustive_reproducible(self):
        for n in (3, 4, 12, 100):
            splits = split_realizations(n, seed=5)
            self.assertEqual(sorted(np.concatenate(list(splits.values())).tolist()), list(range(n)))
            self.assertTrue(all(len(ids) for ids in splits.values()))
            self.assertFalse(set(splits["train"]) & set(splits["test"]))
            for name, ids in splits.items():
                np.testing.assert_array_equal(ids, split_realizations(n, seed=5)[name])

    def test_client_partition_keeps_transmitters_disjoint_and_records_unique(self):
        data = make_synthetic_dataset(count=10, size=12, seed=8)
        split = split_realizations(data.count, seed=2)
        clients = partition_clients(data, split["train"], split["val"], (3, 2), 40, .1, 9)
        self.assertEqual(len(clients), 6)
        observed = []
        for client in clients:
            self.assertGreater(client.count, 0)
            self.assertLessEqual(client.count, 40)
            torch.testing.assert_close(client.rx_xy, client.train_x[:, 2:4])
            for tx in client.train_x[:, :2].numpy():
                self.assertTrue(np.any(np.all(data.tx_xy[split["train"]] == tx, axis=1)))
                self.assertFalse(np.any(np.all(data.tx_xy[split["test"]] == tx, axis=1)))
            for tx in client.val_x[:, :2].numpy():
                self.assertTrue(np.any(np.all(data.tx_xy[split["val"]] == tx, axis=1)))
            observed.extend(map(tuple, client.train_x[:, :4].tolist()))
        self.assertEqual(len(set(observed)), len(observed))
        again = partition_clients(data, split["train"], split["val"], (3, 2), 40, .1, 9)
        for left, right in zip(clients, again):
            torch.testing.assert_close(left.train_x, right.train_x)
        with self.assertRaises(ValueError):
            partition_clients(data, split["train"], split["train"], (2, 2))

    def test_scaler_roundtrip_and_constant_data(self):
        values = torch.tensor([-90., -80., -70.])
        scaler = TargetScaler.fit(values)
        self.assertEqual(scaler.mean, -80.)
        torch.testing.assert_close(scaler.decode(scaler.encode(values)), values)
        self.assertGreater(TargetScaler.fit(np.ones(5)).std, 0)

    def test_partition_streams_maps_and_never_reads_test_targets(self):
        data = make_synthetic_dataset(count=8, size=12, seed=4)
        split = split_realizations(8, seed=7)
        with patch("qv2x.data.get_features", wraps=get_features) as feature_reader:
            clients = partition_clients(data, split["train"], split["val"], (2, 2), 20, .1, 3)
        self.assertEqual(feature_reader.call_count, len(split["train"]) + len(split["val"]))
        for call in feature_reader.call_args_list:
            self.assertEqual(len(call.args[1]), 1)
            self.assertNotIn(call.args[1][0], split["test"])
        changed = data.maps_db.copy()
        changed[split["test"]] += 1e6
        changed_data = MapDataset(changed, data.tx_xy, data.semantics)
        again = partition_clients(changed_data, split["train"], split["val"], (2, 2), 20, .1, 3)
        for first, second in zip(clients, again):
            torch.testing.assert_close(first.train_x, second.train_x)
            torch.testing.assert_close(first.train_y, second.train_y)
            torch.testing.assert_close(first.val_y, second.val_y)

    def test_schema_validation_and_masked_nonfinite(self):
        with self.assertRaises(ValueError):
            MapDataset(np.zeros((2, 4, 4)), np.array([[2, 0], [0, 0]]))
        maps = np.zeros((2, 4, 4), dtype=np.float32)
        maps[0, 0, 0] = np.nan
        data = MapDataset(maps, np.zeros((2, 2)))
        self.assertFalse(data.valid_mask[0, 0, 0])
        self.assertTrue(np.isfinite(data.maps_db).all())


if __name__ == "__main__":
    unittest.main()
