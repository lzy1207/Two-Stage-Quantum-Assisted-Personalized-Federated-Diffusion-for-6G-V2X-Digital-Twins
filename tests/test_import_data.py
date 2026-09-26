import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from qv2x.data import get_features, load_dataset
from qv2x.import_data import import_manifest


class ManifestImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, records):
        path = self.root / "manifest.csv"
        columns = list(dict.fromkeys(key for row in records for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        return path

    def row(self, **kwargs):
        return {"realization_id": "scene1_tx0", "map_path": "radio.npy", "tx_x": .25, "tx_y": .75, **kwargs}

    def test_db_npy_coordinates_and_semantic_masks_roundtrip(self):
        radio = np.arange(12, dtype=np.float32).reshape(3, 4) - 95
        radio[0, 1] = np.nan
        np.save(self.root / "radio.npy", radio)
        building = np.zeros((3, 4), dtype=np.uint8)
        building[1, 2] = 255
        Image.fromarray(building).save(self.root / "buildings.png")
        road = np.zeros((3, 4), dtype=np.float32)
        road[2] = 1
        np.save(self.root / "roads.npy", road)
        path = self.manifest([self.row(building_path="buildings.png", road_path="roads.npy")])
        imported = import_manifest(path, self.root / "out.npz")
        loaded = load_dataset(self.root / "out.npz")
        np.testing.assert_allclose(loaded.tx_xy, [[.25, .75]])
        np.testing.assert_equal(loaded.semantics.shape, (1, 2, 3, 4))
        self.assertFalse(loaded.valid_mask[0, 0, 1])
        self.assertFalse(loaded.valid_mask[0, 1, 2])
        self.assertEqual(loaded.semantics[0, 0, 1, 2], 1)
        np.testing.assert_equal(loaded.maps_db, imported.maps_db)
        self.assertEqual(loaded.metadata["realization_ids"], ["scene1_tx0"])

    def test_png_requires_declared_calibration(self):
        Image.fromarray(np.array([[0, 255], [128, 64]], dtype=np.uint8)).save(self.root / "radio.png")
        path = self.manifest([self.row(map_path="radio.png")])
        with self.assertRaisesRegex(ValueError, "explicit.*calibration"):
            import_manifest(path, self.root / "out.npz")
        self.assertFalse((self.root / "out.npz").exists())
        data = import_manifest(path, self.root / "out.npz", png_db_min=-120, png_db_max=-40)
        # The brightest pixel is not consulted to identify the transmitter.
        np.testing.assert_allclose(data.tx_xy, [[.25, .75]])
        self.assertEqual(data.maps_db[0, 0, 0], -120)
        self.assertEqual(data.maps_db[0, 0, 1], -40)
        np.testing.assert_allclose(data.maps_db[0, 1, 0], -120 + 128 / 255 * 80, atol=1e-5)

    def test_pathloss_and_pixel_coordinates(self):
        np.save(self.root / "radio.npy", np.full((3, 5), 100, dtype=np.float32))
        path = self.manifest([self.row(tx_x=4, tx_y=1)])
        with self.assertRaisesRegex(ValueError, "tx_power_dbm"):
            import_manifest(path, self.root / "out.npz", value_kind="pathloss", coordinates="pixel")
        data = import_manifest(path, self.root / "out.npz", value_kind="pathloss", tx_power_dbm=23, coordinates="pixel")
        np.testing.assert_equal(data.maps_db, -77)
        np.testing.assert_allclose(data.tx_xy, [[1, .5]])
        self.assertEqual(data.metadata["signal_unit"], "dBm")

    def test_explicit_six_features_and_semantics(self):
        np.save(self.root / "radio.npy", np.full((3, 4), -80, dtype=np.float32))
        features = np.arange(72, dtype=np.float32).reshape(6, 3, 4) / 72
        semantic = np.zeros((2, 3, 4), dtype=np.float32)
        np.save(self.root / "features.npy", features)
        np.save(self.root / "semantics.npy", semantic)
        path = self.manifest([self.row(features_path="features.npy", semantics_path="semantics.npy")])
        data = import_manifest(path, self.root / "out.npz")
        np.testing.assert_equal(data.features[0], features)
        np.testing.assert_equal(get_features(data)[0, ..., 4:], features.transpose(1, 2, 0))
        data.maps_db[:] += 27
        np.testing.assert_equal(get_features(data)[0, ..., 4:], features.transpose(1, 2, 0))

    def test_gain_values_preserved_or_shifted_by_explicit_power(self):
        np.save(self.root / "radio.npy", np.full((2, 2), -100, dtype=np.float32))
        path = self.manifest([self.row()])
        gain = import_manifest(path, self.root / "gain.npz", value_kind="gain")
        power = import_manifest(path, self.root / "power.npz", value_kind="gain", tx_power_dbm=30)
        np.testing.assert_equal(gain.maps_db, -100)
        np.testing.assert_equal(power.maps_db, -70)
        self.assertEqual(gain.metadata["signal_unit"], "dB")
        self.assertEqual(power.metadata["signal_unit"], "dBm")

    def test_boolean_geometry_and_fallback_features_do_not_use_radio_values(self):
        np.save(self.root / "radio.npy", np.arange(9, dtype=np.float32).reshape(3, 3) - 100)
        occupancy = np.zeros((3, 3), dtype=bool)
        occupancy[1, 1] = True
        np.save(self.root / "building.npy", occupancy)
        path = self.manifest([self.row(building_path="building.npy")])
        original = import_manifest(path, self.root / "first.npz")
        np.save(self.root / "radio.npy", np.full((3, 3), -45, dtype=np.float32))
        changed = import_manifest(path, self.root / "second.npz")
        np.testing.assert_equal(get_features(original), get_features(changed))
        np.testing.assert_equal(original.valid_mask, changed.valid_mask)
        np.testing.assert_equal(original.semantics, changed.semantics)
        self.assertFalse(changed.valid_mask[0, 1, 1])

    def test_duplicate_ids_and_partial_features_rejected(self):
        np.save(self.root / "radio.npy", np.full((2, 2), -100, dtype=np.float32))
        path = self.manifest([self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "unique"):
            import_manifest(path, self.root / "out.npz")
        path = self.manifest([self.row(features_path="features.npy"), self.row(realization_id="other")])
        with self.assertRaisesRegex(ValueError, "every realization"):
            import_manifest(path, self.root / "out.npz")

    def test_invalid_shapes_coords_and_rgb_maps_rejected(self):
        np.save(self.root / "radio.npy", np.zeros((2, 2), dtype=np.float32))
        path = self.manifest([self.row(tx_x=1.1)])
        with self.assertRaisesRegex(ValueError, "coordinate range"):
            import_manifest(path, self.root / "out.npz")
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(self.root / "rgb.png")
        path = self.manifest([self.row(map_path="rgb.png")])
        with self.assertRaisesRegex(ValueError, "mode L"):
            import_manifest(path, self.root / "out.npz", png_db_min=-120, png_db_max=-40)
        np.save(self.root / "other.npy", np.zeros((3, 2), dtype=np.float32))
        path = self.manifest([self.row(), self.row(realization_id="other", map_path="other.npy")])
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            import_manifest(path, self.root / "out.npz")


if __name__ == "__main__":
    unittest.main()
