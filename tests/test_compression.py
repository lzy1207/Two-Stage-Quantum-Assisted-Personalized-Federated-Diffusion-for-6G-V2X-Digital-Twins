"""Compression round trips and exact Appendix B/Table III communication checks."""
import math
import unittest

import torch

from qv2x.compression import CompressedUpdate, compress_update


class CompressionTests(unittest.TestCase):
    def test_dense_float32_is_exact_without_mask(self):
        vector = torch.tensor([-.2, 0, 4.5, -8.0, .001], dtype=torch.float32)
        packet = compress_update(vector, ratio=1, bits=32)
        self.assertEqual(packet.paper_bits, vector.numel() * 32)
        self.assertEqual(packet.wire_bits, packet.paper_bits + 80)
        self.assertTrue(packet.dense)
        torch.testing.assert_close(packet.decompress(), vector, atol=0, rtol=0)
        restored = CompressedUpdate.from_bytes(packet.to_bytes())
        torch.testing.assert_close(restored.decompress(), vector, atol=0, rtol=0)

    def test_sparse_float32_keeps_largest_magnitudes(self):
        vector = torch.tensor([.1, -5, .3, 2, -.2])
        packet = compress_update(vector, ratio=.4, bits=32)
        self.assertEqual(packet.indices.tolist(), [1, 3])
        self.assertEqual(packet.paper_bits, 2 * 32 + 5)
        self.assertEqual(packet.wire_bits, 80 + 8 + 2 * 32)
        expected = torch.tensor([0, -5, 0, 2, 0], dtype=torch.float32)
        torch.testing.assert_close(packet.decompress(), expected, atol=0, rtol=0)
        torch.testing.assert_close(CompressedUpdate.from_bytes(packet.to_bytes()).decompress(), expected, atol=0, rtol=0)

    def test_int8_quantization_error_and_wire_roundtrip(self):
        vector = torch.linspace(-2.3, 4.7, 101)
        for ratio in (1, .37):
            packet = compress_update(vector, ratio=ratio, bits=8)
            self.assertEqual(packet.indices.numel(), math.ceil(ratio * vector.numel()))
            recovered = packet.decompress()
            error = (recovered[packet.indices] - vector[packet.indices]).abs().max().item()
            self.assertLessEqual(error, packet.scale / 2 + 2e-6)
            restored = CompressedUpdate.from_bytes(packet.to_bytes())
            self.assertEqual(restored.paper_bits, packet.paper_bits)
            self.assertEqual(restored.wire_bits, len(packet.to_bytes()) * 8)
            torch.testing.assert_close(restored.decompress(), recovered, atol=1e-6, rtol=2e-7)

    def test_all_zero_update_has_valid_scale_and_roundtrip(self):
        packet = compress_update(torch.zeros(13), ratio=.5, bits=8)
        self.assertEqual(packet.indices.numel(), 7)
        self.assertGreater(packet.scale, 0)
        restored = CompressedUpdate.from_bytes(packet.to_bytes())
        self.assertEqual(restored.decompress().count_nonzero().item(), 0)

    def test_table_iii_exact_paper_accounting(self):
        vector = torch.linspace(-1, 1, 728)
        cases = (
            (1.0, 32, 23296, .26208, 104.832),
            (1.0, 8, 5824, .06552, 26.208),
            (.5, 32, 12376, .13923, 55.692),
            (.5, 8, 3640, .04095, 16.380),
        )
        for ratio, bits, per_client, per_round_mb, total_mb in cases:
            packet = compress_update(vector, ratio=ratio, bits=bits)
            with self.subTest(ratio=ratio, bits=bits):
                self.assertEqual(packet.paper_bits, per_client)
                self.assertAlmostEqual(packet.paper_bits * 90 / 8 / 1e6, per_round_mb, places=10)
                self.assertAlmostEqual(packet.paper_bits * 90 * 400 / 8 / 1e6, total_mb, places=10)
                self.assertEqual(packet.wire_bits, packet.paper_bits + 80)
        self.assertAlmostEqual(1 - 3640 / 23296, .84375, places=12)

    def test_invalid_input_and_truncated_packets_fail(self):
        for vector, ratio, bits in ((torch.tensor([]), .5, 8),
                                    (torch.tensor([float("nan")]), .5, 8),
                                    (torch.ones(2), 0, 8),
                                    (torch.ones(2), 1.1, 8),
                                    (torch.ones(2), .5, 4)):
            with self.assertRaises(ValueError):
                compress_update(vector, ratio=ratio, bits=bits)
        packet = compress_update(torch.arange(10).float(), ratio=.5, bits=8)
        for encoded in (b"", packet.to_bytes()[:9], packet.to_bytes()[:-1], packet.to_bytes() + b"\x00"):
            with self.assertRaises(ValueError):
                CompressedUpdate.from_bytes(encoded)


if __name__ == "__main__":
    unittest.main()
