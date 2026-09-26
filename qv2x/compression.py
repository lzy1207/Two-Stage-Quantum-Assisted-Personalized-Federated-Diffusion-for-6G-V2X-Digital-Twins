"""Top-K bitmap and symmetric low-bit quantization, with honest bit accounting."""
from dataclasses import dataclass
import math
import struct
import numpy as np
import torch


@dataclass
class CompressedUpdate:
    size: int
    bits: int
    indices: torch.Tensor
    values: torch.Tensor
    scale: float
    dense: bool

    @property
    def paper_bits(self):
        return self.values.numel()*self.bits + (0 if self.dense else self.size)

    @property
    def wire_bits(self):
        return 8*len(self.to_bytes())

    def decompress(self, device="cpu"):
        values = self.values.to(device=device, dtype=torch.float32)
        if self.bits != 32:
            values = values*self.scale
        out = torch.zeros(self.size, device=device)
        out[self.indices.to(device)] = values
        return out

    def to_bytes(self):
        # 4-byte size, 1-byte precision, 1-byte dense flag, 4-byte quant scale.
        header = struct.pack("<IBBf", self.size, self.bits, self.dense, self.scale)
        mask = b""
        if not self.dense:
            bitmap = np.zeros(self.size, dtype=np.uint8)
            bitmap[self.indices.cpu().numpy()] = 1
            mask = np.packbits(bitmap, bitorder="little").tobytes()
        values = self.values.cpu().numpy()
        payload = values.astype("<f4" if self.bits == 32 else "i1").tobytes()
        return header+mask+payload

    @classmethod
    def from_bytes(cls, data):
        if len(data) < 10:
            raise ValueError("Truncated packet")
        size, bits, dense, scale = struct.unpack("<IBBf", data[:10])
        if bits not in (8, 32) or dense not in (0, 1) or size < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid packet header")
        offset = 10
        if dense:
            indices = np.arange(size)
        else:
            nbytes = math.ceil(size/8)
            if len(data) < offset+nbytes:
                raise ValueError("Truncated bitmap")
            mask = np.unpackbits(np.frombuffer(data[offset:offset+nbytes], dtype=np.uint8), bitorder="little")[:size]
            indices = np.flatnonzero(mask)
            offset += nbytes
        if len(data)-offset != len(indices)*(bits//8):
            raise ValueError("Invalid packet length")
        values = np.frombuffer(data[offset:], dtype="<f4" if bits == 32 else "i1").copy()
        return cls(size, bits, torch.from_numpy(indices).long(), torch.from_numpy(values), scale, bool(dense))


def compress_update(vector, ratio=.5, bits=8):
    if vector.ndim != 1 or vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError("Update must be a nonempty finite vector")
    if not 0 < ratio <= 1 or bits not in (8, 32):
        raise ValueError("ratio must be (0,1]; bits must be 8 or 32")
    flat = vector.detach().cpu().float()
    dense = ratio == 1
    count = math.ceil(flat.numel()*ratio)
    indices = torch.arange(flat.numel()) if dense else flat.abs().topk(count).indices.sort().values
    values = flat[indices]
    scale = 1.0
    if bits == 8:
        scale = max(float(values.abs().max())/127, 1e-12)
        values = (values/scale).round().clamp(-127, 127).to(torch.int8)
    return CompressedUpdate(flat.numel(), bits, indices, values, scale, dense)
