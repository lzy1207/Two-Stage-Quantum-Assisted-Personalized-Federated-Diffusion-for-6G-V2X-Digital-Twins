"""Differentiable state-vector simulator; no remote service or quantum SDK needed.

Qubit 0 is the most significant bit. Gates execute RY then RZ for input,
RX then RY then RZ for each layer, followed by the ordered CNOT ring.
The manuscript does not identify observables: we choose Z_i and Z_i Z_(i+1).
All observables are measured from the same computational-basis shots.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def _gate(state, angle, axis, qubit, n):
    batch = state.shape[0]
    angle = angle.expand(batch)
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    zero = torch.zeros_like(c)
    if axis == "X":
        mat = torch.stack((c, -1j*s, -1j*s, c), -1)
    elif axis == "Y":
        mat = torch.stack((c, -s, s, c), -1).to(state.dtype)
    else:
        mat = torch.stack((torch.exp(-.5j*angle), zero, zero, torch.exp(.5j*angle)), -1)
    mat = mat.reshape(batch, 2, 2).to(state.dtype)
    shaped = state.reshape(batch, *([2]*n)).movedim(qubit+1, -1)
    transformed = torch.bmm(shaped.reshape(batch, -1, 2), mat.transpose(1, 2))
    return transformed.reshape(shaped.shape).movedim(-1, qubit+1).reshape(batch, -1)


def _state(encoding, omega, entanglement):
    batch, n, _ = encoding.shape
    dtype = torch.complex128 if encoding.dtype == torch.float64 else torch.complex64
    state = torch.zeros(batch, 2**n, dtype=dtype, device=encoding.device)
    state[:, 0] = 1
    for q in range(n):
        state = _gate(state, encoding[:, q, 0], "Y", q, n)
        state = _gate(state, encoding[:, q, 1], "Z", q, n)
    indices = torch.arange(2**n, device=encoding.device)
    for layer in omega:
        for q in range(n):
            for k, axis in enumerate("XYZ"):
                state = _gate(state, layer[q, k], axis, q, n)
        if entanglement and n > 1:
            for q in range(n):
                target = (q+1) % n
                permutation = indices ^ (((indices >> (n-1-q)) & 1) << (n-1-target))
                state = state[:, permutation]
    return state


def _observables(n, device, dtype):
    indices = torch.arange(2**n, device=device)
    z = torch.stack([1-2*((indices >> (n-1-q)) & 1) for q in range(n)], -1)
    return torch.cat((z, z*z.roll(-1, dims=-1)), -1).to(dtype)


def _measure(encoding, omega, entanglement, shots):
    state = _state(encoding, omega, entanglement)
    probabilities = state.abs().square()
    signs = _observables(encoding.shape[1], encoding.device, encoding.dtype)
    exact = probabilities @ signs
    if shots:
        sampled = torch.multinomial(probabilities.detach(), shots, replacement=True)
        empirical = signs[sampled].mean(1)
        # Adjoint simulation uses the exact expectation Jacobian with a sampled
        # forward pass. Parameter-shift mode below instead shifts sampled circuits.
        return exact + (empirical-exact).detach()
    return exact


class _ParameterShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, omega, entanglement, shots):
        ctx.save_for_backward(angles, omega)
        ctx.entanglement, ctx.shots = entanglement, shots
        return _measure(angles.unsqueeze(-1).expand(-1, -1, 2), omega, entanglement, shots)

    @staticmethod
    def backward(ctx, upstream):
        angles, omega = ctx.saved_tensors
        encoding = angles.unsqueeze(-1).expand(-1, -1, 2).clone()
        grad_angles, grad_omega = torch.zeros_like(angles), torch.zeros_like(omega)
        if ctx.needs_input_grad[0]:
            # Each shared angle occurs in TWO gates; shift each occurrence separately.
            for q in range(angles.shape[1]):
                for gate in range(2):
                    plus, minus = encoding.clone(), encoding.clone()
                    plus[:, q, gate] += math.pi/2
                    minus[:, q, gate] -= math.pi/2
                    delta = (_measure(plus, omega, ctx.entanglement, ctx.shots)
                             - _measure(minus, omega, ctx.entanglement, ctx.shots))/2
                    grad_angles[:, q] += (upstream*delta).sum(-1)
        if ctx.needs_input_grad[1]:
            for i in range(omega.numel()):
                plus, minus = omega.clone(), omega.clone()
                plus.view(-1)[i] += math.pi/2
                minus.view(-1)[i] -= math.pi/2
                delta = (_measure(encoding, plus, ctx.entanglement, ctx.shots)
                         - _measure(encoding, minus, ctx.entanglement, ctx.shots))/2
                grad_omega.view(-1)[i] = (upstream*delta).sum()
        return grad_angles, grad_omega, None, None


class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4, shots=0, gradient_method="adjoint", entanglement=True):
        super().__init__()
        if not 1 <= n_qubits <= 16 or n_layers < 1 or shots < 0:
            raise ValueError("Require 1..16 qubits, positive layers, and nonnegative shots")
        if gradient_method not in ("adjoint", "parameter_shift"):
            raise ValueError("gradient_method must be adjoint or parameter_shift")
        self.n_qubits, self.shots = n_qubits, shots
        self.gradient_method, self.entanglement = gradient_method, entanglement
        self.omega = nn.Parameter(.1*torch.randn(n_layers, n_qubits, 3))

    def forward(self, angles):
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError("Expected angles [batch,n_qubits]")
        if self.gradient_method == "parameter_shift" and torch.is_grad_enabled():
            return _ParameterShift.apply(angles, self.omega, self.entanglement, self.shots)
        return _measure(angles.unsqueeze(-1).expand(-1, -1, 2), self.omega, self.entanglement, self.shots)


class QuantumBackbone(nn.Module):
    def __init__(self, input_dim=10, n_qubits=8, n_layers=4, hidden_dim=32, shots=0,
                 gradient_method="adjoint", entanglement=True):
        super().__init__()
        self.hidden_dim, self.feature_dim = hidden_dim, 2*n_qubits
        self.input_projection = nn.Linear(input_dim, n_qubits)
        self.circuit = QuantumCircuit(n_qubits, n_layers, shots, gradient_method, entanglement)
        self.output_projection = nn.Linear(self.feature_dim, hidden_dim)

    def quantum_features(self, x):
        return self.circuit(math.pi*torch.tanh(self.input_projection(x)))

    def forward(self, x):
        return torch.relu(self.output_projection(self.quantum_features(x)))

    def regularization(self):
        return self.circuit.omega.square().sum()


class ClassicalBackbone(nn.Module):
    """Explicit reconstruction ablation with exactly the same parameter count.

    Elementwise residual/neighbor mixing replaces the PQC. This is NOT an
    implementation of a separately published classical LDM or FL baseline.
    """
    def __init__(self, input_dim=10, n_qubits=8, n_layers=4, hidden_dim=32, **_):
        super().__init__()
        self.hidden_dim, self.feature_dim = hidden_dim, 2*n_qubits
        self.input_projection = nn.Linear(input_dim, n_qubits)
        self.mixing = nn.Parameter(.1*torch.randn(n_layers, n_qubits, 3))
        self.output_projection = nn.Linear(self.feature_dim, hidden_dim)

    def quantum_features(self, x):
        h = torch.tanh(self.input_projection(x))
        for layer in self.mixing:
            h = torch.tanh(h*(1+layer[:, 0])+layer[:, 1]+h.roll(1, -1)*layer[:, 2])
        return torch.cat((h, h.square()), -1)

    def forward(self, x):
        return torch.relu(self.output_projection(self.quantum_features(x)))

    def regularization(self):
        return self.mixing.square().sum()


def make_backbone(config):
    values = dict(config)
    kind = values.pop("kind", "quantum")
    if kind not in ("quantum", "classical"):
        raise ValueError(f"Unknown backbone {kind}")
    return (QuantumBackbone if kind == "quantum" else ClassicalBackbone)(**values)
