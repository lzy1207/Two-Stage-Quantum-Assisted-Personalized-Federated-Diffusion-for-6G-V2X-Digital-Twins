"""Numerical checks of actual unitary simulation and its two gradient modes."""
import math
import unittest

import torch

from qv2x.quantum import (
    ClassicalBackbone,
    QuantumBackbone,
    QuantumCircuit,
    _gate,
    _state,
)


class QuantumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1827)

    def test_default_shared_parameter_count_and_shapes(self):
        model = QuantumBackbone()
        classical = ClassicalBackbone()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 728)
        self.assertEqual(sum(p.numel() for p in classical.parameters()), 728)
        x = torch.randn(2, 10)
        self.assertEqual(model(x).shape, (2, 32))
        self.assertEqual(model.quantum_features(x).shape, (2, 16))
        self.assertEqual(classical.quantum_features(x).shape, (2, 16))

    def test_single_qubit_gates_preserve_analytic_states(self):
        zero = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
        theta = torch.tensor(math.pi, dtype=torch.float64)
        ry = _gate(zero, theta, "Y", 0, 1)
        rx = _gate(zero, theta, "X", 0, 1)
        rz = _gate(zero, theta, "Z", 0, 1)
        torch.testing.assert_close(ry, torch.tensor([[0, 1]], dtype=torch.complex128), atol=1e-12, rtol=0)
        torch.testing.assert_close(rx, torch.tensor([[0, -1j]], dtype=torch.complex128), atol=1e-12, rtol=0)
        torch.testing.assert_close(rz, torch.tensor([[-1j, 0]], dtype=torch.complex128), atol=1e-12, rtol=0)

    def test_full_circuit_preserves_norm(self):
        angles = torch.randn(5, 3, 2, dtype=torch.float64)
        omega = torch.randn(3, 3, 3, dtype=torch.float64)
        for entanglement in (True, False):
            state = _state(angles, omega, entanglement)
            torch.testing.assert_close(state.abs().square().sum(-1), torch.ones(5, dtype=torch.float64), atol=2e-12, rtol=0)

    def test_one_qubit_zero_angles_and_no_self_cnot(self):
        circuit = QuantumCircuit(n_qubits=1, n_layers=1, entanglement=True).double()
        with torch.no_grad():
            circuit.omega.zero_()
        angles = torch.tensor([[0.0], [math.pi / 3], [math.pi]], dtype=torch.float64)
        expected = torch.stack((angles[:, 0].cos(), torch.ones(3, dtype=torch.float64)), -1)
        # For the documented n=1 extension, ring ZZ is Z*Z=identity.
        torch.testing.assert_close(circuit(angles), expected, atol=1e-12, rtol=0)

    def test_parameter_shift_matches_adjoint_for_both_parameter_sets(self):
        exact = QuantumCircuit(n_qubits=2, n_layers=2, gradient_method="adjoint").double()
        shifted = QuantumCircuit(n_qubits=2, n_layers=2, gradient_method="parameter_shift").double()
        shifted.load_state_dict(exact.state_dict())
        xa = torch.tensor([[.31, -.72], [.62, .43]], dtype=torch.float64, requires_grad=True)
        xs = xa.detach().clone().requires_grad_()
        weights = torch.tensor([[.7, -.2, .4, .3], [.2, .1, -.8, .5]], dtype=torch.float64)
        ya, ys = exact(xa), shifted(xs)
        (ya * weights).sum().backward()
        (ys * weights).sum().backward()
        torch.testing.assert_close(ys, ya, atol=1e-12, rtol=1e-12)
        # This catches the incorrect single shift of an angle shared by RY/RZ.
        torch.testing.assert_close(xs.grad, xa.grad, atol=2e-11, rtol=2e-10)
        torch.testing.assert_close(shifted.omega.grad, exact.omega.grad, atol=2e-11, rtol=2e-10)

    def test_shared_input_angle_gradient_matches_finite_difference(self):
        circuit = QuantumCircuit(n_qubits=2, n_layers=1, gradient_method="parameter_shift").double()
        with torch.no_grad():
            circuit.omega.copy_(torch.tensor([[[.38, -.46, .23], [-.51, .37, .42]]], dtype=torch.float64))
        x = torch.tensor([[.41, -.67]], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([[.3, -.7, .2, .4]], dtype=torch.float64)
        (circuit(x) * weights).sum().backward()
        delta = 1e-6
        finite = torch.zeros_like(x)
        with torch.no_grad():
            for q in range(2):
                plus, minus = x.clone(), x.clone()
                plus[0, q] += delta
                minus[0, q] -= delta
                finite[0, q] = ((circuit(plus) - circuit(minus)) * weights).sum() / (2 * delta)
        torch.testing.assert_close(x.grad, finite, atol=2e-9, rtol=2e-8)

    def test_shot_mean_variance_and_discrete_range(self):
        shots, repetitions = 32, 2048
        circuit = QuantumCircuit(n_qubits=1, n_layers=1, shots=shots).double()
        with torch.no_grad():
            circuit.omega.zero_()
            estimates = circuit(torch.full((repetitions, 1), math.pi / 3, dtype=torch.float64))[:, 0]
        expected_mean, expected_var = .5, .75 / shots
        self.assertLess(abs(estimates.mean().item() - expected_mean), 6 * math.sqrt(expected_var / repetitions))
        self.assertLess(abs(estimates.var(unbiased=True).item() - expected_var), .16 * expected_var)
        self.assertTrue(torch.all((estimates >= -1 - 1e-12) & (estimates <= 1 + 1e-12)))
        levels = (estimates + 1) * shots / 2
        torch.testing.assert_close(levels, levels.round(), atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
