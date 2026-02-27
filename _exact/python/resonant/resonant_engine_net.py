"""
WindingEngine: A pure Resonant-Winding Engine implementation.

"""

import random

from applications._core import GoldenExact, ResonantTensor


class ResonantLayer:
    """A single layer in the Resonant Engine."""

    def __init__(self, in_features, out_features, precision=100):
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision

        # Initialize weights and biases as ResonantTensors
        # Using Golden-Measure inspired initialization:
        # Small random integers in the lattice
        w_lattice = []
        for _ in range(out_features * in_features):
            a = random.randint(-2, 2)
            b = random.randint(-1, 1)
            w_lattice.append(GoldenExact.from_integers(a, b))

        b_lattice = [GoldenExact.from_integers(0, 0) for _ in range(out_features)]

        self.weight = ResonantTensor.from_golden_exact(
            w_lattice, [out_features, in_features]
        )
        self.bias = ResonantTensor.from_golden_exact(b_lattice, [out_features])

    def forward(self, x):
        """Forward pass using native Rust linalg."""
        # Y = X @ W^T
        x = x.matmul(self.weight)
        # Y = Y + b
        x.add_bias(self.bias)
        # Y = ReLU(Y)
        x.relu()
        return x


class WindingEngine:
    """
    The WindingEngine manages a hierarchy of ResonantLayers
    performing the DHSR cycle natively on the lattice.
    """

    def __init__(self, dims, precision=100):
        """
        Args:
            dims: List of layer dimensions, e.g., [2, 16, 2]
            precision: Lattice precision
        """
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(ResonantLayer(dims[i], dims[i + 1], precision=precision))
        self.precision = precision

    def forward(self, x, noise_scale=0.0):
        """
        Unified DHSR Cycle:
        1. Projection (D-phase via matmul/relu)
        2. Resonance (H-phase via batch_cpu_cycle)
        """
        for i, layer in enumerate(self.layers):
            # 1. Linear Projection + Activation (Flux logic inside Rust)
            x = layer.forward(x)

            # 2. Resonant Cycle (DHSR)
            # This applies noise (D) and snaps back to Lattice (H)
            if noise_scale > 0:
                x.run_batch_cpu_cycle(noise_scale, self.precision)

        return x

    def get_parameters(self):
        """Return all weights/biases as a flat list for RES optimization."""
        params = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)
        return params

    def set_parameters(self, new_params):
        """Update weights/biases from a list of tensors."""
        idx = 0
        for layer in self.layers:
            layer.weight = new_params[idx]
            layer.bias = new_params[idx + 1]
            idx += 2
