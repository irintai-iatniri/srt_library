"""
Resonant Parameter - Pure Rust-backed learnable parameter.

This module provides ResonantParameter, a wrapper around ResonantTensor
that provides a learnable parameter interface without PyTorch.

.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from ..._core import GoldenExact
from ..resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2


class ResonantParameter:
    """
    A learnable parameter backed by a ResonantTensor.

    The parameter stores its values in the exact Q(φ) lattice and provides
    methods for mutation (RES training) and access.
    """

    def __init__(
        self,
        data: List[float],
        shape: List[int],
        mode_norm_sq: Optional[List[float]] = None,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize ResonantParameter.

        Args:
            data: Initial parameter values (flattened)
            shape: Shape of the parameter tensor
            mode_norm_sq: Mode norms |n|² for each element
            precision: Exact arithmetic precision
            device: Device to store parameter on
        """
        self.precision = precision
        self.device = device
        self._shape = shape

        if mode_norm_sq is None:
            mode_norm_sq = [float(i**2) for i in range(len(data))]

        self._tensor = ResonantTensor(
            data=data,
            shape=shape,
            mode_norm_sq=mode_norm_sq,
            precision=precision,
            device=device,
        )

    @classmethod
    def from_tensor(cls, tensor: ResonantTensor) -> "ResonantParameter":
        """Create a ResonantParameter from an existing ResonantTensor."""
        param = cls.__new__(cls)
        param._tensor = tensor
        param._shape = list(tensor.shape)
        param.precision = tensor.precision
        param.device = tensor.device
        return param

    @property
    def tensor(self) -> ResonantTensor:
        """Access the underlying ResonantTensor."""
        return self._tensor

    @property
    def shape(self) -> List[int]:
        """Get the parameter shape."""
        return self._shape

    @property
    def syntony(self) -> float:
        """Get the current syntony of the parameter."""
        return self._tensor.syntony

    def to_floats(self) -> List[float]:
        """Get the parameter values as floats."""
        return self._tensor.to_floats()

    def to_lattice(self) -> List[GoldenExact]:
        """Get the parameter values as GoldenExact lattice points."""
        return self._tensor.to_lattice()

    def mutate(self, mutation_rate: float = 0.1, magnitude: int = 1):
        """
        Mutate the parameter for evolutionary training (RES).

        Args:
            mutation_rate: Probability of mutating each element
            magnitude: Maximum change in lattice coefficients
        """
        lattice = self._tensor.to_lattice()
        new_lattice = []

        for g in lattice:
            if random.random() < mutation_rate:
                da = random.randint(-magnitude, magnitude)
                db = random.randint(-magnitude, magnitude)
                new_g = g + GoldenExact.from_integers(da, db)
                new_lattice.append(new_g)
            else:
                new_lattice.append(g)

        self._tensor = ResonantTensor.from_golden_exact(new_lattice, self._shape).to(
            self.device
        )

    def clone(self) -> "ResonantParameter":
        """Create a copy of this parameter."""
        return ResonantParameter.from_tensor(
            ResonantTensor.from_golden_exact(self._tensor.to_lattice(), self._shape).to(
                self.device
            )
        )

    def cpu_cycle(self, noise_scale: float = 0.01):
        """Run a full DHSR cycle on the parameter."""
        self._tensor.cpu_cycle(noise_scale, self.precision)

    def to(self, device: str) -> "ResonantParameter":
        """Move parameter to device."""
        if self.device == device:
            return self

        self.device = device
        self._tensor = self._tensor.to(device)
        return self

    def cuda(self, device_id: int = 0) -> "ResonantParameter":
        """Move to CUDA."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "ResonantParameter":
        """Move to CPU."""
        return self.to("cpu")

    def __repr__(self):
        return f"ResonantParameter(shape={self._shape}, device={self.device}, syntony={self.syntony:.4f})"


if __name__ == "__main__":
    # Test the pure ResonantParameter
    print("Testing ResonantParameter...")

    data = [random.gauss(0, 1) for _ in range(16)]
    param = ResonantParameter(data, [4, 4])
    print(f"Parameter: {param}")
    print(f"Initial syntony: {param.syntony:.4f}")

    # Mutate
    param.mutate(mutation_rate=0.3, magnitude=1)
    print(f"After mutation: syntony={param.syntony:.4f}")

    # Clone
    cloned = param.clone()
    print(f"Cloned: {cloned}")

    print("SUCCESS")
