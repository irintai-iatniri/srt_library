"""
Winding Syntony Computer - Compute syntony with winding-aware mode norms.

This module computes syntony S(Ψ) using the winding state structure:

    S(Ψ) = Σᵢ |ψᵢ|² × exp(-|nᵢ|²/φ) / Σᵢ |ψᵢ|²

where |nᵢ|² is the mode norm squared for each feature.

 - Pure Python with ResonantTensor.
"""

from __future__ import annotations

import math
from typing import List

from ..._core import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618


class PureWindingSyntonyComputer:
    """
    Computes syntony with winding-aware mode structure.

    The syntony formula weights energy by golden measure:
        S(Ψ) = Σ |ψ_i|² w(nᵢ) / Σ |ψ_i|²

    where:
        w(n) = exp(-|n|²/φ)  # Golden weight

    High syntony (S → 1): Energy concentrated in low-norm modes
    Low syntony (S → 0): Energy scattered across high-norm modes

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> computer = PureWindingSyntonyComputer(dim=64)
        >>> x = ResonantTensor([0.1] * 64 * 32, [32, 64])
        >>> mode_norms = [float(i**2) for i in range(64)]
        >>> S = computer.forward(x, mode_norms)
        >>> print(f"Syntony: {S:.4f}")
    """

    def __init__(self, dim: int):
        """
        Initialize syntony computer.

        Args:
            dim: Feature dimension
        """
        self.dim = dim

    def forward(self, x: ResonantTensor, mode_norms: List[float]) -> float:
        """
        Compute syntony over batch.

        Args:
            x: Activations tensor (batch, dim) or (dim,)
            mode_norms: Mode norm squared |n|² for each feature (list of dim floats)

        Returns:
            Scalar syntony S ∈ [0, 1]
        """
        x_floats = x.to_floats()

        # Determine shape
        if len(x.shape) == 1:
            # 1D tensor
            batch_size = 1
            dim = len(x_floats)
        elif len(x.shape) == 2:
            # 2D tensor (batch, dim)
            batch_size, dim = x.shape
        else:
            raise ValueError(f"Unsupported shape: {x.shape}. Expected 1D or 2D.")

        # Golden weights: w(n) = exp(-|n|²/φ)
        weights = [math.exp(-norm / PHI) for norm in mode_norms]

        # Energy per feature: |ψᵢ|²
        weighted_energy_sum = 0.0
        total_energy_sum = 0.0

        for b in range(batch_size):
            for d in range(dim):
                if len(x.shape) == 1:
                    idx = d
                else:
                    idx = b * dim + d

                val = x_floats[idx]
                energy = val * val

                weighted_energy_sum += energy * weights[d]
                total_energy_sum += energy

        # Syntony formula
        syntony = weighted_energy_sum / (total_energy_sum + 1e-8)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, syntony))

    def batch_syntony(self, x: ResonantTensor, mode_norms: List[float]) -> List[float]:
        """
        Compute syntony for each sample in batch separately.

        Args:
            x: Activations tensor (batch, dim)
            mode_norms: Mode norm squared |n|² for each feature

        Returns:
            Syntony values for each sample (batch,)
        """
        if len(x.shape) != 2:
            raise ValueError(f"Expected 2D tensor, got shape {x.shape}")

        x_floats = x.to_floats()
        batch_size, dim = x.shape

        # Golden weights
        weights = [math.exp(-norm / PHI) for norm in mode_norms]

        syntonies = []
        for b in range(batch_size):
            weighted_energy = 0.0
            total_energy = 0.0

            for d in range(dim):
                idx = b * dim + d
                val = x_floats[idx]
                energy = val * val

                weighted_energy += energy * weights[d]
                total_energy += energy

            syntony = weighted_energy / (total_energy + 1e-8)
            syntonies.append(max(0.0, min(1.0, syntony)))

        return syntonies

    def __repr__(self) -> str:
        return f"PureWindingSyntonyComputer(dim={self.dim})"


if __name__ == "__main__":
    # Example usage
    from ..._core import ResonantTensor

    computer = PureWindingSyntonyComputer(dim=64)
    print(f"Syntony computer: {computer}")

    # Create sample data
    x = ResonantTensor([0.1] * 64 * 32, [32, 64])  # Batch of 32
    mode_norms = [float(i**2) for i in range(64)]  # |n|² = 0, 1, 4, 9, 16, ...

    # Compute syntony
    S = computer.forward(x, mode_norms)
    print(f"Batch syntony: {S:.4f}")

    # Per-sample syntony
    S_per_sample = computer.batch_syntony(x, mode_norms)
    print(f"Per-sample syntony (first 5): {S_per_sample[:5]}")
    mean_s = sum(S_per_sample) / len(S_per_sample)
    var_s = sum((s - mean_s) ** 2 for s in S_per_sample) / len(S_per_sample)
    std_s = var_s**0.5
    print(f"Mean: {mean_s:.4f}, Std: {std_s:.4f}")

    # Test with concentrated energy (low modes)
    x_concentrated = ResonantTensor([1.0] * 8 + [0.0] * 56, [64])
    S_high = computer.forward(x_concentrated, mode_norms)
    print(f"\nConcentrated energy syntony: {S_high:.4f}")

    # Test with scattered energy (high modes)
    x_scattered = ResonantTensor([0.0] * 56 + [1.0] * 8, [64])
    S_low = computer.forward(x_scattered, mode_norms)
    print(f"Scattered energy syntony: {S_low:.4f}")

    print("\nSUCCESS - PureWindingSyntonyComputer refactored!")
