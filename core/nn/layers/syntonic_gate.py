"""
Syntonic Gate: Adaptive mixing based on local syntony.

Gate = σ(W_g·[x, H(D(x))])
Output = Gate · H(D(x)) + (1 - Gate) · x

 - Pure Rust backend.

Source: CRT.md §7.1
"""

from __future__ import annotations

import math
from typing import Optional

from ..._core import ResonantTensor
from ..layers.resonant_linear import ResonantLinear

PHI = (1 + math.sqrt(5)) / 2


class SyntonicGate:
    """
    Syntonic gating mechanism.

    Adaptively mixes input x with processed output x_harm based on
    how well the processing preserves/enhances syntony.

    Gate = σ(W_g · [x || x_harm])
    Output = Gate · x_harm + (1 - Gate) · x

    High gate → trust the processing (good syntony)
    Low gate → preserve input (processing degraded syntony)

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> gate = SyntonicGate(256)
        >>> data1 = [0.1] * 256 * 32
        >>> x = ResonantTensor(data1, [32, 256])
        >>> data2 = [0.2] * 256 * 32
        >>> x_processed = ResonantTensor(data2, [32, 256])
        >>> y = gate.forward(x, x_processed)
        >>> y.shape
        [32, 256]
    """

    def __init__(
        self, d_model: int, hidden_dim: Optional[int] = None, device: str = "cpu"
    ):
        """
        Initialize syntonic gate.

        Args:
            d_model: Model dimension
            hidden_dim: Hidden dimension for gate network (default: d_model)
            device: Device placement
        """
        hidden_dim = hidden_dim or d_model

        self.d_model = d_model
        self.device = device

        # Gate network: [x, x_processed] -> gate values
        # Input: d_model * 2, Hidden: hidden_dim, Output: d_model
        self.linear1 = ResonantLinear(d_model * 2, hidden_dim, bias=True, device=device)
        self.linear2 = ResonantLinear(hidden_dim, d_model, bias=True, device=device)

    def forward(self, x: ResonantTensor, x_processed: ResonantTensor) -> ResonantTensor:
        """
        Apply syntonic gating.

        Args:
            x: Original input of shape [..., d_model]
            x_processed: Processed output (e.g., H(D(x))) of shape [..., d_model]

        Returns:
            Gated output of shape [..., d_model]
        """
        # Concatenate x and x_processed along last dimension
        concat = ResonantTensor.concat([x, x_processed], dim=-1)

        # Gate network: Linear -> ReLU -> Linear -> Sigmoid
        gate = self.linear1.forward(concat)
        gate.relu()
        gate = self.linear2.forward(gate)
        gate.sigmoid(precision=100)

        # Compute (1 - gate)
        one_minus_gate = gate.one_minus()

        # Output = gate * x_processed + (1 - gate) * x
        term1 = gate.elementwise_mul(x_processed)
        term2 = one_minus_gate.elementwise_mul(x)
        output = term1.elementwise_add(term2)

        return output

    def get_gate_values(
        self, x: ResonantTensor, x_processed: ResonantTensor
    ) -> ResonantTensor:
        """Return gate values for analysis."""
        # Concatenate
        concat = ResonantTensor.concat([x, x_processed], dim=-1)

        # Compute gate
        gate = self.linear1.forward(concat)
        gate.relu()
        gate = self.linear2.forward(gate)
        gate.sigmoid(precision=100)

        return gate

    def __repr__(self) -> str:
        return f"SyntonicGate(d_model={self.d_model}, device={self.device})"


class AdaptiveGate:
    """
    Adaptive syntonic gate with learned syntony estimation.

    Estimates local syntony and uses it to modulate gating.
    More sophisticated than SyntonicGate.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> gate = AdaptiveGate(256)
        >>> x = ResonantTensor([0.1] * 256 * 32, [32, 256])
        >>> x_diff = ResonantTensor([0.2] * 256 * 32, [32, 256])
        >>> x_harm = ResonantTensor([0.3] * 256 * 32, [32, 256])
        >>> y, syntony = gate.forward(x, x_diff, x_harm, return_syntony=True)
    """

    def __init__(
        self,
        d_model: int,
        syntony_temp: float = 1.0,
        min_gate: float = 0.1,
        max_gate: float = 0.9,
        device: str = "cpu",
    ):
        """
        Initialize adaptive gate.

        Args:
            d_model: Model dimension
            syntony_temp: Temperature for syntony-based modulation
            min_gate: Minimum gate value (always some processing)
            max_gate: Maximum gate value (always some preservation)
            device: Device placement
        """
        self.d_model = d_model
        self.syntony_temp = syntony_temp
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.device = device

        # Gate network: [x, x_diff, x_harm] -> gate
        self.gate_linear1 = ResonantLinear(
            d_model * 3, d_model, bias=True, device=device
        )
        self.gate_linear2 = ResonantLinear(d_model, d_model, bias=True, device=device)

        # Syntony estimator: [x, x_diff, x_harm] -> syntony ∈ [0, 1]
        self.syntony_linear1 = ResonantLinear(
            d_model * 3, d_model // 2, bias=True, device=device
        )
        self.syntony_linear2 = ResonantLinear(d_model // 2, 1, bias=True, device=device)

    def forward(
        self,
        x: ResonantTensor,
        x_diff: ResonantTensor,
        x_harm: ResonantTensor,
        return_syntony: bool = False,
    ):
        """
        Apply adaptive gating.

        Args:
            x: Original input
            x_diff: After differentiation D(x)
            x_harm: After harmonization H(D(x))
            return_syntony: Return estimated syntony

        Returns:
            Gated output (and optionally mean syntony estimate)
        """
        # Concatenate all three stages along last dimension
        concat = ResonantTensor.concat([x, x_diff, x_harm], dim=-1)

        # Estimate local syntony: -> [batch, 1]
        syntony_hidden = self.syntony_linear1.forward(concat)
        syntony_hidden.tanh(precision=100)  # Use tanh instead of GELU
        syntony_est = self.syntony_linear2.forward(syntony_hidden)
        syntony_est.sigmoid(precision=100)

        # Compute base gate
        gate_hidden = self.gate_linear1.forward(concat)
        gate_hidden.tanh(precision=100)  # Use tanh instead of GELU
        base_gate = self.gate_linear2.forward(gate_hidden)
        base_gate.sigmoid(precision=100)

        # Modulate by syntony: high syntony → higher gate
        # syntony_est shape: [batch, 1], base_gate shape: [batch, d_model]
        # Broadcast syntony across d_model dimension
        syntony_floats = syntony_est.to_floats()
        base_gate_floats = base_gate.to_floats()

        # Determine batch size and feature dim from base_gate shape
        if len(base_gate.shape) == 2:
            batch_size, d_model = base_gate.shape
        else:
            # Fallback for 1D
            batch_size = len(base_gate_floats) // self.d_model
            d_model = self.d_model

        modulated_gate_data = []
        for b in range(batch_size):
            # Get syntony for this batch element
            s = syntony_floats[b]  # Single value per batch
            syntony_mod = (s - 0.5) * self.syntony_temp

            # Apply to all d_model features
            for f in range(d_model):
                idx = b * d_model + f
                base_val = base_gate_floats[idx]
                # sigmoid(syntony_mod + base_val)
                sig_val = 1.0 / (1.0 + math.exp(-(syntony_mod + base_val)))
                # gate = base * sigmoid_val
                g = base_val * sig_val
                # Clamp to [min, max]
                g = self.min_gate + (self.max_gate - self.min_gate) * g
                modulated_gate_data.append(g)

        gate = ResonantTensor(modulated_gate_data, base_gate.shape)

        # Compute (1 - gate)
        one_minus_gate = gate.one_minus()

        # Gated output: gate * x_harm + (1 - gate) * x
        term1 = gate.elementwise_mul(x_harm)
        term2 = one_minus_gate.elementwise_mul(x)
        output = term1.elementwise_add(term2)

        if return_syntony:
            # Compute mean syntony
            mean_syntony = sum(syntony_floats) / len(syntony_floats)
            return output, mean_syntony
        return output

    def __repr__(self) -> str:
        return f"AdaptiveGate(d_model={self.d_model}, temp={self.syntony_temp}, gate_range=[{self.min_gate}, {self.max_gate}])"


if __name__ == "__main__":
    # Test the pure SyntonicGate
    from ..._core import ResonantTensor

    print("Testing SyntonicGate...")
    gate = SyntonicGate(4, hidden_dim=8)
    print(f"Gate: {gate}")

    # Create inputs
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ResonantTensor(x_data, [2, 4])

    x_proc_data = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    x_processed = ResonantTensor(x_proc_data, [2, 4])

    # Forward pass
    y = gate.forward(x, x_processed)
    print(f"Input syntony: {x.syntony:.4f}")
    print(f"Processed syntony: {x_processed.syntony:.4f}")
    print(f"Output syntony: {y.syntony:.4f}")
    print(f"Output shape: {y.shape}")

    # Get gate values
    gate_vals = gate.get_gate_values(x, x_processed)
    print(
        f"Gate values (mean): {sum(gate_vals.to_floats()) / len(gate_vals.to_floats()):.4f}"
    )

    print("\nTesting AdaptiveGate...")
    adaptive_gate = AdaptiveGate(4, syntony_temp=1.0)
    print(f"Adaptive gate: {adaptive_gate}")

    x_diff = ResonantTensor([0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4], [2, 4])
    y_adaptive, syntony_est = adaptive_gate.forward(
        x, x_diff, x_processed, return_syntony=True
    )
    print(f"Adaptive output syntony: {y_adaptive.syntony:.4f}")
    print(f"Estimated syntony: {syntony_est:.4f}")

    print("\nSUCCESS - SyntonicGate and AdaptiveGate refactored!")
