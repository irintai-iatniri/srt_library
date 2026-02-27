"""
Harmonization Layer: Ĥ operator in neural network form.

Ĥ[x] = x - σ(W_H·x + b_H) + tanh(W_S·x + b_S)

- Sigmoid (σ) damps excessive complexity
- Tanh stabilizes toward syntony projection
- Creates coherence and integration (Whispers)

 - Pure Rust backend.

Source: CRT.md §12.2
"""

from __future__ import annotations

from typing import Optional

from ... import sn
from ..._core import ResonantTensor
from ..layers.resonant_linear import ResonantLinear


class HarmonizationLayer:
    """
    Neural layer implementing harmonization Ĥ.

    Ĥ[x] = x - σ(W_H·x + b_H) + tanh(W_S·x + b_S)

    Two terms:
    1. -σ(W_H·x): Damping term (reduces excessive complexity)
    2. +tanh(W_S·x): Syntony projection (stabilizes toward coherence)

    Properties:
    - Reduces dissonance (damping)
    - Enhances coherence (syntony projection)
    - Creates stable, integrated representations (Whispers)

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> layer = HarmonizationLayer(256)
        >>> data = [0.1] * 256 * 32
        >>> x = ResonantTensor(data, [32, 256])
        >>> y = layer.forward(x)
        >>> y.shape
        [32, 256]
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        beta_scale: float = 1.0,
        gamma_scale: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize harmonization layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias terms
            beta_scale: Damping strength (β in Ĥ formula)
            gamma_scale: Syntony projection strength (γ)
            device: Device placement
        """
        out_features = out_features or in_features

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # Damping pathway: -β·σ(W_H·x + b_H)
        self.damping = ResonantLinear(
            in_features, out_features, bias=bias, device=device
        )
        self.beta_scale = beta_scale

        # Syntony projection: +γ·tanh(W_S·x + b_S)
        self.syntony_proj = ResonantLinear(
            in_features, out_features, bias=bias, device=device
        )
        self.gamma_scale = gamma_scale

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply Ĥ: x → x - β·σ(W_H·x) + γ·tanh(W_S·x)

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Harmonized output tensor
        """
        # Damping pathway: σ(W_H·x)
        damp = self.damping.forward(x)
        damp.sigmoid(precision=100)

        # Scale by beta
        if self.beta_scale != 1.0:
            damp = damp.scalar_mul(self.beta_scale)

        # Syntony projection pathway: tanh(W_S·x)
        syntony = self.syntony_proj.forward(x)
        syntony.tanh(precision=100)

        # Scale by gamma
        if self.gamma_scale != 1.0:
            syntony = syntony.scalar_mul(self.gamma_scale)

        # Harmonization: x - damping + syntony
        # Negate damping
        neg_damp = damp.negate()

        # Combine: x - damp + syntony
        if damp.shape == x.shape:
            result = x.elementwise_add(neg_damp)
            result = result.elementwise_add(syntony)
            return result
        else:
            # If dimensions change, combine damp and syntony only
            result = syntony.elementwise_add(neg_damp)
            return result

    def coherence_gain(self, x: ResonantTensor) -> float:
        """
        Measure coherence increase from harmonization.

        Coherence is measured as reduction in variance.

        Returns:
            Relative variance reduction: (var_before - var_after) / var_before
        """
        h_x = self.forward(x)

        # Compute variance before
        x_floats = x.to_floats()
        x_mean = sum(x_floats) / len(x_floats)
        x_var = sum((v - x_mean) ** 2 for v in x_floats) / len(x_floats)

        # Compute variance after
        h_floats = h_x.to_floats()
        h_mean = sum(h_floats) / len(h_floats)
        h_var = sum((v - h_mean) ** 2 for v in h_floats) / len(h_floats)

        return (x_var - h_var) / (x_var + 1e-8)

    def __repr__(self) -> str:
        return f"HarmonizationLayer(in_features={self.in_features}, out_features={self.out_features}, beta={self.beta_scale}, gamma={self.gamma_scale}, device={self.device})"


class HarmonizationModule(sn.Module):
    """
    Multi-component harmonization for transformer architectures.

    Combines multiple harmonization pathways analogous to
    Σᵢ βᵢ Q̂ᵢ[Ψ] + γ Ŝ_op[Ψ] in continuous Ĥ.

    Architecture:
    1. Multi-path Damping (Multiple β heads)
    2. Shared Syntony Projection (Single γ path)
    3. Output Projection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize Harmonization Module.

        Args:
            d_model: Model dimension
            n_heads: Number of damping heads
            dropout: Dropout probability
            device: Device placement
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device
        self._device = device  # sn.Module support

        # Multi-path Damping: -βᵢ · σ(W_H,i · x)
        # We implement this as n_heads independent linear projections
        # similar to attention heads
        self.damping_heads = sn.ModuleList(
            [
                ResonantLinear(d_model, self.head_dim, device=device)
                for _ in range(n_heads)
            ]
        )

        # Shared Syntony Projection: +γ · tanh(W_S · x)
        self.syntony_proj = ResonantLinear(d_model, d_model, device=device)

        # Output mixing
        self.output_proj = ResonantLinear(d_model, d_model, device=device)
        self.dropout = sn.Dropout(dropout)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply Multi-component Harmonization.
        """
        # 1. Multi-path Damping
        damping_outputs = []
        for head in self.damping_heads:
            h = head.forward(x)
            h.sigmoid(precision=100)  # Damping activation
            damping_outputs.append(h)

        # Concatenate heads [batch, n_heads * head_dim] -> [batch, d_model]
        damping_combined = ResonantTensor.concat(damping_outputs, dim=-1)

        # 2. Shared Syntony Projection
        syntony = self.syntony_proj.forward(x)
        syntony.tanh(precision=100)  # Syntony activation

        # 3. Combine: x - Damping + Syntony
        # Note: We subtract damping (reduction of dissonance) and add syntony (increase of coherence)

        # -Damping
        neg_damping = damping_combined.negate()

        # Combine
        combined = x.elementwise_add(neg_damping)
        combined = combined.elementwise_add(syntony)

        # 4. Output Projection and Dropout
        output = self.output_proj.forward(combined)
        output = self.dropout(output)

        return output


if __name__ == "__main__":
    # Test the pure HarmonizationLayer
    from ..._core import ResonantTensor

    print("Testing HarmonizationLayer...")
    layer = HarmonizationLayer(4, 4, bias=True, beta_scale=0.5, gamma_scale=1.0)
    print(f"Layer: {layer}")

    # Create input
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ResonantTensor(x_data, [2, 4])
    print(f"Input syntony: {x.syntony:.4f}")

    # Forward pass
    y = layer.forward(x)
    print(f"Output shape: {y.shape}")
    print(f"Output syntony: {y.syntony:.4f}")

    # Coherence gain
    coherence = layer.coherence_gain(x)
    print(f"Coherence gain: {coherence:.4f}")

    # Test with dimension change
    print("\nTesting dimension change (4 -> 8)...")
    layer2 = HarmonizationLayer(4, 8, bias=True)
    y2 = layer2.forward(x)
    print(f"Output shape: {y2.shape}")
    print(f"Output syntony: {y2.syntony:.4f}")

    print("\nSUCCESS - HarmonizationLayer refactored!")
