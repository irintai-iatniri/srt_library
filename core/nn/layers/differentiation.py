"""
Differentiation Layer: D̂ operator in neural network form.

D̂[x] = x + ReLU(W_D·x + b_D)

- ReLU introduces non-linearity for complexity generation
- W_D weights serve as αᵢ coupling analogs
- Increases representational complexity (Fire/novelty)

 - Pure Rust backend.

Source: CRT.md §12.2
"""

from __future__ import annotations

from typing import Optional

from ..._core import ResonantTensor
from ..layers.resonant_linear import ResonantLinear


class DifferentiationLayer:
    """
    Neural layer implementing differentiation D̂.

    D̂[x] = x + ReLU(W_D·x + b_D)

    The ReLU introduces non-linearity that generates complexity,
    analogous to D̂ exploring possibility spaces.

    Properties:
    - Increases dimensionality of representational manifold
    - Generates distinctions (Fire)
    - W_D weights control coupling strength to possibility projectors

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> layer = DifferentiationLayer(256)
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
        alpha_scale: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize differentiation layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension (defaults to in_features)
            bias: Include bias term
            alpha_scale: Scaling factor for differentiation strength
            device: Device placement
        """
        out_features = out_features or in_features

        self.in_features = in_features
        self.out_features = out_features
        self.linear = ResonantLinear(
            in_features, out_features, bias=bias, device=device
        )
        self.alpha_scale = alpha_scale
        self.device = device

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply D̂: x → x + α·ReLU(W_D·x + b_D)

        The residual connection preserves input while adding complexity.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor with differentiation applied
        """
        # Linear transformation
        d_x = self.linear.forward(x)

        # ReLU activation (in-place)
        d_x.relu()

        # Scale by alpha
        if self.alpha_scale != 1.0:
            d_x = d_x.scalar_mul(self.alpha_scale)

        # Residual: D̂[x] = x + differentiation
        if d_x.shape == x.shape:
            return x.elementwise_add(d_x)
        else:
            # If dimensions change, use projection only
            return d_x

    def complexity_increase(self, x: ResonantTensor) -> float:
        """
        Measure how much complexity D̂ added.

        Returns:
            Relative increase in norm: ||D̂[x] - x|| / ||x||
        """
        d_x = self.forward(x)

        # Compute norms
        x_floats = x.to_floats()
        d_x_floats = d_x.to_floats()

        # Compute ||d_x - x||
        diff_norm_sq = sum(
            (d_x_floats[i] - x_floats[i]) ** 2 for i in range(len(x_floats))
        )
        diff_norm = diff_norm_sq**0.5

        # Compute ||x||
        x_norm_sq = sum(x_floats[i] ** 2 for i in range(len(x_floats)))
        x_norm = x_norm_sq**0.5

        return diff_norm / (x_norm + 1e-8)

    def __repr__(self) -> str:
        return f"DifferentiationLayer(in_features={self.in_features}, out_features={self.out_features}, alpha_scale={self.alpha_scale}, device={self.device})"


class DifferentiationModule:
    """
    Multi-head differentiation for transformer architectures.

    Applies differentiation with multiple "possibility projectors",
    analogous to Σᵢ αᵢ P̂ᵢ[Ψ] in the continuous D̂ operator.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize multi-head differentiation module.

        Args:
            d_model: Model dimension
            n_heads: Number of differentiation heads
            dropout: Dropout probability
            device: Device placement
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device

        if self.head_dim * n_heads != d_model:
            raise ValueError(
                f"d_model {d_model} must be divisible by n_heads {n_heads}"
            )

        # Multi-head projections (possibility spaces)
        self.projectors = [
            ResonantLinear(d_model, self.head_dim, device=device)
            for _ in range(n_heads)
        ]

        # Recombine projection
        self.out_proj = ResonantLinear(d_model, d_model, device=device)
        self.dropout_p = dropout

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Multi-head differentiation forward pass.

        1. Project input into N possibility spaces (heads)
        2. Apply nonlinear complexity generation (ReLU) in each space
        3. Concatenate and project back to global space
        4. Apply residual connection and normalization

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor
        """
        # 1. & 2. Apply each projector and ReLU
        heads = []
        for proj in self.projectors:
            h = proj.forward(x)
            h.relu()
            heads.append(h)

        # 3. Concatenate heads along feature dimension (last dim)
        # Note: ResonantTensor.concat expects list of tensors and dimension
        concat = ResonantTensor.concat(heads, dim=-1)

        # Project back
        out = self.out_proj.forward(concat)

        # Dropout
        if self.dropout_p > 0:
            out.dropout(self.dropout_p)

        # 4. Residual + Norm
        # x + out
        res = x.elementwise_add(out)

        # Layer Norm (using default behavior for now)
        return res.layer_norm()


if __name__ == "__main__":
    # Test the pure DifferentiationLayer
    from ..._core import ResonantTensor

    print("Testing DifferentiationLayer...")
    layer = DifferentiationLayer(4, 4, bias=True, alpha_scale=0.5)
    print(f"Layer: {layer}")

    # Create input
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ResonantTensor(x_data, [2, 4])
    print(f"Input syntony: {x.syntony:.4f}")

    # Forward pass
    y = layer.forward(x)
    print(f"Output shape: {y.shape}")
    print(f"Output syntony: {y.syntony:.4f}")

    # Complexity increase
    complexity = layer.complexity_increase(x)
    print(f"Complexity increase: {complexity:.4f}")

    # Test with dimension change
    print("\nTesting dimension change (4 -> 8)...")
    layer2 = DifferentiationLayer(4, 8, bias=True)
    y2 = layer2.forward(x)
    print(f"Output shape: {y2.shape}")
    print(f"Output syntony: {y2.syntony:.4f}")

    # Test DifferentiationModule
    print("\nTesting DifferentiationModule...")
    try:
        d_model = 16
        n_heads = 4
        diff_mod = DifferentiationModule(d_model, n_heads, dropout=0.1)

        # improved test data - random gaussian-like
        # [batch=2, seq=4, d_model=16] -> total 128 elements
        dm_data = [float(i % 10) / 10.0 for i in range(128)]
        x_mod = ResonantTensor(dm_data, [2, 4, d_model])

        y_mod = diff_mod.forward(x_mod)
        print(f"Module Input shape: {x_mod.shape}")
        print(f"Module Output shape: {y_mod.shape}")

        assert y_mod.shape == x_mod.shape
        print("DifferentiationModule works!")

    except Exception as e:
        print(f"DifferentiationModule failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nSUCCESS - DifferentiationLayer refactored!")
