"""
Pure Syntonic MLP: Multi-layer perceptron with DHSR structure.

Each layer follows the DHSR cycle:
1. Differentiation (complexity expansion)
2. Harmonization (coherence building)
3. Recursion (integration)

 - Pure Rust backend.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional

from srt_library.core.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    RecursionBlock,
    SyntonicNorm,
)
from srt_library.core.nn.layers.resonant_linear import ResonantLinear
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


from srt_library.core import sn as sn


class PureSyntonicLinear(sn.Module):
    """
    Linear layer with DHSR structure.

    Combines linear transformation with differentiation
    and harmonization for syntonic processing.

    Example:
        >>> from srt_library.core._core import ResonantTensor
        >>> layer = PureSyntonicLinear(256, 128)
        >>> x = ResonantTensor([0.1] * 256 * 32, [32, 256])
        >>> y = layer.forward(x)
        >>> print(f"Syntony: {layer.syntony:.4f}")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_recursion: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize syntonic linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            use_recursion: Use full recursion block
            dropout: Dropout probability (not implemented in pure version)
            bias: Include bias terms
            device: Device placement
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_recursion = use_recursion
        self.dropout_p = dropout
        self.device = device

        # Main linear transformation
        self.linear = ResonantLinear(
            in_features, out_features, bias=bias, device=device
        )

        # DHSR structure
        if use_recursion:
            self.recursion = RecursionBlock(out_features, device=device)
        else:
            self.diff = DifferentiationLayer(out_features, out_features, device=device)
            self.harm = HarmonizationLayer(out_features, out_features, device=device)

        self.norm = SyntonicNorm(out_features, device=device)

        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass with DHSR processing.

        Args:
            x: Input tensor (batch, in_features)

        Returns:
            Output tensor (batch, out_features)
        """
        # Linear transformation
        x = self.linear.forward(x)

        # DHSR processing
        if self.use_recursion:
            x, syntony = self.recursion.forward(x, return_syntony=True)
            self._syntony = syntony
        else:
            x_diff = self.diff.forward(x)
            x_harm = self.harm.forward(x_diff)
            self._syntony = self._compute_syntony(x, x_diff, x_harm)
            x = x_harm

        # Normalization
        x = self.norm.forward(x)

        # Dropout (pure implementation)
        if self.dropout_p > 0 and hasattr(self, "training") and self.training:
            x = self._apply_dropout(x)

        return x

    def _apply_dropout(self, x: ResonantTensor) -> ResonantTensor:
        """Apply dropout with probability dropout_p."""
        import random

        data = x.to_floats()
        # Create random mask: 1 if kept, 0 if dropped
        mask = [1.0 if random.random() > self.dropout_p else 0.0 for _ in data]
        # Scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - self.dropout_p)
        # Apply mask and scale
        dropped = [d * m * scale for d, m in zip(data, mask)]

        # Create new tensor preserving shape, mode norms, and precision
        return ResonantTensor(dropped, list(x.shape), x.get_mode_norms(), x.precision)

    def _compute_syntony(
        self,
        x: ResonantTensor,
        x_diff: ResonantTensor,
        x_harm: ResonantTensor,
    ) -> float:
        """Compute block syntony."""
        x_floats = x.to_floats()
        x_diff_floats = x_diff.to_floats()
        x_harm_floats = x_harm.to_floats()

        # Compute norms
        diff_norm_sq = sum(
            (x_diff_floats[i] - x_floats[i]) ** 2 for i in range(len(x_floats))
        )
        diff_norm = diff_norm_sq**0.5

        harm_diff_norm_sq = sum(
            (x_diff_floats[i] - x_harm_floats[i]) ** 2 for i in range(len(x_floats))
        )
        harm_diff_norm = harm_diff_norm_sq**0.5

        S = 1.0 - diff_norm / (harm_diff_norm + 1e-8)
        return max(0.0, min(1.0, S))

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, recursion={self.use_recursion}"


class PureSyntonicMLP(sn.Module):
    """
    Multi-layer perceptron with syntonic structure.

    Each hidden layer uses RecursionBlocks for DHSR processing.
    Tracks syntony throughout the network.

    Example:
        >>> from srt_library.core._core import ResonantTensor
        >>> model = PureSyntonicMLP(784, [512, 256, 128], 10)
        >>> x = ResonantTensor([0.1] * 784 * 32, [32, 784])
        >>> y = model.forward(x)
        >>> print(f"Model syntony: {model.syntony:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_recursion: bool = True,
        output_activation: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize syntonic MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            use_recursion: Use RecursionBlocks in hidden layers
            output_activation: Final activation ('sigmoid', None)
            device: Device placement
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.device = device

        # Build layers
        self.hidden_layers = sn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layer = PureSyntonicLinear(
                prev_dim,
                hidden_dim,
                use_recursion=use_recursion,
                dropout=dropout,
                device=device,
            )
            self.hidden_layers.append(layer)
            prev_dim = hidden_dim

        # Output layer (no recursion - just linear)
        self.output_layer = ResonantLinear(prev_dim, output_dim, device=device)

        # Track syntony history
        self._layer_syntonies: List[float] = []

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass through syntonic MLP.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Output tensor (batch, output_dim)
        """
        self._layer_syntonies = []

        # Hidden layers with DHSR
        for layer in self.hidden_layers:
            x = layer.forward(x)
            if layer.syntony is not None:
                self._layer_syntonies.append(layer.syntony)

        # Output layer
        x = self.output_layer.forward(x)

        # Output activation
        if self.output_activation == "sigmoid":
            x.sigmoid(precision=100)
        elif self.output_activation == "tanh":
            x.tanh(precision=100)
        # Note: softmax not implemented in pure version yet

        return x

    @property
    def syntony(self) -> float:
        """Get average model syntony."""
        if not self._layer_syntonies:
            return 0.5
        return sum(self._layer_syntonies) / len(self._layer_syntonies)

    @property
    def layer_syntonies(self) -> List[float]:
        """Get per-layer syntony values."""
        return self._layer_syntonies

    def get_intermediate_outputs(
        self,
        x: ResonantTensor,
    ) -> List[ResonantTensor]:
        """
        Get outputs from each layer for analysis.

        Args:
            x: Input tensor

        Returns:
            List of intermediate activations
        """
        outputs = [x]

        for layer in self.hidden_layers:
            x = layer.forward(x)
            outputs.append(x)

        x = self.output_layer.forward(x)
        outputs.append(x)

        return outputs

    def extra_repr(self) -> str:
        return f"input={self.input_dim}, hidden={self.hidden_dims}, output={self.output_dim}"


class PureDeepSyntonicMLP:
    """
    Deep syntonic MLP with skip connections.

    Uses residual connections scaled by golden ratio
    for stable deep training.

    Example:
        >>> from srt_library.core._core import ResonantTensor
        >>> model = PureDeepSyntonicMLP(784, 256, 10, depth=12)
        >>> x = ResonantTensor([0.1] * 784 * 32, [32, 784])
        >>> y = model.forward(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 6,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize deep syntonic MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension (constant)
            output_dim: Output dimension
            depth: Number of hidden layers
            dropout: Dropout probability
            device: Device placement
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.device = device

        self.input_proj = ResonantLinear(input_dim, hidden_dim, device=device)

        # Stack of recursion blocks with residuals
        # Note: Pure RecursionBlock doesn't support dropout parameter
        self.blocks = [RecursionBlock(hidden_dim, device=device) for _ in range(depth)]

        self.output_proj = ResonantLinear(hidden_dim, output_dim, device=device)
        self.norm = SyntonicNorm(hidden_dim, device=device)

        # Golden scaling for residuals
        self._residual_scale = 1.0 / PHI

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Forward pass with golden-scaled residuals."""
        # Project to hidden dimension
        x = self.input_proj.forward(x)

        # Stack of recursion blocks
        for block in self.blocks:
            # Save residual
            residual = x

            # Apply block (with syntony computation)
            x, _ = block.forward(x, return_syntony=True)

            # Golden-scaled residual: x + (1/φ) * residual
            scaled_residual = residual.scalar_mul(self._residual_scale)
            x = x.elementwise_add(scaled_residual)

        # Output
        x = self.norm.forward(x)
        x = self.output_proj.forward(x)

        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across blocks."""
        syntonies = [b.syntony for b in self.blocks if b.syntony is not None]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5

    def __repr__(self) -> str:
        return f"PureDeepSyntonicMLP(input={self.input_dim}, hidden={self.hidden_dim}, output={self.output_dim}, depth={self.depth})"


if __name__ == "__main__":
    # Test the pure syntonic MLP
    from srt_library.core._core import ResonantTensor

    print("=" * 60)
    print("Testing PureSyntonicLinear...")
    print("=" * 60)

    layer = PureSyntonicLinear(4, 8, use_recursion=True)
    print(f"Layer: {layer}")

    x = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4])
    print(f"Input shape: {x.shape}, syntony: {x.syntony:.4f}")

    y = layer.forward(x)
    print(f"Output shape: {y.shape}, syntony: {y.syntony:.4f}")
    if layer.syntony is not None:
        print(f"Layer syntony: {layer.syntony:.4f}")
    else:
        print("Layer syntony: Not computed")

    print("\n" + "=" * 60)
    print("Testing PureSyntonicMLP...")
    print("=" * 60)

    model = PureSyntonicMLP(4, [8, 6], 2, use_recursion=True)
    print(f"Model: {model}")

    x = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4])
    y = model.forward(x)
    print(f"Output shape: {y.shape}, syntony: {y.syntony:.4f}")
    print(f"Layer syntonies: {[f'{s:.4f}' for s in model.layer_syntonies]}")

    # Test intermediate outputs
    intermediates = model.get_intermediate_outputs(x)
    print(f"Number of intermediate outputs: {len(intermediates)}")
    print(f"Intermediate shapes: {[out.shape for out in intermediates]}")

    print("\n" + "=" * 60)
    print("Testing PureDeepSyntonicMLP...")
    print("=" * 60)

    deep_model = PureDeepSyntonicMLP(4, 6, 2, depth=3)
    print(f"Model: {deep_model}")

    y_deep = deep_model.forward(x)
    print(f"Output shape: {y_deep.shape}, syntony: {y_deep.syntony:.4f}")
    print(f"Model syntony: {deep_model.syntony:.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS - All pure MLP architectures working!")
    print("=" * 60)
