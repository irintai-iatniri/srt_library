"""
Recursion Block: R̂ = Ĥ ∘ D̂ in neural network form.

R_layer(x) = H_layer(D_layer(x))

Complete DHSR cycle as a single neural block.

 - Pure Rust backend.

Source: CRT.md §12.2
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from ... import sn
from ..._core import ResonantTensor
from ..layers.differentiation import DifferentiationLayer
from ..layers.harmonization import HarmonizationLayer
from ..layers.syntonic_gate import SyntonicGate


class RecursionBlock(sn.Module):
    """
    Complete DHSR recursion block.

    R̂[x] = Ĥ[D̂[x]]

    Implements one full cycle of:
    1. Differentiation (expand complexity)
    2. Harmonization (build coherence)
    3. Optional Syntonic Gating (adaptive mixing)

    This is the fundamental building block of syntonic networks.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> block = RecursionBlock(256, use_gate=True)
        >>> data = [0.1] * 256 * 32
        >>> x = ResonantTensor(data, [32, 256])
        >>> y, syntony = block.forward(x, return_syntony=True)
        >>> print(f"Block syntony: {syntony:.4f}")
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        use_gate: bool = False,
        alpha_scale: float = 1.0,
        beta_scale: float = 1.0,
        gamma_scale: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize recursion block.

        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension for D/H layers
            out_features: Output dimension
            use_gate: Whether to use syntonic gating (adaptive mixing)
            alpha_scale: Differentiation strength
            beta_scale: Damping strength
            gamma_scale: Syntony projection strength
            device: Device placement
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # D̂ operator
        self.differentiate = DifferentiationLayer(
            in_features, hidden_features, alpha_scale=alpha_scale, device=device
        )

        # Ĥ operator
        self.harmonize = HarmonizationLayer(
            hidden_features,
            out_features,
            beta_scale=beta_scale,
            gamma_scale=gamma_scale,
            device=device,
        )

        self.use_gate = use_gate
        if use_gate:
            self.gate = SyntonicGate(
                out_features, hidden_dim=hidden_features, device=device
            )

        # Track syntony for this block
        self._last_syntony = None
        self.device = device

    def forward(
        self,
        x: ResonantTensor,
        return_syntony: bool = False,
    ) -> Union[ResonantTensor, Tuple[ResonantTensor, float]]:
        """
        Apply R̂ = Ĥ ∘ D̂.

        Args:
            x: Input tensor
            return_syntony: Whether to compute and return block syntony

        Returns:
            Output tensor (and optionally syntony value)
        """
        # D̂: Differentiate (expand complexity)
        x_diff = self.differentiate.forward(x)

        # Ĥ: Harmonize (build coherence)
        x_harm = self.harmonize.forward(x_diff)

        # Gate or direct pass through
        if self.use_gate:
            x_out = self.gate.forward(x, x_harm)
        else:
            x_out = x_harm

        # Compute syntony if requested
        syntony = None
        if return_syntony:
            syntony = self._compute_block_syntony(x, x_diff, x_harm)
            self._last_syntony = syntony

        if return_syntony:
            return x_out, syntony
        return x_out

    def _compute_block_syntony(
        self,
        x: ResonantTensor,
        x_diff: ResonantTensor,
        x_harm: ResonantTensor,
    ) -> float:
        """
        Compute block-level syntony.

        S_block = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

        If dimensions change, use alternate formula:
        S_block = ||D(x)|| / (||D(x) - H(D(x))|| + ε)

        Derivation (from CRT.md §12.2):
        - Numerator ||D(x) - x||: measures differentiation magnitude
        - Denominator ||D(x) - H(D(x))||: measures harmonization effect
        - S → 1 when H perfectly integrates D's output
        - S → 0 when H has no effect (D output = H(D) output)

        Physical meaning:
        - High S: Network representations are coherent
        - Low S: Network representations are fragmented
        """
        # Get floats for computation
        x_floats = x.to_floats()
        x_diff_floats = x_diff.to_floats()
        x_harm_floats = x_harm.to_floats()

        # Check if dimensions match
        if len(x_floats) == len(x_diff_floats):
            # ||D(x) - x||
            diff_norm_sq = sum(
                (x_diff_floats[i] - x_floats[i]) ** 2 for i in range(len(x_floats))
            )
            diff_norm = diff_norm_sq**0.5
        else:
            # Dimensions changed - use ||D(x)|| instead
            diff_norm_sq = sum(x_diff_floats[i] ** 2 for i in range(len(x_diff_floats)))
            diff_norm = diff_norm_sq**0.5

        # ||D(x) - H(D(x))||
        harm_diff_norm_sq = sum(
            (x_diff_floats[i] - x_harm_floats[i]) ** 2
            for i in range(len(x_diff_floats))
        )
        harm_diff_norm = harm_diff_norm_sq**0.5

        # S = 1 - numerator / (denominator + ε)
        epsilon = 1e-8
        syntony = 1.0 - diff_norm / (harm_diff_norm + epsilon)

        return max(0.0, min(1.0, syntony))

    @property
    def syntony(self) -> Optional[float]:
        """Last computed block syntony."""
        return self._last_syntony

    def __repr__(self) -> str:
        return f"RecursionBlock(in={self.differentiate.in_features}, out={self.harmonize.out_features}, device={self.device})"


class DeepRecursionNet(sn.Module):
    """
    Deep network built from stacked RecursionBlocks.

    Implements n iterations of R̂, tracking syntony through layers.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> net = DeepRecursionNet(784, [512, 256, 128], 10)
        >>> data = [0.1] * 784 * 32
        >>> x = ResonantTensor(data, [32, 784])
        >>> y, syntonies = net.forward(x, return_syntonies=True)
        >>> print(f"Mean syntony: {net.mean_syntony:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_gates: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize deep recursion network.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            use_gates: Whether to use syntonic gating
            device: Device placement
        """
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.blocks = sn.ModuleList(
            [
                RecursionBlock(
                    dims[i], dims[i + 1], dims[i + 1], use_gate=use_gates, device=device
                )
                for i in range(len(dims) - 1)
            ]
        )

        self._layer_syntonies = []
        self.device = device

    def forward(
        self,
        x: ResonantTensor,
        return_syntonies: bool = False,
    ) -> Union[ResonantTensor, Tuple[ResonantTensor, List[float]]]:
        """Forward through all recursion blocks."""
        self._layer_syntonies = []

        for block in self.blocks:
            x, syntony = block.forward(x, return_syntony=True)
            self._layer_syntonies.append(syntony)

        if return_syntonies:
            return x, self._layer_syntonies
        return x

    @property
    def mean_syntony(self) -> float:
        """Mean syntony across all blocks."""
        if not self._layer_syntonies:
            return 0.0
        return sum(self._layer_syntonies) / len(self._layer_syntonies)

    @property
    def layer_syntonies(self) -> List[float]:
        """Syntony values for each layer."""
        return self._layer_syntonies

    def __repr__(self) -> str:
        return f"DeepRecursionNet(blocks={len(self.blocks)})"


if __name__ == "__main__":
    # Test the pure RecursionBlock
    from ..._core import ResonantTensor

    print("Testing RecursionBlock...")
    block = RecursionBlock(4, 4, 4, use_gate=False)
    print(f"Block: {block}")

    # Create input
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = ResonantTensor(x_data, [2, 4])
    print(f"Input syntony: {x.syntony:.4f}")

    # Forward pass
    y, syntony = block.forward(x, return_syntony=True)
    print(f"Output shape: {y.shape}")
    print(f"Output syntony: {y.syntony:.4f}")
    print(f"Block syntony: {syntony:.4f}")

    # Test DeepRecursionNet
    print("\nTesting DeepRecursionNet...")
    net = DeepRecursionNet(4, [8, 8], 4, use_gates=False)
    print(f"Net: {net}")

    y2, syntonies = net.forward(x, return_syntonies=True)
    print(f"Output shape: {y2.shape}")
    print(f"Output syntony: {y2.syntony:.4f}")
    print(f"Layer syntonies: {[f'{s:.4f}' for s in syntonies]}")
    print(f"Mean syntony: {net.mean_syntony:.4f}")

    print("\nSUCCESS - RecursionBlock and DeepRecursionNet refactored!")
