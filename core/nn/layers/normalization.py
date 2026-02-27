"""
Syntonic Normalization: Golden-ratio aware normalization layers.

These normalization layers incorporate golden ratio structure
for more natural regularization.

 - Pure Rust backend.

Source: CRT.md §7.1
"""

from __future__ import annotations

import math
from typing import Dict, List

from ..._core import ResonantTensor
from ..layers.resonant_parameter import ResonantParameter

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920


class SyntonicNorm:
    """
    Syntonic normalization layer.

    Similar to LayerNorm but with golden-ratio based scaling
    that promotes syntonic representations.

    The normalization target is φ - q (the syntony attractor)
    rather than zero-mean unit-variance.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> norm = SyntonicNorm(256)
        >>> data = [0.1] * 256 * 32
        >>> x = ResonantTensor(data, [32, 256])
        >>> y = norm(x)
        >>> y.shape
        [32, 256]
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        golden_target: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize syntonic normalization.

        Args:
            normalized_shape: Shape to normalize over (feature dimension)
            eps: Epsilon for numerical stability
            elementwise_affine: Learn affine parameters (gamma/beta)
            golden_target: Use golden-ratio based target variance (1/φ)
            device: Device placement
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.golden_target = golden_target
        self.device = device

        if elementwise_affine:
            # Initialize gamma (weight) to 1.0
            gamma_data = [1.0] * normalized_shape
            self.weight = ResonantParameter(
                gamma_data, [normalized_shape], device=device
            )

            # Initialize beta (bias) to 0.0
            beta_data = [0.0] * normalized_shape
            self.bias = ResonantParameter(beta_data, [normalized_shape], device=device)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply syntonic normalization.

        Uses golden-ratio based scaling for more natural regularization.

        Args:
            x: Input tensor of shape [..., normalized_shape]

        Returns:
            Normalized tensor with same shape as input
        """
        # Use layer_norm with golden_target
        if self.elementwise_affine:
            gamma = self.weight.tensor
            beta = self.bias.tensor
            return x.layer_norm(
                gamma=gamma, beta=beta, eps=self.eps, golden_target=self.golden_target
            )
        else:
            return x.layer_norm(eps=self.eps, golden_target=self.golden_target)

    def __repr__(self) -> str:
        return f"SyntonicNorm({self.normalized_shape}, eps={self.eps}, golden_target={self.golden_target})"


class GoldenNorm:
    """
    Golden ratio normalization.

    Normalizes activations using golden-ratio based statistics:
    - Target mean: 0
    - Target variance: 1/φ

    This creates representations that naturally align with
    syntonic theory predictions.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> norm = GoldenNorm(256)
        >>> data = [0.1] * 256 * 32
        >>> x = ResonantTensor(data, [32, 256])
        >>> y = norm(x)
        >>> # Variance should be approximately 1/φ ≈ 0.618
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize golden normalization.

        Args:
            num_features: Number of features
            momentum: Momentum for running stats
            eps: Epsilon for stability
            affine: Learn affine parameters
            track_running_stats: Track running mean/var
            device: Device placement
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = True  # Track training mode
        self.device = device

        # Target variance is 1/φ
        self.target_var = PHI_INV

        if affine:
            # Initialize weight to sqrt(1/φ) for target variance
            weight_data = [math.sqrt(self.target_var)] * num_features
            self.weight = ResonantParameter(weight_data, [num_features], device=device)

            # Initialize bias to 0
            bias_data = [0.0] * num_features
            self.bias = ResonantParameter(bias_data, [num_features], device=device)
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            # Pure Python state for running statistics
            self.running_mean: List[float] = [0.0] * num_features
            self.running_var: List[float] = [1.0] * num_features
            self.num_batches_tracked: int = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply golden normalization.

        Args:
            x: Input tensor of shape [batch, num_features]

        Returns:
            Normalized tensor with target variance 1/φ
        """
        if self.training or not self.track_running_stats:
            # Use batch statistics - compute mean and var from input
            floats = x.to_floats()
            batch_size = x.shape[0]

            # Compute mean across batch (dim=0)
            mean = [0.0] * self.num_features
            for i in range(batch_size):
                for j in range(self.num_features):
                    mean[j] += floats[i * self.num_features + j]
            mean = [m / batch_size for m in mean]

            # Compute variance across batch
            var = [0.0] * self.num_features
            for i in range(batch_size):
                for j in range(self.num_features):
                    val = floats[i * self.num_features + j]
                    diff = val - mean[j]
                    var[j] += diff * diff
            var = [v / batch_size for v in var]

            if self.track_running_stats:
                # Update running statistics
                for j in range(self.num_features):
                    self.running_mean[j] = (1 - self.momentum) * self.running_mean[
                        j
                    ] + self.momentum * mean[j]
                    self.running_var[j] = (1 - self.momentum) * self.running_var[
                        j
                    ] + self.momentum * var[j]
                self.num_batches_tracked += 1
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Create mean and var tensors for normalization
        # Broadcast mean/var to match input shape
        batch_size = x.shape[0]
        mean_data = []
        var_data = []
        for i in range(batch_size):
            mean_data.extend(mean)
            var_data.extend(var)

        mean_tensor = ResonantTensor(mean_data, x.shape)
        var_tensor = ResonantTensor(var_data, x.shape)

        # Normalize: (x - mean) / sqrt(var + eps)
        # Negate mean
        neg_mean_tensor = mean_tensor.negate()
        x_norm = x.elementwise_add(neg_mean_tensor)

        # Compute 1/sqrt(var + eps)
        floats_var = var_tensor.to_floats()
        inv_std_data = [1.0 / math.sqrt(v + self.eps) for v in floats_var]
        inv_std_tensor = ResonantTensor(inv_std_data, x.shape)

        x_norm = x_norm.elementwise_mul(inv_std_tensor)

        # Apply affine transform to reach target variance (1/φ)
        if self.affine:
            # Broadcast weight and bias to match input shape
            weight_data = self.weight.to_floats() * batch_size
            bias_data = self.bias.to_floats() * batch_size

            weight_tensor = ResonantTensor(weight_data, x.shape)
            bias_tensor = ResonantTensor(bias_data, x.shape)

            x_norm = x_norm.elementwise_mul(weight_tensor)
            x_norm = x_norm.elementwise_add(bias_tensor)

        return x_norm

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode

    def eval(self):
        """Set evaluation mode."""
        self.training = False

    def __repr__(self) -> str:
        return f"GoldenNorm({self.num_features}, target_var=1/φ≈{self.target_var:.4f}, momentum={self.momentum})"


class RecursionLayerNorm:
    """
    Layer normalization with recursion-aware statistics.

    Tracks how statistics evolve through recursion blocks
    and normalizes accordingly.

    Example:
        >>> from syntonic_applications._core import ResonantTensor
        >>> norm = RecursionLayerNorm(256, n_recursions=4)
        >>> data = [0.1] * 256 * 32
        >>> x = ResonantTensor(data, [32, 256])
        >>> y = norm(x, recursion_depth=2)
    """

    def __init__(
        self,
        num_features: int,
        n_recursions: int = 4,
        eps: float = 1e-5,
        device: str = "cpu",
    ):
        """
        Initialize recursion-aware layer norm.

        Args:
            num_features: Number of features
            n_recursions: Number of recursion depths to track
            eps: Epsilon for stability
            device: Device placement
        """
        self.num_features = num_features
        self.n_recursions = n_recursions
        self.eps = eps
        self.device = device

        # Separate parameters for each recursion depth (stored in dict)
        self.weights: Dict[int, ResonantParameter] = {}
        self.biases: Dict[int, ResonantParameter] = {}

        for i in range(n_recursions):
            # Initialize weights to 1.0
            weight_data = [1.0] * num_features
            self.weights[i] = ResonantParameter(
                weight_data, [num_features], device=device
            )

            # Initialize biases to 0.0
            bias_data = [0.0] * num_features
            self.biases[i] = ResonantParameter(bias_data, [num_features], device=device)

        # Golden decay factors for each depth: φ^(-i)
        self.decay_factors = [PHI ** (-i) for i in range(n_recursions)]

    def forward(self, x: ResonantTensor, recursion_depth: int = 0) -> ResonantTensor:
        """
        Apply recursion-aware normalization.

        Args:
            x: Input tensor of shape [..., num_features]
            recursion_depth: Current recursion depth (0-indexed)

        Returns:
            Normalized tensor with depth-specific parameters
        """
        depth = min(recursion_depth, self.n_recursions - 1)

        # Apply standard layer norm
        x_norm = x.layer_norm(eps=self.eps, golden_target=False)

        # Apply depth-specific parameters with golden decay
        decay = self.decay_factors[depth]

        # Get weight and bias for this depth, scale weight by decay
        weight_scaled = self.weights[depth].tensor.scalar_mul(decay)
        bias_floats = self.biases[depth].to_floats()

        # Broadcast to match input shape
        batch_size = x.shape[0]
        weight_broadcast = weight_scaled.to_floats() * batch_size
        bias_broadcast = bias_floats * batch_size

        weight_tensor = ResonantTensor(weight_broadcast, x.shape)
        bias_tensor = ResonantTensor(bias_broadcast, x.shape)

        # Apply affine: x_norm * weight * decay + bias
        x_norm = x_norm.elementwise_mul(weight_tensor)
        x_norm = x_norm.elementwise_add(bias_tensor)

        return x_norm

    def __repr__(self) -> str:
        return (
            f"RecursionLayerNorm({self.num_features}, n_recursions={self.n_recursions})"
        )


if __name__ == "__main__":
    # Test the pure normalization layers
    from ..._core import ResonantTensor

    print("Testing SyntonicNorm...")
    norm1 = SyntonicNorm(4, elementwise_affine=True, golden_target=True)
    x1 = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4])
    y1 = norm1.forward(x1)
    print(f"Input syntony: {x1.syntony:.4f}")
    print(f"Output syntony: {y1.syntony:.4f}")
    print(f"Output shape: {y1.shape}")

    print("\nTesting GoldenNorm...")
    norm2 = GoldenNorm(4, affine=True, track_running_stats=True)
    x2 = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4])
    y2 = norm2.forward(x2)
    print(f"Input syntony: {x2.syntony:.4f}")
    print(f"Output syntony: {y2.syntony:.4f}")
    print(f"Running mean: {norm2.running_mean}")
    print(f"Running var: {norm2.running_var}")

    print("\nTesting RecursionLayerNorm...")
    norm3 = RecursionLayerNorm(4, n_recursions=4)
    x3 = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4])
    y3 = norm3.forward(x3, recursion_depth=2)
    print(f"Input syntony: {x3.syntony:.4f}")
    print(f"Output syntony: {y3.syntony:.4f}")
    print(f"Decay factor at depth 2: {norm3.decay_factors[2]:.4f}")

    print("\nSUCCESS - All normalization layers refactored!")
