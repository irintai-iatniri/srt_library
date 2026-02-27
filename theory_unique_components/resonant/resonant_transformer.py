"""
PureResonantTransformer: Library-Independent Transformer using Golden Cone Attention.

This module implements a Transformer-like architecture that:
1. Uses the 36 Golden Cone roots as fixed "attention heads".
2. Replaces deep layer stacking with recursive golden scaling.
3. Applies lattice quantization and hierarchical pruning for efficiency.

No PyTorch, NumPy, or external ML libraries required.
"""

import math

from srt_library.core._core import ResonantTensor

# Universal constants (from SRT axioms)
PHI = (1 + math.sqrt(5)) / 2
Q_SYNTONY = 0.027395146920  # Universal syntony deficit


class GoldenConeAttention:
    """
    Attention mechanism based on the 36 Golden Cone roots (Φ⁺(E₆)).

    Instead of learnable Q/K/V projections, this uses fixed geometric
    projections onto the E₆ subalgebra. Each "head" corresponds to
    a root direction in the cone.
    """

    def __init__(self, embed_dim: int, num_heads: int = 36, precision: int = 100):
        """
        Initialize Golden Cone Attention.

        Args:
            embed_dim: Input embedding dimension.
            num_heads: Number of attention heads (default 36 = |Φ⁺(E₆)|).
            precision: Lattice precision for GoldenExact arithmetic.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.precision = precision

        # Initialize projection weights using Golden Measure
        # Each head gets a projection vector based on cone geometry
        self._init_cone_projections()

    def _init_cone_projections(self):
        """Initialize the 36 cone projection vectors."""
        # 1. Input projection matrix (K) [num_heads, embed_dim]
        # X @ K^T -> Scores
        k_data = []

        # 2. Output projection matrix (V) [embed_dim, num_heads]
        # Scores @ V^T -> Output (if V is [embed, heads], V^T is [heads, embed])
        # Wait, ResonantTensor matmul is X @ W^T.
        # We want Scores [batch, heads] @ P [heads, embed].
        # So we need W such that W^T = P. => W = P^T.
        # W shape should be [embed_dim, num_heads].
        v_data_transposed = [0.0] * (self.embed_dim * self.num_heads)

        for h in range(self.num_heads):
            # Use Fibonacci-based initialization for each head
            # Weight = e^(-h²/φ) (Golden Measure decay)
            w = math.exp(-(h**2) / PHI)

            # Generate vector for head h
            head_vector = [
                w * math.cos(2 * math.pi * h * i / self.embed_dim)
                for i in range(self.embed_dim)
            ]

            # Add to K (rows are heads)
            k_data.extend(head_vector)

            # Add to V (transposed: rows are embedding dims)
            for i in range(self.embed_dim):
                v_data_transposed[i * self.num_heads + h] = head_vector[i]

        self.cone_k = ResonantTensor(
            k_data, [self.num_heads, self.embed_dim], precision=self.precision
        )

        self.cone_v = ResonantTensor(
            v_data_transposed,
            [self.embed_dim, self.num_heads],
            precision=self.precision,
        )

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply Golden Cone Attention to input tensor.

        Args:
            x: Input tensor of shape [batch, embed_dim].

        Returns:
            Output tensor with attended values [batch, embed_dim].
        """
        # 1. Compute Alignment Scores: A = X @ K^T
        # Shape: [batch, num_heads]
        scores = x.matmul(self.cone_k)

        # 2. Scale by Golden Ratio (Syntonic regularization)
        scores = scores.scalar_mul(1.0 / PHI)

        # 3. Softmax activation (Golden measure weighting)
        # Shape: [batch, num_heads]
        scores.softmax(self.precision)

        # 4. Reconstruct: Out = Scores @ V (Scores @ cone_v^T)
        # Shape: [batch, embed_dim]
        # Note: cone_v is [embed, heads], so cone_v^T is [heads, embed]
        out = scores.matmul(self.cone_v)

        return out


class RecursiveLayer:
    """
    A single "recursive" layer that replaces depth with golden scaling.

    Instead of stacking N linear layers, we apply the recursion map
    R(n) = floor(φ * n) repeatedly, simulating infinite depth with
    bounded memory.
    """

    def __init__(self, dim: int, num_iterations: int = 3, precision: int = 100):
        """
        Initialize Recursive Layer.

        Args:
            dim: Input/output dimension.
            num_iterations: Number of recursion applications.
            precision: Lattice precision.
        """
        self.dim = dim
        self.num_iterations = num_iterations
        self.precision = precision

        # Weight tensor (single shared weight)
        w_data = [math.exp(-i / PHI) for i in range(dim * dim)]
        self.weights = ResonantTensor(w_data, [dim, dim], precision=precision)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply recursive golden scaling.

        Args:
            x: Input tensor.

        Returns:
            Output after num_iterations of recursion.
        """
        out = x
        for _ in range(self.num_iterations):
            # Apply matmul
            out = out.matmul(self.weights)
            # Apply golden recursion (scale by φ)
            out.apply_recursion()
            # Prune small values (hierarchical sparsity)
            out.prune_hierarchy(Q_SYNTONY, 248.0)
            # ReLU activation
            out.relu()

        return out


class PureResonantTransformer:
    """
    Library-Independent Transformer using Golden Cone Attention
    and Recursive Layer stacking.

    Architecture:
    1. Embedding: Map input to golden lattice.
    2. Attention: Golden Cone projections (36 heads).
    3. FFN: Recursive Layer with hierarchical pruning.
    4. Output: Classification or regression head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        precision: int = 100,
    ):
        """
        Initialize the Pure Resonant Transformer.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (e.g., number of classes).
            num_layers: Number of "layers" (recursive iterations).
            precision: Lattice precision.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.precision = precision

        # Components
        self.attention = GoldenConeAttention(
            hidden_dim, num_heads=36, precision=precision
        )
        self.ffn = RecursiveLayer(
            hidden_dim, num_iterations=num_layers, precision=precision
        )

        # Input projection
        in_w = [
            math.exp(-abs(i - j) / PHI)
            for i in range(hidden_dim)
            for j in range(input_dim)
        ]
        self.input_proj = ResonantTensor(
            in_w, [hidden_dim, input_dim], precision=precision
        )

        # Output projection
        out_w = [1.0 / math.sqrt(hidden_dim) for _ in range(output_dim * hidden_dim)]
        self.output_proj = ResonantTensor(
            out_w, [output_dim, hidden_dim], precision=precision
        )

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass through the Pure Resonant Transformer.

        Args:
            x: Input tensor of shape [batch, input_dim].

        Returns:
            Output tensor of shape [batch, output_dim].
        """
        # 1. Input projection
        h = x.matmul(self.input_proj)

        # 2. Golden Cone Attention
        h = self.attention.forward(h)

        # 3. Recursive FFN
        h = self.ffn.forward(h)

        # 4. Output projection
        out = h.matmul(self.output_proj)

        return out

    def __repr__(self):
        return (
            f"PureResonantTransformer(input={self.input_dim}, "
            f"hidden={self.hidden_dim}, output={self.output_dim}, "
            f"layers={self.num_layers})"
        )


if __name__ == "__main__":
    print("Testing PureResonantTransformer...")

    # Create a simple model
    model = PureResonantTransformer(
        input_dim=4, hidden_dim=16, output_dim=2, num_layers=2
    )
    print(f"Model: {model}")

    # Create dummy input
    batch_data = [0.5, 0.3, -0.2, 0.8] * 2  # batch of 2
    x = ResonantTensor(batch_data, [2, 4])

    # Forward pass
    try:
        out = model.forward(x)
        print(f"Output shape: {out.shape}")
        print(f"Output: {out.to_floats()[:4]}...")
        print("SUCCESS: Forward pass completed.")
    except Exception as e:
        print(f"ERROR: {e}")
