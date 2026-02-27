"""
GoldenGELU Activation Function

Theory-Correct GeLU: x * sigmoid(phi * x)

Represents winding probability of a token passing through
W⁴ aperture based on its energy state x.

Mathematical Formulation:
  GeLUφ(x) = x * σ(φ * x)

Where:
  - φ = 1.6180339887 (golden ratio)
  - σ(z) = 1 / (1 + e^(-z)) is sigmoid function
  - x is input tensor

This represents theory-correct GeLU where the scaling factor is
exactly the golden ratio φ, derived from SRT geometry.
"""

from typing import List, Union

from .._core import (
    batched_golden_gelu_forward,
    get_golden_gelu_phi,
    golden_gelu_backward,
    golden_gelu_forward,
)
from ..state import State


class GoldenGELU:
    """
    Theory-Correct GeLU: x * sigmoid(phi * x)

    Represents winding probability of a token passing through
    W⁴ aperture based on its energy state x.
    """

    def __init__(self, precision: int = 100):
        """
        Initialize GoldenGELU activation.

        Args:
            precision: Precision for sigmoid computation (default 100)
        """
        self.precision = precision

        # Get phi from Rust backend if available

        self.phi = get_golden_gelu_phi()

    def _get_phi_value(self) -> float:
        """Get phi value from Rust backend."""
        return get_golden_gelu_phi()

    def forward(
        self,
        x: Union[List[float], "State", None],
        batch_size: int = None,
        n_elements: int = None,
    ) -> Union[List[float], "State", None]:
        """
        Apply GoldenGELU activation: x * sigmoid(phi * x)

        Args:
            x: Input tensor or list of values
            batch_size: Number of tensors in batch (for batched operation)
            n_elements: Number of elements per tensor (for batched operation)

        Returns:
            GoldenGELU-activated values
        """
        # Handle State objects
        if x is None:
            return None

        if State is not None and isinstance(x, State):
            # Convert State to flat list
            flat_list = x.to_list()  # Assuming to_list() returns flat List[float]
            x = flat_list
        elif hasattr(x, "tolist"):
            x = x.tolist()
        elif not isinstance(x, list):
            x = list(x)

        # Use Rust backend exclusively
        if isinstance(x, list):
            if batch_size is not None and n_elements is not None:
                if len(x) != batch_size * n_elements:
                    raise ValueError(
                        f"x length ({len(x)}) must equal batch_size * n_elements ({batch_size} * {n_elements})"
                    )
                return batched_golden_gelu_forward(x, batch_size, n_elements)
            else:
                return golden_gelu_forward(x)

        # If not a list, this shouldn't happen after conversion, but raise error
        raise TypeError(f"Unsupported input type: {type(x)}")

    def backward(
        self,
        inputs: List[float],
        grad_outputs: List[float],
    ) -> List[float]:
        """
        Compute gradients for backpropagation.

        Derivative: d/dx [x * σ(φx)] = σ(φx) + φ * x * σ(φx) * (1 - σ(φx))

        Args:
            inputs: Original input values
            grad_outputs: Gradients from next layer

        Returns:
            Gradients w.r.t. input
        """
        if len(inputs) != len(grad_outputs):
            raise ValueError("inputs and grad_outputs must have same length")
        return golden_gelu_backward(inputs, grad_outputs)


def golden_gelu(
    x: Union[List[float], "State", None],
    precision: int = 100,
    batch_size: int = None,
    n_elements: int = None,
) -> Union[List[float], "State", None]:
    """
    Convenience function for GoldenGELU activation.

    Equivalent to:
        gelu = GoldenGELU(precision=precision)
        return gelu.forward(x, batch_size, n_elements)

    Args:
        x: Input tensor or list of values
        precision: Precision for sigmoid computation (default 100)
        batch_size: Number of tensors in batch (for batched operation)
        n_elements: Number of elements per tensor (for batched operation)

    Returns:
        GoldenGELU-activated values
    """
    gelu = GoldenGELU(precision=precision)
    return gelu.forward(x, batch_size, n_elements)


# Example usage
if __name__ == "__main__":
    # Test GoldenGELU activation
    gelu = GoldenGELU()

    # Forward pass
    inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    outputs = gelu.forward(inputs)

    print("GoldenGELU Activation Test")
    print("=" * 40)
    print(f"PHI = {gelu.phi:.12f}")
    print()
    print("Inputs:  ", [f"{x:8.4f}" for x in inputs])
    print("Outputs: ", [f"{x:8.4f}" for x in outputs])
    print()

    # Backward pass
    grad_outputs = [1.0] * len(inputs)  # dummy gradients
    grads = gelu.backward(inputs, grad_outputs)

    print("Gradients:", [f"{g:8.4f}" for g in grads])
    print()

    # Rust backend status
    print(f"PHI from Rust: {gelu._get_phi_value():.12f}")
