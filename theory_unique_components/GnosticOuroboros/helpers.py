"""
Helper functions for GnosticOuroboros pure Syntonic implementation.

Provides utility functions that replace PyTorch operations with
pure ResonantTensor equivalents.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from srt_library.core.nn.resonant_tensor import ResonantTensor


def compute_tensor_norm(tensor: "ResonantTensor") -> float:
    """
    Compute Frobenius norm of a tensor.

    Replaces torch.norm(tensor).

    Args:
        tensor: ResonantTensor to compute norm of

    Returns:
        Frobenius norm (sqrt of sum of squared elements)
    """
    values = tensor.to_floats()
    return math.sqrt(sum(x * x for x in values))


def tensor_argmax(values: List[float]) -> int:
    """
    Return index of maximum value in a list.

    Replaces torch.argmax(torch.tensor(values)).

    Args:
        values: List of float values

    Returns:
        Index of maximum value
    """
    if not values:
        return 0
    return values.index(max(values))


def create_identity_mask(n: int) -> List[List[bool]]:
    """
    Create n x n identity mask for off-diagonal selection.

    Replaces ~torch.eye(n, dtype=bool).

    Args:
        n: Size of the square matrix

    Returns:
        2D list where mask[i][j] is True if i != j (off-diagonal)
    """
    return [[i != j for j in range(n)] for i in range(n)]


def extract_column(tensor: "ResonantTensor", col: int) -> "ResonantTensor":
    """
    Extract single column from 2D tensor.

    Replaces tensor[:, col].unsqueeze(-1).

    Args:
        tensor: 2D ResonantTensor of shape [batch, features]
        col: Column index to extract

    Returns:
        ResonantTensor of shape [batch, 1]
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    data = tensor.to_floats()
    batch_size = tensor.shape[0]
    num_cols = tensor.shape[1] if len(tensor.shape) > 1 else 1

    col_data = [data[i * num_cols + col] for i in range(batch_size)]
    return ResonantTensor(col_data, [batch_size, 1])


def extract_columns(tensor: "ResonantTensor", cols: List[int]) -> "ResonantTensor":
    """
    Extract multiple columns from 2D tensor.

    Args:
        tensor: 2D ResonantTensor of shape [batch, features]
        cols: List of column indices to extract

    Returns:
        ResonantTensor of shape [batch, len(cols)]
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    data = tensor.to_floats()
    batch_size = tensor.shape[0]
    num_cols = tensor.shape[1] if len(tensor.shape) > 1 else 1

    result_data = []
    for i in range(batch_size):
        for col in cols:
            result_data.append(data[i * num_cols + col])

    return ResonantTensor(result_data, [batch_size, len(cols)])


def broadcast_multiply(
    tensor: "ResonantTensor",
    weights: "ResonantTensor",
    weight_col: int,
) -> "ResonantTensor":
    """
    Multiply tensor by a broadcast weight column.

    Replaces tensor * weights[:, col].unsqueeze(-1).

    Args:
        tensor: Input tensor of shape [batch, features]
        weights: Weight tensor of shape [batch, num_weights]
        weight_col: Column index in weights to use

    Returns:
        ResonantTensor with each row multiplied by corresponding weight
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    tensor_data = tensor.to_floats()
    weights_data = weights.to_floats()

    batch_size = tensor.shape[0]
    features = tensor.shape[1] if len(tensor.shape) > 1 else 1
    num_weight_cols = weights.shape[1] if len(weights.shape) > 1 else 1

    result_data = []
    for i in range(batch_size):
        w = weights_data[i * num_weight_cols + weight_col]
        for j in range(features):
            result_data.append(tensor_data[i * features + j] * w)

    return ResonantTensor(result_data, tensor.shape)


def tensor_clone(tensor: "ResonantTensor") -> "ResonantTensor":
    """
    Create a deep copy of a tensor.

    Replaces tensor.detach().clone().

    Args:
        tensor: ResonantTensor to clone

    Returns:
        New ResonantTensor with same data
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    return ResonantTensor(tensor.to_floats(), list(tensor.shape))


def randn_like(tensor: "ResonantTensor", scale: float = 1.0) -> "ResonantTensor":
    """
    Create random tensor with same shape as input.

    Replaces torch.randn_like(tensor) * scale.

    Args:
        tensor: Template tensor for shape
        scale: Standard deviation of normal distribution

    Returns:
        ResonantTensor with random values
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    size = 1
    for d in tensor.shape:
        size *= d

    data = [random.gauss(0, scale) for _ in range(size)]
    return ResonantTensor(data, list(tensor.shape))


def tensor_add_scalar(tensor: "ResonantTensor", scalar: float) -> "ResonantTensor":
    """
    Add scalar to all elements of tensor.

    Args:
        tensor: Input tensor
        scalar: Value to add

    Returns:
        New tensor with scalar added
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    data = [x + scalar for x in tensor.to_floats()]
    return ResonantTensor(data, list(tensor.shape))


def zeros_like(tensor: "ResonantTensor") -> "ResonantTensor":
    """
    Create zero tensor with same shape as input.

    Replaces torch.zeros_like(tensor).

    Args:
        tensor: Template tensor for shape

    Returns:
        ResonantTensor filled with zeros
    """
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    size = 1
    for d in tensor.shape:
        size *= d

    return ResonantTensor([0.0] * size, list(tensor.shape))


__all__ = [
    "compute_tensor_norm",
    "tensor_argmax",
    "create_identity_mask",
    "extract_column",
    "extract_columns",
    "broadcast_multiply",
    "tensor_clone",
    "randn_like",
    "tensor_add_scalar",
    "zeros_like",
]
