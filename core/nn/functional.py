"""
Syntonic Functional Operations — Pure SRT tensor utilities.

Provides tensor creation and operations that mirror common
functional APIs but produce ResonantTensors natively.

Uses Rust _core backend when available, with pure Python fallback.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2

# Try to import Rust backend
try:
    from srt_library.core._core import (
        py_arange as _rs_arange,
        py_linspace as _rs_linspace,
        py_eye as _rs_eye,
        py_stack as _rs_stack,
        py_silu as _rs_silu,
        py_golden_silu as _rs_golden_silu,
        py_where as _rs_where,
        py_einsum as _rs_einsum,
        py_linear as _rs_linear,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ── Tensor Creation ──────────────────────────────────────────────────

def arange(start: float, end: Optional[float] = None, step: float = 1.0, device: str = "cpu") -> ResonantTensor:
    """Create a 1D ResonantTensor with evenly spaced values."""
    if end is None:
        end = start
        start = 0.0
    if _HAS_RUST:
        vals = _rs_arange(float(start), float(end), float(step))
        return ResonantTensor(vals, [len(vals)], device=device)
    vals = []
    v = float(start)
    while (step > 0 and v < end) or (step < 0 and v > end):
        vals.append(v)
        v += step
    if not vals:
        vals = [0.0]
    return ResonantTensor(vals, [len(vals)], device=device)


def linspace(start: float, end: float, steps: int, device: str = "cpu") -> ResonantTensor:
    """Create a 1D ResonantTensor with linearly spaced values."""
    if _HAS_RUST:
        vals = _rs_linspace(float(start), float(end), steps)
        return ResonantTensor(vals, [steps] if steps > 1 else [1], device=device)
    if steps <= 1:
        return ResonantTensor([float(start)], [1], device=device)
    vals = []
    for i in range(steps):
        t = i / (steps - 1)
        vals.append(start + t * (end - start))
    return ResonantTensor(vals, [steps], device=device)


def eye(n: int, device: str = "cpu") -> ResonantTensor:
    """Create an n×n identity matrix as ResonantTensor."""
    if _HAS_RUST:
        data, shape = _rs_eye(n)
        return ResonantTensor(data, list(shape), device=device)
    data = []
    for i in range(n):
        for j in range(n):
            data.append(1.0 if i == j else 0.0)
    return ResonantTensor(data, [n, n], device=device)


def zeros(shape: List[int], device: str = "cpu") -> ResonantTensor:
    """Create a zero-filled ResonantTensor."""
    return ResonantTensor.zeros(shape, device=device)


def ones(shape: List[int], device: str = "cpu") -> ResonantTensor:
    """Create a ones-filled ResonantTensor."""
    return ResonantTensor.ones(shape, device=device)


def full(shape: List[int], fill_value: float, device: str = "cpu") -> ResonantTensor:
    """Create a ResonantTensor filled with a constant value."""
    size = 1
    for s in shape:
        size *= s
    return ResonantTensor([fill_value] * size, shape, device=device)


def stack(tensors: List[ResonantTensor], dim: int = 0) -> ResonantTensor:
    """
    Stack tensors along a new dimension.

    All tensors must have the same shape. Result gains one new dimension at `dim`.
    """
    if not tensors:
        raise ValueError("Cannot stack empty list")

    base_shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != base_shape:
            raise ValueError(f"Shape mismatch: {t.shape} vs {base_shape}")

    if _HAS_RUST:
        data_list = [t.to_floats() for t in tensors]
        shapes = [list(t.shape) for t in tensors]
        result_data, result_shape = _rs_stack(data_list, shapes, dim)
        return ResonantTensor(result_data, list(result_shape), device=tensors[0].device)

    n = len(tensors)
    new_shape = list(base_shape)
    new_shape.insert(dim, n)

    # For dim=0 (most common), just concatenate data
    if dim == 0:
        all_data = []
        for t in tensors:
            all_data.extend(t.to_floats())
        return ResonantTensor(all_data, new_shape, device=tensors[0].device)

    # General case: interleave data appropriately
    outer_size = 1
    for i in range(dim):
        outer_size *= base_shape[i]
    inner_size = 1
    for i in range(dim, len(base_shape)):
        inner_size *= base_shape[i]

    all_data = []
    for outer in range(outer_size):
        for t_idx in range(n):
            t_data = tensors[t_idx].to_floats()
            start = outer * inner_size
            all_data.extend(t_data[start:start + inner_size])

    return ResonantTensor(all_data, new_shape, device=tensors[0].device)


def cat(tensors: List[ResonantTensor], dim: int = 0) -> ResonantTensor:
    """
    Concatenate tensors along an existing dimension.
    """
    if not tensors:
        raise ValueError("Cannot concatenate empty list")

    # Simple case: dim=0, just concatenate
    if dim == 0:
        all_data = []
        total_first = 0
        for t in tensors:
            all_data.extend(t.to_floats())
            total_first += t.shape[0]
        new_shape = [total_first] + list(tensors[0].shape[1:])
        return ResonantTensor(all_data, new_shape, device=tensors[0].device)

    # General case
    shapes = [t.shape for t in tensors]
    new_shape = list(shapes[0])
    new_shape[dim] = sum(s[dim] for s in shapes)

    # Build output by iterating over outer dims, concatenating along dim
    outer_size = 1
    for i in range(dim):
        outer_size *= shapes[0][i]
    inner_size_per = []
    for t in tensors:
        isz = 1
        for i in range(dim, len(t.shape)):
            isz *= t.shape[i]
        inner_size_per.append(isz)

    all_data = []
    for outer in range(outer_size):
        for t_idx, t in enumerate(tensors):
            t_data = t.to_floats()
            start = outer * inner_size_per[t_idx]
            all_data.extend(t_data[start:start + inner_size_per[t_idx]])

    return ResonantTensor(all_data, new_shape, device=tensors[0].device)


# ── Operations ───────────────────────────────────────────────────────

def linear(input: ResonantTensor, weight: ResonantTensor, bias: Optional[ResonantTensor] = None) -> ResonantTensor:
    """
    Apply linear transformation: output = input @ weight^T + bias.
    """
    if _HAS_RUST:
        bias_data = bias.to_floats() if bias is not None else None
        result_data, result_shape = _rs_linear(
            input.to_floats(), weight.to_floats(), bias_data,
            list(input.shape), list(weight.shape),
        )
        return ResonantTensor(result_data, list(result_shape), device=input.device)
    output = input.matmul(weight)
    if bias is not None:
        bias_data = bias.to_floats()
        out_data = output.to_floats()
        out_shape = output.shape
        cols = out_shape[-1]
        for i in range(len(out_data)):
            out_data[i] += bias_data[i % cols]
        output = ResonantTensor(out_data, out_shape, device=output.device)
    return output


def silu(x: ResonantTensor) -> ResonantTensor:
    """
    SiLU activation: x * sigmoid(x).

    Self-gating aligns with syntonic self-reference in SRT theory.
    """
    if _HAS_RUST:
        return ResonantTensor(_rs_silu(x.to_floats()), list(x.shape), device=x.device)
    data = x.to_floats()
    result = []
    for v in data:
        sig = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, v))))
        result.append(v * sig)
    return ResonantTensor(result, x.shape, device=x.device)


def golden_silu(x: ResonantTensor) -> ResonantTensor:
    """
    Golden SiLU: x * sigmoid(φ*x).

    φ-scaled self-gating for enhanced syntonic resonance.
    """
    if _HAS_RUST:
        return ResonantTensor(_rs_golden_silu(x.to_floats()), list(x.shape), device=x.device)
    data = x.to_floats()
    result = []
    for v in data:
        sig = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, PHI * v))))
        result.append(v * sig)
    return ResonantTensor(result, x.shape, device=x.device)


def relu(x: ResonantTensor) -> ResonantTensor:
    """ReLU activation."""
    data = x.to_floats()
    return ResonantTensor([max(0.0, v) for v in data], x.shape, device=x.device)


def gelu(x: ResonantTensor) -> ResonantTensor:
    """GELU activation (approximate)."""
    data = x.to_floats()
    result = []
    for v in data:
        # Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inner = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)
        result.append(0.5 * v * (1.0 + math.tanh(inner)))
    return ResonantTensor(result, x.shape, device=x.device)


def sigmoid(x: ResonantTensor) -> ResonantTensor:
    """Sigmoid activation (returns new tensor, doesn't mutate)."""
    data = x.to_floats()
    result = []
    for v in data:
        result.append(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, v)))))
    return ResonantTensor(result, x.shape, device=x.device)


def softmax(x: ResonantTensor, dim: int = -1) -> ResonantTensor:
    """Softmax (returns new tensor)."""
    data = x.to_floats()[:]
    result = ResonantTensor(data, x.shape, device=x.device)
    result.softmax(dim=dim)
    return result


def where(condition: ResonantTensor, x: ResonantTensor, y: ResonantTensor) -> ResonantTensor:
    """
    Elementwise conditional: output[i] = x[i] if condition[i] > 0 else y[i].
    """
    if _HAS_RUST:
        return ResonantTensor(
            _rs_where(condition.to_floats(), x.to_floats(), y.to_floats()),
            list(x.shape), device=x.device,
        )
    cond = condition.to_floats()
    x_data = x.to_floats()
    y_data = y.to_floats()
    result = []
    for i in range(len(cond)):
        result.append(x_data[i] if cond[i] > 0 else y_data[i])
    return ResonantTensor(result, x.shape, device=x.device)


def einsum(equation: str, *operands: ResonantTensor) -> ResonantTensor:
    """
    General einsum supporting common patterns and arbitrary contractions.

    Fast paths: matmul (ij,jk->ik), batch matmul (bij,bjk->bik),
    matvec (ij,j->i), dot (i,i->), transpose (ij->ji), outer (i,j->ij),
    batch outer (bti,btj->bij), trace (ii->), diagonal (ii->i),
    row/col sums (ij->i, ij->j).

    Falls back to general contraction loop for other patterns.
    """
    if _HAS_RUST:
        data_list = [op.to_floats() for op in operands]
        shapes = [list(op.shape) for op in operands]
        result_data, result_shape = _rs_einsum(equation, data_list, shapes)
        return ResonantTensor(result_data, list(result_shape), device=operands[0].device)

    # Pure Python fallback — limited patterns
    parts = equation.replace(" ", "").split("->")
    inputs = parts[0].split(",")

    if len(operands) == 2 and len(inputs) == 2:
        a, b = operands
        li, lo = inputs[0], inputs[1]
        out = parts[1] if len(parts) > 1 else ""

        # matmul: ij,jk->ik
        if li == "ij" and lo == "jk" and out == "ik":
            return a.matmul(b.transpose(-2, -1))

        # matvec: ij,j->i
        if li == "ij" and lo == "j" and out == "i":
            b_col = b.view([b.shape[0], 1])
            result = a.matmul(b_col.transpose(-2, -1))
            return result.view([result.shape[0]])

        # dot: i,i->
        if li == "i" and lo == "i" and out == "":
            a_data = a.to_floats()
            b_data = b.to_floats()
            dot = sum(x * y for x, y in zip(a_data, b_data))
            return ResonantTensor([dot], [1], device=a.device)

    if len(operands) == 1 and len(inputs) == 1:
        a = operands[0]
        out = parts[1] if len(parts) > 1 else ""

        # transpose: ij->ji
        if inputs[0] == "ij" and out == "ji":
            return a.transpose(0, 1)

    raise NotImplementedError(f"einsum pattern '{equation}' not yet supported")


def clamp(x: ResonantTensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> ResonantTensor:
    """Clamp values to [min_val, max_val] range."""
    data = x.to_floats()
    result = []
    for v in data:
        if min_val is not None:
            v = max(v, min_val)
        if max_val is not None:
            v = min(v, max_val)
        result.append(v)
    return ResonantTensor(result, x.shape, device=x.device)


def dropout(x: ResonantTensor, p: float = 0.5, training: bool = True) -> ResonantTensor:
    """Apply dropout (pure Python, uses deterministic hash for reproducibility in non-training)."""
    if not training or p == 0.0:
        return x
    import random
    data = x.to_floats()
    scale = 1.0 / (1.0 - p)
    result = []
    for v in data:
        if random.random() < p:
            result.append(0.0)
        else:
            result.append(v * scale)
    return ResonantTensor(result, x.shape, device=x.device)


__all__ = [
    "arange",
    "linspace",
    "eye",
    "zeros",
    "ones",
    "full",
    "stack",
    "cat",
    "linear",
    "silu",
    "relu",
    "gelu",
    "sigmoid",
    "softmax",
    "where",
    "einsum",
    "clamp",
    "dropout",
]
