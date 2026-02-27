"""
Syntonic Network (sn) - Pure Python neural network base classes.

Replaces torch.nn with pure Python/ResonantTensor-based implementations.

This module provides:
- Module: Base class for all syntonic network components
- Parameter: Learnable parameter wrapper
- Sequential: Container for sequential layers
- ModuleList: Container for a list of modules
"""

from __future__ import annotations

from applications import srt_math
from applications._core import Rational
import random
from typing import Any, Dict, Iterator, List, Optional

from applications.nn.resonant_tensor import ResonantTensor

PHI = (1 + srt_math.sqrt(5)) / 2


class Parameter:
    """
    Learnable parameter for syntonic networks.

    Wraps a ResonantTensor as a trainable parameter.

    Mode Norm Theory:
    -----------------
    Parameters use 1D flattened sequential mode norms: mode_norm[i] = i².
    This reflects the recursion hierarchy where earlier parameters (low i)
    are more fundamental than later parameters (high i).

    For a weight matrix [out_features, in_features]:
      - W[0,0] → index 0 → mode_norm = 0 (most fundamental)
      - W[0,1] → index 1 → mode_norm = 1
      - W[1,0] → index 2 → mode_norm = 4
      - etc.

    This is the canonical SRT prescription for parameter tensors.
    See: theory/resonant_engine.md:138-140, winding_net_pure.py:111-115
    """

    def __init__(
        self,
        shape: List[int],
        init: str = "uniform",
        init_scale: Optional[float] = None,
        requires_grad: bool = True,
        device: Optional[str] = None,
        precision: int = 100,
    ):
        """
        Initialize parameter.

        Args:
            shape: Parameter shape
            init: Initialization method ('uniform', 'normal', 'kaiming', 'golden', 'zeros')
            init_scale: Scale for initialization (auto-computed if None)
            requires_grad: Whether parameter is trainable
            precision: ResonantTensor precision
        """
        self.shape = list(shape)
        self.requires_grad = requires_grad
        self.device = device or "cpu"
        self.precision = precision

        # Compute size
        size = 1
        for d in shape:
            size *= d

        # Initialize data
        data = self._initialize(size, init, init_scale)
        mode_norms = [i * i for i in range(size)]  # Keep as exact integers

        self.tensor = ResonantTensor(data, shape, mode_norms, precision)
        if self.device != "cpu":
            self.tensor = self.tensor.to(self.device)

    def _initialize(
        self, size: int, method: str, scale: Optional[float]
    ) -> List[float]:
        """Initialize parameter data."""
        if method == "zeros":
            return [0] * size  # Use exact integer zeros

        if method == "ones":
            return [1] * size  # Use exact integer ones

        # Auto-compute scale
        if scale is None:
            if len(self.shape) >= 2:
                fan_in = self.shape[-2] if len(self.shape) >= 2 else self.shape[-1]
                scale = 1 / srt_math.sqrt(fan_in)  # Use exact division
            else:
                scale = Rational(1, 10)

        if method == "uniform":
            return [random.uniform(-scale, scale) for _ in range(size)]

        if method == "normal":
            return [random.gauss(0, scale) for _ in range(size)]

        if method == "kaiming":
            std = srt_math.sqrt(2 / self.shape[-1] if self.shape else 1)  # Use exact division
            return [random.gauss(0, std) for _ in range(size)]

        if method == "golden":
            # Use mode norm |n|² = i² (not index i) for variance scaling
            # Formula from SRT sub-Gaussian measure (resonant_embedding_pure.py:67)
            # Variance decays quadratically with mode norm: exp(-i²/(2φ))
            return [
                random.gauss(0, scale * srt_math.exp(-(i * i) / (2 * PHI)))
                for i in range(size)
            ]

        return [random.gauss(0, scale) for _ in range(size)]

    def to_list(self) -> List[float]:
        """Get parameter data as list."""
        return self.tensor.to_floats()

    def to(self, device: str) -> "Parameter":
        """Move parameter to device."""
        self.device = device
        self.tensor = self.tensor.to(device)
        return self

    def cuda(self, device_id: int = 0) -> "Parameter":
        """Move parameter to CUDA."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "Parameter":
        """Move parameter to CPU."""
        return self.to("cpu")

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, device={self.device}, requires_grad={self.requires_grad})"


class Module:
    """
    Base class for all syntonic network modules.

    Mirrors torch.nn.Module API for familiarity, but uses
    ResonantTensor for computations.
    """

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}
        self._buffers: Dict[str, Any] = {}
        self._device: str = "cpu"
        self.training = True

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            # Auto-move parameter to module's device
            if (
                hasattr(self, "_device")
                and self._device != "cpu"
                and value.device != self._device
            ):
                value.to(self._device)
            self._parameters[name] = value
        elif isinstance(value, Module):
            # Auto-move submodule to module's device
            if hasattr(self, "_device") and self._device != "cpu":
                value.to(self._device)
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value: Any):
        """Register a buffer (non-trainable state)."""
        self._buffers[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        """Forward pass - override in subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Iterate over module parameters."""
        for param in self._parameters.values():
            yield param

        if recurse:
            for module in self._modules.values():
                for param in module.parameters(recurse=True):
                    yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple]:
        """Iterate over named parameters."""
        for name, param in self._parameters.items():
            yield f"{prefix}{name}", param

        if recurse:
            for name, module in self._modules.items():
                subprefix = f"{prefix}{name}."
                for n, p in module.named_parameters(prefix=subprefix, recurse=True):
                    yield n, p

    def modules(self, recurse: bool = True) -> Iterator["Module"]:
        """Iterate over modules."""
        yield self

        if recurse:
            for module in self._modules.values():
                for m in module.modules(recurse=True):
                    yield m

    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode."""
        return self.train(False)

    def to(self, device: str) -> "Module":
        """Move module and all parameters to device."""
        self._device = device

        for param in self._parameters.values():
            param.to(device)

        for module in self._modules.values():
            module.to(device)

        for name, buf in self._buffers.items():
            if hasattr(buf, "to"):
                self._buffers[name] = buf.to(device)

        return self

    def cuda(self, device_id: int = 0) -> "Module":
        """Move module to CUDA."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "Module":
        """Move module to CPU."""
        return self.to("cpu")

    def extra_repr(self) -> str:
        """Extra info for repr - override in subclasses."""
        return ""

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        extra = self.extra_repr()
        if extra:
            lines.append(f"  {extra}")
        for name, module in self._modules.items():
            mod_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Sequential container for modules."""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
            setattr(self, str(i), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]


class ModuleList(Module):
    """Container for a list of modules."""

    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            for i, module in enumerate(modules):
                self._modules[str(i)] = module
                setattr(self, str(i), module)

    def append(self, module: Module):
        """Append a module."""
        idx = len(self._modules)
        self._modules[str(idx)] = module
        setattr(self, str(idx), module)

    def forward(self, x):
        raise NotImplementedError("ModuleList has no forward method")

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class Dropout(Module):
    """Dropout layer."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        if not self.training or self.p == 0:
            return x

        data = x.to_floats()
        mask = [1 if random.random() > self.p else 0 for _ in data]  # Use exact integers
        scale = 1 / (1 - self.p)  # Use exact division
        dropped = [d * m * scale for d, m in zip(data, mask)]
        return ResonantTensor(
            dropped,
            list(x.shape),
            [i * i for i in range(len(dropped))],  # Keep mode norms exact
            100,
            device=x.device,
        )

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Identity(Module):
    """Identity layer - returns input unchanged."""

    def forward(self, x):
        return x


class ReLU(Module):
    """ReLU activation."""

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation."""

    def __init__(self, precision: int = 100):
        super().__init__()
        self.precision = precision

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        return x.sigmoid(self.precision)


class Tanh(Module):
    """Tanh activation."""

    def __init__(self, precision: int = 100):
        super().__init__()
        self.precision = precision

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        return x.tanh(self.precision)


def compute_spatial_mode_norms(shape: List[int]) -> List[float]:
    """
    Compute spatial mode norms for data tensors (NOT for parameters).

    Use this ONLY for tensors representing spatial data on a torus where
    multi-dimensional frequency structure matters (W⁴ states, field configs).

    For neural network parameters, use default 1D mode norms (automatic).

    Args:
        shape: Tensor shape [d0, d1, d2, ...]

    Returns:
        Mode norms where element at [i0, i1, ...] has:
        |n|² = (i0 - d0//2)² + (i1 - d1//2)² + ...

    Example:
        >>> # For W⁴ winding state representation
        >>> mode_norms = compute_spatial_mode_norms([8, 8, 8, 8])
    """
    import itertools

    mode_norms = []
    ranges = [range(d) for d in shape]

    for indices in itertools.product(*ranges):
        norm_sq = sum((idx - dim // 2) ** 2 for idx, dim in zip(indices, shape))
        mode_norms.append(norm_sq)  # Keep as exact integer

    return mode_norms


__all__ = [
    "Parameter",
    "Module",
    "Sequential",
    "ModuleList",
    "Dropout",
    "Identity",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "PHI",
    "compute_spatial_mode_norms",
]
