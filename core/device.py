"""Device management for Syntonic.

Provides a unified interface for computation devices (CPU/CUDA) with
transparent access to the Rust backend's CUDA capabilities.

Examples:
    >>> from syntonic_applications.core.device import cpu, cuda, device
    >>> state = syn.state([1, 2, 3], device=cpu)
    >>> if cuda_is_available():
    ...     cuda_state = state.cuda()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._core import cuda_device_count as _rust_cuda_count

# Import CUDA functions from Rust backend
from ._core import cuda_is_available as _rust_cuda_available


@dataclass(frozen=True)
class Device:
    """Represents a computation device (CPU or CUDA GPU).

    Immutable device specification used to control where tensor
    computations are performed.

    Attributes:
        type: Device type string ('cpu' or 'cuda').
        index: CUDA device index (0-based), None for CPU.

    Examples:
        >>> cpu_dev = Device('cpu')
        >>> cuda_dev = Device('cuda', index=0)
        >>> print(cuda_dev.name)
        cuda:0
    """

    type: str  # 'cpu' or 'cuda'
    index: Optional[int] = None

    @property
    def name(self) -> str:
        """Get the canonical device name string.

        Returns:
            Device name (e.g., 'cpu', 'cuda:0', 'cuda:1').
        """
        if self.type == "cpu":
            return "cpu"
        return f"cuda:{self.index or 0}"

    @property
    def is_cpu(self) -> bool:
        """Check if this is a CPU device.

        Returns:
            True if CPU device, False otherwise.
        """
        return self.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        """Check if this is a CUDA GPU device.

        Returns:
            True if CUDA device, False otherwise.
        """
        return self.type == "cuda"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"syn.device('{self.name}')"


# Singleton CPU device
cpu = Device("cpu")


def cuda(device_id: int = 0) -> Device:
    """Get a CUDA GPU device.

    Args:
        device_id: CUDA device index (0-based). Default is 0.

    Returns:
        Device instance representing the CUDA GPU.

    Raises:
        RuntimeError: If CUDA is not available.

    Examples:
        >>> if cuda_is_available():
        ...     gpu = cuda(0)
        ...     print(gpu)
        cuda:0
    """
    if not cuda_is_available():
        raise RuntimeError("CUDA not available")
    return Device("cuda", device_id)  # pragma: no cover


def cuda_is_available() -> bool:
    """Check if CUDA is available via the Rust backend.

    Returns:
        True if CUDA is available and usable.
    """
    return _rust_cuda_available()


def cuda_device_count() -> int:
    """Get the number of available CUDA devices.

    Returns:
        Number of CUDA GPUs detected (0 if CUDA unavailable).
    """
    return _rust_cuda_count()


def device(spec: str) -> Device:
    """Parse a device from a string specification.

    Args:
        spec: Device string ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.).

    Returns:
        Device instance matching the specification.

    Raises:
        ValueError: If the device specification is not recognized.

    Examples:
        >>> device('cpu')
        syn.device('cpu')
        >>> device('cuda:1')
        syn.device('cuda:1')
    """
    if spec == "cpu":
        return cpu
    if spec.startswith("cuda"):
        if ":" in spec:
            idx = int(spec.split(":")[1])
            return cuda(idx)
        return cuda(0)
    raise ValueError(f"Unknown device: {spec}")
