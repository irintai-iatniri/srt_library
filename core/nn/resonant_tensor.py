"""
ResonantTensor Python Wrapper

This module provides a clean Python interface around the Rust-backed ResonantTensor
with full type annotations, comprehensive docstrings, and Pythonic operators.

The ResonantTensor is the core data structure of the Resonant Engine, maintaining
dual representations:
- Exact Q(φ) lattice (mathematical purity)
- Ephemeral flux values (CUDA-accelerated differentiation)
"""

from __future__ import annotations

from typing import List, Optional, Union, Callable, Any

from .._core import GoldenExact, py_broadcast_add, py_broadcast_mul
from .._core import ResonantTensor as _RustResonantTensor
from ..exact import FERMAT_PRIMES, LUCAS_SEQUENCE, MERSENNE_PRIMES


class _Function:
    """Base class for autograd functions."""

    @staticmethod
    def forward(ctx: Any, *inputs: ResonantTensor) -> ResonantTensor:
        """Forward pass."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Any, grad_output: ResonantTensor) -> List[Optional[ResonantTensor]]:
        """Backward pass."""
        raise NotImplementedError


class _AddFunction(_Function):
    """Addition operation."""

    @staticmethod
    def forward(ctx: Any, a: ResonantTensor, b: ResonantTensor) -> ResonantTensor:
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx: Any, grad_output: ResonantTensor) -> List[Optional[ResonantTensor]]:
        a, b = ctx.saved_tensors
        return [grad_output, grad_output]


class _MulFunction(_Function):
    """Multiplication operation."""

    @staticmethod
    def forward(ctx: Any, a: ResonantTensor, b: ResonantTensor) -> ResonantTensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Any, grad_output: ResonantTensor) -> List[Optional[ResonantTensor]]:
        a, b = ctx.saved_tensors
        return [grad_output * b, grad_output * a]


class _PowFunction(_Function):
    """Power operation."""

    @staticmethod
    def forward(ctx: Any, a: ResonantTensor, exponent: float) -> ResonantTensor:
        ctx.save_for_backward(a, exponent)
        return a ** exponent

    @staticmethod
    def backward(ctx: Any, grad_output: ResonantTensor) -> List[Optional[ResonantTensor]]:
        a, exponent = ctx.saved_tensors
        grad_a = grad_output * exponent * a.pow(exponent - 1)
        return [grad_a, None]


class _Context:
    """Context for storing intermediate values during autograd."""

    def __init__(self):
        self.saved_tensors: List[ResonantTensor] = []

    def save_for_backward(self, *tensors: ResonantTensor) -> None:
        """Save tensors for backward pass."""
        self.saved_tensors = list(tensors)


class ResonantTensor:
    """
    A tensor that exists in dual representations: exact Q(φ) lattice and ephemeral flux.

    The ResonantTensor maintains mathematical purity through exact golden ratio arithmetic
    while supporting CUDA-accelerated differentiation and harmonization cycles.

    Attributes:
        syntony: Current syntony value S ∈ [0, 1]
        phase: Current phase ("crystallized" or "flux")
        shape: Tensor shape as list of dimensions
        shape: Tensor shape as list of dimensions
        precision: Lattice precision for crystallization
        device: Device location ('cpu' or 'cuda:N')

    Examples:
        >>> # Create from floats with default mode norms
        >>> data = [1.0, 2.0, 3.0, 4.0]
        >>> tensor = ResonantTensor(data, shape=[2, 2])
        >>> print(tensor)
        ResonantTensor(shape=[2, 2], phase=crystallized, syntony=0.8234, precision=100)

        >>> # Run a DHSR cycle
        >>> new_syntony = tensor.cpu_cycle(noise_scale=0.1, precision=100)
        >>> print(f"New syntony: {new_syntony:.4f}")

        >>> # Use Pythonic operators
        >>> a = ResonantTensor.randn([3, 3])
        >>> b = ResonantTensor.randn([3, 3])
        >>> c = a + b  # Element-wise addition
        >>> d = a * 2.0  # Scalar multiplication
        >>> e = a @ b  # Matrix multiplication
    """

    def __init__(
        self,
        data: List[float],
        shape: List[int],
        mode_norm_sq: Optional[List[float]] = None,
        precision: int = 100,
        device: str = "cpu",
        requires_grad: bool = False,
    ):
        """
        Create a ResonantTensor from floating-point data.

        Args:
            data: Flattened tensor values
            shape: Shape of the tensor (e.g., [batch, features])
            mode_norm_sq: Mode norms |n|² for each element (defaults to [i² for i in range(size)])
            precision: Maximum coefficient for golden lattice snapping (default: 100)
            device: Device to place tensor on ('cpu' or 'cuda:N')
            requires_grad: Whether to track gradients for this tensor

        Examples:
            >>> # Basic tensor
            >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

            >>> # Tensor with gradient tracking
            >>> x = ResonantTensor([1.0, 2.0], shape=[2], requires_grad=True)
            >>> y = x * x
            >>> y.backward()
            >>> print(x.grad)
        """
        if device.startswith("cuda"):
            # Parse device index
            if ":" in device:
                idx = int(device.split(":")[1])
            else:
                idx = 0

            # Try direct GPU creation first (more efficient)
            try:
                self._inner = _RustResonantTensor.from_floats_cuda(
                    data, shape, idx, mode_norm_sq, precision
                )
            except (AttributeError, RuntimeError):
                # Fallback to CPU creation + GPU transfer
                self._inner = _RustResonantTensor(data, shape, mode_norm_sq, precision)
                self._inner = self._inner.to_device(idx)
        else:
            self._inner = _RustResonantTensor(data, shape, mode_norm_sq, precision)

        self._device_str = device

        # Autograd attributes
        self._grad: Optional[List[float]] = None
        self._requires_grad: bool = requires_grad
        self._grad_fn: Optional[Callable] = None
        self._ctx: Optional[_Context] = None

    # =========================================================================
    # Gradient Support for Optimization
    # =========================================================================

    @property
    def grad(self) -> Optional["ResonantTensor"]:
        """
        Get the gradient tensor if it exists.

        Returns:
            ResonantTensor containing gradients, or None if no gradient computed
        """
        if self._grad is None:
            return None
        return ResonantTensor(self._grad, list(self.shape), device=self.device)

    @grad.setter
    def grad(self, value: Optional["ResonantTensor"]) -> None:
        """Set the gradient from a ResonantTensor or None."""
        if value is None:
            self._grad = None
        else:
            self._grad = value.to_floats()

    @property
    def requires_grad(self) -> bool:
        """Check if this tensor requires gradient computation."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set whether this tensor requires gradient computation."""
        self._requires_grad = value

    def zero_grad(self) -> None:
        """
        Zero out the gradient.

        Call this before each forward/backward pass to clear accumulated gradients.
        """
        if self._grad is not None:
            self._grad = [0.0] * len(self._grad)

    def accumulate_grad(self, grad_data: List[float]) -> None:
        """
        Accumulate gradient from a backward pass.

        Args:
            grad_data: List of gradient values to accumulate
        """
        if self._grad is None:
            self._grad = grad_data[:]
        else:
            self._grad = [g + dg for g, dg in zip(self._grad, grad_data)]

    def backward(self, gradient: Optional["ResonantTensor"] = None) -> None:
        """
        Compute gradients through backpropagation.

        Implements reverse-mode automatic differentiation for the computational graph.

        Args:
            gradient: Gradient with respect to this tensor (default: ones tensor)

        Examples:
            >>> x = ResonantTensor([2.0], [1], requires_grad=True)
            >>> y = x * x  # y = x²
            >>> y.backward()  # dy/dx = 2x = 4.0
            >>> print(x.grad)  # Should be [4.0]
        """
        if gradient is None:
            # Default gradient is ones tensor
            gradient = ResonantTensor.ones(self.shape, device=self.device)

        if not self.requires_grad:
            return

        # Initialize gradient if not set
        if self._grad is None:
            self._grad = gradient.to_floats()
        else:
            # Accumulate gradient
            grad_list = gradient.to_floats()
            self._grad = [g + dg for g, dg in zip(self._grad, grad_list)]

        # If we have a gradient function, call it
        if self._grad_fn is not None:
            input_grads = self._grad_fn(self._ctx, gradient)
            if input_grads is not None:
                for i, grad in enumerate(input_grads):
                    if grad is not None and hasattr(self, f'_input_{i}'):
                        input_tensor = getattr(self, f'_input_{i}')
                        if input_tensor.requires_grad:
                            input_tensor.backward(grad)

    def detach(self) -> "ResonantTensor":
        """
        Detach tensor from computational graph.

        Returns a new tensor that doesn't require gradients.

        Returns:
            Detached tensor
        """
        return ResonantTensor(
            self.get_data_list(),
            self.shape,
            mode_norm_sq=None,  # Could preserve if needed
            precision=self.precision,
            device=self.device
        )

    def retain_grad(self) -> None:
        """
        Enable gradient retention for this tensor.

        Gradients will be stored even for intermediate tensors.
        """
        self._retain_grad = True

    # =========================================================================
    # Autograd Operations
    # =========================================================================

    def add(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Addition with autograd support.

        Args:
            other: Tensor to add

        Returns:
            Result tensor
        """
        result_data = []
        for a, b in zip(self.get_data_list(), other.get_data_list()):
            result_data.append(a + b)

        requires_grad_result = self._requires_grad or other._requires_grad
        result = ResonantTensor(result_data, self.shape, device=self.device, requires_grad=requires_grad_result)

        if self._requires_grad or other._requires_grad:
            result._grad_fn = _AddFunction.backward
            result._ctx = _Context()
            result._ctx.save_for_backward(self, other)
            result._input_0 = self
            result._input_1 = other
        return result

    def mul(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Multiplication with autograd support.

        Args:
            other: Tensor to multiply

        Returns:
            Result tensor
        """
        result_data = []
        for a, b in zip(self.get_data_list(), other.get_data_list()):
            result_data.append(a * b)

        requires_grad_result = self._requires_grad or other._requires_grad
        result = ResonantTensor(result_data, self.shape, device=self.device, requires_grad=requires_grad_result)

        if self._requires_grad or other._requires_grad:
            result._grad_fn = _MulFunction.backward
            result._ctx = _Context()
            result._ctx.save_for_backward(self, other)
            result._input_0 = self
            result._input_1 = other
        return result

    def sum(self) -> "ResonantTensor":
        """
        Sum all elements with autograd support.

        Returns:
            Scalar tensor containing sum
        """
        # For simplicity, sum to a scalar
        total = sum(self.get_data_list())
        result = ResonantTensor([total], [1], device=self.device, requires_grad=self._requires_grad)

        if self._requires_grad:
            def sum_backward(ctx, grad_output):
                # Gradient is grad_output broadcasted to input shape
                grad_input = ResonantTensor.ones(self.shape, device=self.device) * grad_output.get_data_list()[0]
                return [grad_input]

            result._grad_fn = sum_backward
            result._ctx = _Context()
            result._input_0 = self

        return result

    def get_data_list(self) -> List[float]:
        """
        Get raw data as a mutable list for in-place optimization.

        Returns:
            List of float values (copy of internal data)
        """
        return self.to_floats()

    def set_data_list(self, data: List[float]) -> None:
        """
        Set data from a list (reconstructs the tensor).

        Args:
            data: New data values
        """
        # Reconstruct the tensor with new data
        self._inner = _RustResonantTensor(
            data, list(self.shape), self.get_mode_norms(), self.precision
        )

    @property
    def size(self) -> int:
        """Get total number of elements."""
        result = 1
        for d in self.shape:
            result *= d
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def syntony(self) -> float:
        """Get the current syntony value S ∈ [0, 1]."""
        return self._inner.syntony

    @property
    def inner(self) -> _RustResonantTensor:
        """Access the underlying Rust backend object."""
        return self._inner

    @property
    def phase(self) -> str:
        """Get the current phase ("crystallized" or "flux")."""
        return self._inner.phase

    @property
    def shape(self) -> List[int]:
        """Get the tensor shape."""
        return self._inner.shape

    @property
    def precision(self) -> int:
        """Get the precision used for last crystallization."""
        return self._inner.precision

    @property
    def device(self) -> str:
        """Get the device string (e.g. 'cpu', 'cuda:0')."""
        return self._device_str

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_floats_32(
        cls,
        data: List[float],
        shape: List[int],
        precision: int = 100,
        device: str = "cpu",
    ) -> "ResonantTensor":
        """
        Create from float32 values (alias for constructor).

        Args:
            data: Flattened tensor values
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New ResonantTensor
        """
        return cls(data, shape, precision=precision, device=device)

    @classmethod
    def from_golden_exact(
        cls,
        lattice: List[GoldenExact],
        shape: List[int],
        mode_norm_sq: Optional[List[float]] = None,
    ) -> "ResonantTensor":
        """
        Create from exact Q(φ) lattice values.

        Args:
            lattice: List of GoldenExact values (a + b·φ)
            shape: Tensor shape
            mode_norm_sq: Optional mode norms

        Returns:
            New ResonantTensor in crystallized phase

        Examples:
            >>> from syntonic_applications._core import GoldenExact
            >>> lattice = [GoldenExact.from_integers(1, 0), GoldenExact.from_integers(0, 1)]
            >>> tensor = ResonantTensor.from_golden_exact(lattice, shape=[2])
        """
        instance = cls.__new__(cls)
        instance._inner = _RustResonantTensor.from_golden_exact(
            lattice, shape, mode_norm_sq
        )
        instance._device_str = "cpu"  # Default to cpu for lattice creation
        return instance

    @classmethod
    def zeros(
        cls, shape: List[int], precision: int = 100, device: str = "cpu"
    ) -> "ResonantTensor":
        """
        Create a zero-initialized tensor.

        Args:
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New tensor filled with zeros (using exact GoldenExact::zero)

        Examples:
            >>> zeros = ResonantTensor.zeros([3, 3])
            >>> assert all(v == 0.0 for v in zeros.to_floats())
        """
        instance = cls.__new__(cls)
        instance._inner = _RustResonantTensor.zeros(shape, precision)
        if device != "cpu":
            instance._inner = instance._inner.to_device(device)
        instance._device_str = device
        return instance

    @classmethod
    def ones(
        cls, shape: List[int], precision: int = 100, device: str = "cpu"
    ) -> "ResonantTensor":
        """
        Create a ones-initialized tensor.

        Args:
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New tensor filled with ones

        Examples:
            >>> ones = ResonantTensor.ones([2, 2])
            >>> assert all(abs(v - 1.0) < 0.01 for v in ones.to_floats())
        """
        size = 1
        for dim in shape:
            size *= dim
        return cls([1.0] * size, shape, precision=precision, device=device)

    @classmethod
    def from_floats_cuda(
        cls,
        data: List[float],
        shape: List[int],
        device_idx: int = 0,
        mode_norm_sq: Optional[List[float]] = None,
        precision: int = 100,
    ) -> "ResonantTensor":
        """
        Create a ResonantTensor directly on GPU from floats.

        This bypasses CPU creation → GPU transfer for better performance.

        Args:
            data: Flattened tensor values
            shape: Tensor shape
            device_idx: CUDA device index (default: 0)
            mode_norm_sq: Mode norms |n|² for each element
            precision: Lattice precision

        Returns:
            New ResonantTensor on GPU

        Examples:
            >>> data = [1.0, 2.0, 3.0, 4.0]
            >>> tensor = ResonantTensor.from_floats_cuda(data, [2, 2], device_idx=0)
        """
        instance = cls.__new__(cls)
        instance._inner = _RustResonantTensor.from_floats_cuda(
            data, shape, device_idx, mode_norm_sq, precision
        )
        instance._device_str = f"cuda:{device_idx}"
        return instance

    @classmethod
    def randn(
        cls,
        shape: List[int],
        mean: float = 0.0,
        std: float = 1.0,
        precision: int = 100,
        device: str = "cpu",
    ) -> "ResonantTensor":
        """
        Create a tensor with random Gaussian values.

        Args:
            shape: Tensor shape
            mean: Mean of Gaussian distribution
            std: Standard deviation
            precision: Lattice precision

        Returns:
            New tensor with random values

        Examples:
            >>> # Standard normal
            >>> tensor = ResonantTensor.randn([100, 100])

            >>> # Custom distribution
            >>> tensor = ResonantTensor.randn([10, 10], mean=5.0, std=2.0)
        """
        import random

        size = 1
        for dim in shape:
            size *= dim
        data = [random.gauss(mean, std) for _ in range(size)]
        return cls(data, shape, precision=precision, device=device)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_floats(self) -> List[float]:
        """
        Convert to list of floats (approximate representation).

        Returns:
            List of float values

        Examples:
            >>> tensor = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> floats = tensor.to_floats()
        """
        return self._inner.to_floats()

    def to_list(self) -> List[float]:
        """
        Alias for to_floats().

        Returns:
            List of float values
        """
        return self._inner.to_list()

    def to_lattice(self) -> List[GoldenExact]:
        """
        Get exact Q(φ) lattice values.

        Returns:
            List of GoldenExact values (a + b·φ)

        Examples:
            >>> tensor = ResonantTensor([1.618, 2.0], [2])
            >>> lattice = tensor.to_lattice()
            >>> for g in lattice:
            ...     print(f"{g}")  # Shows exact representation
        """
        return self._inner.get_lattice()

    def get_mode_norms(self) -> List[float]:
        """
        Get mode norm squared values.

        Returns:
            List of |n|² values for each element
        """
        return self._inner.get_mode_norm_sq()

    def to(self, device: str) -> "ResonantTensor":
        """
        Move tensor to device.

        Args:
            device: Target device ('cpu', 'cuda:0', etc.)

        Returns:
            New tensor on target device (or self if already there)
        """
        # Optimized: check if already on device
        if self.device == device:
            return self

        # Call Rust backend
        if device == "cpu":
            new_inner = self._inner.to_cpu()
        elif device.startswith("cuda"):
            if ":" in device:
                idx = int(device.split(":")[1])
            else:
                idx = 0
            new_inner = self._inner.to_device(idx)
        else:
            raise ValueError(f"Unsupported device: {device}")

        new_tensor = ResonantTensor._wrap(new_inner)
        new_tensor._device_str = device
        return new_tensor

    def cuda(self, device_id: int = 0) -> "ResonantTensor":
        """Move to CUDA."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "ResonantTensor":
        """Move to CPU."""
        return self.to("cpu")

    # =========================================================================
    # Phase Transitions (DHSR Cycle)
    # =========================================================================

    def wake_flux(self) -> List[float]:
        """
        Enter D-phase: project lattice → flux values.

        Returns:
            Flux values as list of floats

        Examples:
            >>> tensor = ResonantTensor.ones([2, 2])
            >>> flux = tensor.wake_flux()
            >>> assert tensor.phase == "flux"
        """
        return self._inner.wake_flux_values()

    # Internal wrapper
    @classmethod
    def _wrap(cls, inner: _RustResonantTensor, device: str = "cpu") -> "ResonantTensor":
        """Wrap a Rust tensor without triggering __init__."""
        instance = cls.__new__(cls)
        instance._inner = inner
        instance._device_str = device
        
        # Initialize autograd attributes
        instance._grad = None
        instance._requires_grad = False
        instance._grad_fn = None
        instance._ctx = None
        
        return instance
        """
        Enter H-phase: snap flux → lattice.

        Args:
            values: Flux values to crystallize
            precision: Lattice precision for snapping

        Returns:
            New syntony value after crystallization

        Examples:
            >>> tensor = ResonantTensor.zeros([3])
            >>> flux = tensor.wake_flux()
            >>> # Modify flux somehow
            >>> new_syntony = tensor.crystallize(flux, precision=100)
        """
        return self._inner.crystallize_from_values(values, precision)

    def cpu_cycle(self, noise_scale: float = 0.01, precision: int = 100) -> float:
        """
        Run full D→H cycle in CPU mode.

        This simulates the DHSR (Differentiation-Harmonization-Syntony-Recursion) cycle:
        1. D-phase: Add noise and scale by mode structure
        2. H-phase: Snap back to Q(φ) lattice with attenuation

        Args:
            noise_scale: Scale of stochastic noise in D-phase
            precision: Lattice precision for crystallization

        Returns:
            New syntony value after cycle

        Examples:
            >>> tensor = ResonantTensor.randn([10, 10])
            >>> for _ in range(100):
            ...     syntony = tensor.cpu_cycle(noise_scale=0.1)
            >>> print(f"Final syntony: {syntony:.4f}")
        """
        return self._inner.cpu_cycle(noise_scale, precision)

    def batch_cpu_cycle(
        self, noise_scale: float = 0.01, precision: int = 100
    ) -> List[float]:
        """
        Run batched D→H cycle (for batch dimension).

        Assumes first dimension is batch, applies cycle to each sample independently.

        Args:
            noise_scale: Noise scale
            precision: Lattice precision

        Returns:
            List of syntony values, one per batch sample

        Examples:
            >>> batch_tensor = ResonantTensor.randn([8, 16])  # Batch of 8
            >>> syntonies = batch_tensor.batch_cpu_cycle(noise_scale=0.05)
            >>> assert len(syntonies) == 8
        """
        return self._inner.batch_cpu_cycle(noise_scale, precision)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def matmul(self, weights: "ResonantTensor") -> "ResonantTensor":
        """
        Matrix multiplication: self @ weights.

        Performs Y = X @ W^T where self is X and weights is W.
        All arithmetic is exact in Q(φ).

        Args:
            weights: Weight tensor (out_features, in_features)

        Returns:
            New tensor with result

        Examples:
            >>> x = ResonantTensor.randn([4, 10])  # Batch of 4, 10 features
            >>> w = ResonantTensor.randn([20, 10])  # 10 → 20
            >>> y = x.matmul(w)  # [4, 20]
        """
        result = self._inner.matmul(weights._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def add_bias(self, bias: "ResonantTensor") -> None:
        """
        Add bias in-place.

        Args:
            bias: Bias tensor (must match output dimension)

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> bias = ResonantTensor.randn([10])
            >>> x.add_bias(bias)  # In-place
        """
        self._inner.add_bias(bias._inner)

    # =========================================================================
    # Activations
    # =========================================================================

    def relu(self) -> None:
        """
        Apply ReLU activation in-place.

        Snaps all negative lattice values to zero.

        Examples:
            >>> x = ResonantTensor([-1.0, 2.0, -3.0, 4.0], [4])
            >>> x.relu()
            >>> assert x.to_floats()[0] == 0.0
            >>> assert x.to_floats()[1] > 0.0
        """
        self._inner.relu()

    def sigmoid(self, precision: int = 100) -> None:
        """
        Apply sigmoid activation in-place: σ(x) = 1 / (1 + e^(-x)).

        Args:
            precision: Lattice precision for snapping result

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, -1.0], [3])
            >>> x.sigmoid()
            >>> floats = x.to_floats()
            >>> assert 0.4 < floats[0] < 0.6  # sigmoid(0) ≈ 0.5
        """
        self._inner.sigmoid(precision)

    def tanh(self, precision: int = 100) -> None:
        """
        Apply tanh activation in-place.

        Args:
            precision: Lattice precision

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, -1.0], [3])
            >>> x.tanh()
        """
        self._inner.tanh(precision)

    def golden_gelu(self, precision: int = 100) -> "ResonantTensor":
        """
        Apply Syntonic Golden GELU activation in-place.

        Unlike standard GELU which uses Gaussian approximation,
        this uses the Golden Ratio field for exact resonance.

        Args:
            precision: Lattice precision for snapping.

        Returns:
            Self (for chaining).
        """
        # Checks if backend supports golden_gelu, falls back to gelu if not
        self._inner.golden_gelu(precision)

    def gelu(self, precision: int = 100) -> None:
        """
        Apply GELU activation in-place.

        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        Args:
            precision: Lattice precision

        Examples:
            >>> x = ResonantTensor.randn([10, 10])
            >>> x.gelu()
        """
        self._inner.gelu(precision)

    def softmax(self, dim: Optional[int] = None, precision: int = 32) -> None:
        """
        Apply softmax along a dimension in-place.

        Uses numerically stable computation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

        Args:
            dim: Dimension to apply softmax along. If None, defaults to -1 (last dimension).
            precision: Lattice precision

        Examples:
            >>> # Classification logits
            >>> logits = ResonantTensor([2.0, 1.0, 0.1], [3])
            >>> logits.softmax()
            >>> probs = logits.to_floats()
            >>> assert abs(sum(probs) - 1.0) < 0.01
        """
        self._inner.softmax(dim, precision)

    def sin(self, precision: int = 100) -> "ResonantTensor":
        """
        Sine function: sin(x).

        Computes element-wise sine, snaps result back to Q(φ) lattice.

        Args:
            precision: Lattice precision for snapping

        Returns:
            New tensor with sin applied

        Examples:
            >>> x = ResonantTensor([0.0, 3.14159/2, 3.14159], [3])
            >>> y = x.sin()
            >>> # y ≈ [0.0, 1.0, 0.0]
        """
        result = self._inner.sin(precision)
        return ResonantTensor._wrap(result, device=self.device)

    def cos(self, precision: int = 100) -> "ResonantTensor":
        """
        Cosine function: cos(x).

        Computes element-wise cosine, snaps result back to Q(φ) lattice.

        Args:
            precision: Lattice precision for snapping

        Returns:
            New tensor with cos applied

        Examples:
            >>> x = ResonantTensor([0.0, 3.14159/2, 3.14159], [3])
            >>> y = x.cos()
            >>> # y ≈ [1.0, 0.0, -1.0]
        """
        result = self._inner.cos(precision)
        return ResonantTensor._wrap(result, device=self.device)

    def pow(self, exponent: float, precision: int = 100) -> "ResonantTensor":
        """
        Power function: x^exponent.

        Computes element-wise power, snaps result back to Q(φ) lattice.

        Args:
            exponent: Power to raise each element to
            precision: Lattice precision for snapping

        Returns:
            New tensor with power applied

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> y = x.pow(2.0)  # Square each element
            >>> # y ≈ [1.0, 4.0, 9.0]
        """
        result = self._inner.pow(exponent, precision)
        result_tensor = ResonantTensor._wrap(result, device=self.device)
        
        # Set up autograd
        result_tensor.requires_grad = self._requires_grad
        if self._requires_grad:
            result_tensor._grad_fn = _PowFunction.backward
            result_tensor._ctx = _Context()
            result_tensor._ctx.save_for_backward(self, exponent)
            result_tensor._input_0 = self
        
        return result_tensor

    def sqrt(self, precision: int = 100) -> "ResonantTensor":
        """
        Square root: sqrt(x).

        Computes element-wise square root, snaps result back to Q(φ) lattice.

        Args:
            precision: Lattice precision for snapping

        Returns:
            New tensor with sqrt applied

        Examples:
            >>> x = ResonantTensor([1.0, 4.0, 9.0], [3])
            >>> y = x.sqrt()
            >>> # y ≈ [1.0, 2.0, 3.0]
        """
        result = self._inner.sqrt(precision)
        return ResonantTensor._wrap(result, device=self.device)

    def dropout(self, p: float = 0.5) -> None:
        """
        Apply dropout in-place.

        Randomly zeroes out elements with probability p.
        Scaling is applied to preserve expected sum.

        Args:
            p: Probability of an element to be zeroed.
        """
        self._inner.dropout(p)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def elementwise_add(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise addition: self + other.

        Args:
            other: Tensor with same shape

        Returns:
            New tensor with sum

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = a.elementwise_add(b)  # [4.0, 6.0]
        """
        result = self._inner.elementwise_add(other._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def broadcast_add(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Broadcast addition: self + other with NumPy-style broadcasting.

        Adds a smaller tensor to a larger tensor by broadcasting the smaller
        tensor across dimensions. Keeps computation in Rust backend without
        extracting to Python floats.

        Args:
            other: Tensor to broadcast-add (e.g., shape [1,1] broadcasts to [batch, features])

        Returns:
            New tensor with broadcast sum

        Examples:
            >>> x = ResonantTensor.randn([4, 248])  # [batch, features]
            >>> scalar = ResonantTensor([1.5], [1, 1])  # Scalar as tensor
            >>> result = x.broadcast_add(scalar)  # Adds 1.5 to all elements
        """
        a_data = self.to_floats()
        b_data = other.to_floats()
        a_shape = [int(d) for d in self.shape]
        b_shape = [int(d) for d in other.shape]

        result = py_broadcast_add(a_data, a_shape, b_data, b_shape)
        if result is None:
            raise ValueError(f"Shapes not broadcastable: {a_shape} vs {b_shape}")

        result_data, result_shape = result
        return ResonantTensor(result_data, result_shape, device=self.device)

    def broadcast_mul(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Broadcast multiplication: self * other with NumPy-style broadcasting.

        Args:
            other: Tensor to broadcast-multiply

        Returns:
            New tensor with broadcast product
        """
        a_data = self.to_floats()
        b_data = other.to_floats()
        a_shape = [int(d) for d in self.shape]
        b_shape = [int(d) for d in other.shape]

        result = py_broadcast_mul(a_data, a_shape, b_data, b_shape)
        if result is None:
            raise ValueError(f"Shapes not broadcastable: {a_shape} vs {b_shape}")

        result_data, result_shape = result
        return ResonantTensor(result_data, result_shape, device=self.device)

    def elementwise_mul(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise multiplication (Hadamard product): self * other.

        Args:
            other: Tensor with same shape

        Returns:
            New tensor with element-wise product

        Examples:
            >>> a = ResonantTensor([2.0, 3.0], [2])
            >>> b = ResonantTensor([4.0, 5.0], [2])
            >>> c = a.elementwise_mul(b)  # [8.0, 15.0]
        """
        result = self._inner.elementwise_mul(other._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def scalar_mul(self, scalar: float) -> "ResonantTensor":
        """
        Multiply by scalar: self * scalar.

        Args:
            scalar: Scalar value

        Returns:
            New tensor with all elements multiplied

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> y = x.scalar_mul(2.0)  # [2.0, 4.0, 6.0]
        """
        result = self._inner.scalar_mul(scalar)
        return ResonantTensor._wrap(result, device=self.device)

    def scalar_add(self, scalar: float) -> "ResonantTensor":
        """
        Add scalar to all elements: self + scalar.

        Args:
            scalar: Scalar value

        Returns:
            New tensor with scalar added

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> y = x.scalar_add(10.0)  # [11.0, 12.0, 13.0]
        """
        result = self._inner.scalar_add(scalar)
        return ResonantTensor._wrap(result, device=self.device)

    def negate(self) -> "ResonantTensor":
        """
        Negate: -self.

        Returns:
            New tensor with all elements negated

        Examples:
            >>> x = ResonantTensor([1.0, -2.0, 3.0], [3])
            >>> y = x.negate()  # [-1.0, 2.0, -3.0]
        """
        result = self._inner.negate()
        return ResonantTensor._wrap(result, device=self.device)

    def one_minus(self) -> "ResonantTensor":
        """
        Compute 1 - self.

        Common pattern in gating mechanisms and attention.

        Returns:
            New tensor with 1 - x for each element

        Examples:
            >>> x = ResonantTensor([0.2, 0.5, 0.8], [3])
            >>> y = x.one_minus()  # [0.8, 0.5, 0.2]
        """
        result = self._inner.one_minus()
        return ResonantTensor._wrap(result, device=self.device)

    # =========================================================================
    # Math Functions
    # =========================================================================

    def log(self, precision: Optional[int] = None) -> "ResonantTensor":
        """
        Natural logarithm: ln(x).

        Args:
            precision: Lattice precision (defaults to tensor's precision)

        Returns:
            New tensor with logarithm applied

        Examples:
            >>> x = ResonantTensor([1.0, 2.718, 7.389], [3])
            >>> y = x.log()  # [0.0, 1.0, 2.0]
        """
        result = self._inner.log(precision)
        return ResonantTensor._wrap(result, device=self.device)

    def exp(self, precision: Optional[int] = None) -> "ResonantTensor":
        """
        Natural exponential: e^x.

        Args:
            precision: Lattice precision

        Returns:
            New tensor with exponential applied

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, 2.0], [3])
            >>> y = x.exp()  # [1.0, 2.718, 7.389]
        """
        result = self._inner.exp(precision)
        return ResonantTensor._wrap(result, device=self.device)

    # =========================================================================
    # Prime Resonance Operators (Grand Synthesis)
    # =========================================================================

    def lucas_noise(self, n: int = 5) -> "ResonantTensor":
        """
        Inject novelty/noise scaled by the Lucas sequence (Shadow Sector).

        Magnitude ~ 1 / L_n.
        Higher n = Lower noise (Shadow Fading).

        Args:
            n: Lucas index (default 5, L_5=11 -> scale ~0.09)

        Returns:
            New tensor with added noise.
        """
        if n < 0:
            n = 0
        idx = min(n, len(LUCAS_SEQUENCE) - 1)
        scale = 1.0 / float(LUCAS_SEQUENCE[idx])

        noise = ResonantTensor.randn(self.shape, std=scale, device=self.device)
        return self.elementwise_add(noise)

    def apply_fermat_force(self, n: int = 0) -> "ResonantTensor":
        """
        Apply force scaling governed by Fermat primes (Interaction Strength).

        Scale ~ 1 / F_n.
        F_0 = 3 (Strong), F_4 = 65537 (Weak).

        Args:
            n: Fermat index (0..4)

        Returns:
            Scaled tensor.
        """
        if n < 0:
            n = 0
        idx = min(n, len(FERMAT_PRIMES) - 1)
        coupling = 1.0 / float(FERMAT_PRIMES[idx])
        return self.scalar_mul(coupling)

    def mersenne_stability(self, n: int = 2) -> "ResonantTensor":
        """
        Harmonize/Filter based on Mersenne Prime stability (Matter Stability).

        Aligns tensor towards Mersenne scales.
        (Simplified implementation: Weighted average with One).

        Args:
            n: Mersenne index (2, 3, 5, 7 recommended)

        Returns:
            Stabilized tensor.
        """
        # M_n = 2^n - 1
        # Use as a strong anchor/bias
        # result = self * (1 - alpha) + alpha * M_n
        # Here we use n as "Generation" of stability

        # Approximate effect: Decay towards zero or specific structure?
        # Theory says "Harmonization".
        # We'll use a simple "Crystallize" pass with precision modulated by M_n

        # M_2=3, M_3=7, M_5=31, M_7=127.
        # Higher M = Higher Precision?
        # Let's map n index to M values. (2->0, 3->1...)
        possible_inds = [2, 3, 5, 7]
        if n not in possible_inds:
            # Find closest
            n = min(possible_inds, key=lambda x: abs(x - n))

        idx = possible_inds.index(n)
        mersenne_val = MERSENNE_PRIMES[min(idx, len(MERSENNE_PRIMES) - 1)]

        # Higher Mersenne = More Stable = Less Noise in cycle
        # We perform a crystallization cycle with precision ~ M_n * 10
        return self.cycle(noise_scale=0.0, precision=int(mersenne_val * 10))

    def cycle(
        self, noise_scale: float = 0.01, precision: int = 100
    ) -> "ResonantTensor":
        """
        Run a single DHSR cycle (Unwrap/Re-wrap helper).
        """
        # Uses cpu_cycle logic but returns Tensor instead of syntony?
        # Actually standard cpu_cycle returns Syntony float.
        # We typically want the TENSOR back.
        # But cpu_cycle modifies in-place in Rust?
        # Doc says: "Returns: New syntony value".
        # "1. D-phase... 2. H-phase: Snap back".
        # It modifies self._inner in place!

        # So we clone first to be safe, or modify in place?
        # NN ops usually functional.
        # Let's assume in-place for now or check.
        # If in-place, we return self.

        # Rust signature: &mut self.
        # So it IS in-place modification of lattice.

        # We'll clone for safety if functional style desired, or just modify.
        # Let's return self for chaining.
        self._inner.cpu_cycle(noise_scale, precision)
        return self

    # =========================================================================
    # Advanced Operations
    # =========================================================================

    @staticmethod
    def concat(tensors: List["ResonantTensor"], dim: int = -1) -> "ResonantTensor":
        """
        Concatenate tensors along a dimension.

        Args:
            tensors: List of tensors to concatenate
            dim: Dimension to concatenate along (supports negative indexing)

        Returns:
            New concatenated tensor

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = ResonantTensor.concat([a, b], dim=0)  # Shape: [4]
        """
        if not tensors:
            raise ValueError("concat expects at least one tensor")

        ndim = len(tensors[0].shape)
        if dim < 0:
            dim += ndim

        # PyO3 requires Python context for static methods
        # This is a workaround to get the Python module
        import sys

        if "python.syntonic._core" not in sys.modules:
            pass

        # Get the module that contains ResonantTensor
        # core_module = sys.modules["syntonic._core"]

        # Create Py references for PyO3
        from .._core import ResonantTensor as _RT

        inner_list = [t._inner for t in tensors]

        # Call the static method with Python context
        # Fixed: correct arguments (tensors, dim) without module context
        result = _RT.concat(inner_list, dim)
        return ResonantTensor._wrap(
            result, device=tensors[0].device if tensors else "cpu"
        )

    def index_select(self, indices: List[int], dim: int = 0) -> "ResonantTensor":
        """
        Select slices along a dimension.

        Args:
            indices: Indices to select
            dim: Dimension to select along

        Returns:
            New tensor with selected slices

        Examples:
            >>> x = ResonantTensor.randn([10, 5])
            >>> selected = x.index_select([0, 2, 4], dim=0)  # Shape: [3, 5]
        """
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim

        try:
            result = self._inner.index_select(indices, dim)
            return ResonantTensor._wrap(result, device=self.device)
        except AttributeError:
            # Fallback for when backend doesn't implement index_select
            import itertools

            shape = self.shape

            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]

            # Prepare new shape
            new_shape = list(shape)
            new_shape[dim] = len(indices)

            # Get source data
            lattice = self.to_lattice()
            new_lattice = []

            # Generate coordinates for new tensor
            # The trick: we want to iterate over the NEW tensor's layout
            # But capture values from the OLD tensor using 'indices' map

            # Ranges for iteration: same as shape, but for the selection dim
            # we iterate 0..len(indices). We will use this to look up the real index.
            iter_ranges = [range(s) for s in new_shape]

            for coord in itertools.product(*iter_ranges):
                # Map coordinate to source coordinate
                src_coord = list(coord)
                src_coord[dim] = indices[coord[dim]]  # Look up real index

                # Compute flat index
                flat_idx = sum(c * s for c, s in zip(src_coord, strides))
                new_lattice.append(lattice[flat_idx])

            return ResonantTensor.from_golden_exact(
                new_lattice, new_shape, self.get_mode_norms()
            )

    def view(self, *shape: int) -> "ResonantTensor":
        """
        Returns a new tensor with the same data but different shape.

        Args:
            *shape: New shape dimensions

        Returns:
            ResonantTensor with new shape
        """
        # Handle list vs varargs
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            new_shape = list(shape[0])
        else:
            new_shape = list(shape)

        # Handle -1 (infer dimension)
        if -1 in new_shape:
            total_elements = len(self)
            known_elements = 1
            minus_one_idx = -1
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    if minus_one_idx != -1:
                        raise ValueError("Only one dimension can be -1")
                    minus_one_idx = i
                else:
                    known_elements *= dim

            if total_elements % known_elements != 0:
                raise ValueError(
                    f"Shape mismatch: {total_elements} not divisible by {known_elements}"
                )

            new_shape[minus_one_idx] = total_elements // known_elements

        try:
            # Try Rust implementation if available
            result = self._inner.view(new_shape)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback: re-create with new shape
            # Since data is contiguous in C-order, this is just a metadata change
            # for the lattice list

            # Validate size
            current_size = len(self)
            new_size = 1
            for dim in new_shape:
                new_size *= dim

            if current_size != new_size:
                raise ValueError(
                    f"Shape mismatch: cannot reshape {self.shape} ({current_size}) to {new_shape} ({new_size})"
                )

            # Create new tensor using existing lattice data
            # Note: We duplicate data here because we can't easily share the underlying Rust vector
            # passing it back through Python
            lattice = self.to_lattice()

            # Simple metadata change - mode norms must be resized or reused?
            # If shape changes, mode norms flat list is still valid for element-wise ops,
            # but might need re-indexing for spectral operations.
            # For now, reuse existing mode norms as they are flat.
            return ResonantTensor.from_golden_exact(
                lattice, new_shape, self.get_mode_norms()
            )

    def reshape(self, *shape: int) -> "ResonantTensor":
        """Alias for view()."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> "ResonantTensor":
        """
        Returns a tensor that is a transposed version of input.
        The given dimensions are swapped.

        Args:
            dim0: First dimension to swap
            dim1: Second dimension to swap

        Returns:
            Transposed ResonantTensor
        """
        ndim = len(self.shape)
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim

        try:
            result = self._inner.transpose(dim0, dim1)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback
            import itertools

            shape = list(self.shape)

            # Swap dimensions in shape
            new_shape = list(shape)
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]

            # Perform transpose
            lattice = self.to_lattice()
            new_lattice = []

            # Iterate over NEW shape
            # To simulate nested loops of variable depth:
            ranges = [range(s) for s in new_shape]

            for coord in itertools.product(*ranges):
                # Convert new coord to old coord (swap back)
                old_coord = list(coord)
                old_coord[dim0], old_coord[dim1] = old_coord[dim1], old_coord[dim0]

                # Calculate flat index in source
                flat_idx = sum(c * s for c, s in zip(old_coord, strides))
                new_lattice.append(lattice[flat_idx])

            return ResonantTensor.from_golden_exact(
                new_lattice, new_shape, self.get_mode_norms()
            )

    def permute(self, *dims: int) -> "ResonantTensor":
        """
        Returns a view of the original tensor with its dimensions permuted.

        Args:
            *dims: The desired ordering of dimensions

        Returns:
            Permuted ResonantTensor
        """
        # Handle list vs varargs
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            perm = list(dims[0])
        else:
            perm = list(dims)

        if len(perm) != len(self.shape):
            raise ValueError(
                f"Permutation size {len(perm)} must match tensor dimension {len(self.shape)}"
            )

        ndim = len(self.shape)
        # Normalize negative dimensions
        perm = [d + ndim if d < 0 else d for d in perm]

        try:
            result = self._inner.permute(perm)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback
            import itertools

            shape = self.shape

            # Validate permutation
            if set(perm) != set(range(ndim)):
                raise ValueError(f"Invalid permutation {perm} for {ndim} dimensions")

            new_shape = [shape[i] for i in perm]

            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]

            # Perform permutation
            lattice = self.to_lattice()
            new_lattice = []

            ranges = [range(s) for s in new_shape]

            for coord in itertools.product(*ranges):
                # coord is in new order. We need to map back to source order.
                # If new[i] comes from old[perm[i]], then:
                # new_coord[i] corresponds to axis perm[i] in source.
                # So source_coord[perm[i]] = new_coord[i]

                old_coord = [0] * ndim
                for i, p in enumerate(perm):
                    old_coord[p] = coord[i]

                flat_idx = sum(c * s for c, s in zip(old_coord, strides))
                new_lattice.append(lattice[flat_idx])

            return ResonantTensor.from_golden_exact(
                new_lattice, new_shape, self.get_mode_norms()
            )

    def layer_norm(
        self,
        gamma: Optional["ResonantTensor"] = None,
        beta: Optional["ResonantTensor"] = None,
        eps: float = 1e-8,
        golden_target: bool = True,
    ) -> "ResonantTensor":
        """
        Layer normalization across last dimension.

        Args:
            gamma: Optional scale parameter
            beta: Optional shift parameter
            eps: Small constant for numerical stability
            golden_target: If True, scale to target variance = 1/φ

        Returns:
            New normalized tensor

        Examples:
            >>> x = ResonantTensor.randn([8, 16])
            >>> normalized = x.layer_norm()
        """
        gamma_inner = gamma._inner if gamma else None
        beta_inner = beta._inner if beta else None
        result = self._inner.layer_norm(gamma_inner, beta_inner, eps, golden_target)
        return ResonantTensor._wrap(result)

    def mean(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Mean reduction along a dimension.

        Args:
            dim: Dimension to reduce (None = global mean)
            keepdim: Keep reduced dimension with size 1
            precision: Lattice precision

        Returns:
            New tensor with mean values

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> mean_per_sample = x.mean(dim=1)  # Shape: [4]
            >>> global_mean = x.mean()  # Shape: [1]
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.mean(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def var(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Variance reduction along a dimension (population variance).

        Args:
            dim: Dimension to reduce
            keepdim: Keep reduced dimension
            precision: Lattice precision

        Returns:
            New tensor with variance values

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> var_per_sample = x.var(dim=1)
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.var(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def sum(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Sum reduction along a dimension.

        Args:
            dim: Dimension to reduce (None for global sum)
            keepdim: Keep reduced dimension as size 1
            precision: Lattice precision

        Returns:
            New tensor with sum values

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0, 4.0], [2, 2])
            >>> total = x.sum()  # Global sum
            >>> row_sums = x.sum(dim=1)  # Sum along rows
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.sum(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def max(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Max reduction along a dimension.

        Args:
            dim: Dimension to reduce (None for global max)
            keepdim: Keep reduced dimension as size 1
            precision: Lattice precision

        Returns:
            New tensor with max values

        Examples:
            >>> x = ResonantTensor([1.0, 5.0, 3.0, 4.0], [2, 2])
            >>> global_max = x.max()  # Global maximum
            >>> row_maxes = x.max(dim=1)  # Max along rows
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.max(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def min(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Min reduction along a dimension.

        Args:
            dim: Dimension to reduce (None for global min)
            keepdim: Keep reduced dimension as size 1
            precision: Lattice precision

        Returns:
            New tensor with min values

        Examples:
            >>> x = ResonantTensor([1.0, 5.0, 3.0, 4.0], [2, 2])
            >>> global_min = x.min()  # Global minimum
            >>> row_mins = x.min(dim=1)  # Min along rows
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.min(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def prod(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        Product reduction along a dimension.

        Args:
            dim: Dimension to reduce (None for global product)
            keepdim: Keep reduced dimension as size 1
            precision: Lattice precision

        Returns:
            New tensor with product values

        Examples:
            >>> x = ResonantTensor([2.0, 3.0, 4.0, 5.0], [2, 2])
            >>> global_prod = x.prod()  # Global product
            >>> row_prods = x.prod(dim=1)  # Product along rows
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.prod(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def norm(
        self,
        p: float = 2.0,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None,
    ) -> "ResonantTensor":
        """
        p-norm reduction along a dimension.

        Args:
            p: Norm order (2.0 for Euclidean/L2 norm)
            dim: Dimension to reduce (None for global norm)
            keepdim: Keep reduced dimension as size 1
            precision: Lattice precision

        Returns:
            New tensor with norm values

        Examples:
            >>> x = ResonantTensor([3.0, 4.0], [2])  # [3, 4]
            >>> l2_norm = x.norm(p=2.0)  # sqrt(3^2 + 4^2) = 5.0
            >>> l1_norm = x.norm(p=1.0)  # |3| + |4| = 7.0
        """
        if dim is not None:
            ndim = len(self.shape)
            if dim < 0:
                dim += ndim

        result = self._inner.norm(p, dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def masked_select(self, mask: "ResonantTensor") -> "ResonantTensor":
        """
        Select elements where mask is true.

        Args:
            mask: Boolean mask tensor (same shape as self)

        Returns:
            Flattened tensor containing selected elements

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
            >>> mask = ResonantTensor([1.0, 0.0, 1.0, 0.0], [4])  # Select even indices
            >>> result = x.masked_select(mask)
            >>> # result ≈ [1.0, 3.0]
        """
        result = self._inner.masked_select(mask._inner)
        return ResonantTensor._wrap(result)

    def gather(self, dim: int, index: "ResonantTensor") -> "ResonantTensor":
        """
        Gather elements along a dimension using index tensor.

        Args:
            dim: Dimension to gather along
            index: Index tensor (same shape as self)

        Returns:
            Gathered tensor

        Examples:
            >>> x = ResonantTensor([[1.0, 2.0], [3.0, 4.0]], [2, 2])
            >>> indices = ResonantTensor([[1, 0], [0, 1]], [2, 2])
            >>> result = x.gather(1, indices)
            >>> # result contains [2.0, 1.0, 3.0, 4.0]
        """
        result = self._inner.gather(dim, index._inner)
        return ResonantTensor._wrap(result)

    def scatter(self, dim: int, index: "ResonantTensor", src: "ResonantTensor") -> None:
        """
        Scatter source values into self along a dimension.

        Args:
            dim: Dimension to scatter along
            index: Index tensor
            src: Source tensor

        Examples:
            >>> dest = ResonantTensor.zeros([3])
            >>> indices = ResonantTensor([0, 2], [2])
            >>> src = ResonantTensor([10.0, 20.0], [2])
            >>> dest.scatter(0, indices, src)
            >>> # dest now contains [10.0, 0.0, 20.0]
        """
        self._inner.scatter(dim, index._inner, src._inner)

    def scatter_(self, dim: int, index: "ResonantTensor", src: "ResonantTensor") -> None:
        """
        In-place scatter operation.

        Args:
            dim: Dimension to scatter along
            index: Index tensor
            src: Source tensor
        """
        self._inner.scatter(dim, index._inner, src._inner)

    def masked_scatter_(self, mask: "ResonantTensor", src: "ResonantTensor") -> None:
        """
        In-place masked scatter operation.

        Args:
            mask: Boolean mask tensor
            src: Source tensor to scatter
        """
        self._inner.masked_scatter(mask._inner, src._inner)

    def eq(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise equality comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for equal, 0.0 for not equal)

        Examples:
            >>> a = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> b = ResonantTensor([1.0, 2.5, 3.0], [3])
            >>> result = a.eq(b)  # [1.0, 0.0, 1.0]
        """
        result = self._inner.__eq__(other._inner)
        return ResonantTensor._wrap(result)

    def ne(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise inequality comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for not equal, 0.0 for equal)
        """
        result = self._inner.__ne__(other._inner)
        return ResonantTensor._wrap(result)

    def lt(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise less than comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for less than, 0.0 for greater than or equal)
        """
        result = self._inner.__lt__(other._inner)
        return ResonantTensor._wrap(result)

    def le(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise less than or equal comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for less than or equal, 0.0 for greater than)
        """
        result = self._inner.__le__(other._inner)
        return ResonantTensor._wrap(result)

    def gt(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise greater than comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for greater than, 0.0 for less than or equal)
        """
        result = self._inner.__gt__(other._inner)
        return ResonantTensor._wrap(result)

    def ge(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise greater than or equal comparison.

        Args:
            other: Tensor to compare with

        Returns:
            Boolean tensor (1.0 for greater than or equal, 0.0 for less than)
        """
        result = self._inner.__ge__(other._inner)
        return ResonantTensor._wrap(result)

    # =========================================================================
    # Golden Recursion Operations
    # =========================================================================

    def apply_recursion(self) -> None:
        """
        Apply golden recursion map R(n) = floor(φ·n).

        Scales all lattice values by φ (exactly). This is the Fibonacci scaling
        property of the golden lattice: (a, b) → (b, a+b).

        Examples:
            >>> x = ResonantTensor.ones([3])
            >>> x.apply_recursion()  # Values scaled by φ ≈ 1.618
        """
        self._inner.apply_recursion()

    def apply_inverse_recursion(self) -> None:
        """
        Apply inverse golden recursion map R^{-1}(n) = floor(n/φ).

        Scales all lattice values by 1/φ.

        Examples:
            >>> x = ResonantTensor.ones([3])
            >>> x.apply_inverse_recursion()  # Values scaled by 1/φ ≈ 0.618
        """
        self._inner.apply_inverse_recursion()

    def prune_hierarchy(self, q: float, divisor: float = 248.0) -> None:
        """
        Snap values below threshold to zero (hierarchical pruning).

        Threshold is q/divisor. Values with |v| < threshold are set to zero.

        Args:
            q: Base threshold scale
            divisor: Divisor for threshold (default: 248, related to e^π - π ≈ 19.999)

        Examples:
            >>> x = ResonantTensor.randn([100])
            >>> x.prune_hierarchy(q=1.0, divisor=100.0)  # Prune values < 0.01
        """
        self._inner.prune_hierarchy(q, divisor)

    # =========================================================================
    # Operator Overloading
    # =========================================================================

    def __add__(self, other: Union["ResonantTensor", float]) -> "ResonantTensor":
        """
        Addition operator: self + other.

        Supports both tensor-tensor and tensor-scalar addition.

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = a + b  # Element-wise
            >>> d = a + 10.0  # Scalar
        """
        if isinstance(other, ResonantTensor):
            return self.add(other)
        else:
            # For scalar addition, need to implement scalar_add with autograd
            scalar_tensor = ResonantTensor([float(other)] * len(self.get_data_list()), self.shape, device=self.device)
            return self.add(scalar_tensor)

    def __mul__(self, other: Union["ResonantTensor", float]) -> "ResonantTensor":
        """
        Multiplication operator: self * other.

        Supports both tensor-tensor (Hadamard) and tensor-scalar multiplication.

        Examples:
            >>> a = ResonantTensor([2.0, 3.0], [2])
            >>> b = ResonantTensor([4.0, 5.0], [2])
            >>> c = a * b  # Element-wise
            >>> d = a * 2.0  # Scalar
        """
        if isinstance(other, ResonantTensor):
            return self.mul(other)
        else:
            # For scalar multiplication, need to implement with autograd
            scalar_tensor = ResonantTensor([float(other)] * len(self.get_data_list()), self.shape, device=self.device)
            return self.mul(scalar_tensor)

    def __pow__(self, exponent: float) -> "ResonantTensor":
        """Power operator: self ** exponent."""
        return self.pow(exponent)

    def __neg__(self) -> "ResonantTensor":
        """
        Negation operator: -self.

        Examples:
            >>> x = ResonantTensor([1.0, -2.0], [2])
            >>> y = -x  # [-1.0, 2.0]
        """
        return self.negate()

    def __matmul__(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Matrix multiplication operator: self @ other.

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> w = ResonantTensor.randn([20, 10])
            >>> y = x @ w  # [4, 20]
        """
        return self.matmul(other)

    def __getitem__(self, key) -> "ResonantTensor":
        """
        Support advanced indexing and slicing with NumPy-like behavior.

        Supports:
        - Integer indexing: x[0, 1]
        - Slice indexing: x[0:2, :]
        - Boolean masking: x[x > 0.5]
        - Ellipsis: x[..., 0]

        Examples:
            >>> x = ResonantTensor.randn([4, 4])
            >>> a = x[0]           # First row
            >>> b = x[0:2]         # First two rows
            >>> c = x[:, 1]        # Second column
            >>> mask = x > 0.0     # Boolean tensor
            >>> d = x[mask]        # Elements where condition is true
        """
        # Handle boolean indexing (masking)
        if isinstance(key, ResonantTensor):
            # Boolean mask - select elements where mask is true
            return self.masked_select(key)

        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Handle ellipsis by expanding it
        if Ellipsis in key:
            ellipsis_pos = key.index(Ellipsis)
            # Calculate how many dimensions ellipsis should expand to
            remaining_dims = len(self.shape) - (len(key) - 1)  # -1 for ellipsis
            if remaining_dims < 0:
                remaining_dims = 0
            # Replace ellipsis with appropriate number of slice(None)
            expanded_key = (
                key[:ellipsis_pos]
                + (slice(None),) * remaining_dims
                + key[ellipsis_pos + 1 :]
            )
            key = expanded_key

        current_tensor = self
        reduced_dims = []

        if len(key) > len(self.shape):
            raise IndexError(
                f"Too many indices for tensor of dimension {len(self.shape)}"
            )

        for i, k in enumerate(key):
            if i >= len(current_tensor.shape):
                raise IndexError(
                    f"Index {i} out of bounds for {len(current_tensor.shape)}-D tensor"
                )

            current_dim_size = current_tensor.shape[i]

            if isinstance(k, int):
                # Handle negative index
                if k < 0:
                    k += current_dim_size
                if k < 0 or k >= current_dim_size:
                    raise IndexError(
                        f"Index {k} out of bounds for dimension {i} with size {current_dim_size}"
                    )

                # Select single index
                indices = [k]
                current_tensor = current_tensor.index_select(indices, dim=i)
                reduced_dims.append(i)

            elif isinstance(k, slice):
                start, stop, step = k.indices(current_dim_size)
                indices = list(range(start, stop, step))
                if len(indices) != current_dim_size:
                    current_tensor = current_tensor.index_select(indices, dim=i)

            elif isinstance(k, (list, tuple)) and all(
                isinstance(idx, int) for idx in k
            ):
                # List/tuple of indices
                indices = list(k)
                current_tensor = current_tensor.index_select(indices, dim=i)

        # Squeeze out dimensions that were integer-indexed
        if reduced_dims:
            final_shape = [
                d
                for idx, d in enumerate(current_tensor.shape)
                if idx not in reduced_dims
            ]
            if final_shape and final_shape != current_tensor.shape:
                try:
                    current_tensor = current_tensor.view(final_shape)
                except:
                    # If view fails, keep the original shape
                    pass

        return current_tensor

    def __setitem__(self, key, value: Union["ResonantTensor", float, int]) -> None:
        """
        Advanced assignment with NumPy-like indexing.

        Supports the same indexing as __getitem__:
        - Integer indexing: x[0, 1] = 5.0
        - Slice indexing: x[0:2, :] = tensor
        - Boolean masking: x[x > 0.5] = 0.0
        - Ellipsis: x[..., 0] = tensor

        Args:
            key: Index specification
            value: Value to assign (ResonantTensor, float, or int)

        Examples:
            >>> x = ResonantTensor.zeros([4, 4])
            >>> x[0, 0] = 1.0                    # Single element
            >>> x[0:2, :] = ResonantTensor.ones([2, 4])  # Slice assignment
            >>> x[x < 0.5] = 0.0                 # Boolean masking
        """
        # Convert value to ResonantTensor if needed
        if isinstance(value, (int, float)):
            # Broadcast scalar to appropriate shape
            if isinstance(key, ResonantTensor):
                # Boolean mask - count True values
                mask_size = key.sum().get_data_list()[0]
                value_tensor = ResonantTensor([float(value)] * int(mask_size), [int(mask_size)])
            else:
                # For other indexing, create scalar tensor
                value_tensor = ResonantTensor([float(value)], [1])
        elif isinstance(value, ResonantTensor):
            value_tensor = value
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")

        # Handle boolean indexing (masking)
        if isinstance(key, ResonantTensor):
            # Boolean mask assignment
            self.masked_scatter_(key, value_tensor)
            return

        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Handle ellipsis by expanding it
        if Ellipsis in key:
            ellipsis_pos = key.index(Ellipsis)
            remaining_dims = len(self.shape) - (len(key) - 1)
            if remaining_dims < 0:
                remaining_dims = 0
            expanded_key = (
                key[:ellipsis_pos]
                + (slice(None),) * remaining_dims
                + key[ellipsis_pos + 1 :]
            )
            key = expanded_key

        # For now, implement basic slice assignment
        # More complex indexing would require Rust backend support
        if len(key) == 1 and isinstance(key[0], slice):
            # Simple slice assignment like x[0:2] = value
            start, stop, step = key[0].indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("Step slicing not yet supported in setitem")

            # Broadcast value to match slice size
            slice_size = stop - start
            if value_tensor.size() == 1:
                # Broadcast scalar
                broadcast_value = ResonantTensor([value_tensor.get_data_list()[0]] * slice_size, [slice_size])
            elif value_tensor.shape == [slice_size]:
                broadcast_value = value_tensor
            else:
                raise ValueError(f"Shape mismatch: cannot assign {value_tensor.shape} to slice of size {slice_size}")

            # Use scatter to assign
            indices = list(range(start, stop))
            index_tensor = ResonantTensor([float(i) for i in indices], [len(indices)], dtype=int)
            self.scatter_(0, index_tensor, broadcast_value)

        elif len(key) == 2 and all(isinstance(k, slice) for k in key):
            # 2D slice assignment like x[0:2, 1:3] = value
            row_slice, col_slice = key
            row_start, row_stop, row_step = row_slice.indices(self.shape[0])
            col_start, col_stop, col_step = col_slice.indices(self.shape[1])

            if row_step != 1 or col_step != 1:
                raise NotImplementedError("Step slicing not yet supported in setitem")

            slice_shape = [row_stop - row_start, col_stop - col_start]
            expected_size = slice_shape[0] * slice_shape[1]

            if value_tensor.size() == 1:
                # Broadcast scalar
                broadcast_value = ResonantTensor([value_tensor.get_data_list()[0]] * expected_size, slice_shape)
            elif value_tensor.shape == slice_shape:
                broadcast_value = value_tensor
            else:
                raise ValueError(f"Shape mismatch: cannot assign {value_tensor.shape} to slice of shape {slice_shape}")

            # This is complex - would need advanced scatter implementation
            # For now, raise NotImplementedError for complex slicing
            raise NotImplementedError("Complex slice assignment not yet implemented")

        else:
            # Single element assignment or simple indexing
            # Normalize negative indices
            normalized_key = []
            for i, k in enumerate(key):
                if isinstance(k, int) and k < 0:
                    k += self.shape[i]
                normalized_key.append(k)

            if len(normalized_key) == 1 and isinstance(normalized_key[0], int):
                # Single element: x[5] = value
                index_tensor = ResonantTensor([float(normalized_key[0])], [1], dtype=int)
                self.scatter_(0, index_tensor, value_tensor)
            elif len(normalized_key) == 2 and all(isinstance(k, int) for k in normalized_key):
                # 2D element: x[1, 2] = value
                # Convert to flat index
                flat_index = normalized_key[0] * self.shape[1] + normalized_key[1]
                index_tensor = ResonantTensor([float(flat_index)], [1], dtype=int)
                self.scatter_(0, index_tensor, value_tensor)
            else:
                raise NotImplementedError("Complex indexing assignment not yet implemented")

    # =========================================================================
    # File I/O Operations
    # =========================================================================

    def save(self, path: str, format: str = "binary") -> None:
        """
        Save tensor to file.

        Args:
            path: File path to save to
            format: Format to save in ('binary', 'text', 'json', 'npy')

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> tensor.save("model_weights.bin")
            >>> tensor.save("data.json", format="json")
        """
        import json
        import pickle
        import numpy as np

        data = {
            'data': self.get_data_list(),
            'shape': self.shape,
            'mode_norm_sq': None,  # Could be extended to save mode norms
            'precision': self.precision,
            'device': self.device,
            'syntony': self.syntony,
            'phase': self.phase
        }

        if format == "binary":
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "text":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(data, f)
        elif format == "npy":
            np.save(path, np.array(self.get_data_list()).reshape(self.shape))
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str, format: str = "auto", device: str = "cpu") -> "ResonantTensor":
        """
        Load tensor from file.

        Args:
            path: File path to load from
            format: Format to load from ('auto', 'binary', 'text', 'json', 'npy')
            device: Target device

        Returns:
            Loaded ResonantTensor

        Examples:
            >>> tensor = ResonantTensor.load("model_weights.bin")
            >>> tensor = ResonantTensor.load("data.json", format="json")
        """
        import json
        import pickle
        import numpy as np

        if format == "auto":
            if path.endswith('.bin') or path.endswith('.pkl'):
                format = "binary"
            elif path.endswith('.json'):
                format = "json"
            elif path.endswith('.txt'):
                format = "text"
            elif path.endswith('.npy'):
                format = "npy"
            else:
                format = "binary"  # default

        if format == "binary":
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif format in ["text", "json"]:
            with open(path, 'r') as f:
                data = json.load(f)
        elif format == "npy":
            array = np.load(path)
            return cls(array.flatten().tolist(), array.shape.tolist(), device=device)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return cls(
            data['data'],
            data['shape'],
            mode_norm_sq=data.get('mode_norm_sq'),
            precision=data.get('precision', 100),
            device=device
        )

    # =========================================================================
    # Distributed Training Operations
    # =========================================================================

    def all_reduce(self, op: str = "mean") -> "ResonantTensor":
        """
        All-reduce operation across all processes/devices.

        Args:
            op: Reduction operation ("sum", "mean", "max", "min")

        Returns:
            Reduced tensor

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> reduced = tensor.all_reduce("mean")
        """
        from ..distributed import all_reduce
        return all_reduce(self, op)

    def broadcast(self, src: int = 0) -> "ResonantTensor":
        """
        Broadcast tensor from source rank to all other ranks.

        Args:
            src: Source rank

        Returns:
            Broadcast tensor

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> broadcasted = tensor.broadcast(src=0)
        """
        from ..distributed import broadcast
        return broadcast(self, src)

    def gather(self, dst: int = 0) -> List["ResonantTensor"]:
        """
        Gather tensors from all processes to destination rank.

        Args:
            dst: Destination rank

        Returns:
            List of gathered tensors

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> gathered = tensor.gather(dst=0)
        """
        from ..distributed import all_gather
        return all_gather([self])

    def scatter(self, tensor_list: List["ResonantTensor"]) -> "ResonantTensor":
        """
        Scatter tensors from source rank to all other ranks.

        Args:
            tensor_list: List of tensors to scatter

        Returns:
            Scattered tensor for this rank

        Examples:
            >>> tensors = [ResonantTensor.randn([3, 3]) for _ in range(4)]
            >>> scattered = tensors[0].scatter(tensors)
        """
        # Simplified scatter - return first tensor
        # In practice, this would distribute based on rank
        return tensor_list[0] if tensor_list else self

    def reduce_scatter(self, op: str = "sum") -> "ResonantTensor":
        """
        Reduce-scatter operation.

        Args:
            op: Reduction operation

        Returns:
            Result tensor

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> result = tensor.reduce_scatter("sum")
        """
        from ..distributed import reduce_scatter
        return reduce_scatter(self, op)

    def ring_allreduce(self, op: str = "sum") -> "ResonantTensor":
        """
        Ring all-reduce algorithm for efficient distributed reduction.

        Args:
            op: Reduction operation

        Returns:
            Reduced tensor

        Examples:
            >>> tensor = ResonantTensor.randn([3, 3])
            >>> reduced = tensor.ring_allreduce("sum")
        """
        from ..distributed import ring_allreduce
        return ring_allreduce(self, op)

    def to_shards(self, num_shards: int) -> List["ResonantTensor"]:
        """
        Split tensor into shards for tensor parallelism.

        Args:
            num_shards: Number of shards

        Returns:
            List of tensor shards

        Examples:
            >>> tensor = ResonantTensor.randn([4, 8])
            >>> shards = tensor.to_shards(4)
            >>> assert len(shards) == 4
        """
        from ..distributed import tensor_parallelism
        return tensor_parallelism(self, num_shards)

    def register_for_sync(self, name: str) -> None:
        """
        Register tensor for distributed parameter synchronization.

        Args:
            name: Parameter name for synchronization

        Examples:
            >>> param = ResonantTensor.randn([100, 50])
            >>> param.register_for_sync("layer1.weight")
        """
        from ..distributed import register_parameter
        register_parameter(name, self)

    def push_grad(self, name: str) -> None:
        """
        Push gradient for distributed synchronization.

        Args:
            name: Parameter name

        Examples:
            >>> grad = ResonantTensor.randn([100, 50])
            >>> grad.push_grad("layer1.weight")
        """
        from ..distributed import push_gradient
        push_gradient(name, self)

    def __len__(self) -> int:
        """
        Get size of the first dimension.

        Examples:
            >>> x = ResonantTensor.zeros([3, 4])
            >>> assert len(x) == 3
        """
        if not self.shape:
            return 0
        return self.shape[0]

    def __repr__(self) -> str:
        """String representation."""
        return self._inner.__repr__()

    # =========================================================================
    # Internal Helper
    # =========================================================================

    @staticmethod
    def _wrap(inner: _RustResonantTensor, device: str = "cpu") -> "ResonantTensor":
        """
        Wrap a Rust ResonantTensor in Python wrapper.

        Internal method for wrapping Rust results.

        Args:
            inner: The Rust ResonantTensor to wrap
            device: Device string ('cpu' or 'cuda')
        """
        instance = ResonantTensor.__new__(ResonantTensor)
        instance._inner = inner
        instance._device_str = device
        # Initialize autograd attributes with defaults
        instance._grad = None
        instance._requires_grad = False
        instance._grad_fn = None
        instance._ctx = None
        return instance
