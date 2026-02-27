"""
Syntonic Random Number Generation

Provides SRT-based random number generation with golden ratio sequences,
seeded state management, and various distribution functions.
"""

from __future__ import annotations

from . import srt_math
from typing import List, Optional, Union

from .exact import PHI, PHI_NUMERIC, GoldenExact
from .nn.resonant_tensor import ResonantTensor


class RandomState:
    """
    SRT-based random state with golden ratio sequences.

    Uses golden ratio arithmetic for deterministic but aperiodic sequences
    that maintain mathematical properties across scales.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random state.

        Args:
            seed: Optional integer seed for reproducibility
        """
        # Use time-based seed if none provided (no stdlib random)
        if seed is None:
            import time
            seed = int(time.time() * 1000000) % (2**32 - 1)
        self.seed = seed
        self._state = float(self.seed)  # Use float for state
        self._index = 0

    def _next_golden(self) -> float:
        """Generate next value in golden ratio sequence."""
        # Use golden ratio multiplication for aperiodic sequence
        from .exact import PHI_NUMERIC
        self._state = self._state * PHI_NUMERIC
        self._index += 1
        return self._state

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate uniform random float in [low, high)."""
        val = self._next_golden()
        # Map to [0, 1) using fractional part
        uniform_val = val - srt_math.floor(val)
        return low + uniform_val * (high - low)

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Generate normal distributed random float."""
        # Box-Muller transform using two uniform samples
        u1 = self.uniform(0, 1)
        u2 = self.uniform(0, 1)

        # Avoid log(0) edge case
        u1 = max(u1, 1e-10)
        u2 = max(u2, 1e-10)

        z0 = srt_math.sqrt(-2.0 * srt_math.log(u1)) * srt_math.cos(2.0 * srt_math.pi * u2)
        return mean + z0 * std

    def randint(self, low: int, high: int) -> int:
        """Generate random integer in [low, high)."""
        uniform_val = self.uniform(0, 1)
        return low + int(uniform_val * (high - low))

    def choice(self, seq: List) -> any:
        """Choose random element from sequence."""
        idx = self.randint(0, len(seq))
        return seq[idx]

    def shuffle(self, seq: List) -> List:
        """Shuffle sequence in place using SRT-based randomness."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq


# Global random state
_global_random_state = RandomState()


def seed(seed: int) -> None:
    """
    Set global random seed.

    Args:
        seed: Integer seed value
    """
    global _global_random_state
    _global_random_state = RandomState(seed)


def get_state() -> RandomState:
    """Get current global random state."""
    return _global_random_state


def set_state(state: RandomState) -> None:
    """Set global random state."""
    global _global_random_state
    _global_random_state = state


def randn(shape: List[int], mean: float = 0.0, std: float = 1.0,
          device: str = "cpu") -> ResonantTensor:
    """
    Create tensor with normal distributed random values.

    Args:
        shape: Shape of the tensor
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
        device: Target device

    Returns:
        ResonantTensor with random normal values

    Examples:
        >>> x = randn([3, 3])  # Standard normal
        >>> y = randn([2, 4], mean=1.0, std=0.5)  # Custom parameters
    """
    size = 1
    for dim in shape:
        size *= dim

    data = [_global_random_state.normal(mean, std) for _ in range(size)]
    return ResonantTensor(data, shape, device=device)


def uniform(shape: List[int], low: float = 0.0, high: float = 1.0,
            device: str = "cpu") -> ResonantTensor:
    """
    Create tensor with uniform distributed random values.

    Args:
        shape: Shape of the tensor
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        device: Target device

    Returns:
        ResonantTensor with random uniform values
    """
    size = 1
    for dim in shape:
        size *= dim

    data = [_global_random_state.uniform(low, high) for _ in range(size)]
    return ResonantTensor(data, shape, device=device)


def randint(shape: List[int], low: int, high: int, device: str = "cpu") -> ResonantTensor:
    """
    Create tensor with random integer values.

    Args:
        shape: Shape of the tensor
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        device: Target device

    Returns:
        ResonantTensor with random integer values
    """
    size = 1
    for dim in shape:
        size *= dim

    data = [float(_global_random_state.randint(low, high)) for _ in range(size)]
    return ResonantTensor(data, shape, device=device)


def rand(shape: List[int], device: str = "cpu") -> ResonantTensor:
    """
    Create tensor with uniform random values in [0, 1).

    Args:
        shape: Shape of the tensor
        device: Target device

    Returns:
        ResonantTensor with random values in [0, 1)
    """
    return uniform(shape, 0.0, 1.0, device=device)


def rand_srt(shape: List[int], device: str = "cpu") -> ResonantTensor:
    """
    Create tensor using pure SRT golden ratio sequences.

    Generates values using golden ratio arithmetic for mathematically
    pure random sequences with special properties.

    Args:
        shape: Shape of the tensor
        device: Target device

    Returns:
        ResonantTensor with SRT-based random values
    """
    size = 1
    for dim in shape:
        size *= dim

    data = []
    state = GoldenExact.from_float(1.0)

    for _ in range(size):
        # Generate sequence using golden ratio powers
        state = state * GoldenExact.golden_ratio()
        # Use fractional part for [0, 1) range
        val = state.to_float()
        fractional = val - srt_math.floor(val)
        data.append(fractional)

    return ResonantTensor(data, shape, device=device)


def choice(seq: List, size: Optional[int] = None, replace: bool = True) -> Union[any, List]:
    """
    Choose random elements from sequence.

    Args:
        seq: Sequence to choose from
        size: Number of elements to choose (None for single element)
        replace: Whether to sample with replacement

    Returns:
        Single element or list of elements
    """
    if size is None:
        return _global_random_state.choice(seq)

    if replace:
        return [_global_random_state.choice(seq) for _ in range(size)]
    else:
        # Sample without replacement
        if size > len(seq):
            raise ValueError("Cannot sample more elements than available without replacement")

        indices = list(range(len(seq)))
        selected_indices = []
        for _ in range(size):
            idx = _global_random_state.randint(0, len(indices))
            selected_indices.append(indices.pop(idx))

        return [seq[i] for i in selected_indices]


def shuffle(seq: List) -> List:
    """
    Shuffle sequence using SRT-based randomness.

    Args:
        seq: Sequence to shuffle

    Returns:
        Shuffled sequence (in-place modification)
    """
    return _global_random_state.shuffle(seq)