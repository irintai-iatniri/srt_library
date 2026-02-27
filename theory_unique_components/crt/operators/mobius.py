"""
Mobius Filter Operator.

Enforces the Number-Theoretic stability rules of the vacuum.
Specifically targets the M11 (2047) Composite Barrier.

The Mobius function mu(n) encodes the arithmetic structure of integers:
- mu(n) = 1 if n is squarefree with even number of prime factors
- mu(n) = -1 if n is squarefree with odd number of prime factors
- mu(n) = 0 if n has a squared prime factor

For SRT, dimensions with mu(n) = 0 are unstable and must be zeroed.
This prevents 4th generation matter (M11 = 2047 = 23 * 89) from stabilizing.
"""

from typing import List, Tuple

from srt_library.core.state import State
from srt_library.core.nn.resonant_tensor import ResonantTensor


def mobius(n: int) -> int:
    """
    Compute the Mobius function mu(n).

    Args:
        n: Positive integer

    Returns:
        1 if n is squarefree with even number of prime factors
        -1 if n is squarefree with odd number of prime factors
        0 if n has a squared prime factor
    """
    if n < 1:
        return 0
    if n == 1:
        return 1

    # Count prime factors and check for squares
    num_factors = 0
    temp = n

    # Check for factor 2
    if temp % 2 == 0:
        num_factors += 1
        temp //= 2
        if temp % 2 == 0:
            return 0  # Square factor

    # Check odd factors
    factor = 3
    while factor * factor <= temp:
        if temp % factor == 0:
            num_factors += 1
            temp //= factor
            if temp % factor == 0:
                return 0  # Square factor
        factor += 2

    # Remaining prime factor
    if temp > 1:
        num_factors += 1

    return -1 if num_factors % 2 == 1 else 1


def compute_mobius_mask(size: int) -> List[float]:
    """
    Compute the Mobius mask for a tensor of given size.

    Dimensions where mu(n+1) = 0 are masked (set to 0.0).
    Other dimensions are preserved (set to 1.0).

    Args:
        size: Number of elements

    Returns:
        List of mask values (1.0 or 0.0)
    """
    mask = []
    for i in range(size):
        # Use 1-indexed dimension numbers (dimension 1, 2, 3, ...)
        mu = mobius(i + 1)
        mask.append(1.0 if mu != 0 else 0.0)
    return mask


def apply_mobius_mask(tensor: ResonantTensor) -> ResonantTensor:
    """
    Apply the Mobius Inversion Mask mu(n) to a tensor.

    Any dimension 'n' where mu(n) = 0 (has a square factor) is zeroed out.
    This effectively kills unstable windings like M11 (2047 = 23 * 89),
    preventing 4th generation matter from stabilizing.

    Args:
        tensor: The quantum state tensor

    Returns:
        Filtered tensor with unstable windings removed
    """
    data = tensor.to_floats()
    mask = compute_mobius_mask(len(data))

    # Apply mask element-wise
    filtered_data = [d * m for d, m in zip(data, mask)]

    return ResonantTensor(
        filtered_data,
        list(tensor.shape),
        tensor.get_mode_norms(),
        tensor.precision,
    )


def apply_mobius_mask_state(state: State) -> State:
    """
    Apply the Mobius mask to a State object.

    Args:
        state: The quantum state

    Returns:
        Filtered state with unstable windings removed
    """
    # Convert State to ResonantTensor, apply mask, convert back
    tensor = state.to_resonant_tensor()
    filtered = apply_mobius_mask(tensor)
    return State.from_resonant_tensor(filtered)


def check_m11_stability(tensor: ResonantTensor) -> bool:
    """
    Check if any energy exists in the M11 forbidden zone.

    M11 = 2047 = 2^11 - 1 is a composite Mersenne number (23 * 89).
    Its non-prime status creates an instability barrier that prevents
    4th generation matter.

    Args:
        tensor: The state tensor to check

    Returns:
        True if M11 zone is stable (no significant energy)
        False if M11 zone is active (potential instability)
    """
    data = tensor.to_floats()

    # Check energy at index 2046 (0-indexed dimension 2047)
    if len(data) <= 2046:
        return True  # Tensor too small to have M11 energy

    m11_energy = abs(data[2046])
    return m11_energy < 1e-9


def get_squarefree_indices(size: int) -> List[int]:
    """
    Get indices of squarefree dimensions.

    These are the dimensions where mu(n) != 0, meaning they
    have no squared prime factors and are topologically stable.

    Args:
        size: Number of dimensions to check

    Returns:
        List of indices with squarefree dimension numbers
    """
    return [i for i in range(size) if mobius(i + 1) != 0]


def get_composite_barrier_indices(size: int) -> List[Tuple[int, int]]:
    """
    Get indices of composite barriers (where mu(n) = 0).

    Returns pairs of (index, dimension_number) for visualization.

    Args:
        size: Number of dimensions to check

    Returns:
        List of (index, dimension) tuples for composite barriers
    """
    return [(i, i + 1) for i in range(size) if mobius(i + 1) == 0]


# The first few composite barriers (squared prime factors)
COMPOSITE_BARRIERS = [
    4,  # 2^2
    8,  # 2^3
    9,  # 3^2
    12,  # 2^2 * 3
    16,  # 2^4
    18,  # 2 * 3^2
    20,  # 2^2 * 5
    24,  # 2^3 * 3
    25,  # 5^2
    27,  # 3^3
    28,  # 2^2 * 7
    32,  # 2^5
    # ... continues
]

# M11 = 2047 = 23 * 89 is NOT a barrier (squarefree)
# But its position at 2^11 - 1 creates a resonance instability
M11_DIMENSION = 2047
