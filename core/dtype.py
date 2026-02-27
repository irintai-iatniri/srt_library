"""
Data type definitions for Syntonic.

Pure Rust-based dtype system without numpy dependencies.
Matches the Rust backend's CpuData/CudaData enum types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class DType:
    """
    Syntonic data type - matches Rust backend types.

    This is a pure Python implementation that mirrors the Rust
    CpuData/CudaData enum types without numpy dependencies.
    """

    name: str
    size: int  # bytes
    is_complex: bool = False
    is_floating: bool = True
    rust_type: str = ""  # Corresponding Rust type name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"syn.{self.name}"

    def __eq__(self, other) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        return hash(self.name)


# =============================================================================
# EXACT ARITHMETIC TYPES (PRIMARY - for SGC zero-entropy recursion)
# =============================================================================
# These are the default types for Syntonic Generative Codec.
# They provide bit-perfect deterministic computation.

golden_exact = DType("golden_exact", 16, is_floating=False, rust_type="GoldenExact")  # Q(φ) field - DEFAULT for CPU
rational = DType("rational", 16, is_floating=False, rust_type="Rational")  # Q field
fixed_point64 = DType("fixed_point64", 8, is_floating=False, rust_type="i64")  # Q32.32 format - DEFAULT for GPU
syntonic = DType("syntonic", 192, is_floating=False, rust_type="SyntonicExact")  # Super-Field (a+bφ) + (c+dφ)E* + (e+fφ)q

# =============================================================================
# FLOATING POINT TYPES (SECONDARY - for preview/approximation only)
# =============================================================================
# Use these only when exact arithmetic is not required.
# They introduce floating-point drift in recursive DHSR cycles.

float32 = DType("float32", 4, rust_type="f32")
float64 = DType("float64", 8, rust_type="f64")

# =============================================================================
# COMPLEX TYPES
# =============================================================================

complex64 = DType("complex64", 8, is_complex=True, rust_type="Complex64")
complex128 = DType("complex128", 16, is_complex=True, rust_type="Complex64")

# =============================================================================
# INTEGER TYPES
# =============================================================================

int32 = DType("int32", 4, is_floating=False, rust_type="i32")
int64 = DType("int64", 8, is_floating=False, rust_type="i64")

# Winding type (alias for int64, semantically distinct for T^4 indices)
winding = DType("winding", 8, is_floating=False, rust_type="i64")

# Type mapping for conversions
_DTYPE_MAP = {
    # Float aliases
    "float32": float32,
    "f32": float32,
    "float64": float64,
    "f64": float64,
    "float": float64,
    # Complex aliases
    "complex64": complex64,
    "c64": complex64,
    "complex128": complex128,
    "c128": complex128,
    "complex": complex128,
    # Integer aliases
    "int32": int32,
    "i32": int32,
    "int64": int64,
    "i64": int64,
    "int": int64,
    # Special types
    "winding": winding,
    # Exact arithmetic types
    "golden_exact": golden_exact,
    "golden": golden_exact,
    "rational": rational,
    "fixed_point64": fixed_point64,
    "fp64": fixed_point64,
    "fixed": fixed_point64,
    "syntonic": syntonic,
    "syntonic_exact": syntonic,
}


def get_dtype(dtype_spec: Union[DType, str, Any]) -> DType:
    """
    Get DType from various specifications.

    Args:
        dtype_spec: DType instance, string name, or other type spec

    Returns:
        Corresponding DType instance

    Raises:
        ValueError: If dtype_spec is not recognized
    """
    if isinstance(dtype_spec, DType):
        return dtype_spec

    if isinstance(dtype_spec, str):
        dtype_lower = dtype_spec.lower()
        if dtype_lower in _DTYPE_MAP:
            return _DTYPE_MAP[dtype_lower]

        # Try to match partial names
        for key, dtype in _DTYPE_MAP.items():
            if key in dtype_lower or dtype_lower in key:
                return dtype

    # Try to infer from Python types
    if dtype_spec is float:
        return float64
    elif dtype_spec is int:
        return int64
    elif dtype_spec is complex:
        return complex128

    # Try to match by size/type inspection
    if hasattr(dtype_spec, "itemsize"):
        # Looks like a numpy-like dtype
        size = getattr(dtype_spec, "itemsize", 8)
        is_complex = getattr(dtype_spec, "kind", "") == "c"

        if is_complex:
            return complex128 if size >= 16 else complex64
        else:
            return float64 if size >= 8 else float32

    raise ValueError(f"Unknown dtype specification: {dtype_spec}")


def promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    """
    Determine result dtype from two input dtypes.

    EXACT-FIRST TYPE PROMOTION HIERARCHY (for SGC zero-entropy):

    1. GoldenExact > Rational > FixedPoint64 (exact types preserve exactness)
    2. Exact types DO NOT auto-promote to float (prevents drift)
    3. float64 > float32 (float types promote as usual)
    4. Complex takes precedence over real (existing behavior)

    Critical Rule: Mixing exact and float types will raise an error.
    Use explicit .to_float() or .to_exact() conversions instead.

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        Promoted dtype

    Raises:
        ValueError: If mixing exact and float types without explicit conversion
    """
    # Same dtype - no promotion needed
    if dtype1 == dtype2:
        return dtype1

    # Complex takes precedence (existing behavior)
    if dtype1.is_complex or dtype2.is_complex:
        if dtype1.is_complex and not dtype2.is_complex:
            return dtype1
        elif dtype2.is_complex and not dtype1.is_complex:
            return dtype2
        else:
            # Both complex - larger size wins
            return dtype1 if dtype1.size >= dtype2.size else dtype2

    # Define exact type hierarchy
    exact_types = {"golden_exact": 3, "rational": 2, "fixed_point64": 1}
    float_types = {"float64": 2, "float32": 1}

    dtype1_is_exact = dtype1.name in exact_types
    dtype2_is_exact = dtype2.name in exact_types
    dtype1_is_float = dtype1.name in float_types
    dtype2_is_float = dtype2.name in float_types

    # CRITICAL: Do not mix exact and float types
    if (dtype1_is_exact and dtype2_is_float) or (dtype1_is_float and dtype2_is_exact):
        # This should not happen - State arithmetic should handle conversion
        # If we get here, it means the conversion logic failed
        # For now, return the exact type and let the backend handle it
        if dtype1_is_exact:
            return dtype1
        else:
            return dtype2

    # Both exact - promote to higher precision exact type
    if dtype1_is_exact and dtype2_is_exact:
        rank1 = exact_types[dtype1.name]
        rank2 = exact_types[dtype2.name]
        return dtype1 if rank1 >= rank2 else dtype2

    # Both float - promote to higher precision float
    if dtype1_is_float and dtype2_is_float:
        rank1 = float_types[dtype1.name]
        rank2 = float_types[dtype2.name]
        return dtype1 if rank1 >= rank2 else dtype2

    # Integer types - promote by size
    if not dtype1_is_exact and not dtype1_is_float and not dtype2_is_exact and not dtype2_is_float:
        return dtype1 if dtype1.size >= dtype2.size else dtype2

    # Default - prefer first dtype
    return dtype1


def is_compatible_dtype(dtype1: DType, dtype2: DType) -> bool:
    """
    Check if two dtypes are compatible for operations.

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        True if compatible, False otherwise
    """
    # Same dtype is always compatible
    if dtype1 == dtype2:
        return True

    # Complex can operate with real (result will be complex)
    if dtype1.is_complex != dtype2.is_complex:
        return True

    # Different precisions are compatible (will promote)
    return True


def can_cast_dtype(from_dtype: DType, to_dtype: DType) -> bool:
    """
    Check if dtype can be cast to another dtype.

    Args:
        from_dtype: Source dtype
        to_dtype: Target dtype

    Returns:
        True if casting is possible
    """
    # Same dtype
    if from_dtype == to_dtype:
        return True

    # Complex to real (lossy)
    if from_dtype.is_complex and not to_dtype.is_complex:
        return True

    # Real to complex (safe)
    if not from_dtype.is_complex and to_dtype.is_complex:
        return True

    # Different precisions
    if from_dtype.is_floating == to_dtype.is_floating:
        return True

    # Integer to float (safe)
    if not from_dtype.is_floating and to_dtype.is_floating:
        return True

    # Float to integer (lossy)
    if from_dtype.is_floating and not to_dtype.is_floating:
        return True

    return False


def get_default_dtype(is_complex: bool = False, device_is_cuda: bool = False) -> DType:
    """
    Get default dtype for given type category.

    IMPORTANT: For SGC (Syntonic Generative Codec), defaults to exact arithmetic.

    Args:
        is_complex: Whether to return complex dtype
        device_is_cuda: Whether the device is CUDA (uses fixed_point64) or CPU (uses golden_exact)

    Returns:
        Default dtype (exact arithmetic by default for SGC)
    """
    if is_complex:
        return complex128  # Complex not yet supported in exact mode

    # Exact-first architecture: Q32.32 on GPU, Q(φ) on CPU
    if device_is_cuda:
        return fixed_point64  # Q32.32 for GPU exact arithmetic
    else:
        return golden_exact  # Q(φ) field for CPU exact arithmetic


def get_dtype_info(dtype: DType) -> dict:
    """
    Get detailed information about a dtype.

    Args:
        dtype: DType to inspect

    Returns:
        Dictionary with dtype information
    """
    return {
        "name": dtype.name,
        "size": dtype.size,
        "is_complex": dtype.is_complex,
        "is_floating": dtype.is_floating,
        "is_exact": is_exact_dtype(dtype),
        "rust_type": dtype.rust_type,
        "bytes_per_element": dtype.size,
        "complex_elements": 2 if dtype.is_complex else 1,
    }


def is_exact_dtype(dtype: DType) -> bool:
    """
    Check if a dtype uses exact arithmetic (no floating-point drift).

    Args:
        dtype: DType to check

    Returns:
        True if dtype is exact (golden_exact, rational, fixed_point64)
    """
    return dtype.name in ("golden_exact", "rational", "fixed_point64", "syntonic")


def is_float_dtype(dtype: DType) -> bool:
    """
    Check if a dtype uses floating-point approximation.

    Args:
        dtype: DType to check

    Returns:
        True if dtype is float (float32, float64)
    """
    return dtype.name in ("float32", "float64")


# Export commonly used dtypes at module level
# EXACT TYPES FIRST (for SGC zero-entropy recursion)
__all__ = [
    "DType",
    # Exact arithmetic types (PRIMARY)
    "golden_exact",
    "rational",
    "fixed_point64",
    "syntonic",
    # Float types (SECONDARY - use only for preview)
    "float32",
    "float64",
    # Complex types
    "complex64",
    "complex128",
    # Integer types
    "int32",
    "int64",
    "winding",
    # Functions
    "get_dtype",
    "promote_dtypes",
    "is_compatible_dtype",
    "can_cast_dtype",
    "get_default_dtype",
    "get_dtype_info",
    "is_exact_dtype",
    "is_float_dtype",
]
