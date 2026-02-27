"""
Syntonic - Tensor library for Cosmological and Syntony Recursion Theory

Syntonic provides tensor operations and state evolution primitives for
the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework
used in Cosmological Recursion Theory (CRT) and Syntony Recursion Theory (SRT).

Basic Usage:
    >>> import srt_library.core as syn
    >>> psi = syn.state([1, 2, 3, 4])
    >>> psi.shape
    (4,)
    >>> psi.syntony
    0.5

    >>> # DHSR cycle
    >>> evolved = psi.differentiate().harmonize()
"""

from ._version import __version__, __version_info__

# Flat modules (these are .py files directly in core/)
from . import exact, hypercomplex, linalg
from . import srt_random, fft, jit, distributed, srt_math

# Consciousness/Gnosis (from Rust _core)
from ._core import (
    COLLAPSE_THRESHOLD,
    GNOSIS_GAP,
    gnosis_score,
    is_conscious,
    compute_creativity,
    consciousness_probability,
    # Winding enumeration
    WindingState,
    count_windings,
    enumerate_windings,
    enumerate_windings_by_norm,
    enumerate_windings_exact_norm,
    # Spectral / theta / heat kernel
    heat_kernel_trace,
    heat_kernel_weighted,
    knot_heat_kernel_trace,
    knot_spectral_zeta,
    knot_spectral_zeta_complex,
    partition_function,
    spectral_zeta_weighted,
    theta_series_derivative,
    theta_series_evaluate,
    theta_series_weighted,
    theta_sum_combined,
)

# Core types (from flat files in core/)
from .state import State, state
from .dtype import (
    DType, float32, float64, complex64, complex128,
    int32, int64, winding, get_dtype, promote_dtypes,
)
from .device import (
    Device, cpu, cuda, cuda_is_available, cuda_device_count, device,
)

# Rust backend types and functions
from ._core import (
    RESConfig, ResonantEvolver, RESResult,
    SyntonicExact, SyntonicDual,
    py_gather_f32, py_gather_f64,
    py_load_scatter_gather_kernels,
    py_reduce_consciousness_count_f64,
    py_reduce_e8_norm_f64,
    py_reduce_max_f32, py_reduce_max_f64,
    py_reduce_mean_f32, py_reduce_mean_f64,
    py_reduce_min_f32, py_reduce_min_f64,
    py_reduce_norm_c128,
    py_reduce_norm_l2_f32, py_reduce_norm_l2_f64,
    py_reduce_sum_c128,
    py_reduce_sum_cols_f64,
    py_reduce_sum_f32, py_reduce_sum_f64,
    py_reduce_sum_golden_weighted_f64,
    py_reduce_sum_lucas_shadow_f64,
    py_reduce_sum_mersenne_stable_f64,
    py_reduce_sum_phi_scaled_f64,
    py_reduce_sum_q_corrected_f64,
    py_reduce_sum_rows_f64,
    py_reduce_syntony_deviation_f64,
    py_reduce_syntony_f64,
    py_reduce_variance_golden_target_f64,
    py_scatter_add_f64, py_scatter_f32,
    srt_memory_resonance,
    srt_pool_stats,
    srt_reserve_memory,
    srt_transfer_stats,
    srt_wait_for_resonance,
)

# Extended Hierarchy (from theory_unique_components)
try:
    from srt_library.theory_unique_components.crt.extended_hierarchy import (
        apply_collapse_threshold,
        apply_coxeter_kissing,
        apply_e7_correction,
    )
except ImportError:
    apply_collapse_threshold = None
    apply_coxeter_kissing = None
    apply_e7_correction = None

# Exact arithmetic re-exports
from .exact import (
    E_STAR_NUMERIC, PHI, PHI_INVERSE, PHI_NUMERIC, PHI_SQUARED,
    Q_DEFICIT_NUMERIC, STRUCTURE_DIMENSIONS,
    GoldenExact, Rational,
    correction_factor, fibonacci, golden_number, lucas,
)

# Exceptions
from .exceptions import (
    DeviceError, DTypeError, LinAlgError, ShapeError, SyntonicError,
)

# Hypercomplex re-exports
from .hypercomplex import (
    Octonion, Quaternion, octonion, quaternion,
)

# Core classes and functions
from .nn.resonant_tensor import ResonantTensor

# Neural Networks submodule
try:
    from . import nn
except (ImportError, NameError):
    nn = None

__all__ = [
    # Version
    "__version__", "__version_info__",
    # State
    "State", "state", "ResonantTensor",
    "RESConfig", "RESResult", "ResonantEvolver",
    # DTypes
    "DType", "float32", "float64", "complex64", "complex128",
    "int32", "int64", "winding", "get_dtype", "promote_dtypes",
    # CUDA ops
    "py_gather_f64", "py_gather_f32", "py_scatter_f32",
    "py_scatter_add_f64",
    "py_reduce_sum_f64", "py_reduce_sum_f32",
    "py_reduce_mean_f64", "py_reduce_mean_f32",
    "py_reduce_max_f64", "py_reduce_max_f32",
    "py_reduce_min_f32", "py_reduce_min_f64",
    "py_reduce_norm_l2_f32", "py_reduce_norm_l2_f64",
    "py_reduce_sum_golden_weighted_f64",
    "py_reduce_sum_rows_f64", "py_reduce_sum_cols_f64",
    "py_reduce_sum_phi_scaled_f64",
    "py_reduce_syntony_f64",
    "py_reduce_sum_mersenne_stable_f64",
    "py_reduce_variance_golden_target_f64",
    "py_reduce_sum_lucas_shadow_f64",
    "py_reduce_syntony_deviation_f64",
    "py_reduce_consciousness_count_f64",
    "py_reduce_sum_q_corrected_f64",
    "py_reduce_e8_norm_f64",
    "py_reduce_sum_c128", "py_reduce_norm_c128",
    "py_load_scatter_gather_kernels",
    # Devices
    "Device", "cpu", "cuda", "cuda_is_available", "cuda_device_count",
    "srt_transfer_stats", "srt_reserve_memory",
    "srt_wait_for_resonance", "srt_pool_stats",
    "srt_memory_resonance", "device",
    # Exceptions
    "SyntonicError", "DeviceError", "DTypeError", "ShapeError", "LinAlgError",
    # Submodules
    "linalg", "hypercomplex", "exact", "nn",
    # Hypercomplex types
    "Quaternion", "Octonion", "quaternion", "octonion",
    # Exact arithmetic
    "GoldenExact", "SyntonicExact", "SyntonicDual", "Rational",
    "PHI", "PHI_SQUARED", "PHI_INVERSE", "PHI_NUMERIC",
    "E_STAR_NUMERIC", "Q_DEFICIT_NUMERIC", "STRUCTURE_DIMENSIONS",
    "fibonacci", "lucas", "correction_factor", "golden_number",
    # Consciousness
    "COLLAPSE_THRESHOLD", "GNOSIS_GAP",
    "is_conscious", "gnosis_score", "compute_creativity",
    "consciousness_probability",
    # Extended Hierarchy
    "apply_e7_correction", "apply_collapse_threshold", "apply_coxeter_kissing",
]
