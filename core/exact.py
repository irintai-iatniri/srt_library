"""
Exact arithmetic for Syntonic.

This module provides exact arithmetic over:
- Q: Rational numbers (arbitrary precision)
- Q(φ): Golden field extension (a + b·φ where φ = (1+√5)/2)

The golden ratio φ satisfies the fundamental identity:
    φ² = φ + 1

All computations in this module are exact (no floating-point errors).
Use .eval() to convert to float when needed.

SRT Constants:
- PHI: Golden ratio φ ≈ 1.618033988749895
- PHI_SQUARED: φ² = 1 + φ ≈ 2.618033988749895
- PHI_INVERSE: 1/φ = φ - 1 ≈ 0.618033988749895
- E_STAR_NUMERIC: e^π - π ≈ 19.999099979189474
- Q_DEFICIT_NUMERIC: Syntony deficit q ≈ 0.027395146920

Usage:
    >>> import syntonic as syn
    >>> phi = syn.PHI
    >>> phi_squared = phi * phi
    >>> phi_squared.eval() == phi.eval() + 1  # φ² = φ + 1
    True
    >>> syn.fibonacci(10)
    55
"""

import math

# Try to import PySymExpr if available
from ._core import (
    FundamentalConstant,
    GoldenExact,
    Rational,
    SymExpr,
    srt_correction_factor,
    srt_e,
    srt_phi,
    srt_phi_inv,
    srt_pi,
    srt_q_deficit,
    srt_structure_dimension,
)

# =============================================================================
# Exact Golden Constants
# =============================================================================

# φ = (1 + √5) / 2 ≈ 1.618033988749895
PHI = GoldenExact.golden_ratio()

# φ² = φ + 1 ≈ 2.618033988749895
PHI_SQUARED = GoldenExact.golden_squared()

# φ̂ = 1/φ = φ - 1 ≈ 0.618033988749895
PHI_INVERSE = GoldenExact.coherence_parameter()


# =============================================================================
# Exact SRT Transcendental Constants (SymExpr - Symbolic Arithmetic)
# =============================================================================
# These are transcendental constants that cannot be represented as Rationals
# or GoldenExact. They are kept symbolic using SymExpr which supports full
# algebraic arithmetic while preserving exact relationships.
#
# The Hierarchy of Truth:
#   Rational       → Base atoms (a/b)
#   GoldenExact    → Q(φ) = a + b·φ
#   SymExpr        → Symbolic expressions with π, e, E*, q
#
# SymExpr allows expressions like: (5/2) * E_STAR + PHI * Q_DEFICIT
# without any floating-point approximation.

# q = universal syntony deficit (transcendental)
# Derived from heat kernel regularization on W⁴ × E₈
Q_DEFICIT = SymExpr.q()

# E* = e^π - π (transcendental)
# Spectral Möbius constant from heat kernel regularization
E_STAR = SymExpr.e_star()

# =============================================================================
# Numeric Constants (float approximations - DEPRECATED for SGC)
# =============================================================================
#
# WARNING: For Syntonic Generative Codec (SGC), use exact constants instead:
#   - PHI (not PHI_NUMERIC)
#   - PHI_INVERSE (not PHI_NUMERIC)
#   - Use .eval() explicitly when you need float
#
# Float constants are provided for backward compatibility only.
# They introduce floating-point drift in recursive DHSR cycles.
#
# Example:
#   OLD: mass = PHI_NUMERIC * base  # Float drift
#   NEW: mass = (PHI * base_exact).eval()  # Exact then evaluate
#

# Golden ratio as float
PHI_NUMERIC = srt_phi()  # 1.618033988749895

# Spectral constant E* = e^π - π
E_STAR_NUMERIC = math.exp(math.pi) - math.pi  # 19.999099979189474

# Universal syntony deficit q
Q_DEFICIT_NUMERIC = srt_q_deficit()  # 0.027395146920

# π constant
PI_NUMERIC = srt_pi()  # 3.141592653589793

# Euler's number e
E_NUMERIC = srt_e()  # 2.718281828459045


# =============================================================================
# Helper Functions for Gradual Migration
# =============================================================================


def get_phi_float() -> float:
    """
    DEPRECATED: Get golden ratio as float.

    Use PHI.eval() explicitly instead for clarity.

    Returns:
        φ ≈ 1.618033988749895
    """
    return PHI.eval()


def get_phi_inv_float() -> float:
    """
    DEPRECATED: Get 1/φ as float.

    Use PHI_INVERSE.eval() explicitly instead.

    Returns:
        1/φ ≈ 0.618033988749895
    """
    return PHI_INVERSE.eval()


# =============================================================================
# Prime Sequence Constants
# =============================================================================

# Fermat primes - Gauge force selection rules
FERMAT_PRIMES = [3, 5, 17, 257, 65537]  # F_0 through F_4
FERMAT_INDICES = [0, 1, 2, 3, 4]  # Physics stops at n=5 (composite)

# Mersenne primes - Matter stability rules
MERSENNE_PRIMES = [3, 7, 31, 127]  # M_2, M_3, M_5, M_7
MERSENNE_EXPONENTS = [2, 3, 5, 7]  # p values where 2^p - 1 is prime
M11_BARRIER = 2047  # 23 × 89 - Why 4th generation fails

# Lucas primes - Shadow/dark sector
LUCAS_SEQUENCE = [
    2,
    1,
    3,
    4,
    7,
    11,
    18,
    29,
    47,
    76,
    123,
    199,
    322,
    521,
    843,
    1364,
    2207,
    3571,
]
LUCAS_PRIMES = [2, 3, 7, 11, 29, 47, 199, 521, 2207, 3571]

# Kissing number (Collapse threshold)
D4_KISSING = 24  # K(D₄) - Wave function collapse threshold


# =============================================================================
# Structure Dimensions (for correction factors)
# =============================================================================
# Structure Dimensions (for correction factors)
# =============================================================================

STRUCTURE_DIMENSIONS = {
    "E8_dim": 248,  # dim(E₈) adjoint representation
    "E8_roots": 240,  # |Φ(E₈)| all roots
    "E8_positive": 120,  # |Φ⁺(E₈)| positive roots (chiral)
    "E6_dim": 78,  # dim(E₆) adjoint
    "E6_positive": 36,  # |Φ⁺(E₆)| = Golden Cone roots
    "E6_fundamental": 27,  # dim(27_E₆) fundamental representation
    "D4_kissing": 24,  # K(D₄) kissing number (consciousness threshold)
    "G2_dim": 14,  # dim(G₂) = Aut(O) automorphisms of octonions
    "E7_dim": 133,  # dim(E₇) adjoint
    "E7_roots": 126,  # |Φ(E₇)| all roots
    "E7_positive": 63,  # |Φ⁺(E₇)| positive roots
    "E7_fundamental": 56,  # dim(E₇ fundamental)
    "F4_dim": 52,  # dim(F₄)
    "D4_adjoint": 28,  # dim(SO(8)) = 28
}

# Index mapping for srt_correction_factor
_STRUCTURE_INDICES = {
    "E8_dim": 0,
    "E8_roots": 1,
    "E8_positive": 2,
    "E6_dim": 3,
    "E6_positive": 4,
    "E6_fundamental": 5,
    "D4_kissing": 6,
    "G2_dim": 7,
}

# =============================================================================
# Universal Syntony Correction Hierarchy
# =============================================================================

# Complete geometric factors registry: 60+ factors from E₈ → E₇ → E₆ → SM breaking chain
# Each factor corresponds to a specific geometric structure in the W⁴ × E₈ framework
CORRECTION_FACTORS = {
    # Level 0-10: Ultra-precision (fundamental structures)
    "q_cubed": (Q_DEFICIT_NUMERIC**3, "Third-order vacuum", "Three-loop universal"),
    "q_1000": (Q_DEFICIT_NUMERIC / 1000, "h(E₈)³/27", "Fixed-point stability (proton)"),
    "q_720": (Q_DEFICIT_NUMERIC / 720, "h(E₈)×K(D₄)", "Coxeter-Kissing product"),
    "q_360": (Q_DEFICIT_NUMERIC / 360, "10×36", "Complete cone periodicity"),
    "q_248": (Q_DEFICIT_NUMERIC / 248, "dim(E₈)", "Full E₈ adjoint"),
    "q_240": (Q_DEFICIT_NUMERIC / 240, "|Φ(E₈)|", "Full E₈ root system"),
    "q_133": (Q_DEFICIT_NUMERIC / 133, "dim(E₇)", "Full E₇ adjoint"),
    "q_126": (Q_DEFICIT_NUMERIC / 126, "|Φ(E₇)|", "Full E₇ root system"),
    "q_120": (Q_DEFICIT_NUMERIC / 120, "|Φ⁺(E₈)|", "E₈ positive roots"),
    "q2_phi2": (
        Q_DEFICIT_NUMERIC**2 / PHI_NUMERIC**2,
        "Second-order/double golden",
        "Deep massless",
    ),
    # Level 11-20: High-precision (gauge and matter structures)
    "q_78": (Q_DEFICIT_NUMERIC / 78, "dim(E₆)", "Full E₆ gauge"),
    "q_72": (Q_DEFICIT_NUMERIC / 72, "|Φ(E₆)|", "Full E₆ roots"),
    "q_63": (Q_DEFICIT_NUMERIC / 63, "|Φ⁺(E₇)|", "E₇ positive roots"),
    "q2_phi": (
        Q_DEFICIT_NUMERIC**2 / PHI_NUMERIC,
        "Second-order massless",
        "Neutrino, CMB",
    ),
    "q_56": (Q_DEFICIT_NUMERIC / 56, "dim(E₇ fund)", "E₇ fundamental"),
    "q_52": (Q_DEFICIT_NUMERIC / 52, "dim(F₄)", "F₄ gauge"),
    "q_squared": (Q_DEFICIT_NUMERIC**2, "Second-order vacuum", "Two-loop"),
    "q_36": (Q_DEFICIT_NUMERIC / 36, "|Φ⁺(E₆)|", "36 Golden Cone roots"),
    "q_32": (Q_DEFICIT_NUMERIC / 32, "2⁵", "Five-fold binary"),
    "q_30": (Q_DEFICIT_NUMERIC / 30, "h(E₈)", "Coxeter number"),
    # Level 21-30: Medium-precision (representations and forms)
    "q_28": (Q_DEFICIT_NUMERIC / 28, "dim(D₄)", "SO(8) adjoint"),
    "q_27": (Q_DEFICIT_NUMERIC / 27, "dim(27_E₆)", "E₆ fundamental"),
    "q_24": (Q_DEFICIT_NUMERIC / 24, "K(D₄)", "Kissing number"),
    "q_18": (Q_DEFICIT_NUMERIC / 18, "h(E₇)", "E₇ Coxeter"),
    "q_16": (Q_DEFICIT_NUMERIC / 16, "2⁴", "Four-fold binary"),
    "q_14": (Q_DEFICIT_NUMERIC / 14, "dim(G₂)", "G₂ adjoint"),
    "q_12": (Q_DEFICIT_NUMERIC / 12, "h(E₆)", "E₆ Coxeter"),
    "q_10": (Q_DEFICIT_NUMERIC / 10, "φ²", "Golden square"),
    "q_9": (Q_DEFICIT_NUMERIC / 9, "3²", "Cubic symmetry"),
    "q_8": (Q_DEFICIT_NUMERIC / 8, "2³", "Cubic binary"),
    # Level 31-40: Low-precision (subgroup structures)
    "q_7": (Q_DEFICIT_NUMERIC / 7, "Mersenne prime", "Matter stability"),
    "q_6": (Q_DEFICIT_NUMERIC / 6, "h(D₄)", "D₄ Coxeter"),
    "q_5": (Q_DEFICIT_NUMERIC / 5, "Fermat prime", "Force selection"),
    "q_4": (Q_DEFICIT_NUMERIC / 4, "2²", "Tetrahedral"),
    "q_3": (Q_DEFICIT_NUMERIC / 3, "φ", "Golden ratio"),
    "q_2": (Q_DEFICIT_NUMERIC / 2, "√2", "Square root"),
    "q_phi_inv": (Q_DEFICIT_NUMERIC / PHI_NUMERIC, "1/φ", "Golden inverse"),
    "q_phi_sq": (Q_DEFICIT_NUMERIC * PHI_NUMERIC**2, "φ²·q", "Golden enhanced"),
    "q_phi_cubed": (Q_DEFICIT_NUMERIC * PHI_NUMERIC**3, "φ³·q", "Triple golden"),
    "q_sqrt_phi": (Q_DEFICIT_NUMERIC * PHI_NUMERIC**0.5, "√φ·q", "Golden root"),
    # Level 41-50: Specialized corrections (mixed orders)
    "q_phi_fourth": (Q_DEFICIT_NUMERIC * PHI_NUMERIC**4, "φ⁴·q", "Quadruple golden"),
    "q_phi_fifth": (Q_DEFICIT_NUMERIC * PHI_NUMERIC**5, "φ⁵·q", "Quintuple golden"),
    "q_4pi": (Q_DEFICIT_NUMERIC / (4 * PI_NUMERIC), "q/4π", "Electromagnetic"),
    "q_6pi": (Q_DEFICIT_NUMERIC / (6 * PI_NUMERIC), "q/6π", "Weak mixing"),
    "q_8pi": (Q_DEFICIT_NUMERIC / (8 * PI_NUMERIC), "q/8π", "Strong coupling"),
    "q_e_star": (Q_DEFICIT_NUMERIC / E_STAR_NUMERIC, "q/e*", "Vacuum energy"),
    "q_e": (Q_DEFICIT_NUMERIC / E_NUMERIC, "q/e", "Exponential coupling"),
    "q_pi": (Q_DEFICIT_NUMERIC / PI_NUMERIC, "q/π", "Circular geometry"),
    "q_pi_sq": (Q_DEFICIT_NUMERIC / PI_NUMERIC**2, "q/π²", "Spherical volume"),
    "q_2pi": (Q_DEFICIT_NUMERIC / (2 * PI_NUMERIC), "q/2π", "Angular momentum"),
    # Level 51-60+: Exotic corrections (theoretical structures)
    "q_mersenne_11": (Q_DEFICIT_NUMERIC / 2047, "M₁₁ barrier", "Generation limit"),
    "q_fermat_5": (Q_DEFICIT_NUMERIC / 4294967297, "F₅ composite", "Force cutoff"),
    "q_lucas_17": (Q_DEFICIT_NUMERIC / 1597, "L₁₇ prime", "Dark matter"),
    "q_kissing_24": (Q_DEFICIT_NUMERIC / 24, "K(D₄)", "Consciousness threshold"),
    "q_coxeter_720": (Q_DEFICIT_NUMERIC / 720, "h(E₈)×K(D₄)", "Unified coupling"),
    "q_hierarchy_719": (Q_DEFICIT_NUMERIC / 719, "719 exponent", "Grand unification"),
    "q_e8_e7_bridge": (Q_DEFICIT_NUMERIC / (248 - 133), "E₈-E₇ span", "Breaking scale"),
    "q_e7_e6_bridge": (
        Q_DEFICIT_NUMERIC / (133 - 78),
        "E₇-E₆ span",
        "Intermediate scale",
    ),
    "q_e6_sm_bridge": (Q_DEFICIT_NUMERIC / (78 - 28), "E₆-D₄ span", "Standard model"),
    "q_platonic_sum": (
        Q_DEFICIT_NUMERIC / (4 + 6 + 8 + 12 + 20),
        "Platonic total",
        "Geometric perfection",
    ),
}

# Legacy CORRECTION_HIERARCHY for backward compatibility
# Maps level indices to correction factor values
CORRECTION_HIERARCHY = {
    level: factor_value
    for level, (factor_value, _, _) in enumerate(CORRECTION_FACTORS.values())
}

# Multiplicative suppression factors
SUPPRESSION_FACTORS = {
    "inverse_double_recursion": 1 / (1 + Q_DEFICIT_NUMERIC / PHI_NUMERIC**2),
    "inverse_recursion": 1 / (1 + Q_DEFICIT_NUMERIC / PHI_NUMERIC),
    "base_suppression": 1 / (1 + Q_DEFICIT_NUMERIC),
    "recursion_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC),
    "fixed_point_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC**2),
    "deep_recursion_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC**3),
}

# =============================================================================
# Helper Functions
# =============================================================================


def fibonacci(n: int) -> int:
    """
    Compute Fibonacci number F_n using exact golden ratio arithmetic.

    Uses the identity: φ^n = F_{n-1} + F_n·φ

    Args:
        n: Non-negative integer

    Returns:
        F_n (Fibonacci number)

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
        >>> fibonacci(50)
        12586269025
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return 0
    if n == 1:
        return 1

    # φ^n = F_{n-1} + F_n·φ
    # So F_n is the coefficient of φ
    phi_n = PHI.phi_to_power(n)
    # phi_coefficient returns (numerator, denominator) tuple
    num, denom = phi_n.phi_coefficient
    return num // denom


def lucas(n: int) -> int:
    """
    Compute Lucas number L_n using exact golden ratio arithmetic.

    Uses the identity: L_n = φ^n + (φ')^n = 2·a_n + b_n
    where φ^n = a_n + b_n·φ

    Args:
        n: Non-negative integer

    Returns:
        L_n (Lucas number)

    Examples:
        >>> lucas(0)
        2
        >>> lucas(1)
        1
        >>> lucas(10)
        123
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return 2
    if n == 1:
        return 1

    # φ^n = a + b·φ
    # (φ')^n = a - b·(φ - 1) = (a + b) - b·φ  [since φ' = -1/φ = 1 - φ]
    # L_n = φ^n + (φ')^n = 2a + b
    phi_n = PHI.phi_to_power(n)
    # rational_coefficient and phi_coefficient return (num, denom) tuples
    a_num, a_denom = phi_n.rational_coefficient
    b_num, b_denom = phi_n.phi_coefficient
    a = a_num // a_denom
    b = b_num // b_denom
    return 2 * a + b


def correction_factor(structure: str, sign: int = 1) -> float:
    """
    Compute SRT correction factor (1 ± q/N).

    These factors arise from heat kernel regularization in SRT.

    Args:
        structure: One of 'E8_dim', 'E8_roots', 'E8_positive',
                  'E6_dim', 'E6_positive', 'E6_fundamental',
                  'D4_kissing', 'G2_dim'
        sign: +1 for enhancement (1 + q/N), -1 for suppression (1 - q/N)

    Returns:
        Correction factor value

    Examples:
        >>> correction_factor('E8_positive', -1)  # Chiral suppression
        0.9997716...
        >>> correction_factor('E6_positive', +1)  # Intra-generation
        1.0007609...
    """
    if structure not in _STRUCTURE_INDICES:
        raise ValueError(
            f"Unknown structure: {structure}. "
            f"Valid options: {list(_STRUCTURE_INDICES.keys())}"
        )
    index = _STRUCTURE_INDICES[structure]
    return srt_correction_factor(index, sign)


def get_correction_factor(level: int) -> float:
    """
    Get the correction factor for a given hierarchy level.

    Args:
        level: Hierarchy level (0-57)

    Returns:
        Correction factor value (1 ± correction)

    Examples:
        >>> get_correction_factor(35)  # q/8
        0.996577...
        >>> get_correction_factor(47)  # q
        0.027395...
    """
    return CORRECTION_HIERARCHY.get(level, 1.0)


def get_suppression_factor(name: str) -> float:
    """
    Get the suppression factor for a given name.

    Args:
        name: One of 'inverse_double_recursion', 'inverse_recursion', 'base_suppression',
              'recursion_penalty', 'fixed_point_penalty', 'deep_recursion_penalty'

    Returns:
        Suppression factor value (< 1)

    Examples:
        >>> get_suppression_factor('inverse_recursion')  # 1/(1+q/φ)
        0.9830...
        >>> get_suppression_factor('recursion_penalty')  # 1/(1+qφ)
        0.9578...
    """
    return SUPPRESSION_FACTORS.get(name, 1.0)


def apply_nested_corrections(value: float, factors: list) -> float:
    """
    Apply a sequence of correction factors from the geometric hierarchy.

    Args:
        value: Base value to correct
        factors: List of factor names from CORRECTION_FACTORS.keys()

    Returns:
        Value with corrections applied: value × ∏(1 + factor_value)

    Examples:
        >>> apply_nested_corrections(1.0, ['q_248', 'q_133', 'q_78'])
        # Applies E₈ → E₇ → E₆ corrections
    """
    result = value
    for factor_name in factors:
        if factor_name in CORRECTION_FACTORS:
            factor_value, _, _ = CORRECTION_FACTORS[factor_name]
            result *= 1.0 + factor_value
    return result


def get_geometric_factor(name: str) -> tuple[float, str, str]:
    """
    Get a specific geometric correction factor by name.

    Args:
        name: Factor name from CORRECTION_FACTORS.keys()

    Returns:
        Tuple of (factor_value, geometric_origin, physical_interpretation)

    Examples:
        >>> factor, origin, physics = get_geometric_factor('q_248')
        >>> print(f"E₈ correction: {factor:.2e} ({physics})")
        E₈ correction: 1.10e-04 (Full E₈ adjoint)
    """
    if name not in CORRECTION_FACTORS:
        raise ValueError(f"Unknown geometric factor: {name}")
    return CORRECTION_FACTORS[name]


def apply_unified_hierarchy(value: float, target_scale: str = "sm") -> float:
    """
    Apply the complete SRT hierarchy corrections for a target energy scale.

    Args:
        value: Physical value to correct
        target_scale: Target unification scale
            - 'e8': Full E₈ unification
            - 'e7': Intermediate E₇ scale
            - 'e6': E₆ grand unification
            - 'sm': Standard Model (D₄/SO(8))

    Returns:
        Value corrected for the full hierarchy chain

    Examples:
        >>> proton_mass = apply_unified_hierarchy(938.272, 'sm')
        # Applies E₈ → E₇ → E₆ → SM corrections
    """
    hierarchy_chains = {
        "e8": [],  # No corrections (exact at Planck scale)
        "e7": ["q_248", "q_133"],  # E₈ → E₇ breaking
        "e6": ["q_248", "q_133", "q_78"],  # E₈ → E₇ → E₆ breaking
        "sm": ["q_248", "q_133", "q_78", "q_28"],  # Full chain to SM
    }

    if target_scale not in hierarchy_chains:
        raise ValueError(f"Unknown target scale: {target_scale}")

    return apply_nested_corrections(value, hierarchy_chains[target_scale])


def golden_number(a: int, b: int) -> GoldenExact:
    """
    Create a GoldenExact representing a + b·φ

    Args:
        a: Integer coefficient
        b: Golden ratio coefficient

    Returns:
        GoldenExact representing a + b·φ

    Examples:
        >>> g = golden_number(1, 2)  # 1 + 2φ
        >>> g.eval()
        4.23606797749979
    """
    return GoldenExact.from_integers(a, b)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    "Rational",
    "GoldenExact",
    "FundamentalConstant",
    "SymExpr",
    # Exact constants
    "PHI",
    "PHI_SQUARED",
    "PHI_INVERSE",
    "Q_DEFICIT",
    "E_STAR",
    # Numeric constants
    "PHI_NUMERIC",
    "E_STAR_NUMERIC",
    "Q_DEFICIT_NUMERIC",
    "PI_NUMERIC",
    "E_NUMERIC",
    # SRT computation functions (from Rust backend)
    "srt_phi",
    "srt_phi_inv",
    "srt_pi",
    "srt_e",
    "srt_q_deficit",
    "srt_correction_factor",
    "srt_structure_dimension",
    # Prime sequence constants
    "FERMAT_PRIMES",
    "FERMAT_INDICES",
    "MERSENNE_PRIMES",
    "MERSENNE_EXPONENTS",
    "M11_BARRIER",
    "LUCAS_SEQUENCE",
    "LUCAS_PRIMES",
    "D4_KISSING",
    # Structure data
    "STRUCTURE_DIMENSIONS",
    "CORRECTION_HIERARCHY",
    "SUPPRESSION_FACTORS",
    # CORRECTION_FACTORS registry
    "CORRECTION_FACTORS",
    # Functions
    "fibonacci",
    "lucas",
    "correction_factor",
    "get_correction_factor",
    "get_suppression_factor",
    "get_geometric_factor",
    "golden_number",
    "apply_nested_corrections",
    "apply_unified_hierarchy",
    # Deprecated helpers (for migration)
    "get_phi_float",
    "get_phi_inv_float",
]
