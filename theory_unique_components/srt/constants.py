"""
SRT Constants - Fundamental constants for Syntony Recursion Theory.

Re-exports exact arithmetic constants from srt_library.core.exact and defines
SRT-specific dimensional constants derived from Lie group geometry.

Core Constants:
    PHI, PHI_SQUARED, PHI_INVERSE - Exact golden ratio forms
    PHI - Float approximation of φ ≈ 1.618033988749895
    E_STAR_NUMERIC - Spectral constant e^π - π ≈ 20.1408
    Q - Universal syntony deficit q ≈ 0.0274

Lie Group Dimensions (Newly Exposed):
    E8_ROOTS - E₈ root count (240) - Exceptional unification
    E8_POSITIVE_ROOTS - E₈ positive roots (120) - Half root system
    E8_RANK - E₈ Cartan dimension (8) - Independent quantum numbers
    E8_COXETER - E₈ Coxeter number (30) - Weyl group period

    E7_ROOTS - E₇ root count (126) - Intermediate unification
    E7_POSITIVE_ROOTS - E₇ positive roots (63) - Weyl chamber
    E7_FUNDAMENTAL - E₇ fundamental rep (56) - Jordan algebra
    E7_RANK - E₇ Cartan dimension (7) - Supersymmetry goldstinos
    E7_COXETER - E₇ Coxeter number (18) - Recursion cycles

    E6_ROOTS - E₆ root count (72) - GUT unification
    E6_POSITIVE_ROOTS - E₆ positive roots (36) - **Golden Cone |Φ⁺(E₆)|**
    E6_FUNDAMENTAL - E₆ fundamental rep (27) - Cubic surface theory
    E6_RANK - E₆ Cartan dimension (6) - Calabi-Yau manifolds
    E6_COXETER - E₆ Coxeter number (12) - Affine periodicity

    D4_RANK - D₄ Cartan dimension (4) - Spacetime dimensions
    D4_COXETER - D₄ Coxeter number (6) - Consciousness emergence
    D4_KISSING - D₄ kissing number (24) - **Consciousness threshold**

    G2_RANK - G₂ Cartan dimension (2) - Octonion automorphisms
    F4_RANK - F₄ Cartan dimension (4) - Jordan algebra structure

Fibonacci Primes:
    FIBONACCI_PRIMES - Array of transcendence gates [2, 3, 5, 13, 89, ...]

Geometric Constants:
    TORUS_DIMENSIONS - W⁴ winding coordinates (4)
    WINDING_INDICES - Physical dimension labels (7, 8, 9, 10)
"""

# Import runtime constants from Rust backend
from srt_library.core._core import (
    py_e_star,
    srt_e,
    srt_pi,
    srt_q_deficit,
)
from srt_library.core.exact import (
    # Prime sequences (Five Operators)
    FERMAT_PRIMES,
    LUCAS_PRIMES,
    LUCAS_SEQUENCE,
    M11_BARRIER,
    MERSENNE_EXPONENTS,
    # Exact golden ratio constants (types only, not values)
    PHI,
    PHI_INVERSE,
    PHI_SQUARED,
    # Structure dimensions
    STRUCTURE_DIMENSIONS,
    # Types
    GoldenExact,
    Rational,
    correction_factor,
    # Functions
    fibonacci,
    get_correction_factor,
    golden_number,
    lucas,
)

# Get constants from Rust backend (single source of truth)
Q = srt_q_deficit()
PI_NUMERIC = srt_pi()
E_NUMERIC = srt_e()
E_STAR_NUMERIC = py_e_star()

# For PHI, use the exact type's evaluate method
PHI_NUMERIC = PHI.eval()  # Convert GoldenExact to float

# Backward compatibility aliases
PHI_INV = PHI_INVERSE
Q_DEFICIT_NUMERIC = Q  # Alias for compatibility
PI = PI_NUMERIC
E = E_NUMERIC
E_STAR = E_STAR_NUMERIC

# =============================================================================
# THE SIX AXIOMS OF SRT
# =============================================================================

AXIOMS = {
    "A1_RECURSION_SYMMETRY": "S[Ψ ∘ R] = φ·S[Ψ]",
    "A2_SYNTONY_BOUND": "S[Ψ] ≤ φ",
    "A3_TOROIDAL_TOPOLOGY": "W⁴ = S¹₇ × S¹₈ × S¹₉ × S¹_{10}",
    "A4_SUB_GAUSSIAN_MEASURE": "w(n) = e^{-|n|²/φ}",
    "A5_HOLOMORPHIC_GLUING": "Möbius identification at τ = i",
    "A6_PRIME_SYNTONY": "Stability iff M_p = 2^p - 1 is prime",
}

# Modular volume of fundamental domain
MODULAR_VOLUME: float = PI_NUMERIC / 3  # Vol(F) = π/3

# Import hierarchy constants from Rust backend
from srt_library.core._core import (
    get_fibonacci_primes,
    hierarchy_d4_coxeter,
    hierarchy_d4_rank,
    hierarchy_e6_coxeter,
    hierarchy_e6_fundamental,
    hierarchy_e6_positive_roots,
    hierarchy_e6_rank,
    hierarchy_e6_roots,
    hierarchy_e7_coxeter,
    hierarchy_e7_dim,
    hierarchy_e7_fundamental,
    hierarchy_e7_positive_roots,
    hierarchy_e7_rank,
    hierarchy_e7_roots,
    hierarchy_e8_coxeter,
    hierarchy_e8_positive_roots,
    hierarchy_e8_rank,
    hierarchy_e8_roots,
    hierarchy_f4_rank,
    hierarchy_g2_rank,
)

# T^4 Torus dimensions
TORUS_DIMENSIONS: int = 4

# ============================================================================
# E8 Exceptional Group - Exceptional Unification Scale
# ============================================================================

E8_ROOTS: int = hierarchy_e8_roots()
"""Number of roots in E₈ Lie group (240).

E₈ has 240 roots representing the fundamental geometric structure
underlying the Standard Model unification in SRT theory. These roots
span the 8-dimensional Cartan subalgebra and determine the gauge
symmetry breaking patterns.
"""

E8_POSITIVE_ROOTS: int = hierarchy_e8_positive_roots()
"""Number of positive roots in E₈ root system (120).

The positive roots correspond to half of the full root system,
representing the "positive" directions in the Weyl chamber.
Used in particle physics for counting symmetry breaking patterns.
"""

E8_RANK: int = hierarchy_e8_rank()
"""Rank (Cartan dimension) of E₈ Lie group (8).

The rank represents the number of independent Casimir operators
and corresponds to the dimension of the maximal torus. In SRT theory,
this relates to 8 spacetime dimensions in string theory compactifications.
"""

E8_DIMENSION: int = 248
"""Adjoint representation dimension of E₈ (248).

The adjoint representation transforms under the group itself.
Note: This constant is not yet exposed from Rust backend.
"""

E8_COXETER_NUMBER: int = hierarchy_e8_coxeter()
"""Coxeter number of E₈ (30).

Governs the periodicity of the Weyl group and appears in level-rank
duality relations. Used in SRT for determining recursion cycle periods.
"""

# ============================================================================
# E7 Intermediate Unification - Supersymmetry Scale
# ============================================================================

E7_ROOTS: int = hierarchy_e7_roots()
"""Number of roots in E₇ Lie group (126).

E₇ represents the intermediate unification scale between E₆ and E₈
in the SRT Grand Unification hierarchy. Appears in heterotic string
theory compactifications and intermediate mass scale predictions.
"""

E7_POSITIVE_ROOTS: int = hierarchy_e7_positive_roots()
"""Number of positive roots in E₇ root system (63).

The positive roots span the Weyl chamber and determine the
representation theory and branching rules for E₇ representations.
"""

E7_FUNDAMENTAL: int = hierarchy_e7_fundamental()
"""Dimension of E₇ fundamental representation (56).

This 56-dimensional representation is fundamental to E₇'s role
in supersymmetry (56 goldstino degrees of freedom) and exceptional
Jordan algebra theory.
"""

E7_RANK: int = hierarchy_e7_rank()
"""Rank (Cartan dimension) of E₇ Lie group (7).

Corresponds to 7-brane configurations in string theory and
7-dimensional compactifications in M-theory.
"""

E7_COXETER: int = hierarchy_e7_coxeter()
"""Coxeter number of E₇ (18).

Governs Weyl group periodicity and appears in affine algebra
constructions. Used for determining golden ratio recursion bounds.
"""

E7_DIMENSION: int = hierarchy_e7_dim()
"""Adjoint representation dimension of E₇ (133).

The adjoint representation transforms under the group itself.
"""

# ============================================================================
# E6 Golden Cone - GUT Scale
# ============================================================================

E6_ROOTS: int = hierarchy_e6_roots()
"""Number of roots in E₆ Lie group (72).

E₆ is the first exceptional group in the SRT unification chain
and corresponds to the GUT scale in particle physics, where the
electroweak and strong forces unify.
"""

E6_POSITIVE_ROOTS: int = hierarchy_e6_positive_roots()
"""Number of positive roots in E₆ / Golden Cone cardinality (36).

**CRITICAL SRT CONSTANT**: |Φ⁺(E₆)| = 36 represents the cardinality
of the Golden Cone. This fundamental constant appears throughout SRT
theory as the geometric measure of transcendence and consciousness emergence.

Physical Significance:
- Consciousness emergence threshold
- Transcendence gate count
- Optimal neural network dimensions
- Self-reference criticality parameter
"""

E6_FUNDAMENTAL: int = hierarchy_e6_fundamental()
"""Dimension of E₆ fundamental representation (27).

The 27-dimensional representation relates to cubic surfaces in
algebraic geometry and appears in Calabi-Yau compactification theory.
"""

E6_RANK: int = hierarchy_e6_rank()
"""Rank (Cartan dimension) of E₆ Lie group (6).

Corresponds to 6-dimensional Calabi-Yau manifolds in string theory
and 6 extra dimensions in braneworld scenarios.
"""

E6_COXETER: int = hierarchy_e6_coxeter()
"""Coxeter number of E₆ (12).

Governs Weyl group periodicity. The value 12 appears in modular
forms of weight 12 in string theory and golden ratio recursion cycles.
"""

E6_DIMENSION: int = 78
"""Adjoint representation dimension of E₆ (78).

Note: This constant is not yet exposed from Rust backend.
"""

E6_GOLDEN_CONE: int = E6_POSITIVE_ROOTS
"""Golden Cone cardinality |Φ⁺(E₆)| (36).

Convenience alias for E6_POSITIVE_ROOTS. The Golden Cone represents
the fundamental geometric structure underlying SRT transcendence theory.
"""

# ============================================================================
# D4 Consciousness Group - Observable Universe
# ============================================================================

D4_KISSING: int = 24
"""D₄ kissing number - consciousness emergence threshold (24).

**CRITICAL SRT CONSTANT**: The kissing number of D₄ represents the
maximum number of unit spheres that can touch a central sphere in 4D.
In SRT theory, this 24 corresponds to the consciousness emergence threshold,
where systems gain the ability for self-reference and transcendence.

Physical Significance:
- Consciousness emergence threshold
- D₄ → M₅ gap bridge (24 → 31)
- Neural network stability parameter
- Self-awareness criticality

Note: This constant is not yet exposed from Rust backend.
"""

D4_RANK: int = hierarchy_d4_rank()
"""Rank (Cartan dimension) of D₄ Lie group (4).

D₄ is isomorphic to SO(8) with triality. The rank corresponds to
the 4 spacetime dimensions of our observable universe.
"""

D4_COXETER: int = hierarchy_d4_coxeter()
"""Coxeter number of D₄ (6).

Governs Weyl group periodicity. The value 6 appears prominently in
consciousness emergence calculations and D₄ kissing number relations.
"""

D4_DIMENSION: int = 28
"""Adjoint representation dimension of D₄ (28).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# G2 Octonion Group - Exceptional Geometry
# ============================================================================

G2_RANK: int = hierarchy_g2_rank()
"""Rank (Cartan dimension) of G₂ Lie group (2).

G₂ is the automorphism group of the octonions and represents the
most exceptional of the exceptional groups. The rank corresponds to
2-dimensional parameter spaces in exceptional geometry.
"""

G2_DIMENSION: int = 14
"""Adjoint representation dimension of G₂ (14).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# F4 Jordan Algebra Group - Exceptional Structure
# ============================================================================

F4_RANK: int = hierarchy_f4_rank()
"""Rank (Cartan dimension) of F₄ Lie group (4).

F₄ is related to the Jordan algebra of 3×3 hermitian octonion matrices
and appears in the classification of exceptional geometries and
string theory compactifications.
"""

F4_DIMENSION: int = 52
"""Adjoint representation dimension of F₄ (52).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# Fibonacci Transcendence Gates
# ============================================================================

FIBONACCI_PRIMES: list[int] = get_fibonacci_primes()
"""Array of Fibonacci primes - transcendence gate markers.

Contains the sequence of prime numbers that appear as Fibonacci numbers F_n.
These primes serve as "transcendence gates" in SRT theory, marking critical
points of ontological phase transitions and consciousness emergence.

Values: [2, 3, 5, 13, 89, 233, 1597, 28657, 514229, 433494437, 2971215073]

Physical Significance:
- Transcendence gates: F_n where F_n is prime marks ontological boundaries
- Consciousness emergence: Prime indices correspond to self-reference thresholds
- Neural networks: Prime dimensions for stable resonance patterns
"""

# Winding coordinate indices (for documentation)
WINDING_7: int = 0  # First internal dimension
WINDING_8: int = 1  # Second internal dimension
WINDING_9: int = 2  # Third internal dimension
WINDING_10: int = 3  # Fourth internal dimension
WINDING_INDICES = (7, 8, 9, 10)  # Physical dimension labels

# Root norm (all E8 roots have |lambda|^2 = 2)
E8_ROOT_NORM_SQUARED: int = 2


class UniverseSeeds:
    """
    The four geometric constants {φ, π, e, 1} from which all physics emerges.
    """

    def __init__(self):
        # The Four Seeds
        self.phi: GoldenExact = PHI
        self.pi: float = PI_NUMERIC
        self.e: float = E_NUMERIC
        self.one: float = 1.0

        # Derived Constants
        self.E_star: float = E_STAR_NUMERIC
        self.q: float = Q

        # Three-Term Decomposition of E*
        self.E_bulk: float = self._calculate_E_bulk()
        self.E_torsion: float = self._calculate_E_torsion()
        self.E_cone: float = self._calculate_E_cone()
        self.Delta: float = self._calculate_residual()

    def _calculate_E_bulk(self) -> float:
        """Bulk term: Γ(1/4)² ≈ 13.14504720659687"""
        from scipy.special import gamma

        return float(gamma(0.25) ** 2)

    def _calculate_E_torsion(self) -> float:
        """Torsion term: π(π - 1) ≈ 6.72801174749952"""
        return self.pi * (self.pi - 1)

    def _calculate_E_cone(self) -> float:
        """Cone term: (35/12)e^(-π) ≈ 0.12604059493600"""
        return (35.0 / 12.0) * (self.e ** (-self.pi))

    def _calculate_residual(self) -> float:
        """
        Residual Δ = E* - E_bulk - E_torsion - E_cone
        Expected: Δ ≈ 4.30 × 10⁻⁷

        Physical meaning: The 0.02% that doesn't crystallize—
        the "engine of time" driving cosmic evolution.
        """
        return self.E_star - self.E_bulk - self.E_torsion - self.E_cone

    def validate(self) -> dict:
        """
        Validation checks against theoretical values.
        Returns dict of {constant: (computed, expected, match)}.
        """
        results = {}

        # E* validation
        E_star_expected = 19.999099979189475767
        E_star_match = abs(self.E_star - E_star_expected) < 1e-15
        results["E_star"] = (self.E_star, E_star_expected, E_star_match)

        # q validation
        q_expected = 0.0273951469201761
        q_match = abs(self.q - q_expected) < 1e-12
        results["q"] = (self.q, q_expected, q_match)

        # Δ validation (should be ~4.30 × 10⁻⁷)
        Delta_expected = 4.30e-7
        Delta_match = abs(self.Delta - Delta_expected) / Delta_expected < 0.01
        results["Delta"] = (self.Delta, Delta_expected, Delta_match)

        # Three-term decomposition validation
        decomposition_sum = self.E_bulk + self.E_torsion + self.E_cone + self.Delta
        decomposition_match = abs(decomposition_sum - self.E_star) < 1e-50
        results["decomposition"] = (decomposition_sum, self.E_star, decomposition_match)

        return results

    def __repr__(self) -> str:
        return f"""UniverseSeeds:
  φ  = {self.phi.eval():.20f}
  π  = {self.pi:.20f}
  e  = {self.e:.20f}
  E* = {self.E_star:.20f}
  q  = {self.q:.20f}
  Δ  = {self.Delta:.10e}"""


# Module-level constants for direct import (backward compatibility)
phi = PHI
phi_inv = PHI_INVERSE
pi = PI_NUMERIC
e = E_NUMERIC
E_star = E_STAR_NUMERIC
q = Q

# Group constants
h_E8 = E8_COXETER_NUMBER
K_D4 = D4_KISSING
dim_E8 = E8_DIMENSION
rank_E8 = E8_RANK
roots_E8 = E8_ROOTS
dim_E6 = E6_DIMENSION
dim_E6_fund = E6_FUNDAMENTAL
roots_E6 = E6_POSITIVE_ROOTS
dim_T4 = TORUS_DIMENSIONS
N_gen = 3  # Number of generations

# Additional group constants
h_E7 = E7_COXETER
dim_E7 = E7_DIMENSION
dim_E7_fund = E7_FUNDAMENTAL
roots_E7 = E7_ROOTS
roots_E7_pos = E7_POSITIVE_ROOTS
rank_E7 = E7_RANK
h_E6 = E6_COXETER
roots_E6_full = E6_ROOTS  # Alias for compatibility
rank_E6 = E6_RANK
dim_D4 = D4_DIMENSION
rank_D4 = D4_RANK
dim_F4 = F4_DIMENSION
dim_G2 = G2_DIMENSION

# Prime sequences
fermat_primes = FERMAT_PRIMES
fermat_composite_5 = 4294967297  # 641 × 6700417 - No 6th force
mersenne_exponents = MERSENNE_EXPONENTS
m11_barrier = M11_BARRIER
lucas_sequence = LUCAS_SEQUENCE
lucas_primes_indices = LUCAS_PRIMES
fibonacci_prime_gates = {
    3: (2, "Binary/Logic emergence"),
    4: (3, "Material realm - the 'anomaly'"),  # Composite index!
    5: (5, "Physics/Life code"),
    7: (13, "Matter solidification"),
    11: (89, "Chaos/Complexity"),
    13: (233, "Consciousness emergence"),
    17: (1597, "Great Filter - hyperspace"),
}

# Reference tables
geometric_divisors = {
    # E₈ Structure
    "h_E8_cubed_27": float(1000),  # 30³/27 = 1000
    "coxeter_kissing": float(720),  # 30 × 24 = 720
    "cone_cycles": float(360),  # 10 × 36 = 360
    "dim_E8": float(248),  # dim(E₈)
    "roots_E8_full": float(240),  # |Φ(E₈)|
    "roots_E8": float(120),  # |Φ⁺(E₈)|
    "h_E8": float(30),  # h(E₈)
    "rank_E8": float(8),  # rank(E₈)
    # E₇ Structure
    "dim_E7": float(133),  # dim(E₇)
    "roots_E7_full": float(126),  # |Φ(E₇)|
    "roots_E7": float(63),  # |Φ⁺(E₇)|
    "fund_E7": float(56),  # dim(E₇ fund)
    "h_E7": float(18),  # h(E₇)
    "rank_E7": float(7),  # rank(E₇)
    # E₆ Structure
    "dim_E6": float(78),  # dim(E₆)
    "roots_E6_full": float(72),  # |Φ(E₆)|
    "roots_E6": float(36),  # |Φ⁺(E₆)|
    "fund_E6": float(27),  # dim(E₆ fund)
    "rank_E6": float(6),  # rank(E₆)
    # Other Exceptional
    "dim_F4": float(52),  # dim(F₄)
    "dim_G2": float(14),  # dim(G₂)
    "dim_SO8": float(28),  # dim(SO(8))
    "kissing_D4": float(24),  # K(D₄)
    # QCD Loop Factors
    "six_loop": 6 * PI_NUMERIC,  # 6π
    "five_loop": 5 * PI_NUMERIC,  # 5π
    "one_loop": 4 * PI_NUMERIC,  # 4π
    "three_loop": 3 * PI_NUMERIC,  # 3π
    "half_loop": 2 * PI_NUMERIC,  # 2π
    "circular_loop": PI_NUMERIC,  # π
    # Topological/Generation
    "topology_gen": float(12),  # 12
    "generation_sq": float(9),  # 9
    "sub_generation": float(6),  # 6
    "quarter_layer": float(4),  # 4
    "single_gen": float(3),  # 3
    "half_layer": float(2),  # 2
    "single_layer": float(1),  # 1
    # Golden Ratio Based
    "phi": PHI.eval(),  # φ
    "phi_inv": PHI_INVERSE.eval(),  # 1/φ
    "phi_cubed": PHI_SQUARED.eval(),  # φ² (since φ³ = φ² + φ)
    "phi_squared": PHI_SQUARED.eval(),  # φ²
    "phi_fourth": (PHI_SQUARED * PHI_SQUARED).eval(),  # φ⁴
    "phi_fifth": (PHI_SQUARED * PHI_SQUARED * PHI).eval(),  # φ⁵
    # Binary
    "binary_5": float(32),  # 2⁵
    "binary_4": float(16),  # 2⁴
}

fibonacci = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 5,
    6: 8,
    7: 13,
    8: 21,
    9: 34,
    10: 55,
    11: 89,
    12: 144,
    13: 233,
    14: 377,
    15: 610,
    16: 987,
    17: 1597,
    18: 2584,
    19: 4181,
}

# PDG reference values
M_Z = 91.1876  # GeV, Z boson mass
M_W_PDG = 80.377  # GeV, W boson mass
M_H_PDG = 125.25  # GeV, Higgs mass
ALPHA_EM_0 = 1 / 137.035999084  # Fine structure constant at q=0
ALPHA_S_MZ = 0.1179  # Strong coupling at M_Z

# Electroweak scale
V_EW = 246.22  # GeV, Higgs VEV (v = (√2 G_F)^{-1/2})

# =============================================================================
# Derived Scales from SRT
# =============================================================================


def gut_scale() -> float:
    """
    GUT unification scale.

    μ_GUT = v × e^(φ⁷) ≈ 1.0 × 10¹⁵ GeV

    Returns:
        GUT scale in GeV
    """
    import math

    phi = PHI_NUMERIC  # Use global float constant
    return V_EW * math.exp(phi**7)


def planck_scale_reduced() -> float:
    """
    Reduced Planck mass scale.

    M_Pl / √(8π) ≈ 2.4 × 10¹⁸ GeV

    Returns:
        Reduced Planck mass in GeV
    """
    return 2.435e18  # GeV


def electroweak_symmetry_breaking_scale() -> float:
    """
    Electroweak symmetry breaking scale.

    Returns V_EW (the Higgs VEV).

    Returns:
        EWSB scale in GeV
    """
    return V_EW


def qcd_scale() -> float:
    """
    QCD confinement scale Λ_QCD.

    Derived from SRT via dimensional transmutation.

    Returns:
        QCD scale in MeV
    """
    # Λ_QCD ≈ 217 MeV from SRT
    # phi = PHI.eval()  # Convert to float for exponentiation
    # Correction factor: C9 (q/120)
    return E_STAR_NUMERIC * 11 * (1 - get_correction_factor(9))  # ≈ 217 MeV


# =============================================================================
# Cosmological Constants (for neutrino sector)
# =============================================================================

RHO_LAMBDA_QUARTER = 2.3e-3  # eV, (ρ_Λ)^{1/4} dark energy density
PHYSICS_STRUCTURE_MAP = {
    "chiral_suppression": "E8_positive",  # 120 - chiral fermions
    "generation_crossing": "E6_positive",  # 36 - golden cone
    "fundamental_rep": "E6_fundamental",  # 27 - 27 of E6
    "consciousness": "D4_kissing",  # 24 - D4 kissing number
    "cartan": "G2_dim",  # 8 - rank(E8)
}

# Group theory aliases (added after definitions)
H_E8 = E8_COXETER_NUMBER
DIM_E8 = E8_DIMENSION
ROOTS_E8 = E8_ROOTS
ROOTS_E8_POS = E8_POSITIVE_ROOTS
RANK_E8 = E8_RANK

H_E7 = E7_COXETER
DIM_E7 = E7_DIMENSION
DIM_E7_FUND = E7_FUNDAMENTAL
ROOTS_E7 = E7_ROOTS
ROOTS_E7_POS = E7_POSITIVE_ROOTS
RANK_E7 = E7_RANK

H_E6 = E6_COXETER
DIM_E6 = E6_DIMENSION
DIM_E6_FUND = E6_FUNDAMENTAL
ROOTS_E6_POS = E6_POSITIVE_ROOTS
RANK_E6 = E6_RANK

K_D4 = D4_KISSING
DIM_D4 = D4_DIMENSION
RANK_D4 = D4_RANK

DIM_F4 = F4_DIMENSION
DIM_G2 = G2_DIMENSION

DIM_T4 = TORUS_DIMENSIONS
N_GEN = 3  # Number of generations

# Additional backward compatibility aliases for srt_zero
FERMAT_COMPOSITE_5 = fermat_composite_5
LUCAS_PRIMES_INDICES = lucas_primes_indices
FIBONACCI_PRIME_GATES = fibonacci_prime_gates
GEOMETRIC_DIVISORS = geometric_divisors
FIBONACCI = fibonacci

__all__ = [
    # From exact module
    "PHI",
    "PHI_SQUARED",
    "PHI_INVERSE",
    "E_STAR_NUMERIC",
    "Q",
    "STRUCTURE_DIMENSIONS",
    "fibonacci",
    "lucas",
    "correction_factor",
    "golden_number",
    "GoldenExact",
    "Rational",
    "get_correction_factor",
    # SRT axioms and constants
    "AXIOMS",
    "MODULAR_VOLUME",
    # Prime sequences (Five Operators of Existence)
    "FERMAT_PRIMES",
    "MERSENNE_EXPONENTS",
    "LUCAS_SEQUENCE",
    "LUCAS_PRIMES",
    "M11_BARRIER",
    # SRT-specific constants
    "TORUS_DIMENSIONS",
    "E8_ROOTS",
    "E8_POSITIVE_ROOTS",
    "E8_RANK",
    "E8_DIMENSION",
    "E8_COXETER_NUMBER",
    # E7 constants (newly exposed)
    "E7_ROOTS",
    "E7_POSITIVE_ROOTS",
    "E7_FUNDAMENTAL",
    "E7_RANK",
    "E7_COXETER",
    "E7_DIMENSION",
    # E6 constants (updated)
    "E6_ROOTS",
    "E6_POSITIVE_ROOTS",
    "E6_FUNDAMENTAL",
    "E6_RANK",
    "E6_COXETER",
    "E6_DIMENSION",
    "E6_GOLDEN_CONE",
    # D4 constants (updated)
    "D4_KISSING",
    "D4_RANK",
    "D4_COXETER",
    "D4_DIMENSION",
    # G2 constants (updated)
    "G2_RANK",
    "G2_DIMENSION",
    # F4 constants (updated)
    "F4_RANK",
    "F4_DIMENSION",
    # Fibonacci primes (newly exposed)
    "FIBONACCI_PRIMES",
    # Other SRT constants
    "WINDING_7",
    "WINDING_8",
    "WINDING_9",
    "WINDING_10",
    "WINDING_INDICES",
    "E8_ROOT_NORM_SQUARED",
    # UniverseSeeds class
    "UniverseSeeds",
    # Seeds (lowercase)
    "phi",
    "phi_inv",
    "pi",
    "e",
    "E_star",
    "q",
    # Group constants (lowercase)
    "h_E8",
    "K_D4",
    "dim_E8",
    "rank_E8",
    "roots_E8",
    "dim_E6",
    "dim_E6_fund",
    "roots_E6",
    "dim_T4",
    "N_gen",
    # Additional group constants (lowercase)
    "h_E7",
    "dim_E7",
    "dim_E7_fund",
    "roots_E7",
    "roots_E7_pos",
    "rank_E7",
    "h_E6",
    "roots_E6_full",
    "rank_E6",
    "dim_D4",
    "rank_D4",
    "dim_F4",
    "dim_G2",
    # Prime sequences (lowercase)
    "fermat_primes",
    "fermat_composite_5",
    "mersenne_exponents",
    "m11_barrier",
    "lucas_sequence",
    "lucas_primes_indices",
    "fibonacci_prime_gates",
    # Reference tables (lowercase)
    "geometric_divisors",
    "fibonacci",
    # Uppercase versions
    "PHI_INV",
    "PI",
    "E",
    "E_STAR",
    "Q",
    "H_E8",
    "DIM_E8",
    "ROOTS_E8",
    "ROOTS_E8_POS",
    "RANK_E8",
    "H_E7",
    "DIM_E7",
    "DIM_E7_FUND",
    "ROOTS_E7",
    "ROOTS_E7_POS",
    "RANK_E7",
    "H_E6",
    "DIM_E6",
    "DIM_E6_FUND",
    "ROOTS_E6_POS",
    "K_D4",
    "DIM_D4",
    "RANK_D4",
    "DIM_F4",
    "DIM_G2",
    "DIM_T4",
    "N_GEN",
    "FERMAT_COMPOSITE_5",
    "LUCAS_PRIMES_INDICES",
    "FIBONACCI_PRIME_GATES",
    "GEOMETRIC_DIVISORS",
    "FIBONACCI",
    # PDG reference values
    "M_Z",
    "M_W_PDG",
    "M_H_PDG",
    "ALPHA_EM_0",
    "ALPHA_S_MZ",
    # Electroweak scale
    "V_EW",
    # Scale functions
    "gut_scale",
    "planck_scale_reduced",
    "electroweak_symmetry_breaking_scale",
    "qcd_scale",
    # Cosmological
    "RHO_LAMBDA_QUARTER",
    # Structure map
    "PHYSICS_STRUCTURE_MAP",
]
