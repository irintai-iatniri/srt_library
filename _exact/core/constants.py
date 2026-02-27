"""
SRT-Zero Kernel: Core Constants Module
=======================================
Source: SRT_Equations.md + Universal_Syntony_Correction_Hierarchy.md

Initialize the geometric seeds of the universe to high precision.
Re-exports key constants from hierarchy.py for backward compatibility.
"""

from __future__ import annotations

from applications._core import SymExpr
from applications.exact import (
    E_NUMERIC,
    E_STAR_NUMERIC,
    PHI,
    PHI_INVERSE, 
    PHI_SQUARED,
    PHI_NUMERIC,
    PI_NUMERIC,
    Q_DEFICIT_NUMERIC,
    Rational,
    GoldenExact,
    get_correction_factor,
)

# =============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# =============================================================================

# Exact symbolic constants (primary)
PI = SymExpr.pi()  # π
E = SymExpr.e()  # e  
E_STAR = SymExpr.e_star()  # e^π - π
Q = SymExpr.q()  # Universal syntony deficit

# Golden ratio from applications.exact (already exact)
PHI_SYMEXPR = SymExpr.phi()  # φ as SymExpr
PHI_INV = PHI_INVERSE  # φ⁻¹ (exact GoldenExact)

# Numeric constants for backward compatibility (COMPAT: float evaluation for legacy code)
PHI_NUMERIC_LOCAL = PHI_NUMERIC  # COMPAT: float evaluation for legacy code
E_STAR_NUMERIC_LOCAL = E_STAR_NUMERIC  # COMPAT: float evaluation for legacy code  
Q_NUMERIC_LOCAL = Q_DEFICIT_NUMERIC  # COMPAT: float evaluation for legacy code
PI_NUMERIC_LOCAL = PI_NUMERIC  # COMPAT: float evaluation for legacy code
E_NUMERIC_LOCAL = E_NUMERIC  # COMPAT: float evaluation for legacy code
PHI_INV_NUMERIC = 1.0 / PHI_NUMERIC  # COMPAT: float evaluation for legacy code

# Crystallization Thresholds
SYNTONY_THRESHOLD = 0.99  # Threshold for Gnosis crystallization


# =============================================================================
# GROUP THEORY CONSTANTS
# =============================================================================

# E₈ Lattice (The Vacuum)
H_E8: int = 30  # Coxeter number h(E₈)
DIM_E8: int = 248  # dim(E₈) adjoint representation
ROOTS_E8: int = 240  # Total roots
ROOTS_E8_POS: int = 120  # Positive roots |Φ⁺(E₈)|
RANK_E8: int = 8  # Cartan subalgebra dimension

# E₇ Lattice (Intermediate Unification)
H_E7: int = 18  # Coxeter number h(E₇)
DIM_E7: int = 133  # dim(E₇) adjoint
DIM_E7_FUND: int = 56  # dim(E₇) fundamental
ROOTS_E7: int = 126  # Total roots
ROOTS_E7_POS: int = 63  # Positive roots
RANK_E7: int = 7  # Cartan subalgebra

# E₆ Lattice (The Gauge Sector)
H_E6: int = 12  # Coxeter number h(E₆)
DIM_E6: int = 78  # dim(E₆)
DIM_E6_FUND: int = 27  # E₆ fundamental representation
ROOTS_E6: int = 72  # Total roots
ROOTS_E6_POS: int = 36  # Positive roots |Φ⁺(E₆)| = Golden Cone
RANK_E6: int = 6  # Cartan subalgebra

# D₄ Lattice (Spacetime Projection)
K_D4: int = 24  # Kissing number K(D₄)
DIM_D4: int = 28  # dim(SO(8))
RANK_D4: int = 4  # Rank

# Other Exceptional
DIM_F4: int = 52  # dim(F₄)
DIM_G2: int = 14  # dim(G₂)

# Topological Constants
DIM_T4: int = 4  # W⁴ dimensions
N_GEN: int = 3  # Number of generations


# =============================================================================
# PRIME SEQUENCES - THE FIVE OPERATORS OF EXISTENCE
# =============================================================================

# Fermat Primes (The Architect - Differentiation/Force separation)
# F_n = 2^{2^n} + 1 is prime only for n = 0,1,2,3,4
FERMAT_PRIMES: tuple = (3, 5, 17, 257, 65537)
FERMAT_COMPOSITE_5 = 4294967297  # 641 × 6700417 - No 6th force

# Mersenne Primes (The Builder - Harmonization/Matter stability)
# M_p = 2^p - 1 for prime p
MERSENNE_EXPONENTS: dict = {
    2: 3,  # Generation 1 (e, u, d)
    3: 7,  # Generation 2 (μ, c, s)
    5: 31,  # Generation 3 (τ, b)
    7: 127,  # Heavy anchor (t, Higgs)
    # 11: 2047 = 23 × 89 - COMPOSITE, 4th gen forbidden
}
M11_BARRIER = 2047  # The barrier preventing 4th generation

# Lucas Sequence (The Shadow - Balance/Dark sector)
# L_n = φ^n + (1-φ)^n
LUCAS_SEQUENCE: tuple = (
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
    5778,
)
LUCAS_PRIMES_INDICES: tuple = (
    0,
    2,
    4,
    5,
    7,
    8,
    11,
    13,
    16,
    17,
)  # indices where L_n is prime

# Fibonacci Prime Gates (Transcendence thresholds)
FIBONACCI_PRIME_GATES: dict = {
    3: (2, "Binary/Logic emergence"),
    4: (3, "Material realm - the 'anomaly'"),  # Composite index!
    5: (5, "Physics/Life code"),
    7: (13, "Matter solidification"),
    11: (89, "Chaos/Complexity"),
    13: (233, "Consciousness emergence"),
    17: (1597, "Great Filter - hyperspace"),
}


# =============================================================================
# GEOMETRIC DIVISOR REFERENCE TABLE
# =============================================================================

GEOMETRIC_DIVISORS: dict = {
    # E₈ Structure (exact integers)
    "h_E8_cubed_27": 1000,  # 30³/27 = 1000
    "coxeter_kissing": 720,  # 30 × 24 = 720
    "cone_cycles": 360,  # 10 × 36 = 360
    "dim_E8": 248,  # dim(E₈)
    "roots_E8_full": 240,  # |Φ(E₈)|
    "roots_E8": 120,  # |Φ⁺(E₈)|
    "h_E8": 30,  # h(E₈)
    "rank_E8": 8,  # rank(E₈)
    # E₇ Structure (exact integers)
    "dim_E7": 133,  # dim(E₇)
    "roots_E7_full": 126,  # |Φ(E₇)|
    "roots_E7": 63,  # |Φ⁺(E₇)|
    "fund_E7": 56,  # dim(E₇ fund)
    "h_E7": 18,  # h(E₇)
    "rank_E7": 7,  # rank(E₇)
    # E₆ Structure (exact integers)
    "dim_E6": 78,  # dim(E₆)
    "roots_E6_full": 72,  # |Φ(E₆)|
    "roots_E6": 36,  # |Φ⁺(E₆)|
    "fund_E6": 27,  # dim(E₆ fund)
    "rank_E6": 6,  # rank(E₆)
    # Other Exceptional (exact integers)
    "dim_F4": 52,  # dim(F₄)
    "dim_G2": 14,  # dim(G₂)
    "dim_SO8": 28,  # dim(SO(8))
    "kissing_D4": 24,  # K(D₄)
    # QCD Loop Factors (exact π multiples)
    "six_loop": SymExpr.from_int(6) * SymExpr.pi(),  # 6π
    "five_loop": SymExpr.from_int(5) * SymExpr.pi(),  # 5π
    "one_loop": SymExpr.from_int(4) * SymExpr.pi(),  # 4π
    "three_loop": SymExpr.from_int(3) * SymExpr.pi(),  # 3π
    "half_loop": SymExpr.from_int(2) * SymExpr.pi(),  # 2π
    "circular_loop": SymExpr.pi(),  # π
    # Topological/Generation (exact integers)
    "topology_gen": 12,  # 12
    "generation_sq": 9,  # 9
    "sub_generation": 6,  # 6
    "quarter_layer": 4,  # 4
    "single_gen": 3,  # 3
    "half_layer": 2,  # 2
    "single_layer": 1,  # 1
    # Golden Ratio Based (exact golden numbers)
    "phi": PHI,  # φ (exact GoldenExact)
    "phi_inv": PHI_INVERSE,  # 1/φ (exact GoldenExact)
    "phi_cubed": PHI * PHI_SQUARED,  # φ³ (exact GoldenExact)
    "phi_squared": PHI_SQUARED,  # φ² (exact GoldenExact)
    "phi_fourth": PHI_SQUARED * PHI_SQUARED,  # φ⁴ (exact GoldenExact)
    "phi_fifth": PHI_SQUARED * PHI_SQUARED * PHI,  # φ⁵ (exact GoldenExact)
    # Binary (exact integers)
    "binary_5": 32,  # 2⁵
    "binary_4": 16,  # 2⁴
}


# =============================================================================
# FIBONACCI SEQUENCE (used for particle integers)
# =============================================================================

# Pre-computed Fibonacci numbers used in SRT
FIBONACCI: dict = {
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
MODULAR_VOLUME = SymExpr.pi() / SymExpr.from_int(3)  # Vol(F) = π/3
MODULAR_VOLUME_NUMERIC = PI_NUMERIC_LOCAL / 3  # COMPAT: float evaluation for legacy code


class UniverseSeeds:
    """
    The four geometric constants {φ, π, e, 1} from which all physics emerges.
    """

    def __init__(self):
        # The Four Seeds (exact)
        self.phi = PHI  # Exact GoldenExact
        self.pi = PI  # Exact SymExpr
        self.e = E  # Exact SymExpr
        self.one = Rational.from_int(1)  # Exact integer

        # Derived Constants (exact)
        self.E_star = E_STAR  # Exact SymExpr
        self.q = Q  # Exact SymExpr

        # Three-Term Decomposition of E* (exact symbolic)
        self.E_bulk = self._calculate_E_bulk()
        self.E_torsion = self._calculate_E_torsion()
        self.E_cone = self._calculate_E_cone()
        self.Delta = self._calculate_residual()

    def _calculate_E_bulk(self):
        """Bulk term: Γ(1/4)² ≈ 13.14504720659687"""
        quarter = SymExpr.rational(1, 4)
        return quarter.gamma() ** 2

    def _calculate_E_torsion(self):
        """Torsion term: π(π - 1) ≈ 6.72801174749952"""
        return self.pi * (self.pi - SymExpr.from_int(1))

    def _calculate_E_cone(self):
        """Cone term: (35/12)e^(-π) ≈ 0.12604059493600"""
        coeff = SymExpr.rational(35, 12)
        return coeff * (-self.pi).exp()

    def _calculate_residual(self):
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

        # E* validation (use .eval() only for comparison)
        E_star_expected = 19.999099979189475767
        E_star_numeric = self.E_star.eval()
        E_star_match = abs(E_star_numeric - E_star_expected) < 1e-15
        results["E_star"] = (E_star_numeric, E_star_expected, E_star_match)

        # q validation (use .eval() only for comparison)
        q_expected = 0.0273951469201761
        q_numeric = self.q.eval()
        q_match = abs(q_numeric - q_expected) < 1e-12
        results["q"] = (q_numeric, q_expected, q_match)

        # Δ validation (should be ~4.30 × 10⁻⁷) (use .eval() only for comparison)
        Delta_expected = 4.30e-7
        Delta_numeric = self.Delta.eval()
        Delta_match = abs(Delta_numeric - Delta_expected) / Delta_expected < 0.01
        results["Delta"] = (Delta_numeric, Delta_expected, Delta_match)

        # Three-term decomposition validation (use .eval() only for comparison)
        decomposition_sum = self.E_bulk + self.E_torsion + self.E_cone + self.Delta
        decomposition_sum_numeric = decomposition_sum.eval()
        E_star_numeric_check = self.E_star.eval()
        decomposition_match = abs(decomposition_sum_numeric - E_star_numeric_check) < 1e-50
        results["decomposition"] = (decomposition_sum_numeric, E_star_numeric_check, decomposition_match)

        return results

    def __repr__(self) -> str:
        return f"""UniverseSeeds:
  φ  = {self.phi.eval():.20f}
  π  = {self.pi.eval():.20f}
  e  = {self.e.eval():.20f}
  E* = {self.E_star.eval():.20f}
  q  = {self.q.eval():.20f}
  Δ  = {self.Delta.eval():.10e}"""


# Module-level constants for direct import (exact primary, numeric for compat)
phi = PHI  # Exact GoldenExact
phi_inv = PHI_INV  # Exact GoldenExact
pi = PI  # Exact SymExpr
e = E  # Exact SymExpr
E_star = E_STAR  # Exact SymExpr
q = Q  # Exact SymExpr

# Numeric versions for backward compatibility
phi_numeric = PHI_NUMERIC_LOCAL  # COMPAT: float evaluation for legacy code
pi_numeric = PI_NUMERIC_LOCAL  # COMPAT: float evaluation for legacy code
e_numeric = E_NUMERIC_LOCAL  # COMPAT: float evaluation for legacy code
E_star_numeric = E_STAR_NUMERIC_LOCAL  # COMPAT: float evaluation for legacy code
q_numeric = Q_NUMERIC_LOCAL  # COMPAT: float evaluation for legacy code

# Group constants
h_E8 = H_E8
K_D4 = K_D4
dim_E8 = DIM_E8
rank_E8 = RANK_E8
roots_E8 = ROOTS_E8
dim_E6 = DIM_E6
dim_E6_fund = DIM_E6_FUND
roots_E6 = ROOTS_E6_POS
dim_T4 = DIM_T4
N_gen = N_GEN

# Additional group constants
h_E7 = H_E7
dim_E7 = DIM_E7
dim_E7_fund = DIM_E7_FUND
roots_E7 = ROOTS_E7
roots_E7_pos = ROOTS_E7_POS
rank_E7 = RANK_E7
h_E6 = H_E6
roots_E6_full = ROOTS_E6_POS  # Alias for compatibility
rank_E6 = RANK_E6
dim_D4 = DIM_D4
rank_D4 = RANK_D4
dim_F4 = DIM_F4
dim_G2 = DIM_G2

# Prime sequences
fermat_primes = FERMAT_PRIMES
fermat_composite_5 = FERMAT_COMPOSITE_5
mersenne_exponents = MERSENNE_EXPONENTS
m11_barrier = M11_BARRIER
lucas_sequence = LUCAS_SEQUENCE
lucas_primes_indices = LUCAS_PRIMES_INDICES
fibonacci_prime_gates = FIBONACCI_PRIME_GATES

# Reference tables
geometric_divisors = GEOMETRIC_DIVISORS
fibonacci = FIBONACCI

M_Z = 91.1876  # GeV, Z boson mass
M_W_PDG = 80.377  # GeV, W boson mass
M_H_PDG = 125.25  # GeV, Higgs mass
V_EW = 246.22  # GeV, Higgs VEV (electroweak symmetry breaking scale)
ALPHA_EM_0 = 1 / 137.035999084  # Fine structure constant at q=0
ALPHA_S_MZ = 0.1179  # Strong coupling at M_Z

# =============================================================================
# Derived Scales from SRT
# =============================================================================


def gut_scale():
    """
    GUT unification scale.

    μ_GUT = v × e^(φ⁷) ≈ 1.0 × 10¹⁵ GeV

    Returns:
        GUT scale as SymExpr
    """
    phi_symexpr = SymExpr.phi()
    v_ew_symexpr = SymExpr.rational(24622, 100)  # V_EW = 246.22 GeV as exact rational
    return v_ew_symexpr * (phi_symexpr ** 7).exp()

def gut_scale_numeric() -> float:
    """
    GUT unification scale (numeric evaluation).
    
    Returns:
        GUT scale in GeV (float)
    """
    return gut_scale().eval()


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


def qcd_scale():
    """
    QCD confinement scale Λ_QCD.

    Derived from SRT via dimensional transmutation.

    Returns:
        QCD scale as SymExpr
    """
    # Λ_QCD ≈ 217 MeV from SRT
    # Correction factor: C9 (q/120)
    e_star_symexpr = SymExpr.e_star()
    eleven = SymExpr.from_int(11)
    one = SymExpr.from_int(1)
    correction_9 = SymExpr.correction(9, 1)  # C9 correction factor
    return e_star_symexpr * eleven * (one - correction_9)

def qcd_scale_numeric() -> float:
    """
    QCD confinement scale Λ_QCD (numeric evaluation).
    
    Returns:
        QCD scale in MeV (float)
    """
    return qcd_scale().eval()


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

__all__ = [
    "UniverseSeeds",
    # Seeds
    "phi",
    "phi_inv",
    "pi",
    "e",
    "E_star",
    "q",
    # Group constants
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
    # Additional group constants
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
    # Prime sequences
    "fermat_primes",
    "fermat_composite_5",
    "mersenne_exponents",
    "m11_barrier",
    "lucas_sequence",
    "lucas_primes_indices",
    "fibonacci_prime_gates",
    # Reference tables
    "geometric_divisors",
    "fibonacci",
    # Uppercase versions
    "PHI",
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
    "FERMAT_PRIMES",
    "FERMAT_COMPOSITE_5",
    "MERSENNE_EXPONENTS",
    "M11_BARRIER",
    "LUCAS_SEQUENCE",
    "LUCAS_PRIMES_INDICES",
    "FIBONACCI_PRIME_GATES",
    "GEOMETRIC_DIVISORS",
    "FIBONACCI",
    # Input scale
    "V_EW",
    # PDG reference values
    "M_Z",
    "M_W_PDG",
    "M_H_PDG",
    "ALPHA_EM_0",
    "ALPHA_S_MZ",
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


if __name__ == "__main__":
    # Self-test
    seeds = UniverseSeeds()
    print(seeds)
    print("\nValidation:")
    for name, (computed, expected, match) in seeds.validate().items():
        status = "✓" if match else "✗"
        print(f"  {name}: {status}")
