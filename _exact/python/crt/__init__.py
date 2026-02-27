"""
CRT Core - Cosmological Recursion Theory operators and metrics.

Provides the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework
for state evolution and analysis.

Components:
- **Operators**: D̂ (differentiation), Ĥ (harmonization), R̂ (recursion)
- **Metrics**: Syntony S(Ψ), Gnosis layers (0-3)
- **Evolution**: Trajectory tracking, attractor finding

Basic Usage:
    >>> import syntonic_applications as syn
    >>> from syntonic_applications.crt import create_dhsr_system

    >>> # Create DHSR system
    >>> R_op, S_comp, G_comp = create_dhsr_system()

    >>> # Create a state
    >>> psi = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

    >>> # Apply recursion
    >>> evolved = R_op.apply(psi)

    >>> # Compute metrics
    >>> syntony = S_comp.compute(psi)
    >>> gnosis = G_comp.compute_layer(psi)

    >>> # Or use the evolver for full trajectory
    >>> from syntonic_applications.crt import DHSREvolver
    >>> evolver = DHSREvolver()
    >>> trajectory = evolver.evolve(psi, n_steps=100)
    >>> print(trajectory.summary())
"""

from typing import Tuple

# Extended hierarchy corrections
from applications._core import (
    hierarchy_apply_collapse_threshold_correction,
    hierarchy_apply_coxeter_kissing_correction,
    hierarchy_apply_e7_correction,
    hierarchy_coxeter_kissing_720,
    hierarchy_d4_coxeter,
    hierarchy_d4_dim,
    hierarchy_d4_kissing,
    hierarchy_d4_rank,
    hierarchy_e6_coxeter,
    hierarchy_e6_dim,
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
    hierarchy_e8_dim,
    hierarchy_e8_positive_roots,
    hierarchy_e8_rank,
    # Newly exposed hierarchy constants
    hierarchy_e8_roots,
    hierarchy_exponent,
    hierarchy_f4_dim,
    hierarchy_f4_rank,
    hierarchy_g2_dim,
    hierarchy_g2_rank,
)
from applications.crt.extended_hierarchy import (
    apply_collapse_threshold,
    apply_coxeter_kissing,
    apply_e7_correction,
)

# Operators
from applications.crt.operators import (
    K_D4,
    DampingProjector,
    DifferentiationOperator,
    FourierProjector,
    GnosisComputer,
    HarmonizationOperator,
    LaplacianOperator,
    OperatorBase,
    RecursionOperator,
    SyntonyComputer,
    create_damping_cascade,
    create_mode_projectors,
    default_differentiation_operator,
    default_gnosis_computer,
    default_harmonization_operator,
    default_recursion_operator,
    syntony_entropy,
    syntony_quick,
    syntony_spectral,
)

# Structure dimensions dictionary (comprehensive)
STRUCTURE_DIMENSIONS = {
    # E8 Family
    "E8_DIM": hierarchy_e8_dim(),  # 248
    "E8_ROOTS": hierarchy_e8_roots(),  # 240
    "E8_POSITIVE": hierarchy_e8_positive_roots(),  # 120
    "E8_RANK": hierarchy_e8_rank(),  # 8
    "E8_COXETER": hierarchy_e8_coxeter(),  # 30
    # E7 Family
    "E7_DIM": hierarchy_e7_dim(),  # 133
    "E7_ROOTS": hierarchy_e7_roots(),  # 126
    "E7_POSITIVE": hierarchy_e7_positive_roots(),  # 63
    "E7_FUNDAMENTAL": hierarchy_e7_fundamental(),  # 56
    "E7_RANK": hierarchy_e7_rank(),  # 7
    "E7_COXETER": hierarchy_e7_coxeter(),  # 18
    # E6 Family
    "E6_DIM": hierarchy_e6_dim(),  # 78
    "E6_ROOTS": hierarchy_e6_roots(),  # 72
    "E6_POSITIVE": hierarchy_e6_positive_roots(),  # 36 (Golden Cone)
    "E6_FUNDAMENTAL": hierarchy_e6_fundamental(),  # 27
    "E6_RANK": hierarchy_e6_rank(),  # 6
    "E6_COXETER": hierarchy_e6_coxeter(),  # 12
    # D4 Family
    "D4_DIM": hierarchy_d4_dim(),  # 28
    "D4_KISSING": hierarchy_d4_kissing(),  # 24
    "D4_RANK": hierarchy_d4_rank(),  # 4
    "D4_COXETER": hierarchy_d4_coxeter(),  # 6
    # G2 Family
    "G2_DIM": hierarchy_g2_dim(),  # 14
    "G2_RANK": hierarchy_g2_rank(),  # 2
    # F4 Family
    "F4_DIM": hierarchy_f4_dim(),  # 52
    "F4_RANK": hierarchy_f4_rank(),  # 4
    # Derived quantities
    "COXETER_KISSING": hierarchy_coxeter_kissing_720(),  # 720
    "HIERARCHY_EXPONENT": hierarchy_exponent(),  # 719
}

# ============================================================================
# STRUCTURE_DIMENSIONS Documentation
# ============================================================================
"""
Complete SRT Lie group dimensional hierarchy dictionary.

This dictionary contains all fundamental constants derived from the
exceptional Lie group geometry underlying SRT theory. Each constant
has deep physical and mathematical significance:

Critical SRT Constants:
----------------------
- E6_POSITIVE (36): Golden Cone cardinality |Φ⁺(E₆)|
  * Consciousness emergence threshold
  * Transcendence gate count
  * Self-reference criticality parameter

- D4_KISSING (24): Consciousness emergence threshold
  * D₄ kissing number - maximum sphere packings in 4D
  * D₄ → M₅ gap bridge (24 → 31)
  * Neural network stability parameter

- COXETER_KISSING (720): Hierarchy scale parameter
  * Product of E₈ Coxeter (30) × D₄ kissing (24)
  * Universal SRT scaling constant

Neural Network Applications:
---------------------------
- E8_DIM (248): Optimal embedding dimensions for transformers
- E6_POSITIVE (36): Stable layer sizes in resonance networks
- D4_KISSING (24): Attention head counts for stable patterns
- E7_FUNDAMENTAL (56): Memory dimension constraints

Physics Applications:
--------------------
- E8_ROOTS (240): Gauge symmetry breaking pattern counting
- E7_FUNDAMENTAL (56): Supersymmetry goldstino degrees of freedom
- E6_FUNDAMENTAL (27): Cubic surface theory in algebraic geometry
- E6_POSITIVE (36): Unification predictions and phase transitions

Mathematical Structure:
---------------------
Each Lie group constant derives from the root system geometry:
- ROOTS: Total number of roots in the root lattice
- POSITIVE: Roots in the positive Weyl chamber
- RANK: Dimension of Cartan subalgebra
- COXETER: Period of Weyl group action
- FUNDAMENTAL: Dimension of fundamental representation

Example Usage:
-------------
>>> from syntonic_applications.crt import STRUCTURE_DIMENSIONS
>>> golden_cone = STRUCTURE_DIMENSIONS['E6_POSITIVE']  # 36
>>> consciousness_threshold = STRUCTURE_DIMENSIONS['D4_KISSING']  # 24
"""

# Evolution
from applications.crt.dhsr_fused.dhsr_evolution import (
    DHSREvolver,
    SyntonyTrajectory,
    default_evolver,
)


def create_dhsr_system(
    alpha_0: float = 0.1,
    beta_0: float = 0.618,
    zeta_0: float = 0.01,
    gamma_0: float = 0.1,
    num_modes: int = 8,
    num_dampers: int = 3,
) -> Tuple[RecursionOperator, SyntonyComputer, GnosisComputer]:
    """
    Create a complete DHSR system with specified parameters.

    This is the recommended way to create a configured DHSR system
    for evolution and analysis.

    Args:
        alpha_0: Base differentiation coupling strength
        beta_0: Base harmonization damping strength (default: 1/φ)
        zeta_0: Base Laplacian diffusion coefficient
        gamma_0: Syntony projection strength
        num_modes: Number of Fourier modes for differentiation
        num_dampers: Number of damping levels for harmonization

    Returns:
        (RecursionOperator, SyntonyComputer, GnosisComputer) tuple

    Example:
        >>> R_op, S_comp, G_comp = create_dhsr_system()
        >>> evolved = R_op.apply(state)
        >>> syntony = S_comp.compute(state)
        >>> layer = G_comp.compute_layer(state)
    """
    # Create operators
    D_op = DifferentiationOperator(
        alpha_0=alpha_0,
        zeta_0=zeta_0,
        num_modes=num_modes,
    )

    H_op = HarmonizationOperator(
        beta_0=beta_0,
        gamma_0=gamma_0,
        num_dampers=num_dampers,
    )

    R_op = RecursionOperator(diff_op=D_op, harm_op=H_op)

    # Create metrics
    S_comp = SyntonyComputer(diff_op=D_op, harm_op=H_op)
    G_comp = GnosisComputer(recursion_op=R_op, syntony_computer=S_comp)

    return R_op, S_comp, G_comp


def create_evolver(
    alpha_0: float = 0.1,
    beta_0: float = 0.618,
    **kwargs,
) -> DHSREvolver:
    """
    Create a configured DHSR evolver.

    Args:
        alpha_0: Base differentiation strength
        beta_0: Base harmonization strength
        **kwargs: Additional parameters for create_dhsr_system

    Returns:
        DHSREvolver configured with specified parameters
    """
    R_op, S_comp, G_comp = create_dhsr_system(
        alpha_0=alpha_0,
        beta_0=beta_0,
        **kwargs,
    )

    return DHSREvolver(
        recursion_op=R_op,
        syntony_computer=S_comp,
        gnosis_computer=G_comp,
    )


__all__ = [
    # Operators
    "OperatorBase",
    "FourierProjector",
    "DampingProjector",
    "LaplacianOperator",
    "create_mode_projectors",
    "create_damping_cascade",
    "DifferentiationOperator",
    "default_differentiation_operator",
    "HarmonizationOperator",
    "default_harmonization_operator",
    "RecursionOperator",
    "default_recursion_operator",
    "SyntonyComputer",
    "syntony_entropy",
    "syntony_spectral",
    "syntony_quick",
    "GnosisComputer",
    "default_gnosis_computer",
    "K_D4",
    # Evolution
    "SyntonyTrajectory",
    "DHSREvolver",
    "default_evolver",
    # Factories
    "create_dhsr_system",
    "create_evolver",
    # Extended Hierarchy Corrections (Rust backend)
    "hierarchy_apply_collapse_threshold_correction",
    "hierarchy_apply_coxeter_kissing_correction",
    "hierarchy_apply_e7_correction",
    # Extended Hierarchy Python Wrappers
    "apply_collapse_threshold",
    "apply_coxeter_kissing",
    "apply_e7_correction",
    # Hierarchy Constants
    "hierarchy_coxeter_kissing_720",
    "hierarchy_d4_coxeter",
    "hierarchy_d4_dim",
    "hierarchy_d4_kissing",
    "hierarchy_d4_rank",
    "hierarchy_e6_coxeter",
    "hierarchy_e6_dim",
    "hierarchy_e6_fundamental",
    "hierarchy_e6_positive_roots",
    "hierarchy_e6_rank",
    "hierarchy_e6_roots",
    "hierarchy_e7_coxeter",
    "hierarchy_e7_dim",
    "hierarchy_e7_fundamental",
    "hierarchy_e7_positive_roots",
    "hierarchy_e7_rank",
    "hierarchy_e7_roots",
    "hierarchy_e8_coxeter",
    "hierarchy_e8_dim",
    "hierarchy_e8_positive_roots",
    "hierarchy_e8_rank",
    "hierarchy_e8_roots",
    "hierarchy_exponent",
    "hierarchy_f4_dim",
    "hierarchy_f4_rank",
    "hierarchy_g2_dim",
    "hierarchy_g2_rank",
    # Structure Dimensions
    "STRUCTURE_DIMENSIONS",
]
