"""
CRT Operators - DHSR operator implementations.

Provides:
- OperatorBase: Abstract base class for all operators
- FourierProjector: Fourier mode projection
- DampingProjector: High-frequency damping
- LaplacianOperator: Discrete Laplacian ∇²
- DifferentiationOperator: D̂ operator
- HarmonizationOperator: Ĥ operator
- RecursionOperator: R̂ = Ĥ ∘ D̂
"""

from srt_library.theory_unique_components.crt.dhsr_fused.dhsr_loop import (
    DHSRLoop,
    compute_optimal_alpha,
    compute_optimal_strength,
    differentiate_step,
    evolve_state,
    harmonize_step,
    single_dhsr_cycle,
)
from srt_library.theory_unique_components.crt.operators.base import OperatorBase
from srt_library.theory_unique_components.crt.operators.differentiation import (
    DifferentiationOperator,
    default_differentiation_operator,
)
from srt_library.theory_unique_components.crt.operators.gnosis import (
    K_D4,
    GnosisComputer,
    default_gnosis_computer,
)
from srt_library.theory_unique_components.crt.operators.harmonization import (
    HarmonizationOperator,
    default_harmonization_operator,
)
from srt_library.theory_unique_components.crt.operators.mobius import (
    apply_mobius_mask,
    check_m11_stability,
    compute_mobius_mask,
    get_composite_barrier_indices,
    get_squarefree_indices,
    mobius,
)
from srt_library.theory_unique_components.crt.operators.projectors import (
    DampingProjector,
    FourierProjector,
    LaplacianOperator,
    create_damping_cascade,
    create_mode_projectors,
)
from srt_library.theory_unique_components.crt.operators.recursion import (
    RecursionOperator,
    default_recursion_operator,
)
from srt_library.theory_unique_components.crt.operators.syntony import (
    SyntonyComputer,
    syntony_entropy,
    syntony_quick,
    syntony_spectral,
)

__all__ = [
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
    # Mobius filter
    "mobius",
    "apply_mobius_mask",
    "check_m11_stability",
    "compute_mobius_mask",
    "get_squarefree_indices",
    "get_composite_barrier_indices",
    # DHSR loop
    "DHSRLoop",
    "evolve_state",
    "single_dhsr_cycle",
    "differentiate_step",
    "harmonize_step",
    "compute_optimal_alpha",
    "compute_optimal_strength",
]
