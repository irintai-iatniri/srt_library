"""
DHSR Fused - Efficient combined DHSR cycles.

Provides:
- evolve_state: High-level state evolution
- DHSRLoop: Stateful iteration tracking
- DHSREvolver: Trajectory-based evolution
- SyntonyTrajectory: Evolution history analysis
- Reference implementations for DHSR operators
"""

from applications.crt.dhsr_fused.dhsr_evolution import (
    DHSREvolver,
    SyntonyTrajectory,
)
from applications.crt.dhsr_fused.dhsr_loop import (
    DHSRLoop,
    compute_optimal_alpha,
    compute_optimal_strength,
    differentiate_step,
    evolve_state,
    harmonize_step,
    single_dhsr_cycle,
)
from applications.crt.dhsr_fused.dhsr_reference import (
    PHI,
    PHI_INV,
    Q_DEFICIT,
    DHSRTrajectory,
    compute_syntony,
    differentiate,
    harmonize,
    recurse,
)

__all__ = [
    "DHSRLoop",
    "compute_optimal_alpha",
    "compute_optimal_strength",
    "differentiate_step",
    "evolve_state",
    "harmonize_step",
    "single_dhsr_cycle",
    "DHSREvolver",
    "SyntonyTrajectory",
    "PHI",
    "PHI_INV",
    "Q_DEFICIT",
    "DHSRTrajectory",
    "compute_syntony",
    "differentiate",
    "harmonize",
    "recurse",
]
