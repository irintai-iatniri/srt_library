"""
SRT Spectral - Theta series, heat kernels, and Möbius regularization.

The spectral theory in SRT connects:
- Theta series Θ₄(t) on the W⁴ torus
- Heat kernel traces Tr[exp(-t·L²)]
- Knot Laplacian eigenvalues
- Möbius regularization for extracting E* = e^π - π

Classes:
    ThetaSeries - Θ₄(t) = Σₙ w(n)·exp(-π|n|²/t)
    HeatKernel - K(t) = Tr[exp(-t·L²)]
    KnotLaplacian - L²_knot operator on winding states
    MobiusRegularizer - Möbius inversion for E* extraction

Functions:
    theta_series() - Factory for ThetaSeries
    heat_kernel() - Factory for HeatKernel
    knot_laplacian() - Factory for KnotLaplacian
    mobius_regularizer() - Factory for MobiusRegularizer
    compute_e_star() - Quick E* computation
"""

from srt_library.theory_unique_components.srt.spectral.heat_kernel import (
    HeatKernel,
    heat_kernel,
)
from srt_library.theory_unique_components.srt.spectral.knot_laplacian import (
    KnotLaplacian,
    knot_laplacian,
)
from srt_library.theory_unique_components.srt.spectral.mobius import (
    MobiusRegularizer,
    compute_e_star,
    mobius_regularizer,
)
from srt_library.theory_unique_components.srt.spectral.theta_series import (
    ThetaSeries,
    theta_series,
)

__all__ = [
    "ThetaSeries",
    "theta_series",
    "HeatKernel",
    "heat_kernel",
    "KnotLaplacian",
    "knot_laplacian",
    "MobiusRegularizer",
    "mobius_regularizer",
    "compute_e_star",
]
