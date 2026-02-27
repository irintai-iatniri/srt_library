"""
SRT Functional - Syntony functional S[Ψ] with bound S ≤ φ.

The syntony functional measures the coherence of a quantum state
with respect to the golden recursion structure. It satisfies the
fundamental bound S[Ψ] ≤ φ for all physical states.

Classes:
    SyntonyFunctional - S[Ψ] = φ · Tr[exp(-⟨n,L²⟩/φ)] / Tr[exp(-⟨0,L²⟩/φ)]

Functions:
    syntony_functional() - Factory for SyntonyFunctional
    compute_syntony() - Quick S[Ψ] computation
"""

from srt_library.theory_unique_components.srt.functional.syntony import (
    SyntonyFunctional,
    compute_syntony,
    syntony_functional,
)

__all__ = [
    "SyntonyFunctional",
    "syntony_functional",
    "compute_syntony",
]
