"""
Winding State - Winding numbers on the T^4 torus.

A winding state |n> = |n_7, n_8, n_9, n_10> represents a configuration
of winding numbers on the internal 4-torus. These are the fundamental
quantum numbers in SRT from which all charges derive.

This module re-exports the high-performance Rust implementation from
srt_library.core. The Rust WindingState provides ~50x speedup for
enumeration operations.

Example:
    >>> from srt_library.theory_unique_components.srt.geometry import winding_state
    >>> n = winding_state(1, 2, 0, -1)
    >>> n.norm_squared
    6
    >>> n.generation
    2
"""

from __future__ import annotations

from typing import Dict, List

# Re-export Rust WindingState and enumeration functions
from srt_library.core import (
    WindingState,
    count_windings,
    enumerate_windings_exact_norm,
)
from srt_library.core import (
    enumerate_windings as _enumerate_windings,
)
from srt_library.core import (
    enumerate_windings_by_norm as _enumerate_windings_by_norm,
)

# Re-export WindingState class for type annotations
__all__ = [
    "WindingState",
    "winding_state",
    "enumerate_windings",
    "enumerate_windings_by_norm",
    "enumerate_windings_exact_norm",
    "count_windings",
]


def winding_state(n7: int, n8: int = 0, n9: int = 0, n10: int = 0) -> WindingState:
    """
    Create a winding state |n_7, n_8, n_9, n_10>.

    Factory function for WindingState with optional default zeros.

    Args:
        n7: Winding number on S^1_7
        n8: Winding number on S^1_8 (default: 0)
        n9: Winding number on S^1_9 (default: 0)
        n10: Winding number on S^1_10 (default: 0)

    Returns:
        WindingState instance

    Example:
        >>> n = winding_state(1, 2)  # |1,2,0,0>
        >>> n.norm_squared
        5
    """
    return WindingState(n7, n8, n9, n10)


def enumerate_windings(max_norm: int = 10) -> List[WindingState]:
    """
    Enumerate all winding states with |n| <= max_norm.

    Returns winding states with norm <= max_norm.
    Uses high-performance Rust implementation (~50x faster than Python).

    Args:
        max_norm: Maximum norm to enumerate

    Returns:
        List of WindingState instances with norm <= max_norm

    Example:
        >>> states = enumerate_windings(1)
        >>> len(states)
        9
    """
    return _enumerate_windings(max_norm)


def enumerate_windings_by_norm(max_norm_sq: int) -> Dict[int, List[WindingState]]:
    """
    Enumerate all winding states with |n|^2 <= max_norm_sq, grouped by norm squared.

    Args:
        max_norm_sq: Maximum squared norm to enumerate

    Returns:
        Dictionary mapping norm_squared values to lists of WindingState

    Example:
        >>> by_norm = enumerate_windings_by_norm(4)
        >>> len(by_norm[1])  # States with |n|^2 = 1
        8
    """
    return _enumerate_windings_by_norm(max_norm_sq)
