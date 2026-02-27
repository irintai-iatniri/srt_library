"""
Golden Recursion Map - R: n -> floor(phi * n) on Z^4.

The golden recursion map is fundamental to SRT mass hierarchies.
Applying R repeatedly expands winding states (φ > 1), with the
orbit depth determining the mass scale m ~ e^(-phi^k).

Uses exact golden ratio arithmetic from srt_library.core.exact.GoldenExact.

Example:
    >>> from srt_library.theory_unique_components.srt.golden import golden_recursion
    >>> from srt_library.theory_unique_components.srt.geometry import winding_state
    >>> R = golden_recursion()
    >>> n = winding_state(5, 3, 0, 0)
    >>> R.apply(n)
    WindingState(n7=8, n8=4, n9=0, n10=0)  # floor(phi * (5,3,0,0))
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from srt_library.core.exact import PHI, GoldenExact
from srt_library.theory_unique_components.srt.geometry.winding import WindingState, enumerate_windings

if TYPE_CHECKING:
    pass


class GoldenRecursionMap:
    """
    Golden recursion map R: Z^4 -> Z^4 defined by R(n) = floor(phi * n).

    The golden recursion is applied component-wise. Since φ > 1, repeated
    application expands states (increasing |n|²), not contracts them.
    The orbit structure determines the mass generation.

    Uses exact golden ratio φ from srt_library.core.exact.GoldenExact.

    Properties:
        - R preserves the integer lattice Z^4
        - R(n) expands: |R(n)|² > |n|² for non-zero n
        - The orbit depth k determines mass scale: m ~ e^(-phi^k)

    Attributes:
        phi: Golden ratio float value (from exact PHI.eval())
        phi_exact: Exact GoldenExact representation of φ

    Example:
        >>> R = GoldenRecursionMap()
        >>> n = WindingState(10, 0, 0, 0)
        >>> R.orbit(n)
        [|10,0,0,0>, |16,0,0,0>, |25,0,0,0>, ...]  # Orbit until periodic
    """

    def __init__(self, phi: Optional[float] = None):
        """
        Initialize golden recursion map.

        Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()

    @property
    def phi(self) -> float:
        """Golden ratio float value."""
        return self._phi

    @property
    def phi_exact(self) -> GoldenExact:
        """Exact GoldenExact representation of the golden ratio."""
        return self._phi_exact

    # Maximum winding component to prevent overflow (i64 limit / phi)
    _MAX_COMPONENT = (
        5_000_000_000_000_000_000  # ~5e18, safe for i64 after phi multiplication
    )

    def apply(self, n: WindingState) -> WindingState:
        """
        Apply R(n) = floor(phi * n) component-wise.

        Args:
            n: Winding state

        Returns:
            R(n) = (floor(phi*n_7), floor(phi*n_8), floor(phi*n_9), floor(phi*n_10))

        Raises:
            OverflowError: If any component exceeds safe range
        """
        # Check for overflow before computing
        if (
            abs(n.n7) > self._MAX_COMPONENT
            or abs(n.n8) > self._MAX_COMPONENT
            or abs(n.n9) > self._MAX_COMPONENT
            or abs(n.n10) > self._MAX_COMPONENT
        ):
            raise OverflowError(
                "Winding component exceeds safe range for golden recursion"
            )

        return WindingState(
            int(math.floor(self._phi * n.n7)),
            int(math.floor(self._phi * n.n8)),
            int(math.floor(self._phi * n.n9)),
            int(math.floor(self._phi * n.n10)),
        )

    def apply_inverse_approx(self, n: WindingState) -> WindingState:
        """
        Approximate inverse R^{-1}(n) = floor(n / phi) = floor(n * (phi - 1)).

        Note: This is only approximate since R is not bijective.

        Args:
            n: Winding state

        Returns:
            Approximate inverse mapping
        """
        inv_phi = self._phi - 1  # 1/phi = phi - 1
        return WindingState(
            int(math.floor(n.n7 * inv_phi)),
            int(math.floor(n.n8 * inv_phi)),
            int(math.floor(n.n9 * inv_phi)),
            int(math.floor(n.n10 * inv_phi)),
        )

    def orbit(
        self,
        n: WindingState,
        max_depth: int = 100,
    ) -> List[WindingState]:
        """
        Compute orbit n -> R(n) -> R^2(n) -> ... until cycle, max_depth, or overflow.

        Args:
            n: Starting winding state
            max_depth: Maximum iterations

        Returns:
            List of states in the orbit (including initial state)
        """
        trajectory = [n]
        seen: Set[Tuple[int, int, int, int]] = {n.n}
        current = n

        for _ in range(max_depth):
            try:
                next_state = self.apply(current)
            except OverflowError:
                # Orbit diverged beyond representable range
                break
            if next_state.n in seen:
                # Found a cycle
                trajectory.append(next_state)
                break
            trajectory.append(next_state)
            seen.add(next_state.n)
            current = next_state

        return trajectory

    def orbit_depth(
        self,
        n: WindingState,
        max_depth: int = 100,
    ) -> int:
        """
        Number of iterations to reach a fixed point or cycle.

        This determines the generation k where mass ~ e^(-phi^k).

        Args:
            n: Starting winding state
            max_depth: Maximum iterations

        Returns:
            Depth k, or -1 if max_depth reached without convergence or overflow
        """
        seen: Set[Tuple[int, int, int, int]] = {n.n}
        current = n

        for k in range(max_depth):
            try:
                next_state = self.apply(current)
            except OverflowError:
                # Overflow means orbit diverged beyond representable range
                return -1
            if next_state.n in seen:
                return k
            seen.add(next_state.n)
            current = next_state

        return -1

    def is_fixed_point(self, n: WindingState) -> bool:
        """
        Check if R(n) = n.

        Args:
            n: Winding state to check

        Returns:
            True if n is a fixed point of R
        """
        return self.apply(n) == n

    def fixed_points(self, max_norm: int = 10) -> List[WindingState]:
        """
        Find all fixed points R(n) = n with |n| <= max_norm.

        The fixed points of the golden recursion are special winding
        states that don't change under R.

        Args:
            max_norm: Maximum norm to search

        Returns:
            List of fixed point WindingState
        """
        fixed = []
        for n in enumerate_windings(max_norm):
            if self.is_fixed_point(n):
                fixed.append(n)
        return fixed

    def periodic_points(
        self,
        period: int,
        max_norm: int = 10,
    ) -> List[WindingState]:
        """
        Find periodic points with given period.

        A point n is periodic with period p if R^p(n) = n but R^k(n) != n
        for 0 < k < p.

        Args:
            period: Period to search for
            max_norm: Maximum norm to search

        Returns:
            List of periodic WindingState with given period
        """
        periodic = []
        for n in enumerate_windings(max_norm):
            # Check if R^period(n) = n
            current = n
            for _ in range(period):
                current = self.apply(current)
            if current != n:
                continue
            # Check that period is minimal
            current = n
            is_smaller_period = False
            for k in range(1, period):
                current = self.apply(current)
                if current == n:
                    is_smaller_period = True
                    break
            if not is_smaller_period:
                periodic.append(n)
        return periodic

    def mass_scaling(self, n: WindingState, max_depth: int = 100) -> float:
        """
        Compute mass scale m ~ e^(-phi^k) where k = orbit_depth.

        The mass hierarchy in SRT comes from the golden recursion depth.
        The vacuum state (0,0,0,0) has mass scaling 1.0 (no suppression).

        Args:
            n: Winding state
            max_depth: Maximum depth for orbit calculation

        Returns:
            Mass scaling factor in (0, 1]
        """
        # Vacuum has no mass suppression
        if n.is_zero():
            return 1.0
        k = self.orbit_depth(n, max_depth)
        if k < 0:
            return 0.0  # No convergence
        # k=0 means fixed point, k>0 means deeper recursion
        return math.exp(-(self._phi**k))

    def generation(self, n: WindingState, max_depth: int = 100) -> int:
        """
        Determine generation number from orbit depth.

        Generations are labeled 1, 2, 3, ... corresponding to
        electron, muon, tau, etc.

        Args:
            n: Winding state
            max_depth: Maximum depth

        Returns:
            Generation number (1-indexed), or 0 if vacuum
        """
        if n.is_zero():
            return 0
        depth = self.orbit_depth(n, max_depth)
        if depth < 0:
            return -1  # No convergence
        # Map depth to generation (depth 0, 1, 2 -> gen 1, 2, 3)
        return max(1, depth + 1)

    def classify_orbit(
        self,
        n: WindingState,
        max_depth: int = 100,
    ) -> str:
        """
        Classify the orbit type of a winding state.

        Returns:
            - 'fixed': n is a fixed point
            - 'periodic-k': n has period k > 1
            - 'convergent': orbit converges to a fixed point
            - 'divergent': orbit doesn't converge in max_depth
        """
        if self.is_fixed_point(n):
            return "fixed"

        orbit = self.orbit(n, max_depth)
        if len(orbit) <= max_depth:
            # Orbit terminated - found cycle
            last = orbit[-1]
            if last == orbit[-2]:
                return "convergent"
            # Find period
            for i, state in enumerate(orbit[:-1]):
                if state == last:
                    period = len(orbit) - 1 - i
                    return f"periodic-{period}"
        return "divergent"

    def __repr__(self) -> str:
        return f"GoldenRecursionMap(phi={self._phi:.6f})"

    def __call__(self, n: WindingState) -> WindingState:
        """Allow R(n) syntax."""
        return self.apply(n)


def golden_recursion(phi: Optional[float] = None) -> GoldenRecursionMap:
    """
    Create a GoldenRecursionMap instance.

    Factory function for GoldenRecursionMap.
    Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

    Args:
        phi: Golden ratio float value. If None, uses PHI.eval().

    Returns:
        GoldenRecursionMap instance
    """
    return GoldenRecursionMap(phi=phi)


def apply_golden_recursion(
    n: WindingState,
    phi: Optional[float] = None,
) -> WindingState:
    """
    Apply golden recursion R(n) = floor(phi * n).

    Convenience function for single application.
    Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

    Args:
        n: Winding state
        phi: Golden ratio float value. If None, uses PHI.eval().

    Returns:
        R(n)
    """
    R = GoldenRecursionMap(phi=phi)
    return R.apply(n)
