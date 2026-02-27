"""
Golden Cone - The 36 E8 roots in the golden cone = Phi+(E6).

The golden cone is defined by Q(lambda) >= 0 where Q is the SRT
quadratic form. Exactly 36 positive E8 roots satisfy this condition,
and these form the positive roots of E6.

Example:
    >>> from syntonic_applications.srt.lattice import golden_cone, e8_lattice
    >>> cone = golden_cone()
    >>> len(cone.roots)
    36
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from applications.exact import PHI_NUMERIC, PHI
from applications._core import SymExpr, GoldenExact, Rational
from applications.srt.lattice.quadratic_form import QuadraticForm

if TYPE_CHECKING:
    from applications.srt.lattice.e8 import E8Lattice, E8Root


class GoldenProjector:
    """
    Golden projector P_phi: R^8 -> R^4.

    Decomposes E8 vectors into parallel (physical) and perpendicular
    (internal) components via the golden ratio embedding.

    The golden embedding uses:
        P_phi projects onto the 4D physical subspace
        P_perp projects onto the 4D internal subspace

    These projections are orthogonal: P_phi + P_perp = I
    And satisfy: P_phi^2 = P_phi, P_perp^2 = P_perp

    Attributes:
        phi: Golden ratio value
        quadratic_form: The associated quadratic form Q

    Example:
        >>> proj = GoldenProjector()
        >>> v = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> proj.project_parallel(v)
        (0.447..., 0.447..., 0.0, 0.0)
    """

    def __init__(self, phi: Optional[float] = None):
        """
        Initialize golden projector.

        Args:
            phi: Golden ratio value. If None, uses exact PHI.
        """
        if phi is None:
            # Use exact phi for construction, convert to float for compatibility
            self._phi_exact = PHI  # Exact GoldenExact
            self._phi = PHI.eval()  # COMPAT: float evaluation for legacy code
        else:
            self._phi = phi
            self._phi_exact = None  # User provided float value
        self._Q = QuadraticForm(phi=self._phi)

    @property
    def phi(self) -> float:
        """Golden ratio value (numeric)."""
        return self._phi
    
    @property
    def phi_exact(self):
        """Exact golden ratio value (GoldenExact if available)."""
        return self._phi_exact

    @property
    def quadratic_form(self) -> QuadraticForm:
        """The associated quadratic form."""
        return self._Q

    def project_parallel(self, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        P_phi(v) - project to physical 4D subspace.

        Args:
            v: 8-dimensional vector

        Returns:
            4-dimensional projection
        """
        return self._Q.project_parallel(v)

    def project_perpendicular(self, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        P_perp(v) - project to internal 4D subspace.

        Args:
            v: 8-dimensional vector

        Returns:
            4-dimensional projection
        """
        return self._Q.project_perpendicular(v)

    def projection_matrix_parallel(self) -> List[List[float]]:
        """4x8 projection matrix P_phi."""
        return self._Q.projection_matrix_parallel()

    def projection_matrix_perpendicular(self) -> List[List[float]]:
        """4x8 projection matrix P_perp."""
        return self._Q.projection_matrix_perpendicular()

    def norm_parallel(self, v: Tuple[float, ...]) -> float:
        """Norm |P_phi(v)|."""
        return math.sqrt(self._Q.parallel_norm_squared(v))

    def norm_perpendicular(self, v: Tuple[float, ...]) -> float:
        """Norm |P_perp(v)|."""
        return math.sqrt(self._Q.perpendicular_norm_squared(v))

    def compute_Q(self, v: Tuple[float, ...]) -> float:
        """Compute Q(v) = |P_phi(v)|^2 - |P_perp(v)|^2."""
        return self._Q.evaluate(v)

    def is_in_cone(self, v: Tuple[float, ...], tol: float = 1e-10) -> bool:  # eval boundary
        """Check if v is in the golden cone (Q >= 0)."""
        return self._Q.is_in_cone(v, tol)

    def __repr__(self) -> str:
        return f"GoldenProjector(phi={self._phi:.6f})"


class GoldenCone:
    """
    The golden cone in E8: 36 roots satisfying Q(lambda) >= 0.

    These are the positive roots of E6 embedded in E8 via the golden
    projector. The golden cone defines the physical particle spectrum
    in SRT.

    Key properties:
        - 36 roots = |Phi+(E6)|
        - All roots have Q(lambda) >= 0
        - Related to exceptional Jordan algebra

    Attributes:
        roots: List of 36 E8 roots in the cone
        projector: The golden projector

    Example:
        >>> from syntonic_applications.srt.lattice import e8_lattice
        >>> cone = GoldenCone()
        >>> len(cone.roots)
        36
    """

    def __init__(
        self,
        e8_lattice: Optional["E8Lattice"] = None,
        phi: Optional[float] = None,
        tolerance: float = 1e-10,  # eval boundary
    ):
        """
        Initialize golden cone.

        Args:
            e8_lattice: E8Lattice to filter. If None, creates new one.
            phi: Golden ratio value. If None, uses exact PHI.
            tolerance: Tolerance for cone membership test.
        """
        from applications.srt.lattice.e8 import E8Lattice

        if phi is None:
            # Use exact phi for construction, convert to float for compatibility
            self._phi_exact = PHI  # Exact GoldenExact
            self._phi = PHI.eval()  # COMPAT: float evaluation for legacy code
        else:
            self._phi = phi
            self._phi_exact = None  # User provided float value
        self._tolerance = tolerance
        self._projector = GoldenProjector(phi=self._phi)
        self._Q = self._projector.quadratic_form

        # Get E8 lattice
        if e8_lattice is None:
            e8_lattice = E8Lattice()
        self._e8 = e8_lattice

        # Filter all 240 roots to those in golden cone (gives 36)
        self._roots = self._filter_cone_roots()

    def _filter_cone_roots(self) -> List["E8Root"]:
        """Filter all E8 roots to those in the golden cone (36 roots)."""
        cone_roots = []
        for root in self._e8.roots:  # All 240 roots, not just positive
            if self.is_in_cone(root):
                cone_roots.append(root)
        return cone_roots

    @property
    def roots(self) -> List["E8Root"]:
        """36 E8 roots in the golden cone."""
        return self._roots.copy()

    @property
    def num_roots(self) -> int:
        """Number of roots in cone (36)."""
        return len(self._roots)

    @property
    def projector(self) -> GoldenProjector:
        """The golden projector."""
        return self._projector

    @property
    def quadratic_form(self) -> QuadraticForm:
        """The quadratic form Q."""
        return self._Q

    def is_in_cone(self, root: "E8Root") -> bool:
        """
        Test if root satisfies all cone constraints.

        Args:
            root: An E8 root

        Returns:
            True if Q(root) >= 0
        """
        v = root.to_float()  # export boundary
        return self._Q.is_in_cone(v, self._tolerance)

    def filter_roots(self, roots: List["E8Root"]) -> List["E8Root"]:
        """
        Filter a list of roots to only those in the cone.

        Args:
            roots: List of E8 roots

        Returns:
            Filtered list with only cone roots
        """
        return [r for r in roots if self.is_in_cone(r)]

    def cone_Q_values(self) -> List[Tuple["E8Root", float]]:
        """
        Get roots with their Q values.

        Returns:
            List of (root, Q(root)) tuples
        """
        return [(r, self._Q.evaluate_root(r)) for r in self._roots]

    def classify_roots(self) -> dict:
        """
        Classify all positive E8 roots by cone membership.

        Returns:
            Dictionary with 'in_cone' and 'out_of_cone' lists
        """
        in_cone = []
        out_of_cone = []
        for root in self._e8.positive_roots:
            if self.is_in_cone(root):
                in_cone.append(root)
            else:
                out_of_cone.append(root)
        return {"in_cone": in_cone, "out_of_cone": out_of_cone}

    def boundary_roots(self, tolerance: float = 1e-8) -> List["E8Root"]:  # eval boundary
        """
        Find roots on or near the cone boundary (Q approx 0).

        Args:
            tolerance: How close to 0 counts as boundary

        Returns:
            List of boundary roots
        """
        boundary = []
        for root in self._roots:
            Q = self._Q.evaluate_root(root)
            if abs(Q) < tolerance:
                boundary.append(root)
        return boundary

    def mean_Q(self) -> float:
        """Average Q value for cone roots."""
        if not self._roots:
            return 0
        return sum(self._Q.evaluate_root(r) for r in self._roots) / len(self._roots)

    def __len__(self) -> int:
        """Number of roots in cone."""
        return len(self._roots)

    def __iter__(self):
        """Iterate over cone roots."""
        return iter(self._roots)

    def __repr__(self) -> str:
        return f"GoldenCone(roots={len(self._roots)}, phi={self._phi:.6f})"


def golden_projector(phi: Optional[float] = None) -> GoldenProjector:
    """
    Create a GoldenProjector instance.

    Factory function for GoldenProjector.

    Args:
        phi: Golden ratio value. If None, uses exact PHI.

    Returns:
        GoldenProjector instance
    """
    return GoldenProjector(phi=phi)


def golden_cone(
    e8: Optional["E8Lattice"] = None,
    phi: Optional[float] = None,
) -> GoldenCone:
    """
    Create a GoldenCone instance.

    Factory function for GoldenCone.

    Args:
        e8: E8Lattice to filter. If None, creates new one.
        phi: Golden ratio value. If None, uses PHI_NUMERIC.

    Returns:
        GoldenCone instance

    Example:
        >>> cone = golden_cone()
        >>> len(cone.roots)
        36
    """
    return GoldenCone(e8_lattice=e8, phi=phi)
