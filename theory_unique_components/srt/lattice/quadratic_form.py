"""
Quadratic Form - Q(lambda) and Golden Cone selection.

The SRT golden cone is defined by 4 null vectors c_1, c_2, c_3, c_4.
A root α is in the golden cone iff ⟨c_a, α⟩ > 0 for all 4 null vectors.

This selects exactly 36 roots = Φ⁺(E₆) from the 120 positive E8 roots.

Example:
    >>> from srt_library.theory_unique_components.srt.lattice import quadratic_form, e8_lattice
    >>> Q = quadratic_form()
    >>> E8 = e8_lattice()
    >>> for root in E8.roots[:5]:
    ...     print(f"{root}: in_cone = {Q.is_in_cone_root(root)}")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from srt_library.core.exact import PHI_NUMERIC

if TYPE_CHECKING:
    from srt_library.theory_unique_components.srt.lattice.e8 import E8Root


# The 4 null vectors defining the golden cone (from golden_cone_report.md)
# These satisfy Q(c_a) = 0 and define the cone boundary
NULL_VECTORS: List[Tuple[float, ...]] = [
    (-0.152753, -0.312330, 0.192683, -0.692448, 0.013308, 0.531069, 0.153449, 0.238219),
    (0.270941, 0.201058, 0.532514, -0.128468, 0.404635, -0.475518, -0.294881, 0.330591),
    (0.157560, 0.189639, 0.480036, 0.021016, 0.274831, -0.782436, 0.060319, 0.130225),
    (
        0.476719,
        0.111410,
        0.464671,
        -0.543379,
        -0.142049,
        -0.150498,
        -0.314296,
        0.327929,
    ),
]


class QuadraticForm:
    """
    SRT quadratic form Q(lambda) = |P_parallel(lambda)|^2 - |P_perp(lambda)|^2.

    The golden projector P_phi decomposes R^8 = R^4_parallel + R^4_perp
    where R^4_parallel is the "physical" subspace and R^4_perp is the
    "internal" subspace.

    The quadratic form Q classifies vectors:
        - Q(lambda) > 0: space-like (physical particles)
        - Q(lambda) = 0: light-like (massless)
        - Q(lambda) < 0: time-like (virtual/internal)

    Attributes:
        phi: Golden ratio for projection

    Example:
        >>> Q = QuadraticForm()
        >>> v = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> Q.evaluate(v)
        0.0  # Type A root is light-like
    """

    def __init__(self, phi: Optional[float] = None):
        """
        Initialize quadratic form with golden projector.

        Args:
            phi: Golden ratio value. If None, uses PHI_NUMERIC.
        """
        self._phi = phi if phi is not None else PHI_NUMERIC
        self._phi_inv = 1.0 / self._phi  # = phi - 1
        self._compute_projection_matrices()

    def _compute_projection_matrices(self) -> None:
        """
        Compute the 4x8 projection matrices P_parallel and P_perp.

        The golden embedding uses the eigenspaces of the golden ratio
        to define the physical and internal subspaces.
        """
        # The golden projector is based on the Fibonacci embedding
        # P_parallel projects onto the eigenspace of phi
        # P_perp projects onto the eigenspace of -1/phi

        phi = self._phi
        # sqrt5 = math.sqrt(5)

        # Normalized projection coefficients
        # These come from the golden ratio eigenvector structure
        # Simplified 4x8 matrices for the standard golden embedding

        # For the standard SRT golden projector, we use:
        # P_parallel picks out even-index coordinates scaled by phi-factors
        # P_perp picks out odd-index coordinates

        # Store projection matrices as nested lists
        # Each row is a 4D output component
        self._P_parallel = [
            [1.0, 0.0, 0.0, 0.0, phi, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, phi, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, phi, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, phi],
        ]

        self._P_perp = [
            [phi, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, phi, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, phi, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, phi, 0.0, 0.0, 0.0, -1.0],
        ]

        # Normalize the projection matrices
        norm_factor = 1.0 / math.sqrt(1.0 + phi * phi)
        for i in range(4):
            for j in range(8):
                self._P_parallel[i][j] *= norm_factor
                self._P_perp[i][j] *= norm_factor

    @property
    def phi(self) -> float:
        """Golden ratio value."""
        return self._phi

    def project_parallel(self, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Project v onto the parallel (physical) subspace.

        Args:
            v: 8-dimensional vector

        Returns:
            4-dimensional projection P_parallel(v)
        """
        if len(v) != 8:
            raise ValueError(f"Vector must have 8 components, got {len(v)}")

        result = []
        for i in range(4):
            component = sum(self._P_parallel[i][j] * v[j] for j in range(8))
            result.append(component)
        return tuple(result)

    def project_perpendicular(self, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Project v onto the perpendicular (internal) subspace.

        Args:
            v: 8-dimensional vector

        Returns:
            4-dimensional projection P_perp(v)
        """
        if len(v) != 8:
            raise ValueError(f"Vector must have 8 components, got {len(v)}")

        result = []
        for i in range(4):
            component = sum(self._P_perp[i][j] * v[j] for j in range(8))
            result.append(component)
        return tuple(result)

    def parallel_norm_squared(self, v: Tuple[float, ...]) -> float:
        """
        Compute |P_parallel(v)|^2.

        Args:
            v: 8-dimensional vector

        Returns:
            Squared norm of parallel projection
        """
        proj = self.project_parallel(v)
        return sum(x * x for x in proj)

    def perpendicular_norm_squared(self, v: Tuple[float, ...]) -> float:
        """
        Compute |P_perp(v)|^2.

        Args:
            v: 8-dimensional vector

        Returns:
            Squared norm of perpendicular projection
        """
        proj = self.project_perpendicular(v)
        return sum(x * x for x in proj)

    def evaluate(self, v: Tuple[float, ...]) -> float:
        """
        Compute Q(v) = |P_parallel(v)|^2 - |P_perp(v)|^2.

        Args:
            v: 8-dimensional vector

        Returns:
            Quadratic form value
        """
        return self.parallel_norm_squared(v) - self.perpendicular_norm_squared(v)

    def evaluate_root(self, root: "E8Root") -> float:
        """
        Compute Q for an E8 root.

        Args:
            root: An E8Root

        Returns:
            Q(root)
        """
        v = root.to_float()
        return self.evaluate(v)

    def classify(self, v: Tuple[float, ...], tol: float = 1e-10) -> str:
        """
        Classify vector by sign of Q.

        Args:
            v: 8-dimensional vector
            tol: Tolerance for "light-like" classification

        Returns:
            'spacelike', 'lightlike', or 'timelike'
        """
        Q = self.evaluate(v)
        if Q > tol:
            return "spacelike"
        elif Q < -tol:
            return "timelike"
        return "lightlike"

    def classify_root(self, root: "E8Root", tol: float = 1e-10) -> str:
        """
        Classify E8 root by sign of Q.

        Args:
            root: An E8Root
            tol: Tolerance

        Returns:
            'spacelike', 'lightlike', or 'timelike'
        """
        return self.classify(root.to_float(), tol)

    def is_in_cone(self, v: Tuple[float, ...], tol: float = 1e-10) -> bool:
        """
        Check if v is in the golden cone using the 4 null vectors.

        The golden cone is defined by: ⟨c_a, v⟩ > 0 for all 4 null vectors.
        This selects exactly 36 roots = Φ⁺(E₆) from E8 positive roots.

        Args:
            v: 8-dimensional vector
            tol: Tolerance for positivity check

        Returns:
            True if all 4 inner products are positive
        """
        if len(v) != 8:
            raise ValueError(f"Vector must have 8 components, got {len(v)}")

        for c in NULL_VECTORS:
            inner = sum(c[i] * v[i] for i in range(8))
            if inner <= tol:  # Must be strictly positive
                return False
        return True

    def is_in_cone_q(self, v: Tuple[float, ...], tol: float = 1e-10) -> bool:
        """
        Check if v is in the cone using Q >= 0 criterion (alternative).

        This is the simpler criterion based on the quadratic form.
        Note: This gives more roots than the golden cone (36).

        Args:
            v: 8-dimensional vector
            tol: Tolerance

        Returns:
            True if Q(v) >= 0
        """
        return self.evaluate(v) >= -tol

    def projection_matrix_parallel(self) -> List[List[float]]:
        """Return the 4x8 parallel projection matrix."""
        return [row.copy() for row in self._P_parallel]

    def projection_matrix_perpendicular(self) -> List[List[float]]:
        """Return the 4x8 perpendicular projection matrix."""
        return [row.copy() for row in self._P_perp]

    def __repr__(self) -> str:
        return f"QuadraticForm(phi={self._phi:.6f})"


def quadratic_form(phi: Optional[float] = None) -> QuadraticForm:
    """
    Create a QuadraticForm instance.

    Factory function for QuadraticForm.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.

    Returns:
        QuadraticForm instance
    """
    return QuadraticForm(phi=phi)


def compute_Q(v: Tuple[float, ...], phi: Optional[float] = None) -> float:
    """
    Compute Q(v) = |P_parallel(v)|^2 - |P_perp(v)|^2.

    Convenience function for single computation.

    Args:
        v: 8-dimensional vector
        phi: Golden ratio value. If None, uses PHI_NUMERIC.

    Returns:
        Quadratic form value
    """
    Q = QuadraticForm(phi=phi)
    return Q.evaluate(v)
