"""
Correction Factors - SRT correction factors (1 +/- q/N).

These corrections arise from heat kernel regularization on lattices and
appear in all physical predictions. The universal syntony deficit q appears
in combinations like (1 + q/N) or (1 - q/N) where N is the dimension of
the relevant algebraic structure.

Example:
    >>> from syntonic_applications.srt.corrections import correction_factors
    >>> cf = correction_factors()
    >>> cf.plus('E8')   # 1 + q/248
    >>> cf.minus('D4')  # 1 - q/24
"""

from typing import List, Optional

from applications._core import Q_DEFICIT_NUMERIC, STRUCTURE_DIMENSIONS, Rational


class CorrectionFactors:
    """
    SRT correction factors (1 +/- q/N) for various algebraic structures.

    The universal syntony deficit q = 20 - E* where E* = e^pi - pi appears
    in physical predictions multiplied by structure-dependent factors.

    Attributes:
        q: The universal syntony deficit (default: Q_DEFICIT_NUMERIC)
        structures: Dictionary mapping structure names to dimensions

    Example:
        >>> cf = CorrectionFactors()
        >>> cf.plus('E8')   # Returns 1 + q/248
        1.0000036...
        >>> cf.minus('D4')  # Returns 1 - q/24
        0.9999625...
    """

    def __init__(self, q_deficit: Optional[float] = None):
        """
        Initialize correction factor calculator.

        Args:
            q_deficit: Universal syntony deficit. If None, uses Q_DEFICIT_NUMERIC.
        """
        self._q = q_deficit if q_deficit is not None else Q_DEFICIT_NUMERIC
        self._structures = dict(STRUCTURE_DIMENSIONS)

    @property
    def q(self) -> float:
        """The universal syntony deficit."""
        return self._q

    @property
    def available_structures(self) -> List[str]:
        """List of available structure names."""
        return list(self._structures.keys())

    def get_dimension(self, structure: str) -> int:
        """
        Get dimension N for a structure name.

        Args:
            structure: Name of the algebraic structure (e.g., 'E8', 'D4')

        Returns:
            The dimension of the structure

        Raises:
            KeyError: If structure is not recognized
        """
        if structure not in self._structures:
            raise KeyError(
                f"Unknown structure '{structure}'. "
                f"Available: {self.available_structures}"
            )
        return self._structures[structure]

    def plus(self, structure: str) -> float:
        """
        Compute (1 + q/N) for given structure.

        This correction enhances the contribution from the structure.

        Args:
            structure: Name of the algebraic structure

        Returns:
            1 + q/N where N is the structure dimension
        """
        N = self.get_dimension(structure)
        return 1.0 + self._q / N

    def minus(self, structure: str) -> float:
        """
        Compute (1 - q/N) for given structure.

        This correction suppresses the contribution from the structure.

        Args:
            structure: Name of the algebraic structure

        Returns:
            1 - q/N where N is the structure dimension
        """
        N = self.get_dimension(structure)
        return 1.0 - self._q / N

    def factor(self, structure: str, sign: int = 1) -> float:
        """
        Compute (1 + sign*q/N) for given structure.

        Args:
            structure: Name of the algebraic structure
            sign: +1 for enhancement, -1 for suppression

        Returns:
            1 + sign*q/N
        """
        if sign == 1:
            return self.plus(structure)
        elif sign == -1:
            return self.minus(structure)
        else:
            N = self.get_dimension(structure)
            return 1.0 + sign * self._q / N

    def compound(
        self,
        plus_structures: List[str],
        minus_structures: List[str],
    ) -> float:
        """
        Compute product of (1 + q/N_i) * (1 - q/N_j) factors.

        Many SRT predictions involve products of multiple correction factors.

        Args:
            plus_structures: List of structures for (1 + q/N) factors
            minus_structures: List of structures for (1 - q/N) factors

        Returns:
            Product of all correction factors

        Example:
            >>> cf = CorrectionFactors()
            >>> cf.compound(['E8'], ['D4', 'SU3'])
            # Returns (1 + q/248) * (1 - q/24) * (1 - q/8)
        """
        result = Rational(1, 1)
        for s in plus_structures:
            result *= self.plus(s)
        for s in minus_structures:
            result *= self.minus(s)
        return result

    def ratio(self, numerator: str, denominator: str) -> float:
        """
        Compute (1 + q/N_num) / (1 + q/N_den).

        Args:
            numerator: Structure for numerator factor
            denominator: Structure for denominator factor

        Returns:
            Ratio of the two (1 + q/N) factors
        """
        return self.plus(numerator) / self.plus(denominator)

    def add_structure(self, name: str, dimension: int) -> None:
        """
        Add a custom structure.

        Args:
            name: Name for the structure
            dimension: Dimension N of the structure
        """
        self._structures[name] = dimension

    def __repr__(self) -> str:
        return f"CorrectionFactors(q={self._q:.6e})"


def correction_factors(q: Optional[float] = None) -> CorrectionFactors:
    """
    Create a CorrectionFactors calculator.

    Factory function for creating CorrectionFactors instances.

    Args:
        q: Universal syntony deficit. If None, uses Q_DEFICIT_NUMERIC.

    Returns:
        CorrectionFactors instance

    Example:
        >>> cf = correction_factors()
        >>> cf.plus('E8')
        1.0000036...
    """
    return CorrectionFactors(q_deficit=q)
