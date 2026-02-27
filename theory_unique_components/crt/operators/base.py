"""
Abstract base class for CRT operators.

All DHSR operators (Differentiation, Harmonization, Recursion) inherit from
OperatorBase to ensure a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srt_library.core.state import State


class OperatorBase(ABC):
    """
    Abstract base class for DHSR operators.

    All CRT operators must implement:
    - apply(): The core operator application
    - __call__(): Convenience wrapper around apply()

    Example:
        >>> D_op = DifferentiationOperator(alpha_0=0.1)
        >>> evolved = D_op(state)  # Same as D_op.apply(state)
    """

    @abstractmethod
    def apply(self, state: "State", **kwargs) -> "State":
        """
        Apply the operator to a state.

        Args:
            state: Input state Ψ
            **kwargs: Operator-specific parameters

        Returns:
            Transformed state
        """
        pass

    def __call__(self, state: "State", **kwargs) -> "State":
        """
        Convenience method to apply operator.

        Equivalent to calling apply(state, **kwargs).
        """
        return self.apply(state, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
