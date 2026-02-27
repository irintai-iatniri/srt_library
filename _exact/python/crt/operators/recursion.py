"""
Recursion Operator R̂ for CRT.

Implements the complete DHSR cycle:
R̂ = Ĥ ∘ D̂

The recursion operator combines differentiation and harmonization
into a single transformation that can be iterated to find attractors.

Key properties:
- R̂[Ψ] = Ĥ[D̂[Ψ]] (composition)
- Fixed points: R̂[Ψ*] = Ψ* represent stable configurations
- Iteration: R̂ⁿ[Ψ] converges to attractors
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from applications.crt.operators.base import OperatorBase
from applications.crt.operators.differentiation import (
    DifferentiationOperator,
    default_differentiation_operator,
)
from applications.crt.operators.harmonization import (
    HarmonizationOperator,
    default_harmonization_operator,
)

if TYPE_CHECKING:
    from applications.core.state import State


class RecursionOperator(OperatorBase):
    """
    Recursion Operator R̂ = Ĥ ∘ D̂.

    Performs one complete DHSR cycle by:
    1. Applying differentiation D̂
    2. Applying harmonization Ĥ to the result

    The recursion operator is the fundamental iteration step in CRT.
    Iterating R̂ drives the system toward attractor states where
    differentiation and harmonization are in balance.

    Example:
        >>> R_op = RecursionOperator()
        >>> evolved = R_op.apply(state)

        >>> # Or iterate to find attractor
        >>> attractor, n_iters, converged = R_op.find_fixed_point(state)
    """

    def __init__(
        self,
        diff_op: Optional[DifferentiationOperator] = None,
        harm_op: Optional[HarmonizationOperator] = None,
    ):
        """
        Create a recursion operator.

        Args:
            diff_op: Differentiation operator (default: standard D̂)
            harm_op: Harmonization operator (default: standard Ĥ)
        """
        self.diff_op = diff_op or default_differentiation_operator()
        self.harm_op = harm_op or default_harmonization_operator()

    def apply(self, state: "State", **kwargs) -> "State":
        """
        Apply recursion operator R̂ = Ĥ ∘ D̂.

        R̂[Ψ] = Ĥ[D̂[Ψ]]

        Args:
            state: Input state Ψ

        Returns:
            Recursed state R̂[Ψ]
        """
        # Apply D̂[Ψ]
        d_state = self.diff_op.apply(state)

        # Compute differentiation magnitude for adaptive harmonization
        delta_d = (d_state - state).norm()

        # Apply Ĥ[D̂[Ψ]] with differentiation info
        return self.harm_op.apply(d_state, delta_d=delta_d)

    def apply_with_info(self, state: "State") -> Tuple["State", dict]:
        """
        Apply R̂ and return detailed information.

        Returns:
            (result_state, info_dict) where info contains:
            - d_state: D̂[Ψ]
            - diff_magnitude: ||D̂[Ψ] - Ψ||
            - harm_magnitude: ||Ĥ[D̂[Ψ]] - D̂[Ψ]||
            - total_change: ||R̂[Ψ] - Ψ||
        """
        # Apply D̂
        d_state = self.diff_op.apply(state)
        diff_magnitude = (d_state - state).norm()

        # Apply Ĥ
        result = self.harm_op.apply(d_state, delta_d=diff_magnitude)
        harm_magnitude = (result - d_state).norm()
        total_change = (result - state).norm()

        info = {
            "d_state": d_state,
            "diff_magnitude": diff_magnitude,
            "harm_magnitude": harm_magnitude,
            "total_change": total_change,
        }

        return result, info

    def iterate(
        self,
        state: "State",
        n_steps: int = 10,
    ) -> List["State"]:
        """
        Iterate R̂ n times, returning trajectory.

        Args:
            state: Initial state Ψ₀
            n_steps: Number of iterations

        Returns:
            List [Ψ₀, R̂[Ψ₀], R̂²[Ψ₀], ..., R̂ⁿ[Ψ₀]]
        """
        trajectory = [state]
        current = state

        for _ in range(n_steps):
            current = self.apply(current)
            trajectory.append(current)

        return trajectory

    def find_fixed_point(
        self,
        state: "State",
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> Tuple["State", int, bool]:
        """
        Find fixed point R̂[Ψ*] = Ψ*.

        Iterates R̂ until convergence or max iterations.

        Args:
            state: Initial state Ψ₀
            tol: Convergence tolerance ||R̂[Ψ] - Ψ|| < tol
            max_iter: Maximum iterations

        Returns:
            (fixed_point, n_iterations, converged)
        """
        current = state

        for n in range(max_iter):
            next_state = self.apply(current)
            change = (next_state - current).norm()

            if change < tol:
                return next_state, n + 1, True

            current = next_state

        return current, max_iter, False

    def find_attractor_basin(
        self,
        state: "State",
        tol: float = 1e-6,
        max_iter: int = 1000,
        record_trajectory: bool = False,
    ) -> Tuple["State", dict]:
        """
        Find attractor and characterize the basin.

        Returns more detailed information about convergence.

        Args:
            state: Initial state
            tol: Convergence tolerance
            max_iter: Maximum iterations
            record_trajectory: Whether to store full trajectory

        Returns:
            (attractor, info) where info contains:
            - n_iterations: Number of iterations to converge
            - converged: Whether convergence was achieved
            - final_change: ||R̂[Ψ*] - Ψ*|| at termination
            - trajectory: List of states (if record_trajectory=True)
            - syntony_history: Syntony at each step
        """
        current = state
        trajectory = [state] if record_trajectory else []
        syntony_history = [current.syntony]

        for n in range(max_iter):
            next_state = self.apply(current)
            change = (next_state - current).norm()

            if record_trajectory:
                trajectory.append(next_state)
            syntony_history.append(next_state.syntony)

            if change < tol:
                return next_state, {
                    "n_iterations": n + 1,
                    "converged": True,
                    "final_change": change,
                    "trajectory": trajectory if record_trajectory else None,
                    "syntony_history": syntony_history,
                }

            current = next_state

        return current, {
            "n_iterations": max_iter,
            "converged": False,
            "final_change": (self.apply(current) - current).norm(),
            "trajectory": trajectory if record_trajectory else None,
            "syntony_history": syntony_history,
        }

    def period_finder(
        self,
        state: "State",
        max_period: int = 10,
        tol: float = 1e-6,
    ) -> Optional[int]:
        """
        Find periodic orbit period if state is on a limit cycle.

        Args:
            state: State to check
            max_period: Maximum period to search for
            tol: Tolerance for period detection

        Returns:
            Period if found, None otherwise
        """
        # Generate trajectory
        trajectory = self.iterate(state, n_steps=max_period + 1)

        # Check for periods 1, 2, ..., max_period
        for period in range(1, max_period + 1):
            if len(trajectory) > period:
                diff = (trajectory[-1] - trajectory[-(period + 1)]).norm()
                if diff < tol:
                    return period

        return None

    def __repr__(self) -> str:
        return f"RecursionOperator(D={self.diff_op}, H={self.harm_op})"


def default_recursion_operator() -> RecursionOperator:
    """
    Create a recursion operator with default SRT parameters.

    Returns:
        RecursionOperator with standard D̂ and Ĥ
    """
    return RecursionOperator()
