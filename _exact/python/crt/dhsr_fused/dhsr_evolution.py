"""
Evolution and Trajectory tracking for CRT.

Provides:
- SyntonyTrajectory: Records state evolution through DHSR cycles
- DHSREvolver: Evolves states with full tracking and analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from applications.core.state import State
    from applications.crt.operators.gnosis import GnosisComputer
    from applications.crt.operators.recursion import RecursionOperator
    from applications.crt.operators.syntony import SyntonyComputer


@dataclass
class SyntonyTrajectory:
    """
    Records evolution of a state through DHSR cycles.

    Stores the complete history of states, syntony values,
    gnosis layers, and phase accumulation through iteration.

    Example:
        >>> evolver = DHSREvolver()
        >>> trajectory = evolver.evolve(initial_state, n_steps=100)
        >>> print(f"Final syntony: {trajectory.final_syntony}")
        >>> print(f"Converged: {trajectory.converged}")
    """

    states: List["State"] = field(default_factory=list)
    syntony_values: List[float] = field(default_factory=list)
    gnosis_values: List[int] = field(default_factory=list)
    phase_values: List[float] = field(default_factory=list)
    change_magnitudes: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def n_steps(self) -> int:
        """Number of evolution steps."""
        return max(0, len(self.states) - 1)

    @property
    def initial_state(self) -> Optional["State"]:
        """Get initial state."""
        return self.states[0] if self.states else None

    @property
    def final_state(self) -> Optional["State"]:
        """Get final state."""
        return self.states[-1] if self.states else None

    @property
    def initial_syntony(self) -> Optional[float]:
        """Get initial syntony."""
        return self.syntony_values[0] if self.syntony_values else None

    @property
    def final_syntony(self) -> Optional[float]:
        """Get final syntony."""
        return self.syntony_values[-1] if self.syntony_values else None

    @property
    def final_gnosis(self) -> Optional[int]:
        """Get final gnosis layer."""
        return self.gnosis_values[-1] if self.gnosis_values else None

    @property
    def final_phase(self) -> Optional[float]:
        """Get final phase."""
        return self.phase_values[-1] if self.phase_values else None

    @property
    def converged(self) -> bool:
        """
        Check if trajectory has converged.

        Convergence is detected when the change magnitude
        becomes very small.
        """
        if len(self.change_magnitudes) < 2:
            return False
        return self.change_magnitudes[-1] < 1e-6

    @property
    def syntony_delta(self) -> Optional[float]:
        """Change in syntony from start to end."""
        if self.initial_syntony is None or self.final_syntony is None:
            return None
        return self.final_syntony - self.initial_syntony

    @property
    def syntony_trend(self) -> str:
        """Describe syntony trend: 'increasing', 'decreasing', or 'stable'."""
        delta = self.syntony_delta
        if delta is None:
            return "unknown"
        if delta > 0.01:
            return "increasing"
        elif delta < -0.01:
            return "decreasing"
        else:
            return "stable"

    def convergence_rate(self) -> Optional[float]:
        """
        Estimate convergence rate from change magnitudes.

        Returns geometric decay rate if converging.
        """
        if len(self.change_magnitudes) < 3:
            return None

        # Use last few steps to estimate rate
        recent = self.change_magnitudes[-5:]
        if len(recent) < 2:
            return None

        # Compute average decay ratio
        ratios = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 1e-12:
                ratios.append(recent[i] / recent[i - 1])

        if not ratios:
            return None

        return sum(ratios) / len(ratios)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"SyntonyTrajectory: {self.n_steps} steps",
            (
                f"  Initial syntony: {self.initial_syntony:.4f}"
                if self.initial_syntony
                else ""
            ),
            f"  Final syntony: {self.final_syntony:.4f}" if self.final_syntony else "",
            f"  Trend: {self.syntony_trend}",
            f"  Converged: {self.converged}",
            (
                f"  Final gnosis: Layer {self.final_gnosis}"
                if self.final_gnosis is not None
                else ""
            ),
        ]
        return "\n".join(line for line in lines if line)


class DHSREvolver:
    """
    Evolves states through DHSR cycles with trajectory tracking.

    Provides high-level evolution interface with:
    - Full trajectory recording
    - Convergence detection
    - Attractor finding
    - Analysis tools

    Example:
        >>> evolver = DHSREvolver()
        >>> trajectory = evolver.evolve(state, n_steps=100)
        >>> attractor, traj = evolver.find_attractor(state)
    """

    def __init__(
        self,
        recursion_op: Optional["RecursionOperator"] = None,
        syntony_computer: Optional["SyntonyComputer"] = None,
        gnosis_computer: Optional["GnosisComputer"] = None,
    ):
        """
        Create a DHSR evolver.

        Args:
            recursion_op: Recursion operator (default: standard R̂)
            syntony_computer: Syntony computer (optional)
            gnosis_computer: Gnosis computer (default: standard)
        """
        self._recursion_op = recursion_op
        self._syntony_computer = syntony_computer
        self._gnosis_computer = gnosis_computer

    @property
    def recursion_op(self) -> "RecursionOperator":
        """Get recursion operator, creating default if needed."""
        if self._recursion_op is None:
            from applications.crt.operators.recursion import default_recursion_operator

            self._recursion_op = default_recursion_operator()
        return self._recursion_op

    @property
    def gnosis_computer(self) -> "GnosisComputer":
        """Get gnosis computer, creating default if needed."""
        if self._gnosis_computer is None:
            from applications.crt.operators.gnosis import default_gnosis_computer

            self._gnosis_computer = default_gnosis_computer()
        return self._gnosis_computer

    def evolve(
        self,
        initial_state: "State",
        n_steps: int = 100,
        early_stop: bool = True,
        tol: float = 1e-6,
    ) -> SyntonyTrajectory:
        """
        Evolve state through DHSR cycles.

        Args:
            initial_state: Starting state Ψ₀
            n_steps: Maximum number of evolution steps
            early_stop: Stop early if converged
            tol: Convergence tolerance

        Returns:
            SyntonyTrajectory recording evolution
        """
        trajectory = SyntonyTrajectory()
        current = initial_state

        # Record initial state
        trajectory.states.append(current)
        trajectory.syntony_values.append(current.syntony)
        trajectory.gnosis_values.append(self.gnosis_computer.compute_layer(current))
        trajectory.phase_values.append(0)

        accumulated_phase = 0

        for step in range(n_steps):
            # Apply R̂
            next_state = self.recursion_op.apply(current)

            # Compute change
            change = (next_state - current).norm()
            trajectory.change_magnitudes.append(change)

            # Accumulate phase based on syntony
            S = current.syntony
            delta_phase = math.pi * S * max(0, 1 - change / max(current.norm(), 1e-12))
            accumulated_phase += delta_phase

            # Record state
            trajectory.states.append(next_state)
            trajectory.syntony_values.append(next_state.syntony)
            trajectory.gnosis_values.append(
                self.gnosis_computer.compute_layer(next_state)
            )
            trajectory.phase_values.append(accumulated_phase)

            # Check convergence
            if early_stop and change < tol:
                break

            current = next_state

        return trajectory

    def find_attractor(
        self,
        initial_state: "State",
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> Tuple["State", SyntonyTrajectory]:
        """
        Find attractor state from initial condition.

        Evolves until convergence or max iterations.

        Args:
            initial_state: Starting state
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            (attractor_state, trajectory)
        """
        trajectory = self.evolve(
            initial_state,
            n_steps=max_iter,
            early_stop=True,
            tol=tol,
        )

        return trajectory.final_state, trajectory

    def analyze_stability(
        self,
        state: "State",
        perturbation_scale: float = 0.01,
        n_perturbations: int = 10,
        n_steps: int = 50,
    ) -> dict:
        """
        Analyze stability of a state under perturbations.

        Args:
            state: State to analyze
            perturbation_scale: Scale of random perturbations
            n_perturbations: Number of perturbations to try
            n_steps: Steps to evolve each perturbation

        Returns:
            dict with stability analysis
        """
        import random

        from applications.core.state import state as create_state

        base_syntony = state.syntony
        trajectories = []

        for _ in range(n_perturbations):
            # Create perturbed state
            flat = state.to_list()
            perturbed = [
                x + perturbation_scale * (2 * random.random() - 1) for x in flat
            ]
            perturbed_state = create_state(
                perturbed, dtype=state.dtype, shape=state.shape
            )

            # Evolve
            traj = self.evolve(perturbed_state, n_steps=n_steps)
            trajectories.append(traj)

        # Analyze results
        final_syntonies = [t.final_syntony for t in trajectories]
        mean_final = sum(final_syntonies) / len(final_syntonies)
        variance = sum((s - mean_final) ** 2 for s in final_syntonies) / len(
            final_syntonies
        )

        return {
            "base_syntony": base_syntony,
            "mean_final_syntony": mean_final,
            "syntony_variance": variance,
            "stable": variance < 0.01,
            "convergence_rate": sum(1 for t in trajectories if t.converged)
            / n_perturbations,
        }

    def find_all_attractors(
        self,
        initial_states: List["State"],
        tol: float = 1e-6,
        cluster_tol: float = 0.01,
    ) -> List[Tuple["State", int]]:
        """
        Find distinct attractors from multiple initial conditions.

        Args:
            initial_states: List of starting states
            tol: Convergence tolerance
            cluster_tol: Tolerance for clustering attractors

        Returns:
            List of (attractor, count) tuples
        """
        attractors = []

        for state in initial_states:
            attractor, _ = self.find_attractor(state, tol=tol)

            # Check if this attractor is new
            is_new = True
            for i, (existing, count) in enumerate(attractors):
                diff = (attractor - existing).norm()
                if diff < cluster_tol:
                    # Merge with existing
                    attractors[i] = (existing, count + 1)
                    is_new = False
                    break

            if is_new:
                attractors.append((attractor, 1))

        return attractors

    def __repr__(self) -> str:
        return "DHSREvolver()"


def default_evolver() -> DHSREvolver:
    """Create DHSR evolver with default settings."""
    return DHSREvolver()
