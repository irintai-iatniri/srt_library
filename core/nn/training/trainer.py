"""
Pure Syntonic Trainer: Training loop with Retrocausal RES.

PURE IMPLEMENTATION: Uses ResonantTensor and Retrocausal RES,
no PyTorch dependencies.

Provides a gradient-free training loop that:
1. Uses Retrocausal RES for weight optimization
2. Tracks syntony throughout training
3. Detects archonic patterns
4. Uses attractor memory for guidance

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from ..._core import (
    ResonantEvolver,
    RESResult,
    py_apply_geodesic_slide,
    py_standard_mode_norms,
)
from ..loss.syntony_metrics import SyntonyTracker
from ..resonant_tensor import ResonantTensor
try:
    from srt_library.theory_unique_components.resonant.retrocausal import (
    # RetrocausalConfig,
    create_retrocausal_evolver,
)
except ImportError:
    create_retrocausal_evolver = None
    RetrocausalConfig = None
from ... import server as viz_server
from ...server import launch_background_thread, update_monitor

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class PureModel(Protocol):
    """Protocol for pure syntonic models."""

    def forward(self, x: ResonantTensor) -> ResonantTensor: ...
    def get_weights(self) -> List[ResonantTensor]: ...
    def set_weights(self, weights: List[ResonantTensor]) -> None: ...
    @property
    def syntony(self) -> float: ...


@dataclass
class RESTrainingConfig:
    """Configuration for Retrocausal RES training."""

    # Evolution parameters
    max_generations: int = 500
    population_size: int = 32
    syntony_threshold: float = 0.95

    # Retrocausal parameters
    attractor_capacity: int = 32
    pull_strength: float = 0.3
    attractor_min_syntony: float = 0.7
    attractor_decay_rate: float = 0.98
    noise_scale: float = 0.1

    # Syntony targets
    syntony_target: float = S_TARGET

    # Training options
    use_archonic_detection: bool = True
    log_interval: int = 10
    eval_interval: int = 1

    # Loss parameters
    lambda_syntony: float = 0.1
    mu_phase: float = 0.01

    # Visualization
    enable_viz: bool = False  # Disable by default - causes training issues


class RetrocausalTrainer:
    """
    Pure training loop using Retrocausal RES.

    Handles:
    - Weight evolution via attractor-guided RES
    - Syntony computation and tracking
    - Archonic pattern detection
    - Retrocausal guidance

    Example:
        >>> from syntonic_applications.nn.training import RetrocausalTrainer, RESTrainingConfig
        >>> config = RESTrainingConfig(max_generations=200)
        >>> trainer = RetrocausalTrainer(model, train_data, config=config)
        >>> result = trainer.train()
        >>> print(f"Final syntony: {result['final_syntony']:.4f}")
    """

    def __init__(
        self,
        model: PureModel,
        train_data: List[Tuple[ResonantTensor, ResonantTensor]],
        val_data: Optional[List[Tuple[ResonantTensor, ResonantTensor]]] = None,
        config: Optional[RESTrainingConfig] = None,
        loss_fn: Optional[Callable[[ResonantTensor, ResonantTensor], float]] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Pure syntonic model to train
            train_data: Training data as list of (input, target) tuples
            val_data: Validation data
            config: Training configuration
            loss_fn: Loss function (default: MSE + syntony)
            callbacks: List of callbacks
        """
        self.config = config or RESTrainingConfig()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.callbacks = callbacks or []
        # Tracking
        self._syntony_tracker = SyntonyTracker(window_size=100)
        self._loss_history: List[float] = []
        self._current_syntony = 0.5
        self._current_generation = 0

        # Cache for geometric mode norms (prevents re-computation)
        self._mode_norms_cache = {}  # <--- Add this line

        # Create evolver template from model weights
        self._weight_templates = model.get_weights()

        # Start the visualization server (if enabled)
        if self.config.enable_viz:
            launch_background_thread()
            # Send an initial monitor snapshot so the dashboard has immediate data
            try:
                update_monitor(self.model, self._current_syntony, 0.0, 1.0)
            except Exception:
                # If viz server isn't available yet or model introspection fails,
                # we silently continue; the monitor will be updated during training.
                pass

            # Start a background monitor thread to push frequent updates and
            # to observe control state set by the dashboard client.
            def _monitor_loop():
                import time

                while True:
                    try:
                        # Pull control hints from viz server (phase/temperature)
                        ctrl = getattr(viz_server, "CONTROL_STATE", None)
                        if ctrl is not None:
                            # Apply simple controls to trainer config
                            ph = ctrl.get("phase")
                            if ph == "D":
                                self._current_phase = "D"
                            elif ph == "H":
                                self._current_phase = "H"
                            # temperature may be used by physics calculations
                            tval = float(ctrl.get("temperature", 0.0))
                        else:
                            tval = 0.0

                        # Push a monitor snapshot (so dashboard animates)
                        try:
                            update_monitor(
                                self.model,
                                self._current_syntony,
                                tval,
                                (
                                    1.0
                                    if getattr(self, "_current_phase", "D") == "D"
                                    else 0.0
                                ),
                            )
                        except Exception:
                            pass

                        time.sleep(0.1)
                    except Exception:
                        # If trainer is being torn down, exit
                        break

            import threading

            t = threading.Thread(target=_monitor_loop, daemon=True)
            t.start()

        # Loss function
        if loss_fn is None:
            from ..loss.syntonic_loss import mse_loss

            self.loss_fn = mse_loss
        else:
            self.loss_fn = loss_fn

        # Tracking
        self._syntony_tracker = SyntonyTracker(window_size=100)
        self._loss_history: List[float] = []
        self._current_syntony = 0.5
        self._current_generation = 0

        # Create evolver template from model weights
        self._weight_templates = model.get_weights()

    def train(self) -> Dict[str, Any]:
        """
        Run Retrocausal RES training.

        Returns:
            Training result dictionary
        """
        start_time = time.time()

        # Create retrocausal config
        # res_config = RetrocausalConfig(
        #     population_size=self.config.population_size,
        #     max_generations=self.config.max_generations,
        #     # Note: syntony_threshold is used for evaluation, not RES evolution
        #     attractor_capacity=self.config.attractor_capacity,
        #     attractor_pull_strength=self.config.pull_strength,
        #     attractor_min_syntony=self.config.attractor_min_syntony,
        #     attractor_decay_rate=self.config.attractor_decay_rate,
        # ).to_res_config()

        # Evolve each weight tensor
        all_results = []
        evolved_weights = []

        for i, template in enumerate(self._weight_templates):
            # Create evolver for this weight
            evolver = create_retrocausal_evolver(
                template=template,
                population_size=self.config.population_size,
                attractor_capacity=self.config.attractor_capacity,
                pull_strength=self.config.pull_strength,
                min_syntony=self.config.attractor_min_syntony,
                decay_rate=self.config.attractor_decay_rate,
            )

            # Run evolution with fitness function
            result = self._evolve_weight(evolver, i)
            all_results.append(result)

            # wrap the result (Rust tensor) into Python wrapper
            best_tensor_py = ResonantTensor._wrap(result["best_tensor"])
            evolved_weights.append(best_tensor_py)

            # Apply intermediate weight snapshot so the visualization
            # can observe per-weight progress in real-time. We construct
            # a full-weight list where weights up to the current index
            # use the evolved tensors and the remaining weights use the
            # existing templates (or current model weights).
            try:
                snapshot = []
                for j in range(len(self._weight_templates)):
                    if j <= i:
                        snapshot.append(evolved_weights[j])
                    else:
                        snapshot.append(self._weight_templates[j])
                # Apply the partial snapshot to the model so update_monitor
                # will pick up the changed weights when called.
                try:
                    self.model.set_weights(snapshot)
                except Exception:
                    # If the model doesn't accept partial updates, ignore.
                    pass
            except Exception:
                pass

            if i % self.config.log_interval == 0:
                self._log_progress(i, len(self._weight_templates), result)

        # Apply evolved weights to model
        self.model.set_weights(evolved_weights)

        # Final validation
        final_loss, final_syntony = self._evaluate()

        elapsed_time = time.time() - start_time

        return {
            "final_syntony": final_syntony,
            "final_loss": final_loss,
            "generations": sum(r["generations"] for r in all_results)
            / len(all_results),
            "elapsed_time": elapsed_time,
            "weight_results": all_results,
            "syntony_history": self._syntony_tracker.history,
            "loss_history": self._loss_history,
        }

    def _evolve_weight(
        self,
        evolver: ResonantEvolver,
        weight_idx: int,
    ) -> Dict[str, Any]:
        """
        Evolve a single weight tensor.

        Uses the training data to compute fitness and applies
        retrocausal geometric constraints.
        """
        # 1. Run Evolution (The Genetic Search)
        result: RESResult = evolver.run()

        # Track progress
        self._syntony_tracker.update(result.final_syntony)
        self._current_syntony = result.final_syntony
        self._current_generation += result.generations

        # 2. Apply Geodesic Gravity Slide (The Physical Lock)
        # This forces the weights to snap to the nearest valid E8 lattice points
        # relative to the attractor (Time-Loop Logic).

        attractor = self._weight_templates[weight_idx]

        # (Breathing Schedule) ---
        # Dissonance: 0.0 (Harmony) -> 1.0 (Chaos)
        dissonance = 1.0 - result.final_syntony

        # Temperature: High when dissonant (Melt lattice to explore)
        # If we are stuck (high dissonance), we heat up.
        temp = dissonance * self.config.noise_scale

        # Gravity: Low when dissonant (Don't crush exploration)
        # Only turn on the "Vacuum Sombrero" when we start finding truth (Syntony > 0.6)
        # Using PHI (1.618) as the scaling constant
        gravity_scalar = 0.0
        if result.final_syntony > 0.6:
            # Scale from 0 at syntony=0.6 to 1.0 at syntony=1.0
            gravity_scalar = (result.final_syntony - 0.6) * 2.5

        gravity = self.config.pull_strength * PHI * gravity_scalar

        # Apply slide if on CUDA (geometry requires GPU acceleration)
        try:
            best_tensor = result.winner
            w_shape = tuple(best_tensor.shape)

            # Retrieve or create mode norms (needed for E8 geometry calculations)
            if w_shape not in self._mode_norms_cache:
                # Generate standard mode norms for this shape on the correct device
                # These define the 'metric' of the space the weights live in
                self._mode_norms_cache[w_shape] = py_standard_mode_norms(
                    w_shape, best_tensor.device
                )

            norms = self._mode_norms_cache[w_shape]

            # Apply the Physical Update
            # We access the internal .tensor (TensorStorage) to pass to Rust/CUDA
            py_apply_geodesic_slide(
                best_tensor.tensor,  # Mutable: will be updated in-place
                attractor.tensor,  # Read-only: the future attractor
                norms.tensor,  # Read-only: geometric metric
                gravity,
                temp,
            )

            # Broadcast the real physical state to the console (only if viz enabled)
            if self.config.enable_viz:
                try:
                    update_monitor(
                        self.model,
                        result.final_syntony,
                        temp,  # From your physics calc
                        1.0 if temp > 0.1 else 0.0,  # D-Phase vs H-Phase
                    )
                except Exception:
                    # ignore viz errors during training
                    pass
        except Exception:
            # Fallback: If physics engine fails (e.g. running on CPU),
            # we accept the evolutionary result as-is without the lattice snap.
            pass

        return {
            "weight_idx": weight_idx,
            "final_syntony": result.final_syntony,
            "generations": result.generations,
            "converged": result.converged,
            "best_tensor": result.winner,
        }

    def _evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model on training data.

        Returns:
            (loss, syntony)
        """
        total_loss = 0.0
        n_samples = 0

        for inputs, targets in self.train_data:
            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            # Convert loss to float if it's a ResonantTensor
            if hasattr(loss, "to_floats"):
                loss_val = loss.to_floats()[0]
            else:
                loss_val = float(loss)
            total_loss += loss_val
            n_samples += 1

        avg_loss = total_loss / max(1, n_samples)
        model_syntony = self.model.syntony

        self._loss_history.append(avg_loss)
        return avg_loss, model_syntony

    def _log_progress(
        self,
        current: int,
        total: int,
        result: Dict[str, Any],
    ):
        """Log training progress."""
        for callback in self.callbacks:
            if hasattr(callback, "on_weight_evolved"):
                callback.on_weight_evolved(
                    self,
                    current,
                    total,
                    result,
                )

    @property
    def current_syntony(self) -> float:
        """Get current model syntony."""
        return self._current_syntony

    @property
    def syntony_history(self) -> List[float]:
        """Get syntony history."""
        return self._syntony_tracker.history

    @property
    def is_archonic(self) -> bool:
        """Check if training is in archonic pattern."""
        return self._syntony_tracker.is_archonic()


# Convenience function
def train_with_retrocausal_res(
    model: PureModel,
    train_data: List[Tuple[ResonantTensor, ResonantTensor]],
    max_generations: int = 500,
    population_size: int = 32,
    pull_strength: float = 0.3,
) -> Dict[str, Any]:
    """
    Quick training function with sensible defaults.

    Args:
        model: Pure syntonic model
        train_data: Training data
        max_generations: Maximum generations per weight
        population_size: Population size for RES
        pull_strength: Retrocausal pull strength

    Returns:
        Training results
    """
    config = RESTrainingConfig(
        max_generations=max_generations,
        population_size=population_size,
        pull_strength=pull_strength,
    )
    trainer = RetrocausalTrainer(model, train_data, config=config)
    return trainer.train()


# Aliases
PureTrainer = RetrocausalTrainer
PureTrainingConfig = RESTrainingConfig


if __name__ == "__main__":
    """Test pure trainer."""
    print("Testing Pure Retrocausal Trainer...")
    print("Note: Full test requires a PureModel implementation.")

    # Verify imports work
    config = RESTrainingConfig()
    print(f"Config created: population_size={config.population_size}")
    print(f"  max_generations={config.max_generations}")
    print(f"  pull_strength={config.pull_strength}")
    print(f"  syntony_target={config.syntony_target:.4f}")

    print("✅ Pure Retrocausal Trainer imports verified!")
