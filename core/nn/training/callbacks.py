"""
Pure Training Callbacks for Syntonic Networks.

Callbacks for monitoring syntony, detecting archonic patterns,
and controlling training flow. Pure Python - no PyTorch dependencies.

Source: CRT.md §12.2
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..training.trainer import RetrocausalTrainer

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "RetrocausalTrainer"):
        """Called at start of training."""
        pass

    def on_train_end(self, trainer: "RetrocausalTrainer"):
        """Called at end of training."""
        pass

    def on_generation_begin(self, trainer: "RetrocausalTrainer", generation: int):
        """Called at start of each generation (replaces epoch)."""
        pass

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Called at end of each generation."""
        pass


class SyntonyCallback(Callback):
    """
    Monitor and log syntony during training.

    Tracks syntony evolution and provides statistics.

    Example:
        >>> callback = SyntonyCallback(log_interval=10)
        >>> trainer = RetrocausalTrainer(model, config, callbacks=[callback])
    """

    def __init__(
        self,
        log_interval: int = 10,
        target_syntony: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Initialize syntony callback.

        Args:
            log_interval: Generations between logs
            target_syntony: Target syntony (default: φ - q)
            verbose: Print updates
        """
        self.log_interval = log_interval
        self.target_syntony = target_syntony if target_syntony is not None else S_TARGET
        self.verbose = verbose

        self.history: List[Dict[str, float]] = []
        self._best_syntony = 0.0
        self._best_generation = 0

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Record syntony at end of generation."""
        syntony = metrics.get("syntony", 0.0)

        record = {
            "generation": generation,
            "syntony": syntony,
            "fitness": metrics.get("fitness", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
        }
        self.history.append(record)

        # Track best
        if syntony > self._best_syntony:
            self._best_syntony = syntony
            self._best_generation = generation

        # Log
        if self.verbose and (generation + 1) % self.log_interval == 0:
            gap = self.target_syntony - syntony
            status = "↑" if syntony > self.target_syntony - 0.1 else "↓"
            print(
                f"Gen {generation+1}: S={syntony:.4f} (target gap: {gap:.4f}) {status}"
            )

    def on_train_end(self, trainer: "RetrocausalTrainer"):
        """Print summary at end of training."""
        if self.verbose and self.history:
            final = self.history[-1]
            print("\nTraining complete:")
            print(f"  Final syntony: {final['syntony']:.4f}")
            print(
                f"  Best syntony: {self._best_syntony:.4f} (gen {self._best_generation + 1})"
            )
            print(f"  Target: {self.target_syntony:.4f}")

    @property
    def best_syntony(self) -> float:
        """Get best syntony achieved."""
        return self._best_syntony

    @property
    def syntony_trend(self) -> float:
        """Get recent syntony trend."""
        if len(self.history) < 10:
            return 0.0

        recent = [h["syntony"] for h in self.history[-10:]]
        mid = len(recent) // 2
        return sum(recent[mid:]) / len(recent[mid:]) - sum(recent[:mid]) / len(
            recent[:mid]
        )


class ArchonicEarlyStop(Callback):
    """
    Stop training when archonic pattern is detected.

    Detects when the network is cycling without syntony improvement
    and triggers early stopping or escape mechanisms.

    Example:
        >>> callback = ArchonicEarlyStop(patience=20)
        >>> trainer = RetrocausalTrainer(model, config, callbacks=[callback])
    """

    def __init__(
        self,
        patience: int = 20,
        variance_threshold: float = 0.01,
        min_improvement: float = 0.001,
        escape_noise_scale: float = 0.01,
        escape_instead: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize archonic early stop.

        Args:
            patience: Generations of no improvement before stopping
            variance_threshold: Threshold for detecting cycling
            min_improvement: Minimum syntony improvement required
            escape_noise_scale: Noise scale for escape mechanism
            escape_instead: Inject noise instead of stopping
            verbose: Print warnings
        """
        self.patience = patience
        self.variance_threshold = variance_threshold
        self.min_improvement = min_improvement
        self.escape_noise_scale = escape_noise_scale
        self.escape_instead = escape_instead
        self.verbose = verbose

        self._syntony_history: List[float] = []
        self._no_improvement_count = 0
        self._best_syntony = 0.0
        self._should_stop = False
        self._archonic_count = 0

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Check for archonic pattern at end of generation."""
        syntony = metrics.get("syntony", 0.0)
        self._syntony_history.append(syntony)

        # Check improvement
        if syntony > self._best_syntony + self.min_improvement:
            self._best_syntony = syntony
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        # Check for archonic pattern
        is_archonic = self._detect_archonic()

        if is_archonic:
            self._archonic_count += 1

            if self.verbose:
                print(f"⚠ Archonic pattern detected at generation {generation + 1}")

            if self.escape_instead:
                self._trigger_escape(trainer)
            elif self._no_improvement_count >= self.patience:
                self._should_stop = True
                if self.verbose:
                    print(
                        f"⛔ Early stopping: archonic pattern for {self.patience} gens"
                    )

    def _detect_archonic(self) -> bool:
        """Detect archonic cycling pattern."""
        if len(self._syntony_history) < 20:
            return False

        recent = self._syntony_history[-20:]
        mean_S = sum(recent) / len(recent)
        var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

        # Check for high variance (cycling)
        if var_S < self.variance_threshold:
            return False

        # Check for no trend (stuck)
        mid = len(recent) // 2
        trend = sum(recent[mid:]) / len(recent[mid:]) - sum(recent[:mid]) / len(
            recent[:mid]
        )

        return abs(trend) < self.min_improvement

    def _trigger_escape(self, trainer: "RetrocausalTrainer"):
        """Inject noise to escape archonic pattern (pure Python version)."""
        # import random

        if hasattr(trainer, "weights") and trainer.weights is not None:
            # Get weight data and add noise
            # data = trainer.weights.to_list()
            # noise_scale = self.escape_noise_scale
            # noisy = [v + random.gauss(0, noise_scale) for v in data]
            # Update via evolver if available
            if hasattr(trainer, "evolver") and trainer.evolver is not None:
                # The evolver will handle the next mutation
                pass

    @property
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop

    @property
    def archonic_count(self) -> int:
        """Number of archonic detections."""
        return self._archonic_count


class SyntonyCheckpoint(Callback):
    """
    Save checkpoints based on syntony.

    Saves model weights as JSON when syntony improves or at regular intervals.

    Example:
        >>> callback = SyntonyCheckpoint('checkpoints/', save_best=True)
        >>> trainer = RetrocausalTrainer(model, config, callbacks=[callback])
    """

    def __init__(
        self,
        save_dir: str,
        save_best: bool = True,
        save_interval: int = 10,
        min_syntony: float = 0.0,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory for checkpoints
            save_best: Save on syntony improvement
            save_interval: Generations between regular saves
            min_syntony: Minimum syntony to trigger save
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_interval = save_interval
        self.min_syntony = min_syntony

        self._best_syntony = 0.0

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Save checkpoint if criteria met."""
        syntony = metrics.get("syntony", 0.0)

        # Save best
        if (
            self.save_best
            and syntony > self._best_syntony
            and syntony > self.min_syntony
        ):
            self._best_syntony = syntony
            self._save_checkpoint(trainer, generation, syntony, "best_model.json")

        # Regular interval save
        if (generation + 1) % self.save_interval == 0:
            self._save_checkpoint(
                trainer, generation, syntony, f"checkpoint_gen_{generation+1}.json"
            )

    def _save_checkpoint(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        syntony: float,
        filename: str,
    ):
        """Save weights to JSON file."""
        path = self.save_dir / filename

        # Get weights as list
        weights_data = []
        if hasattr(trainer, "weights") and trainer.weights is not None:
            weights_data = trainer.weights.to_list()

        checkpoint = {
            "generation": generation,
            "syntony": syntony,
            "fitness": getattr(trainer, "_best_fitness", 0.0),
            "weights": weights_data,
            "shape": (
                list(trainer.weights.shape()) if hasattr(trainer, "weights") else []
            ),
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)


class MetricsLogger(Callback):
    """
    Log training metrics to file.

    Saves detailed metrics in JSON format for analysis.

    Example:
        >>> callback = MetricsLogger('logs/training_metrics.json')
        >>> trainer = RetrocausalTrainer(model, config, callbacks=[callback])
    """

    def __init__(
        self,
        log_path: str,
        include_weights: bool = False,
    ):
        """
        Initialize metrics logger.

        Args:
            log_path: Path to log file
            include_weights: Log weight statistics
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.include_weights = include_weights

        self.logs: List[Dict[str, Any]] = []

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Log metrics at end of generation."""
        log_entry = {
            "generation": generation,
            "metrics": dict(metrics),
        }

        # Weight statistics
        if self.include_weights and hasattr(trainer, "weights"):
            weights = trainer.weights.to_list()
            if weights:
                mean_w = sum(weights) / len(weights)
                var_w = sum((w - mean_w) ** 2 for w in weights) / len(weights)
                max_w = max(abs(w) for w in weights)
                log_entry["weight_norm_mean"] = math.sqrt(var_w)
                log_entry["weight_max"] = max_w

        self.logs.append(log_entry)

    def on_train_end(self, trainer: "RetrocausalTrainer"):
        """Save logs to file."""
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=2)


class FitnessPlateauCallback(Callback):
    """
    Detect fitness plateau and adjust evolution parameters.

    When fitness stops improving, increases mutation rate or
    population diversity to escape local optima.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mutation_boost: float = 2.0,
        verbose: bool = True,
    ):
        """
        Initialize plateau callback.

        Args:
            patience: Generations without improvement
            min_delta: Minimum fitness improvement
            mutation_boost: Factor to boost mutation rate
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mutation_boost = mutation_boost
        self.verbose = verbose

        self._best_fitness = float("-inf")
        self._no_improvement = 0
        self._boosts_applied = 0

    def on_generation_end(
        self,
        trainer: "RetrocausalTrainer",
        generation: int,
        metrics: Dict[str, float],
    ):
        """Check for plateau and boost if needed."""
        fitness = metrics.get("fitness", 0.0)

        if fitness > self._best_fitness + self.min_delta:
            self._best_fitness = fitness
            self._no_improvement = 0
        else:
            self._no_improvement += 1

        if self._no_improvement >= self.patience:
            self._apply_boost(trainer, generation)
            self._no_improvement = 0

    def _apply_boost(self, trainer: "RetrocausalTrainer", generation: int):
        """Apply mutation boost to escape plateau."""
        self._boosts_applied += 1

        if self.verbose:
            print(
                f"🔄 Plateau detected at gen {generation + 1}, applying boost #{self._boosts_applied}"
            )

        # Boost mutation if evolver is accessible
        if hasattr(trainer, "evolver") and trainer.evolver is not None:
            evolver = trainer.evolver
            if hasattr(evolver, "mutation_rate"):
                evolver.mutation_rate *= self.mutation_boost


# Convenience function for common callback sets
def default_callbacks(
    save_dir: Optional[str] = None,
    log_path: Optional[str] = None,
    verbose: bool = True,
) -> List[Callback]:
    """
    Create a default set of callbacks.

    Args:
        save_dir: Directory for checkpoints (optional)
        log_path: Path for metrics log (optional)
        verbose: Print updates

    Returns:
        List of callbacks
    """
    callbacks = [
        SyntonyCallback(log_interval=10, verbose=verbose),
        ArchonicEarlyStop(patience=20, verbose=verbose),
        FitnessPlateauCallback(patience=10, verbose=verbose),
    ]

    if save_dir:
        callbacks.append(SyntonyCheckpoint(save_dir, save_best=True))

    if log_path:
        callbacks.append(MetricsLogger(log_path))

    return callbacks
