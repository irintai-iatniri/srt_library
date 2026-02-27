"""
Network Health Monitoring: Track syntonic network health.

Monitors various aspects of network health including:
- Syntony levels
- Gradient health
- Weight distribution
- Activation patterns

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


@dataclass
class HealthReport:
    """
    Comprehensive health report for a syntonic network.

    Attributes:
        overall_health: Overall health score (0-1)
        syntony_health: Syntony-related health
        gradient_health: Gradient-related health
        weight_health: Weight distribution health
        is_healthy: Whether network is healthy
        warnings: List of warnings
        recommendations: List of recommendations
    """

    overall_health: float = 0.0
    syntony_health: float = 0.0
    gradient_health: float = 0.0
    weight_health: float = 0.0
    activation_health: float = 0.0
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        s = f"Network Health: {status} ({self.overall_health:.2%})\n"
        s += f"  Syntony: {self.syntony_health:.2%}\n"
        s += f"  Gradients: {self.gradient_health:.2%}\n"
        s += f"  Weights: {self.weight_health:.2%}\n"
        s += f"  Activations: {self.activation_health:.2%}\n"

        if self.warnings:
            s += "\nWarnings:\n"
            for w in self.warnings:
                s += f"  - {w}\n"

        if self.recommendations:
            s += "\nRecommendations:\n"
            for r in self.recommendations:
                s += f"  - {r}\n"

        return s


class SyntonyMonitor:
    """
    Monitor syntony levels over time.

    Tracks syntony history and computes statistics.

    Example:
        >>> monitor = SyntonyMonitor()
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     monitor.update(model.syntony)
        >>> print(f"Average syntony: {monitor.mean_syntony:.4f}")
    """

    def __init__(
        self,
        target: float = S_TARGET,
        window_size: int = 100,
    ):
        """
        Initialize monitor.

        Args:
            target: Target syntony
            window_size: Window for statistics
        """
        self.target = target
        self.window_size = window_size

        self._history: List[float] = []
        self._peak_syntony = 0.0
        self._steps_above_target = 0

    def update(self, syntony: float):
        """Update with new syntony value."""
        self._history.append(syntony)

        if syntony > self._peak_syntony:
            self._peak_syntony = syntony

        if syntony >= self.target:
            self._steps_above_target += 1

    @property
    def current_syntony(self) -> float:
        """Get current syntony."""
        return self._history[-1] if self._history else 0.0

    @property
    def mean_syntony(self) -> float:
        """Get mean syntony."""
        if not self._history:
            return 0.0
        window = self._history[-self.window_size :]
        return sum(window) / len(window)

    @property
    def variance(self) -> float:
        """Get syntony variance."""
        if len(self._history) < 2:
            return 0.0
        window = self._history[-self.window_size :]
        mean = sum(window) / len(window)
        return sum((s - mean) ** 2 for s in window) / len(window)

    @property
    def trend(self) -> float:
        """Get recent trend."""
        if len(self._history) < 10:
            return 0.0
        window = self._history[-self.window_size :]
        mid = len(window) // 2
        return sum(window[mid:]) / len(window[mid:]) - sum(window[:mid]) / len(
            window[:mid]
        )

    @property
    def peak_syntony(self) -> float:
        """Get peak syntony achieved."""
        return self._peak_syntony

    @property
    def fraction_above_target(self) -> float:
        """Fraction of steps above target."""
        if not self._history:
            return 0.0
        return self._steps_above_target / len(self._history)

    @property
    def history(self) -> List[float]:
        """Get full history."""
        return self._history

    def get_statistics(self) -> Dict[str, float]:
        """Get all statistics."""
        return {
            "current": self.current_syntony,
            "mean": self.mean_syntony,
            "variance": self.variance,
            "trend": self.trend,
            "peak": self.peak_syntony,
            "fraction_above_target": self.fraction_above_target,
            "target": self.target,
        }

    def reset(self):
        """Reset monitor."""
        self._history = []
        self._peak_syntony = 0.0
        self._steps_above_target = 0


class NetworkHealth:
    """
    Comprehensive network health analyzer.

    Checks multiple aspects of network health:
    - Syntony levels and trends
    - Gradient statistics
    - Weight distributions
    - Activation patterns

    Example:
        >>> health = NetworkHealth(model)
        >>> report = health.check()
        >>> if not report.is_healthy:
        ...     print(report)
        ...     apply_fixes(report.recommendations)
    """

    def __init__(
        self,
        model: nn.Module,
        syntony_target: float = S_TARGET,
    ):
        """
        Initialize health analyzer.

        Args:
            model: Neural network to monitor
            syntony_target: Target syntony level
        """
        self.model = model
        self.syntony_target = syntony_target

        self._syntony_monitor = SyntonyMonitor(syntony_target)
        self._last_report: Optional[HealthReport] = None

    def check(
        self,
        inputs: Optional[torch.Tensor] = None,
    ) -> HealthReport:
        """
        Perform comprehensive health check.

        Args:
            inputs: Optional input for activation analysis

        Returns:
            HealthReport
        """
        warnings = []
        recommendations = []

        # 1. Syntony health
        syntony_health, syntony_warnings = self._check_syntony()
        warnings.extend(syntony_warnings)

        # 2. Gradient health
        gradient_health, grad_warnings = self._check_gradients()
        warnings.extend(grad_warnings)

        # 3. Weight health
        weight_health, weight_warnings = self._check_weights()
        warnings.extend(weight_warnings)

        # 4. Activation health (if inputs provided)
        if inputs is not None:
            activation_health, act_warnings = self._check_activations(inputs)
            warnings.extend(act_warnings)
        else:
            activation_health = 1.0

        # Overall health
        overall_health = (
            0.4 * syntony_health
            + 0.2 * gradient_health
            + 0.2 * weight_health
            + 0.2 * activation_health
        )

        is_healthy = overall_health > 0.6 and syntony_health > 0.4

        # Generate recommendations
        if syntony_health < 0.5:
            recommendations.append(
                "Syntony below target - consider noise injection or lr increase"
            )
        if gradient_health < 0.5:
            recommendations.append("Gradient issues detected - check gradient clipping")
        if weight_health < 0.5:
            recommendations.append(
                "Weight distribution unhealthy - consider re-initialization"
            )
        if activation_health < 0.5:
            recommendations.append("Activation issues - check for dead/saturated units")

        report = HealthReport(
            overall_health=overall_health,
            syntony_health=syntony_health,
            gradient_health=gradient_health,
            weight_health=weight_health,
            activation_health=activation_health,
            is_healthy=is_healthy,
            warnings=warnings,
            recommendations=recommendations,
        )

        self._last_report = report
        return report

    def _check_syntony(self) -> tuple:
        """Check syntony health."""
        warnings = []

        # Get model syntony
        syntonies = []
        for module in self.model.modules():
            if hasattr(module, "syntony") and module.syntony is not None:
                syntonies.append(module.syntony)

        if not syntonies:
            return 0.5, ["No syntony tracking found in model"]

        mean_syntony = sum(syntonies) / len(syntonies)
        min_syntony = min(syntonies)
        max_syntony = max(syntonies)

        # Update monitor
        self._syntony_monitor.update(mean_syntony)

        # Health score based on proximity to target
        gap = self.syntony_target - mean_syntony
        health = max(0.0, 1.0 - gap / self.syntony_target)

        # Warnings
        if mean_syntony < self.syntony_target - 0.2:
            warnings.append(
                f"Syntony significantly below target: {mean_syntony:.4f} < {self.syntony_target:.4f}"
            )

        if max_syntony - min_syntony > 0.3:
            warnings.append(
                f"High syntony variance across layers: [{min_syntony:.4f}, {max_syntony:.4f}]"
            )

        if self._syntony_monitor.trend < -0.01:
            warnings.append("Syntony trending downward")

        return health, warnings

    def _check_gradients(self) -> tuple:
        """Check gradient health."""
        warnings = []

        grad_norms = []
        has_nan = False
        has_zero = False

        for param in self.model.parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms.append(norm)

                if math.isnan(norm) or math.isinf(norm):
                    has_nan = True
                if norm == 0:
                    has_zero = True

        if not grad_norms:
            return 1.0, []  # No gradients to check

        if has_nan:
            warnings.append("NaN/Inf gradients detected!")
            return 0.0, warnings

        mean_norm = sum(grad_norms) / len(grad_norms)
        max_norm = max(grad_norms)

        # Health based on gradient magnitude
        health = 1.0

        if max_norm > 100:
            health *= 0.5
            warnings.append(f"Exploding gradients: max norm = {max_norm:.2f}")

        if mean_norm < 1e-7:
            health *= 0.5
            warnings.append(f"Vanishing gradients: mean norm = {mean_norm:.2e}")

        if has_zero:
            health *= 0.9
            warnings.append("Some zero gradients detected")

        return health, warnings

    def _check_weights(self) -> tuple:
        """Check weight distribution health."""
        warnings = []

        weight_stats = []
        has_nan = False

        for name, param in self.model.named_parameters():
            if param.dim() >= 2:
                mean = param.mean().item()
                std = param.std().item()

                if math.isnan(mean) or math.isnan(std):
                    has_nan = True
                    continue

                weight_stats.append(
                    {
                        "name": name,
                        "mean": mean,
                        "std": std,
                    }
                )

        if has_nan:
            warnings.append("NaN weights detected!")
            return 0.0, warnings

        if not weight_stats:
            return 1.0, []

        health = 1.0

        for stat in weight_stats:
            # Check for mean drift
            if abs(stat["mean"]) > 1.0:
                health *= 0.9
                warnings.append(f"Large mean in {stat['name']}: {stat['mean']:.4f}")

            # Check for too small/large std
            if stat["std"] < 1e-4:
                health *= 0.9
                warnings.append(f"Very small std in {stat['name']}: {stat['std']:.6f}")
            elif stat["std"] > 10:
                health *= 0.9
                warnings.append(f"Very large std in {stat['name']}: {stat['std']:.4f}")

        return health, warnings

    def _check_activations(self, inputs: torch.Tensor) -> tuple:
        """Check activation health."""
        warnings = []
        # activation_stats = []

        # Hook to capture activations
        activations = []

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach())

        handles = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU)):
                handles.append(module.register_forward_hook(hook))

        # Forward pass
        with torch.no_grad():
            self.model(inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        health = 1.0

        for i, act in enumerate(activations):
            if act.dim() < 1:
                continue

            mean = act.mean().item()
            std = act.std().item()
            dead_fraction = (act.abs() < 1e-6).float().mean().item()
            saturated_fraction = (act.abs() > 10).float().mean().item()

            # Check distribution stats
            if abs(mean) > 10.0:
                warnings.append(f"Layer {i}: High mean activation ({mean:.2f})")
            if std > 20.0:
                warnings.append(f"Layer {i}: High activation variance ({std:.2f})")

            # Check for dead units
            if dead_fraction > 0.5:
                health *= 0.8
                warnings.append(f"Layer {i}: {dead_fraction:.1%} dead units")

            # Check for saturation
            if saturated_fraction > 0.1:
                health *= 0.9
                warnings.append(f"Layer {i}: {saturated_fraction:.1%} saturated units")

        return health, warnings

    @property
    def syntony_monitor(self) -> SyntonyMonitor:
        """Get syntony monitor."""
        return self._syntony_monitor

    @property
    def last_report(self) -> Optional[HealthReport]:
        """Get last health report."""
        return self._last_report
