"""
Escape Mechanisms: Methods for escaping archonic patterns.

When a network is stuck in an archonic configuration,
these mechanisms help it escape to higher syntony states.

Source: CRT.md §10, §12.2
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


class EscapeMechanism(ABC):
    """
    Base class for archonic escape mechanisms.

    Subclasses implement specific escape strategies.
    """

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Apply escape mechanism.

        Args:
            model: Neural network
            optimizer: Optional optimizer to modify

        Returns:
            Dict with escape statistics
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset mechanism state."""
        pass


class NoiseInjection(EscapeMechanism):
    """
    Inject noise into weights or gradients.

    Helps escape local minima by perturbing the
    network state.

    Example:
        >>> escape = NoiseInjection(scale=0.01, target='weights')
        >>> if detector.is_archonic:
        ...     stats = escape.apply(model)
        ...     print(f"Injected noise with scale {stats['scale']}")
    """

    def __init__(
        self,
        scale: float = 0.01,
        target: str = "weights",  # 'weights' or 'gradients'
        golden_scaling: bool = True,
        adaptive: bool = True,
    ):
        """
        Initialize noise injection.

        Args:
            scale: Base noise scale
            target: Where to inject ('weights' or 'gradients')
            golden_scaling: Scale noise by golden ratio per layer
            adaptive: Adapt scale based on severity
        """
        self.base_scale = scale
        self.target = target
        self.golden_scaling = golden_scaling
        self.adaptive = adaptive

        self._injection_count = 0
        self._last_scale = scale

    def apply(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        severity: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Apply noise injection.

        Args:
            model: Neural network
            optimizer: Optional optimizer (unused for weight injection)
            severity: Archonic severity (0-1) for adaptive scaling

        Returns:
            Statistics dict
        """
        # Compute scale
        if self.adaptive:
            scale = self.base_scale * (1 + severity)
        else:
            scale = self.base_scale

        self._last_scale = scale
        self._injection_count += 1

        # Count parameters modified
        n_params = 0
        total_noise_norm = 0.0

        # Apply to each parameter
        params = list(model.parameters())
        n_layers = len(params)

        for i, param in enumerate(params):
            if not param.requires_grad:
                continue

            # Golden scaling: less noise in later layers
            if self.golden_scaling:
                layer_scale = scale * (PHI ** (-(i / n_layers)))
            else:
                layer_scale = scale

            # Generate noise
            if self.target == "weights":
                noise = torch.randn_like(param.data) * layer_scale * param.data.std()
                param.data.add_(noise)
                total_noise_norm += noise.norm().item()
            elif self.target == "gradients" and param.grad is not None:
                noise = torch.randn_like(param.grad) * layer_scale * param.grad.std()
                param.grad.add_(noise)
                total_noise_norm += noise.norm().item()

            n_params += param.numel()

        return {
            "scale": scale,
            "target": self.target,
            "n_params": n_params,
            "total_noise_norm": total_noise_norm,
            "injection_count": self._injection_count,
        }

    def reset(self):
        """Reset injection state."""
        self._injection_count = 0
        self._last_scale = self.base_scale


class LearningRateShock(EscapeMechanism):
    """
    Apply learning rate shock to escape archonic patterns.

    Temporarily increases learning rate to push
    network out of local minimum.

    Example:
        >>> escape = LearningRateShock(multiplier=3.0, duration=10)
        >>> if detector.is_archonic:
        ...     escape.apply(model, optimizer)
        >>> # Call step() each iteration
        >>> escape.step(optimizer)  # Restores lr after duration
    """

    def __init__(
        self,
        multiplier: float = 3.0,
        duration: int = 10,
        decay: str = "linear",  # 'linear', 'exponential', 'step'
    ):
        """
        Initialize lr shock.

        Args:
            multiplier: LR multiplier during shock
            duration: Number of steps for shock
            decay: How to decay back to original lr
        """
        self.multiplier = multiplier
        self.duration = duration
        self.decay = decay

        self._original_lrs: List[float] = []
        self._shock_active = False
        self._remaining_steps = 0
        self._shock_count = 0

    def apply(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Apply learning rate shock.

        Args:
            model: Neural network (unused)
            optimizer: Optimizer to modify (required)

        Returns:
            Statistics dict
        """
        if optimizer is None:
            raise ValueError("Optimizer required for LearningRateShock")

        if self._shock_active:
            return {"status": "shock_already_active"}

        # Store original learning rates
        self._original_lrs = [group["lr"] for group in optimizer.param_groups]

        # Apply shock
        for group in optimizer.param_groups:
            group["lr"] *= self.multiplier

        self._shock_active = True
        self._remaining_steps = self.duration
        self._shock_count += 1

        return {
            "status": "shock_applied",
            "multiplier": self.multiplier,
            "duration": self.duration,
            "new_lrs": [group["lr"] for group in optimizer.param_groups],
            "shock_count": self._shock_count,
        }

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Step the shock mechanism.

        Call this each training iteration.

        Args:
            optimizer: The optimizer

        Returns:
            True if shock is still active
        """
        if not self._shock_active:
            return False

        self._remaining_steps -= 1

        if self._remaining_steps <= 0:
            # Restore original learning rates
            for group, original_lr in zip(optimizer.param_groups, self._original_lrs):
                group["lr"] = original_lr
            self._shock_active = False
            return False

        # Apply decay
        if self.decay == "linear":
            progress = 1 - (self._remaining_steps / self.duration)
            for group, original_lr in zip(optimizer.param_groups, self._original_lrs):
                shocked_lr = original_lr * self.multiplier
                group["lr"] = shocked_lr - (shocked_lr - original_lr) * progress

        elif self.decay == "exponential":
            decay_factor = (1 / self.multiplier) ** (1 / self.duration)
            for group in optimizer.param_groups:
                group["lr"] *= decay_factor

        # 'step' decay: stay at shocked lr until end
        return True

    def reset(self):
        """Reset shock state."""
        self._original_lrs = []
        self._shock_active = False
        self._remaining_steps = 0
        self._shock_count = 0

    @property
    def is_active(self) -> bool:
        """Check if shock is active."""
        return self._shock_active


class WeightPerturbation(EscapeMechanism):
    """
    Structured weight perturbation for escape.

    Perturbs weights in directions that increase
    syntony (based on gradient of syntony loss).

    Example:
        >>> escape = WeightPerturbation(magnitude=0.1)
        >>> if detector.is_archonic:
        ...     escape.apply(model)
    """

    def __init__(
        self,
        magnitude: float = 0.1,
        direction: str = "syntony",  # 'syntony', 'random', 'anti_gradient'
    ):
        """
        Initialize weight perturbation.

        Args:
            magnitude: Perturbation magnitude
            direction: Perturbation direction strategy
        """
        self.magnitude = magnitude
        self.direction = direction
        self._perturbation_count = 0

    def apply(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Apply structured perturbation.

        Args:
            model: Neural network
            optimizer: Optional optimizer

        Returns:
            Statistics dict
        """
        self._perturbation_count += 1

        total_perturb_norm = 0.0
        n_params = 0

        for param in model.parameters():
            if not param.requires_grad:
                continue

            if self.direction == "random":
                # Random direction
                direction = torch.randn_like(param.data)
                direction = direction / (direction.norm() + 1e-8)

            elif self.direction == "anti_gradient" and param.grad is not None:
                # Move against gradient (escape local min)
                direction = -param.grad
                direction = direction / (direction.norm() + 1e-8)

            else:
                # Default to random
                direction = torch.randn_like(param.data)
                direction = direction / (direction.norm() + 1e-8)

            # Apply perturbation
            perturbation = direction * self.magnitude * param.data.std()
            param.data.add_(perturbation)

            total_perturb_norm += perturbation.norm().item()
            n_params += param.numel()

        return {
            "magnitude": self.magnitude,
            "direction": self.direction,
            "total_perturb_norm": total_perturb_norm,
            "n_params": n_params,
            "perturbation_count": self._perturbation_count,
        }

    def reset(self):
        """Reset perturbation state."""
        self._perturbation_count = 0


class CombinedEscape(EscapeMechanism):
    """
    Combine multiple escape mechanisms.

    Applies mechanisms in sequence with adaptive
    selection based on severity.

    Example:
        >>> escape = CombinedEscape([
        ...     NoiseInjection(scale=0.01),
        ...     LearningRateShock(multiplier=2.0),
        ... ])
        >>> escape.apply(model, optimizer, severity=0.8)
    """

    def __init__(
        self,
        mechanisms: List[EscapeMechanism],
        severity_thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize combined escape.

        Args:
            mechanisms: List of escape mechanisms
            severity_thresholds: Thresholds for each mechanism
        """
        self.mechanisms = mechanisms

        if severity_thresholds is None:
            # Default: apply all for high severity
            n = len(mechanisms)
            self.severity_thresholds = [i / n for i in range(n)]
        else:
            self.severity_thresholds = severity_thresholds

    def apply(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        severity: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Apply appropriate mechanisms based on severity.

        Args:
            model: Neural network
            optimizer: Optimizer
            severity: Archonic severity (0-1)

        Returns:
            Combined statistics
        """
        results = {"mechanisms_applied": []}

        for mechanism, threshold in zip(self.mechanisms, self.severity_thresholds):
            if severity >= threshold:
                result = mechanism.apply(model, optimizer)
                results["mechanisms_applied"].append(
                    {
                        "type": type(mechanism).__name__,
                        "result": result,
                    }
                )

        results["severity"] = severity
        results["n_applied"] = len(results["mechanisms_applied"])

        return results

    def reset(self):
        """Reset all mechanisms."""
        for mechanism in self.mechanisms:
            mechanism.reset()
