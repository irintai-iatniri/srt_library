"""
Gradient-Based Trainer using SRT Autograd Kernels

This module provides gradient-based training functionality using the CUDA autograd
kernels, complementing the RES (Resonant Evolution Strategy) approach.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class GradientConfig:
    """Configuration for gradient-based training."""

    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    device_idx: int = 0


class GradientTrainer:
    """
    Gradient-based trainer using SRT autograd CUDA kernels.

    This trainer provides standard backpropagation using the autograd kernels
    defined in Rust, enabling gradient-based optimization alongside RES training.
    """

    def __init__(self, config: Optional[GradientConfig] = None):
        """
        Initialize the gradient trainer.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or GradientConfig()
        self._momentum_buffers: Dict[str, List[float]] = {}

        # Import autograd functions from _core
        try:
            from ..._core import (
                py_backward_add,
                py_backward_mul,
                py_backward_softmax,
                py_backward_layernorm,
                py_backward_phi_residual,
                py_load_autograd_kernels,
            )
            self._backward_add = py_backward_add
            self._backward_mul = py_backward_mul
            self._backward_softmax = py_backward_softmax
            self._backward_layernorm = py_backward_layernorm
            self._backward_phi_residual = py_backward_phi_residual
            self._load_kernels = py_load_autograd_kernels
            self._kernels_loaded = False
        except ImportError as e:
            raise ImportError(f"Failed to import autograd functions from _core: {e}")

    def load_kernels(self) -> List[str]:
        """
        Load all autograd CUDA kernels.

        Returns:
            List of loaded kernel names.
        """
        if not self._kernels_loaded:
            kernels = self._load_kernels(self.config.device_idx)
            self._kernels_loaded = True
            return kernels
        return []

    def backward_add(
        self, grad_output: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute gradients for element-wise addition.

        For z = x + y:
        - grad_x = grad_output
        - grad_y = grad_output

        Args:
            grad_output: Gradient from the next layer.

        Returns:
            Tuple of (grad_x, grad_y).
        """
        return self._backward_add(grad_output, self.config.device_idx)

    def backward_mul(
        self, grad_output: List[float], x: List[float], y: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute gradients for element-wise multiplication.

        For z = x * y:
        - grad_x = grad_output * y
        - grad_y = grad_output * x

        Args:
            grad_output: Gradient from the next layer.
            x: First input tensor.
            y: Second input tensor.

        Returns:
            Tuple of (grad_x, grad_y).
        """
        return self._backward_mul(grad_output, x, y, self.config.device_idx)

    def backward_softmax(
        self, grad_output: List[float], softmax_output: List[float]
    ) -> List[float]:
        """
        Compute gradients for softmax operation.

        Args:
            grad_output: Gradient from the next layer.
            softmax_output: The softmax output from forward pass.

        Returns:
            Gradient with respect to softmax input.
        """
        return self._backward_softmax(grad_output, softmax_output, self.config.device_idx)

    def backward_layernorm(
        self,
        grad_output: List[float],
        input_data: List[float],
        normalized: List[float],
        gamma: List[float],
        mean: float,
        inv_std: float,
    ) -> Tuple[List[float], List[float], float]:
        """
        Compute gradients for layer normalization.

        Args:
            grad_output: Gradient from the next layer.
            input_data: Original input to layer norm.
            normalized: Normalized values from forward pass.
            gamma: Scale parameter.
            mean: Mean computed during forward pass.
            inv_std: Inverse standard deviation from forward pass.

        Returns:
            Tuple of (grad_input, grad_gamma, grad_beta).
        """
        return self._backward_layernorm(
            grad_output, input_data, normalized, gamma, mean, inv_std,
            self.config.device_idx
        )

    def backward_phi_residual(
        self, grad_output: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute gradients for phi-residual connection.

        For output = input + phi * layer_output:
        - grad_input = grad_output
        - grad_layer = grad_output * phi

        Args:
            grad_output: Gradient from the next layer.

        Returns:
            Tuple of (grad_input, grad_layer_output).
        """
        return self._backward_phi_residual(grad_output, self.config.device_idx)

    def clip_gradients(self, gradients: List[float]) -> List[float]:
        """
        Clip gradients by norm.

        Args:
            gradients: List of gradient values.

        Returns:
            Clipped gradients.
        """
        if not gradients:
            return gradients

        norm = sum(g * g for g in gradients) ** 0.5
        if norm > self.config.gradient_clip:
            scale = self.config.gradient_clip / norm
            return [g * scale for g in gradients]
        return gradients

    def apply_weight_update(
        self,
        weights: List[float],
        gradients: List[float],
        param_name: str = "default",
    ) -> List[float]:
        """
        Apply SGD with momentum weight update.

        Args:
            weights: Current weight values.
            gradients: Gradient values.
            param_name: Name for momentum buffer tracking.

        Returns:
            Updated weights.
        """
        if len(weights) != len(gradients):
            raise ValueError("weights and gradients must have same length")

        # Clip gradients
        clipped_grads = self.clip_gradients(gradients)

        # Initialize momentum buffer if needed
        if param_name not in self._momentum_buffers:
            self._momentum_buffers[param_name] = [0.0] * len(weights)

        momentum_buf = self._momentum_buffers[param_name]

        # SGD with momentum and weight decay
        updated_weights = []
        for i, (w, g) in enumerate(zip(weights, clipped_grads)):
            # Apply weight decay
            g = g + self.config.weight_decay * w

            # Apply momentum
            momentum_buf[i] = self.config.momentum * momentum_buf[i] + g

            # Update weight
            w_new = w - self.config.learning_rate * momentum_buf[i]
            updated_weights.append(w_new)

        return updated_weights
