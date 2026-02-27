"""
Syntony Visualization: Tools for visualizing syntonic networks.

Provides plotting utilities for:
- Syntony history
- Layer-wise syntony
- Archonic regions
- Training dynamics

Requires matplotlib for plotting.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


def _check_matplotlib():
    """Check if matplotlib is available."""
    import importlib.util

    return importlib.util.find_spec("matplotlib") is not None


def plot_syntony_history(
    history: List[float],
    target: float = S_TARGET,
    title: str = "Syntony Over Training",
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot syntony history over training.

    Args:
        history: List of syntony values
        target: Target syntony line
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib figure or None if matplotlib unavailable
    """
    if not _check_matplotlib():
        print("matplotlib not available for plotting")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Plot syntony
    ax.plot(history, "b-", label="Syntony", alpha=0.7)

    # Target line
    ax.axhline(
        y=target, color="g", linestyle="--", label=f"Target (φ-q ≈ {target:.3f})"
    )

    # Warning zone
    ax.axhline(
        y=target - 0.1, color="orange", linestyle=":", alpha=0.5, label="Warning zone"
    )

    # Moving average
    if len(history) > 10:
        window = min(50, len(history) // 5)
        ma = [
            sum(history[max(0, i - window) : i + 1]) / min(i + 1, window + 1)
            for i in range(len(history))
        ]
        ax.plot(ma, "r-", label=f"MA({window})", alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Syntony")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layer_syntonies(
    layer_syntonies: List[float],
    layer_names: Optional[List[str]] = None,
    target: float = S_TARGET,
    title: str = "Layer-wise Syntony",
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot syntony per layer.

    Args:
        layer_syntonies: Syntony values per layer
        layer_names: Optional layer names
        target: Target syntony
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save

    Returns:
        matplotlib figure or None
    """
    if not _check_matplotlib():
        print("matplotlib not available for plotting")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    n_layers = len(layer_syntonies)
    x = range(n_layers)

    # Color by syntony level
    colors = [
        "green" if s >= target - 0.1 else "orange" if s >= target - 0.2 else "red"
        for s in layer_syntonies
    ]

    ax.bar(x, layer_syntonies, color=colors, alpha=0.7)

    # Target line
    ax.axhline(y=target, color="g", linestyle="--", label="Target")

    # Labels
    if layer_names:
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha="right")
    else:
        ax.set_xlabel("Layer")

    ax.set_ylabel("Syntony")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_archonic_regions(
    history: List[float],
    archonic_mask: Optional[List[bool]] = None,
    target: float = S_TARGET,
    variance_threshold: float = 0.01,
    title: str = "Archonic Pattern Detection",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot syntony with archonic regions highlighted.

    Args:
        history: Syntony history
        archonic_mask: Optional pre-computed archonic mask
        target: Target syntony
        variance_threshold: Threshold for archonic detection
        title: Plot title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib figure or None
    """
    if not _check_matplotlib():
        print("matplotlib not available for plotting")
        return None

    import matplotlib.pyplot as plt

    from ..analysis.archonic_detector import detect_archonic_pattern

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot syntony
    ax1.plot(history, "b-", alpha=0.7, label="Syntony")
    ax1.axhline(y=target, color="g", linestyle="--", alpha=0.5, label="Target")

    # Detect archonic regions if mask not provided
    if archonic_mask is None:
        archonic_mask = []
        for i in range(len(history)):
            if i < 50:
                archonic_mask.append(False)
            else:
                report = detect_archonic_pattern(history[: i + 1], window_size=50)
                archonic_mask.append(report.is_archonic)

    # Highlight archonic regions
    archonic_regions = []
    in_region = False
    start = 0

    for i, is_archonic in enumerate(archonic_mask):
        if is_archonic and not in_region:
            start = i
            in_region = True
        elif not is_archonic and in_region:
            archonic_regions.append((start, i))
            in_region = False

    if in_region:
        archonic_regions.append((start, len(history)))

    for start, end in archonic_regions:
        ax1.axvspan(
            start,
            end,
            alpha=0.3,
            color="red",
            label="Archonic" if start == archonic_regions[0][0] else "",
        )

    ax1.set_ylabel("Syntony")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Plot rolling variance
    window = 50
    variances = []
    for i in range(len(history)):
        if i < window:
            variances.append(0)
        else:
            w = history[i - window : i]
            mean = sum(w) / len(w)
            var = sum((x - mean) ** 2 for x in w) / len(w)
            variances.append(var)

    ax2.plot(variances, "orange", alpha=0.7, label="Variance")
    ax2.axhline(
        y=variance_threshold, color="r", linestyle="--", alpha=0.5, label="Threshold"
    )

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class SyntonyViz:
    """
    Comprehensive visualization suite for syntonic networks.

    Example:
        >>> viz = SyntonyViz()
        >>> viz.update(epoch=1, syntony=0.8, loss=0.5)
        >>> viz.plot_all()
    """

    def __init__(self):
        """Initialize visualization suite."""
        self._history: Dict[str, List[float]] = {
            "syntony": [],
            "loss": [],
            "lr": [],
        }
        self._layer_syntonies: List[List[float]] = []
        self._epochs: List[int] = []

    def update(
        self,
        epoch: int,
        syntony: Optional[float] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        layer_syntonies: Optional[List[float]] = None,
    ):
        """
        Update with new metrics.

        Args:
            epoch: Current epoch
            syntony: Model syntony
            loss: Training loss
            lr: Learning rate
            layer_syntonies: Per-layer syntonies
        """
        self._epochs.append(epoch)

        if syntony is not None:
            self._history["syntony"].append(syntony)
        if loss is not None:
            self._history["loss"].append(loss)
        if lr is not None:
            self._history["lr"].append(lr)
        if layer_syntonies is not None:
            self._layer_syntonies.append(layer_syntonies)

    def plot_all(
        self,
        figsize: tuple = (14, 10),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create comprehensive visualization.

        Args:
            figsize: Figure size
            save_path: Optional save path

        Returns:
            matplotlib figure or None
        """
        if not _check_matplotlib():
            print("matplotlib not available for plotting")
            return None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Syntony history
        if self._history["syntony"]:
            ax = axes[0, 0]
            ax.plot(
                self._epochs[: len(self._history["syntony"])],
                self._history["syntony"],
                "b-",
                alpha=0.7,
            )
            ax.axhline(y=S_TARGET, color="g", linestyle="--", alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Syntony")
            ax.set_title("Syntony Over Training")
            ax.grid(True, alpha=0.3)

        # Loss history
        if self._history["loss"]:
            ax = axes[0, 1]
            ax.semilogy(
                self._epochs[: len(self._history["loss"])],
                self._history["loss"],
                "r-",
                alpha=0.7,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)

        # Learning rate
        if self._history["lr"]:
            ax = axes[1, 0]
            ax.plot(
                self._epochs[: len(self._history["lr"])],
                self._history["lr"],
                "g-",
                alpha=0.7,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        # Latest layer syntonies
        if self._layer_syntonies:
            ax = axes[1, 1]
            latest = self._layer_syntonies[-1]
            colors = [
                (
                    "green"
                    if s >= S_TARGET - 0.1
                    else "orange" if s >= S_TARGET - 0.2 else "red"
                )
                for s in latest
            ]
            ax.bar(range(len(latest)), latest, color=colors, alpha=0.7)
            ax.axhline(y=S_TARGET, color="g", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Syntony")
            ax.set_title("Layer-wise Syntony (Latest)")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_syntony_heatmap(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Plot layer syntony evolution as heatmap.

        Args:
            figsize: Figure size
            save_path: Optional save path

        Returns:
            matplotlib figure or None
        """
        if not _check_matplotlib() or not self._layer_syntonies:
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=figsize)

        # Stack layer syntonies
        data = np.array(self._layer_syntonies).T

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Layer")
        ax.set_title("Layer Syntony Evolution")

        plt.colorbar(im, ax=ax, label="Syntony")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def reset(self):
        """Reset all history."""
        self._history = {"syntony": [], "loss": [], "lr": []}
        self._layer_syntonies = []
        self._epochs = []

    @property
    def history(self) -> Dict[str, List[float]]:
        """Get metric history."""
        return self._history
