"""
Archonic Pattern Detection: Identify "stuck" configurations.

Archonic patterns (from CRT.md §10) are configurations where:
- Syntony is below target (S < φ - q)
- High variance without improvement (cycling)
- Network is trapped in local minimum

Detection enables early intervention.

Source: CRT.md §10, §12.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


@dataclass
class ArchonicReport:
    """
    Report of archonic pattern analysis.

    Attributes:
        is_archonic: Whether archonic pattern detected
        severity: Severity score (0-1)
        duration: How many steps in archonic state
        mean_syntony: Average syntony during detection
        variance: Syntony variance
        trend: Recent syntony trend
        recommendation: Suggested intervention
    """

    is_archonic: bool
    severity: float = 0.0
    duration: int = 0
    mean_syntony: float = 0.0
    variance: float = 0.0
    trend: float = 0.0
    recommendation: str = ""

    def __str__(self) -> str:
        if not self.is_archonic:
            return "No archonic pattern detected"
        return (
            f"ARCHONIC PATTERN DETECTED\n"
            f"  Severity: {self.severity:.3f}\n"
            f"  Duration: {self.duration} steps\n"
            f"  Mean syntony: {self.mean_syntony:.4f} (target: {S_TARGET:.4f})\n"
            f"  Variance: {self.variance:.6f}\n"
            f"  Trend: {self.trend:+.6f}\n"
            f"  Recommendation: {self.recommendation}"
        )


def detect_archonic_pattern(
    syntony_history: List[float],
    window_size: int = 50,
    variance_threshold: float = 0.01,
    trend_threshold: float = 0.001,
    min_gap: float = 0.1,
) -> ArchonicReport:
    """
    Detect archonic pattern in syntony history.

    Args:
        syntony_history: List of syntony values over time
        window_size: Window for analysis
        variance_threshold: Minimum variance for cycling detection
        trend_threshold: Maximum trend for "stuck" detection
        min_gap: Minimum gap below target for archonic

    Returns:
        ArchonicReport with detection results
    """
    if len(syntony_history) < window_size:
        return ArchonicReport(is_archonic=False)

    # Get recent window
    recent = syntony_history[-window_size:]

    # Compute statistics
    mean_S = sum(recent) / len(recent)
    variance = sum((s - mean_S) ** 2 for s in recent) / len(recent)

    # Compute trend (first half vs second half)
    mid = len(recent) // 2
    first_half = recent[:mid]
    second_half = recent[mid:]
    trend = sum(second_half) / len(second_half) - sum(first_half) / len(first_half)

    # Gap from target
    target_gap = S_TARGET - mean_S

    # Archonic conditions:
    # 1. Below target by min_gap
    # 2. High variance (cycling)
    # 3. Low trend (stuck)
    is_archonic = (
        target_gap > min_gap
        and variance > variance_threshold
        and abs(trend) < trend_threshold
    )

    if not is_archonic:
        return ArchonicReport(
            is_archonic=False,
            mean_syntony=mean_S,
            variance=variance,
            trend=trend,
        )

    # Compute severity (0-1)
    severity = min(
        1.0,
        (
            0.4 * (target_gap / S_TARGET)
            + 0.3 * min(1.0, variance / variance_threshold)
            + 0.3 * (1.0 - abs(trend) / trend_threshold)
        ),
    )

    # Determine duration (how long has this been archonic?)
    duration = _compute_archonic_duration(
        syntony_history, variance_threshold, trend_threshold
    )

    # Generate recommendation
    recommendation = _generate_recommendation(severity, duration, variance, trend)

    return ArchonicReport(
        is_archonic=True,
        severity=severity,
        duration=duration,
        mean_syntony=mean_S,
        variance=variance,
        trend=trend,
        recommendation=recommendation,
    )


def _compute_archonic_duration(
    history: List[float],
    var_threshold: float,
    trend_threshold: float,
) -> int:
    """Count consecutive steps in archonic state."""
    if len(history) < 20:
        return 0

    duration = 0
    for end in range(len(history), 20, -10):
        window = history[max(0, end - 50) : end]
        if len(window) < 20:
            break

        mean_S = sum(window) / len(window)
        var_S = sum((s - mean_S) ** 2 for s in window) / len(window)

        mid = len(window) // 2
        trend = sum(window[mid:]) / len(window[mid:]) - sum(window[:mid]) / len(
            window[:mid]
        )

        if (
            var_S > var_threshold
            and abs(trend) < trend_threshold
            and mean_S < S_TARGET - 0.1
        ):
            duration += 10
        else:
            break

    return duration


def _generate_recommendation(
    severity: float,
    duration: int,
    variance: float,
    trend: float,
) -> str:
    """Generate intervention recommendation."""
    if severity > 0.7:
        if duration > 100:
            return "Critical: Consider architecture change or reset"
        return "Inject noise and increase learning rate significantly"
    elif severity > 0.4:
        if variance > 0.02:
            return "High cycling: Reduce learning rate, add noise injection"
        return "Moderate stuck: Increase learning rate by 50%"
    else:
        return "Minor archonic: Monitor closely, consider slight lr increase"


class ArchonicDetector:
    """
    Real-time archonic pattern detector.

    Monitors syntony during training and triggers
    alerts when archonic patterns are detected.

    Example:
        >>> detector = ArchonicDetector()
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     report = detector.update(model.syntony)
        ...     if report.is_archonic:
        ...         print(report)
        ...         apply_intervention(report)
    """

    def __init__(
        self,
        window_size: int = 50,
        variance_threshold: float = 0.01,
        trend_threshold: float = 0.001,
        alert_cooldown: int = 20,
    ):
        """
        Initialize archonic detector.

        Args:
            window_size: Analysis window size
            variance_threshold: Threshold for cycling detection
            trend_threshold: Threshold for stuck detection
            alert_cooldown: Steps between alerts
        """
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.trend_threshold = trend_threshold
        self.alert_cooldown = alert_cooldown

        self._history: List[float] = []
        self._steps_since_alert = 0
        self._total_archonic_steps = 0
        self._alert_count = 0

    def update(self, syntony: float) -> ArchonicReport:
        """
        Update detector with new syntony value.

        Args:
            syntony: Current model syntony

        Returns:
            ArchonicReport (may or may not indicate archonic)
        """
        self._history.append(syntony)
        self._steps_since_alert += 1

        # Detect pattern
        report = detect_archonic_pattern(
            self._history,
            self.window_size,
            self.variance_threshold,
            self.trend_threshold,
        )

        if report.is_archonic:
            self._total_archonic_steps += 1

            # Only alert if past cooldown
            if self._steps_since_alert >= self.alert_cooldown:
                self._steps_since_alert = 0
                self._alert_count += 1
                return report

        return ArchonicReport(
            is_archonic=False,
            mean_syntony=report.mean_syntony,
            variance=report.variance,
            trend=report.trend,
        )

    def reset(self):
        """Reset detector state."""
        self._history = []
        self._steps_since_alert = 0
        self._total_archonic_steps = 0
        self._alert_count = 0

    @property
    def history(self) -> List[float]:
        """Get syntony history."""
        return self._history

    @property
    def archonic_fraction(self) -> float:
        """Fraction of steps in archonic state."""
        if not self._history:
            return 0.0
        return self._total_archonic_steps / len(self._history)

    @property
    def alert_count(self) -> int:
        """Number of alerts triggered."""
        return self._alert_count

    def get_summary(self) -> Dict[str, Any]:
        """Get detector summary."""
        return {
            "total_steps": len(self._history),
            "archonic_steps": self._total_archonic_steps,
            "archonic_fraction": self.archonic_fraction,
            "alert_count": self._alert_count,
            "current_syntony": self._history[-1] if self._history else None,
        }
