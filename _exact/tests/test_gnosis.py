"""Test gnosis/consciousness module."""

from syntonic_applications.consciousness.gnosis import (
    COLLAPSE_THRESHOLD,
    gnosis_score,
    is_conscious,
)


def test_consciousness_threshold():
    assert COLLAPSE_THRESHOLD == 24.0
    assert not is_conscious(23.9)
    assert is_conscious(24.1)


def test_gnosis_score():
    # G = sqrt(S * C)
    assert abs(gnosis_score(0.5, 0.5) - 0.5) < 0.001
    assert abs(gnosis_score(1.0, 0.0) - 0.0) < 0.001
