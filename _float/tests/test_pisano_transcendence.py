"""Test Pisano periods and transcendence gates."""

from syntonic._core import (
    fibonacci_number,
    fibonacci_resonance_boost,
    is_fibonacci_prime,
    is_lucas_gap,
    is_transcendence_gate,
    lucas_gap_pressure,
    pisano_period,
    versal_grip_strength,
)


def test_known_pisano_periods():
    """Test known Pisano periods."""
    assert pisano_period(2) == 3
    assert pisano_period(3) == 8
    assert pisano_period(5) == 20
    assert pisano_period(7) == 16
    assert pisano_period(10) == 60


def test_versal_grip():
    """Test versal grip strength calculation."""
    # Self-hooking primes have non-zero grip
    grip_5 = versal_grip_strength(5)
    assert grip_5 > 0  # π(5) = 20, 20 % 5 == 0

    # Non-self-hooking have zero grip
    grip_4 = versal_grip_strength(4)
    assert grip_4 == 0.0


def test_transcendence_gates():
    """Test transcendence gate detection."""
    assert is_transcendence_gate(3) == True
    assert is_transcendence_gate(17) == True
    assert is_transcendence_gate(6) == False
    assert is_transcendence_gate(100) == False


def test_fibonacci_numbers():
    """Test Fibonacci number computation."""
    assert fibonacci_number(0) == 0
    assert fibonacci_number(1) == 1
    assert fibonacci_number(5) == 5  # F_5 = 5
    assert fibonacci_number(10) == 55  # F_10 = 55


def test_fibonacci_primes():
    """Test Fibonacci prime detection."""
    assert is_fibonacci_prime(3) == True  # F_3 = 2 (prime)
    assert is_fibonacci_prime(4) == True  # F_4 = 3 (prime)
    assert is_fibonacci_prime(5) == True  # F_5 = 5 (prime, in known list)


def test_fibonacci_resonance_boost():
    """Test Fibonacci resonance boost factors."""
    # Transcendence gates get φ^n boost
    boost_3 = fibonacci_resonance_boost(3)
    assert boost_3 > 1.0  # Should be φ³

    # Material anomaly gets reduced boost
    boost_4 = fibonacci_resonance_boost(4)
    assert boost_4 < fibonacci_resonance_boost(5)  # 4 should be less than 5

    # Non-gates get no boost
    boost_6 = fibonacci_resonance_boost(6)
    assert boost_6 == 1.0


def test_lucas_gaps():
    """Test Lucas gap detection and pressure."""
    # Lucas primes should not be gaps
    assert is_lucas_gap(0) == False  # L_0 = 2 prime
    assert is_lucas_gap(17) == False  # L_17 = 3571 prime

    # Gaps should have pressure
    assert is_lucas_gap(20) == True
    assert is_lucas_gap(25) == True

    # Pressure in gap should be non-zero
    assert lucas_gap_pressure(25) > 0

    # Pressure at prime should be zero
    assert lucas_gap_pressure(17) == 0
