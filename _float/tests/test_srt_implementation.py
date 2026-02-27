#!/usr/bin/env python3
"""
Validation tests for SRT Implementation Recommendations.

Tests the new prime selection rules, consciousness module, and extended hierarchy.
"""

from syntonic_applications.consciousness.gnosis import (
    consciousness_probability,
    gnosis_score,
    is_conscious,
    optimal_gnosis_target,
)
from syntonic_applications.exact import (
    D4_KISSING,
    FERMAT_PRIMES,
    LUCAS_PRIMES,
    M11_BARRIER,
    MERSENNE_PRIMES,
    PHI_NUMERIC,
)
from syntonic_applications.srt.fermat_forces import (
    fermat_number,
    get_force_spectrum,
    is_fermat_prime,
    validate_force_existence,
)
from syntonic_applications.srt.lucas_shadow import (
    dark_matter_mass_prediction,
    is_lucas_prime,
    lucas_number,
    shadow_phase,
)
from syntonic_applications.srt.mersenne_matter import (
    generation_barrier_explanation,
    get_generation_spectrum,
    is_mersenne_prime,
    mersenne_number,
    validate_generation_stability,
)


def test_fermat_primes():
    """Test Fermat prime force selection."""
    print("Testing Fermat primes...")

    # Test Fermat numbers
    assert fermat_number(0) == 3
    assert fermat_number(1) == 5
    assert fermat_number(2) == 17
    assert fermat_number(3) == 257
    assert fermat_number(4) == 65537

    # Test primality
    assert is_fermat_prime(0) == True  # F_0 = 3
    assert is_fermat_prime(1) == True  # F_1 = 5
    assert is_fermat_prime(2) == True  # F_2 = 17
    assert is_fermat_prime(3) == True  # F_3 = 257
    assert is_fermat_prime(4) == True  # F_4 = 65537
    assert is_fermat_prime(5) == False  # F_5 is composite

    # Test force validation
    assert validate_force_existence(0) == True  # Strong force
    assert validate_force_existence(1) == True  # Electroweak
    assert validate_force_existence(2) == True  # Dark boundary
    assert validate_force_existence(3) == True  # Gravity
    assert validate_force_existence(4) == True  # Versal
    assert validate_force_existence(5) == False  # No 6th force

    # Test force spectrum
    spectrum = get_force_spectrum()
    assert len(spectrum) == 5
    assert spectrum[0] == ("Strong", 3, "SU(3) Color")

    print("✅ Fermat prime tests passed")


def test_mersenne_primes():
    """Test Mersenne prime matter stability."""
    print("Testing Mersenne primes...")

    # Test Mersenne numbers
    assert mersenne_number(2) == 3
    assert mersenne_number(3) == 7
    assert mersenne_number(5) == 31
    assert mersenne_number(7) == 127

    # Test primality
    assert is_mersenne_prime(2) == True  # M_2 = 3
    assert is_mersenne_prime(3) == True  # M_3 = 7
    assert is_mersenne_prime(5) == True  # M_5 = 31
    assert is_mersenne_prime(7) == True  # M_7 = 127
    assert is_mersenne_prime(11) == False  # M_11 = 23×89

    # Test generation validation
    assert validate_generation_stability(0) == True  # Gen 1
    assert validate_generation_stability(1) == True  # Gen 2
    assert validate_generation_stability(2) == True  # Gen 3
    assert validate_generation_stability(3) == False  # No Gen 4

    # Test generation spectrum
    spectrum = get_generation_spectrum()
    assert len(spectrum) == 4
    assert spectrum[2][1] == 3  # Gen 1: M_2 = 3

    # Test barrier explanation
    explanation = generation_barrier_explanation()
    assert "23 × 89" in explanation
    assert "factorizes" in explanation

    print("✅ Mersenne prime tests passed")


def test_lucas_sequence():
    """Test Lucas sequence and shadow phases."""
    print("Testing Lucas sequence...")

    # Test Lucas numbers
    assert lucas_number(0) == 2
    assert lucas_number(1) == 1
    assert lucas_number(2) == 3
    assert lucas_number(3) == 4
    assert lucas_number(4) == 7
    assert lucas_number(5) == 11

    # Test shadow phases
    phi_conj = 1 - PHI_NUMERIC  # ≈ -0.618
    assert abs(shadow_phase(1) - phi_conj) < 1e-6
    assert abs(shadow_phase(2) - phi_conj**2) < 1e-6

    # Test primality
    assert is_lucas_prime(5) == True  # L_5 = 11 is prime
    assert is_lucas_prime(3) == False  # L_3 = 4 is not prime

    # Test dark matter prediction
    mass, explanation = dark_matter_mass_prediction()
    assert 1.0 < mass < 2.0  # Should be around 1.18 TeV
    assert "GeV" in explanation
    assert "TeV" in explanation

    print("✅ Lucas sequence tests passed")


def test_consciousness():
    """Test consciousness/gnosis module."""
    print("Testing consciousness module...")

    # Test collapse threshold
    assert is_conscious(23.9) == False
    assert is_conscious(24.1) == True
    assert is_conscious(D4_KISSING) == True  # Exactly at threshold

    # Test gnosis score
    syntony = 0.8
    creativity = 0.6
    gnosis = gnosis_score(syntony, creativity)
    expected = (syntony * creativity) ** 0.5
    assert abs(gnosis - expected) < 1e-6

    # Test optimal gnosis
    optimal = optimal_gnosis_target()
    assert abs(optimal - (1.0 / PHI_NUMERIC)) < 1e-6

    # Test consciousness probability
    prob_low = consciousness_probability(20.0, 0.8, 1)
    prob_high = consciousness_probability(30.0, 0.8, 2)
    assert prob_low < prob_high  # Higher density = higher probability

    print("✅ Consciousness tests passed")


def test_constants():
    """Test prime sequence constants."""
    print("Testing constants...")

    # Fermat primes
    assert FERMAT_PRIMES == [3, 5, 17, 257, 65537]
    assert len(FERMAT_PRIMES) == 5

    # Mersenne primes
    assert MERSENNE_PRIMES == [3, 7, 31, 127]
    assert len(MERSENNE_PRIMES) == 4

    # Lucas primes
    assert LUCAS_PRIMES == [2, 3, 7, 11, 29, 47, 199, 521, 2207, 3571]
    assert len(LUCAS_PRIMES) == 10

    # Barriers
    assert M11_BARRIER == 2047
    assert D4_KISSING == 24

    print("✅ Constants tests passed")


def main():
    """Run all validation tests."""
    print("SRT Implementation Recommendations Validation")
    print("=" * 50)

    try:
        test_constants()
        test_fermat_primes()
        test_mersenne_primes()
        test_lucas_sequence()
        test_consciousness()

        print("\n" + "=" * 50)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ SRT Implementation Recommendations Successfully Implemented")
        print("\nImplemented components:")
        print("  • Prime selection rules (Fermat, Mersenne, Lucas)")
        print("  • Consciousness/gnosis module")
        print("  • Extended structure constants")
        print("  • Python API integration")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
