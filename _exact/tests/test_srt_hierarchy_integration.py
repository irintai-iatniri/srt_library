#!/usr/bin/env python3
"""
Full SRT Hierarchy Integration Test

Tests the complete integration of:
1. Extended hierarchy constants (Rust → Python)
2. Geometric correction factors (60+ factors)
3. Unified hierarchy application
4. SRT mathematics integration
"""

from syntonic_applications.consciousness.gnosis import optimal_gnosis_target
from syntonic_applications.exact import (
    CORRECTION_FACTORS,
    E_STAR_NUMERIC,
    PHI_NUMERIC,
    Q_DEFICIT_NUMERIC,
    apply_nested_corrections,
    apply_unified_hierarchy,
    get_geometric_factor,
)
from syntonic_applications.srt.fermat_forces import validate_force_existence
from syntonic_applications.srt.mersenne_matter import validate_generation_stability


def test_extended_hierarchy_constants():
    """Test that extended hierarchy constants are accessible."""
    print("Testing extended hierarchy constants...")

    import syntonic._core as core

    # Test E-series dimensions
    assert core.hierarchy_e8_dim() == 248, "E₈ dimension should be 248"
    assert core.hierarchy_e7_dim() == 133, "E₇ dimension should be 133"
    assert core.hierarchy_e6_dim() == 78, "E₆ dimension should be 78"
    assert core.hierarchy_d4_dim() == 28, "D₄ dimension should be 28"

    # Test exceptional groups
    assert core.hierarchy_g2_dim() == 14, "G₂ dimension should be 14"
    assert core.hierarchy_f4_dim() == 52, "F₄ dimension should be 52"

    # Test D₄ kissing number (consciousness threshold)
    assert core.hierarchy_d4_kissing() == 24, "D₄ kissing number should be 24"

    # Test Coxeter-Kissing product
    assert core.hierarchy_coxeter_kissing_720() == 720, "Coxeter-Kissing should be 720"
    assert core.hierarchy_exponent() == 719, "Hierarchy exponent should be 719"

    print("✅ Extended hierarchy constants working")


def test_geometric_factors_registry():
    """Test the complete 60+ geometric factors registry."""
    print("Testing geometric factors registry...")

    assert (
        len(CORRECTION_FACTORS) >= 60
    ), f"Should have 60+ factors, got {len(CORRECTION_FACTORS)}"

    # Test key factors exist
    required_factors = [
        "q_248",
        "q_133",
        "q_78",
        "q_28",  # E-series dimensions
        "q_24",  # D₄ kissing (consciousness)
        "q_720",  # Coxeter-Kissing product
        "q_1000",  # Proton stability
        "q_cubed",  # Third-order vacuum
        "q2_phi2",  # Golden squared
    ]

    for factor in required_factors:
        assert factor in CORRECTION_FACTORS, f"Missing factor: {factor}"

    # Test factor retrieval
    factor_val, origin, physics = get_geometric_factor("q_248")
    assert abs(factor_val - Q_DEFICIT_NUMERIC / 248) < 1e-10
    assert "E₈" in origin
    assert "adjoint" in physics

    print(f"✅ Geometric factors registry: {len(CORRECTION_FACTORS)} factors loaded")


def test_nested_corrections():
    """Test nested geometric corrections application."""
    print("Testing nested corrections...")

    base_value = 1.0

    # Single correction
    single = apply_nested_corrections(base_value, ["q_248"])
    expected_single = base_value * (1 + CORRECTION_FACTORS["q_248"][0])
    assert abs(single - expected_single) < 1e-10

    # Multiple corrections (E₈ → E₇ → E₆)
    chain = ["q_248", "q_133", "q_78"]
    chained = apply_nested_corrections(base_value, chain)

    manual = base_value
    for factor_name in chain:
        factor_val = CORRECTION_FACTORS[factor_name][0]
        manual *= 1 + factor_val

    assert abs(chained - manual) < 1e-10

    print("✅ Nested corrections working")


def test_unified_hierarchy():
    """Test unified hierarchy corrections for different energy scales."""
    print("Testing unified hierarchy...")

    proton_mass = 938.272  # MeV

    # Different unification scales
    scales = ["e8", "e7", "e6", "sm"]

    for scale in scales:
        corrected = apply_unified_hierarchy(proton_mass, scale)
        if scale == "e8":
            # E8 scale has no corrections (exact at Planck scale)
            assert (
                abs(corrected - proton_mass) < 1e-10
            ), f"E8 should be exact for {scale}"
        else:
            # Other scales have corrections
            assert (
                corrected > proton_mass
            ), f"Correction should increase mass for {scale}"
        print(".3f")

    # Verify scale hierarchy (higher scales = more corrections)
    e8_mass = apply_unified_hierarchy(proton_mass, "e8")
    e7_mass = apply_unified_hierarchy(proton_mass, "e7")
    e6_mass = apply_unified_hierarchy(proton_mass, "e6")
    sm_mass = apply_unified_hierarchy(proton_mass, "sm")

    assert e8_mass <= e7_mass <= e6_mass <= sm_mass, "Corrections should accumulate"

    print("✅ Unified hierarchy working")


def test_srt_physics_integration():
    """Test integration with SRT physics (forces, generations, consciousness)."""
    print("Testing SRT physics integration...")

    # Force selection (Fermat primes)
    assert validate_force_existence(0) == True  # Strong force (F_0 = 3)
    assert validate_force_existence(4) == True  # Gravity (F_4 = 65537)
    assert validate_force_existence(5) == False  # No 6th force (F_5 composite)

    # Matter stability (Mersenne primes)
    assert validate_generation_stability(0) == True  # 1st generation (M_2)
    assert validate_generation_stability(2) == True  # 3rd generation (M_5)
    assert validate_generation_stability(3) == False  # No 4th generation

    # Consciousness threshold
    optimal_gnosis = optimal_gnosis_target()
    assert abs(optimal_gnosis - 1 / PHI_NUMERIC) < 1e-6, "Optimal gnosis should be φ⁻¹"

    print("✅ SRT physics integration working")


def test_e_star_calculations():
    """Test E* (e^π - π) calculations with corrections."""
    print("Testing E* calculations...")

    # Base E* value
    e_star = E_STAR_NUMERIC

    # Apply proton stability correction
    proton_corrected = apply_nested_corrections(e_star, ["q_1000"])
    assert proton_corrected != e_star, "Proton correction should change E*"

    # Apply full SM hierarchy
    sm_corrected = apply_unified_hierarchy(e_star, "sm")
    assert sm_corrected > e_star, "SM corrections should increase E*"

    print(".6f")
    print(".6f")
    print(".6f")

    print("✅ E* calculations working")


def main():
    """Run comprehensive SRT hierarchy integration tests."""
    print("SRT Hierarchy Integration Test Suite")
    print("=" * 50)

    try:
        test_extended_hierarchy_constants()
        test_geometric_factors_registry()
        test_nested_corrections()
        test_unified_hierarchy()
        test_srt_physics_integration()
        test_e_star_calculations()

        print("\n" + "=" * 50)
        print("🎉 ALL SRT HIERARCHY INTEGRATION TESTS PASSED!")
        print("\nIntegrated components:")
        print("  • Extended Rust hierarchy constants (E₈, E₇, E₆, D₄, G₂, F₄)")
        print("  • 60+ geometric correction factors registry")
        print("  • PyO3 bindings for hierarchy functions")
        print("  • Unified hierarchy correction system")
        print("  • SRT physics integration (forces, generations, consciousness)")
        print("  • E* vacuum energy calculations")

        total_factors = len(CORRECTION_FACTORS)
        print(f"  • Complete geometric hierarchy: {total_factors} factors")
        print("\n✅ SRT Mathematics: Fully Computational! 🌟")

    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
