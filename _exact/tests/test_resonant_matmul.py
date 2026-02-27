#!/usr/bin/env python3
"""
Test script for ResonantTensor matmul functionality.

This script tests the matrix multiplication operation of ResonantTensor,
which performs exact arithmetic in the Q(φ) lattice.
"""

import math

from syntonic_applications.nn.resonant_tensor import ResonantTensor


def test_basic_matmul():
    """Test basic matrix multiplication."""
    print("Testing basic ResonantTensor matmul...")

    # Create test matrices
    # A: 2x3 matrix (input)
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    A = ResonantTensor(a_data, shape=[2, 3])

    # B: 2x3 matrix (weights: out_features=2, in_features=3)
    # Since matmul does A @ B^T, and we want A @ B^T where B^T is 3x2
    # So B should be 2x3, B^T will be 3x2, A @ B^T = [2,3] @ [3,2] = [2,2]
    b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    B = ResonantTensor(b_data, shape=[2, 3])

    # Perform matrix multiplication: A @ B^T should give 2x2 result
    C = A.matmul(B)

    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"C shape: {C.shape}")

    # Expected result: A @ B^T
    # A = [[1, 2, 3], [4, 5, 6]]
    # B^T = [[7, 10], [8, 11], [9, 12]]
    # C[0][0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    # C[0][1] = 1*10 + 2*11 + 3*12 = 10 + 22 + 36 = 68
    # C[1][0] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    # C[1][1] = 4*10 + 5*11 + 6*12 = 40 + 55 + 72 = 167

    expected = [50.0, 68.0, 122.0, 167.0]
    actual = C.to_floats()

    print(f"Expected: {expected}")
    print(f"Actual: {actual}")

    # Check results (allowing some tolerance due to lattice snapping)
    for i, (exp, act) in enumerate(zip(expected, actual)):
        diff = abs(exp - act)
        if diff > 2.0:  # Allow reasonable tolerance
            print(f"ERROR: Element {i} mismatch: expected {exp}, got {act}")
            return False

    print("✓ Basic matmul test passed!")
    return True


def test_matmul_with_golden_ratio():
    """Test matmul with golden ratio values."""
    print("\nTesting matmul with golden ratio values...")

    PHI = (1 + math.sqrt(5)) / 2

    # Create matrices with golden ratio elements
    a_data = [1.0, PHI, 0.0, PHI, 1.0, 0.0]
    A = ResonantTensor(a_data, shape=[2, 3])

    b_data = [PHI, 0.0, 0.0, 1.0, 1.0, 0.0]
    B = ResonantTensor(b_data, shape=[2, 3])  # out_features=2, in_features=3

    C = A.matmul(B)

    print(f"A contains golden ratio: {PHI in a_data}")
    print(f"B contains golden ratio: {PHI in b_data}")
    print(f"Result shape: {C.shape}")

    # Check that result is valid
    actual = C.to_floats()
    print(f"Result values: {actual}")

    # All values should be finite
    if not all(math.isfinite(v) for v in actual):
        print("ERROR: Non-finite values in result")
        return False

    print("✓ Golden ratio matmul test passed!")
    return True


def test_matmul_shapes():
    """Test matmul with different tensor shapes."""
    print("\nTesting matmul with various shapes...")

    test_cases = [
        # (A_shape, B_shape, expected_C_shape)
        # A @ B^T where B is [out_features, in_features], B^T is [in_features, out_features]
        # So [m, n] @ [n, p] = [m, p], so B should be [p, n]
        ([2, 3], [4, 3], [2, 4]),  # A[2,3] @ B^T[3,4] = [2,4], so B=[4,3]
        ([1, 5], [1, 5], [1, 1]),  # A[1,5] @ B^T[5,1] = [1,1], so B=[1,5]
        ([3, 1], [3, 1], [3, 3]),  # A[3,1] @ B^T[1,3] = [3,3], so B=[3,1]
        ([4, 2], [5, 2], [4, 5]),  # A[4,2] @ B^T[2,5] = [4,5], so B=[5,2]
    ]

    for a_shape, b_shape, expected_c_shape in test_cases:
        # Create random tensors
        A = ResonantTensor.randn(a_shape)
        B = ResonantTensor.randn(b_shape)

        C = A.matmul(B)

        if C.shape != expected_c_shape:
            print(
                f"ERROR: Shape mismatch for {a_shape} @ {b_shape}^T: expected {expected_c_shape}, got {C.shape}"
            )
            return False

        print(f"✓ {a_shape} @ {b_shape}^T → {C.shape}")

    print("✓ Shape tests passed!")
    return True


def test_matmul_operator():
    """Test the @ operator for matmul."""
    print("\nTesting @ operator...")

    A = ResonantTensor.randn([3, 4])
    B = ResonantTensor.randn([2, 4])

    # Test both matmul method and @ operator
    C1 = A.matmul(B)
    C2 = A @ B

    # Results should be identical
    floats1 = C1.to_floats()
    floats2 = C2.to_floats()

    if len(floats1) != len(floats2):
        print("ERROR: Different result lengths")
        return False

    for f1, f2 in zip(floats1, floats2):
        if abs(f1 - f2) > 1e-10:
            print(f"ERROR: @ operator gives different result: {f1} vs {f2}")
            return False

    print("✓ @ operator test passed!")
    return True


def test_matmul_syntony():
    """Test that matmul preserves syntony computation."""
    print("\nTesting syntony preservation...")

    A = ResonantTensor.randn([5, 5])
    B = ResonantTensor.randn([5, 5])

    syntony_a = A.syntony
    syntony_b = B.syntony

    C = A.matmul(B)
    syntony_c = C.syntony

    print(f"A syntony: {syntony_a:.4f}")
    print(f"B syntony: {syntony_b:.4f}")
    print(f"C syntony: {syntony_c:.4f}")

    # All syntonies should be valid
    if not (
        0.0 <= syntony_a <= 1.0 and 0.0 <= syntony_b <= 1.0 and 0.0 <= syntony_c <= 1.0
    ):
        print("ERROR: Invalid syntony values")
        return False

    print("✓ Syntony preservation test passed!")
    return True


def main():
    """Run all tests."""
    print("Running ResonantTensor matmul tests...\n")

    tests = [
        test_basic_matmul,
        test_matmul_with_golden_ratio,
        test_matmul_shapes,
        test_matmul_operator,
        test_matmul_syntony,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
