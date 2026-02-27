"""
Tests for Sedenion (16-dimensional hypercomplex number) implementation.

Sedenions are the fourth step in the Cayley-Dickson construction:
    Real → Complex → Quaternion → Octonion → Sedenion
    1D      2D         4D          8D         16D

Key properties tested:
- Basic arithmetic (add, subtract, negate, scale)
- Cayley-Dickson multiplication
- Norm, conjugate, inverse
- Non-associativity and non-commutativity
- ZERO DIVISORS (the key feature distinguishing sedenions from octonions)
"""

import math

import pytest

from syntonic_applications.hypercomplex import (
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    Sedenion,
    sedenion,
)


class TestSedenionBasic:
    """Basic sedenion operations."""

    def test_creation(self):
        """Test sedenion creation."""
        s = Sedenion(
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        )
        assert s.e0 == 1.0
        assert s.e1 == 2.0
        assert s.e15 == 16.0

    def test_factory_function(self):
        """Test sedenion factory function."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert s.e0 == 1.0
        assert s.e15 == 16.0

    def test_factory_defaults(self):
        """Test factory function with default values."""
        s = sedenion(5)
        assert s.e0 == 5.0
        assert s.e1 == 0.0
        assert s.e15 == 0.0

    def test_real_property(self):
        """Test real part property."""
        s = sedenion(3.14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        assert s.real == 3.14

    def test_imag_property(self):
        """Test imaginary part property."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        imag = s.imag
        assert len(imag) == 15
        assert imag[0] == 2.0  # e1
        assert imag[14] == 16.0  # e15

    def test_octonion_halves(self):
        """Test extraction of octonion halves."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        low = s.octonion_low()
        high = s.octonion_high()
        assert low == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        assert high == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    def test_to_list(self):
        """Test conversion to list."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        lst = s.to_list()
        assert lst == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]

    def test_from_list(self):
        """Test creation from list."""
        s = Sedenion.from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        assert s.e0 == 1.0
        assert s.e15 == 16.0

    def test_from_list_wrong_length(self):
        """Test from_list with wrong length raises error."""
        with pytest.raises(ValueError):
            Sedenion.from_list([1, 2, 3])

    def test_from_octonions(self):
        """Test creation from two octonion halves."""
        s = Sedenion.from_octonions(
            [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]
        )
        assert s.e0 == 1.0
        assert s.e8 == 9.0

    def test_basis(self):
        """Test basis element creation."""
        for i in range(16):
            b = Sedenion.basis(i)
            for j in range(16):
                expected = 1.0 if i == j else 0.0
                assert b.to_list()[j] == expected

    def test_basis_invalid(self):
        """Test basis with invalid index raises error."""
        with pytest.raises(ValueError):
            Sedenion.basis(16)

    def test_one(self):
        """Test unit element."""
        one = Sedenion.one()
        assert one.e0 == 1.0
        assert all(x == 0.0 for x in one.imag)

    def test_zero(self):
        """Test zero element."""
        zero = Sedenion.zero()
        assert all(x == 0.0 for x in zero.to_list())


class TestSedenionBasisElements:
    """Test predefined basis elements."""

    def test_basis_elements_exist(self):
        """Test that all basis elements are defined."""
        basis = [S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15]
        for i, b in enumerate(basis):
            lst = b.to_list()
            assert lst[i] == 1.0
            for j in range(16):
                if j != i:
                    assert lst[j] == 0.0

    def test_s0_is_identity(self):
        """Test that S0 is the multiplicative identity."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        result = s * S0
        assert result.approx_eq(s, 1e-10)


class TestSedenionArithmetic:
    """Test sedenion arithmetic operations."""

    def test_addition(self):
        """Test sedenion addition."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = sedenion(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        c = a + b
        assert all(x == 17.0 for x in c.to_list())

    def test_subtraction(self):
        """Test sedenion subtraction."""
        a = sedenion(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
        b = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        c = a - b
        expected = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]
        for i, val in enumerate(c.to_list()):
            assert abs(val - expected[i]) < 1e-10

    def test_negation(self):
        """Test sedenion negation."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = -a
        for i, val in enumerate(b.to_list()):
            assert val == -(i + 1)

    def test_scalar_multiplication(self):
        """Test sedenion scalar multiplication."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = a.scale(2.0)
        for i, val in enumerate(b.to_list()):
            assert val == 2.0 * (i + 1)


class TestSedenionNorm:
    """Test sedenion norm operations."""

    def test_norm_sq(self):
        """Test norm squared."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        # Sum of squares: 1 + 4 + 9 + ... + 256 = sum(i^2 for i in 1..16) = 1496
        expected = sum(i * i for i in range(1, 17))
        assert abs(s.norm_sq() - expected) < 1e-10

    def test_norm(self):
        """Test Euclidean norm."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        expected = math.sqrt(sum(i * i for i in range(1, 17)))
        assert abs(s.norm() - expected) < 1e-10

    def test_zero_norm(self):
        """Test norm of zero sedenion."""
        zero = Sedenion.zero()
        assert zero.norm() == 0.0

    def test_unit_basis_norm(self):
        """Test that basis elements have unit norm."""
        for i in range(16):
            b = Sedenion.basis(i)
            assert abs(b.norm() - 1.0) < 1e-10


class TestSedenionConjugate:
    """Test sedenion conjugation."""

    def test_conjugate(self):
        """Test conjugate negates imaginary parts."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        c = s.conjugate()
        assert c.e0 == 1.0  # Real part unchanged
        assert c.e1 == -2.0  # Imaginary parts negated
        assert c.e15 == -16.0

    def test_double_conjugate(self):
        """Test that conj(conj(s)) = s."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert s.conjugate().conjugate().approx_eq(s, 1e-10)

    def test_s_times_conj_s(self):
        """Test that s * conj(s) equals norm_sq times identity."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        product = s * s.conjugate()
        # Should be real with value = norm_sq
        assert abs(product.e0 - s.norm_sq()) < 1e-10
        # Imaginary parts should be ~0
        for x in product.imag:
            assert abs(x) < 1e-10


class TestSedenionMultiplication:
    """Test sedenion multiplication (Cayley-Dickson)."""

    def test_identity_multiplication(self):
        """Test multiplication by identity."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        one = Sedenion.one()
        assert (s * one).approx_eq(s, 1e-10)
        assert (one * s).approx_eq(s, 1e-10)

    def test_basis_e1_squared(self):
        """Test that e1² = -1."""
        result = S1 * S1
        expected = sedenion(-1)
        assert result.approx_eq(expected, 1e-10)

    def test_all_imaginary_basis_squared(self):
        """Test that all imaginary basis elements square to -1."""
        for i in range(1, 16):
            bi = Sedenion.basis(i)
            result = bi * bi
            assert abs(result.e0 - (-1.0)) < 1e-10, f"e{i}² should equal -1"
            for j in range(1, 16):
                assert (
                    abs(result.to_list()[j]) < 1e-10
                ), f"e{i}² imaginary part should be 0"

    def test_non_commutativity(self):
        """Test that sedenions are non-commutative."""
        # e1 * e2 != e2 * e1 for most basis pairs
        result_12 = S1 * S2
        result_21 = S2 * S1
        # These should differ (commutator is non-zero)
        diff = result_12 - result_21
        assert diff.norm() > 1e-10, "Sedenions should be non-commutative"

    def test_commutator(self):
        """Test commutator computation."""
        a = sedenion(1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        b = sedenion(0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0)
        comm = Sedenion.commutator(a, b)
        # Commutator should be non-zero
        assert comm.norm() > 1e-10

    def test_non_associativity(self):
        """Test that sedenions are non-associative."""
        a = S1
        b = S2
        c = S8
        # (a*b)*c vs a*(b*c)
        left = (a * b) * c
        right = a * (b * c)
        # These should differ for some choices
        diff = left - right
        # Note: Not all triples are non-associative, so we test associator
        assoc = Sedenion.associator(a, b, c)
        # The associator may or may not be zero for this specific triple
        # Let's use a known non-associative triple
        a2 = S1
        b2 = S9
        c2 = S10
        assoc2 = Sedenion.associator(a2, b2, c2)
        # For a more robust test, we check that SOME triple is non-associative
        # by testing several combinations
        found_non_assoc = False
        for i in range(1, 8):
            for j in range(8, 12):
                for k in range(12, 16):
                    ai = Sedenion.basis(i)
                    bj = Sedenion.basis(j)
                    ck = Sedenion.basis(k)
                    assoc = Sedenion.associator(ai, bj, ck)
                    if assoc.norm() > 1e-10:
                        found_non_assoc = True
                        break
                if found_non_assoc:
                    break
            if found_non_assoc:
                break
        assert found_non_assoc, "Should find at least one non-associative triple"


class TestSedenionInverse:
    """Test sedenion inverse operations."""

    def test_inverse_basic(self):
        """Test basic inverse computation."""
        s = sedenion(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        inv = s.inverse()
        product = s * inv
        assert product.approx_eq(Sedenion.one(), 1e-10)

    def test_inverse_general(self):
        """Test inverse for general sedenion (non-zero-divisor)."""
        # Use a sedenion that's unlikely to be a zero divisor
        s = sedenion(
            1,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
        )
        inv = s.inverse()
        product = s * inv
        # Should be close to identity
        assert abs(product.e0 - 1.0) < 1e-6
        for x in product.imag:
            assert abs(x) < 1e-6

    def test_normalize(self):
        """Test normalization."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        n = s.normalize()
        assert abs(n.norm() - 1.0) < 1e-10

    def test_normalize_zero(self):
        """Test normalizing zero returns zero."""
        zero = Sedenion.zero()
        n = zero.normalize()
        assert n.norm() == 0.0


class TestSedenionZeroDivisors:
    """Test sedenion zero divisor detection and handling.

    This is the KEY distinguishing feature of sedenions vs octonions.
    Zero divisors are non-zero elements a, b such that a * b = 0.
    """

    def test_zero_divisor_pair_method(self):
        """Test the built-in zero divisor pair generator."""
        a, b = Sedenion.zero_divisor_pair()
        # Both should be non-zero
        assert a.norm() > 1e-10
        assert b.norm() > 1e-10
        # Product should be zero
        product = a * b
        assert (
            product.norm() < 1e-10
        ), f"Zero divisor product should be zero, got norm {product.norm()}"

    def test_canonical_zero_divisor(self):
        """Test a canonical zero divisor pair: (e3 + e10) * (e6 - e15) = 0."""
        a = sedenion(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)  # e3 + e10
        b = sedenion(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1)  # e6 - e15
        product = a * b
        assert (
            product.norm() < 1e-10
        ), f"Canonical zero divisor product should be zero, got {product.norm()}"

    def test_has_zero_divisor_with(self):
        """Test zero divisor detection method."""
        a, b = Sedenion.zero_divisor_pair()
        assert a.has_zero_divisor_with(b), "Should detect zero divisor pair"

    def test_has_zero_divisor_with_false(self):
        """Test that non-zero-divisor pairs are not flagged."""
        a = sedenion(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        b = sedenion(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert not a.has_zero_divisor_with(
            b
        ), "e0 and e1 should not be zero divisor pair"

    def test_is_potential_zero_divisor(self):
        """Test potential zero divisor detection."""
        # e0 + e8 has equal-norm halves, so is a potential zero divisor
        a = sedenion(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        assert a.is_potential_zero_divisor()

        # Pure real is not a potential zero divisor
        b = sedenion(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert not b.is_potential_zero_divisor()

    def test_composition_property_fails(self):
        """Test that ||ab|| != ||a||·||b|| for zero divisors.

        This is the fundamental property that fails for sedenions.
        """
        a, b = Sedenion.zero_divisor_pair()
        norm_a = a.norm()
        norm_b = b.norm()
        norm_product = (a * b).norm()

        # For a division algebra: ||ab|| = ||a||·||b||
        # For sedenions with zero divisors: ||ab|| < ||a||·||b||
        expected = norm_a * norm_b
        assert (
            norm_product < expected * 0.01
        ), f"Zero divisor: ||ab||={norm_product} should be << ||a||·||b||={expected}"


class TestSedenionDot:
    """Test sedenion dot product."""

    def test_dot_product(self):
        """Test dot product computation."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = sedenion(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        dot = Sedenion.dot(a, b)
        expected = sum(range(1, 17))  # 1+2+...+16 = 136
        assert abs(dot - expected) < 1e-10

    def test_dot_self_equals_norm_sq(self):
        """Test that dot(s, s) = norm_sq(s)."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert abs(Sedenion.dot(s, s) - s.norm_sq()) < 1e-10


class TestSedenionPure:
    """Test pure (imaginary) sedenion detection."""

    def test_is_pure_true(self):
        """Test pure sedenion detection."""
        s = sedenion(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        assert s.is_pure()

    def test_is_pure_false(self):
        """Test non-pure sedenion."""
        s = sedenion(1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        assert not s.is_pure()


class TestSedenionRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ returns valid string."""
        s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        repr_str = repr(s)
        assert "Sedenion" in repr_str
        assert "1.0" in repr_str or "1.00" in repr_str


class TestSedenionApproxEq:
    """Test approximate equality."""

    def test_approx_eq_true(self):
        """Test approximate equality for equal sedenions."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert a.approx_eq(b, 1e-10)

    def test_approx_eq_within_tolerance(self):
        """Test approximate equality within tolerance."""
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = sedenion(1.0001, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert a.approx_eq(b, 0.001)
        assert not a.approx_eq(b, 1e-5)


class TestSedenionDivision:
    """Test sedenion division."""

    def test_division_by_scalar(self):
        """Test division by a real sedenion."""
        s = sedenion(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
        two = sedenion(2)
        result = s / two
        expected = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        assert result.approx_eq(expected, 1e-10)

    def test_division_warning(self):
        """Division by zero divisor may not work as expected.

        This is documented behavior - division is not well-defined for zero divisors.
        """
        # This test just verifies the code doesn't crash
        a, b = Sedenion.zero_divisor_pair()
        # Division by a zero divisor element may give unexpected results
        # but shouldn't crash
        try:
            result = a / b
            # Result may be NaN or unexpected values
        except ZeroDivisionError:
            pass  # This is also acceptable


class TestCayleyDicksonConstruction:
    """Test properties specific to Cayley-Dickson construction."""

    def test_octonion_embedding(self):
        """Test that sedenions with zero high part multiply like octonions."""
        # When high octonion part is zero, multiplication should follow octonion rules
        a = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0)
        b = sedenion(8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        product = a * b
        # High part should remain zero
        high = product.octonion_high()
        for x in high:
            assert abs(x) < 1e-10

    def test_power_associativity(self):
        """Test power-associativity: (s*s)*s = s*(s*s).

        Unlike general associativity, sedenions ARE power-associative.
        """
        s = sedenion(
            1,
            0.5,
            0.3,
            0.2,
            0.1,
            0.05,
            0.03,
            0.02,
            0.01,
            0.005,
            0.003,
            0.002,
            0.001,
            0.0005,
            0.0003,
            0.0002,
        )
        s2 = s * s
        left = s2 * s
        right = s * s2
        assert left.approx_eq(right, 1e-8), "Sedenions should be power-associative"
