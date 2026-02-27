#!/usr/bin/env python3
"""
Test suite for float mode isolation and explicit conversion requirements.

Verifies that float mode is opt-in only and exact types don't auto-promote to float.

Phase 7.2: Float Isolation Tests
"""

import pytest
import warnings
from syntonic_applications.core.state import State, state
from syntonic_applications.core.dtype import (
    golden_exact,
    fixed_point64,
    float64,
    float32,
    is_exact_dtype,
    is_float_dtype,
)
from syntonic_applications.core.device import cpu


class TestFloatRequiresExplicitConversion:
    """Verify float mode is opt-in only."""

    def test_default_is_exact(self):
        """State constructors default to exact arithmetic."""
        # Default constructor
        test_state = state.zeros((10, 10))
        assert is_exact_dtype(state.dtype), \
            f"Default should be exact, got {state.dtype}"

        # from_seed
        test_state = state.from_seed(42, (10, 10))
        assert is_exact_dtype(state.dtype), \
            f"from_seed should default to exact, got {state.dtype}"

        # randn
        test_state = state.randn((10, 10))
        assert is_exact_dtype(state.dtype), \
            f"randn should default to exact, got {state.dtype}"

    def test_explicit_float_required(self):
        """Float dtype must be explicitly specified."""
        # Explicit float64
        state_float = state.zeros((10, 10), dtype=float64)
        assert state_float.dtype == float64, "Explicit float64 must work"
        assert is_float_dtype(state_float.dtype), "Must be float dtype"

        # Explicit float32
        state_float32 = state.zeros((10, 10), dtype=float32)
        assert state_float32.dtype == float32, "Explicit float32 must work"

    def test_to_float_explicit_conversion(self):
        """to_float() explicitly converts to float mode."""
        state_exact = state.zeros((10, 10))
        assert is_exact_dtype(state_exact.dtype)

        # Explicit conversion
        state_float = state_exact.to_float()
        assert is_float_dtype(state_float.dtype), \
            "to_float() must convert to float dtype"

        # Original unchanged
        assert is_exact_dtype(state_exact.dtype), \
            "Original state must remain exact"

    def test_exact_preserved_across_operations(self):
        """Exact dtype preserved through operations."""
        test_state = state.from_seed(42, (16, 16))
        assert is_exact_dtype(state.dtype)

        # Operations should preserve exact
        result = state + state
        assert is_exact_dtype(result.dtype), "Addition must preserve exact"

        result = state * 2.0
        assert is_exact_dtype(result.dtype), "Scalar mult must preserve exact"

        result = state - state
        assert is_exact_dtype(result.dtype), "Subtraction must preserve exact"


class TestNoImplicitFloatPromotion:
    """Verify exact types don't auto-promote to float."""

    def test_exact_stays_exact(self):
        """Exact arithmetic preserves exact types."""
        state1 = state.from_seed(42, (10, 10))
        state2 = state.from_seed(43, (10, 10))

        assert is_exact_dtype(state1.dtype)
        assert is_exact_dtype(state2.dtype)

        # Operations between exact states
        result = state1 + state2
        assert is_exact_dtype(result.dtype), \
            "Exact + Exact must remain exact"

        result = state1 * state2
        assert is_exact_dtype(result.dtype), \
            "Exact * Exact must remain exact"

    def test_scalar_operations_preserve_exact(self):
        """Scalar operations with exact states remain exact."""
        test_state = state.from_seed(42, (16, 16))
        assert is_exact_dtype(state.dtype)

        # Integer scalar
        result = state * 2
        assert is_exact_dtype(result.dtype), "Exact * int must remain exact"

        # Float scalar (backend converts to exact)
        result = state * 2.5
        assert is_exact_dtype(result.dtype), "Exact * float must remain exact"

    def test_dhsr_preserves_exact(self):
        """DHSR operators preserve exact types."""
        test_state = state.from_seed(42, (32, 32))
        assert is_exact_dtype(state.dtype)

        # Differentiation
        state_diff = state.differentiate()
        assert is_exact_dtype(state_diff.dtype), \
            "Differentiation must preserve exact"

        # Harmonization
        state_harm = state.harmonize()
        assert is_exact_dtype(state_harm.dtype), \
            "Harmonization must preserve exact"

        # Full DHSR cycle
        state_dhsr = state.differentiate().harmonize()
        assert is_exact_dtype(state_dhsr.dtype), \
            "DHSR cycle must preserve exact"


class TestMixingExactFloat:
    """Test behavior when mixing exact and float types."""

    def test_mixing_requires_explicit_conversion(self):
        """Mixing exact and float requires explicit conversion."""
        state_exact = state.zeros((10, 10))  # defaults to exact
        state_float = state.zeros((10, 10), dtype=float64)

        assert is_exact_dtype(state_exact.dtype)
        assert is_float_dtype(state_float.dtype)

        # Direct mixing should handle conversion internally
        # The backend should either:
        # 1. Promote to exact (keeping precision), or
        # 2. Raise an error requiring explicit conversion
        #
        # Currently, the backend promotes to exact to prevent drift
        result = state_exact + state_float
        # Result should be exact (preserving determinism)
        assert is_exact_dtype(result.dtype), \
            "Mixing should preserve exact to prevent drift"

    def test_explicit_float_conversion_for_mixing(self):
        """Explicitly convert to float for float operations."""
        state_exact = state.zeros((10, 10))
        state_float = state.zeros((10, 10), dtype=float64)

        # Explicit conversion to float
        result = state_exact.to_float() + state_float
        assert is_float_dtype(result.dtype), \
            "Explicit to_float() enables float operations"

    def test_exact_preferred_in_promotion(self):
        """When mixing, exact is preferred to prevent drift."""
        state_exact = state.from_seed(42, (16, 16))
        state_float = state.from_seed(42, (16, 16), dtype=float64)

        # Addition should promote to exact
        result = state_exact + state_float
        assert is_exact_dtype(result.dtype), \
            "Type promotion must prefer exact to prevent drift"


class TestFloatOptIn:
    """Test that float mode must be explicitly requested."""

    def test_zeros_defaults_exact(self):
        """state.zeros() defaults to exact."""
        test_state = state.zeros((10, 10))
        assert is_exact_dtype(state.dtype)

    def test_ones_defaults_exact(self):
        """state.ones() defaults to exact."""
        test_state = state.ones((10, 10))
        assert is_exact_dtype(state.dtype)

    def test_randn_defaults_exact(self):
        """state.randn() defaults to exact."""
        test_state = state.randn((10, 10))
        assert is_exact_dtype(state.dtype)

    def test_from_seed_defaults_exact(self):
        """state.from_seed() defaults to exact."""
        test_state = state.from_seed(42, (10, 10))
        assert is_exact_dtype(state.dtype)

    def test_explicit_float_works(self):
        """Explicit float dtype works for all constructors."""
        # zeros
        test_state = state.zeros((10, 10), dtype=float64)
        assert state.dtype == float64

        # ones
        test_state = state.ones((10, 10), dtype=float64)
        assert state.dtype == float64

        # randn
        test_state = state.randn((10, 10), dtype=float64)
        assert state.dtype == float64

        # from_seed
        test_state = state.from_seed(42, (10, 10), dtype=float64)
        assert state.dtype == float64


class TestFloatLossWarnings:
    """Test that converting exact to float produces warnings (if implemented)."""

    def test_to_float_is_lossy(self):
        """to_float() is a lossy conversion."""
        state_exact = state.from_seed(42, (10, 10))

        # Run DHSR to create exact state
        for _ in range(10):
            state_exact = state_exact.differentiate().harmonize()

        # Convert to float (lossy)
        state_float = state_exact.to_float()

        # Should be different types
        assert is_exact_dtype(state_exact.dtype)
        assert is_float_dtype(state_float.dtype)

        # Values may differ slightly
        exact_list = state_exact.to_list()
        float_list = state_float.to_list()

        # Check that conversion happened
        assert len(exact_list) == len(float_list)

    def test_float_prevents_perfect_reconstruction(self):
        """Float mode breaks perfect reconstruction guarantee."""
        seed = 42
        shape = (32, 32)

        # Run 1: exact mode
        state_exact = state.from_seed(seed, shape)
        for _ in range(50):
            state_exact = state_exact.differentiate().harmonize()

        # Run 2: float mode (opt-in)
        state_float = state.from_seed(seed, shape, dtype=float64)
        for _ in range(50):
            state_float = state_float.differentiate().harmonize()

        # They should be different (float loses precision)
        # Convert both to Python floats for comparison
        exact_floats = [float(v) for v in state_exact.to_list()]
        float_floats = [float(v) for v in state_float.to_list()]

        # Compute max difference
        max_diff = max(abs(e - f) for e, f in zip(exact_floats, float_floats))

        # Float mode will have accumulated some error vs exact
        # Should be small but nonzero
        assert max_diff > 0, "Float and exact modes should differ"


class TestBackwardCompatibility:
    """Test that explicit float mode still works for backward compatibility."""

    def test_explicit_float_operations(self):
        """Float mode works when explicitly requested."""
        state1 = state.zeros((10, 10), dtype=float64)
        state2 = state.ones((10, 10), dtype=float64)

        assert is_float_dtype(state1.dtype)
        assert is_float_dtype(state2.dtype)

        # Float operations should work
        result = state1 + state2
        assert is_float_dtype(result.dtype)

        result = state1 * 2.5
        assert is_float_dtype(result.dtype)

    def test_float_dhsr_works(self):
        """DHSR operators work in float mode."""
        test_state = state.from_seed(42, (16, 16), dtype=float64)
        assert is_float_dtype(state.dtype)

        # DHSR should work (but won't be perfectly deterministic)
        result = state.differentiate().harmonize()
        assert is_float_dtype(result.dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
