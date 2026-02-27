#!/usr/bin/env python3
"""
Quick verification test for GnosticOuroboros shape fixes.
Uses a smaller dimension for faster initialization.
"""

from syntonic_applications.nn.resonant_tensor import ResonantTensor

# Test the individual components before full model

print("=" * 60)
print("Testing Shape-Fixed Components")
print("=" * 60)

# 1. Test ResonantTensor mean operation
print("\n1. Testing ResonantTensor.mean()...")
x = ResonantTensor.randn([4, 64])  # [seq, dim]
print(f"   Input shape: {x.shape}")
try:
    x_mean = x.mean(dim=0)
    print(f"   Mean shape: {x_mean.shape}")
    assert x_mean.shape == [64], f"Expected [64], got {x_mean.shape}"
    print("   ✓ Mean operation works")
except Exception as e:
    print(f"   ✗ Mean failed: {e}")

# 2. Test add_bias for broadcasting
print("\n2. Testing add_bias for broadcasting...")
try:
    x_copy = ResonantTensor(x.to_floats(), list(x.shape))  # Clone
    pull = ResonantTensor.randn([64])
    print(f"   Tensor: {x_copy.shape}, Bias: {pull.shape}")

    # Test add_bias (in-place broadcast)
    x_copy.add_bias(pull)
    print(f"   After add_bias: {x_copy.shape}")
    assert x_copy.shape == [4, 64]
    print("   ✓ add_bias broadcast works")
except Exception as e:
    print(f"   ✗ Broadcast failed: {e}")

# 3. Test ResonantLinear shape handling
print("\n3. Testing ResonantLinear projections...")
from syntonic_applications.nn.layers.resonant_linear import ResonantLinear

try:
    proj = ResonantLinear(64, 256)  # dim -> dim*4
    y = proj.forward(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == [4, 256], f"Expected [4, 256], got {y.shape}"
    print("   ✓ Projection works")
except Exception as e:
    print(f"   ✗ Projection failed: {e}")

# 4. Test evolver with matching template
print("\n4. Testing evolver with correct template size...")
from syntonic_applications.resonant.retrocausal import create_retrocausal_evolver

try:
    template = ResonantTensor([0.0] * 256, [256])  # dim*4
    evolver = create_retrocausal_evolver(template=template, population_size=8)

    # Create input matching template
    test_input = ResonantTensor.randn([256])
    harmonized = evolver.harmonize(test_input._inner)
    result = ResonantTensor._wrap(harmonized)
    print(
        f"   Template: {template.shape}, Input: {test_input.shape}, Output: {result.shape}"
    )
    assert result.shape == [256]
    print("   ✓ Evolver harmonize works with matching shapes")
except Exception as e:
    print(f"   ✗ Evolver failed: {e}")

print("\n" + "=" * 60)
print("Component tests complete!")
print("=" * 60)
print("\nNote: Full GnosticOuroboros init takes ~8 minutes due to 18 plane evolvers.")
print("The shape fixes should now allow the forward pass to work correctly.")
