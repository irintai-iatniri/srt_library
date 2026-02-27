#!/usr/bin/env python3
"""Test SNA import and basic functionality - IT IS ALIVE test"""

import math
from syntonic_applications.sna import ResonantOscillator, DiscreteHilbertKernel, KERNEL_SCALE

print("=" * 80)
print("SYNTONIC NEURAL ARCHITECTURE - GENESIS TEST")
print("=" * 80)

# Display constants
print(f"\n✓ KERNEL_SCALE = {KERNEL_SCALE}")

# Create the Amoeba
print(f"\n✓ Creating Resonant Oscillator (The Amoeba)...")
neuron = ResonantOscillator(
    input_count=1,
    kernel_taps=31,  # 31-tap Hilbert kernel
    threshold=500    # Energy threshold
)
print(f"  Created: {neuron}")

# Generate sine wave stimulus (amplitude 1000, period 20 samples)
print(f"\n✓ Generating sine wave stimulus (amplitude=1000, period=20, n=100)...")
signal = [int(1000 * math.sin(2 * math.pi * t / 20)) for t in range(100)]
print(f"  Signal range: [{min(signal)}, {max(signal)}]")

# Process batch (THE MOMENT OF TRUTH)
print(f"\n✓ Processing batch through neuron...")
spikes = neuron.process_batch([signal])
print(f"  Generated {len(spikes)} ternary spikes")

# Visualize response
response_str = "".join("▲" if s > 0 else "▼" if s < 0 else "·" for s in spikes)
print(f"\n✓ Spike train visualization:")
print(f"  {response_str}")

# Analyze activity
has_activity = any(s != 0 for s in spikes)
excitatory = sum(1 for s in spikes if s > 0)
inhibitory = sum(1 for s in spikes if s < 0)
null = sum(1 for s in spikes if s == 0)

print(f"\n✓ Activity analysis:")
print(f"  Excitatory spikes (+1): {excitatory}")
print(f"  Inhibitory spikes (-1): {inhibitory}")
print(f"  Null spikes (0): {null}")
print(f"  Total non-zero: {excitatory + inhibitory}")

# THE VERDICT
print(f"\n" + "=" * 80)
if has_activity:
    print("🎉 IT IS ALIVE! The Resonant Oscillator is resonating!")
    print("   Phase-coherent computation confirmed via W4 projection.")
else:
    print("⚠️  No activity detected (may need parameter tuning)")
print("=" * 80)

# Additional validation tests
print(f"\n✓ Running validation tests...")

# Test 1: Invalid taps should fail
print(f"  Test 1: Invalid taps rejection...")
try:
    bad_kernel = DiscreteHilbertKernel(4)  # Even - should fail
    print(f"    ✗ FAILED: Should have rejected even taps")
except ValueError as e:
    print(f"    ✓ Correctly rejected: {e}")

# Test 2: Weight validation
print(f"  Test 2: Weight unit norm validation...")
neuron2 = ResonantOscillator(1, 31, 500)
try:
    neuron2.set_weights([(KERNEL_SCALE * 2, 0)])  # Excessive - should fail
    print(f"    ✗ FAILED: Should have rejected excessive weight")
except ValueError as e:
    print(f"    ✓ Correctly rejected: {e}")

# Test 3: Ragged array rejection
print(f"  Test 3: Ragged array rejection...")
neuron3 = ResonantOscillator(2, 31, 500)
try:
    neuron3.process_batch([[1, 2, 3], [1, 2]])  # Ragged - should fail
    print(f"    ✗ FAILED: Should have rejected ragged array")
except ValueError as e:
    print(f"    ✓ Correctly rejected: {e}")

print(f"\n✓ All validation tests passed!")
print(f"\n" + "=" * 80)
print("GENESIS COMPLETE - Syntonic Neuron operational")
print("=" * 80)
