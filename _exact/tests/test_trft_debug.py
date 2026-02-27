#!/usr/bin/env python3
"""
Debug TRFT implementation - check intermediate values
"""

from syntonic._core import py_generate_resonance_ladder, py_ternary_decompose
import math

# Create a simple ramp signal (easier to debug than sine)
print("=== Test with Ramp Signal ===")
signal = list(range(0, 100))  # 0, 1, 2, ..., 99
print(f"Signal: first 10 = {signal[:10]}")
print(f"Signal: last 10 = {signal[-10:]}")

# Try decomposition
hints = []
layers = py_ternary_decompose(
    signal,
    hints,
    max_layers=5,
    energy_threshold_dominant=0.20,
    energy_threshold_subtle=0.05
)

print(f"Found {len(layers)} layers")
print()

# Try with a stronger signal (scaled up)
print("=== Test with Scaled Sine Wave ===")
length = 50
scale = 100000  # Much larger scale
signal = [int(scale * math.sin(2 * math.pi * i / length)) for i in range(length)]
print(f"Signal: min={min(signal)}, max={max(signal)}")
print(f"Signal: first 10 = {signal[:10]}")

layers = py_ternary_decompose(
    signal,
    hints,
    max_layers=5,
    energy_threshold_dominant=0.20,
    energy_threshold_subtle=0.05
)

print(f"Found {len(layers)} layers")
for i, (a, b, h, gx0, gy0, energy) in enumerate(layers):
    print(f"  Layer {i}: triple=({a},{b},{h}), gx0={gx0}, gy0={gy0}, energy={energy:.3f}")
print()

# Try with step function (simple pattern)
print("=== Test with Step Function ===")
signal = [0]*25 + [10000]*25 + [0]*25 + [10000]*25
print(f"Signal: {signal}")

layers = py_ternary_decompose(
    signal,
    hints,
    max_layers=5,
    energy_threshold_dominant=0.10,  # Lower threshold
    energy_threshold_subtle=0.02
)

print(f"Found {len(layers)} layers")
for i, (a, b, h, gx0, gy0, energy) in enumerate(layers):
    print(f"  Layer {i}: triple=({a},{b},{h}), gx0={gx0}, gy0={gy0}, energy={energy:.3f}")
