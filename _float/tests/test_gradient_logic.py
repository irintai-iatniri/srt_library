#!/usr/bin/env python3
"""
Direct test of gradient extraction logic (Python simulation)
"""

def extract_gradient(signal):
    """Simulate Rust gradient extraction"""
    if not signal:
        return []
    
    gradient = []
    for i in range(len(signal) - 1):
        delta = signal[i+1] - signal[i]
        if delta > 0:
            gradient.append(1)
        elif delta < 0:
            gradient.append(-1)
        else:
            gradient.append(0)
    
    # Duplicate last value
    gradient.append(gradient[-1] if gradient else 0)
    return gradient

def compute_mad(gradient):
    """Compute Median Absolute Deviation"""
    if not gradient:
        return 0
    
    # Compute median
    sorted_g = sorted(gradient)
    n = len(sorted_g)
    if n % 2 == 0:
        median = (sorted_g[n//2 - 1] + sorted_g[n//2]) / 2
    else:
        median = sorted_g[n//2]
    
    # Compute absolute deviations
    deviations = sorted([abs(g - median) for g in gradient])
    
    # Return median of deviations
    if len(deviations) % 2 == 0:
        mad = (deviations[len(deviations)//2 - 1] + deviations[len(deviations)//2]) / 2
    else:
        mad = deviations[len(deviations)//2]
    
    return mad

# Test 1: Ramp
print("=== Ramp Signal ===")
signal = list(range(0, 100))
gradient = extract_gradient(signal)
print(f"Gradient: first 10 = {gradient[:10]}")
print(f"Gradient: last 10 = {gradient[-10:]}")
print(f"Gradient unique values: {set(gradient)}")
mad = compute_mad(gradient)
threshold = mad / 2
print(f"MAD = {mad}, Threshold = {threshold}")
print()

# Test 2: Sine wave
import math
print("=== Scaled Sine Wave ===")
length = 50
scale = 100000
signal = [int(scale * math.sin(2 * math.pi * i / length)) for i in range(length)]
gradient = extract_gradient(signal)
print(f"Gradient: first 10 = {gradient[:10]}")
print(f"Gradient unique values: {set(gradient)}")
mad = compute_mad(gradient)
threshold = mad / 2
print(f"MAD = {mad}, Threshold = {threshold}")
print()

# Test 3: Step function
print("=== Step Function ===")
signal = [0]*25 + [10000]*25 + [0]*25 + [10000]*25
gradient = extract_gradient(signal)
print(f"Gradient length: {len(gradient)}")
print(f"Gradient: {gradient}")
print(f"Gradient unique values: {set(gradient)}")
mad = compute_mad(gradient)
threshold = mad / 2
print(f"MAD = {mad}, Threshold = {threshold}")

# The issue: if threshold == 0, decomposition stops immediately!
print("\n" + "="*50)
print("Analysis:")
print("- Ramp: all gradients are 1, MAD=0, threshold=0 → STOPS")
print("- Sine: gradients vary, MAD should be > 0")
print("- Step: mostly 0 with few 1/-1, MAD might be 0 → STOPS")
print("="*50)
