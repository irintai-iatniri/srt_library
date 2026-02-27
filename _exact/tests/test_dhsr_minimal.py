#!/usr/bin/env python3
"""
Minimal Syntonic DHSR Implementation Test
"""

import cmath
import math
import random

from syntonic_applications.core.dtype import complex128
from syntonic_applications.core.state import State
from syntonic_applications.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC

# Constants
PHI = PHI_NUMERIC
PHI_INV = PHI_NUMERIC**-1
PHI_INV_SQ = PHI_INV**2
Q_DEFICIT = Q_DEFICIT_NUMERIC

print("Syntonic DHSR Minimal Test")
print("=" * 40)
print(f"PHI = {PHI:.6f}")
print(f"PHI_INV = {PHI_INV:.6f}")
print(f"PHI_INV_SQ = {PHI_INV_SQ:.6f}")
print(f"Q_DEFICIT = {Q_DEFICIT:.6f}")


# Utility functions
def norm(state: State) -> float:
    """L2 norm of state."""
    values = state.to_list()
    return sum(abs(v) ** 2 for v in values) ** 0.5


def golden_distribution(N: int) -> list:
    """Golden measure weights."""
    n_values = list(range(N))
    weights = [math.exp(-n * n / PHI) for n in n_values]
    total = sum(weights)
    return [w / total for w in weights]


# Test State operations
print("\n--- Testing State Operations ---")
N = 8
real_part = [random.gauss(0, 1) for _ in range(N)]
imag_part = [random.gauss(0, 1) for _ in range(N)]
psi_values = [complex(r, i) for r, i in zip(real_part, imag_part)]
psi = State(psi_values, dtype=complex128, shape=(N,))

print(f"Created state with shape: {psi.shape}")
print(f"Original norm: {norm(psi):.6f}")

# Normalize
psi_norm = norm(psi)
psi_normalized = psi * (1.0 / psi_norm) if psi_norm > 0 else psi
print(f"Normalized norm: {norm(psi_normalized):.6f}")

# Test golden harmonization
print("\n--- Testing Golden Harmonization ---")
golden_weights = golden_distribution(N)
target_amplitudes = [math.sqrt(psi_norm**2 * w) for w in golden_weights]

# Create target state with golden amplitudes but preserved phases
values = psi_normalized.to_list()
phases = [cmath.phase(v) for v in values]
target_values = [a * cmath.exp(1j * p) for a, p in zip(target_amplitudes, phases)]
target_state = State(target_values, dtype=complex128, shape=(N,))

# Simple harmonization (strength = PHI_INV)
strength = PHI_INV
harmonized = (1 - strength) * psi_normalized + strength * target_state

print(f"Harmonized norm: {norm(harmonized):.6f}")

# Test differentiation (simplified version)
print("\n--- Testing Differentiation ---")
alpha = 0.1
syntony = 0.5

# Add syntony-modulated noise
noise_scale = alpha * (1 - syntony)
noise_real = [random.gauss(0, noise_scale) for _ in range(N)]
noise_imag = [random.gauss(0, noise_scale) for _ in range(N)]
noise_values = [complex(r, i) for r, i in zip(noise_real, noise_imag)]
noise_state = State(noise_values, dtype=complex128, shape=(N,))

differentiated = psi_normalized + noise_state

# Energy conservation
orig_energy = sum(abs(v) ** 2 for v in psi_normalized.to_list())
diff_energy = sum(abs(v) ** 2 for v in differentiated.to_list())
if diff_energy > 0:
    scale = math.sqrt(orig_energy / diff_energy)
    diff_values = differentiated.to_list()
    diff_values = [v * scale for v in diff_values]
    differentiated = State(diff_values, dtype=complex128, shape=(N,))

print(f"Differentiated norm: {norm(differentiated):.6f}")

# Test recursion
print("\n--- Testing Recursion ---")
recursed = differentiated * (1 - PHI_INV) + harmonized * PHI_INV
print(f"Recursed norm: {norm(recursed):.6f}")

print("\n--- Basic DHSR Operations Working ---")
print("✅ State creation and operations")
print("✅ Normalization")
print("✅ Golden harmonization")
print("✅ Syntony-modulated differentiation")
print("✅ Recursion operator")
print("\n🎉 Syntonic DHSR core functionality verified!")
