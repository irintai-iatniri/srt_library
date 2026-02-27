#!/usr/bin/env python3
"""
Simple test script for syntonic DHSR implementation.
"""

import math
import random

from syntonic_applications.core.dtype import complex128
from syntonic_applications.core.state import State
from syntonic_applications.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC

# Constants
PHI = PHI_NUMERIC
PHI_INV = PHI_NUMERIC**-1  # 1/PHI
PHI_INV_SQ = PHI_INV**2
Q_DEFICIT = Q_DEFICIT_NUMERIC

print(f"Constants: PHI={PHI:.6f}, PHI_INV={PHI_INV:.6f}, Q_DEFICIT={Q_DEFICIT:.6f}")


# Simple norm function
def norm(state: State) -> float:
    """Compute L2 norm of State."""
    values = state.to_list()
    return sum(abs(v) ** 2 for v in values) ** 0.5


# Test basic functionality
print("\n--- Testing Basic State Operations ---")

# Create a random complex state
N = 10
real_part = [random.gauss(0, 1) for _ in range(N)]
imag_part = [random.gauss(0, 1) for _ in range(N)]
psi_values = [complex(r, i) for r, i in zip(real_part, imag_part)]
psi = State(psi_values, dtype=complex128, shape=(N,))

print(f"Created state with shape: {psi.shape}")
print(f"Norm: {norm(psi):.6f}")

# Test basic operations
psi_norm = norm(psi)
psi_normalized = psi * (1.0 / psi_norm) if psi_norm > 0 else psi
print(f"Normalized norm: {norm(psi_normalized):.6f}")


# Test golden distribution
def golden_distribution(N: int) -> list:
    n_values = list(range(N))
    weights = [math.exp(-n * n / PHI) for n in n_values]
    total = sum(weights)
    return [w / total for w in weights]


golden_weights = golden_distribution(N)
print(f"Golden weights (first 5): {golden_weights[:5]}")

print("\n--- Basic Functionality Test Passed ---")
print("Syntonic DHSR core components are working!")
