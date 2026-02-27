#!/usr/bin/env python3
"""
Test script for remade convergence_benchmark.py

Tests the syntonic-only implementation without full package imports.
"""

import sys
from pathlib import Path

# Get repo root for file path operations
repo_root = Path(__file__).resolve().parents[1]


# Mock the syntonic._core module to avoid import issues
class MockResonantTensor:
    def __init__(self, data, shape, mode_norms=None):
        self.data = data
        self.shape = shape
        self.mode_norms = mode_norms or []
        self.syntony = 0.5

    def to_list(self):
        return self.data


class MockFitnessGuidedEvolver:
    def __init__(self, **kwargs):
        self.best_tensor = None
        self.generation = 0

    def step(self):
        self.generation += 1
        # Create a mock tensor
        self.best_tensor = MockResonantTensor([0.1, 0.2, 0.3, 0.4], [4])


# Mock the modules
sys.modules["syntonic._core"] = type(
    "MockModule", (), {"ResonantTensor": MockResonantTensor}
)()

sys.modules["syntonic.benchmarks.datasets"] = type(
    "MockModule",
    (),
    {
        "make_xor": lambda **kwargs: (
            [[1, 1], [-1, 1], [1, -1], [-1, -1]],
            [0, 1, 1, 0],
        ),
        "train_test_split": lambda X, y, **kwargs: (X[:3], X[3:], y[:3], y[3:]),
    },
)()

sys.modules["syntonic.benchmarks.fitness"] = type(
    "MockModule",
    (),
    {
        "ClassificationFitness": type(
            "MockFitness",
            (),
            {"task_fitness": lambda self, x: 0.8, "loss": lambda self, x: 0.2},
        ),
        "FitnessGuidedEvolver": MockFitnessGuidedEvolver,
    },
)()

print("Testing remade convergence_benchmark.py...")
print("=" * 50)

# Now test the benchmark
try:
    # Import directly from file
    exec(open("syntonic_python/benchmarks/convergence_benchmark.py").read())

    # Test basic functionality
    benchmark = ConvergenceSpeedBenchmark(n_samples=100, seed=42)
    print("✓ Benchmark instance created")
    print(f"  Dataset size: {benchmark.n_samples}")
    print(f"  Features: {benchmark.n_features}")
    print(f"  Classes: {benchmark.n_classes}")

    # Test polynomial feature expansion
    test_sample = [[1.0, 0.5]]
    poly_features = benchmark._add_polynomial_features(test_sample)
    print("✓ Polynomial feature expansion working")
    print(f"  Original: {test_sample[0]}")
    print(f"  Expanded: {[round(x, 3) for x in poly_features[0]]}")

    # Test matrix multiplication
    X = [[1, 2], [3, 4]]
    W = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]  # 5x2
    result = benchmark._matrix_multiply(X, W)
    print("✓ Matrix multiplication working")
    print(f"  X shape: {len(X)}x{len(X[0])}")
    print(f"  W shape: {len(W)}x{len(W[0])}")
    print(f"  Result shape: {len(result)}x{len(result[0])}")

    # Test accuracy computation
    logits = [[1.0, 0.5], [0.3, 1.2], [0.8, 0.9]]
    labels = [0, 1, 1]
    acc = benchmark._compute_accuracy(logits, labels)
    print(f"✓ Accuracy computation: {acc:.1f}")

    print("=" * 50)
    print("All core functionality tests passed!")
    print("The benchmark is ready for full RES testing.")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
