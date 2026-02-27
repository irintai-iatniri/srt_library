#!/usr/bin/env python3
"""
Performance benchmarks for exact arithmetic vs float arithmetic.

Measures and documents the performance tradeoffs of the exact-first architecture.

Phase 7.3: Performance Benchmarks

Target: <100ms per DHSR cycle on 1024x1024 state with Q32.32 GPU
"""

import time
import pytest
from typing import Tuple, List
from syntonic_applications.core.state import State, state
from syntonic_applications.core.dtype import golden_exact, fixed_point64, float64
from syntonic_applications.core.device import cpu, cuda
from syntonic._core import cuda_is_available


def benchmark_operation(func, num_iterations: int = 10) -> Tuple[float, float]:
    """
    Benchmark an operation.

    Args:
        func: Function to benchmark
        num_iterations: Number of iterations to average over

    Returns:
        (mean_time_ms, std_time_ms)
    """
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time


class TestBasicOperationPerformance:
    """Benchmark basic operations: exact vs float."""

    def test_state_creation_performance(self):
        """Compare state creation speed."""
        shape = (256, 256)
        num_iterations = 20

        # Exact mode (CPU)
        exact_time, exact_std = benchmark_operation(
            lambda: state.from_seed(42, shape, device=cpu),
            num_iterations
        )

        # Float mode (CPU)
        float_time, float_std = benchmark_operation(
            lambda: state.from_seed(42, shape, dtype=float64, device=cpu),
            num_iterations
        )

        print(f"\nState Creation ({shape}):")
        print(f"  Exact (CPU):  {exact_time:.2f} ± {exact_std:.2f} ms")
        print(f"  Float (CPU):  {float_time:.2f} ± {float_std:.2f} ms")
        print(f"  Ratio:        {exact_time/float_time:.2f}x")

        # Document results
        assert exact_time > 0, "Benchmark must complete"
        assert float_time > 0, "Benchmark must complete"

    def test_arithmetic_performance(self):
        """Compare arithmetic operation speed."""
        shape = (512, 512)

        # Create states
        state_exact = state.from_seed(42, shape, device=cpu)
        state_float = state.from_seed(42, shape, dtype=float64, device=cpu)

        # Benchmark addition
        exact_time, _ = benchmark_operation(
            lambda: state_exact + state_exact,
            num_iterations=10
        )

        float_time, _ = benchmark_operation(
            lambda: state_float + state_float,
            num_iterations=10
        )

        print(f"\nArithmetic Addition ({shape}):")
        print(f"  Exact (CPU):  {exact_time:.2f} ms")
        print(f"  Float (CPU):  {float_time:.2f} ms")
        print(f"  Overhead:     {exact_time/float_time:.2f}x")

        # Benchmark multiplication
        exact_time_mul, _ = benchmark_operation(
            lambda: state_exact * 2.5,
            num_iterations=10
        )

        float_time_mul, _ = benchmark_operation(
            lambda: state_float * 2.5,
            num_iterations=10
        )

        print(f"\nScalar Multiplication ({shape}):")
        print(f"  Exact (CPU):  {exact_time_mul:.2f} ms")
        print(f"  Float (CPU):  {float_time_mul:.2f} ms")
        print(f"  Overhead:     {exact_time_mul/float_time_mul:.2f}x")


class TestDHSRCyclePerformance:
    """Benchmark DHSR cycle performance."""

    def test_single_dhsr_cycle_cpu(self):
        """Benchmark single DHSR cycle on CPU."""
        shape = (256, 256)

        state_exact = state.from_seed(42, shape, device=cpu)
        state_float = state.from_seed(42, shape, dtype=float64, device=cpu)

        # Benchmark exact
        exact_time, exact_std = benchmark_operation(
            lambda: state_exact.differentiate().harmonize(),
            num_iterations=10
        )

        # Benchmark float
        float_time, float_std = benchmark_operation(
            lambda: state_float.differentiate().harmonize(),
            num_iterations=10
        )

        print(f"\nDHSR Cycle CPU ({shape}):")
        print(f"  Exact:   {exact_time:.2f} ± {exact_std:.2f} ms")
        print(f"  Float:   {float_time:.2f} ± {float_std:.2f} ms")
        print(f"  Ratio:   {exact_time/float_time:.2f}x")

        # Document that exact is slower but deterministic
        # Expected: 25-67x slower per plan
        print(f"  Status:  {'✓ Acceptable' if exact_time < 10000 else '⚠ Too slow'}")

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_single_dhsr_cycle_gpu(self):
        """Benchmark single DHSR cycle on GPU (Q32.32 fixed-point)."""
        shape = (1024, 1024)

        state_exact = state.from_seed(42, shape, device=cuda)
        state_float = state.from_seed(42, shape, dtype=float64, device=cuda)

        # Benchmark exact (Q32.32)
        exact_time, exact_std = benchmark_operation(
            lambda: state_exact.differentiate().harmonize(),
            num_iterations=10
        )

        # Benchmark float
        float_time, float_std = benchmark_operation(
            lambda: state_float.differentiate().harmonize(),
            num_iterations=10
        )

        print(f"\nDHSR Cycle GPU ({shape}):")
        print(f"  Exact (Q32.32):  {exact_time:.2f} ± {exact_std:.2f} ms")
        print(f"  Float:           {float_time:.2f} ± {float_std:.2f} ms")
        print(f"  Overhead:        {exact_time/float_time:.2f}x")

        # Target: <100ms for 1024x1024
        print(f"  Target:          <100ms")
        print(f"  Status:          {'✓ PASS' if exact_time < 100 else '⚠ SLOW'}")

        # Q32.32 should have minimal overhead on GPU
        # Expected: ~1x for basic ops, ~10x for transcendentals
        assert exact_time > 0, "Benchmark must complete"

    def test_long_dhsr_chain_cpu(self):
        """Benchmark 100 DHSR cycles on CPU."""
        shape = (128, 128)
        num_cycles = 100

        state_exact = state.from_seed(42, shape, device=cpu)
        state_float = state.from_seed(42, shape, dtype=float64, device=cpu)

        # Benchmark exact
        def exact_chain():
            state = state.from_seed(42, shape, device=cpu)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        exact_time, _ = benchmark_operation(exact_chain, num_iterations=3)

        # Benchmark float
        def float_chain():
            state = state.from_seed(42, shape, dtype=float64, device=cpu)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        float_time, _ = benchmark_operation(float_chain, num_iterations=3)

        print(f"\n{num_cycles} DHSR Cycles CPU ({shape}):")
        print(f"  Exact:   {exact_time:.1f} ms ({exact_time/num_cycles:.2f} ms/cycle)")
        print(f"  Float:   {float_time:.1f} ms ({float_time/num_cycles:.2f} ms/cycle)")
        print(f"  Ratio:   {exact_time/float_time:.2f}x")

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_long_dhsr_chain_gpu(self):
        """Benchmark 100 DHSR cycles on GPU."""
        shape = (512, 512)
        num_cycles = 100

        # Exact (Q32.32)
        def exact_chain():
            state = state.from_seed(42, shape, device=cuda)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        exact_time, _ = benchmark_operation(exact_chain, num_iterations=3)

        # Float
        def float_chain():
            state = state.from_seed(42, shape, dtype=float64, device=cuda)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        float_time, _ = benchmark_operation(float_chain, num_iterations=3)

        print(f"\n{num_cycles} DHSR Cycles GPU ({shape}):")
        print(f"  Exact (Q32.32):  {exact_time:.1f} ms ({exact_time/num_cycles:.2f} ms/cycle)")
        print(f"  Float:           {float_time:.1f} ms ({float_time/num_cycles:.2f} ms/cycle)")
        print(f"  Overhead:        {exact_time/float_time:.2f}x")


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
class TestCPUGPUComparison:
    """Compare CPU exact (GoldenExact) vs GPU exact (Q32.32)."""

    def test_cpu_vs_gpu_exact(self):
        """Compare GoldenExact CPU vs FixedPoint64 GPU."""
        shape = (256, 256)

        state_cpu = state.from_seed(42, shape, device=cpu)
        state_gpu = state.from_seed(42, shape, device=cuda)

        # Benchmark CPU
        cpu_time, _ = benchmark_operation(
            lambda: state_cpu.differentiate().harmonize(),
            num_iterations=5
        )

        # Benchmark GPU
        gpu_time, _ = benchmark_operation(
            lambda: state_gpu.differentiate().harmonize(),
            num_iterations=5
        )

        print(f"\nCPU vs GPU Exact Arithmetic ({shape}):")
        print(f"  CPU (GoldenExact):   {cpu_time:.2f} ms")
        print(f"  GPU (FixedPoint64):  {gpu_time:.2f} ms")
        print(f"  Speedup:             {cpu_time/gpu_time:.2f}x")

        # GPU should be faster even with exact arithmetic
        print(f"  Result:              {'✓ GPU faster' if gpu_time < cpu_time else '⚠ GPU slower'}")


class TestMemoryUsage:
    """Benchmark memory usage for exact vs float."""

    def test_memory_footprint(self):
        """Compare memory footprint."""
        shape = (1024, 1024)

        # Create states
        state_exact = state.from_seed(42, shape, device=cpu)
        state_float = state.from_seed(42, shape, dtype=float64, device=cpu)

        # Rough memory estimates
        size = shape[0] * shape[1]

        # golden_exact: 16 bytes per element (Q(φ): a + b·φ with i64 coefficients)
        # fixed_point64: 8 bytes per element (Q32.32)
        # float64: 8 bytes per element

        exact_mem_mb = (size * 16) / (1024 * 1024)
        float_mem_mb = (size * 8) / (1024 * 1024)

        print(f"\nMemory Footprint ({shape}):")
        print(f"  Exact (GoldenExact):  ~{exact_mem_mb:.2f} MB (16 bytes/element)")
        print(f"  Float (float64):      ~{float_mem_mb:.2f} MB (8 bytes/element)")
        print(f"  Ratio:                {exact_mem_mb/float_mem_mb:.2f}x")

        print(f"\nNote: GPU FixedPoint64 (Q32.32) uses 8 bytes/element, same as float64")


class TestScalability:
    """Test performance scaling with state size."""

    def test_size_scaling(self):
        """Measure performance across different sizes."""
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

        print("\nPerformance Scaling (DHSR cycle, CPU):")
        print(f"{'Size':<15} {'Exact (ms)':<15} {'Float (ms)':<15} {'Ratio':<10}")
        print("-" * 55)

        for shape in sizes:
            state_exact = state.from_seed(42, shape, device=cpu)
            state_float = state.from_seed(42, shape, dtype=float64, device=cpu)

            exact_time, _ = benchmark_operation(
                lambda: state_exact.differentiate().harmonize(),
                num_iterations=5
            )

            float_time, _ = benchmark_operation(
                lambda: state_float.differentiate().harmonize(),
                num_iterations=5
            )

            ratio = exact_time / float_time
            print(f"{str(shape):<15} {exact_time:<15.2f} {float_time:<15.2f} {ratio:<10.2f}x")


class TestReconstructionOverhead:
    """Measure overhead of perfect reconstruction."""

    def test_reconstruction_cost(self):
        """Measure cost of deterministic reconstruction."""
        seed = 42
        shape = (256, 256)
        num_cycles = 50

        # Exact reconstruction (bit-perfect)
        def exact_reconstruct():
            state = state.from_seed(seed, shape, device=cpu)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        exact_time, _ = benchmark_operation(exact_reconstruct, num_iterations=3)

        # Float reconstruction (approximate)
        def float_reconstruct():
            state = state.from_seed(seed, shape, dtype=float64, device=cpu)
            for _ in range(num_cycles):
                state = state.differentiate().harmonize()
            return state

        float_time, _ = benchmark_operation(float_reconstruct, num_iterations=3)

        print(f"\nReconstruction Cost ({num_cycles} cycles, {shape}):")
        print(f"  Exact (bit-perfect):   {exact_time:.1f} ms")
        print(f"  Float (approximate):   {float_time:.1f} ms")
        print(f"  Overhead:              {exact_time/float_time:.2f}x")
        print(f"  Tradeoff:              Exact is slower but guarantees zero-entropy")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 70)
    print("EXACT ARITHMETIC PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print("\nPhase 7.3: Performance Benchmarks")
    print("\nTarget: <100ms per DHSR cycle on 1024x1024 state with Q32.32 GPU")
    print("=" * 70)

    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_all_benchmarks()
