#!/usr/bin/env python3
"""
Comprehensive Test Suite for SRT Library Rust & CUDA Kernels (Final Refinement)
"""

import sys
import time
import math
from typing import Callable, List, Tuple, Any
from dataclasses import dataclass

# ============================================================================
# Test Framework
# ============================================================================

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_ms: float
    error: str = ""


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.current_category = "General"
    
    def set_category(self, category: str):
        self.current_category = category
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print('='*60)
    
    def run_test(self, name: str, test_fn: Callable[[], Any], 
                 expected: Any = None, check_fn: Callable[[Any], bool] = None):
        """Run a single test and record the result."""
        start = time.perf_counter()
        try:
            result = test_fn()
            duration = (time.perf_counter() - start) * 1000
            
            # Check result if needed
            if check_fn is not None:
                passed = check_fn(result)
            elif expected is not None:
                passed = result == expected
            else:
                passed = True  # No assertion, just checking it runs
            
            if passed:
                print(f"  ✓ {name} ({duration:.2f}ms)")
            else:
                print(f"  ✗ {name} - unexpected result: {result}")
            
            self.results.append(TestResult(
                name=name, category=self.current_category,
                passed=passed, duration_ms=duration,
                error="" if passed else f"Expected {expected}, got {result}"
            ))
            return result
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            error_msg = str(e)
            print(f"  ✗ {name} - ERROR: {error_msg}")
            self.results.append(TestResult(
                name=name, category=self.current_category,
                passed=False, duration_ms=duration, error=error_msg
            ))
            return None
    
    def summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)
        
        for cat, tests in categories.items():
            cat_passed = sum(1 for t in tests if t.passed)
            cat_total = len(tests)
            status = "✓" if cat_passed == cat_total else "✗"
            print(f"  {status} {cat}: {cat_passed}/{cat_total}")
        
        print("-"*60)
        print(f"Total: {passed}/{total} passed ({100*passed/total:.1f}%)" if total > 0 else "No tests run.")
        
        if failed > 0:
            print("\nFailed tests (first 100 chars):")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.category}/{r.name}: {r.error[:100]}")
        
        return passed, failed


# ============================================================================
# Import SRT Library
# ============================================================================

srt = None
try:
    import srt_library._core as srt
    print("✓ Successfully imported srt_library._core")
except ImportError:
    try:
        import _core as srt
        print("✓ Successfully imported _core")
    except ImportError as e:
        print(f"✗ Failed to import SRT core module: {e}")
        print("  Run 'maturin develop' in the rust/ directory first")
        sys.exit(1)

runner = TestRunner()

# ============================================================================
# 1. Exact Arithmetic Tests
# ============================================================================

runner.set_category("Exact Arithmetic")

# Rational
runner.run_test("Rational creation", lambda: srt.Rational(3, 4))
runner.run_test("Rational creation (int)", lambda: srt.Rational(42, 1))

# GoldenExact
runner.run_test("GoldenExact creation", lambda: srt.GoldenExact.from_integers(1, 0))
runner.run_test("GoldenExact phi", lambda: srt.GoldenExact.from_integers(0, 1))
runner.run_test("GoldenExact phi squared", 
    lambda: srt.GoldenExact.from_integers(0, 1).phi_coefficient)

# FundamentalConstant
runner.run_test("FundamentalConstant PI", lambda: srt.FundamentalConstant.pi())
runner.run_test("FundamentalConstant PHI", lambda: srt.FundamentalConstant.phi())

# ============================================================================
# 2. SRT Constants and Scalar Math
# ============================================================================

runner.set_category("SRT Constants")

runner.run_test("srt_phi", lambda: srt.srt_phi(), check_fn=lambda x: abs(x - 1.618033988749895) < 1e-10)
runner.run_test("srt_phi_inv", lambda: srt.srt_phi_inv(), check_fn=lambda x: abs(x - 0.618033988749895) < 1e-10)
runner.run_test("srt_q_deficit", lambda: srt.srt_q_deficit(), check_fn=lambda x: 0.02 < x < 0.03)

runner.set_category("Scalar Math")

runner.run_test("srt_sqrt(4)", lambda: srt.srt_sqrt(4.0), expected=2.0)
runner.run_test("srt_exp(0)", lambda: srt.srt_exp(0.0), expected=1.0)
runner.run_test("srt_sin(0)", lambda: srt.srt_sin(0.0), expected=0.0)
runner.run_test("srt_gcd(48, 18)", lambda: srt.srt_gcd(48, 18), expected=6)
runner.run_test("srt_factorial(5)", lambda: srt.srt_factorial(5), expected=120)

# ============================================================================
# 3. Hierarchy Corrections
# ============================================================================

runner.set_category("Hierarchy Corrections")

runner.run_test("hierarchy_e8_dim", lambda: srt.hierarchy_e8_dim(), expected=248)
runner.run_test("hierarchy_e7_dim", lambda: srt.hierarchy_e7_dim(), expected=133)
runner.run_test("hierarchy_e8_roots", lambda: srt.hierarchy_e8_roots(), expected=240)
runner.run_test("hierarchy_mersenne_m5", lambda: srt.hierarchy_mersenne_m5(), expected=31)

# ============================================================================
# 4. Spectral Operations
# ============================================================================

runner.set_category("Spectral Operations")

try:
    windings = list(srt.enumerate_windings(2))[:5]
    has_windings = len(windings) > 0
except:
    has_windings = False
    windings = []

if has_windings:
    runner.run_test("theta_series_evaluate", lambda: srt.theta_series_evaluate(windings, 0.1))
    # theta_series_weighted(windings, weights, t)
    runner.run_test("theta_series_weighted", lambda: srt.theta_series_weighted(windings, [1.0]*len(windings), 0.1))
    runner.run_test("heat_kernel_trace", lambda: srt.heat_kernel_trace(windings, 0.1, 1.0))
    runner.run_test("spectral_zeta", lambda: srt.spectral_zeta(windings, 2.0, 1.0))
    runner.run_test("partition_function", lambda: srt.partition_function(windings))
else:
    print("  (Skipping spectral tests - no windings available)")

# ============================================================================
# 5. Winding States
# ============================================================================

runner.set_category("Winding States")

runner.run_test("enumerate_windings", lambda: list(srt.enumerate_windings(2))[:5], check_fn=lambda x: len(x) >= 1)
runner.run_test("WindingState creation", lambda: srt.WindingState(1, 0, 0, 0))

# ============================================================================
# 6. Number Theory
# ============================================================================

runner.set_category("Number Theory")

runner.run_test("py_mobius(6)", lambda: srt.py_mobius(6), expected=1)
runner.run_test("py_is_square_free(6)", lambda: srt.py_is_square_free(6), expected=True)
runner.run_test("is_fermat_prime(1)", lambda: srt.is_fermat_prime(1), expected=True) # F_1 = 5
runner.run_test("py_lucas_number(5)", lambda: srt.py_lucas_number(5), expected=11)
runner.run_test("pisano_period(5)", lambda: srt.pisano_period(5), expected=20)

# ============================================================================
# 7. Linear Algebra
# ============================================================================

runner.set_category("Linear Algebra")

# Use TensorStorage for linalg
try:
    # from_list(data, shape, dtype, device)
    A_ts = srt.TensorStorage.from_list([1.0, 2.0, 3.0, 4.0], [2, 2], "f64", "cpu")
    B_ts = srt.TensorStorage.from_list([5.0, 6.0, 7.0, 8.0], [2, 2], "f64", "cpu")
    
    runner.run_test("TensorStorage MatMul", lambda: A_ts.matmul(B_ts))
    runner.run_test("TensorStorage Add", lambda: A_ts.add(B_ts))
    
    # Static linalg functions
    runner.run_test("linalg_mm", lambda: srt.linalg_mm(A_ts, B_ts))
    
    # mm_add signature: (A, B, C, alpha, beta)
    C_ts = srt.TensorStorage.from_list([1.0]*4, [2, 2], "f64", "cpu")
    runner.run_test("linalg_mm_add", lambda: srt.linalg_mm_add(A_ts, B_ts, C_ts, 1.0, 1.0))

except Exception as e:
    print(f"  ✗ Linalg Setup ERROR: {e}")

# ============================================================================
# 8. Loss Functions
# ============================================================================

runner.set_category("Loss Functions")

runner.run_test("py_mse_loss", lambda: srt.py_mse_loss([1.0, 2.0], [1.1, 2.1]))
runner.run_test("py_softmax", lambda: srt.py_softmax([1.0, 2.0, 3.0]))
runner.run_test("py_cross_entropy_loss", lambda: srt.py_cross_entropy_loss([0.1, 0.2, 0.7], [0.0, 0.0, 1.0]))

# ============================================================================
# 9. Golden GELU
# ============================================================================

runner.set_category("Golden GELU")

runner.run_test("golden_gelu_forward", lambda: srt.golden_gelu_forward([0.0, 1.0, 2.0]))
# batched_golden_gelu_forward(batch, batch_size, n_elements)
runner.run_test("batched_golden_gelu_forward", lambda: srt.batched_golden_gelu_forward([0.0, 1.0], 1, 2))

# ============================================================================
# 10. Resonant Engine
# ============================================================================

runner.set_category("Resonant Engine")

runner.run_test("ResonantTensor creation", lambda: srt.ResonantTensor([1.0, 2.0, 3.0, 4.0], [2, 2]))
runner.run_test("ResonantTensor syntony", lambda: srt.ResonantTensor([1.0, 2.0, 3.0, 4.0], [2, 2]).syntony)

# ============================================================================
# 11. CUDA Operations
# ============================================================================

runner.set_category("CUDA")

if srt.cuda_is_available():
    runner.run_test("cuda_device_count", lambda: srt.cuda_device_count())
    runner.run_test("validate_kernels", lambda: srt.validate_kernels(0))
    runner.run_test("py_load_reduction_kernels", lambda: srt.py_load_reduction_kernels(0))
else:
    print("  (CUDA not available)")

# ============================================================================
# 12. SNA
# ============================================================================

runner.set_category("SNA")

try:
    runner.run_test("DiscreteHilbertKernel", lambda: srt.sna.DiscreteHilbertKernel(31))
    runner.run_test("ResonantOscillator", lambda: srt.sna.ResonantOscillator(1, 31, 500))
except Exception as e:
    print(f"  ✗ SNA ERROR: {e}")

# ============================================================================
# Summary
# =============================================================

runner.summary()
