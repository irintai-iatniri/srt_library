# Exact Arithmetic API Reference

**Quick reference for exact arithmetic in Syntonic**

## Core Types

### State Class

```python
from syntonic.core import State

# Default: exact arithmetic
state = State([1.0, 2.0, 3.0])  # dtype=golden_exact (CPU)
state = State([1.0, 2.0, 3.0], device='cuda')  # dtype=fixed_point64 (GPU)

# Explicit float (opt-in)
from syntonic.core.dtype import float64
state = State([1.0, 2.0, 3.0], dtype=float64)

# Seed-based construction (deterministic)
state = State.from_seed(seed=42, shape=(256, 256))
```

### Conversion Methods

```python
# Exact to float (lossy)
state_float = state.to_float()

# Float to exact (approximate)
state_exact = state.to_exact()

# Exact to Q32.32
state_fixed = state.to_fixed_point()
```

## DType System

### Exact Types

```python
from syntonic.core.dtype import golden_exact, rational, fixed_point64

# Type hierarchy (highest to lowest precision)
# 1. golden_exact  - Q(φ) field, unlimited precision
# 2. rational      - Q field, exact fractions
# 3. fixed_point64 - Q32.32, bounded range
```

### Type Checking

```python
from syntonic.core.dtype import is_exact_dtype, is_float_dtype

if is_exact_dtype(state.dtype):
    print("Using exact arithmetic")

if is_float_dtype(state.dtype):
    print("Using float approximation")
```

### Type Promotion

```python
from syntonic.core.dtype import promote_dtypes, golden_exact, rational

# Exact + exact = higher precision exact
result_dtype = promote_dtypes(rational, golden_exact)
# → golden_exact

# Float + float = higher precision float
result_dtype = promote_dtypes(float32, float64)
# → float64
```

## Exact Constants

### Golden Ratio Constants

```python
from syntonic.exact import PHI, PHI_INVERSE, PHI_SQUARED

# Exact constants (use these\!)
phi = PHI                  # φ = (1+√5)/2
phi_inv = PHI_INVERSE      # 1/φ = φ - 1
phi_sq = PHI_SQUARED       # φ² = φ + 1

# Evaluate to float when needed
phi_float = PHI.eval()     # → 1.618033988749895

# Deprecated (backward compat)
from syntonic.exact import get_phi_float
phi = get_phi_float()      # DEPRECATED: use PHI.eval()
```

### Legacy Float Constants (Deprecated)

```python
from syntonic.exact import PHI_NUMERIC, E_STAR_NUMERIC, Q_DEFICIT_NUMERIC

# These are DEPRECATED - use exact constants instead
# They introduce floating-point drift in recursive operations

# OLD (deprecated)
mass = PHI_NUMERIC * base

# NEW (exact-first)
from syntonic.exact import PHI
mass = (PHI * base_exact).eval()
```

## DHSR Operators

### Differentiation

```python
from syntonic.crt.operators import DifferentiationOperator
from syntonic.exact import GoldenExact

# Exact by default
D = DifferentiationOperator(
    alpha_0=0.1,      # Can be float (converted to exact internally)
    zeta_0=0.01,
    num_modes=8
)

# Or with exact coefficients
D = DifferentiationOperator(
    alpha_0=GoldenExact.nearest(0.1, 1<<30),
    zeta_0=GoldenExact.nearest(0.01, 1<<30)
)

# Apply (preserves exact types)
result = D.apply(state)
```

### Harmonization

```python
from syntonic.crt.operators import HarmonizationOperator

# Exact mode (recommended for SGC)
H = HarmonizationOperator(
    beta_0=0.618,      # Can use float (converted internally)
    gamma_0=0.1,
    exact=True         # DEFAULT: use exact arithmetic
)

# Float mode (legacy)
H = HarmonizationOperator(
    beta_0=0.618,
    exact=False        # Opt-in to float approximation
)

# Apply (preserves exact types)
result = H.apply(state, syntony=0.5)
```

### Full DHSR Cycle

```python
# Exact DHSR cycle
D = DifferentiationOperator(alpha_0=0.1)
H = HarmonizationOperator(exact=True)

# Iterate (maintains exactness)
for _ in range(100):
    state = H.apply(D.apply(state))

# Result is still exact - zero drift\!
assert is_exact_dtype(state.dtype)
```

## Arithmetic Operations

### State Arithmetic

```python
# All operations preserve exact types
result = state1 + state2      # Exact + exact = exact
result = state * 2.5          # Exact * scalar = exact
result = state1 @ state2      # Matrix multiply (exact)
result = state / 3.0          # Division (exact)

# Type conversions handled automatically
exact_state = State([1, 2], dtype=golden_exact)
float_state = State([3, 4], dtype=float64)

# Mixing types: converts float to exact
result = exact_state + float_state  # Result is golden_exact
```

### Scalar Operations

```python
from syntonic.exact import GoldenExact

# Float scalars converted to exact internally
result = state * 0.618  # Backend uses GoldenExact.nearest(0.618)

# Explicit exact scalars
alpha = GoldenExact.nearest(0.618, 1<<30)
result = state * alpha.eval()
```

## GPU Acceleration

### Q32.32 on CUDA

```python
# Automatic Q32.32 on GPU
state = State([1.0, 2.0, 3.0], device='cuda')
assert state.dtype.name == 'fixed_point64'

# All operations remain exact
for _ in range(1000):
    state = state * 1.618
    state = state / 1.618

# Zero drift - result is exact
```

### Device Transfer

```python
# CPU (golden_exact) to GPU (fixed_point64)
cpu_state = State([1, 2, 3])  # golden_exact
gpu_state = cpu_state.cuda()  # Converts to fixed_point64

# GPU to CPU
cpu_state = gpu_state.cpu()   # Converts to golden_exact
```

## Helper Functions

### GoldenExact Operations

```python
from syntonic.exact import GoldenExact

# Create from integers: a + b·φ
g = GoldenExact.from_integers(1, 1)  # 1 + 1·φ = 2.618...

# Approximate float as exact
g = GoldenExact.nearest(0.618, max_coeff=1<<30)

# Arithmetic (exact)
g1 = GoldenExact.from_integers(1, 0)  # 1
g2 = GoldenExact.from_integers(0, 1)  # φ
g3 = g1 + g2                          # 1 + φ
g4 = g2 * g2                          # φ² = φ + 1 (identity\!)

# Evaluate to float
print(g3.eval())  # 2.618033988749895
```

### Rational Operations

```python
from syntonic.exact import Rational

# Create rational
r = Rational.new(3, 4)  # 3/4

# Arithmetic
r1 = Rational.new(1, 2)
r2 = Rational.new(1, 3)
r3 = r1 + r2  # 5/6

# Convert to float
print(r3.to_f64())  # 0.8333...
```

## Default Behavior Summary

| Context | Default DType | Notes |
|---------|--------------|-------|
| `State([...])` CPU | `golden_exact` | Q(φ) field |
| `State([...], device='cuda')` | `fixed_point64` | Q32.32 |
| `State([...], dtype=float64)` | `float64` | Explicit opt-in |
| Complex data | `complex128` | Not yet exact |
| DHSR operators | Exact | Use `exact=True` for harmonization |
| Constants (`PHI`) | Exact | Use `.eval()` for float |

## Common Patterns

### Pattern: Exact Computation, Float Display

```python
# Compute with exact arithmetic
state = State([1.0, 2.0, 3.0])
for _ in range(100):
    state = state.differentiate().harmonize()

# Display with float
print(state.to_float().to_list())
```

### Pattern: GPU Acceleration

```python
# Use Q32.32 for performance
state = State.from_seed(42, (1024, 1024), device='cuda')

# Fast exact DHSR cycles
for _ in range(1000):
    state = state.differentiate().harmonize()

# Transfer to CPU if needed
result = state.cpu()
```

### Pattern: Deterministic Reconstruction

```python
# Compress to seed
def compress(data):
    # ... find optimal seed ...
    return seed

# Reconstruct exactly
def decompress(seed, shape):
    state = State.from_seed(seed, shape)
    # Apply DHSR evolution
    for _ in range(n_cycles):
        state = state.differentiate().harmonize()
    return state

# Guarantee: decompress(compress(data)) == data (bit-perfect)
```

## Error Handling

### Common Errors

```python
# Error: Mixing exact and float without conversion
try:
    result = exact_state + float_state
except TypeError:
    # Fix: explicit conversion
    result = exact_state + float_state.to_exact()

# Error: Overflow in GoldenExact
try:
    for _ in range(10000):
        state = state * 1.618
except OverflowError:
    # Fix: use FixedPoint64 (Q32.32) on GPU
    state = state.cuda()  # Bounded range
```

## See Also

- [Architecture Document](exact_arithmetic_architecture.md) - Design rationale
- [Migration Guide](migrating_to_exact.md) - Update existing code
- [Status Document](../EXACT_ARITHMETIC_STATUS.md) - Implementation progress

---

*Last Updated: 2026-02-03*  
*Version: 1.0*
