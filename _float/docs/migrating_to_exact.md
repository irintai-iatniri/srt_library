# Migration Guide: Transitioning to Exact Arithmetic

**For Syntonic Library Users**

This guide helps you update existing code to use the new exact-first architecture introduced in Syntonic v0.1.0 (Phases 1-6).

## Quick Reference

| Old (Float) | New (Exact) | Notes |
|-------------|-------------|-------|
| `State([1, 2, 3])` | Same, now defaults to exact | No change needed |
| `dtype=float64` | `dtype=None` (exact) or explicit `dtype=float64` | Explicit opt-in for float |
| `PHI_NUMERIC * x` | `(PHI * x_exact).eval()` | Use exact PHI, evaluate at end |
| `DifferentiationOperator()` | Same, now uses exact | No change needed |
| `HarmonizationOperator()` | `HarmonizationOperator(exact=True)` | Explicit exact mode |

## Migration Steps

### Step 1: Update State Constructors

#### Before (Float by default)
```python
from syntonic.core import State

# Was: float64 by default
state = State([1.0, 2.0, 3.0])
```

#### After (Exact by default)
```python
from syntonic.core import State

# Now: golden_exact by default (CPU) or fixed_point64 (GPU)
state = State([1.0, 2.0, 3.0])  # Automatically exact

# Explicit float if needed
from syntonic.core.dtype import float64
state_float = State([1.0, 2.0, 3.0], dtype=float64)

# Or convert existing state
state_float = state.to_float()
```

**Action Required**: 
- Review all `State()` constructors
- Add explicit `dtype=float64` if float is required
- Otherwise, no change needed (will use exact automatically)

### Step 2: Update Constant Usage

#### Before (Numeric float constants)
```python
from syntonic.exact import PHI_NUMERIC, PHI_INVERSE_NUMERIC

mass = PHI_NUMERIC * base_mass
ratio = PHI_INVERSE_NUMERIC * scale
```

#### After (Exact constants)
```python
from syntonic.exact import PHI, PHI_INVERSE

# Option A: Keep exact, evaluate at end
mass_exact = PHI * base_mass_exact
mass_float = mass_exact.eval()

# Option B: Evaluate immediately (if base_mass is float)
mass = (PHI).eval() * base_mass

# Option C: Use legacy function (backward compat)
from syntonic.exact import get_phi_float
mass = get_phi_float() * base_mass  # DEPRECATED but works
```

**Action Required**:
- Search for `PHI_NUMERIC`, `E_STAR_NUMERIC`, `Q_DEFICIT_NUMERIC`
- Replace with exact constants and explicit `.eval()`
- Or use `get_phi_float()` for quick migration (deprecated)

### Step 3: Update DHSR Operators

#### Before (Mixed exact/float)
```python
from syntonic.crt.operators import DifferentiationOperator, HarmonizationOperator

D = DifferentiationOperator(alpha_0=0.1)
H = HarmonizationOperator(beta_0=0.618)

result = H.apply(D.apply(state))
```

#### After (Explicit exact mode)
```python
from syntonic.crt.operators import DifferentiationOperator, HarmonizationOperator

# Differentiation: already exact by default
D = DifferentiationOperator(alpha_0=0.1)

# Harmonization: set exact=True explicitly
H = HarmonizationOperator(beta_0=0.618, exact=True)

result = H.apply(D.apply(state))
```

**Action Required**:
- Add `exact=True` to `HarmonizationOperator` calls
- No change needed for `DifferentiationOperator` (already exact)

### Step 4: Handle Type Conversions

#### Explicit Conversions
```python
# Exact to float (lossy)
state_float = state.to_float()

# Float to exact (approximate)
state_exact = state_float.to_exact()

# Exact to Q32.32 (for GPU)
state_fixed = state.to_fixed_point()
```

#### Check Dtype
```python
from syntonic.core.dtype import is_exact_dtype, is_float_dtype

if is_exact_dtype(state.dtype):
    print("Using exact arithmetic")
elif is_float_dtype(state.dtype):
    print("Using float approximation")
```

### Step 5: Update Visualization Code

For preview/plotting code that doesn't need exactness:

#### Before
```python
import matplotlib.pyplot as plt

# Assumed float
plt.plot(state.to_list())
```

#### After
```python
import matplotlib.pyplot as plt

# Explicit float conversion for visualization
plt.plot(state.to_float().to_list())
```

**Action Required**:
- Add `.to_float()` before visualization/plotting
- Keeps visualization fast, computation exact

### Step 6: Test Determinism

Add tests to verify exact arithmetic works:

```python
def test_deterministic_reconstruction():
    """Verify exact arithmetic produces identical results."""
    from syntonic.core import State
    
    seed = 42
    shape = (64, 64)
    
    # Run 1
    state1 = State.from_seed(seed, shape)
    for _ in range(10):
        state1 = state1.differentiate().harmonize()
    
    # Run 2
    state2 = State.from_seed(seed, shape)
    for _ in range(10):
        state2 = state2.differentiate().harmonize()
    
    # Must be bit-identical
    assert state1.to_list() == state2.to_list()
```

## Common Migration Patterns

### Pattern 1: Physics Module Update

#### Before
```python
# physics/fermions/leptons.py
from syntonic.exact import PHI_NUMERIC

def compute_mass():
    phi = PHI_NUMERIC
    return base * phi**5
```

#### After
```python
# physics/fermions/leptons.py
from syntonic.exact import PHI

def compute_mass():
    phi = PHI.eval()  # Explicit conversion
    return base * phi**5
```

### Pattern 2: Neural Network Module

#### Before
```python
class SyntonicLayer:
    def __init__(self):
        self.weight = State.zeros((10, 10))  # Was float
    
    def forward(self, x):
        return self.weight @ x
```

#### After
```python
class SyntonicLayer:
    def __init__(self, exact=True):
        # Explicit dtype control
        dtype = None if exact else float64
        self.weight = State.zeros((10, 10), dtype=dtype)
    
    def forward(self, x):
        # Type compatibility handled automatically
        return self.weight @ x
```

### Pattern 3: Data Loading

#### Before
```python
def load_data(path):
    data = np.load(path)
    return State(data)  # Was float
```

#### After
```python
def load_data(path, exact=True):
    data = np.load(path)
    
    if exact:
        # Default: exact arithmetic
        return State(data)
    else:
        # Explicit float for legacy compatibility
        return State(data, dtype=float64)
```

## Backward Compatibility

The migration is designed to be **mostly backward compatible**:

### ✓ Works Automatically
- `State()` constructors (now exact by default)
- DHSR operator calls (upgraded automatically)
- Arithmetic operations (preserve types correctly)

### ⚠️ May Need Changes
- Code using `PHI_NUMERIC` directly
- Hardcoded `dtype=float64` (will override exact default)
- Type-sensitive tests (now using exact types)

### ❌ Breaking Changes
- Type promotion rules changed (exact > float)
- Default dtype changed (golden_exact, not float64)
- Mixing exact/float requires explicit conversion

## Performance Considerations

### Expected Slowdowns

| Component | Old (Float) | New (Exact) | Ratio |
|-----------|-------------|-------------|-------|
| State creation | 1 ms | 1 ms | 1x |
| Arithmetic (CPU) | 1 ms | 25-67 ms | 25-67x |
| Arithmetic (GPU) | 1 ms | 1 ms | 1x |
| DHSR cycle (CPU) | 10 ms | 250-670 ms | 25-67x |
| DHSR cycle (GPU) | 15 ms | 15 ms | 1x |

### Optimization Strategies

1. **Use GPU**: Q32.32 has zero overhead on CUDA
   ```python
   state = State(data, device='cuda')  # Fast + exact
   ```

2. **Float for Preview**: Quick visualization
   ```python
   preview = state.to_float()  # Fast approximation
   ```

3. **Exact for Compression**: Slow but correct
   ```python
   seed = compress_exact(state)  # Bit-perfect
   ```

## Troubleshooting

### Issue 1: "Dtype mismatch" Error

**Cause**: Mixing exact and float types without conversion.

**Fix**: 
```python
# Before (error)
exact_state + float_state

# After (explicit)
exact_state + float_state.to_exact()
# OR
exact_state.to_float() + float_state
```

### Issue 2: Overflow in GoldenExact

**Cause**: i128 overflow after many iterations.

**Fix**: Switch to FixedPoint64 (Q32.32):
```python
# Before (CPU, may overflow)
state = State(data)  # golden_exact

# After (GPU, bounded)
state = State(data, device='cuda')  # fixed_point64
```

### Issue 3: Slow Performance on CPU

**Cause**: GoldenExact is 25-67x slower than float.

**Fix**: Use GPU or opt-in to float:
```python
# Option A: GPU (fast + exact)
state = State(data, device='cuda')

# Option B: Float (fast, not exact)
state = State(data, dtype=float64)
```

### Issue 4: Tests Failing

**Cause**: Type-sensitive assertions.

**Fix**: Update test expectations:
```python
# Before
assert state.dtype == float64

# After
from syntonic.core.dtype import golden_exact
assert state.dtype == golden_exact
```

## Verification Checklist

After migration, verify:

- [ ] All `State()` constructors reviewed
- [ ] Constants updated to use exact types
- [ ] DHSR operators use `exact=True`
- [ ] Visualization code has `.to_float()`
- [ ] Tests updated for exact types
- [ ] Performance measured and acceptable
- [ ] Determinism tests pass
- [ ] No unexpected dtype errors

## Getting Help

If you encounter issues:

1. Check [Architecture Document](exact_arithmetic_architecture.md)
2. Review [Implementation Status](../EXACT_ARITHMETIC_STATUS.md)
3. See example in `tests/test_exact_arithmetic.py`
4. Ask on GitHub: https://github.com/anthropics/syntonic/issues

## Example: Complete Migration

### Before (Float-based)
```python
# old_code.py
from syntonic.core import State
from syntonic.exact import PHI_NUMERIC
from syntonic.crt.operators import DifferentiationOperator, HarmonizationOperator

# Float by default
state = State([1.0, 2.0, 3.0])

# Numeric constants
alpha = 0.1 * PHI_NUMERIC

# DHSR operators
D = DifferentiationOperator(alpha_0=alpha)
H = HarmonizationOperator()

# Compute
result = H.apply(D.apply(state))
print(result.to_list())
```

### After (Exact-first)
```python
# new_code.py
from syntonic.core import State
from syntonic.exact import PHI
from syntonic.crt.operators import DifferentiationOperator, HarmonizationOperator

# Exact by default
state = State([1.0, 2.0, 3.0])  # Automatically golden_exact

# Exact constants
alpha_exact = PHI * 0.1
alpha = alpha_exact.eval()  # Evaluate when needed

# DHSR operators with exact mode
D = DifferentiationOperator(alpha_0=alpha)
H = HarmonizationOperator(exact=True)  # Explicit exact

# Compute (maintains exactness)
result = H.apply(D.apply(state))

# Convert to float for display
print(result.to_float().to_list())
```

---

*Last Updated: 2026-02-03*  
*Version: 1.0*  
*Compatibility: Syntonic v0.1.0+*
