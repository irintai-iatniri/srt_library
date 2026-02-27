# Syntonic — SRT Library

**The computational substrate for Syntony Recursion Theory**

Derives Standard Model physics from four mathematical seeds — **φ**, **π**, **e**, **1** — with zero free parameters.

## Quick Start

```python
import srt_library.core as syn

# Create a state
psi = syn.state([1, 2, 3, 4])
print(psi.syntony)

# Exact golden arithmetic
phi = syn.PHI
assert (phi * phi).eval() == phi.eval() + 1  # φ² = φ + 1

# Quaternion algebra
q = syn.Quaternion(1, 2, 3, 4)

# Consciousness check
print(syn.gnosis_score(0.8, 0.6))
print(syn.COLLAPSE_THRESHOLD)  # 24 (D₄ kissing number)
```

## Build

```bash
cd srt_library/_exact
CUDA_PATH=/usr/local/cuda maturin develop --release
```

## Layout

- **core/** — 65 flat Python modules + nn/ subpackage (43 modules)
- **_exact/** — Exact arithmetic Rust/CUDA (71 .rs, 33 .cu → 461 PyO3 bindings)
- **_float/** — Float arithmetic Rust/CUDA
- **theory_unique_components/** — SRT geometry, CRT operators, resonant engine (63 modules)
