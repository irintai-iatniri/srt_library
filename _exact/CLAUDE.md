# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Requirements

- **Python**: 3.12+
- **Rust**: 1.91+ with cargo
- **CUDA**: 12.0+ (optional, tested on RTX 3070 Ti sm_86)
- **System packages**: `libopenblas-dev libssl-dev pkg-config`
- **Build tool**: maturin 1.11.5+

## Build Commands

```bash
# Development install (requires Rust toolchain + maturin)
cd srt_library
maturin develop --release

# With CUDA support
CUDA_PATH=/usr/local/cuda maturin develop --release

# Release install
pip install .

# Install with optional dependencies
pip install ".[dev]"      # Development (pytest, ruff, mypy, black, hypothesis)
pip install ".[all]"
pip install ".[docs]"     # Documentation (Sphinx)

# Run tests
pytest tests/
pytest tests/test_srt_implementation.py    # Single test file
pytest -v --cov=syntonic                   # With coverage

# Linting and formatting
ruff check python/
black python/
mypy python/syntonic/
```

## Limitations

Do not use PyTorch, NumPy, SciPy, or any other external libraries. Use only the libraries provided in this repository. If the code requires these libraries, do not use them. If a function is needed from one of these libraries implement it in the CUDA kernels and Rust.

## Architecture

Syntonic is a hybrid Python/Rust library implementing Syntony Recursion Theory (SRT), a mathematical framework for deriving Standard Model physics from geometric structures.

### Language Split

- **Rust (`rust/src/`)**: Performance-critical tensor operations, hypercomplex types (Quaternion, Octonion, Sedenion), resonant tensor core, spectral computations, exact arithmetic. Compiled via maturin/PyO3 to `syntonic._core` (~460 bindings).
- **Python (`python/`)**: High-level API, CRT/SRT operators, physics derivations, neural network modules.
- **CUDA (`rust/kernels/`)**: 35 custom kernel files compiled to PTX for sm_75/80/86/90.

### Core Modules

| Module | Purpose |
|--------|---------|
| `syntonic.core` | `State` (tensor wrapper), `DType` system, `Device` (cpu/cuda), `ResonantTensor`, `RESConfig` |
| `syntonic.linalg` | Linear algebra (eig, svd, qr, cholesky, solve) — numpy-free |
| `syntonic.hypercomplex` | Quaternions, Octonions, Sedenions (implemented in Rust) |
| `syntonic.exact` | Exact arithmetic: `Rational`, `GoldenExact` (a + b·φ), Fibonacci/Lucas |
| `syntonic.crt` | DHSR framework: Differentiation/Harmonization/Syntony/Recursion operators |
| `syntonic.srt` | SRT geometry: W⁴ torus, E₈/D₄ lattices, golden cone, theta series, heat kernels |
| `domains.physics` | **Physics engine**: 96 particle derivations, auto-miner, 60+ level correction hierarchy |
| `domains.physics.sm` | Standard Model subpackage: fermions, bosons, hadrons, mixing (CKM/PMNS), neutrinos |

### SRT Physics Engine (`domains/physics/`)

The physics validation suite has been relocated to `domains/physics/` (moved from `srt_library/physics/`):

| File | Purpose |
|------|---------|
| `engine.py` | `DerivationEngine` — derives any particle from geometric formula |
| `catalog.py` | 108 particle configurations with PDG values and formula types |
| `hierarchy.py` | Universal Syntony Correction Hierarchy (60+ levels) |
| `corrections.py` | Complete correction factor implementation |
| `auto.py` | Auto-miner: brute-force formula discovery |
| `validate.py` | `ValidationSuite` — runs 129 tests, 95.3% pass rate |
| `operators.py` | Five Operators of Existence from Recursion Axiom |
| `geometry.py` | Geometric invariants catalog |
| `cli.py` | CLI: `python -m domains.physics.cli derive proton` |
| `results/derivations.json` | 96 validated derivations with formulas |

### Relocated Components

| Component | Old Location | New Location |
|-----------|-------------|--------------|
| Physics engine | `srt_library/physics/` | `domains/physics/` |
| Float arithmetic | `srt_library/float_arithmetic/` | `srt_library_float/` (top-level) |
| gnostic_ouroboros | (scattered) | `/home/Andrew/lib/gnostic_ouroboros/` |

### Validation Commands

```bash
# Full validation suite (129 tests)
cd ouroboros_prime
python -m domains.physics.validate

# Derive a single particle
python -c "
from domains.physics.engine import DerivationEngine
engine = DerivationEngine()
result = engine.derive('proton')
print(f'Proton: {result.final_value:.6f} MeV')  # 938.271 MeV
"

# CLI
python -m domains.physics.cli validate
python -m domains.physics.cli derive proton
python -m domains.physics.cli mine 125250 --tolerance 0.1
python -m domains.physics.cli list
```

### Validation Results (January 2026)

- **129 tests, 123 passed (95.3%)**
- **96 particles**: 78 EXACT (<0.01%), 5 VERY GOOD (<0.1%), 3 GOOD (<0.5%), 4 ACCEPTABLE (<1%), 6 outliers (≥1%)
- **Median error**: 0.0004%
- **Notable**: Proton 0.0001%, W boson 0.000001%, Tau 0.000002%, Top 0.000003%, Higgs 0.000005%
- **6 failures**: R_b (6.4%), V_us (4.0%), V_cb (2.4%), GW170817 echo (2.3%), CMB H₂/H₁ (1.9%), CMB H₃/H₁ (1.5%)

### Rust Backend Structure

| Path | Purpose |
|------|---------|
| `rust/src/lib.rs` | Main entry point, PyO3 module exports (~460 bindings) |
| `rust/src/tensor/` | Core tensor storage with BLAS/LAPACK integration |
| `rust/src/resonant/` | Resonant tensor operations: crystallize, attractor, syntony, softmax |
| `rust/src/linalg/` | Eigenvalue decomposition, SVD, matmul with φ-scaling |
| `rust/src/exact/` | GoldenExact (Q(φ)), Rational, SymExpr, ternary solver |
| `rust/src/hypercomplex/` | Quaternion, Octonion, Sedenion implementations |
| `rust/src/spectral.rs` | Spectral computations for SRT |
| `rust/src/winding.rs` | W⁴ winding number operations |
| `rust/src/hierarchy.rs` | Correction hierarchy in Rust (~38K lines) |
| `rust/src/vibe.rs` | HCB2 Vibe substrate (Phase-native computation) |
| `rust/src/sna/` | Syntonic Neural Architecture |

### Key Abstractions

- **State**: The fundamental object representing information configurations. Wraps Rust tensor storage.
- **ResonantTensor**: Core tensor type with 65 methods (matmul, softmax, gelu, golden_gelu, syntony, crystallize, wake_flux, etc.)
- **DHSR cycle**: `psi.differentiate().harmonize()` or `RecursionOperator.apply(psi)` for state evolution.
- **DerivationEngine**: `engine.derive("proton")` derives particle mass from pure geometry.
- **GoldenExact**: Exact arithmetic in Q(φ) where φ² = φ + 1 exactly.

### Mathematical Constants

All physics derives from four seeds: **φ** (golden ratio), **π**, **e**, **1**

- `PHI` = (1+√5)/2 ≈ 1.618033988749895
- `E_STAR` = e^π − π ≈ 19.999099979189474
- `Q` = (2φ + e/(2φ²)) / (φ⁴ × E*) ≈ 0.027395146920

From these, every particle mass follows: `E*/D × N × (1 ± q/divisor)`

### Import Fix Notes

The `srt_physics/hierarchy.py` imports constants from `syntonic_applications.core.constants`. If you see `ImportError: cannot import name 'H_E8'` or similar, ensure hierarchy.py imports all needed group theory constants:

```python
from syntonic_applications.core.constants import (
    PHI, PHI_INV, PI, E, E_STAR, Q, GEOMETRIC_DIVISORS,
    H_E8, DIM_E8, ROOTS_E8, ROOTS_E8_POS, RANK_E8,
    DIM_E6, DIM_E6_FUND, ROOTS_E6_POS,
    K_D4, RANK_D4, DIM_T4, N_GEN,
    FIBONACCI, MERSENNE_EXPONENTS, LUCAS_PRIMES_INDICES,
)
```

## Mode Norm Theory for Neural Networks

### TL;DR
- **Data tensors**: May use spatial mode norms if representing W⁴ states (rare)
- **ResonantTensor**: Tracks precision level and mode norm weighting

### Golden Initialization

The `init='golden'` method uses the SRT sub-Gaussian measure:

```python
variance[i] = scale * exp(-|n|²/(2φ)) = scale * exp(-(i*i)/(2*PHI))
```

This concentrates weight in low-mode parameters (fundamentals) and rapidly
decreases for high-mode parameters (complex interactions).

## Project Structure

```
srt_library/
├── syntonic/          # Compiled Python package (contains _core.so)
├── python/            # Python source modules
│   ├── crt/           # DHSR operators and metrics
│   ├── exact/         # Exact arithmetic
│   ├── hypercomplex/  # Quaternion/Octonion wrappers
│   ├── lattice/       # E₈, D₄, Golden Cone
│   ├── linalg/        # Linear algebra
│   ├── resonant/      # Resonant engine modules
│   ├── spectral/      # Heat kernel, theta series, Möbius
│   └── sn/            # Syntonic Neural
├── rust/              # Rust source
│   ├── src/           # All Rust modules
│   └── kernels/       # 35 CUDA kernel files
├── exact_arithmetic/  # Exact-precision Rust/CUDA workspace
├── experiments/       # Experimental code
├── core/              # Python core (State, DType, Device)
├── tests/             # Test suite
├── docs/              # Documentation
└── pyproject.toml     # Package configuration (maturin)

# Related packages (outside srt_library):
ouroboros_prime/
├── srt_library_float/ # Float-precision variant (extracted)
├── domains/physics/   # SRT-Zero physics engine (96 particles, moved here)
└── /home/Andrew/lib/gnostic_ouroboros/  # Gnostic Ouroboros core
```
