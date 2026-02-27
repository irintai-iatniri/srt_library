# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Development install (requires Rust toolchain)
maturin develop

# Release install
pip install .

# Install with optional dependencies
pip install ".[dev]"      # Development (pytest, ruff, mypy, black, hypothesis)
pip install ".[all]"
pip install ".[docs]"     # Documentation (Sphinx)

# Run tests
pytest tests/
pytest tests/test_core/test_state.py           # Single test file
pytest tests/test_core/test_state.py::test_foo # Single test function
pytest -v --cov=syntonic                       # With coverage

# Linting and formatting
ruff check syntonic/
black syntonic/
mypy syntonic/
```
## Limitations
Do not use PyTorch, NumPY, SciPy, or any other external libraries. Use only the libraries provided in this repository. If the code requires these libraries, do not use them. If a function is needed from one of these libraries implement it in the CUDA kernels and Rust. 

## Architecture

Syntonic is a hybrid Python/Rust library implementing Syntony Recursion Theory (SRT), a mathematical framework for deriving Standard Model physics from geometric structures.

### Language Split

- **Rust (`rust/src/`)**: Performance-critical tensor operations, hypercomplex types (Quaternion, Octonion), resonant tensor core, spectral computations, and exact arithmetic. Compiled via maturin/PyO3 to `syntonic._core`.
- **Python (`syntonic/`)**: High-level API, CRT/SRT operators, physics derivations, neural network modules, and applications.

### Core Modules

| Module | Purpose |
|--------|---------|
| `syntonic.core` | `State` (tensor wrapper), `DType` system (float64, complex128, winding), `Device` (cpu/cuda), `ResonantTensor`, `RESConfig`, `ResonantEvolver` |
| `syntonic.linalg` | Linear algebra (eig, svd, qr, cholesky, solve). Core ops are numpy-free; expm/logm require scipy |
| `syntonic.hypercomplex` | Quaternions and Octonions (implemented in Rust) |
| `syntonic.exact` | Exact arithmetic: `Rational`, `GoldenExact` (a + b*φ), Fibonacci/Lucas sequences |
| `syntonic.crt` | DHSR framework: Differentiation/Harmonization/Syntony/Recursion operators and metrics |
| `syntonic.srt` | SRT geometry: W⁴ torus, E₈/D₄ lattices, golden cone (36 roots), theta series, heat kernels |
| `syntonic.physics` | Standard Model derivation: fermions, bosons, hadrons, mixing matrices (CKM/PMNS), neutrinos, running couplings |
| `syntonic.nn` | Neural network layers, architectures, losses, and training utilities (requires PyTorch) |
| `syntonic.applications` | Applied domains: biology, chemistry, condensed matter, consciousness, ecology, thermodynamics |

### Rust Backend Structure

| Path | Purpose |
|------|---------|
| `rust/src/lib.rs` | Main entry point, PyO3 module exports |
| `rust/src/tensor/` | Core tensor storage with BLAS/LAPACK integration |
| `rust/src/resonant/` | Resonant tensor operations and evolution |
| `rust/src/linalg/` | Eigenvalue decomposition, SVD, etc. |
| `rust/src/exact/` | Exact golden ratio arithmetic, rationals |
| `rust/src/hypercomplex/` | Quaternion and Octonion implementations |
| `rust/src/spectral.rs` | Spectral computations for SRT |
| `rust/src/winding.rs` | W⁴ winding number operations |

### Key Abstractions

- **State**: The fundamental object representing information configurations. Wraps Rust tensor storage.
- **ResonantTensor**: Core tensor type with mode norm tracking and precision control.
- **DHSR cycle**: `psi.differentiate().harmonize()` or `RecursionOperator.apply(psi)` for state evolution.
- **SRTSystem**: `create_srt_system()` initializes E₈ lattice (240 roots), golden cone (36 roots), spectral operators.
- **StandardModel**: `syn.physics.StandardModel()` derives all 25+ SM parameters from geometry (zero free parameters).

### Mathematical Constants

- `PHI` / `PHI_NUMERIC`: Golden ratio φ = (1+√5)/2
- `PHI_SQUARED`, `PHI_INVERSE`: φ² and 1/φ
- `E_STAR_NUMERIC`: Spectral constant e^π - π ≈ 19.999
- `Q_DEFICIT_NUMERIC`: Universal syntony deficit q ≈ 0.027395
- `STRUCTURE_DIMENSIONS`: Fundamental structure dimensions from SRT

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

### For Data Tensors (Advanced)

If you need spatial mode norms for W⁴ winding state representations:

```python
from syntonic.sn import compute_spatial_mode_norms
mode_norms = compute_spatial_mode_norms([8, 8, 8, 8])  # 4D torus
tensor = ResonantTensor(data, shape, mode_norms, precision)
```

## System Requirements

- Python 3.10+
- Rust 1.70+ with cargo
- System libraries: `libopenblas-dev`, `libssl-dev`
- Optional: CUDA toolkit for GPU support (feature flag `cuda`)
- Optional: PyTorch 2.0+ for neural network modules

## Project Structure

```
srt_library/
├── syntonic/          # Python source (installed as "syntonic" package)
│   ├── core/                 # State, DType, Device
│   ├── crt/                  # DHSR operators and metrics
│   ├── srt/                  # SRT geometry and spectral
│   ├── physics/              # Standard Model derivations
│   ├── nn/                   # Neural network modules
│   ├── applications/         # Applied domain implementations
│   ├── exact/                # Exact arithmetic
│   ├── hypercomplex/         # Quaternion/Octonion (Python wrappers)
│   └── linalg/               # Linear algebra
├── rust/src/                 # Rust source
│   ├── tensor/               # Tensor storage and operations
│   ├── resonant/             # Resonant tensor core
│   ├── exact/                # Exact arithmetic
│   ├── hypercomplex/         # Quaternion/Octonion
│   └── linalg/               # BLAS/LAPACK bindings
├── tests/                    # Test suite
└── pyproject.toml            # Package configuration (maturin)
```