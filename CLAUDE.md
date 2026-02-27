# CLAUDE.md — srt_library

## Structure (Feb 25 2026)

```
srt_library/
├── core/                        Unified Python package (65 modules)
│   ├── __init__.py              Re-exports State, DType, exact types, consciousness, etc.
│   ├── _core/                   Compiled Rust extension (.so, 461 exports)
│   ├── state.py                 State: fundamental information object
│   ├── dtype.py                 DType system (float32/64, complex, winding, golden_exact)
│   ├── device.py                CPU/CUDA device management
│   ├── constants.py             φ, E*, q, group theory constants
│   ├── exact.py                 Exact arithmetic: GoldenExact, Rational, SymExpr, φ, primes
│   ├── hypercomplex.py          Quaternion, Octonion, Sedenion wrappers
│   ├── linalg.py                Linear algebra (matmul, bmm, corrected mm)
│   ├── sn.py                    Syntonic Neural base (ResonantOscillator, Hilbert kernel)
│   ├── srt_math.py              SRT math functions (replaces Python math)
│   ├── srt_random.py            Golden-ratio RNG
│   ├── fft.py                   FFT implementation
│   ├── jit.py                   JIT compilation
│   ├── distributed.py           Distributed compute
│   ├── exceptions.py            Error types
│   ├── server.py                Visualization websocket server
│   ├── nn/                      Neural networks (43 modules)
│   │   ├── resonant_tensor.py   Python wrapper around Rust ResonantTensor
│   │   ├── golden_gelu.py       Golden GELU activation
│   │   ├── layers/              DHSR layers, normalization, gates
│   │   ├── loss/                Syntony loss, phase alignment, regularization
│   │   ├── optim/               Golden momentum, retrocausal RES
│   │   ├── training/            Trainers, callbacks, metrics
│   │   ├── winding/             Winding number layers, Fermat/Mersenne/Lucas
│   │   ├── activations/         Gnosis GELU
│   │   └── analysis/            Archonic detection, escape, health
│   └── prompt_core/             Prompt templates
├── _exact/                      Exact arithmetic Rust/CUDA build
│   ├── rust/src/                71 Rust source files
│   ├── rust/kernels/            33 CUDA .cu files → PTX (sm_75/80/86/90)
│   ├── syntonic/                Compiled package (_core.so, 461 bindings)
│   └── pyproject.toml           maturin config
├── _float/                      Float arithmetic Rust/CUDA build
│   ├── rust/src/                Float-precision Rust source
│   └── float_arithmetic/        Inner Cargo workspace
└── theory_unique_components/    Theory-specific modules (63 files)
    ├── srt/                     SRT: constants, geometry, lattice, spectral, golden
    ├── crt/                     CRT: DHSR operators (D̂, Ĥ, Ŝ, R̂, Gnosis, Möbius)
    ├── resonant/                Resonant engine, transformer, retrocausal, embedding
    ├── GnosticOuroboros/        Bimodal quaternion architecture components
    ├── syntonic_transformer.py  Theory-specific architectures
    ├── syntonic_mlp.py
    ├── syntonic_attention.py
    ├── syntonic_cnn.py
    ├── embeddings.py
    └── cosmological_block.py
```

## Build

```bash
cd srt_library/_exact
CUDA_PATH=/usr/local/cuda maturin develop --release
# Produces: syntonic/_core.cpython-312-x86_64-linux-gnu.so
```

Requirements: Python 3.12+, Rust 1.91+, CUDA 12.0+, maturin 1.11.5+

## Constraint

No PyTorch, NumPy, SciPy. Everything via Rust/CUDA `_core` or pure Python.

## Import Patterns

```python
# Package root
import srt_library.core as syn
psi = syn.state([1, 2, 3, 4])
print(psi.syntony)

# Exact arithmetic
from srt_library.core import PHI, GoldenExact, Rational
phi_sq = PHI * PHI  # exact: φ² = φ + 1

# Rust backend directly
from srt_library.core._core import ResonantTensor, WindingState

# Neural networks
from srt_library.core.nn import ResonantTensor, GoldenGELU
from srt_library.core.nn.layers import DifferentiationLayer, HarmonizationLayer

# Theory modules (separate package)
from srt_library.theory_unique_components.crt.operators import DifferentiationOperator
from srt_library.theory_unique_components.srt.lattice.e8 import E8Lattice
```

## Key Constants

All from {φ, π, e, 1}:
- `PHI` = (1+√5)/2 ≈ 1.618034
- `E_STAR` = e^π − π ≈ 19.999100
- `Q` = (2φ + e/(2φ²)) / (φ⁴ × E*) ≈ 0.027395
- `COLLAPSE_THRESHOLD` = K(D₄) = 24

## Related Packages

| Package | Location | Purpose |
|---------|----------|---------|
| Physics engine | `applications/domains/physics/` | SM derivation (96 particles) |
| SRT-Zero | `applications/srt_zero/` | Particle mass calculator |
| SNA | `architectures/sna/` | Syntonic Neural Architecture |
| GnosticOuroboros | `architectures/GnosticOuroboros/` | Bimodal quaternion arch |
| irintAI | `irintAI/` | Agent, MCP, OS, market |
