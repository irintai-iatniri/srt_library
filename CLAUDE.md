# CLAUDE.md — srt_library

Tensor library for Syntony Recursion Theory (SRT) and Cosmological Recursion Theory (CRT).
Zero external math dependencies — all operations via Rust/CUDA `_core` or pure Python.

## Constraint

**No PyTorch, NumPy, SciPy.** If a function is needed, implement it in Rust/CUDA kernels.

## Build

```bash
cd srt_library/_exact
CUDA_PATH=/usr/local/cuda maturin develop --release
# Produces: syntonic/_core.cpython-312-x86_64-linux-gnu.so (~460 bindings)
```

Requirements: Python 3.10+ (dev on 3.12), Rust 1.91+, CUDA 12.0+, maturin >=1.0

```bash
# Tests / lint
pip install ".[dev]"
pytest tests/ -v --cov=syntonic
ruff check python/ && black python/ && mypy python/syntonic/
```

## Structure

```
srt_library/
├── core/                        Python package (71 modules)
│   ├── __init__.py              Re-exports State, DType, _core bindings, nn, exact, etc.
│   ├── _core/                   Compiled Rust extension (.so, ~460 exports)
│   ├── state.py                 State: fundamental information object
│   ├── dtype.py                 DType system (golden_exact, rational, fixed_point64,
│   │                            syntonic, float32/64, complex64/128, int32/64, winding)
│   ├── device.py                CPU/CUDA device management
│   ├── constants.py             PHI, E*, Q, group theory, prime sequences, axioms
│   ├── exact.py                 GoldenExact (a+bphi), Rational, SymExpr, fibonacci, lucas
│   ├── hypercomplex.py          Quaternion, Octonion, Sedenion wrappers
│   ├── linalg.py                matmul, bmm, corrected mm
│   ├── sn.py                    Parameter, Module, Sequential base classes
│   ├── srt_math.py              Pure Python math (replaces stdlib math)
│   ├── srt_random.py            Golden-ratio RNG
│   ├── fft.py / jit.py          FFT, JIT compilation
│   ├── distributed.py           Distributed compute
│   ├── server.py                Visualization WebSocket server
│   ├── exceptions.py            SyntonicError, DeviceError, DTypeError, ShapeError, LinAlgError
│   └── nn/                      Neural networks (49 modules)
│       ├── resonant_tensor.py   Python wrapper around Rust ResonantTensor
│       ├── functional.py        Functional interfaces (gelu, softmax, etc.)
│       ├── golden_gelu.py       Golden GELU activation
│       ├── layers/              16 modules: attention, conv, embedding, recurrent,
│       │                        differentiation, harmonization, normalization,
│       │                        resonant_linear, gnosis, prime gates, pixel_ops
│       ├── loss/                syntonic_loss, phase_alignment, regularization, metrics
│       ├── optim/               Golden momentum optimizer
│       ├── training/            Trainer, gradient_trainer, callbacks, metrics
│       ├── winding/             WindingNet, DHSR block, Fermat/Mersenne/Lucas layers
│       ├── activations/         Gnosis GELU
│       └── analysis/            Archonic detection, escape, health, visualization
├── _exact/                      Exact arithmetic Rust/CUDA build
│   ├── rust/src/                72 Rust files
│   │   ├── lib.rs               PyO3 entry point (~460 bindings)
│   │   ├── exact/               GoldenExact, Rational, SymExpr, SyntonicExact (13 files)
│   │   ├── resonant/            ResonantTensor, syntony, crystallize, RES evolver (15 files)
│   │   ├── tensor/              Storage, nn_ops, conv, reduction, CUDA device mgmt (19 files)
│   │   ├── hypercomplex/        Quaternion, Octonion, Sedenion (4 files)
│   │   ├── autograd/            Backpropagation (2 files)
│   │   ├── linalg/              matmul with phi-scaling (2 files)
│   │   ├── sna/                 Syntonic Neural Architecture (3 files)
│   │   └── hierarchy.rs         Correction hierarchy (~38K lines)
│   ├── rust/kernels/            37 CUDA .cu files (sm_75/80/86/90)
│   │   (core_ops, matmul, conv, attention, golden_gelu, dhsr,
│   │    e8_projection, gnosis, winding, recurrent, embedding, etc.)
│   ├── syntonic/                Compiled output (_core.so)
│   ├── pyproject.toml           maturin config (package: "syntonic", v0.2.0)
│   └── rust/Cargo.toml          syntonic-core: pyo3 0.21, ndarray 0.15, cudarc 0.18.2
├── _float/                      Float arithmetic variant
│   ├── rust/src/                148 Rust files (float-precision)
│   └── rust/kernels/            66 CUDA .cu files
└── theory_unique_components/    Theory-specific modules (63 files)
    ├── srt/                     Constants, Fermat forces, Lucas shadow, Mersenne matter
    ├── crt/                     Extended correction hierarchy
    ├── resonant/                Resonant engine, transformer, DHSR block, retrocausal
    ├── GnosticOuroboros/         Bimodal quaternion architecture
    ├── syntonic_transformer.py / syntonic_mlp.py / syntonic_attention.py / syntonic_cnn.py
    ├── embeddings.py
    └── cosmological_block.py
```

## Import Patterns

```python
import srt_library.core as syn
psi = syn.state([1, 2, 3, 4])
psi.syntony          # 0.5
psi.differentiate().harmonize()  # DHSR cycle

# Exact arithmetic
from srt_library.core import PHI, GoldenExact, Rational
phi_sq = PHI * PHI   # exact: phi^2 = phi + 1

# Rust backend directly
from srt_library.core._core import ResonantTensor, WindingState, RESConfig

# Neural networks
from srt_library.core.nn import ResonantTensor, GoldenGELU
from srt_library.core.nn.layers import DifferentiationLayer, HarmonizationLayer
from srt_library.core.nn.layers.attention import MultiHeadSyntonicAttention
from srt_library.core.nn.layers.conv import Conv1d, Conv2d
from srt_library.core.nn.layers.recurrent import LSTM, GRU

# Theory modules
from srt_library.theory_unique_components.crt.extended_hierarchy import apply_e7_correction
from srt_library.theory_unique_components.resonant.resonant_engine_net import ResonantEngineNet
```

## Key Constants

All physics derives from four seeds: {phi, pi, e, 1}

| Constant | Value | Definition |
|----------|-------|------------|
| `PHI` | 1.618033988749895 | (1+sqrt(5))/2 |
| `E_STAR` | 19.999099979189475 | e^pi - pi |
| `Q` | 0.027395146920 | (2phi + e/(2phi^2)) / (phi^4 * E*) |
| `COLLAPSE_THRESHOLD` | 24 | K(D4) kissing number |

Group theory: E8 (h=30, dim=248, roots=240), E7 (h=18, dim=133), E6 (h=12, dim=78, fund=27), D4 (K=24)

## Key Abstractions

- **State** (`core/state.py`): Fundamental information object wrapping Rust tensor storage
- **ResonantTensor** (`core/nn/resonant_tensor.py`): Core tensor with 65+ methods (matmul, softmax, gelu, golden_gelu, syntony, crystallize, wake_flux)
- **GoldenExact** (`core/exact.py`): Exact arithmetic in Q(phi) where phi^2 = phi + 1
- **Module / Parameter / Sequential** (`core/sn.py`): Neural network base classes (like torch.nn but SRT-native)
- **DHSR cycle**: Differentiation -> Harmonization -> Syntony -> Recursion
- **RES** (`_core`): Retrocausal Evolver for training (RESConfig, ResonantEvolver, RESResult)

## DType System

| DType | Size | Description | Default |
|-------|------|-------------|---------|
| golden_exact | 16B | Q(phi) field | CPU default |
| rational | 16B | Q field | |
| fixed_point64 | 8B | Q32.32 format | GPU default |
| syntonic | 192B | Super-Field | |
| float32/64 | 4/8B | IEEE floats | secondary/preview |
| complex64/128 | 8/16B | Complex | |
| winding | - | Winding numbers | |

## Neural Network Conventions

- DHSR framework: layers in `core/nn/layers/` implement Differentiation, Harmonization, Syntony, Recursion
- Golden initialization: `variance[i] = scale * exp(-i^2 / (2*PHI))` (sub-Gaussian measure)
- Winding layers in `core/nn/winding/`: Fermat (force separation), Mersenne (matter stability), Lucas (dark sector)
- Losses in `core/nn/loss/`: SyntonicLoss, PhaseAlignmentLoss, ArchonicPenalty
- Optimizer: GoldenMomentum in `core/nn/optim/`

## Axioms

```
A1  Recursion Symmetry    S[Psi . R] = phi * S[Psi]
A2  Syntony Bound         S[Psi] <= phi
A3  Toroidal Topology     W^4 = S^1_7 x S^1_8 x S^1_9 x S^1_10
A4  Sub-Gaussian Measure  w(n) = e^{-|n|^2/phi}
A5  Holomorphic Gluing    Mobius identification at tau = i
A6  Prime Syntony         Stability iff M_p = 2^p - 1 is prime
```
