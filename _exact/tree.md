# srt_library - Directory Tree

*Last updated: 2026-02-25*

## Current Structure

```
srt_library/
├── _core/                         # Compiled extension stub
│   └── __init__.py
├── core/                          # Python core abstractions
│   ├── backend.py                 # Backend selection (Rust/CUDA/CPU)
│   ├── constants.py               # SRT constants (φ, E*, q)
│   ├── device.py                  # Device management (CPU/CUDA)
│   ├── dtype.py                   # DType system
│   ├── state.py                   # State: fundamental information object
│   └── types.py                   # Type definitions
├── docs/                          # Documentation
│   ├── README.md                  # Docs overview
│   ├── exact_arithmetic_api.md    # Exact arithmetic API reference
│   ├── exact_arithmetic_architecture.md  # SGC architecture
│   └── migrating_to_exact.md      # Migration guide
├── exact_arithmetic/              # Exact-precision Rust/CUDA workspace
│   ├── rust/
│   │   ├── kernels/               # 35 CUDA kernel files (.cu)
│   │   │   └── ptx/               # Compiled PTX (sm_75/80/86/90)
│   │   └── src/                   # Rust source
│   │       ├── autograd/          # Automatic differentiation
│   │       ├── exact/             # GoldenExact, Rational, SymExpr, etc.
│   │       ├── hypercomplex/      # Quaternion, Octonion, Sedenion
│   │       ├── linalg/            # Matrix operations
│   │       ├── resonant/          # Resonant tensor, attractor, crystallize
│   │       ├── sna/               # Syntonic Neural Architecture
│   │       ├── tensor/            # Core tensor storage + CUDA integration
│   │       ├── lib.rs             # Main library entry (~460 PyO3 bindings)
│   │       ├── spectral.rs        # Heat kernels, theta series
│   │       ├── vibe.rs            # HCB2 Vibe substrate
│   │       └── winding.rs         # W⁴ winding states
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── README.md
├── experiments/                   # Experimental code
│   ├── complex_conv.py
│   ├── complex_linear/
│   └── theory_of_mind_experiment.py
├── python/                        # Python API layer (14 submodules)
│   ├── consciousness/             # gnosis.py
│   ├── corrections/               # factors.py
│   ├── crt/                       # DHSR operators, extended hierarchy
│   │   ├── dhsr_fused/            # Fused DHSR evolution/loop/reference
│   │   └── operators/             # D̂, Ĥ, Ŝ, R̂, Gnosis, Möbius, projectors
│   ├── exact/                     # Exact arithmetic wrappers
│   ├── geometry/                  # torus.py, winding.py
│   ├── golden/                    # measure.py, recursion.py
│   ├── hypercomplex/              # Quaternion/Octonion Python wrappers
│   ├── lattice/                   # e8.py, d4.py, golden_cone.py
│   ├── linalg/                    # Linear algebra (numpy-free)
│   ├── resonant/                  # DHSR block, transformer, embedding, retrocausal
│   ├── sn/                        # Syntonic Neural
│   ├── spectral/                  # heat_kernel.py, theta_series.py, mobius.py
│   ├── constants.py               # Python-side constants
│   └── golden_random.py           # Golden-ratio random number generation
├── rust/                          # Main Rust backend (active build target)
│   ├── kernels/                   # 35 CUDA kernel files
│   │   └── ptx/                   # Compiled PTX for sm_75/80/86/90
│   ├── src/                       # Rust source
│   │   ├── autograd/              # backward.rs
│   │   ├── exact/                 # 12 files: golden.rs, rational.rs, etc.
│   │   ├── hypercomplex/          # quaternion.rs, octonion.rs, sedenion.rs
│   │   ├── linalg/                # matmul.rs
│   │   ├── resonant/              # 13 files: tensor.rs, attractor.rs, etc.
│   │   ├── sna/                   # network.rs, resonant_oscillator.rs
│   │   ├── tensor/                # storage.rs, cuda/, srt_kernels.rs, etc.
│   │   └── lib.rs                 # ~460 PyO3 bindings
│   ├── tests/
│   │   └── verify_exact_math.rs
│   └── Cargo.toml
├── src/                           # Standalone Rust source
│   └── vibe.rs                    # HCB2 Vibe substrate stub
├── syntonic/                      # Compiled Python package
│   ├── __init__.py
│   ├── _core.cpython-312-x86_64-linux-gnu.so  # Compiled Rust extension
│   └── crt/operators/harmonization.py
├── tests/                         # 25 Python test files
│   ├── test_srt_implementation.py
│   ├── test_syntonic_basic.py
│   ├── test_convergence_benchmark.py
│   ├── test_grand_synthesis.py
│   └── ... (21 more test files)
├── CLAUDE.md                      # Claude Code guidance
├── Cargo.toml                     # Workspace Cargo.toml
├── LICENSE-COMMERCIAL.md          # Commercial license
├── LICENSE-RESEARCH.md            # Research license
├── README.md                      # Main documentation
├── REFACTORING_PLAN.md            # Refactoring roadmap
├── RUST_CODE_STRUCTURE.md         # Rust code structure reference
├── __init__.py                    # Package init
├── compile_*_kernels.{py,sh}      # CUDA kernel compilation scripts
├── generate_api_index.py          # API index generator
├── pyproject.toml                 # maturin build configuration
├── test_all_kernels.py            # Kernel test runner
└── tree.md                        # This file
```

## Relocated Components

| Component | Old Location | New Location |
|-----------|-------------|--------------|
| **Physics engine** | `srt_library/physics/` | `domains/physics/` |
| **Float arithmetic** | `srt_library/float_arithmetic/` | `srt_library_float/` |
| **srt_physics** | `domains/srt_physics/` | `domains/physics/` (consolidated) |
| **gnostic_ouroboros** | (various) | `/home/Andrew/lib/gnostic_ouroboros/` |

## Related Packages (outside srt_library)

```
ouroboros_prime/
├── srt_library/                   # ← THIS PACKAGE (exact arithmetic)
├── srt_library_float/             # Float-precision variant
├── domains/
│   └── physics/                   # SRT-Zero physics engine (96 particles)
│       ├── engine.py              # DerivationEngine
│       ├── catalog.py             # 108 particle configs
│       ├── hierarchy.py           # 60+ level corrections
│       ├── validate.py            # ValidationSuite (129 tests, 95.3%)
│       └── results/derivations.json
├── syntonic_applications/         # Application layer
└── ...

/home/Andrew/lib/
└── gnostic_ouroboros/             # Gnostic Ouroboros core
    ├── core.py
    ├── layers.py
    ├── physics.py
    └── tensor.py
```

## Build Output

After `maturin develop --release`:
- `syntonic/_core.cpython-312-x86_64-linux-gnu.so` (23.6 MB)
- CUDA PTX compiled for sm_75, sm_80, sm_86, sm_90
