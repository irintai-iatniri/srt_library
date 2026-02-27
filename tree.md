# srt_library — Directory Tree

*Last updated: 2026-02-25*

```
srt_library/
├── core/                              UNIFIED PYTHON PACKAGE (289 modules)
│   ├── __init__.py
│   ├── _core/                         Compiled Rust extension
│   │   ├── _core.cpython-312-*.so     ~460 PyO3 bindings (23.6 MB)
│   │   └── _libs/                     Bundled libs (CUDA, OpenBLAS, gfortran)
│   ├── core/                          Foundational types
│   │   ├── state.py                   State: fundamental information object
│   │   ├── dtype.py                   DType system
│   │   ├── device.py                  CPU/CUDA device management
│   │   ├── constants.py               φ, E*, q and group theory constants
│   │   ├── backend.py                 Backend selection
│   │   └── types.py                   Type definitions
│   ├── nn/                            Neural networks (58 modules)
│   │   ├── activations/               Golden GELU, syntonic activations
│   │   ├── architectures/             GnosticOuroboros, etc.
│   │   ├── layers/                    Linear, conv, embedding, attention
│   │   ├── loss/                      Syntony loss, resonant loss
│   │   ├── optim/                     DHSR-based optimizers
│   │   ├── training/                  Training loops
│   │   ├── winding/                   Winding number layers
│   │   ├── golden_gelu.py
│   │   └── resonant_tensor.py
│   ├── sna/                           Syntonic Neural Architecture (162 modules)
│   │   ├── core/                      SNA core
│   │   ├── interfaces/                External interfaces
│   │   ├── irintAI/                   Agent integration
│   │   ├── irintai_mcp/              MCP tools
│   │   ├── services/                  Background services
│   │   ├── sgc_builder/              SGC codec builder
│   │   ├── soma/                      Embodiment layer
│   │   ├── tools/                     10 tool categories
│   │   │   ├── affective/
│   │   │   ├── agency/
│   │   │   ├── body/
│   │   │   ├── governance/
│   │   │   ├── knowledge/
│   │   │   ├── memory/
│   │   │   ├── oracle_system/
│   │   │   ├── processing/
│   │   │   └── search/
│   │   ├── visualization/
│   │   └── world_reports/
│   ├── physics/                       Standard Model (29 modules)
│   │   ├── bosons/                    gauge.py, higgs.py
│   │   ├── fermions/                  leptons.py, quarks.py, windings.py
│   │   ├── hadrons/                   masses.py
│   │   ├── mixing/                    ckm.py, pmns.py
│   │   ├── neutrinos/                 masses.py
│   │   ├── running/                   rg.py
│   │   ├── validation/                pdg.py
│   │   └── srt_physics.py
│   ├── crt/                           DHSR operators
│   ├── srt/                           SRT theory: corrections, geometry, lattice, spectral
│   ├── consciousness/                 gnosis.py
│   ├── exact/                         Exact arithmetic wrappers
│   ├── hypercomplex/                  Quaternion/Octonion
│   ├── linalg/                        Linear algebra (numpy-free)
│   ├── resonant/                      Resonant engine, transformer, retrocausal
│   ├── sn/                            Syntonic Neural base
│   ├── markets/                       s_market.py
│   ├── viz/                           Visualization
│   ├── golden.py, fft.py, distributed.py, domains.py, exceptions.py, jit.py
│   ├── srt_math.py, srt_random.py
│   └── tree.md
│
├── _exact/                            EXACT ARITHMETIC BUILD
│   ├── rust/
│   │   ├── src/                       71 Rust source files
│   │   │   ├── lib.rs                 PyO3 module root
│   │   │   ├── exact/                 GoldenExact, Rational, SymExpr, etc.
│   │   │   ├── hypercomplex/          Quaternion, Octonion, Sedenion
│   │   │   ├── tensor/               Storage, CUDA, data loading
│   │   │   ├── resonant/             Attractor, crystallize, syntony
│   │   │   ├── sna/                  Network, resonant oscillator
│   │   │   ├── autograd/             Backward pass
│   │   │   ├── linalg/               Matrix multiply
│   │   │   ├── spectral.rs           Heat kernel, theta series
│   │   │   ├── hierarchy.rs          Correction hierarchy
│   │   │   ├── vibe.rs               HCB2 substrate
│   │   │   └── winding.rs            W⁴ winding states
│   │   └── kernels/                   33 CUDA .cu files
│   │       └── ptx/                   Compiled for sm_75/80/86/90
│   ├── exact_arithmetic/              Inner Cargo workspace
│   ├── syntonic/                      Compiled package output
│   ├── python/                        Legacy Python wrappers
│   ├── tests/                         25 test files
│   ├── Cargo.toml, pyproject.toml
│   └── *.md                           Historical docs (pre-restructure)
│
├── _float/                            FLOAT ARITHMETIC BUILD
│   ├── rust/src/                      Float-precision Rust
│   ├── float_arithmetic/              Inner Cargo workspace
│   ├── nn/                            Float NN (resonant_tensor.py)
│   └── tests/
│
└── theory_unique_components/          THEORY-SPECIFIC MODULES
    ├── SRT/                           Syntony Recursion Theory
    │   ├── constants.py
    │   ├── corrections/               Correction factors
    │   ├── geometry/                  Torus, winding
    │   ├── golden/                    Measure, recursion
    │   ├── lattice/                   E₈, D₄, golden cone
    │   ├── spectral/                  Heat kernel, theta, Möbius
    │   ├── fermat_forces.py
    │   ├── lucas_shadow.py
    │   ├── mersenne_matter.py
    │   ├── prime_selection.py
    │   └── transcendence.py
    └── crt/                           Cosmological Recursion Theory
        ├── dhsr_fused/                Fused DHSR evolution/loop/reference
        ├── operators/                 D̂, Ĥ, Ŝ, R̂, Gnosis, Möbius
        └── extended_hierarchy.py
```
