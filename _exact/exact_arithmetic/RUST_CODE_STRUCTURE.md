# Syntonic Rust Code Structure and Functions

## Overview

The Syntonic library uses **Rust as its high-performance backend** for tensor operations, mathematical computations, and GPU acceleration. Located in `/srt_library/rust/`, the Rust codebase implements core SRT (Syntony Recursion Theory) operations with zero dependency on NumPy, PyTorch, or floating-point approximations.

### Project Setup

- **Location**: `/srt_library/rust/`
- **Cargo Workspace**: Root workspace in `/srt_library/Cargo.toml` with Rust package in `/srt_library/rust/`
- **Package Name**: `syntonic-core`
- **Output Library**: `_core` (compiled as PyO3 extension module)
- **Build System**: `maturin` for seamless Rust/Python integration
- **Default Features**: CPU and CUDA support enabled
- **Rust Edition**: 2021

---

## Directory Structure

```
srt_library/rust/
├── src/                          # Main Rust source code
│   ├── lib.rs                   # Library root with module declarations
│   ├── vibe.rs                  # Fixed-point complex number arithmetic
│   ├── exact/                   # Exact arithmetic module
│   ├── tensor/                  # Tensor storage and operations
│   ├── spectral.rs              # Spectral operations (heat kernels, theta series)
│   ├── autograd.rs              # Automatic differentiation
│   ├── golden_gelu.rs           # Golden-measure GELU activation
│   ├── hierarchy.rs             # Hierarchy corrections and constants
│   ├── hypercomplex.rs          # Hypercomplex numbers (Quaternions, Octonions, Sedenions)
│   ├── linalg.rs                # Linear algebra operations
│   ├── math_utils.rs            # Mathematical utilities
│   ├── prime_selection.rs       # Prime number enumeration
│   ├── resonant.rs              # Resonant neural networks
│   ├── transcendence.rs         # Transcendental constant computations
│   ├── winding.rs               # Winding state enumeration
│   ├── gnosis.rs                # Masking and knowledge representation
│   ├── sna.rs                   # SNA operations
│   └── resonance_test.rs        # Integration tests
├── kernels/                      # Pre-compiled CUDA kernels (PTX format)
├── bin/                          # Executable binaries
└── Cargo.toml                    # Rust package manifest
```

---

## Core Modules

### 1. **lib.rs** - Module Root and FFI Layer
**Location**: `/srt_library/rust/src/lib.rs`

**Purpose**: Central hub that:
- Declares all submodules (public and private)
- Exposes functions to Python via PyO3
- Registers CUDA kernels and operations
- Initializes the Python module `_core`

**Key Exports**:
- PyO3 module initialization for Python integration
- Tensor operations (`TensorStorage`, `srt_compute_syntony`, `srt_dhsr_cycle`)
- CUDA operations (`srt_apply_correction`, `srt_e8_batch_projection`)
- Autograd functions (`py_backward_add`, `py_backward_mul`)
- Spectral operations (eigenvalues, heat kernels, theta series)
- Hypercomplex numbers (Quaternion, Octonion, Sedenion)
- Broadcasting operations (inflationary broadcast for consciousness)
- Causal history tracking for DHSR computation graphs
- Winding state enumeration
- Data loading and CSV parsing

---

### 2. **exact/** - Exact Arithmetic Module
**Location**: `/srt_library/rust/src/exact/`

**Purpose**: Implements non-floating-point arithmetic required for rigorous SRT mathematics. Unlike IEEE-754 floats, these types preserve exact relationships between fundamental constants.

**Submodules**:

#### **constants.rs**
- Defines the five fundamental constants:
  - **π** (pi) - circle constant, toroidal topology
  - **e** (euler) - natural base, exponential evolution  
  - **φ** (phi) - golden ratio, recursion symmetry (algebraic: x² - x - 1 = 0)
  - **E\*** (e_star) - spectral Möbius constant = e^π - π
  - **q** - universal syntony deficit ≈ 0.027395, the fundamental scale of SRT
- Enums for `FundamentalConstant` and correction levels
- Provides rigorous mathematical definitions

#### **rational.rs** - `Rational` Type
- Represents Q (rational numbers) with exact integer arithmetic
- All denominators are positive
- Arithmetic: addition, subtraction, multiplication, division, modulo
- Conversion to/from floats with controlled precision loss
- Simplification and GCD operations
- Type: `Rational(i128, i128)` = numerator/denominator

#### **golden.rs** - `GoldenExact` Type
- Represents Q(√5) = Q(φ), the golden field
- Form: `a + b√5` where a, b are rationals
- Implements exact field arithmetic for golden ratio relationships
- Critical for: mode frequencies, recursive scaling, harmonic ratios
- Preserves algebraic identity: φ² = φ + 1

#### **fixed.rs** - `FixedPoint64` Type
- Q32.32 fixed-point representation (32 bits integer, 32 bits fractional)
- Avoids floating-point rounding errors for real-time systems
- Operations: addition, multiplication (with post-shift), exponentiation, logarithm
- Range: -2^31 to 2^31 with 2^-32 precision
- Used throughout tensor operations for reproducibility

#### **symexpr.rs** - `SymExpr` Type
- Symbolic expressions with π, e, E*, q, and operations
- Lazily evaluated for performance
- Form: `Term(coeff, π_power, e_power, q_power, ...)`
- Supports compilation to fixed-point or floating-point
- Critical for spectral zeta functions and partition functions

#### **transcendental.rs**
- Computes transcendental constants from rational expressions
- Approximates π, e, E* to high precision
- Used for initialization and validation

#### **traits.rs** - `ExactScalar` Trait
- Common interface for all exact types
- Methods: `zero()`, `one()`, `from_f64()`, `to_f64()`, etc.
- Enables generic operations across Rational, GoldenExact, FixedPoint64, SymExpr

#### **syntonic.rs** - `SyntonicExact` Type
- Combines golden field with transcendental constants
- Represents Q(φ, π, e, E*, q)
- Complete number system for SRT mathematics

#### **Additional Submodules**:
- **dual.rs**: `SyntonicDual` - dual number type for automatic differentiation
- **pythagorean.rs**: `PythagoreanTriple` - generates frequency ladders from Pythagorean triples
- **rotator.rs**: `RationalRotator` - rational approximations of rotations
- **ternary_solver.rs**: `TernarySolver` - solves ternary quadratic equations

---

### 3. **tensor/** - Tensor Storage and Operations
**Location**: `/srt_library/rust/src/tensor/`

**Purpose**: NumPy-free tensor operations with CPU and optional CUDA acceleration.

**Submodules**:

#### **storage.rs** - `TensorStorage` Type
- **Size**: 6053 lines (largest module)
- Core tensor operations without NumPy
- Supports N-dimensional arrays with multiple data types
- Features:
  - **CPU Backend**: ndarray + ndarray_linalg for dense operations
  - **CUDA Backend**: cudarc for GPU acceleration with device management
  - **Memory Pooling**: Optimized device memory allocation
  - **Multi-GPU**: Distributed operations across multiple GPUs
  - **Kernel Caching**: Pre-compiled PTX kernels for different GPU architectures
  - **Async Operations**: Overlapped GPU transfers with computation
- Operations: matrix multiplication, eigendecomposition, QR, SVD, determinant, inverse, solve
- Fixed-point operations: `srt_compute_syntony_fp64`, `srt_dhsr_cycle_fp64`, `srt_differentiation_fp64`, `srt_harmonization_fp64`, `srt_laplacian_1d_fp64`
- Correction application: `srt_apply_correction`
- Golden measure: `srt_golden_gaussian_weights`, `srt_scale_phi`
- Memory resonance: `srt_memory_resonance`, `srt_pool_stats`, `srt_reserve_memory`

#### **srt_kernels.rs**
- CUDA kernels for SRT operations
- Pre-computed constants: PHI (golden ratio), Q_DEFICIT
- Kernel implementations:
  - Syntony computation (core DHSR metric)
  - DHSR cycle execution (D̂ → Ĥ → Syntony → Recursion)
  - E8 batch projection (geometric embedding)
  - Theta series evaluation
  - Correction operators
- Supports both single-precision and double-precision kernels

#### **cuda/** - CUDA Infrastructure
- **device_manager.rs**: Device allocation, CUDA context management
- **async_transfer.rs**: Asynchronous GPU memory transfers with overlap
- **multi_gpu.rs**: Multi-GPU coordination
- Memory pooling and stream management
- Custom CUDA error handling

#### **data_loading.rs**
- `SRTBinaryLoader`: Loads binary tensor data
- `SRTCSVParser`: Parses CSV files
- `SRTDataPipeline`: Streaming data pipeline
- `GoldenExactConverter`: Converts data to exact arithmetic
- `DataBatch`, `DataType`, `Endianness`: Data format specifications

#### **broadcast.rs**
- **Inflationary Broadcasting**: Extends tensors with golden-measure weighting
- **Consciousness Broadcasting**: Special broadcasting for consciousness field
- Symmetry-preserving tensor expansion

#### **causal_history.rs**
- `PyCausalHistoryTracker`: Tracks computation graphs for DHSR
- `d4_consciousness_threshold`: D4 algebra thresholds
- Enables end-to-end gradient computation through DHSR cycles

#### **precision_policy.rs**
- `PrecisionPolicy`: Controls floating-point precision per operation
- `get_srt_operation_policy()`: Returns optimal precision for SRT operations
- Balances accuracy vs. performance

#### **srt_optimization.rs**
- `GoldenMomentum`: Golden-measure momentum optimizer
- Replaces Adam with SRT-native optimization
- Convergence guarantees via syntony dynamics

#### **conv.rs**
- Convolution operations for neural networks
- Supports CUDA acceleration

#### **reduction.rs**
- Reduction operations (sum, mean, max, min)
- CUDA-optimized

---

### 4. **spectral.rs** - Spectral Analysis
**Location**: `/srt_library/rust/src/spectral.rs`

**Purpose**: High-performance spectral operations for theta series, heat kernels, and zeta functions.

**Key Functions**:
- `theta_series_evaluate()`: Θ(t) = Σ_n w(n) exp(-π|n|²/t)
- `theta_series_weighted()`: Weighted theta series with custom weights
- `theta_series_derivative()`: Derivative ∂Θ/∂t
- `heat_kernel_trace()`: K(t) = Σ_n exp(-λ_n t)
- `heat_kernel_weighted()`: Weighted heat kernel
- `heat_kernel_derivative()`: Derivative ∂K/∂t
- `spectral_zeta()`: ζ(s) = Σ_n λ_n^(-s) (spectral zeta function)
- `spectral_zeta_weighted()`: Weighted spectral zeta
- `partition_function()`: Z(t) = Tr(e^(-tĤ))
- `compute_eigenvalues()`: Eigendecomposition of spectral operators
- `compute_golden_weights()`: Golden-measure weights ∝ exp(-n²/φ)
- `count_by_generation()`: Generation statistics from winding numbers
- `filter_by_generation()`: Filter eigenmodes by generation

**Dependencies**:
- `WindingState`: For mode enumeration
- `FixedPoint64`: Fixed-point arithmetic for stability
- PHI, PI constants from `srt_kernels`

---

### 5. **vibe.rs** - Fixed-Point Complex Numbers
**Location**: `/srt_library/rust/src/vibe.rs`

**Purpose**: Fixed-point (Q32.32) complex arithmetic for real-time, deterministic computations.

**Type**: `Vibe`
- **Fields**: `real: i128`, `imag: i128` (Q32.32 format)
- **Shift**: 32 bits = 2^32 scale factor
- **Precision**: 2^-32 ≈ 2.3×10^-10
- **Q-Deficit Constant**: 117658758 (represents q in fixed-point)

**Operations**:
- Arithmetic: addition, subtraction, multiplication
- Rotation: `rotate_90()` (complex rotation by 90°)
- Drag Dynamics: `apply_drag()` (multiplies by 1 - q, harmonic dissipation)
- Conversions: to/from f64 with scaling

**Use Case**: Intermediate calculations avoiding floating-point rounding.

---

### 6. **autograd.rs** - Automatic Differentiation
**Location**: `/srt_library/rust/src/autograd.rs`

**Purpose**: Reverse-mode automatic differentiation for neural networks.

**Key Functions**:
- `py_backward_add()`: Gradient for addition
- `py_backward_mul()`: Gradient for multiplication
- `py_backward_softmax()`: Softmax gradient
- `py_backward_layernorm()`: Layer normalization gradient
- `py_backward_phi_residual()`: Golden-measure residual gradient
- `py_load_autograd_kernels()`: CUDA kernel loader for backprop

**Integration**: Works with `PyCausalHistoryTracker` for end-to-end differentiation.

---

### 7. **resonant.rs** - Resonant Neural Networks
**Location**: `/srt_library/rust/src/resonant/`

**Purpose**: DHSR-based neural network layers with guaranteed convergence.

**Features**:
- Mode normalization (ensures Σ|ψ_n|² = 1)
- Syntony tracking (recursive stability metric)
- DHSR layer composition
- Golden-measure weight initialization

---

### 8. **hierarchy.rs** - Hierarchy Corrections
**Location**: `/srt_library/rust/src/hierarchy.rs`

**Purpose**: Applies corrections that embed Standard Model physics into SRT geometry.

**Key Constants**:
- **Mersenne Primes** (generation count): M₂=3, M₃=7, M₅=31, M₇=127
- **Lucas Numbers** (shadow phase): L₄=7, L₅=11, L₆=18, L₇=29, L₁₁=199
- **Fermat Primes** (forces): F₀=3, F₁=5, F₂=17, F₃=257, F₄=65537
- **Derived Constants**: φ², φ³, φ⁴, φ⁵, E*×N batch computations

**Corrections Applied**:
- `q²/φ` - Second-order syntony deficit
- `q·φ` - Golden-syntony coupling
- `4q` - Fourth-generation leakage
- Suppression factors for rare decays
- Nested correction chains for multi-level hierarchies

**CPU Implementation**: Full functionality (CUDA pending cudarc API updates).

---

### 9. **hypercomplex.rs** - Higher-Dimensional Numbers
**Location**: `/srt_library/rust/src/hypercomplex/`

**Purpose**: Implementations of hypercomplex algebras for E8 geometry.

**Types**:
- **Quaternion**: Hamilton algebra, 4D rotation representation
- **Octonion**: Normed algebra, E₈ coordinates
- **Sedenion**: 16D algebra, higher hierarchies

**Operations**: Arithmetic, normalization, conjugation, basis transformations.

---

### 10. **linalg.rs** - Linear Algebra
**Location**: `/srt_library/rust/src/linalg.rs`

**Purpose**: Core linear algebra without NumPy.

**Key Functions**:
- `matmul()`: Matrix multiplication
- Eigendecomposition, QR, SVD via ndarray_linalg
- BLAS/LAPACK integration for performance

---

### 11. **math_utils.rs** - Mathematical Utilities
**Location**: `/srt_library/rust/src/math_utils.rs`

**Purpose**: Helper functions for SRT operations.

**Functions**:
- Trigonometric and hyperbolic functions
- Exponential and logarithm
- Special function computation
- Complex arithmetic helpers
- Registration of math utilities module for Python

---

### 12. **prime_selection.rs** - Prime Number Enumeration
**Location**: `/srt_library/rust/src/prime_selection.rs`

**Purpose**: Generate prime numbers and their relationships.

**Functions**:
- Prime generation algorithms
- Mersenne prime enumeration
- Generation-constrained prime selection
- Extended prime sequences
- Registration of extended prime selection for Python

---

### 13. **transcendence.rs** - Transcendental Computations
**Location**: `/srt_library/rust/src/transcendence.rs`

**Purpose**: Compute transcendental constants and their combinations.

**Functions**:
- π computation
- e computation
- E* = e^π - π
- Combinations and derivatives
- Registration of transcendence module for Python

---

### 14. **winding.rs** - Winding State Enumeration
**Location**: `/srt_library/rust/src/winding.rs`

**Purpose**: Enumerate winding numbers on toroidal geometries W⁴.

**Type**: `WindingState`
- Represents integer winding vectors (n₁, n₂, n₃, n₄) on W⁴
- Norm-squared: |n|² = n₁² + n₂² + n₃² + n₄²

**Iterators**:
- `enumerate_windings()`: All windings up to norm bound
- `enumerate_windings_by_norm()`: Group by norm squared
- `enumerate_windings_exact_norm()`: Fix exact norm
- `count_windings()`: Count windings within radius

**Use Cases**: Theta series evaluation, spectral mode enumeration, partition functions.

---

### 15. **gnosis.rs** - Knowledge Representation
**Location**: `/srt_library/rust/src/gnosis.rs`

**Purpose**: Masking and knowledge encoding for consciousness field.

**Features**:
- Attention masks based on SRT geometry
- Knowledge vector projection
- Registration of gnosis module for Python

---

### 16. **golden_gelu.rs** - Golden GELU Activation
**Location**: `/srt_library/rust/src/golden_gelu.rs`

**Purpose**: Smooth activation function based on golden measure.

**Function**: GELU with golden-ratio scaling
- `GELU[x] ≈ x · Φ((√(2/π))(x + 0.044715x³))`
- Golden version: weighted by exp(-|n|²/φ)

---

### 17. **sna.rs** - SNA Operations
**Location**: `/srt_library/rust/src/sna.rs`

**Purpose**: Spinor/Tensor algebra operations for E8 geometry.

---

### 18. **resonance_test.rs** - Integration Tests
**Location**: `/srt_library/rust/src/resonance_test.rs`

**Purpose**: Tests for resonant operations, DHSR cycles, and tensor operations.

---

## Build and Compilation

### Build Process

1. **Rust Compilation**:
   ```bash
   cd /srt_library
   cargo build --release  # Compile Rust backend
   ```

2. **Python Integration** (via maturin):
   ```bash
   maturin develop  # Compile + install to Python site-packages
   ```

3. **CUDA Kernels**:
   - Pre-compiled PTX kernels in `/kernels/ptx/`
   - Loaded at runtime based on GPU compute capability
   - Supports SM75 (RTX 2070+), SM80 (A100), SM86 (RTX 3090), SM90 (H100)

### Dependencies

**External Crates**:
- **pyo3**: Python-Rust FFI
- **ndarray**, **ndarray_linalg**: Dense linear algebra (CPU)
- **cudarc**: CUDA runtime and kernel management
- **num-complex**: Complex number arithmetic
- **half**: Half-precision floats
- **lazy_static**: Global initialization
- **rayon**: Parallel iteration (optional)

**No Dependencies On**:
- NumPy
- PyTorch
- TensorFlow
- SciPy
- Standard `math` module (except constants)

---

## FFI (Foreign Function Interface) to Python

The Rust code exports functions to Python via PyO3 module `_core`, organized by category:

### Core Tensor Operations
- `TensorStorage`: Main tensor type
- `srt_compute_syntony()`: Core DHSR metric
- `srt_dhsr_cycle()`: Full D̂→Ĥ→S→Ř cycle
- `srt_apply_correction()`: Hierarchy corrections

### CUDA Operations
- `cuda_is_available()`, `cuda_device_count()`
- `srt_e8_batch_projection()`: E8 embedding
- `srt_scale_phi()`: Golden scaling
- `srt_golden_gaussian_weights()`: Mode weights

### Backpropagation
- `py_backward_add()`, `py_backward_mul()`, `py_backward_softmax()`
- `py_backward_layernorm()`, `py_backward_phi_residual()`

### Broadcasting
- `py_consciousness_inflationary_broadcast()`
- `py_golden_inflationary_broadcast()`

### Data Loading
- `SRTBinaryLoader`, `SRTCSVParser`, `SRTDataPipeline`
- `StreamingCSVIterator`

### Spectral Operations
All functions in `spectral.rs` module exported to Python

### Causal History
- `create_causal_tracker()`
- `PyCausalHistoryTracker` - Python-facing tracker

### Memory Management
- `srt_memory_resonance()`, `srt_reserve_memory()`
- `srt_pool_stats()`, `srt_transfer_stats()`

---

## Performance Characteristics

### CPU Operations
- **Linear Algebra**: Full BLAS/LAPACK via ndarray_linalg
- **Parallelization**: rayon for multi-threaded operations
- **Cache Optimization**: Proper memory layout for CPU cache
- **Fixed-Point Math**: Avoid floating-point rounding

### GPU Operations (CUDA)
- **Kernel Compilation**: NVRTC (runtime compilation) via cudarc
- **Memory Pooling**: Reduce allocation overhead
- **Async Transfers**: Overlap GPU ↔ CPU transfers with computation
- **Multi-GPU**: Peer-copy and distributed operations
- **Caching**: PTX kernels cached per device

### Accuracy
- **Exact Arithmetic**: Rational, GoldenExact, SymExpr for theory
- **Fixed-Point**: Q32.32 for reproducible real-time systems
- **Floating-Point**: Only for final output or numerical stability when absolutely necessary

---

## Key Design Principles

1. **No External Dependencies**: Library is self-contained (except basic Rust std)
2. **Exact Math First**: All constants derive from φ, π, e, E*, q
3. **Hardware-Aware**: CPU vs. GPU operations optimized separately
4. **Zero Free Parameters**: Constants from SRT geometry, not learning
5. **Convergence Guaranteed**: DHSR dynamics ensure stable equilibrium
6. **Transparent to Python**: All Rust complexity hidden behind simple Python API

---

## Development Workflow

### Testing
```bash
# Run Rust tests
cargo test --release

# Run Python integration tests
pytest tests/
```

### Debugging
- Use `RUST_BACKTRACE=1` for panic backtraces
- CUDA kernel debugging with `cudb`
- Profiling with `perf` or NVIDIA Nsight

### Common Build Issues
- **PTX Compilation Fails**: Update NVIDIA driver
- **Maturin Not Found**: `pip install maturin`
- **CUDA Not Detected**: Set `CUDA_PATH` environment variable

---

## Conclusion

The Syntonic Rust backend is the computational engine implementing SRT mathematics with zero approximation and guaranteed correctness. It bridges the gap between pure mathematics and practical computation, enabling physically-grounded neural networks without free parameters or floating-point errors.
