# SRT Library — The Phase-State Architectural Paradigm

**The computational substrate for Syntony Recursion Theory (SRT)**

The SRT Library is a high-performance framework designed to derive Standard Model physics from four mathematical seeds — **φ**, **π**, **e**, and **1** — with zero free parameters. It implements the **Phase-State paradigm**, where computation is viewed as the evolution of resonant states within a syntonic manifold.

---

## 🏗️ Architecture

The library is organized into four primary layers, bridging theoretical physics with high-performance GPU-accelerated computation.

### 1. `core/` — Unified Python Intelligence
The high-level entry point for the library. It provides a familiar Pythonic interface for complex operations.
- **State Management**: Orchestrates `state.py` (the fundamental information object).
- **Neural Networks (`nn/`)**: Implementation of Gnostic architectures, resonant layers, and Golden GELU activations.
- **Physics Engine**: Modules for bosons, fermions, and the Standard Model hierarchy.

### 2. `_exact/` — The Precision Engine (Rust/CUDA)
Modern AI demands more than floating-point approximations. `_exact` provides a fixed-point and rational arithmetic backend.
- **Rust Backend**: Massive PyO3-based extension layer providing ~460 high-performance bindings.
- **CUDA Kernels**: 40+ specialized `.cu` kernels for exact tensor operations.
- **Autograd**: A custom industrial-grade reverse-mode automatic differentiation system.

### 3. `_float/` — The Performance Path
A mirrored implementation optimized for standard floating-point hardware, ensuring compatibility and benchmarking parity with traditional deep learning.

### 4. `theory_unique_components/` — The SRT Core
The implementation of SRT-specific logic that defines the library's unique analytical power.
- **CRT Operators**: Cosmological Recursion Theory (D̂, Ĥ, Ŝ, R̂).
- **Geometry**: Torus winding, E₈ lattice projections, and Möebius spectral analysis.

---

## 🦀 The Rust Backend (`_exact/rust`)

The heart of the library's precision is written in performance-critical Rust.
- **Custom Tensor Storage**: Native memory management for multidimensional arrays with support for complex-valued and exact-type storage.
- **HIerarchical Correction**: The `hierarchy.rs` module manages the recursive correction factors essential for deriving physical constants.
- **Spectral Analysis**: High-performance implementations of heat kernels and theta series.

---

## ⚡ CUDA Kernels (`_exact/rust/kernels`)

The library leverages low-level NVIDIA hardware features for maximal throughput:
- **`dhsr.cu`**: Dynamic Harmonic Shift Resonance — a proprietary fused operator for state evolution.
- **`e8_projection.cu`**: Efficient mapping of high-dimensional states onto the E₈ lattice.
- **`attention.cu`**: Syntonic-aware attention mechanisms optimized for phase-state coherence.
- **PTX Workflow**: Kernels are dynamically JIT-compiled or loaded from pre-compiled PTX (supporting `sm_75` through `sm_90`).

---

## 🚀 Quick Start

Ensure you have a modern NVIDIA GPU and the CUDA Toolkit (12.x+) installed.

### Installation

```bash
cd _exact
# Install the library into your environment via maturin
CUDA_PATH=/usr/local/cuda maturin develop --release
```

### Basic Usage

```python
import srt_library.core as syn

# Initialize a state on the GPU
psi = syn.state([1.0, 0.0, 0.5, 0.1], device="cuda")

# Apply a Golden GELU activation
y = syn.nn.functional.golden_gelu(psi)

# Access theoretical constants
phi = syn.PHI  # 1.6180339887...
```

---

## 📜 Maintenance

To maintain a clean and up-to-date view of the rapidly evolving directory structure, use the automated tree generator:

```bash
python3 generate_tree.py  # Updates tree.md
```

Detailed file-by-file documentation can be found in [tree.md](file:///media/Andrew/Backup/Programs/srt_library/tree.md).
