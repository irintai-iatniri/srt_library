//! Resonant Engine Module
//!
//! Hardware-native SRT/CRT architecture where:
//! - GPU = D̂ (Differentiation) - chaotic flux generator
//! - CPU = Ĥ (Harmonization) - exact lattice enforcer via crystallization
//! - PCIe bus = Phase boundary with φ-dwell timing enforcement
//!
//! # Core Components
//!
//! - [`ResonantTensor`]: Dual-state tensor with exact lattice (CPU) and ephemeral flux (GPU)
//! - [`ResonantPhase`]: Phase enumeration (Crystallized, Flux, Transitioning)
//! - [`ResonanceEnforcer`]: φ-dwell timing controller
//! - [`ResonantEvolver`]: Resonant Evolution Strategy (RES) for discrete learning
//!
//! # Phase Transitions
//!
//! ```text
//! ┌─────────────────┐     wake_flux()      ┌─────────────────┐
//! │   CRYSTALLIZED  │ ─────────────────────► │      FLUX       │
//! │  (CPU/Exact)    │                       │   (GPU/Float)   │
//! │  GoldenExact    │                       │   CudaSlice     │
//! └────────┬────────┘                       └────────┬────────┘
//!          │                                         │
//!          │        crystallize()                    │
//!          │◄────────────────────────────────────────┤
//!          │        (snap f64→GoldenExact)           │
//!          │                                         │
//!          │        destroy_shadow()                 │
//!          │        (flux = None)                    │
//!          │                            ─────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use syntonic::resonant::{ResonantTensor, ResonantEngine};
//!
//! // Create tensor from floats
//! let values = vec![1.0, 2.0, 3.0, 4.0];
//! let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
//! let mut tensor = ResonantTensor::from_floats(&values, vec![4], mode_norms, 100);
//!
//! // D-phase: wake flux (transfer to GPU)
//! tensor.wake_flux(device, 0.01)?;
//!
//! // H-phase: crystallize (snap back to exact lattice)
//! let syntony = tensor.crystallize(1000)?;
//! ```

mod attractor;
mod crystallize;
pub mod e8_lattice;
pub mod e8_lattice_nn;
mod evolver;
pub mod golden_norm;
pub mod loss;
pub mod number_theory;
pub mod phi_ops;
pub mod py_wrappers;
mod retrocausal;
pub mod syntonic_softmax;
pub mod syntony;
mod tensor;

pub use evolver::{RESConfig, RESResult, ResonantEvolver};
pub use golden_norm::GoldenNormMode;
pub use phi_ops::{phi_residual, phi_residual_relu, PhiResidualMode};
pub use syntonic_softmax::{syntonic_softmax_py, SyntonicSoftmaxMode, SyntonicSoftmaxState};
pub use tensor::{ResonantError, ResonantPhase, ResonantTensor};

// E8 Lattice and Golden Projector PyO3 wrappers
pub use e8_lattice::{
    py_compute_8d_weight, py_e8_generate_roots, py_e8_generate_weights, py_golden_project_parallel,
    py_golden_project_perp, py_golden_projector_phi, py_golden_projector_q, py_is_in_golden_cone,
};

// Neural Network E8 Lattice and Golden Projector PyO3 wrappers
pub use e8_lattice_nn::{
    py_compute_8d_weight_nn, py_e8_generate_roots_nn, py_e8_generate_weights_nn,
    py_golden_project_parallel_nn, py_golden_project_perp_nn, py_golden_projector_phi_nn,
    py_golden_projector_q_nn, py_is_in_golden_cone_nn,
};

// Re-export key constants
pub const PHI: f64 = 1.6180339887498948482;
pub const PHI_INV: f64 = 0.6180339887498948482;
pub const PHI_INV_SQ: f64 = 0.3819660112501051518;
