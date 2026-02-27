//! Syntonic Neural Architecture (SNA)
//!
//! Biologically-inspired neural computation using integer-based phasor mathematics.
//! 
//! # Core Components
//!
//! - **DiscreteHilbertKernel**: Converts real scalar signals into complex analytic signals
//!   (W4 projection) using integer-only Hilbert transform approximation.
//!
//! - **ResonantOscillator**: The fundamental "neuron" unit that performs phase-coherent
//!   resonance computation. Inputs are weighted by complex impedances, summed as phasors,
//!   and converted to ternary action potentials {-1, 0, +1}.
//!
//! # Architecture Philosophy
//!
//! Unlike standard neural networks that use floating-point dot products, SNA neurons
//! compute **vector resonance** in complex integer space. Aligned phasors reinforce
//! (constructive interference), misaligned phasors cancel (destructive interference).
//!
//! The output is not a continuous activation but a **ternary spike**:
//! - +1: Excitatory (forward phase, positive real component)
//! - -1: Inhibitory (reverse phase, negative real component)  
//! - 0: Syntonic Null (below threshold or exact zero-crossing)
//!
//! # Mathematical Foundation
//!
//! All arithmetic uses fixed-point integers scaled by `KERNEL_SCALE = 1024 = 2^10`.
//! This provides ~10 bits of fractional precision while maintaining exact computation.
//!
//! Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
//! Energy: |z|² = Re²(z) + Im²(z)
//! Phase: sign(Re(z)) determines spike polarity

pub mod resonant_oscillator;
pub mod network;

pub use resonant_oscillator::{DiscreteHilbertKernel, ResonantOscillator, KERNEL_SCALE};
pub use network::SyntonicNetwork;