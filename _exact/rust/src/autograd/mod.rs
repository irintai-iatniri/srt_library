//! SRT Autograd Module
//!
//! Provides gradient-based training infrastructure using CUDA autograd kernels.
//! This module enables standard backpropagation alongside the RES (Resonant Evolution Strategy)
//! training approach.

mod backward;

pub use backward::*;
