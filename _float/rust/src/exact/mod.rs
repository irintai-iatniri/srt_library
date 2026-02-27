//! Exact Arithmetic Module for Syntony Recursion Theory
//!
//! This module provides exact (non-floating-point) arithmetic types
//! required for rigorous SRT computations. Standard IEEE-754 floats
//! cannot represent algebraic numbers like φ or preserve the exact
//! relationships between transcendental constants.
//!
//! # Type Hierarchy
//!
//! ```text
//! Rational (Q)
//!     ↓
//! GoldenExact (Q(√5) = Q(φ))
//!     ↓
//! SymExpr (symbolic expressions with π, e, E*, q)
//! ```
//!
//! # The Five Fundamental Constants
//!
//! SRT is built on five constants:
//! - π (pi) - circle constant, toroidal topology
//! - e (euler) - natural base, exponential evolution
//! - φ (phi) - golden ratio, recursion symmetry (algebraic: x² - x - 1 = 0)
//! - E* (e_star) - spectral Möbius constant = e^π - π
//! - q - universal syntony deficit, THE fundamental scale of SRT

pub mod constants;
pub mod golden;
pub mod rational;
pub mod symexpr;
pub mod fixed;
pub mod transcendental;
pub mod traits;
pub mod syntonic;
pub mod dual;
pub mod pythagorean;
pub mod rotator;
pub mod ternary_solver;

pub use constants::{CorrectionLevel, FundamentalConstant, Structure};
pub use golden::GoldenExact;
pub use rational::Rational;
pub use fixed::FixedPoint64;
pub use traits::ExactScalar;
pub use symexpr::{PySymExpr, SymExpr};
pub use syntonic::SyntonicExact;
pub use dual::SyntonicDual;
pub use pythagorean::{PythagoreanTriple, generate_resonance_ladder, generate_resonance_ladder_adaptive, find_closest_frequency};
pub use rotator::RationalRotator;
pub use ternary_solver::{TernarySolver, WaveLayer, BASIS_SCALE};
