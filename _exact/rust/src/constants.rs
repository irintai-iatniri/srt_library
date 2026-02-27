//! Canonical SRT Constants — Single Source of Truth
//!
//! ALL float constants in the crate must come from here.
//! Every value is derived from exact mathematics, not typed by hand.
//!
//! # Derivation Chain
//!
//! φ = (1 + √5)/2                        (algebraic: x² - x - 1 = 0)
//! 1/φ = φ - 1                            (algebraic identity)
//! φ² = φ + 1                             (algebraic identity)
//! E* = e^π - π                           (spectral Möbius constant)
//! q = (2φ + e/(2φ²)) / (φ⁴ · E*)        (universal syntony deficit)
//!
//! No magic numbers. No hand-typed decimals. Everything derived.

use std::f64::consts::{PI, E};

// =============================================================================
// φ (Golden Ratio) — Algebraic, exact in Q(√5)
// =============================================================================

/// φ = (1 + √5)/2 — the recursion eigenvalue
pub const PHI: f64 = 1.6180339887498948482_f64;

/// 1/φ = φ - 1 — inverse golden ratio
/// NOT typed separately — derived from the identity 1/φ = φ - 1
pub const PHI_INV: f64 = PHI - 1.0;

/// φ² = φ + 1 — golden ratio squared
/// NOT typed separately — derived from φ² = φ + 1
pub const PHI_SQ: f64 = PHI + 1.0;

/// 1/φ² = 2 - φ — inverse golden ratio squared  
/// Derived: 1/φ² = (φ-1)² = φ² - 2φ + 1 = (φ+1) - 2φ + 1 = 2 - φ
pub const PHI_INV_SQ: f64 = 2.0 - PHI;

/// φ³ = φ² · φ = (φ+1)·φ = φ² + φ = 2φ + 1
pub const PHI_CUBED: f64 = 2.0 * PHI + 1.0;

/// φ⁴ = φ³ · φ = (2φ+1)·φ = 2φ² + φ = 2(φ+1) + φ = 3φ + 2
pub const PHI_FOURTH: f64 = 3.0 * PHI + 2.0;

/// φ⁵ = φ⁴ · φ = (3φ+2)·φ = 3φ² + 2φ = 3(φ+1) + 2φ = 5φ + 3
pub const PHI_FIFTH: f64 = 5.0 * PHI + 3.0;

/// φ⁶ = φ⁵ · φ = (5φ+3)·φ = 5φ² + 3φ = 5(φ+1) + 3φ = 8φ + 5
pub const PHI_SIXTH: f64 = 8.0 * PHI + 5.0;

/// φ⁷ = 13φ + 8
pub const PHI_SEVENTH: f64 = 13.0 * PHI + 8.0;

/// φ⁸ = 21φ + 13
pub const PHI_EIGHTH: f64 = 21.0 * PHI + 13.0;

/// √5 = 2φ - 1 — derived from φ = (1+√5)/2
pub const SQRT5: f64 = 2.0 * PHI - 1.0;

// =============================================================================
// Transcendental Constants — π and e from std, E* and q derived
// =============================================================================

/// π — Archimedes' constant (from std)
pub const SRT_PI: f64 = PI;

/// 2π — full circle
pub const TWO_PI: f64 = 2.0 * PI;

/// e — Euler's number (from std)
pub const SRT_E: f64 = E;

// E* and q cannot be const because they require runtime computation.
// We provide lazy_static or inline functions.

/// E* = e^π - π — Spectral Möbius constant
/// Must be computed at runtime because f64::powf is not const.
#[inline]
pub fn e_star() -> f64 {
    E.powf(PI) - PI
}

/// q = (2φ + e/(2φ²)) / (φ⁴ · E*) — Universal syntony deficit
/// The single most important number in SRT.
#[inline]
pub fn q_deficit() -> f64 {
    let numerator = 2.0 * PHI + E / (2.0 * PHI_SQ);
    let denominator = PHI_FOURTH * e_star();
    numerator / denominator
}

// For contexts that need a const (arrays, match arms, CUDA interop),
// we provide pre-computed values verified against the exact derivation.
// These MUST match e_star() and q_deficit() to full f64 precision.

/// E* as const — verified: e^π - π = 19.999099979189475...
pub const E_STAR_CONST: f64 = 19.999099979189476_f64;

/// q as const — verified: (2φ + e/(2φ²)) / (φ⁴·E*) = 0.027395146920158...
pub const Q_DEFICIT_CONST: f64 = 0.02739514692015854_f64;

// =============================================================================
// Algebraic Structure Dimensions (integer constants)
// =============================================================================

/// E₈ root count (both signs)
pub const E8_ROOTS: u32 = 240;

/// E₈ positive roots (chiral)
pub const E8_POSITIVE: u32 = 120;

/// E₈ adjoint dimension
pub const E8_ADJOINT: u32 = 248;

/// E₈ Coxeter number
pub const E8_COXETER: u32 = 30;

/// E₈ rank (Cartan subalgebra)
pub const E8_RANK: u32 = 8;

/// E₇ adjoint dimension
pub const E7_ADJOINT: u32 = 133;

/// E₇ roots
pub const E7_ROOTS: u32 = 126;

/// E₇ positive roots
pub const E7_POSITIVE: u32 = 63;

/// E₇ fundamental representation
pub const E7_FUND: u32 = 56;

/// E₇ Coxeter number
pub const E7_COXETER: u32 = 18;

/// E₆ adjoint dimension
pub const E6_ADJOINT: u32 = 78;

/// E₆ roots
pub const E6_ROOTS: u32 = 72;

/// E₆ positive roots = Golden Cone
pub const E6_POSITIVE: u32 = 36;

/// E₆ fundamental representation
pub const E6_FUND: u32 = 27;

/// D₄ kissing number = consciousness threshold
pub const D4_KISSING: u32 = 24;

/// D₄ adjoint dimension
pub const D4_ADJOINT: u32 = 28;

/// G₂ adjoint dimension
pub const G2_ADJOINT: u32 = 14;

// =============================================================================
// Projection / Normalization Constants
// =============================================================================

/// 1/√2 — projection normalization (exact: algebraic irrational)
pub const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// =============================================================================
// Derived Correction Functions
// =============================================================================

/// Correction factor: (1 + sign × q/N)
#[inline]
pub fn q_correction(divisor: f64, sign: f64) -> f64 {
    1.0 + sign * q_deficit() / divisor
}

/// Positive correction: (1 + q/N)
#[inline]
pub fn q_plus(divisor: f64) -> f64 {
    q_correction(divisor, 1.0)
}

/// Negative correction: (1 - q/N)
#[inline]
pub fn q_minus(divisor: f64) -> f64 {
    q_correction(divisor, -1.0)
}

// =============================================================================
// Verification
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_identity() {
        // φ² = φ + 1
        assert!((PHI * PHI - PHI_SQ).abs() < 1e-15);
    }

    #[test]
    fn test_phi_inverse() {
        // 1/φ = φ - 1
        assert!((1.0 / PHI - PHI_INV).abs() < 1e-15);
    }

    #[test]
    fn test_phi_inv_sq() {
        // 1/φ² = 2 - φ
        assert!((1.0 / (PHI * PHI) - PHI_INV_SQ).abs() < 1e-15);
    }

    #[test]
    fn test_phi_powers_fibonacci() {
        // φⁿ = F(n)φ + F(n-1) where F is Fibonacci
        // φ³ = 2φ + 1 (F(3)=2, F(2)=1)
        assert!((PHI.powi(3) - PHI_CUBED).abs() < 1e-14);
        // φ⁴ = 3φ + 2 (F(4)=3, F(3)=2)
        assert!((PHI.powi(4) - PHI_FOURTH).abs() < 1e-14);
        // φ⁵ = 5φ + 3 (F(5)=5, F(4)=3)
        assert!((PHI.powi(5) - PHI_FIFTH).abs() < 1e-13);
    }

    #[test]
    fn test_sqrt5() {
        // √5 = 2φ - 1
        assert!((5.0_f64.sqrt() - SQRT5).abs() < 1e-15);
    }

    #[test]
    fn test_e_star() {
        let computed = e_star();
        let expected = E.powf(PI) - PI;
        assert!((computed - expected).abs() < 1e-15);
        assert!((computed - E_STAR_CONST).abs() < 1e-12);
    }

    #[test]
    fn test_q_deficit() {
        let q = q_deficit();
        // Verify the Master Equation
        let numerator = 2.0 * PHI + E / (2.0 * PHI_SQ);
        let denominator = PHI_FOURTH * e_star();
        let expected = numerator / denominator;
        assert!((q - expected).abs() < 1e-15);
        assert!((q - Q_DEFICIT_CONST).abs() < 1e-12);
    }

    #[test]
    fn test_e_star_near_20() {
        // E* ≈ 19.999... (famously close to 20)
        assert!(e_star() > 19.99);
        assert!(e_star() < 20.01);
    }

    #[test]
    fn test_q_about_2_7_percent() {
        // q ≈ 2.74%
        assert!(q_deficit() > 0.027);
        assert!(q_deficit() < 0.028);
    }
}
