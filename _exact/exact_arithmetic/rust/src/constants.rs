//! Canonical SRT Constants — Single Source of Truth
//!
//! ALL constants in the crate must come from here using exact types.
//! Every value is derived from exact mathematics, not floating point.
//!
//! # Derivation Chain
//!
//! φ = (1 + √5)/2                        (algebraic: x² - x - 1 = 0)
//! 1/φ = φ - 1                            (algebraic identity)
//! φ² = φ + 1                             (algebraic identity)
//! E* = e^π - π                           (spectral Möbius constant)
//! q = (2φ + e/(2φ²)) / (φ⁴ · E*)        (universal syntony deficit)
//!
//! No magic numbers. No hand-typed decimals. Everything exact.

use crate::exact::golden::GoldenExact;
use crate::exact::symexpr::SymExpr;
use crate::exact::rational::Rational;

// =============================================================================
// φ (Golden Ratio) — Algebraic, exact in Q(√5)
// =============================================================================

/// φ = 0 + 1·φ — the recursion eigenvalue
pub fn srt_phi() -> GoldenExact {
    GoldenExact::phi()
}

/// 1/φ = φ - 1 — inverse golden ratio (coherence parameter)
pub fn srt_phi_inv() -> GoldenExact {
    GoldenExact::phi_hat()
}

/// φ² = 1 + φ — golden ratio squared
pub fn srt_phi_squared() -> GoldenExact {
    GoldenExact::phi_squared()
}

/// 1/φ² = 2 - φ — inverse golden ratio squared  
/// Derived: 1/φ² = (φ-1)² = φ² - 2φ + 1 = (φ+1) - 2φ + 1 = 2 - φ
pub fn srt_phi_inv_squared() -> GoldenExact {
    GoldenExact::from_ints(2, -1)
}

/// φ³ = 1 + 2φ — third power using Fibonacci coefficients
pub fn srt_phi_cubed() -> GoldenExact {
    GoldenExact::phi_power(3)
}

/// φ⁴ = 2 + 3φ — fourth power using Fibonacci coefficients
pub fn srt_phi_fourth() -> GoldenExact {
    GoldenExact::phi_power(4)
}

/// φ⁵ = 3 + 5φ — fifth power using Fibonacci coefficients
pub fn srt_phi_fifth() -> GoldenExact {
    GoldenExact::phi_power(5)
}

/// φ⁶ = 5 + 8φ — sixth power using Fibonacci coefficients
pub fn srt_phi_sixth() -> GoldenExact {
    GoldenExact::phi_power(6)
}

/// φ⁷ = 8 + 13φ — seventh power using Fibonacci coefficients
pub fn srt_phi_seventh() -> GoldenExact {
    GoldenExact::phi_power(7)
}

/// φ⁸ = 13 + 21φ — eighth power using Fibonacci coefficients
pub fn srt_phi_eighth() -> GoldenExact {
    GoldenExact::phi_power(8)
}

/// √5 = -1 + 2φ — exact representation in Q(φ)
pub fn srt_sqrt5() -> GoldenExact {
    GoldenExact::sqrt5()
}

// =============================================================================
// Transcendental Constants — Exact symbolic representations
// =============================================================================

/// π — Archimedes' constant (exact symbolic atom)
pub fn srt_pi() -> SymExpr {
    SymExpr::pi()
}

/// 2π — full circle
pub fn srt_two_pi() -> SymExpr {
    SymExpr::from_int(2).mul(SymExpr::pi())
}

/// e — Euler's number (exact symbolic atom)
pub fn srt_e() -> SymExpr {
    SymExpr::e()
}

/// E* = e^π - π — Spectral Möbius constant (exact symbolic)
pub fn srt_e_star() -> SymExpr {
    SymExpr::e_star()
}

/// q = (2φ + e/(2φ²)) / (φ⁴ · E*) — Universal syntony deficit (exact symbolic)
/// The single most important number in SRT.
pub fn srt_q_deficit() -> SymExpr {
    SymExpr::q()
}

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

/// 1/√2 — projection normalization (exact: will be symbolic when sqrt is implemented)
/// For now, return as symbolic expression that can be evaluated
pub fn srt_inv_sqrt2() -> SymExpr {
    // 1/√2 = √2/2 - this will be exact when we implement symbolic sqrt
    // For now, create as symbolic expression
    SymExpr::from_int(1).div(SymExpr::from_int(2).pow(Rational::new(1, 2)))
}

// =============================================================================
// Derived Correction Functions
// =============================================================================

/// Correction factor: (1 + sign × q/N) - exact symbolic
pub fn srt_q_correction(divisor: i128, positive_sign: bool) -> SymExpr {
    let one = SymExpr::from_int(1);
    let q = SymExpr::q();
    let n = SymExpr::from_int(divisor);
    let q_over_n = q.div(n);
    
    if positive_sign {
        one.add(q_over_n)
    } else {
        one.sub(q_over_n)
    }
}

/// Positive correction: (1 + q/N) - exact symbolic
pub fn srt_q_plus(divisor: i128) -> SymExpr {
    srt_q_correction(divisor, true)
}

/// Negative correction: (1 - q/N) - exact symbolic
pub fn srt_q_minus(divisor: i128) -> SymExpr {
    srt_q_correction(divisor, false)
}

// =============================================================================
// Verification
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_identity() {
        // φ² = φ + 1 (exact test)
        let phi = srt_phi();
        let phi_squared = phi * phi;
        let phi_plus_one = phi + GoldenExact::one();
        assert_eq!(phi_squared, phi_plus_one);
    }

    #[test]
    fn test_phi_inverse() {
        // 1/φ = φ - 1 (exact test)
        let phi = srt_phi();
        let phi_inv = srt_phi_inv();
        let expected = phi - GoldenExact::one();
        assert_eq!(phi_inv, expected);
    }

    #[test]
    fn test_phi_powers_fibonacci() {
        // φⁿ = F(n)φ + F(n-1) where F is Fibonacci (exact tests)
        // φ³ = 1 + 2φ (F(3)=2, F(2)=1)
        assert_eq!(srt_phi_cubed(), GoldenExact::from_ints(1, 2));
        // φ⁴ = 2 + 3φ (F(4)=3, F(3)=2)
        assert_eq!(srt_phi_fourth(), GoldenExact::from_ints(2, 3));
        // φ⁵ = 3 + 5φ (F(5)=5, F(4)=3)
        assert_eq!(srt_phi_fifth(), GoldenExact::from_ints(3, 5));
    }

    #[test]
    fn test_sqrt5() {
        // √5 = -1 + 2φ (exact test)
        let sqrt5 = srt_sqrt5();
        let expected = GoldenExact::from_ints(-1, 2);
        assert_eq!(sqrt5, expected);
    }

    #[test]
    fn test_e_star_expansion() {
        // E* = e^π - π (symbolic identity test)
        let e_star = srt_e_star();
        let e_star_expanded = SymExpr::e().exp_of().sub(SymExpr::pi());
        // Test that they evaluate to the same value
        let diff = (e_star.eval_f64() - e_star_expanded.eval_f64()).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_q_expanded_identity() {
        // q = (2φ + e/(2φ²)) / (φ⁴ · E*) (symbolic identity test)
        let q = srt_q_deficit();
        let q_expanded = SymExpr::q_expanded();
        // Test that they evaluate to the same value
        let diff = (q.eval_f64() - q_expanded.eval_f64()).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_e_star_near_20() {
        // E* ≈ 19.999... (famously close to 20)
        let e_star_val = srt_e_star().eval_f64();
        assert!(e_star_val > 19.99);
        assert!(e_star_val < 20.01);
    }

    #[test]
    fn test_q_about_2_7_percent() {
        // q ≈ 2.74%
        let q_val = srt_q_deficit().eval_f64();
        assert!(q_val > 0.027);
        assert!(q_val < 0.028);
    }

    #[test]
    fn test_correction_factors() {
        // Test that correction factors maintain symbolic structure
        let q_plus_8 = srt_q_plus(8);
        let q_minus_240 = srt_q_minus(240);
        
        // Should be symbolic expressions that can be evaluated
        assert!(q_plus_8.eval_f64() > 1.0); // 1 + q/8 > 1
        assert!(q_minus_240.eval_f64() < 1.0); // 1 - q/240 < 1
    }
}
