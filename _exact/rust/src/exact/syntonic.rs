use pyo3::prelude::*;
use crate::exact::golden::GoldenExact;
use crate::exact::rational::Rational;
use crate::exact::fixed::FixedPoint64;
use std::ops::{Add, Sub, Mul, Div};

/// SyntonicExact represents the complete Master Equation state:
/// Value = (a + bφ) + (c + dφ)E* + (e + fφ)q
///
/// This allows us to perform arithmetic with transcendentals without float drift.
#[pyclass]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyntonicExact {
    /// The primary golden component (Source term)
    #[pyo3(get)]
    pub base: GoldenExact,
    /// The E* component (e^π - π) - Winding transition weight
    #[pyo3(get)]
    pub e_star_coeff: GoldenExact,
    /// The q component (Syntony Deficit) - Ground noise weight
    #[pyo3(get)]
    pub q_deficit_coeff: GoldenExact,
}

#[pymethods]
impl SyntonicExact {
    #[new]
    pub fn new(base: GoldenExact) -> Self {
        Self {
            base,
            e_star_coeff: GoldenExact::zero(), // Default to zero e*
            q_deficit_coeff: GoldenExact::zero(), // Default to zero deficit
        }
    }

    /// Evaluates to float for the Shadow Domain (Visualization/Output only)
    pub fn to_f64(&self) -> f64 {
        const E_STAR_F64: f64 = 19.99909997918947; 
        const Q_DEFICIT_F64: f64 = 0.027395146920;

        self.base.to_f64() + 
        (self.e_star_coeff.to_f64() * E_STAR_F64) + 
        (self.q_deficit_coeff.to_f64() * Q_DEFICIT_F64)
    }

    /// Alias for to_f64, specifically for Shadow Domain visualization
    pub fn eval(&self) -> f64 {
        self.to_f64()
    }

    /// Evaluates to FixedPoint64 for deterministic computation (No floats)
    pub fn to_fixed_point(&self) -> FixedPoint64 {
        // Constants in Q32.32 format derived from exact values
        // PHI = 1.6180339887... * 2^32 = 6949005072
        const PHI_FIXED: i64 = 6949005072;
        // E* = 19.999099979... * 2^32 = 85903930402
        const E_STAR_FIXED: i64 = 85903930402;
        // q = 0.027395147... * 2^32 = 117671542
        const Q_DEFICIT_FIXED: i64 = 117671542;

        let base_fixed = {
            // GoldenExact a + b*phi
            // We need to convert Rationals a,b to FixedPoint64
            // Rational to FixedPoint: (num * 2^32) / denom
            let a_fp = FixedPoint64::from_rational(self.base.rational_part());
            let b_fp = FixedPoint64::from_rational(self.base.phi_part());
            a_fp + b_fp * FixedPoint64(PHI_FIXED)
        };
        
        let e_star_term = {
             // Approximation: just use the 'a' component of the coefficient for speed?
             // No, we should do full expansion: (a + b*phi) * E*
             // But E* is a scalar.
             let coeff_fp = self.e_star_coeff.to_fixed_point_internal(PHI_FIXED);
             coeff_fp * FixedPoint64(E_STAR_FIXED)
        };
        
        let q_term = {
             let coeff_fp = self.q_deficit_coeff.to_fixed_point_internal(PHI_FIXED);
             coeff_fp * FixedPoint64(Q_DEFICIT_FIXED)
        };

        base_fixed + e_star_term + q_term
    }
}

// Implement Arithmetic for the Super-Field
impl Add for SyntonicExact {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            base: self.base + other.base,
            e_star_coeff: self.e_star_coeff + other.e_star_coeff,
            q_deficit_coeff: self.q_deficit_coeff + other.q_deficit_coeff,
        }
    }
}

impl Sub for SyntonicExact {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            base: self.base - other.base,
            e_star_coeff: self.e_star_coeff - other.e_star_coeff,
            q_deficit_coeff: self.q_deficit_coeff - other.q_deficit_coeff,
        }
    }
}

impl std::ops::Neg for SyntonicExact {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            base: -self.base,
            e_star_coeff: -self.e_star_coeff,
            q_deficit_coeff: -self.q_deficit_coeff,
        }
    }
}

impl std::ops::Div for SyntonicExact {
    type Output = Self;
    fn div(self, _other: Self) -> Self {
        // Division in the super-field is complex (requires inverse of A+Bφ+...)
        // TODO: Implement proper inverse or geometric series approximation
        todo!("SyntonicExact division not yet implemented")
    }
}

impl Mul<Rational> for SyntonicExact {
    type Output = Self;
    fn mul(self, scalar: Rational) -> Self {
        Self {
            base: self.base * scalar,
            e_star_coeff: self.e_star_coeff * scalar,
            q_deficit_coeff: self.q_deficit_coeff * scalar,
        }
    }
}

impl Div<Rational> for SyntonicExact {
    type Output = Self;
    fn div(self, scalar: Rational) -> Self {
        Self {
            base: self.base / scalar,
            e_star_coeff: self.e_star_coeff / scalar,
            q_deficit_coeff: self.q_deficit_coeff / scalar,
        }
    }
}

impl Mul<SyntonicExact> for SyntonicExact {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // Syntonic multiplication: treat as field element modulo the transcendentals
        // For now, we multiply the base GoldenExact components
        // (Full cross-product would be: (a + bE* + cq)(d + eE* + fq) = complex)
        // Simplified version: just use base components for typical scalar multiplication
        Self {
            base: self.base * other.base,
            e_star_coeff: self.e_star_coeff * other.base + self.base * other.e_star_coeff,
            q_deficit_coeff: self.q_deficit_coeff * other.base + self.base * other.q_deficit_coeff,
        }
    }
}