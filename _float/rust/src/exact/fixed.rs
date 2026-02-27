use crate::exact::traits::ExactScalar;
use core::ops::{Add, Sub, Mul, Div, Neg};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint64(pub i64);

impl FixedPoint64 {
    pub const FRACTIONAL_BITS: u32 = 32;
    pub const SCALING_FACTOR: i64 = 1 << 32;
    
    pub const ZERO: FixedPoint64 = FixedPoint64(0);
    pub const ONE: FixedPoint64 = FixedPoint64(1 << 32);
    
    pub fn new(raw: i64) -> Self {
        FixedPoint64(raw)
    }

    pub fn sin(self) -> Self {
        crate::exact::transcendental::cordic_sin_cos(self).0
    }

    pub fn cos(self) -> Self {
        crate::exact::transcendental::cordic_sin_cos(self).1
    }

    pub fn sqrt(self) -> Self {
        crate::exact::transcendental::sqrt(self)
    }

    pub fn exp(self) -> Self {
        crate::exact::transcendental::exp(self)
    }

    pub fn acos(self) -> Self {
        crate::exact::transcendental::acos(self)
    }

    pub fn powi(self, exp: i64) -> Self {
        if exp == 0 { return Self::ONE; }
        if exp < 0 {
            return (Self::ONE / self).powi(-exp);
        }
        
        let mut base = self;
        let mut result = Self::ONE;
        let mut e = exp;
        
        while e > 0 {
            if e & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result
    }

    /// Create from integer value
    pub fn from_int(n: i64) -> Self {
        FixedPoint64(n << Self::FRACTIONAL_BITS)
    }

    /// Create from Rational (exact integer conversion)
    pub fn from_rational(r: crate::exact::rational::Rational) -> Self {
        // (num * 2^32) / denom
        // Use i128 to prevent overflow during multiplication
        let num = r.numerator() as i128;
        let denom = r.denominator() as i128;
        let scaled = num << Self::FRACTIONAL_BITS;
        FixedPoint64((scaled / denom) as i64)
    }

    /// Convert to f64 (lossy)
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALING_FACTOR as f64
    }

    /// Create from f64 (lossy)
    pub fn from_f64(v: f64) -> Self {
        FixedPoint64((v * Self::SCALING_FACTOR as f64) as i64)
    }
}

impl Add for FixedPoint64 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        FixedPoint64(self.0 + other.0)
    }
}

impl Sub for FixedPoint64 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        FixedPoint64(self.0 - other.0)
    }
}

impl Mul for FixedPoint64 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let product = (self.0 as i128 * other.0 as i128) >> FixedPoint64::FRACTIONAL_BITS;
        FixedPoint64(product as i64)
    }
}

impl Div for FixedPoint64 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let numerator = (self.0 as i128) << FixedPoint64::FRACTIONAL_BITS;
        let quotient = numerator / (other.0 as i128);
        FixedPoint64(quotient as i64)
    }
}

impl Neg for FixedPoint64 {
    type Output = Self;
    fn neg(self) -> Self {
        FixedPoint64(-self.0)
    }
}

impl ExactScalar for FixedPoint64 {
    fn zero() -> Self { Self::ZERO }
    fn one() -> Self { Self::ONE }
    
    fn from_f64(v: f64) -> Self {
        FixedPoint64((v * Self::SCALING_FACTOR as f64) as i64)
    }
    
    fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALING_FACTOR as f64
    }
}
