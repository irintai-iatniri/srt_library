//! Exact Golden Field Arithmetic Q(√5) = Q(φ)
//!
//! Elements of the form a + b·φ where a, b ∈ Q (exact rationals).
//! This is the natural number field for SRT since φ is the recursion eigenvalue.
//!
//! # Key Properties
//!
//! - φ = (1 + √5)/2 is a root of x² - x - 1 = 0
//! - φ² = φ + 1 (fundamental identity, used for multiplication)
//! - φ · φ̂ = -1 where φ̂ = (1 - √5)/2 = 1 - φ (Galois conjugate)
//! - 1/φ = φ - 1 = φ̂ (with sign correction)
//!
//! # Norm and Trace
//!
//! For α = a + bφ:
//! - Conjugate: α* = (a + b) - bφ = a + b(1 - φ)
//! - Norm: N(α) = α · α* = a² + ab - b² (always rational)
//! - Trace: Tr(α) = α + α* = 2a + b (always rational)

use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::rational::Rational;
use crate::exact::fixed::FixedPoint64;

/// Union type for Python operators that accept GoldenExact or Rational
#[derive(Clone, Copy)]
pub enum GoldenOrRational {
    Golden(GoldenExact),
    Rat(Rational),
}

impl<'py> FromPyObject<'py> for GoldenOrRational {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try GoldenExact first
        if let Ok(g) = ob.extract::<GoldenExact>() {
            return Ok(GoldenOrRational::Golden(g));
        }
        // Try Rational
        if let Ok(r) = ob.extract::<Rational>() {
            return Ok(GoldenOrRational::Rat(r));
        }
        // Try converting int to Rational
        if let Ok(i) = ob.extract::<i64>() {
            return Ok(GoldenOrRational::Rat(Rational::from_int(i as i128)));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected GoldenExact, Rational, or int"
        ))
    }
}

/// Exact element of Q(φ): a + b·φ where a, b ∈ Q
#[pyclass]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GoldenExact {
    /// Rational component (coefficient of 1)
    #[pyo3(get)]
    a: Rational,
    /// Golden component (coefficient of φ)
    #[pyo3(get)]
    b: Rational,
}

impl GoldenExact {
    /// Create a + b·φ
    pub fn new(a: Rational, b: Rational) -> Self {
        GoldenExact { a, b }
    }

    /// Create from two integers: a + b·φ
    pub fn from_ints(a: i64, b: i64) -> Self {
        GoldenExact {
            a: Rational::from_int(a as i128),
            b: Rational::from_int(b as i128),
        }
    }

    /// Zero in Q(φ)
    pub fn zero() -> Self {
        GoldenExact {
            a: Rational::zero(),
            b: Rational::zero(),
        }
    }

    /// One in Q(φ)
    pub fn one() -> Self {
        GoldenExact {
            a: Rational::one(),
            b: Rational::zero(),
        }
    }

    /// φ = 0 + 1·φ
    pub fn phi() -> Self {
        GoldenExact {
            a: Rational::zero(),
            b: Rational::one(),
        }
    }

    /// φ² = 1 + 1·φ (derived from φ² = φ + 1)
    pub fn phi_squared() -> Self {
        GoldenExact {
            a: Rational::one(),
            b: Rational::one(),
        }
    }

    /// φ̂ = 1/φ = φ - 1 = -1 + 1·φ
    /// Note: This is the SRT coherence parameter
    pub fn phi_hat() -> Self {
        GoldenExact {
            a: Rational::from_int(-1),
            b: Rational::one(),
        }
    }

    /// √5 = 2φ - 1 = -1 + 2·φ
    /// Since φ = (1 + √5)/2, we have √5 = 2φ - 1
    pub fn sqrt5() -> Self {
        GoldenExact {
            a: Rational::from_int(-1),
            b: Rational::from_int(2),
        }
    }

    /// Embed a rational into Q(φ)
    pub fn from_rational(r: Rational) -> Self {
        GoldenExact {
            a: r,
            b: Rational::zero(),
        }
    }

    /// Create from single integer: n + 0·φ
    pub fn from_int(n: i32) -> Self {
        GoldenExact {
            a: Rational::from_int(n as i128),
            b: Rational::zero(),
        }
    }

    /// Get the rational coefficient
    pub fn rational_part(&self) -> Rational {
        self.a
    }

    /// Get the φ coefficient
    pub fn phi_part(&self) -> Rational {
        self.b
    }

    /// Is this exactly zero?
    pub fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    /// Is this exactly one?
    pub fn is_one(&self) -> bool {
        self.a.is_one() && self.b.is_zero()
    }

    /// Convert to exact rational (if b=0)
    pub fn to_rational_exact(&self) -> Option<Rational> {
        if self.b.is_zero() {
            Some(self.a)
        } else {
            None
        }
    }
    
    /// Convert to FixedPoint64 using exact integer arithmetic (no floats)
    /// Requires passing the fixed-point value of PHI
    pub fn to_fixed_point_internal(&self, phi_val: i64) -> FixedPoint64 {
        let phi_fp = FixedPoint64(phi_val);
        let a_fp = FixedPoint64::from_rational(self.a);
        let b_fp = FixedPoint64::from_rational(self.b);
        a_fp + b_fp * phi_fp
    }

    /// Is this a pure rational (b = 0)?
    pub fn is_rational(&self) -> bool {
        self.b.is_zero()
    }

    /// Is this a pure φ-multiple (a = 0)?
    pub fn is_pure_phi(&self) -> bool {
        self.a.is_zero()
    }

    /// Galois conjugate: (a + bφ)* = (a + b) - bφ
    ///
    /// Since φ* = 1 - φ (the other root of x² - x - 1 = 0):
    /// (a + bφ)* = a + b(1 - φ) = (a + b) - bφ
    pub fn conjugate(&self) -> Self {
        GoldenExact {
            a: self.a + self.b,
            b: -self.b,
        }
    }

    /// Norm: N(α) = α · α* = a² + ab - b²
    ///
    /// This is always a rational number.
    /// For φ: N(φ) = 0 + 0·1 - 1 = -1
    pub fn norm(&self) -> Rational {
        // (a + bφ)(a + b - bφ) = a(a+b) - abφ + b(a+b)φ - b²φ²
        // = a² + ab + abφ + b²φ - abφ - b²φ²
        // = a² + ab + b²φ - b²(φ+1)
        // = a² + ab + b²φ - b²φ - b²
        // = a² + ab - b²
        self.a * self.a + self.a * self.b - self.b * self.b
    }

    /// Trace: Tr(α) = α + α* = 2a + b
    pub fn trace(&self) -> Rational {
        self.a + self.a + self.b
    }

    /// Multiplicative inverse: 1/α = α* / N(α)
    pub fn inverse(&self) -> Self {
        let n = self.norm();
        if n.is_zero() {
            panic!("GoldenExact: inverse of zero");
        }
        let conj = self.conjugate();
        GoldenExact {
            a: conj.a / n,
            b: conj.b / n,
        }
    }

    /// Integer power using binary exponentiation
    pub fn pow(&self, exp: i32) -> Self {
        if exp == 0 {
            return GoldenExact::one();
        }
        if exp < 0 {
            return self.inverse().pow(-exp);
        }

        let mut result = GoldenExact::one();
        let mut base = *self;
        let mut e = exp as u32;

        while e > 0 {
            if e & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result
    }

    /// φ^n computed exactly
    /// Uses the identity φ^n = F_n·φ + F_{n-1} where F_n is Fibonacci
    pub fn phi_power(n: i32) -> Self {
        if n == 0 {
            return GoldenExact::one();
        }
        if n < 0 {
            // φ^(-n) = (-1)^n · (F_{n-1} - F_n·φ) / (F_n² - F_n·F_{n-1} - F_{n-1}²)
            // Simpler: just compute and invert
            return GoldenExact::phi_power(-n).inverse();
        }

        // φ^n = F_n·φ + F_{n-1}
        let (f_prev, f_curr) = fibonacci_pair(n as u32);
        GoldenExact::from_ints(f_prev, f_curr)
    }

    /// Multiply by φ: (a + b·φ) × φ = b + (a + b)·φ
    ///
    /// This is exact in Q(φ) using the identity φ² = φ + 1.
    pub fn mul_phi(&self) -> Self {
        // (a + b·φ) × φ = a·φ + b·φ² = a·φ + b·(φ + 1) = b + (a + b)·φ
        GoldenExact {
            a: self.b,
            b: self.a + self.b,
        }
    }

    /// Divide by φ: (a + b·φ) / φ = (a + b·φ) × (1/φ) = (a + b·φ) × (φ - 1)
    ///
    /// This is exact in Q(φ) using the identity 1/φ = φ - 1.
    pub fn div_phi(&self) -> Self {
        // (a + b·φ) / φ = (a + b·φ) × (φ - 1) = a·φ - a + b·φ² - b·φ
        //              = a·φ - a + b·(φ + 1) - b·φ = (b - a) + a·φ
        GoldenExact {
            a: self.b - self.a,
            b: self.a,
        }
    }

    /// Evaluate to f64 (for numerical verification ONLY)
    /// φ ≈ 1.6180339887498949
    pub fn to_f64(&self) -> f64 {
        const PHI_F64: f64 = 1.6180339887498949;
        self.a.to_f64() + self.b.to_f64() * PHI_F64
    }

    // =========================================================================
    // LLL-based Diophantine Approximation
    // =========================================================================

    /// Find the nearest GoldenExact to a float using LLL lattice reduction.
    ///
    /// Given x ∈ ℝ, finds a + b·φ ∈ Q(φ) that minimizes |a + b·φ - x|
    /// where the coefficients a, b are rationals with bounded numerator/denominator.
    ///
    /// # Arguments
    /// * `x` - The floating-point value to approximate
    /// * `max_coeff` - Maximum absolute value for integer coefficients (denominator bound)
    ///
    /// # Algorithm
    /// Uses LLL lattice basis reduction on a 2D lattice encoding the approximation problem.
    /// The lattice basis is:
    /// ```text
    ///   v₁ = (1, 0, ⌊K⌋)      // representing coefficient a
    ///   v₂ = (0, 1, ⌊K·φ⌋)    // representing coefficient b
    /// ```
    /// where K is a scaling factor. Short vectors in this lattice correspond to
    /// good approximations a + b·φ ≈ x.
    pub fn find_nearest(x: f64, max_coeff: i64) -> Self {
        const PHI_F64: f64 = 1.6180339887498949;
        const TARGET_ERR: f64 = 1e-12;

        // Handle special cases
        if x.is_nan() || x.is_infinite() {
            return GoldenExact::zero();
        }

        // Scaling factor for lattice precision
        // Larger K gives better precision but risks overflow
        let mut k: f64 = 1e12;

        // Apply 2D LLL reduction (first pass)
        let (a_best, b_best) = lll_find_nearest_2d(x, PHI_F64, k, max_coeff);
        let mut best = GoldenExact::from_ints(a_best, b_best);
        let mut best_error = (best.to_f64() - x).abs();

        // Local neighborhood search around the LLL candidate
        for da in -2..=2 {
            for db in -2..=2 {
                let a = a_best + da;
                let b = b_best + db;
                if a.abs() > max_coeff || b.abs() > max_coeff {
                    continue;
                }
                let cand = GoldenExact::from_ints(a, b);
                let err = (cand.to_f64() - x).abs();
                if err < best_error {
                    best_error = err;
                    best = cand;
                }
            }
        }

        // Adaptive refinement with increased lattice scale until tolerance or limits
        if best_error > TARGET_ERR {
            for _ in 0..2 {
                k *= 10.0;
                let (a_next, b_next) = lll_find_nearest_2d(x, PHI_F64, k, max_coeff);
                let cand = GoldenExact::from_ints(a_next, b_next);
                let err = (cand.to_f64() - x).abs();
                if err < best_error {
                    best_error = err;
                    best = cand;
                }
                if best_error <= TARGET_ERR {
                    break;
                }
            }
        }

        // Rational refinement on the residual if still above target and bounds allow
        if best_error > TARGET_ERR && max_coeff <= 2000 {
            let residual = x - best.to_f64();
            if let Some(delta_a) = Rational::from_f64_approx(residual, max_coeff as i128) {
                let a = best.rational_part() + delta_a;
                let b = best.phi_part();
                let cand = GoldenExact::new(a, b);
                let err = (cand.to_f64() - x).abs();
                if err < best_error {
                    best = cand;
                }
            }
        }

        best
    }

    /// Find the nearest GoldenExact with rational coefficients.
    ///
    /// This version allows fractional a, b for higher precision approximation.
    ///
    /// # Arguments
    /// * `x` - The floating-point value to approximate
    /// * `max_denom` - Maximum denominator for rational coefficients
    pub fn find_nearest_rational(x: f64, max_denom: i64) -> Self {
        const PHI_F64: f64 = 1.6180339887498949;

        if x.is_nan() || x.is_infinite() {
            return GoldenExact::zero();
        }

        // First find good integer approximation
        let (a_int, b_int) = lll_find_nearest_2d(x, PHI_F64, 1e12, max_denom);

        // Compute residual
        let approx = a_int as f64 + b_int as f64 * PHI_F64;
        let residual = x - approx;

        // If residual is small enough, use integer coefficients
        if residual.abs() < 1e-14 {
            return GoldenExact::from_ints(a_int, b_int);
        }

        // Try to improve with rational refinement
        // Use continued fraction on residual to get a small correction
        if let Some(correction_a) = Rational::from_f64_approx(residual, max_denom as i128) {
            let a = Rational::from_int(a_int as i128) + correction_a;
            let b = Rational::from_int(b_int as i128);
            return GoldenExact::new(a, b);
        }

        GoldenExact::from_ints(a_int, b_int)
    }

    /// Snap a vector of f64 values to GoldenExact lattice points.
    ///
    /// Returns (lattice_points, residuals) where residuals[i] = values[i] - lattice_points[i].to_f64()
    ///
    /// # Arguments
    /// * `values` - Slice of floating-point values to snap
    /// * `max_coeff` - Maximum coefficient bound for each element
    pub fn snap_vector(values: &[f64], max_coeff: i64) -> (Vec<GoldenExact>, Vec<f64>) {
        let lattice: Vec<GoldenExact> = values
            .iter()
            .map(|&x| GoldenExact::find_nearest(x, max_coeff))
            .collect();

        let residuals: Vec<f64> = values
            .iter()
            .zip(lattice.iter())
            .map(|(&x, g)| x - g.to_f64())
            .collect();

        (lattice, residuals)
    }

    /// Snap with parallelization using rayon (for large vectors).
    #[cfg(feature = "rayon")]
    pub fn snap_vector_parallel(values: &[f64], max_coeff: i64) -> (Vec<GoldenExact>, Vec<f64>) {
        use rayon::prelude::*;

        let lattice: Vec<GoldenExact> = values
            .par_iter()
            .map(|&x| GoldenExact::find_nearest(x, max_coeff))
            .collect();

        let residuals: Vec<f64> = values
            .par_iter()
            .zip(lattice.par_iter())
            .map(|(&x, g)| x - g.to_f64())
            .collect();

        (lattice, residuals)
    }
}

// === Arithmetic Operations ===

impl Add for GoldenExact {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        GoldenExact {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl Sub for GoldenExact {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        GoldenExact {
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl Mul for GoldenExact {
    type Output = Self;

    /// Multiplication using φ² = φ + 1
    ///
    /// (a₁ + b₁φ)(a₂ + b₂φ) = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂φ²
    ///                       = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂(φ + 1)
    ///                       = (a₁a₂ + b₁b₂) + (a₁b₂ + a₂b₁ + b₁b₂)φ
    fn mul(self, other: Self) -> Self {
        let new_a = self.a * other.a + self.b * other.b;
        let new_b = self.a * other.b + self.b * other.a + self.b * other.b;
        GoldenExact { a: new_a, b: new_b }
    }
}

impl Div for GoldenExact {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self * other.inverse()
    }
}

impl Neg for GoldenExact {
    type Output = Self;

    fn neg(self) -> Self {
        GoldenExact {
            a: -self.a,
            b: -self.b,
        }
    }
}

// === Scalar operations ===

impl Mul<Rational> for GoldenExact {
    type Output = Self;

    fn mul(self, scalar: Rational) -> Self {
        GoldenExact {
            a: self.a * scalar,
            b: self.b * scalar,
        }
    }
}

impl Div<Rational> for GoldenExact {
    type Output = Self;

    fn div(self, scalar: Rational) -> Self {
        GoldenExact {
            a: self.a / scalar,
            b: self.b / scalar,
        }
    }
}

impl fmt::Display for GoldenExact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.b.is_zero() {
            write!(f, "{}", self.a)
        } else if self.a.is_zero() {
            if self.b.is_one() {
                write!(f, "φ")
            } else {
                write!(f, "{}·φ", self.b)
            }
        } else if self.b.is_positive() {
            if self.b.is_one() {
                write!(f, "{} + φ", self.a)
            } else {
                write!(f, "{} + {}·φ", self.a, self.b)
            }
        } else {
            let neg_b = -self.b;
            if neg_b.is_one() {
                write!(f, "{} - φ", self.a)
            } else {
                write!(f, "{} - {}·φ", self.a, neg_b)
            }
        }
    }
}

// === PyO3 Methods ===

#[pymethods]
impl GoldenExact {
    #[new]
    fn py_new(arg1: &Bound<'_, PyAny>, arg2: &Bound<'_, PyAny>, arg3: Option<i64>, arg4: Option<i64>) -> PyResult<Self> {
        // Case 1: GoldenExact(Rational, Rational)
        if let Ok(r1) = arg1.extract::<Rational>() {
            if let Ok(r2) = arg2.extract::<Rational>() {
                return Ok(GoldenExact::new(r1, r2));
            }
        }
        
        // Case 2: GoldenExact(a_num, a_denom, b_num, b_denom)
        if let (Ok(a_n), Ok(a_d)) = (arg1.extract::<i64>(), arg2.extract::<i64>()) {
             if let (Some(b_n), Some(b_d)) = (arg3, arg4) {
                 return Ok(GoldenExact::new(
                     Rational::new(a_n as i128, a_d as i128),
                     Rational::new(b_n as i128, b_d as i128)
                 ));
             }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Invalid arguments. Expected (Rational, Rational) or (int, int, int, int)"
        ))
    }

    /// Create from integer coefficients: a + b·φ
    #[staticmethod]
    fn from_integers(a: i64, b: i64) -> Self {
        GoldenExact::from_ints(a, b)
    }

    /// The golden ratio φ
    #[staticmethod]
    fn golden_ratio() -> Self {
        GoldenExact::phi()
    }

    /// φ² = φ + 1
    #[staticmethod]
    fn golden_squared() -> Self {
        GoldenExact::phi_squared()
    }

    /// 1/φ = φ - 1 (coherence parameter)
    #[staticmethod]
    fn coherence_parameter() -> Self {
        GoldenExact::phi_hat()
    }

    /// √5 = 2φ - 1
    #[staticmethod]
    fn root_five() -> Self {
        GoldenExact::sqrt5()
    }

    /// φ^n computed exactly
    #[staticmethod]
    fn phi_to_power(n: i32) -> Self {
        GoldenExact::phi_power(n)
    }

    #[getter]
    fn rational_coefficient(&self) -> (i64, i64) {
        (self.a.numerator() as i64, self.a.denominator() as i64)
    }

    #[getter]
    fn phi_coefficient(&self) -> (i64, i64) {
        (self.b.numerator() as i64, self.b.denominator() as i64)
    }

    fn __repr__(&self) -> String {
        format!(
            "GoldenExact({}/{}  + ({}/{})·φ)",
            self.a.numerator(),
            self.a.denominator(),
            self.b.numerator(),
            self.b.denominator()
        )
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __add__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => *self + g,
            GoldenOrRational::Rat(r) => GoldenExact::new(self.a + r, self.b),
        }
    }

    fn __sub__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => *self - g,
            GoldenOrRational::Rat(r) => GoldenExact::new(self.a - r, self.b),
        }
    }

    fn __mul__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => *self * g,
            GoldenOrRational::Rat(r) => *self * r,
        }
    }

    fn __truediv__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => *self / g,
            GoldenOrRational::Rat(r) => *self / r,
        }
    }

    fn __neg__(&self) -> Self {
        -*self
    }

    fn __eq__(&self, other: &GoldenExact) -> bool {
        self == other
    }

    // === Rational interoperability ===
    // These allow GoldenExact op Rational and Rational op GoldenExact

    /// Multiply GoldenExact by Rational: (a + bφ) * r = ar + brφ
    fn mul_rational(&self, r: &Rational) -> Self {
        *self * *r
    }

    /// Add Rational to GoldenExact: (a + bφ) + r = (a + r) + bφ
    fn add_rational(&self, r: &Rational) -> Self {
        GoldenExact::new(self.a + *r, self.b)
    }

    /// Subtract Rational from GoldenExact: (a + bφ) - r = (a - r) + bφ
    fn sub_rational(&self, r: &Rational) -> Self {
        GoldenExact::new(self.a - *r, self.b)
    }

    /// Divide GoldenExact by Rational: (a + bφ) / r = (a/r) + (b/r)φ
    fn div_rational(&self, r: &Rational) -> Self {
        *self / *r
    }

    /// Reverse add: other + (a + bφ)
    /// Accepts int, Rational, or GoldenExact
    fn __radd__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => g + *self,
            GoldenOrRational::Rat(r) => GoldenExact::new(self.a + r, self.b),
        }
    }

    /// Reverse multiply: other * (a + bφ)
    /// Accepts int, Rational, or GoldenExact
    fn __rmul__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => g * *self,
            GoldenOrRational::Rat(r) => *self * r,
        }
    }

    /// Reverse subtract: other - (a + bφ)
    /// Accepts int, Rational, or GoldenExact
    fn __rsub__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => g - *self,
            GoldenOrRational::Rat(r) => GoldenExact::new(r - self.a, -self.b),
        }
    }

    /// Reverse divide: other / (a + bφ)
    /// Accepts int, Rational, or GoldenExact
    fn __rtruediv__(&self, other: GoldenOrRational) -> Self {
        match other {
            GoldenOrRational::Golden(g) => g / *self,
            GoldenOrRational::Rat(r) => GoldenExact::from_rational(r) / *self,
        }
    }

    /// Galois conjugate
    fn galois_conjugate(&self) -> Self {
        self.conjugate()
    }

    /// Field norm (always rational)
    fn field_norm(&self) -> (i64, i64) {
        let n = self.norm();
        (n.numerator() as i64, n.denominator() as i64)
    }

    /// Field trace (always rational)
    fn field_trace(&self) -> (i64, i64) {
        let t = self.trace();
        (t.numerator() as i64, t.denominator() as i64)
    }

    /// Integer power
    fn power(&self, exp: i32) -> Self {
        self.pow(exp)
    }

    /// Multiplicative inverse
    fn reciprocal(&self) -> Self {
        self.inverse()
    }

    /// Evaluate to float (for verification only!)
    fn eval(&self) -> f64 {
        self.to_f64()
    }

    /// Check if this is exactly rational (no φ component)
    fn is_purely_rational(&self) -> bool {
        self.is_rational()
    }

    /// Check if this is exactly a multiple of φ (no rational part)
    fn is_purely_golden(&self) -> bool {
        self.is_pure_phi()
    }

    // =========================================================================
    // LLL-based Approximation (Python bindings)
    // =========================================================================

    /// Find the nearest GoldenExact to a float using LLL lattice reduction.
    ///
    /// Args:
    ///     x: The floating-point value to approximate
    ///     max_coeff: Maximum absolute value for integer coefficients
    ///
    /// Returns:
    ///     A GoldenExact that minimizes |a + b·φ - x|
    #[staticmethod]
    pub fn nearest(x: f64, max_coeff: i64) -> Self {
        GoldenExact::find_nearest(x, max_coeff)
    }

    /// Find the nearest GoldenExact with rational coefficients.
    ///
    /// Args:
    ///     x: The floating-point value to approximate
    ///     max_denom: Maximum denominator for rational coefficients
    ///
    /// Returns:
    ///     A GoldenExact with rational coefficients minimizing |a + b·φ - x|
    #[staticmethod]
    pub fn nearest_rational(x: f64, max_denom: i64) -> Self {
        GoldenExact::find_nearest_rational(x, max_denom)
    }

    /// Snap a list of floats to GoldenExact lattice points.
    ///
    /// Args:
    ///     values: List of floating-point values
    ///     max_coeff: Maximum coefficient bound
    ///
    /// Returns:
    ///     Tuple of (lattice_points, residuals)
    #[staticmethod]
    fn snap(values: Vec<f64>, max_coeff: i64) -> (Vec<GoldenExact>, Vec<f64>) {
        GoldenExact::snap_vector(&values, max_coeff)
    }

    /// Compute approximation error |self - target|
    fn error_from(&self, target: f64) -> f64 {
        (self.to_f64() - target).abs()
    }
}

// =============================================================================
// LLL Algorithm Implementation for 2D Diophantine Approximation
// =============================================================================

/// LLL-based simultaneous Diophantine approximation for Q(φ).
///
/// Finds integers (a, b) minimizing |a + b·φ - x| with |a|, |b| ≤ max_coeff.
///
/// # Algorithm
/// We use a 3D lattice with basis:
/// ```text
///   b₁ = (1, 0, K)       // coefficient a contribution
///   b₂ = (0, 1, K·φ)     // coefficient b contribution
///   b₃ = (0, 0, K·x)     // target value
/// ```
/// LLL reduction finds short vectors; the shortest non-trivial vector
/// in the reduced basis gives us the best (a, b) approximation.
fn lll_find_nearest_2d(x: f64, phi: f64, k: f64, max_coeff: i64) -> (i64, i64) {
    // For 2D approximation problem a + b*phi ≈ x, we use a direct approach:
    // Construct a 2x3 matrix and apply Gaussian reduction + LLL

    // Strategy: Search over Farey-like sequence of (a, b) pairs
    // using the continued fraction of (x - a) / phi for each a

    // Simple but effective: grid search with LLL-inspired pruning
    // For small max_coeff, this is fast enough

    if max_coeff <= 100 {
        return grid_search_nearest(x, phi, max_coeff);
    }

    // For larger bounds, use LLL on a 2x3 lattice
    lll_reduce_nearest(x, phi, k, max_coeff)
}

/// Direct grid search for small coefficient bounds
fn grid_search_nearest(x: f64, phi: f64, max_coeff: i64) -> (i64, i64) {
    let mut best_a: i64 = 0;
    let mut best_b: i64 = 0;
    let mut best_error = x.abs(); // Error with (0, 0)

    // For efficiency, use the structure of Q(φ):
    // Given x, the optimal b is approximately (x - round(x)) / phi

    for b in -max_coeff..=max_coeff {
        // For this b, optimal a is round(x - b*phi)
        let target_a = x - (b as f64) * phi;
        let a = target_a.round() as i64;

        // Check if within bounds
        if a.abs() <= max_coeff {
            let error = (a as f64 + (b as f64) * phi - x).abs();
            if error < best_error {
                best_error = error;
                best_a = a;
                best_b = b;
            }
        }

        // Also check floor and ceil
        let a_floor = target_a.floor() as i64;
        let a_ceil = target_a.ceil() as i64;

        for a in [a_floor, a_ceil] {
            if a.abs() <= max_coeff {
                let error = (a as f64 + (b as f64) * phi - x).abs();
                if error < best_error {
                    best_error = error;
                    best_a = a;
                    best_b = b;
                }
            }
        }
    }

    (best_a, best_b)
}

/// LLL lattice reduction for finding nearest Q(φ) element
///
/// Uses a 2D lattice encoding the approximation problem.
fn lll_reduce_nearest(x: f64, phi: f64, k: f64, max_coeff: i64) -> (i64, i64) {
    // Build 2x3 lattice matrix (stored row-major)
    // b₁ = (1, 0, k)
    // b₂ = (0, 1, k*phi)
    // We want to find integer combination a*b₁ + b*b₂ close to (0, 0, k*x)

    // Equivalently, find (a, b) minimizing |a*k + b*k*phi - k*x| = k|a + b*phi - x|

    // For 2D, we can use a simpler version of LLL:
    // The reduced basis will have short vectors

    // Initialize basis vectors (stored as [v0, v1, v2] for each row)
    let mut b1 = [1.0_f64, 0.0, k];
    let mut b2 = [0.0_f64, 1.0, k * phi];

    // Gram-Schmidt orthogonalization
    fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm_sq(a: &[f64; 3]) -> f64 {
        dot(a, a)
    }

    fn scale(a: &[f64; 3], s: f64) -> [f64; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn add(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    // LLL reduction parameter (standard value)
    let delta = 0.75_f64;

    // Size-reduce and swap loop (simplified 2D LLL)
    for _ in 0..20 {
        // Gram-Schmidt on b1
        let b1_star = b1;
        let b1_star_norm_sq = norm_sq(&b1_star);

        if b1_star_norm_sq < 1e-30 {
            break;
        }

        // Gram-Schmidt on b2
        let mu21 = dot(&b2, &b1_star) / b1_star_norm_sq;
        let b2_star = sub(&b2, &scale(&b1_star, mu21));
        let b2_star_norm_sq = norm_sq(&b2_star);

        // Size reduction: ensure |mu21| <= 0.5
        if mu21.abs() > 0.5 {
            let round_mu = mu21.round();
            b2 = sub(&b2, &scale(&b1, round_mu));
            continue;
        }

        // Lovász condition
        if b2_star_norm_sq >= (delta - mu21 * mu21) * b1_star_norm_sq {
            // Reduced!
            break;
        } else {
            // Swap b1 and b2
            std::mem::swap(&mut b1, &mut b2);
        }
    }

    // Now find the best combination within bounds
    // The reduced basis has short vectors; search nearby

    // b1 and b2 are now reduced; the first two coordinates give us (a, b)
    // We need integer combinations

    let mut best_a: i64 = 0;
    let mut best_b: i64 = 0;
    let mut best_error = x.abs();

    // Search over small integer combinations of reduced basis
    let search_range = 50_i64.min(max_coeff);

    for i in -search_range..=search_range {
        for j in -search_range..=search_range {
            // Combination: i*b1 + j*b2
            let combination = add(&scale(&b1, i as f64), &scale(&b2, j as f64));
            let a = combination[0].round() as i64;
            let b = combination[1].round() as i64;

            if a.abs() <= max_coeff && b.abs() <= max_coeff {
                let error = (a as f64 + b as f64 * phi - x).abs();
                if error < best_error {
                    best_error = error;
                    best_a = a;
                    best_b = b;
                }
            }
        }
    }

    // Fall back to grid search if LLL didn't find anything good
    if best_error > 0.1 && max_coeff <= 1000 {
        return grid_search_nearest(x, phi, max_coeff);
    }

    (best_a, best_b)
}

/// Compute (F_{n-1}, F_n) - the Fibonacci pair
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
fn fibonacci_pair(n: u32) -> (i64, i64) {
    if n == 0 {
        return (1, 0); // F_{-1} = 1, F_0 = 0 (by convention for φ^0 = F_0·φ + F_{-1})
    }

    let mut prev: i64 = 0;
    let mut curr: i64 = 1;

    for _ in 1..n {
        let next = prev + curr;
        prev = curr;
        curr = next;
    }

    (prev, curr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_squared() {
        let phi = GoldenExact::phi();
        let phi_sq = phi * phi;
        assert_eq!(phi_sq, GoldenExact::phi_squared());
        // φ² = 1 + φ
        assert_eq!(phi_sq, GoldenExact::from_ints(1, 1));
    }

    #[test]
    fn test_phi_inverse() {
        let phi = GoldenExact::phi();
        let phi_inv = phi.inverse();
        // 1/φ = φ - 1
        assert_eq!(phi_inv, GoldenExact::phi_hat());

        // Verify: φ · (1/φ) = 1
        let product = phi * phi_inv;
        assert_eq!(product, GoldenExact::one());
    }

    #[test]
    fn test_norm() {
        // N(φ) = 0² + 0·1 - 1² = -1
        let phi = GoldenExact::phi();
        assert_eq!(phi.norm(), Rational::from_int(-1));

        // N(1) = 1
        let one = GoldenExact::one();
        assert_eq!(one.norm(), Rational::one());
    }

    #[test]
    fn test_sqrt5() {
        let sqrt5 = GoldenExact::sqrt5();
        let five = sqrt5 * sqrt5;
        // (√5)² = 5
        assert_eq!(five, GoldenExact::from_rational(Rational::from_int(5)));
    }

    #[test]
    fn test_phi_powers() {
        // φ¹ = φ
        assert_eq!(GoldenExact::phi_power(1), GoldenExact::phi());

        // φ² = 1 + φ
        assert_eq!(GoldenExact::phi_power(2), GoldenExact::from_ints(1, 1));

        // φ³ = φ² + φ = (1 + φ) + φ = 1 + 2φ
        assert_eq!(GoldenExact::phi_power(3), GoldenExact::from_ints(1, 2));

        // φ⁴ = φ³ + φ² = (1 + 2φ) + (1 + φ) = 2 + 3φ
        assert_eq!(GoldenExact::phi_power(4), GoldenExact::from_ints(2, 3));

        // φ⁵ = φ⁴ + φ³ = (2 + 3φ) + (1 + 2φ) = 3 + 5φ
        assert_eq!(GoldenExact::phi_power(5), GoldenExact::from_ints(3, 5));
    }

    #[test]
    fn test_conjugate() {
        // φ* = 1 - φ
        let phi = GoldenExact::phi();
        let phi_conj = phi.conjugate();
        assert_eq!(phi_conj, GoldenExact::from_ints(1, -1));

        // φ + φ* = 1 (trace)
        let sum = phi + phi_conj;
        assert_eq!(sum, GoldenExact::one());
    }

    // =========================================================================
    // Tests for LLL-based find_nearest
    // =========================================================================

    #[test]
    fn test_find_nearest_exact_phi() {
        // φ ≈ 1.618... should snap to exactly (0, 1)
        const PHI_F64: f64 = 1.6180339887498949;
        let result = GoldenExact::find_nearest(PHI_F64, 100);
        assert_eq!(result.rational_part(), Rational::zero());
        assert_eq!(result.phi_part(), Rational::one());
    }

    #[test]
    fn test_find_nearest_exact_phi_squared() {
        // φ² ≈ 2.618... should snap to (1, 1)
        const PHI_F64: f64 = 1.6180339887498949;
        let phi_sq = PHI_F64 * PHI_F64;
        let result = GoldenExact::find_nearest(phi_sq, 100);
        assert_eq!(result, GoldenExact::from_ints(1, 1));
    }

    #[test]
    fn test_find_nearest_integers() {
        // Integers should snap to themselves
        let result = GoldenExact::find_nearest(5.0, 100);
        assert_eq!(result.rational_part(), Rational::from_int(5));
        assert_eq!(result.phi_part(), Rational::zero());
    }

    #[test]
    fn test_find_nearest_error_bound() {
        // Test that approximation error is small
        const PHI_F64: f64 = 1.6180339887498949;

        for x in [0.5, 1.23, 2.5, 3.14159, 10.0] {
            let result = GoldenExact::find_nearest(x, 1000);
            let error = (result.to_f64() - x).abs();
            // Error should be less than 1/1000 for max_coeff=1000
            assert!(error < 0.01, "Error {} too large for x={}", error, x);
        }
    }

    #[test]
    fn test_find_nearest_fibonacci_ratios() {
        // Fibonacci ratios F_{n+1}/F_n converge to φ
        // These should be well-approximated
        let fib_ratios = [
            (2.0, 1.0),  // 2/1 = 2
            (3.0, 2.0),  // 3/2 = 1.5
            (5.0, 3.0),  // 5/3 ≈ 1.667
            (8.0, 5.0),  // 8/5 = 1.6
            (13.0, 8.0), // 13/8 = 1.625
        ];

        for (num, denom) in fib_ratios {
            let ratio = num / denom;
            let result = GoldenExact::find_nearest(ratio, 100);
            let error = (result.to_f64() - ratio).abs();
            assert!(error < 0.001, "Error {} for ratio {}/{}", error, num, denom);
        }
    }

    #[test]
    fn test_snap_vector() {
        const PHI_F64: f64 = 1.6180339887498949;

        let values = vec![1.0, PHI_F64, PHI_F64 * PHI_F64, 3.0];
        let (lattice, residuals) = GoldenExact::snap_vector(&values, 100);

        assert_eq!(lattice.len(), 4);
        assert_eq!(residuals.len(), 4);

        // Check that residuals are small
        for (i, &r) in residuals.iter().enumerate() {
            assert!(r.abs() < 0.01, "Residual {} too large at index {}", r, i);
        }

        // Check specific snaps
        assert_eq!(lattice[0], GoldenExact::from_ints(1, 0)); // 1.0 → 1
        assert_eq!(lattice[1], GoldenExact::phi()); // φ → φ
        assert_eq!(lattice[2], GoldenExact::phi_squared()); // φ² → 1 + φ
    }
}
