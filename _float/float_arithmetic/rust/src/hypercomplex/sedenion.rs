use pyo3::prelude::*;

/// Sedenion: 16-dimensional hypercomplex number from Cayley-Dickson construction
///
/// Components: e0 (real), e1-e15 (imaginary basis elements)
/// Constructed as pairs of octonions: Sedenion = (Octonion_low, Octonion_high)
///
/// # WARNING: ZERO DIVISORS
///
/// Unlike quaternions and octonions, sedenions have ZERO DIVISORS.
/// This means there exist non-zero sedenions a, b such that a * b = 0.
/// Consequences:
/// - Division is not always well-defined
/// - inverse() may produce unexpected results for zero divisor elements
/// - The composition property ||ab|| = ||a|| ||b|| does NOT hold in general
/// - Sedenions are NOT a division algebra
///
/// Use `has_zero_divisor_with(other)` to check before critical operations.
///
/// # SRT Note
///
/// Sedenions are NOT currently used in SRT theory. Octonions are the "end of the line"
/// for exceptional geometry (E₈, G₂). This implementation enables mathematical
/// exploration and completeness of the Cayley-Dickson construction.
///
/// # Cayley-Dickson Construction
///
/// For sedenions (a,b) and (c,d) where a,b,c,d are octonions:
/// ```text
/// (a,b) * (c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
/// ```
///
/// Properties lost at the sedenion level:
/// - Alternativity (octonions are alternative, sedenions are not)
/// - Power-associativity is preserved
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Sedenion {
    #[pyo3(get, set)]
    pub e0: f64,
    #[pyo3(get, set)]
    pub e1: f64,
    #[pyo3(get, set)]
    pub e2: f64,
    #[pyo3(get, set)]
    pub e3: f64,
    #[pyo3(get, set)]
    pub e4: f64,
    #[pyo3(get, set)]
    pub e5: f64,
    #[pyo3(get, set)]
    pub e6: f64,
    #[pyo3(get, set)]
    pub e7: f64,
    #[pyo3(get, set)]
    pub e8: f64,
    #[pyo3(get, set)]
    pub e9: f64,
    #[pyo3(get, set)]
    pub e10: f64,
    #[pyo3(get, set)]
    pub e11: f64,
    #[pyo3(get, set)]
    pub e12: f64,
    #[pyo3(get, set)]
    pub e13: f64,
    #[pyo3(get, set)]
    pub e14: f64,
    #[pyo3(get, set)]
    pub e15: f64,
}

// Internal helper: octonion multiplication for Cayley-Dickson
// Returns the 8 components of the product of two octonions
#[inline]
fn octonion_mul(
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    b5: f64,
    b6: f64,
    b7: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    // Octonion multiplication using Fano plane structure
    // Same table as in octonion.rs

    // e0 (real part)
    let r0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7;

    // e1
    let r1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6;

    // e2
    let r2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5;

    // e3
    let r3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4;

    // e4
    let r4 = a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3;

    // e5
    let r5 = a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2;

    // e6
    let r6 = a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1;

    // e7
    let r7 = a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0;

    (r0, r1, r2, r3, r4, r5, r6, r7)
}

// Internal helper: octonion conjugate
#[inline]
fn octonion_conj(
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    (a0, -a1, -a2, -a3, -a4, -a5, -a6, -a7)
}

// Internal helper: octonion subtraction
#[inline]
fn octonion_sub(
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    b5: f64,
    b6: f64,
    b7: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    (
        a0 - b0,
        a1 - b1,
        a2 - b2,
        a3 - b3,
        a4 - b4,
        a5 - b5,
        a6 - b6,
        a7 - b7,
    )
}

// Internal helper: octonion addition
#[inline]
fn octonion_add(
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    b5: f64,
    b6: f64,
    b7: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    (
        a0 + b0,
        a1 + b1,
        a2 + b2,
        a3 + b3,
        a4 + b4,
        a5 + b5,
        a6 + b6,
        a7 + b7,
    )
}

#[pymethods]
impl Sedenion {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        e0: f64,
        e1: f64,
        e2: f64,
        e3: f64,
        e4: f64,
        e5: f64,
        e6: f64,
        e7: f64,
        e8: f64,
        e9: f64,
        e10: f64,
        e11: f64,
        e12: f64,
        e13: f64,
        e14: f64,
        e15: f64,
    ) -> Self {
        Sedenion {
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
            e8,
            e9,
            e10,
            e11,
            e12,
            e13,
            e14,
            e15,
        }
    }

    /// Real (scalar) part
    #[getter]
    pub fn real(&self) -> f64 {
        self.e0
    }

    /// Imaginary part as list [e1, ..., e15]
    #[getter]
    pub fn imag(&self) -> Vec<f64> {
        vec![
            self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8, self.e9,
            self.e10, self.e11, self.e12, self.e13, self.e14, self.e15,
        ]
    }

    /// Low octonion components (e0-e7)
    pub fn octonion_low(&self) -> Vec<f64> {
        vec![
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
        ]
    }

    /// High octonion components (e8-e15)
    pub fn octonion_high(&self) -> Vec<f64> {
        vec![
            self.e8, self.e9, self.e10, self.e11, self.e12, self.e13, self.e14, self.e15,
        ]
    }

    /// Addition
    fn __add__(&self, other: &Sedenion) -> Self {
        Sedenion::new(
            self.e0 + other.e0,
            self.e1 + other.e1,
            self.e2 + other.e2,
            self.e3 + other.e3,
            self.e4 + other.e4,
            self.e5 + other.e5,
            self.e6 + other.e6,
            self.e7 + other.e7,
            self.e8 + other.e8,
            self.e9 + other.e9,
            self.e10 + other.e10,
            self.e11 + other.e11,
            self.e12 + other.e12,
            self.e13 + other.e13,
            self.e14 + other.e14,
            self.e15 + other.e15,
        )
    }

    /// Subtraction
    fn __sub__(&self, other: &Sedenion) -> Self {
        Sedenion::new(
            self.e0 - other.e0,
            self.e1 - other.e1,
            self.e2 - other.e2,
            self.e3 - other.e3,
            self.e4 - other.e4,
            self.e5 - other.e5,
            self.e6 - other.e6,
            self.e7 - other.e7,
            self.e8 - other.e8,
            self.e9 - other.e9,
            self.e10 - other.e10,
            self.e11 - other.e11,
            self.e12 - other.e12,
            self.e13 - other.e13,
            self.e14 - other.e14,
            self.e15 - other.e15,
        )
    }

    /// Negation
    fn __neg__(&self) -> Self {
        Sedenion::new(
            -self.e0, -self.e1, -self.e2, -self.e3, -self.e4, -self.e5, -self.e6, -self.e7,
            -self.e8, -self.e9, -self.e10, -self.e11, -self.e12, -self.e13, -self.e14, -self.e15,
        )
    }

    /// Scalar multiplication (right)
    fn __rmul__(&self, scalar: f64) -> Self {
        Sedenion::new(
            scalar * self.e0,
            scalar * self.e1,
            scalar * self.e2,
            scalar * self.e3,
            scalar * self.e4,
            scalar * self.e5,
            scalar * self.e6,
            scalar * self.e7,
            scalar * self.e8,
            scalar * self.e9,
            scalar * self.e10,
            scalar * self.e11,
            scalar * self.e12,
            scalar * self.e13,
            scalar * self.e14,
            scalar * self.e15,
        )
    }

    /// Scale by scalar
    pub fn scale(&self, scalar: f64) -> Self {
        self.__rmul__(scalar)
    }

    /// Cayley-Dickson multiplication (non-associative, non-alternative!)
    ///
    /// For sedenions (a,b) and (c,d) where a,b,c,d are octonions:
    /// (a,b) * (c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
    ///
    /// WARNING: Due to zero divisors, ||s*t|| may not equal ||s||*||t||
    fn __mul__(&self, other: &Sedenion) -> Self {
        // Extract octonion halves
        // self = (a, b) where a = (e0..e7), b = (e8..e15)
        // other = (c, d) where c = (e0..e7), d = (e8..e15)

        let (a0, a1, a2, a3, a4, a5, a6, a7) = (
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
        );
        let (b0, b1, b2, b3, b4, b5, b6, b7) = (
            self.e8, self.e9, self.e10, self.e11, self.e12, self.e13, self.e14, self.e15,
        );
        let (c0, c1, c2, c3, c4, c5, c6, c7) = (
            other.e0, other.e1, other.e2, other.e3, other.e4, other.e5, other.e6, other.e7,
        );
        let (d0, d1, d2, d3, d4, d5, d6, d7) = (
            other.e8, other.e9, other.e10, other.e11, other.e12, other.e13, other.e14, other.e15,
        );

        // Cayley-Dickson formula: (a,b) * (c,d) = (a*c - conj(d)*b, d*a + b*conj(c))

        // Compute a*c
        let ac = octonion_mul(
            a0, a1, a2, a3, a4, a5, a6, a7, c0, c1, c2, c3, c4, c5, c6, c7,
        );

        // Compute conj(d)
        let d_conj = octonion_conj(d0, d1, d2, d3, d4, d5, d6, d7);

        // Compute conj(d) * b
        let d_conj_b = octonion_mul(
            d_conj.0, d_conj.1, d_conj.2, d_conj.3, d_conj.4, d_conj.5, d_conj.6, d_conj.7, b0, b1,
            b2, b3, b4, b5, b6, b7,
        );

        // Compute result_low = a*c - conj(d)*b
        let result_low = octonion_sub(
            ac.0, ac.1, ac.2, ac.3, ac.4, ac.5, ac.6, ac.7, d_conj_b.0, d_conj_b.1, d_conj_b.2,
            d_conj_b.3, d_conj_b.4, d_conj_b.5, d_conj_b.6, d_conj_b.7,
        );

        // Compute d*a
        let da = octonion_mul(
            d0, d1, d2, d3, d4, d5, d6, d7, a0, a1, a2, a3, a4, a5, a6, a7,
        );

        // Compute conj(c)
        let c_conj = octonion_conj(c0, c1, c2, c3, c4, c5, c6, c7);

        // Compute b * conj(c)
        let b_c_conj = octonion_mul(
            b0, b1, b2, b3, b4, b5, b6, b7, c_conj.0, c_conj.1, c_conj.2, c_conj.3, c_conj.4,
            c_conj.5, c_conj.6, c_conj.7,
        );

        // Compute result_high = d*a + b*conj(c)
        let result_high = octonion_add(
            da.0, da.1, da.2, da.3, da.4, da.5, da.6, da.7, b_c_conj.0, b_c_conj.1, b_c_conj.2,
            b_c_conj.3, b_c_conj.4, b_c_conj.5, b_c_conj.6, b_c_conj.7,
        );

        Sedenion::new(
            result_low.0,
            result_low.1,
            result_low.2,
            result_low.3,
            result_low.4,
            result_low.5,
            result_low.6,
            result_low.7,
            result_high.0,
            result_high.1,
            result_high.2,
            result_high.3,
            result_high.4,
            result_high.5,
            result_high.6,
            result_high.7,
        )
    }

    /// Norm squared: sum of squares of all components
    pub fn norm_sq(&self) -> f64 {
        self.e0 * self.e0
            + self.e1 * self.e1
            + self.e2 * self.e2
            + self.e3 * self.e3
            + self.e4 * self.e4
            + self.e5 * self.e5
            + self.e6 * self.e6
            + self.e7 * self.e7
            + self.e8 * self.e8
            + self.e9 * self.e9
            + self.e10 * self.e10
            + self.e11 * self.e11
            + self.e12 * self.e12
            + self.e13 * self.e13
            + self.e14 * self.e14
            + self.e15 * self.e15
    }

    /// Euclidean norm
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Conjugate: negate all imaginary parts
    pub fn conjugate(&self) -> Self {
        Sedenion::new(
            self.e0, -self.e1, -self.e2, -self.e3, -self.e4, -self.e5, -self.e6, -self.e7,
            -self.e8, -self.e9, -self.e10, -self.e11, -self.e12, -self.e13, -self.e14, -self.e15,
        )
    }

    /// Multiplicative inverse: conj / norm_sq
    ///
    /// WARNING: For zero divisors, this may not behave as expected.
    /// The product s * s.inverse() may not equal 1 for zero divisors.
    /// Use has_zero_divisor_with() to check before relying on inverse.
    pub fn inverse(&self) -> Self {
        let n2 = self.norm_sq();
        let conj = self.conjugate();
        Sedenion::new(
            conj.e0 / n2,
            conj.e1 / n2,
            conj.e2 / n2,
            conj.e3 / n2,
            conj.e4 / n2,
            conj.e5 / n2,
            conj.e6 / n2,
            conj.e7 / n2,
            conj.e8 / n2,
            conj.e9 / n2,
            conj.e10 / n2,
            conj.e11 / n2,
            conj.e12 / n2,
            conj.e13 / n2,
            conj.e14 / n2,
            conj.e15 / n2,
        )
    }

    /// Division: self * other.inverse()
    ///
    /// WARNING: Division may not be well-defined for zero divisors.
    fn __truediv__(&self, other: &Sedenion) -> Self {
        let inv = other.inverse();
        self.__mul__(&inv)
    }

    /// Normalize to unit sedenion
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            return *self;
        }
        Sedenion::new(
            self.e0 / n,
            self.e1 / n,
            self.e2 / n,
            self.e3 / n,
            self.e4 / n,
            self.e5 / n,
            self.e6 / n,
            self.e7 / n,
            self.e8 / n,
            self.e9 / n,
            self.e10 / n,
            self.e11 / n,
            self.e12 / n,
            self.e13 / n,
            self.e14 / n,
            self.e15 / n,
        )
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Sedenion({:.6} + {:.6}e1 + {:.6}e2 + {:.6}e3 + {:.6}e4 + {:.6}e5 + {:.6}e6 + {:.6}e7 + \
                     {:.6}e8 + {:.6}e9 + {:.6}e10 + {:.6}e11 + {:.6}e12 + {:.6}e13 + {:.6}e14 + {:.6}e15)",
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
            self.e8, self.e9, self.e10, self.e11, self.e12, self.e13, self.e14, self.e15
        )
    }

    /// Associator: [a, b, c] = (a*b)*c - a*(b*c)
    /// Non-zero for sedenions (measures non-associativity)
    #[staticmethod]
    pub fn associator(a: &Sedenion, b: &Sedenion, c: &Sedenion) -> Sedenion {
        let ab = a.__mul__(b);
        let bc = b.__mul__(c);
        let ab_c = ab.__mul__(c);
        let a_bc = a.__mul__(&bc);
        ab_c.__sub__(&a_bc)
    }

    /// Commutator: [a, b] = a*b - b*a
    /// Non-zero for sedenions (measures non-commutativity)
    #[staticmethod]
    pub fn commutator(a: &Sedenion, b: &Sedenion) -> Sedenion {
        let ab = a.__mul__(b);
        let ba = b.__mul__(a);
        ab.__sub__(&ba)
    }

    /// Check if product with other is (approximately) zero
    ///
    /// This is the key diagnostic for zero divisors: if a*b ≈ 0 and
    /// both a and b are non-zero, then (a, b) form a zero divisor pair.
    pub fn has_zero_divisor_with(&self, other: &Sedenion) -> bool {
        let product = self.__mul__(other);
        let product_norm = product.norm();
        let self_norm = self.norm();
        let other_norm = other.norm();

        // Both must be non-zero, but product must be essentially zero
        // Use relative tolerance
        let tol = 1e-10 * (self_norm * other_norm).max(1e-10);

        self_norm > 1e-10 && other_norm > 1e-10 && product_norm < tol
    }

    /// Check if this sedenion is a potential zero divisor
    ///
    /// A sedenion of the form (a, b) where a and b are octonions
    /// can be a zero divisor if ||a|| = ||b|| (same norm halves).
    /// This is a necessary but not sufficient condition.
    ///
    /// Known zero divisor pairs include:
    /// - (e0 + e8) * (e0 - e8) = 0
    /// - More generally: (a, a) * (b, -b) = 0 for some octonions a, b
    pub fn is_potential_zero_divisor(&self) -> bool {
        // Compute norms of octonion halves
        let low_norm_sq = self.e0 * self.e0
            + self.e1 * self.e1
            + self.e2 * self.e2
            + self.e3 * self.e3
            + self.e4 * self.e4
            + self.e5 * self.e5
            + self.e6 * self.e6
            + self.e7 * self.e7;
        let high_norm_sq = self.e8 * self.e8
            + self.e9 * self.e9
            + self.e10 * self.e10
            + self.e11 * self.e11
            + self.e12 * self.e12
            + self.e13 * self.e13
            + self.e14 * self.e14
            + self.e15 * self.e15;

        // If halves have equal norm, this could be a zero divisor
        let tol = 1e-10 * (low_norm_sq + high_norm_sq).max(1e-10);
        (low_norm_sq - high_norm_sq).abs() < tol
    }

    /// Check if sedenion is purely imaginary (real part ~ 0)
    pub fn is_pure(&self) -> bool {
        self.e0.abs() < 1e-10
    }

    /// Dot product (real part of a * conj(b))
    #[staticmethod]
    pub fn dot(a: &Sedenion, b: &Sedenion) -> f64 {
        a.e0 * b.e0
            + a.e1 * b.e1
            + a.e2 * b.e2
            + a.e3 * b.e3
            + a.e4 * b.e4
            + a.e5 * b.e5
            + a.e6 * b.e6
            + a.e7 * b.e7
            + a.e8 * b.e8
            + a.e9 * b.e9
            + a.e10 * b.e10
            + a.e11 * b.e11
            + a.e12 * b.e12
            + a.e13 * b.e13
            + a.e14 * b.e14
            + a.e15 * b.e15
    }

    /// Create a sedenion from two lists of 8 components (octonion halves)
    #[staticmethod]
    pub fn from_octonions(low: Vec<f64>, high: Vec<f64>) -> PyResult<Self> {
        if low.len() != 8 || high.len() != 8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each octonion must have exactly 8 components",
            ));
        }
        Ok(Sedenion::new(
            low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7], high[0], high[1],
            high[2], high[3], high[4], high[5], high[6], high[7],
        ))
    }

    /// Create a sedenion from a list of 16 components
    #[staticmethod]
    pub fn from_list(components: Vec<f64>) -> PyResult<Self> {
        if components.len() != 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sedenion requires exactly 16 components",
            ));
        }
        Ok(Sedenion::new(
            components[0],
            components[1],
            components[2],
            components[3],
            components[4],
            components[5],
            components[6],
            components[7],
            components[8],
            components[9],
            components[10],
            components[11],
            components[12],
            components[13],
            components[14],
            components[15],
        ))
    }

    /// Convert to a list of 16 components
    pub fn to_list(&self) -> Vec<f64> {
        vec![
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8,
            self.e9, self.e10, self.e11, self.e12, self.e13, self.e14, self.e15,
        ]
    }

    /// Create a known zero divisor pair
    ///
    /// A verified zero divisor pair in sedenions:
    /// a = e3 + e10 (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    /// b = e6 - e15 (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1)
    /// Then a * b = 0
    ///
    /// This is one of the canonical zero divisor pairs arising from
    /// the Cayley-Dickson construction.
    #[staticmethod]
    pub fn zero_divisor_pair() -> (Sedenion, Sedenion) {
        // e3 + e10
        let a = Sedenion::new(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        );
        // e6 - e15
        let b = Sedenion::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
        );
        (a, b)
    }

    /// Create sedenion basis element e_i
    #[staticmethod]
    pub fn basis(i: usize) -> PyResult<Self> {
        if i > 15 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Basis index must be 0-15",
            ));
        }
        let mut components = [0.0f64; 16];
        components[i] = 1.0;
        Ok(Sedenion::new(
            components[0],
            components[1],
            components[2],
            components[3],
            components[4],
            components[5],
            components[6],
            components[7],
            components[8],
            components[9],
            components[10],
            components[11],
            components[12],
            components[13],
            components[14],
            components[15],
        ))
    }

    /// Identity element (e0 = 1)
    #[staticmethod]
    pub fn one() -> Self {
        Sedenion::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    }

    /// Zero element
    #[staticmethod]
    pub fn zero() -> Self {
        Sedenion::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    }

    /// Check approximate equality
    pub fn approx_eq(&self, other: &Sedenion, tol: f64) -> bool {
        (self.e0 - other.e0).abs() < tol
            && (self.e1 - other.e1).abs() < tol
            && (self.e2 - other.e2).abs() < tol
            && (self.e3 - other.e3).abs() < tol
            && (self.e4 - other.e4).abs() < tol
            && (self.e5 - other.e5).abs() < tol
            && (self.e6 - other.e6).abs() < tol
            && (self.e7 - other.e7).abs() < tol
            && (self.e8 - other.e8).abs() < tol
            && (self.e9 - other.e9).abs() < tol
            && (self.e10 - other.e10).abs() < tol
            && (self.e11 - other.e11).abs() < tol
            && (self.e12 - other.e12).abs() < tol
            && (self.e13 - other.e13).abs() < tol
            && (self.e14 - other.e14).abs() < tol
            && (self.e15 - other.e15).abs() < tol
    }
}

// Implement standard Rust traits for non-Python use
impl std::ops::Add for Sedenion {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.__add__(&other)
    }
}

impl std::ops::Sub for Sedenion {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.__sub__(&other)
    }
}

impl std::ops::Neg for Sedenion {
    type Output = Self;
    fn neg(self) -> Self {
        self.__neg__()
    }
}

impl std::ops::Mul for Sedenion {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.__mul__(&other)
    }
}

impl std::ops::Mul<f64> for Sedenion {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        self.__rmul__(scalar)
    }
}

impl std::ops::Div for Sedenion {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.__truediv__(&other)
    }
}

impl Default for Sedenion {
    fn default() -> Self {
        Sedenion::zero()
    }
}

impl PartialEq for Sedenion {
    fn eq(&self, other: &Self) -> bool {
        self.e0 == other.e0
            && self.e1 == other.e1
            && self.e2 == other.e2
            && self.e3 == other.e3
            && self.e4 == other.e4
            && self.e5 == other.e5
            && self.e6 == other.e6
            && self.e7 == other.e7
            && self.e8 == other.e8
            && self.e9 == other.e9
            && self.e10 == other.e10
            && self.e11 == other.e11
            && self.e12 == other.e12
            && self.e13 == other.e13
            && self.e14 == other.e14
            && self.e15 == other.e15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_basic() {
        let s = Sedenion::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq!(s.e0, 1.0);
        assert_eq!(s.e15, 16.0);
    }

    #[test]
    fn test_sedenion_conjugate() {
        let s = Sedenion::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let c = s.conjugate();
        assert_eq!(c.e0, 1.0); // Real part unchanged
        assert_eq!(c.e1, -2.0); // Imaginary negated
        assert_eq!(c.e15, -16.0);
    }

    #[test]
    fn test_sedenion_norm() {
        // Sum of squares: 1 + 4 + 9 + ... + 256 = 1496
        let s = Sedenion::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let expected_norm_sq = (1..=16).map(|i| (i * i) as f64).sum::<f64>();
        assert!((s.norm_sq() - expected_norm_sq).abs() < 1e-10);
    }

    #[test]
    fn test_sedenion_identity() {
        let one = Sedenion::one();
        let s = Sedenion::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let result = s.__mul__(&one);
        assert!(s.approx_eq(&result, 1e-10));
    }

    #[test]
    fn test_zero_divisor() {
        // Known zero divisor pair: (e3 + e10) * (e6 - e15) = 0
        let (a, b) = Sedenion::zero_divisor_pair();
        let product = a.__mul__(&b);
        assert!(
            product.norm() < 1e-10,
            "Zero divisor product should be zero, got {}",
            product.norm()
        );
        assert!(
            a.has_zero_divisor_with(&b),
            "Should detect zero divisor pair"
        );
    }

    #[test]
    fn test_sedenion_s_conj_s_equals_norm_sq() {
        let s = Sedenion::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let product = s.__mul__(&s.conjugate());
        // s * conj(s) should be real and equal to norm_sq
        let expected_norm_sq = s.norm_sq();
        assert!(
            (product.e0 - expected_norm_sq).abs() < 1e-10,
            "s*conj(s) real part should equal norm_sq"
        );
        // Imaginary parts should be ~0
        assert!(
            product.imag().iter().all(|x| x.abs() < 1e-10),
            "s*conj(s) imaginary parts should be zero"
        );
    }
}
