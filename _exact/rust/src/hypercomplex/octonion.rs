use pyo3::prelude::*;

/// Octonion: 8-dimensional non-associative division algebra
/// Components: e0 (real), e1-e7 (imaginary basis elements)
/// Multiplication follows Cayley-Dickson construction from quaternions
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Octonion {
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
}

#[pymethods]
impl Octonion {
    #[new]
    pub fn new(e0: f64, e1: f64, e2: f64, e3: f64, e4: f64, e5: f64, e6: f64, e7: f64) -> Self {
        Octonion {
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
        }
    }

    /// Real (scalar) part
    #[getter]
    pub fn real(&self) -> f64 {
        self.e0
    }

    /// Imaginary part as list [e1, e2, e3, e4, e5, e6, e7]
    #[getter]
    pub fn imag(&self) -> Vec<f64> {
        vec![
            self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
        ]
    }

    /// Addition
    fn __add__(&self, other: &Octonion) -> Self {
        Octonion::new(
            self.e0 + other.e0,
            self.e1 + other.e1,
            self.e2 + other.e2,
            self.e3 + other.e3,
            self.e4 + other.e4,
            self.e5 + other.e5,
            self.e6 + other.e6,
            self.e7 + other.e7,
        )
    }

    /// Subtraction
    fn __sub__(&self, other: &Octonion) -> Self {
        Octonion::new(
            self.e0 - other.e0,
            self.e1 - other.e1,
            self.e2 - other.e2,
            self.e3 - other.e3,
            self.e4 - other.e4,
            self.e5 - other.e5,
            self.e6 - other.e6,
            self.e7 - other.e7,
        )
    }

    /// Negation
    fn __neg__(&self) -> Self {
        Octonion::new(
            -self.e0, -self.e1, -self.e2, -self.e3, -self.e4, -self.e5, -self.e6, -self.e7,
        )
    }

    /// Cayley-Dickson multiplication (non-associative!)
    /// Uses Fano plane multiplication table
    fn __mul__(&self, other: &Octonion) -> Self {
        // Octonion multiplication table based on Fano plane
        // e_i * e_j = -delta_ij + epsilon_ijk * e_k
        // where epsilon_ijk is the Fano plane structure constants

        let a = self;
        let b = other;

        // Real part: a0*b0 - sum(ai*bi)
        let e0 = a.e0 * b.e0
            - a.e1 * b.e1
            - a.e2 * b.e2
            - a.e3 * b.e3
            - a.e4 * b.e4
            - a.e5 * b.e5
            - a.e6 * b.e6
            - a.e7 * b.e7;

        // e1 component
        let e1 = a.e0 * b.e1 + a.e1 * b.e0 + a.e2 * b.e3 - a.e3 * b.e2 + a.e4 * b.e5
            - a.e5 * b.e4
            - a.e6 * b.e7
            + a.e7 * b.e6;

        // e2 component
        let e2 = a.e0 * b.e2 - a.e1 * b.e3 + a.e2 * b.e0 + a.e3 * b.e1 + a.e4 * b.e6 + a.e5 * b.e7
            - a.e6 * b.e4
            - a.e7 * b.e5;

        // e3 component
        let e3 = a.e0 * b.e3 + a.e1 * b.e2 - a.e2 * b.e1 + a.e3 * b.e0 + a.e4 * b.e7 - a.e5 * b.e6
            + a.e6 * b.e5
            - a.e7 * b.e4;

        // e4 component
        let e4 = a.e0 * b.e4 - a.e1 * b.e5 - a.e2 * b.e6 - a.e3 * b.e7
            + a.e4 * b.e0
            + a.e5 * b.e1
            + a.e6 * b.e2
            + a.e7 * b.e3;

        // e5 component
        let e5 = a.e0 * b.e5 + a.e1 * b.e4 - a.e2 * b.e7 + a.e3 * b.e6 - a.e4 * b.e1 + a.e5 * b.e0
            - a.e6 * b.e3
            + a.e7 * b.e2;

        // e6 component
        let e6 = a.e0 * b.e6 + a.e1 * b.e7 + a.e2 * b.e4 - a.e3 * b.e5 - a.e4 * b.e2
            + a.e5 * b.e3
            + a.e6 * b.e0
            - a.e7 * b.e1;

        // e7 component
        let e7 = a.e0 * b.e7 - a.e1 * b.e6 + a.e2 * b.e5 + a.e3 * b.e4 - a.e4 * b.e3 - a.e5 * b.e2
            + a.e6 * b.e1
            + a.e7 * b.e0;

        Octonion::new(e0, e1, e2, e3, e4, e5, e6, e7)
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
    }

    /// Euclidean norm
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Conjugate: negate all imaginary parts
    pub fn conjugate(&self) -> Self {
        Octonion::new(
            self.e0, -self.e1, -self.e2, -self.e3, -self.e4, -self.e5, -self.e6, -self.e7,
        )
    }

    /// Multiplicative inverse: conj / norm_sq
    pub fn inverse(&self) -> Self {
        let n2 = self.norm_sq();
        let conj = self.conjugate();
        Octonion::new(
            conj.e0 / n2,
            conj.e1 / n2,
            conj.e2 / n2,
            conj.e3 / n2,
            conj.e4 / n2,
            conj.e5 / n2,
            conj.e6 / n2,
            conj.e7 / n2,
        )
    }

    /// Division: self * other.inverse()
    fn __truediv__(&self, other: &Octonion) -> Self {
        let inv = other.inverse();
        self.__mul__(&inv)
    }

    /// Normalize to unit octonion
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            return *self;
        }
        Octonion::new(
            self.e0 / n,
            self.e1 / n,
            self.e2 / n,
            self.e3 / n,
            self.e4 / n,
            self.e5 / n,
            self.e6 / n,
            self.e7 / n,
        )
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Octonion({:.6} + {:.6}e1 + {:.6}e2 + {:.6}e3 + {:.6}e4 + {:.6}e5 + {:.6}e6 + {:.6}e7)",
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7
        )
    }

    /// Associator: [a, b, c] = (a*b)*c - a*(b*c)
    /// Non-zero for octonions (measures non-associativity)
    #[staticmethod]
    pub fn associator(a: &Octonion, b: &Octonion, c: &Octonion) -> Octonion {
        let ab = a.__mul__(b);
        let bc = b.__mul__(c);
        let ab_c = ab.__mul__(c);
        let a_bc = a.__mul__(&bc);
        ab_c.__sub__(&a_bc)
    }

    /// Check if octonion is purely imaginary (real part ~ 0)
    pub fn is_pure(&self) -> bool {
        self.e0.abs() < 1e-10
    }

    /// Dot product (real part of a * conj(b))
    #[staticmethod]
    pub fn dot(a: &Octonion, b: &Octonion) -> f64 {
        a.e0 * b.e0
            + a.e1 * b.e1
            + a.e2 * b.e2
            + a.e3 * b.e3
            + a.e4 * b.e4
            + a.e5 * b.e5
            + a.e6 * b.e6
            + a.e7 * b.e7
    }

    /// Cross product (imaginary part of a * conj(b) - b * conj(a)) / 2
    /// Returns the G2-symmetric part
    #[staticmethod]
    pub fn cross(a: &Octonion, b: &Octonion) -> Octonion {
        let a_bconj = a.__mul__(&b.conjugate());
        let b_aconj = b.__mul__(&a.conjugate());
        let diff = a_bconj.__sub__(&b_aconj);
        Octonion::new(
            0.0,
            diff.e1 / 2.0,
            diff.e2 / 2.0,
            diff.e3 / 2.0,
            diff.e4 / 2.0,
            diff.e5 / 2.0,
            diff.e6 / 2.0,
            diff.e7 / 2.0,
        )
    }
}
