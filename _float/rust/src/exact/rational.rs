//! Exact Rational Numbers (Q)
//!
//! Arbitrary-precision rational numbers for exact arithmetic.
//! Uses i128 for numerator and denominator to handle large intermediate values.

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Compute greatest common divisor using Euclidean algorithm
fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Exact rational number: num/denom where gcd(num, denom) = 1 and denom > 0
#[pyclass]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Rational {
    #[pyo3(get)]
    num: i128,
    #[pyo3(get)]
    denom: i128,
}

impl Rational {
    /// Create a new rational, automatically reducing to lowest terms
    pub fn new(num: i128, denom: i128) -> Self {
        if denom == 0 {
            panic!("Rational: division by zero");
        }

        let g = gcd(num.abs(), denom.abs());
        let sign = if denom < 0 { -1 } else { 1 };

        Rational {
            num: sign * num / g,
            denom: sign * denom / g,
        }
    }

    /// Create from integer
    pub fn from_int(n: i128) -> Self {
        Rational { num: n, denom: 1 }
    }

    /// Zero
    pub fn zero() -> Self {
        Rational { num: 0, denom: 1 }
    }

    /// One
    pub fn one() -> Self {
        Rational { num: 1, denom: 1 }
    }

    /// Numerator
    pub fn numerator(&self) -> i128 {
        self.num
    }

    /// Denominator
    pub fn denominator(&self) -> i128 {
        self.denom
    }

    /// Is this rational exactly zero?
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Is this rational exactly one?
    pub fn is_one(&self) -> bool {
        self.num == 1 && self.denom == 1
    }

    /// Is this a positive rational?
    pub fn is_positive(&self) -> bool {
        self.num > 0
    }

    /// Is this a negative rational?
    pub fn is_negative(&self) -> bool {
        self.num < 0
    }

    /// Is this an integer (denominator = 1)?
    pub fn is_integer(&self) -> bool {
        self.denom == 1
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        Rational {
            num: self.num.abs(),
            denom: self.denom,
        }
    }

    /// Reciprocal (1/x)
    pub fn recip(&self) -> Self {
        if self.num == 0 {
            panic!("Rational: reciprocal of zero");
        }
        Rational::new(self.denom, self.num)
    }

    /// Integer power
    pub fn pow(&self, exp: i32) -> Self {
        if exp == 0 {
            return Rational::one();
        }
        if exp < 0 {
            return self.recip().pow(-exp);
        }

        // Binary exponentiation
        let mut result = Rational::one();
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

    /// Floor: largest integer <= self
    pub fn floor(&self) -> i128 {
        if self.num >= 0 {
            self.num / self.denom
        } else {
            // For negative numbers, we need to round toward negative infinity
            (self.num - self.denom + 1) / self.denom
        }
    }

    /// Ceiling: smallest integer >= self
    pub fn ceil(&self) -> i128 {
        if self.num <= 0 {
            self.num / self.denom
        } else {
            (self.num + self.denom - 1) / self.denom
        }
    }

    /// Convert to f64 (for numerical verification ONLY)
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.denom as f64
    }

    /// Convert to integer approximation scaled by factor
    /// Used for BASIS_SCALE conversions: rational * scale → i128
    /// Example: Rational(1,2) with scale=1000 → 500
    pub fn to_integer_approx(&self, scale: i128) -> i128 {
        (self.num * scale) / self.denom
    }

    /// Try to parse from f64 with given tolerance (for importing legacy data)
    /// Returns None if no exact representation found within tolerance
    pub fn from_f64_approx(x: f64, max_denom: i128) -> Option<Self> {
        if x.is_nan() || x.is_infinite() {
            return None;
        }

        // Continued fraction approximation
        let sign = if x < 0.0 { -1 } else { 1 };
        let x = x.abs();

        let mut p0: i128 = 0;
        let mut q0: i128 = 1;
        let mut p1: i128 = 1;
        let mut q1: i128 = 0;

        let mut xi = x;

        for _ in 0..50 {
            let a = xi.floor() as i128;
            let p2 = a * p1 + p0;
            let q2 = a * q1 + q0;

            if q2 > max_denom {
                break;
            }

            p0 = p1;
            q0 = q1;
            p1 = p2;
            q1 = q2;

            let frac = xi - a as f64;
            if frac.abs() < 1e-15 {
                break;
            }
            xi = 1.0 / frac;
        }

        Some(Rational::new(sign * p1, q1))
    }
}

// === Arithmetic Operations ===

impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Rational::new(
            self.num * other.denom + other.num * self.denom,
            self.denom * other.denom,
        )
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Rational::new(
            self.num * other.denom - other.num * self.denom,
            self.denom * other.denom,
        )
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Rational::new(self.num * other.num, self.denom * other.denom)
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if other.num == 0 {
            panic!("Rational: division by zero");
        }
        Rational::new(self.num * other.denom, self.denom * other.num)
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Rational {
            num: -self.num,
            denom: self.denom,
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare a/b with c/d: compare a*d with c*b
        let lhs = self.num * other.denom;
        let rhs = other.num * self.denom;
        lhs.cmp(&rhs)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.denom)
        }
    }
}

// === From implementations ===

impl From<i32> for Rational {
    fn from(n: i32) -> Self {
        Rational::from_int(n as i128)
    }
}

impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Rational::from_int(n as i128)
    }
}

impl From<i128> for Rational {
    fn from(n: i128) -> Self {
        Rational::from_int(n)
    }
}

// === PyO3 Methods ===

#[pymethods]
impl Rational {
    #[new]
    fn py_new(num: i64, denom: i64) -> Self {
        Rational::new(num as i128, denom as i128)
    }

    #[staticmethod]
    fn from_integer(n: i64) -> Self {
        Rational::from_int(n as i128)
    }

    #[getter]
    fn get_numerator(&self) -> i64 {
        self.numerator() as i64
    }

    #[getter]
    fn get_denominator(&self) -> i64 {
        self.denominator() as i64
    }

    fn __repr__(&self) -> String {
        format!("Rational({}, {})", self.num, self.denom)
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __add__(&self, other: &Rational) -> Self {
        *self + *other
    }

    fn __sub__(&self, other: &Rational) -> Self {
        *self - *other
    }

    fn __mul__(&self, other: &Rational) -> Self {
        *self * *other
    }

    fn __truediv__(&self, other: &Rational) -> Self {
        *self / *other
    }

    fn __neg__(&self) -> Self {
        -*self
    }

    fn __eq__(&self, other: &Rational) -> bool {
        self == other
    }

    fn __lt__(&self, other: &Rational) -> bool {
        self < other
    }

    fn __le__(&self, other: &Rational) -> bool {
        self <= other
    }

    fn __gt__(&self, other: &Rational) -> bool {
        self > other
    }

    fn __ge__(&self, other: &Rational) -> bool {
        self >= other
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Evaluate to float (for verification only)
    fn eval(&self) -> f64 {
        self.to_f64()
    }

    /// Integer power
    fn power(&self, exp: i32) -> Self {
        self.pow(exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);

        assert_eq!(a + b, Rational::new(5, 6));
        assert_eq!(a - b, Rational::new(1, 6));
        assert_eq!(a * b, Rational::new(1, 6));
        assert_eq!(a / b, Rational::new(3, 2));
    }

    #[test]
    fn test_negative() {
        let r = Rational::new(-3, 4);
        assert!(r.is_negative());
        assert_eq!(r.abs(), Rational::new(3, 4));
    }

    #[test]
    fn test_power() {
        let r = Rational::new(2, 3);
        assert_eq!(r.pow(2), Rational::new(4, 9));
        assert_eq!(r.pow(-1), Rational::new(3, 2));
        assert_eq!(r.pow(0), Rational::one());
    }

    #[test]
    fn test_reduction() {
        let r = Rational::new(4, 8);
        assert_eq!(r.numerator(), 1);
        assert_eq!(r.denominator(), 2);
    }
}
