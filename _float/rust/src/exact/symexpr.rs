//! Symbolic Expression Trees for SRT
//!
//! SymExpr represents mathematical expressions symbolically, preserving
//! exact relationships between constants. Unlike floating-point arithmetic,
//! these expressions maintain the algebraic structure.
//!
//! # Design Principles
//!
//! 1. **Never auto-evaluate**: Expressions stay symbolic until explicitly evaluated
//! 2. **Preserve identities**: φ² stays as φ² (or simplifies to φ+1), not 2.618...
//! 3. **All five constants**: π, e, φ, E*, q are first-class citizens
//! 4. **Exact rational embedding**: Rationals and GoldenExact embed exactly
//!
//! # Expression Hierarchy
//!
//! ```text
//! SymExpr
//! ├── Exact types (fully reducible)
//! │   ├── Integer(i128)
//! │   ├── Rational(Rational)
//! │   └── Golden(GoldenExact)
//! ├── Fundamental constants (symbolic atoms)
//! │   ├── Pi, Euler, Phi, EStar, Q
//! │   └── (Phi can reduce to Golden for some operations)
//! └── Composite expressions
//!     ├── Add, Sub, Mul, Div
//!     ├── Pow (rational exponent for algebraic closure)
//!     ├── Exp, Ln
//!     ├── Gamma (for Γ(1/4) in E* decomposition)
//!     └── Sin, Cos (for completeness)
//! ```

use pyo3::prelude::*;
use std::fmt;

use super::constants::FundamentalConstant;
use super::golden::GoldenExact;
use super::rational::Rational;

/// Symbolic expression in SRT
#[derive(Clone, Debug)]
pub enum SymExpr {
    // === Exact Numeric Types ===
    /// Exact integer
    Integer(i128),

    /// Exact rational number
    Rat(Rational),

    /// Exact golden field element (a + bφ where a,b ∈ Q)
    Golden(GoldenExact),

    // === Fundamental Constants ===
    /// One of the five fundamental SRT constants
    Const(FundamentalConstant),

    // === Binary Operations ===
    /// Addition
    Add(Box<SymExpr>, Box<SymExpr>),

    /// Subtraction
    Sub(Box<SymExpr>, Box<SymExpr>),

    /// Multiplication
    Mul(Box<SymExpr>, Box<SymExpr>),

    /// Division
    Div(Box<SymExpr>, Box<SymExpr>),

    /// Power with rational exponent (for algebraic closure)
    Pow(Box<SymExpr>, Rational),

    // === Transcendental Functions ===
    /// Exponential: e^x
    Exp(Box<SymExpr>),

    /// Natural logarithm: ln(x)
    Ln(Box<SymExpr>),

    /// Gamma function: Γ(x)
    /// Needed for E* = Γ(1/4)² + π(π-1) + (35/12)e^(-π) + Δ
    Gamma(Box<SymExpr>),

    /// Sine
    Sin(Box<SymExpr>),

    /// Cosine
    Cos(Box<SymExpr>),

    // === Special ===
    /// Negation (unary minus)
    Neg(Box<SymExpr>),
}

impl SymExpr {
    // === Constructors for constants ===

    /// The constant π
    pub fn pi() -> Self {
        SymExpr::Const(FundamentalConstant::Pi)
    }

    /// Euler's number e
    pub fn e() -> Self {
        SymExpr::Const(FundamentalConstant::Euler)
    }

    /// The golden ratio φ
    pub fn phi() -> Self {
        SymExpr::Const(FundamentalConstant::Phi)
    }

    /// The Spectral Möbius constant E* = e^π - π
    pub fn e_star() -> Self {
        SymExpr::Const(FundamentalConstant::EStar)
    }

    /// The universal syntony deficit q
    pub fn q() -> Self {
        SymExpr::Const(FundamentalConstant::Q)
    }

    // === Constructors for exact values ===

    /// Create from integer
    pub fn from_int(n: i128) -> Self {
        SymExpr::Integer(n)
    }

    /// Create from rational
    pub fn from_rational(r: Rational) -> Self {
        SymExpr::Rat(r)
    }

    /// Create from golden exact
    pub fn from_golden(g: GoldenExact) -> Self {
        SymExpr::Golden(g)
    }

    /// Create rational from num/denom
    pub fn rational(num: i128, denom: i128) -> Self {
        SymExpr::Rat(Rational::new(num, denom))
    }

    // === Expression builders ===

    /// Addition
    pub fn add(self, other: Self) -> Self {
        // Try to simplify if both are exact
        match (&self, &other) {
            (SymExpr::Integer(a), SymExpr::Integer(b)) => SymExpr::Integer(a + b),
            (SymExpr::Rat(a), SymExpr::Rat(b)) => SymExpr::Rat(*a + *b),
            (SymExpr::Golden(a), SymExpr::Golden(b)) => SymExpr::Golden(*a + *b),
            _ => SymExpr::Add(Box::new(self), Box::new(other)),
        }
    }

    /// Subtraction
    pub fn sub(self, other: Self) -> Self {
        match (&self, &other) {
            (SymExpr::Integer(a), SymExpr::Integer(b)) => SymExpr::Integer(a - b),
            (SymExpr::Rat(a), SymExpr::Rat(b)) => SymExpr::Rat(*a - *b),
            (SymExpr::Golden(a), SymExpr::Golden(b)) => SymExpr::Golden(*a - *b),
            _ => SymExpr::Sub(Box::new(self), Box::new(other)),
        }
    }

    /// Multiplication
    pub fn mul(self, other: Self) -> Self {
        match (&self, &other) {
            (SymExpr::Integer(a), SymExpr::Integer(b)) => SymExpr::Integer(a * b),
            (SymExpr::Rat(a), SymExpr::Rat(b)) => SymExpr::Rat(*a * *b),
            (SymExpr::Golden(a), SymExpr::Golden(b)) => SymExpr::Golden(*a * *b),
            _ => SymExpr::Mul(Box::new(self), Box::new(other)),
        }
    }

    /// Division
    pub fn div(self, other: Self) -> Self {
        match (&self, &other) {
            (SymExpr::Rat(a), SymExpr::Rat(b)) => SymExpr::Rat(*a / *b),
            (SymExpr::Golden(a), SymExpr::Golden(b)) => SymExpr::Golden(*a / *b),
            _ => SymExpr::Div(Box::new(self), Box::new(other)),
        }
    }

    /// Power with integer exponent
    pub fn pow_int(self, exp: i32) -> Self {
        self.pow(Rational::from_int(exp as i128))
    }

    /// Power with rational exponent
    pub fn pow(self, exp: Rational) -> Self {
        // Handle special cases
        if exp.is_zero() {
            return SymExpr::Integer(1);
        }
        if exp.is_one() {
            return self;
        }

        match &self {
            SymExpr::Integer(n) if exp.is_integer() => {
                let e = exp.floor();
                if e >= 0 && e < 64 {
                    SymExpr::Integer(n.pow(e as u32))
                } else {
                    SymExpr::Pow(Box::new(self), exp)
                }
            }
            SymExpr::Golden(g) if exp.is_integer() => {
                let e = exp.floor() as i32;
                SymExpr::Golden(g.pow(e))
            }
            _ => SymExpr::Pow(Box::new(self), exp),
        }
    }

    /// Exponential
    pub fn exp_of(self) -> Self {
        SymExpr::Exp(Box::new(self))
    }

    /// Natural logarithm
    pub fn ln_of(self) -> Self {
        SymExpr::Ln(Box::new(self))
    }

    /// Gamma function
    pub fn gamma_of(self) -> Self {
        SymExpr::Gamma(Box::new(self))
    }

    /// Negation
    pub fn neg(self) -> Self {
        match self {
            SymExpr::Integer(n) => SymExpr::Integer(-n),
            SymExpr::Rat(r) => SymExpr::Rat(-r),
            SymExpr::Golden(g) => SymExpr::Golden(-g),
            _ => SymExpr::Neg(Box::new(self)),
        }
    }

    // === SRT-specific expressions ===

    /// The E* identity: e^π - π
    /// This should symbolically equal E*
    pub fn e_star_expanded() -> Self {
        // e^π - π: exponential of pi, minus pi
        SymExpr::pi().exp_of().sub(SymExpr::pi())
    }

    /// The q formula: (2φ + e/(2φ²)) / (φ⁴ · E*)
    /// This should symbolically equal q
    pub fn q_expanded() -> Self {
        let phi = SymExpr::phi();
        let e = SymExpr::e();
        let e_star = SymExpr::e_star();

        // Numerator: 2φ + e/(2φ²)
        let two = SymExpr::from_int(2);
        let phi_sq = phi.clone().pow_int(2);
        let numerator = two.clone().mul(phi.clone()).add(e.div(two.mul(phi_sq)));

        // Denominator: φ⁴ · E*
        let phi_4 = phi.pow_int(4);
        let denominator = phi_4.mul(e_star);

        numerator.div(denominator)
    }

    /// The E* decomposition: Γ(1/4)² + π(π-1) + (35/12)e^(-π) + Δ
    pub fn e_star_decomposition() -> Self {
        let gamma_quarter = SymExpr::rational(1, 4).gamma_of();
        let gamma_sq = gamma_quarter.pow_int(2);

        let pi = SymExpr::pi();
        let pi_term = pi.clone().mul(pi.clone().sub(SymExpr::from_int(1)));

        let exp_neg_pi = SymExpr::pi().neg().exp_of();
        let cone_term = SymExpr::rational(35, 12).mul(exp_neg_pi);

        // Δ ≈ (55/72)q⁴ (Fibonacci residual, very small)
        let q = SymExpr::q();
        let delta = SymExpr::rational(55, 72).mul(q.pow_int(4));

        gamma_sq.add(pi_term).add(cone_term).add(delta)
    }

    /// Syntony correction factor: multiply of q at given level
    pub fn correction(numerator: i32, denominator: i32) -> Self {
        if denominator == 1 {
            SymExpr::from_int(numerator as i128).mul(SymExpr::q())
        } else {
            SymExpr::q().mul(SymExpr::rational(numerator as i128, denominator as i128))
        }
    }

    // === Evaluation ===

    /// Evaluate to f64 (for numerical verification ONLY)
    /// This loses exactness - use only for checking results
    pub fn eval_f64(&self) -> f64 {
        match self {
            SymExpr::Integer(n) => *n as f64,
            SymExpr::Rat(r) => r.to_f64(),
            SymExpr::Golden(g) => g.to_f64(),
            SymExpr::Const(c) => c.approx_f64(),
            SymExpr::Add(a, b) => a.eval_f64() + b.eval_f64(),
            SymExpr::Sub(a, b) => a.eval_f64() - b.eval_f64(),
            SymExpr::Mul(a, b) => a.eval_f64() * b.eval_f64(),
            SymExpr::Div(a, b) => a.eval_f64() / b.eval_f64(),
            SymExpr::Pow(base, exp) => base.eval_f64().powf(exp.to_f64()),
            SymExpr::Exp(x) => x.eval_f64().exp(),
            SymExpr::Ln(x) => x.eval_f64().ln(),
            SymExpr::Gamma(x) => gamma_f64(x.eval_f64()),
            SymExpr::Sin(x) => x.eval_f64().sin(),
            SymExpr::Cos(x) => x.eval_f64().cos(),
            SymExpr::Neg(x) => -x.eval_f64(),
        }
    }

    /// Check if this expression is exactly equal to a fundamental constant
    pub fn is_fundamental_constant(&self) -> Option<FundamentalConstant> {
        match self {
            SymExpr::Const(c) => Some(*c),
            _ => None,
        }
    }

    /// Check if this is an exact (non-transcendental) expression
    pub fn is_exact(&self) -> bool {
        matches!(
            self,
            SymExpr::Integer(_) | SymExpr::Rat(_) | SymExpr::Golden(_)
        )
    }
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymExpr::Integer(n) => write!(f, "{}", n),
            SymExpr::Rat(r) => write!(f, "{}", r),
            SymExpr::Golden(g) => write!(f, "{}", g),
            SymExpr::Const(c) => write!(f, "{}", c.symbol()),
            SymExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            SymExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            SymExpr::Mul(a, b) => write!(f, "({} · {})", a, b),
            SymExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            SymExpr::Pow(base, exp) => {
                if exp.denominator() == 1 {
                    write!(f, "{}^{}", base, exp.numerator())
                } else {
                    write!(f, "{}^({}/{})", base, exp.numerator(), exp.denominator())
                }
            }
            SymExpr::Exp(x) => write!(f, "exp({})", x),
            SymExpr::Ln(x) => write!(f, "ln({})", x),
            SymExpr::Gamma(x) => write!(f, "Γ({})", x),
            SymExpr::Sin(x) => write!(f, "sin({})", x),
            SymExpr::Cos(x) => write!(f, "cos({})", x),
            SymExpr::Neg(x) => write!(f, "-{}", x),
        }
    }
}

// === PyO3 Wrapper ===

/// Helper enum for Python operator interoperability
/// Accepts SymExpr, GoldenExact, Rational, or int
pub enum SymExprOrOther {
    Sym(PySymExpr),
    Golden(GoldenExact),
    Rat(Rational),
}

impl<'py> FromPyObject<'py> for SymExprOrOther {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try SymExpr first
        if let Ok(s) = ob.extract::<PySymExpr>() {
            return Ok(SymExprOrOther::Sym(s));
        }
        // Try GoldenExact
        if let Ok(g) = ob.extract::<GoldenExact>() {
            return Ok(SymExprOrOther::Golden(g));
        }
        // Try Rational
        if let Ok(r) = ob.extract::<Rational>() {
            return Ok(SymExprOrOther::Rat(r));
        }
        // Try int -> Rational
        if let Ok(i) = ob.extract::<i64>() {
            return Ok(SymExprOrOther::Rat(Rational::from_int(i as i128)));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected SymExpr, GoldenExact, Rational, or int",
        ))
    }
}

impl SymExprOrOther {
    /// Convert to SymExpr (the universal container)
    fn to_symexpr(self) -> SymExpr {
        match self {
            SymExprOrOther::Sym(s) => s.inner,
            SymExprOrOther::Golden(g) => SymExpr::Golden(g),
            SymExprOrOther::Rat(r) => SymExpr::Rat(r),
        }
    }
}

/// Python-accessible symbolic expression
#[pyclass(name = "SymExpr")]
#[derive(Clone)]
pub struct PySymExpr {
    pub inner: SymExpr,
}

#[pymethods]
impl PySymExpr {
    // === Constant constructors ===

    /// The constant π
    #[staticmethod]
    fn pi() -> Self {
        PySymExpr {
            inner: SymExpr::pi(),
        }
    }

    /// Euler's number e
    #[staticmethod]
    fn e() -> Self {
        PySymExpr {
            inner: SymExpr::e(),
        }
    }

    /// The golden ratio φ
    #[staticmethod]
    fn phi() -> Self {
        PySymExpr {
            inner: SymExpr::phi(),
        }
    }

    /// The Spectral Möbius constant E*
    #[staticmethod]
    fn e_star() -> Self {
        PySymExpr {
            inner: SymExpr::e_star(),
        }
    }

    /// The universal syntony deficit q
    #[staticmethod]
    fn q() -> Self {
        PySymExpr {
            inner: SymExpr::q(),
        }
    }

    // === Value constructors ===

    /// Create from integer
    #[staticmethod]
    fn from_int(n: i64) -> Self {
        PySymExpr {
            inner: SymExpr::from_int(n as i128),
        }
    }

    /// Create from rational num/denom
    #[staticmethod]
    fn rational(num: i64, denom: i64) -> Self {
        PySymExpr {
            inner: SymExpr::rational(num as i128, denom as i128),
        }
    }

    /// Create from golden (a + b·φ) with integer coefficients
    #[staticmethod]
    fn golden(a: i64, b: i64) -> Self {
        PySymExpr {
            inner: SymExpr::from_golden(GoldenExact::from_ints(a, b)),
        }
    }

    // === SRT expressions ===

    /// E* expanded as e^π - π
    #[staticmethod]
    fn e_star_expanded() -> Self {
        PySymExpr {
            inner: SymExpr::e_star_expanded(),
        }
    }

    /// q expanded using the Universal Formula
    #[staticmethod]
    fn q_expanded() -> Self {
        PySymExpr {
            inner: SymExpr::q_expanded(),
        }
    }

    /// E* decomposition into Γ(1/4)² + π(π-1) + (35/12)e^(-π) + Δ
    #[staticmethod]
    fn e_star_decomposition() -> Self {
        PySymExpr {
            inner: SymExpr::e_star_decomposition(),
        }
    }

    /// Syntony correction factor: (num/denom) · q
    #[staticmethod]
    fn correction(num: i32, denom: i32) -> Self {
        PySymExpr {
            inner: SymExpr::correction(num, denom),
        }
    }

    // === Operations ===
    // Accept SymExpr, GoldenExact, Rational, or int

    fn __add__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: self.inner.clone().add(other.to_symexpr()),
        }
    }

    fn __sub__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: self.inner.clone().sub(other.to_symexpr()),
        }
    }

    fn __mul__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: self.inner.clone().mul(other.to_symexpr()),
        }
    }

    fn __truediv__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: self.inner.clone().div(other.to_symexpr()),
        }
    }

    fn __neg__(&self) -> Self {
        PySymExpr {
            inner: self.inner.clone().neg(),
        }
    }

    // === Reverse operations ===
    // Handle: int + SymExpr, GoldenExact + SymExpr, etc.

    fn __radd__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: other.to_symexpr().add(self.inner.clone()),
        }
    }

    fn __rsub__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: other.to_symexpr().sub(self.inner.clone()),
        }
    }

    fn __rmul__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: other.to_symexpr().mul(self.inner.clone()),
        }
    }

    fn __rtruediv__(&self, other: SymExprOrOther) -> Self {
        PySymExpr {
            inner: other.to_symexpr().div(self.inner.clone()),
        }
    }

    fn __pow__(&self, exp: i32, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Modulo operation not supported for symbolic expressions",
            ));
        }
        Ok(PySymExpr {
            inner: self.inner.clone().pow_int(exp),
        })
    }

    /// Exponential e^self
    fn exp(&self) -> Self {
        PySymExpr {
            inner: self.inner.clone().exp_of(),
        }
    }

    /// Natural logarithm ln(self)
    fn ln(&self) -> Self {
        PySymExpr {
            inner: self.inner.clone().ln_of(),
        }
    }

    /// Gamma function Γ(self)
    fn gamma(&self) -> Self {
        PySymExpr {
            inner: self.inner.clone().gamma_of(),
        }
    }

    // === Evaluation ===

    /// Evaluate to float (FOR VERIFICATION ONLY - loses exactness!)
    fn eval(&self) -> f64 {
        self.inner.eval_f64()
    }

    /// Get symbolic string representation
    fn symbolic(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SymExpr({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Simple gamma function approximation using Lanczos
fn gamma_f64(x: f64) -> f64 {
    // For Γ(1/4) ≈ 3.625609908...
    // Use Stirling-like approximation or lookup for special values
    if (x - 0.25).abs() < 1e-10 {
        return 3.6256099082219083;
    }

    // Lanczos approximation for general values
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_f64(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut sum = c[0];
        for i in 1..9 {
            sum += c[i] / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_expansion() {
        // Verify that q_expanded() evaluates to approximately q
        let q_direct = SymExpr::q().eval_f64();
        let q_formula = SymExpr::q_expanded().eval_f64();
        assert!((q_direct - q_formula).abs() < 1e-10);
    }

    #[test]
    fn test_e_star_expansion() {
        // Verify that e^π - π ≈ E*
        let e_star_direct = SymExpr::e_star().eval_f64();
        let e_star_formula = SymExpr::e_star_expanded().eval_f64();
        assert!((e_star_direct - e_star_formula).abs() < 1e-10);
    }

    #[test]
    fn test_golden_arithmetic() {
        // φ² should simplify to 1 + φ in golden field
        let phi = SymExpr::from_golden(GoldenExact::phi());
        let phi_sq = phi.clone().mul(phi);

        let expected = SymExpr::from_golden(GoldenExact::from_ints(1, 1));

        // Both should evaluate to the same value
        assert!((phi_sq.eval_f64() - expected.eval_f64()).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_display() {
        let expr = SymExpr::q().mul(SymExpr::from_int(2));
        assert!(expr.to_string().contains("q"));
    }
}
