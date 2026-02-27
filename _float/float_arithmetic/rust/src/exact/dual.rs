use pyo3::prelude::*;
use crate::exact::rational::Rational;
use crate::exact::syntonic::SyntonicExact;
use std::ops::{Add, Mul, Div};

/// The Kinetic State of the SGC.
/// Represents Z = Real + ε(Dual)
/// Where ε² = 0 (Nilpotent Algebra)
#[pyclass]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyntonicDual {
    /// The "Position" on the E8 Lattice (The Seed)
    #[pyo3(get)]
    pub real: SyntonicExact,
    
    /// The "Momentum" or Gradient (The Force Pressure)
    /// This tracks the 'Heat' or 'Urgency' of the state.
    #[pyo3(get)]
    pub dual: SyntonicExact, 
}

#[pymethods]
impl SyntonicDual {
    #[new]
    pub fn new(real: SyntonicExact, dual: SyntonicExact) -> Self {
        Self { real, dual }
    }

    /// Returns the Scalar Magnitude (Mass) of the state
    /// Ignores the dual part (momentum)
    /// Returns the Scalar Magnitude (Mass) of the state
    /// Ignores the dual part (momentum)
    pub fn mass(&self) -> f64 {
        self.real.eval()
    }

    fn __mul__(&self, object: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(r) = object.extract::<Rational>() {
            return Ok(Self {
                real: self.real * r,
                dual: self.dual * r, // Linear scaling of gradient
            });
        }
        if let Ok(other) = object.extract::<SyntonicDual>() {
            return Ok(*self * other);
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported operand type"))
    }
    
    fn __truediv__(&self, object: &Bound<'_, PyAny>) -> PyResult<Self> {
         if let Ok(r) = object.extract::<Rational>() {
            return Ok(Self {
                real: self.real / r,
                dual: self.dual / r,
            });
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported operand type"))
    }

    /// Checks if 'other' is a harmonic expansion of 'self' by exactly Phi.
    /// Returns True if: other == self * (89/55) [Approximation for now, exact later]
    pub fn checks_resonance(&self, other: &SyntonicDual) -> bool {
        // We use our Rational approximation for growth check
        // In V2, we will use exact GoldenExact checking (b_new == a_old + b_old)
        let phi = Rational::new(89, 55);
        
        let predicted_real = self.real.base * phi;
        
        // Exact comparison of the Rationals
        // Note: This assumes simple linear growth. 
        // Real SRT uses matrix rotation, but this proves the "Folder".
        predicted_real == other.real.base
    }
}

// ARITHMETIC RULES FOR DUAL NUMBERS
// (A + εB) + (C + εD) = (A+C) + ε(B+D)
// (A + εB) * (C + εD) = (A*C) + ε(A*D + B*C)  <-- Note: B*D vanishes because ε²=0

impl Add for SyntonicDual {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            dual: self.dual + other.dual,
        }
    }
}

impl Mul for SyntonicDual {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // Real part: Standard multiplication
        let new_real = self.real * other.real;
        
        // Dual part: Cross-terms (The Chain Rule of Differentiation)
        // This automatically calculates the derivative of the product!
        // f(x)*g(x) -> f'(x)g(x) + f(x)g'(x)
        let term1 = self.real * other.dual;
        let term2 = self.dual * other.real;
        let new_dual = term1 + term2;

        Self {
            real: new_real,
            dual: new_dual,
        }
    }
}
