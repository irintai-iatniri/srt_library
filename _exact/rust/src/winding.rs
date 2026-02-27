//! Winding State - Winding numbers on the T^4 torus.
//!
//! A winding state |n> = |n_7, n_8, n_9, n_10> represents a configuration
//! of winding numbers on the internal 4-torus. These are the fundamental
//! quantum numbers in SRT from which all charges derive.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::tensor::srt_kernels::PHI;

/// A 4D winding state on T^4 torus.
///
/// Represents a configuration of integer winding numbers on the
/// four internal circles of the torus. These quantum numbers
/// determine charge, mass generation, and golden recursion properties.
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct WindingState {
    #[pyo3(get)]
    pub n7: i64,
    #[pyo3(get)]
    pub n8: i64,
    #[pyo3(get)]
    pub n9: i64,
    #[pyo3(get)]
    pub n10: i64,
}

impl PartialEq for WindingState {
    fn eq(&self, other: &Self) -> bool {
        self.n7 == other.n7 && self.n8 == other.n8 && self.n9 == other.n9 && self.n10 == other.n10
    }
}

impl Eq for WindingState {}

impl Hash for WindingState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n7.hash(state);
        self.n8.hash(state);
        self.n9.hash(state);
        self.n10.hash(state);
    }
}

#[pymethods]
impl WindingState {
    /// Create a new winding state.
    #[new]
    #[pyo3(signature = (n7, n8=0, n9=0, n10=0))]
    pub fn new(n7: i64, n8: i64, n9: i64, n10: i64) -> Self {
        WindingState { n7, n8, n9, n10 }
    }

    /// Tuple of all winding numbers (n_7, n_8, n_9, n_10).
    #[getter]
    pub fn n(&self) -> (i64, i64, i64, i64) {
        (self.n7, self.n8, self.n9, self.n10)
    }

    /// Squared norm |n|^2 = n_7^2 + n_8^2 + n_9^2 + n_10^2.
    #[getter]
    pub fn norm_squared(&self) -> i64 {
        self.n7 * self.n7 + self.n8 * self.n8 + self.n9 * self.n9 + self.n10 * self.n10
    }

    /// Euclidean norm |n|.
    #[getter]
    pub fn norm(&self) -> f64 {
        (self.norm_squared() as f64).sqrt()
    }

    /// Maximum absolute value of winding components.
    #[getter]
    pub fn max_component(&self) -> i64 {
        self.n7
            .abs()
            .max(self.n8.abs())
            .max(self.n9.abs())
            .max(self.n10.abs())
    }

    /// Recursion generation depth k.
    ///
    /// The generation determines the mass hierarchy: m ~ e^(-phi^k).
    /// For winding n, generation k = 1 + floor(log_phi(max|n_i|)) for |n| > 0,
    /// and k = 0 for the vacuum |n| = 0.
    #[getter]
    pub fn generation(&self) -> i64 {
        let max_n = self.max_component();
        if max_n == 0 {
            return 0;
        }
        // k = 1 + floor(log_phi(max|n_i|))
        1 + ((max_n as f64).ln() / PHI.ln()) as i64
    }

    /// Golden measure weight w(n) = exp(-|n|^2 / phi).
    #[pyo3(signature = (phi=None))]
    pub fn golden_weight(&self, phi: Option<f64>) -> f64 {
        let phi_val = phi.unwrap_or(PHI);
        (-(self.norm_squared() as f64) / phi_val).exp()
    }

    /// Inner product n . m = sum_i n_i * m_i.
    pub fn inner_product(&self, other: &WindingState) -> i64 {
        self.n7 * other.n7 + self.n8 * other.n8 + self.n9 * other.n9 + self.n10 * other.n10
    }

    /// Check if this is the vacuum state |0,0,0,0>.
    pub fn is_zero(&self) -> bool {
        self.n7 == 0 && self.n8 == 0 && self.n9 == 0 && self.n10 == 0
    }

    /// True if this is the vacuum state |0,0,0,0>.
    #[getter]
    pub fn is_vacuum(&self) -> bool {
        self.is_zero()
    }

    /// Convert to tuple.
    pub fn to_tuple(&self) -> (i64, i64, i64, i64) {
        (self.n7, self.n8, self.n9, self.n10)
    }

    // === Constructors ===

    /// Create the vacuum state |0,0,0,0>.
    #[staticmethod]
    pub fn zero() -> Self {
        WindingState::new(0, 0, 0, 0)
    }

    /// Create unit winding state along given axis (0-3 for n_7, n_8, n_9, n_10).
    #[staticmethod]
    pub fn unit(index: usize) -> PyResult<Self> {
        match index {
            0 => Ok(WindingState::new(1, 0, 0, 0)),
            1 => Ok(WindingState::new(0, 1, 0, 0)),
            2 => Ok(WindingState::new(0, 0, 1, 0)),
            3 => Ok(WindingState::new(0, 0, 0, 1)),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Index must be 0-3",
            )),
        }
    }

    /// Create from tuple.
    #[staticmethod]
    pub fn from_tuple(t: (i64, i64, i64, i64)) -> Self {
        WindingState::new(t.0, t.1, t.2, t.3)
    }

    // === Arithmetic ===

    fn __add__(&self, other: &WindingState) -> Self {
        WindingState::new(
            self.n7 + other.n7,
            self.n8 + other.n8,
            self.n9 + other.n9,
            self.n10 + other.n10,
        )
    }

    fn __sub__(&self, other: &WindingState) -> Self {
        WindingState::new(
            self.n7 - other.n7,
            self.n8 - other.n8,
            self.n9 - other.n9,
            self.n10 - other.n10,
        )
    }

    fn __neg__(&self) -> Self {
        WindingState::new(-self.n7, -self.n8, -self.n9, -self.n10)
    }

    fn __mul__(&self, scalar: i64) -> Self {
        WindingState::new(
            self.n7 * scalar,
            self.n8 * scalar,
            self.n9 * scalar,
            self.n10 * scalar,
        )
    }

    fn __rmul__(&self, scalar: i64) -> Self {
        self.__mul__(scalar)
    }

    // === Comparison ===

    fn __eq__(&self, other: &WindingState) -> bool {
        self == other
    }

    fn __ne__(&self, other: &WindingState) -> bool {
        self != other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    // === Display ===

    fn __repr__(&self) -> String {
        format!(
            "WindingState(n7={}, n8={}, n9={}, n10={})",
            self.n7, self.n8, self.n9, self.n10
        )
    }

    fn __str__(&self) -> String {
        format!("|{},{},{},{}>", self.n7, self.n8, self.n9, self.n10)
    }

    // === Sequence Protocol ===

    fn __len__(&self) -> usize {
        4
    }

    fn __getitem__(&self, index: i32) -> PyResult<i64> {
        match index {
            0 | -4 => Ok(self.n7),
            1 | -3 => Ok(self.n8),
            2 | -2 => Ok(self.n9),
            3 | -1 => Ok(self.n10),
            _ => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Index out of range",
            )),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<WindingStateIterator> {
        Ok(WindingStateIterator {
            state: *slf,
            index: 0,
        })
    }
}

/// Iterator for WindingState components.
#[pyclass]
pub struct WindingStateIterator {
    state: WindingState,
    index: usize,
}

#[pymethods]
impl WindingStateIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<i64> {
        let result = match slf.index {
            0 => Some(slf.state.n7),
            1 => Some(slf.state.n8),
            2 => Some(slf.state.n9),
            3 => Some(slf.state.n10),
            _ => None,
        };
        slf.index += 1;
        result
    }
}

// =============================================================================
// Enumeration Functions
// =============================================================================

/// Enumerate all winding states with |n| <= max_norm.
///
/// Returns a list of WindingState instances with norm <= max_norm.
#[pyfunction]
#[pyo3(signature = (max_norm=10))]
pub fn enumerate_windings(max_norm: i64) -> Vec<WindingState> {
    let max_sq = max_norm * max_norm;
    let mut result = Vec::new();

    for n7 in -max_norm..=max_norm {
        for n8 in -max_norm..=max_norm {
            for n9 in -max_norm..=max_norm {
                for n10 in -max_norm..=max_norm {
                    let norm_sq = n7 * n7 + n8 * n8 + n9 * n9 + n10 * n10;
                    if norm_sq <= max_sq {
                        result.push(WindingState { n7, n8, n9, n10 });
                    }
                }
            }
        }
    }

    result
}

/// Enumerate all winding states with |n|^2 <= max_norm_sq, grouped by norm squared.
///
/// Returns a dictionary mapping norm_squared values to lists of WindingState.
#[pyfunction]
pub fn enumerate_windings_by_norm(max_norm_sq: i64) -> HashMap<i64, Vec<WindingState>> {
    let mut result: HashMap<i64, Vec<WindingState>> = HashMap::new();
    let max_single = ((max_norm_sq as f64).sqrt().ceil()) as i64;

    for n7 in -max_single..=max_single {
        for n8 in -max_single..=max_single {
            for n9 in -max_single..=max_single {
                for n10 in -max_single..=max_single {
                    let norm_sq = n7 * n7 + n8 * n8 + n9 * n9 + n10 * n10;
                    if norm_sq <= max_norm_sq {
                        result
                            .entry(norm_sq)
                            .or_default()
                            .push(WindingState { n7, n8, n9, n10 });
                    }
                }
            }
        }
    }

    result
}

/// Enumerate winding states with exact |n|^2 = target_norm_sq.
#[pyfunction]
pub fn enumerate_windings_exact_norm(target_norm_sq: i64) -> Vec<WindingState> {
    let mut result = Vec::new();
    let max_single = ((target_norm_sq as f64).sqrt().ceil()) as i64;

    for n7 in -max_single..=max_single {
        for n8 in -max_single..=max_single {
            for n9 in -max_single..=max_single {
                for n10 in -max_single..=max_single {
                    let norm_sq = n7 * n7 + n8 * n8 + n9 * n9 + n10 * n10;
                    if norm_sq == target_norm_sq {
                        result.push(WindingState { n7, n8, n9, n10 });
                    }
                }
            }
        }
    }

    result
}

/// Count winding states with |n| <= max_norm.
#[pyfunction]
#[pyo3(signature = (max_norm=10))]
pub fn count_windings(max_norm: i64) -> usize {
    let max_sq = max_norm * max_norm;
    let mut count = 0usize;

    for n7 in -max_norm..=max_norm {
        for n8 in -max_norm..=max_norm {
            for n9 in -max_norm..=max_norm {
                for n10 in -max_norm..=max_norm {
                    let norm_sq = n7 * n7 + n8 * n8 + n9 * n9 + n10 * n10;
                    if norm_sq <= max_sq {
                        count += 1;
                    }
                }
            }
        }
    }

    count
}
