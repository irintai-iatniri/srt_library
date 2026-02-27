//! Linear algebra module for Syntonic tensor operations.
//!
//! This module provides matrix multiplication and related operations,
//! with all constants derived from the exact symbolic infrastructure.
//!
//! # Design Principle
//!
//! Every numerical value traces back to the five fundamental SRT constants:
//! - φ (phi) from `GoldenExact::phi()`
//! - φ⁻¹ (phi inverse) from `GoldenExact::phi_hat()`
//! - q (syntony deficit) from `FundamentalConstant::Q`
//! - π from `FundamentalConstant::Pi`
//! - E* from `FundamentalConstant::EStar`
//!
//! No hardcoded floating-point constants are used.

pub mod matmul;

use crate::exact::constants::Structure;
use crate::tensor::storage::TensorStorage;
use pyo3::prelude::*;

// =============================================================================
// PyO3 Wrapper Functions for Python Access
// =============================================================================

/// Core matrix multiplication: C = A × B
#[pyfunction]
#[pyo3(name = "linalg_mm")]
pub fn py_mm(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm(a, b).map_err(|e| e.into())
}

/// GEMM: C = α × (A × B) + β × C
#[pyfunction]
#[pyo3(name = "linalg_mm_add")]
pub fn py_mm_add(
    a: &TensorStorage,
    b: &TensorStorage,
    c: &TensorStorage,
    alpha: f64,
    beta: f64,
) -> PyResult<TensorStorage> {
    matmul::mm_add(a, b, c, alpha, beta).map_err(|e| e.into())
}

/// Transposed-None matmul: C = Aᵀ × B
#[pyfunction]
#[pyo3(name = "linalg_mm_tn")]
pub fn py_mm_tn(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_tn(a, b).map_err(|e| e.into())
}

/// None-Transposed matmul: C = A × Bᵀ
#[pyfunction]
#[pyo3(name = "linalg_mm_nt")]
pub fn py_mm_nt(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_nt(a, b).map_err(|e| e.into())
}

/// Transposed-Transposed matmul: C = Aᵀ × Bᵀ
#[pyfunction]
#[pyo3(name = "linalg_mm_tt")]
pub fn py_mm_tt(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_tt(a, b).map_err(|e| e.into())
}

/// Hermitian-None matmul: C = A† × B
#[pyfunction]
#[pyo3(name = "linalg_mm_hn")]
pub fn py_mm_hn(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_hn(a, b).map_err(|e| e.into())
}

/// None-Hermitian matmul: C = A × B†
#[pyfunction]
#[pyo3(name = "linalg_mm_nh")]
pub fn py_mm_nh(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_nh(a, b).map_err(|e| e.into())
}

/// Batched matrix multiplication: C[i] = A[i] × B[i]
#[pyfunction]
#[pyo3(name = "linalg_bmm")]
pub fn py_bmm(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::bmm(a, b).map_err(|e| e.into())
}

/// φ-scaled matmul: φⁿ × (A × B)
///
/// Uses exact Fibonacci formula for φⁿ via GoldenExact.
#[pyfunction]
#[pyo3(name = "linalg_mm_phi")]
pub fn py_mm_phi(a: &TensorStorage, b: &TensorStorage, n: i32) -> PyResult<TensorStorage> {
    matmul::mm_phi(a, b, n).map_err(|e| e.into())
}

/// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
///
/// The fundamental bracket for SRT φ-Lie algebra representations.
#[pyfunction]
#[pyo3(name = "linalg_phi_bracket")]
pub fn py_phi_bracket(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::phi_bracket(a, b).map_err(|e| e.into())
}

/// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
#[pyfunction]
#[pyo3(name = "linalg_phi_antibracket")]
pub fn py_phi_antibracket(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::phi_antibracket(a, b).map_err(|e| e.into())
}

/// Correction factor matmul: (1 ± q/N) × (A × B)
///
/// Uses Structure for dimension N. Sign: 1 for +, -1 for -.
#[pyfunction]
#[pyo3(name = "linalg_mm_corrected")]
pub fn py_mm_corrected(
    a: &TensorStorage,
    b: &TensorStorage,
    structure: Structure,
    sign: i8,
) -> PyResult<TensorStorage> {
    matmul::mm_corrected(a, b, structure, sign).map_err(|e| e.into())
}

/// Complex phase matmul: e^{iπn/φ} × (A × B)
///
/// Applies a golden-ratio-modulated phase rotation.
#[pyfunction]
#[pyo3(name = "linalg_mm_golden_phase")]
pub fn py_mm_golden_phase(a: &TensorStorage, b: &TensorStorage, n: i32) -> PyResult<TensorStorage> {
    matmul::mm_golden_phase(a, b, n).map_err(|e| e.into())
}

/// Golden-weighted matmul: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(−k²/φ)
///
/// Each summation index k is weighted by a golden Gaussian.
#[pyfunction]
#[pyo3(name = "linalg_mm_golden_weighted")]
pub fn py_mm_golden_weighted(a: &TensorStorage, b: &TensorStorage) -> PyResult<TensorStorage> {
    matmul::mm_golden_weighted(a, b).map_err(|e| e.into())
}

/// Projection sum: Ψ + Σₖ αₖ × (Pₖ × Ψ)
///
/// Used for DHSR projection summation over lattice points.
/// Takes a list of projector TensorStorages and a list of coefficients.
#[pyfunction]
#[pyo3(name = "linalg_projection_sum")]
pub fn py_projection_sum(
    psi: &TensorStorage,
    projectors: &Bound<'_, pyo3::types::PyList>,
    coefficients: Vec<f64>,
) -> PyResult<TensorStorage> {
    // Extract TensorStorages from Python list using PyRef pattern
    // then clone internally to build owned Vec
    let n = projectors.len();
    let mut proj_vec: Vec<TensorStorage> = Vec::with_capacity(n);

    for i in 0..n {
        let item = projectors.get_item(i)?;
        let tensor_ref: pyo3::PyRef<'_, TensorStorage> = item.extract()?;
        proj_vec.push(tensor_ref.clone_storage_internal());
    }

    matmul::projection_sum(psi, &proj_vec, &coefficients).map_err(|e| e.into())
}

/// Generalized matrix multiply: C = α × op(A) × op(B) + β × C
///
/// BLAS-style GEMM with transpose specification using strings:
/// - "N" - use matrix as-is
/// - "T" - transpose the matrix
/// - "H" - conjugate transpose (Hermitian, for complex)
///
/// If c is None, computes C = α × op(A) × op(B).
#[pyfunction]
#[pyo3(name = "linalg_mm_gemm")]
pub fn py_mm_gemm(
    a: &TensorStorage,
    b: &TensorStorage,
    trans_a: &str,
    trans_b: &str,
    alpha: f64,
    beta: f64,
    c: Option<&TensorStorage>,
) -> PyResult<TensorStorage> {
    let ta = match trans_a.to_uppercase().as_str() {
        "N" | "NONE" => matmul::Transpose::None,
        "T" | "TRANS" => matmul::Transpose::Trans,
        "H" | "CONJTRANS" | "HERMITIAN" => matmul::Transpose::ConjTrans,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid transpose '{}': use 'N', 'T', or 'H'",
                trans_a
            )))
        }
    };
    let tb = match trans_b.to_uppercase().as_str() {
        "N" | "NONE" => matmul::Transpose::None,
        "T" | "TRANS" => matmul::Transpose::Trans,
        "H" | "CONJTRANS" | "HERMITIAN" => matmul::Transpose::ConjTrans,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid transpose '{}': use 'N', 'T', or 'H'",
                trans_b
            )))
        }
    };
    matmul::mm_gemm(a, b, ta, tb, alpha, beta, c).map_err(|e| e.into())
}

/// Direct q-deficit correction: (1 ± q/N) × (A × B)
///
/// Unlike `mm_corrected` which uses Structure enum, this allows
/// specifying dimension N directly.
///
/// Standard dimensions:
/// - E₈: 248 (adjoint), 240 (roots), 120 (positive), 8 (rank)
/// - E₆: 78 (adjoint), 36 (golden cone), 27 (fundamental)
/// - D₄: 24 (kissing number)
/// - G₂: 14 (adjoint)
#[pyfunction]
#[pyo3(name = "linalg_mm_q_corrected_direct")]
pub fn py_mm_q_corrected_direct(
    a: &TensorStorage,
    b: &TensorStorage,
    n: u32,
    sign: i8,
) -> PyResult<TensorStorage> {
    matmul::mm_q_corrected_direct(a, b, n, sign).map_err(|e| e.into())
}

/// Direct q-deficit scalar: (1 ± q/N)
///
/// Returns the raw correction factor without matrix multiplication.
/// Useful for applying q-corrections to individual values.
#[pyfunction]
#[pyo3(name = "linalg_q_correction_scalar")]
pub fn py_q_correction_scalar(n: u32, sign: i8) -> f64 {
    matmul::q_correction_scalar(n, sign)
}
