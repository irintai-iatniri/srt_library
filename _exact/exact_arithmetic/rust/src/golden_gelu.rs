//! GoldenGELU activation module
//!
//! Provides Rust interface to CUDA kernels for GoldenGELU activation
//!
//! GoldenGELU: x * sigmoid(phi * x)
//!
//! Represents winding probability of a token passing through W⁴ aperture
//! based on its energy state x.
//!
//! Mathematical Formulation:
//!   GeLUφ(x) = x * σ(φ * x)
//!
//! Where:
//!   - φ = 1.6180339887 (golden ratio)
//!   - σ(z) = 1 / (1 + e^(-z)) is sigmoid function
//!   - x is input tensor

use crate::tensor::srt_kernels::PHI;
use pyo3::prelude::*;

// =============================================================================
// PYTHON WRAPPERS
// =============================================================================

/// GoldenGELU forward pass: x * sigmoid(phi * x)
///
/// Represents winding probability of a token passing through W⁴ aperture
/// based on its energy state x.
///
/// Mathematical Formulation:
///   GeLUφ(x) = x * σ(φ * x)
///
/// Where:
///   - φ = 1.6180339887 (golden ratio)
///   - σ(z) = 1 / (1 + e^(-z)) is sigmoid function
///   - x is input tensor
///
/// Args:
///   values: List of float64 values
///
/// Returns:
///   GoldenGELU-activated values
#[pyfunction]
#[pyo3(name = "golden_gelu_forward")]
pub fn golden_gelu_forward(values: Vec<f64>) -> PyResult<Vec<f64>> {
    // CPU-only implementation for now
    // GPU activation pending cudarc API compatibility fixes
    let mut outputs = Vec::with_capacity(values.len());
    for &x in values.iter() {
        // Scale by phi: phi * x
        let scaled = PHI * x;

        // Compute sigmoid: 1 / (1 + exp(-scaled))
        let exp_neg_scaled = (-scaled).exp();
        let gate = 1.0 / (1.0 + exp_neg_scaled);

        // Apply gate: x * gate
        outputs.push(x * gate);
    }
    Ok(outputs)
}

/// GoldenGELU backward pass for training
///
/// Derivative: d/dx [x * σ(φx)] = σ(φx) + φ * x * σ(φx) * (1 - σ(φx))
///
/// Args:
///   inputs: Original input values
///   grad_outputs: Gradients from next layer
///
/// Returns:
///   Gradient w.r.t. input
#[pyfunction]
#[pyo3(name = "golden_gelu_backward")]
pub fn golden_gelu_backward(inputs: Vec<f64>, grad_outputs: Vec<f64>) -> PyResult<Vec<f64>> {
    if inputs.len() != grad_outputs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inputs and grad_outputs must have same length",
        ));
    }

    // CPU-only implementation
    let mut grad_inputs = Vec::with_capacity(inputs.len());
    for (&x, &grad_out) in inputs.iter().zip(grad_outputs.iter()) {
        // Scale by phi
        let scaled = PHI * x;

        // Compute sigmoid: σ(φx)
        let exp_neg_scaled = (-scaled).exp();
        let gate = 1.0 / (1.0 + exp_neg_scaled);

        // Compute derivative: σ + φ * x * σ * (1 - σ)
        let gate_complement = 1.0 - gate;
        let derivative = gate + PHI * x * gate * gate_complement;

        // Chain rule: grad_output * derivative
        grad_inputs.push(grad_out * derivative);
    }
    Ok(grad_inputs)
}

/// Batched GoldenGELU forward pass for efficiency
///
/// Args:
///   batch: Flattened list of input tensors [batch_size, n_elements]
///   batch_size: Number of tensors in batch
///   n_elements: Number of elements per tensor
///
/// Returns:
///   Flattened GoldenGELU-activated outputs
#[pyfunction]
#[pyo3(name = "batched_golden_gelu_forward")]
pub fn batched_golden_gelu_forward(
    batch: Vec<f64>,
    batch_size: usize,
    n_elements: usize,
) -> PyResult<Vec<f64>> {
    if batch.len() != batch_size * n_elements {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "batch.len() ({}) must equal batch_size * n_elements ({} * {})",
            batch.len(),
            batch_size,
            n_elements
        )));
    }

    // CPU-only implementation: process each tensor
    let mut outputs = Vec::with_capacity(batch.len());
    for &x in batch.iter() {
        let scaled = PHI * x;
        let exp_neg_scaled = (-scaled).exp();
        let gate = 1.0 / (1.0 + exp_neg_scaled);
        outputs.push(x * gate);
    }
    Ok(outputs)
}

/// Get the golden ratio constant used in GoldenGELU
///
/// Returns:
///   φ = 1.6180339887498948482
#[pyfunction]
#[pyo3(name = "get_golden_gelu_phi")]
pub fn get_golden_gelu_phi() -> PyResult<f64> {
    use crate::exact::GoldenExact;
    Ok(GoldenExact::phi().to_f64())
}
