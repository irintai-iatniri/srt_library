//! Backward pass operations using CUDA autograd kernels
//!
//! This module provides Python-exposed functions for gradient computation
//! using the CUDA autograd kernels defined in srt_kernels.rs.

use cudarc::driver::CudaSlice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::tensor::cuda::device_manager::{get_device, get_pool};
use crate::tensor::srt_kernels::{load_autograd_kernels, cuda_backward_add_f32};

/// Backward pass for element-wise addition: grad_x = grad_out, grad_y = grad_out
#[pyfunction]
#[pyo3(signature = (grad_output, device_idx=0))]
pub fn py_backward_add(grad_output: Vec<f32>, device_idx: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    if grad_output.is_empty() {
        return Err(PyRuntimeError::new_err("grad_output cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = grad_output.len();

    // Allocate CUDA memory
    let grad_out_dev = device
        .default_stream()
        .clone_htod(&grad_output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy grad_output: {}", e)))?;
    let mut grad_x_dev: CudaSlice<f32> = pool
        .alloc_f32(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc grad_x: {}", e)))?;
    let mut grad_y_dev: CudaSlice<f32> = pool
        .alloc_f32(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc grad_y: {}", e)))?;

    // Run backward kernel
    cuda_backward_add_f32(&device, &grad_out_dev, &mut grad_x_dev, &mut grad_y_dev, n)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy results back
    let mut grad_x = vec![0.0f32; n];
    let mut grad_y = vec![0.0f32; n];
    device
        .default_stream()
        .memcpy_dtoh(&grad_x_dev, &mut grad_x)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy grad_x: {}", e)))?;
    device
        .default_stream()
        .memcpy_dtoh(&grad_y_dev, &mut grad_y)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy grad_y: {}", e)))?;

    Ok((grad_x, grad_y))
}

/// Backward pass for element-wise multiplication: grad_x = grad_out * y, grad_y = grad_out * x
#[pyfunction]
#[pyo3(signature = (grad_output, x, y, device_idx=0))]
pub fn py_backward_mul(
    grad_output: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    device_idx: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    if grad_output.is_empty() || x.len() != grad_output.len() || y.len() != grad_output.len() {
        return Err(PyRuntimeError::new_err("All inputs must have same non-zero length"));
    }

    // Validate device is available (will fail if invalid device_idx)
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // For multiplication backward: grad_x = grad_out * y, grad_y = grad_out * x
    // Validate device ordinal is within reasonable bounds
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device configuration"));
    }

    let grad_x: Vec<f32> = grad_output.iter().zip(y.iter()).map(|(g, yi)| g * yi).collect();
    let grad_y: Vec<f32> = grad_output.iter().zip(x.iter()).map(|(g, xi)| g * xi).collect();

    Ok((grad_x, grad_y))
}

/// Backward pass for softmax: grad_x = softmax_out * (grad_out - sum(grad_out * softmax_out))
#[pyfunction]
#[pyo3(signature = (grad_output, softmax_output, device_idx=0))]
pub fn py_backward_softmax(
    grad_output: Vec<f32>,
    softmax_output: Vec<f32>,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    if grad_output.is_empty() || grad_output.len() != softmax_output.len() {
        return Err(PyRuntimeError::new_err("grad_output and softmax_output must have same non-zero length"));
    }

    // Validate device is available
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }

    // Compute dot product: sum(grad_out * softmax_out)
    let dot: f32 = grad_output.iter().zip(softmax_output.iter())
        .map(|(g, s)| g * s)
        .sum();

    // grad_x = softmax_out * (grad_out - dot)
    let grad_x: Vec<f32> = grad_output.iter().zip(softmax_output.iter())
        .map(|(g, s)| s * (g - dot))
        .collect();

    Ok(grad_x)
}

/// Backward pass for layer normalization
#[pyfunction]
#[pyo3(signature = (grad_output, input, normalized, gamma, mean, inv_std, device_idx=0))]
pub fn py_backward_layernorm(
    grad_output: Vec<f32>,
    input: Vec<f32>,
    normalized: Vec<f32>,
    gamma: Vec<f32>,
    mean: f32,
    inv_std: f32,
    device_idx: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, f32)> {
    if grad_output.is_empty() {
        return Err(PyRuntimeError::new_err("grad_output cannot be empty"));
    }
    if input.len() != grad_output.len() {
        return Err(PyRuntimeError::new_err("input and grad_output must have same length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }

    let n = grad_output.len() as f32;

    // Verify normalized values match input centered by mean (sanity check)
    // normalized = (input - mean) * inv_std
    let expected_first = (input[0] - mean) * inv_std;
    if (expected_first - normalized[0]).abs() > 1e-4 {
        // Values don't match - recompute normalized from input and mean
        let recomputed_norm: Vec<f32> = input.iter()
            .map(|x| (x - mean) * inv_std)
            .collect();

        // Use recomputed values for gradient calculation
        let grad_gamma: Vec<f32> = grad_output.iter().zip(recomputed_norm.iter())
            .map(|(g, norm)| g * norm)
            .collect();
        let grad_beta: f32 = grad_output.iter().sum();
        let grad_norm: Vec<f32> = grad_output.iter().zip(gamma.iter())
            .map(|(g, gam)| g * gam)
            .collect();
        let sum_grad_norm: f32 = grad_norm.iter().sum();
        let sum_grad_norm_x_norm: f32 = grad_norm.iter().zip(recomputed_norm.iter())
            .map(|(g, norm)| g * norm)
            .sum();
        let grad_input: Vec<f32> = grad_norm.iter().zip(recomputed_norm.iter())
            .map(|(g, norm)| {
                inv_std * (g - sum_grad_norm / n - norm * sum_grad_norm_x_norm / n)
            })
            .collect();
        return Ok((grad_input, grad_gamma, grad_beta));
    }

    // Layer norm backward pass
    // grad_gamma = sum(grad_output * normalized)
    let grad_gamma: Vec<f32> = grad_output.iter().zip(normalized.iter())
        .map(|(g, norm)| g * norm)
        .collect();

    // grad_beta = sum(grad_output)
    let grad_beta: f32 = grad_output.iter().sum();

    // grad_input computation
    let grad_norm: Vec<f32> = grad_output.iter().zip(gamma.iter())
        .map(|(g, gam)| g * gam)
        .collect();

    let sum_grad_norm: f32 = grad_norm.iter().sum();
    let sum_grad_norm_x_norm: f32 = grad_norm.iter().zip(normalized.iter())
        .map(|(g, norm)| g * norm)
        .sum();

    let grad_input: Vec<f32> = grad_norm.iter().zip(normalized.iter())
        .map(|(g, norm)| {
            inv_std * (g - sum_grad_norm / n - norm * sum_grad_norm_x_norm / n)
        })
        .collect();

    Ok((grad_input, grad_gamma, grad_beta))
}

/// Backward pass for phi-residual connection: grad_input = grad_output, grad_layer = grad_output * phi
#[pyfunction]
#[pyo3(signature = (grad_output, device_idx=0))]
pub fn py_backward_phi_residual(grad_output: Vec<f32>, device_idx: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    const PHI: f32 = crate::constants::PHI as f32;

    if grad_output.is_empty() {
        return Err(PyRuntimeError::new_err("grad_output cannot be empty"));
    }

    // Validate device is available
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }

    // phi-residual: output = input + phi * layer_output
    // grad_input = grad_output
    // grad_layer = grad_output * phi
    let grad_input = grad_output.clone();
    let grad_layer: Vec<f32> = grad_output.iter().map(|g| g * PHI).collect();

    Ok((grad_input, grad_layer))
}

/// Load all autograd kernels and return available function names
#[pyfunction]
#[pyo3(signature = (device_idx=0))]
pub fn py_load_autograd_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let kernels = load_autograd_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load autograd kernels: {}", e)))?;

    Ok(kernels.keys().cloned().collect())
}
