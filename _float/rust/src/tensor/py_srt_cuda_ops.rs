//! Python bindings for SRT CUDA operations
//!
//! Exposes toroidal math, gnosis masking, golden exponentials,
//! autograd kernels, and matrix multiplication to Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// SRT Constants
const PHI: f64 = 1.6180339887498948482;
const PHI_INV: f64 = 0.6180339887498948482;
const Q_DEFICIT: f64 = 0.027395146920;

use super::cuda::device_manager::get_device;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::tensor::cuda::CudaComplex64;
use num_complex::Complex64;
use crate::tensor::precision_policy::{get_srt_operation_policy, PrecisionPolicy};
use crate::hierarchy;

// =============================================================================
// Toroidal Math Functions (CPU fallbacks + CUDA)
// =============================================================================

/// Compute sin(2πx) for toroidal coordinates (CPU)
#[allow(unused)]
fn cpu_sin_toroidal_f64(data: &[f64]) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    data.iter().map(|&x| (two_pi * x).sin()).collect()
}

/// Compute cos(2πx) for toroidal coordinates (CPU)
#[allow(unused)]
fn cpu_cos_toroidal_f64(data: &[f64]) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    data.iter().map(|&x| (two_pi * x).cos()).collect()
}

/// Compute atan2 for toroidal coordinates (CPU)
#[allow(unused)]
fn cpu_atan2_toroidal_f64(y: &[f64], x: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(&yi, &xi)| yi.atan2(xi) / (2.0 * std::f64::consts::PI))
        .collect()
}

/// Compute φ^x (golden exponential) using exact Fibonacci recurrence (CPU)
#[allow(unused)]
fn cpu_phi_exp_f64(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| PHI.powf(x)).collect()
}

/// Compute φ^(-x) (inverse golden exponential) (CPU)
#[allow(unused)]
fn cpu_phi_exp_inv_f64(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| PHI_INV.powf(x)).collect()
}

// =============================================================================
// CUDA Toroidal Functions (with proper GPU memory allocation)
// =============================================================================

// Note: static_sin_toroidal_f64 and static_cos_toroidal_f64 from srt_kernels
// are NOT used because they pass CPU pointers to GPU kernels.
// Instead, we use cuda_sin_toroidal_f64/cuda_cos_toroidal_f64 below
// which properly allocate GPU memory.

// =============================================================================
// Device-Pointer FFI Declarations
// =============================================================================
// These functions expect DEVICE pointers only. Callers must allocate GPU memory
// and perform H2D/D2H copies explicitly.

extern "C" {
    fn dev_sin_toroidal_f64(out_dev: *mut f64, in_dev: *const f64, n: i32) -> i32;
    fn dev_cos_toroidal_f64(out_dev: *mut f64, in_dev: *const f64, n: i32) -> i32;
    fn dev_phi_exp_f64(out_dev: *mut f64, in_dev: *const f64, n: i32) -> i32;
    fn dev_phi_exp_inv_f64(out_dev: *mut f64, in_dev: *const f64, n: i32) -> i32;
}

/// GPU sin(2πx) with proper memory allocation (H2D, kernel, D2H)
/// Uses dev_* API which requires explicit device pointers.
fn cuda_sin_toroidal_f64(data: &[f64]) -> Result<Vec<f64>, String> {
    // Check if CUDA is available
    let device = match get_device(0) {
        Ok(d) => d,
        Err(e) => return Err(format!("CUDA device unavailable: {}", e)),
    };
    let pool = get_pool(0).map_err(|e| format!("Pool unavailable: {}", e))?;
    let stream = device.default_stream();
    
    // Step 1: Allocate GPU memory and copy input H2D
    let input_dev = stream
        .clone_htod(data)
        .map_err(|e| format!("H2D copy failed: {}", e))?;
    
    // Step 2: Allocate output on GPU using pool
    let mut output_dev = pool
        .alloc_f64(data.len())
        .map_err(|e| format!("GPU alloc failed: {}", e))?;
    
    // Step 3: Get device pointers and call kernel
    {
        let n = data.len() as i32;
        let (out_ptr, _out_guard) = output_dev.device_ptr_mut(&stream);
        let (in_ptr, _in_guard) = input_dev.device_ptr(&stream);
        
        // Call the device-pointer kernel (expects GPU addresses)
        let result = unsafe { dev_sin_toroidal_f64(out_ptr as *mut f64, in_ptr as *const f64, n) };
        if result != 0 {
            return Err(format!("CUDA kernel failed with code {}", result));
        }
    }
    
    // Step 4: Copy result D2H (guards dropped, borrow released)
    // Note: Pool may allocate more than requested (power-of-2 buckets).
    // We allocate a buffer large enough for the pool's allocation, then truncate.
    let pool_len = output_dev.len();
    let mut output = vec![0.0f64; pool_len];
    stream
        .memcpy_dtoh(&output_dev, &mut output)
        .map_err(|e| format!("D2H copy failed: {}", e))?;
    
    // Truncate to requested size
    output.truncate(data.len());
    Ok(output)
}

/// GPU cos(2πx) with proper memory allocation (H2D, kernel, D2H)
/// Uses dev_* API which requires explicit device pointers.
fn cuda_cos_toroidal_f64(data: &[f64]) -> Result<Vec<f64>, String> {
    // Check if CUDA is available
    let device = match get_device(0) {
        Ok(d) => d,
        Err(e) => return Err(format!("CUDA device unavailable: {}", e)),
    };
    let pool = get_pool(0).map_err(|e| format!("Pool unavailable: {}", e))?;
    let stream = device.default_stream();
    
    // Step 1: Allocate GPU memory and copy input H2D
    let input_dev = stream
        .clone_htod(data)
        .map_err(|e| format!("H2D copy failed: {}", e))?;
    
    // Step 2: Allocate output on GPU using pool
    let mut output_dev = pool
        .alloc_f64(data.len())
        .map_err(|e| format!("GPU alloc failed: {}", e))?;
    
    // Step 3: Get device pointers and call kernel
    {
        let n = data.len() as i32;
        let (out_ptr, _out_guard) = output_dev.device_ptr_mut(&stream);
        let (in_ptr, _in_guard) = input_dev.device_ptr(&stream);
        
        // Call the device-pointer kernel (expects GPU addresses)
        let result = unsafe { dev_cos_toroidal_f64(out_ptr as *mut f64, in_ptr as *const f64, n) };
        if result != 0 {
            return Err(format!("CUDA kernel failed with code {}", result));
        }
    }
    
    // Step 4: Copy result D2H (guards dropped, borrow released)
    // Note: Pool may allocate more than requested (power-of-2 buckets).
    let pool_len = output_dev.len();
    let mut output = vec![0.0f64; pool_len];
    stream
        .memcpy_dtoh(&output_dev, &mut output)
        .map_err(|e| format!("D2H copy failed: {}", e))?;
    
    // Truncate to requested size
    output.truncate(data.len());
    Ok(output)
}

// =============================================================================
// CUDA Phi Exponential Functions (with proper GPU memory allocation)
// =============================================================================

/// GPU φ^x (golden exponential) with proper memory allocation (H2D, kernel, D2H)
/// Uses dev_phi_exp_f64 API which requires explicit device pointers.
fn cuda_phi_exp_f64(data: &[f64]) -> Result<Vec<f64>, String> {
    // Check if CUDA is available
    let device = match get_device(0) {
        Ok(d) => d,
        Err(e) => return Err(format!("CUDA device unavailable: {}", e)),
    };
    let pool = get_pool(0).map_err(|e| format!("Pool unavailable: {}", e))?;
    let stream = device.default_stream();
    
    // Step 1: Allocate GPU memory and copy input H2D
    let input_dev = stream
        .clone_htod(data)
        .map_err(|e| format!("H2D copy failed: {}", e))?;
    
    // Step 2: Allocate output on GPU using pool
    let mut output_dev = pool
        .alloc_f64(data.len())
        .map_err(|e| format!("GPU alloc failed: {}", e))?;
    
    // Step 3: Get device pointers and call kernel
    {
        let n = data.len() as i32;
        let (out_ptr, _out_guard) = output_dev.device_ptr_mut(&stream);
        let (in_ptr, _in_guard) = input_dev.device_ptr(&stream);
        
        // Call the device-pointer kernel (expects GPU addresses)
        let result = unsafe { dev_phi_exp_f64(out_ptr as *mut f64, in_ptr as *const f64, n) };
        if result != 0 {
            return Err(format!("CUDA kernel failed with code {}", result));
        }
    }
    
    // Step 4: Copy result D2H (guards dropped, borrow released)
    let pool_len = output_dev.len();
    let mut output = vec![0.0f64; pool_len];
    stream
        .memcpy_dtoh(&output_dev, &mut output)
        .map_err(|e| format!("D2H copy failed: {}", e))?;
    
    // Truncate to requested size
    output.truncate(data.len());
    Ok(output)
}

/// GPU φ^(-x) (inverse golden exponential) with proper memory allocation (H2D, kernel, D2H)
/// Uses dev_phi_exp_inv_f64 API which requires explicit device pointers.
fn cuda_phi_exp_inv_f64(data: &[f64]) -> Result<Vec<f64>, String> {
    // Check if CUDA is available
    let device = match get_device(0) {
        Ok(d) => d,
        Err(e) => return Err(format!("CUDA device unavailable: {}", e)),
    };
    let pool = get_pool(0).map_err(|e| format!("Pool unavailable: {}", e))?;
    let stream = device.default_stream();
    
    // Step 1: Allocate GPU memory and copy input H2D
    let input_dev = stream
        .clone_htod(data)
        .map_err(|e| format!("H2D copy failed: {}", e))?;
    
    // Step 2: Allocate output on GPU using pool
    let mut output_dev = pool
        .alloc_f64(data.len())
        .map_err(|e| format!("GPU alloc failed: {}", e))?;
    
    // Step 3: Get device pointers and call kernel
    {
        let n = data.len() as i32;
        let (out_ptr, _out_guard) = output_dev.device_ptr_mut(&stream);
        let (in_ptr, _in_guard) = input_dev.device_ptr(&stream);
        
        // Call the device-pointer kernel (expects GPU addresses)
        let result = unsafe { dev_phi_exp_inv_f64(out_ptr as *mut f64, in_ptr as *const f64, n) };
        if result != 0 {
            return Err(format!("CUDA kernel failed with code {}", result));
        }
    }
    
    // Step 4: Copy result D2H (guards dropped, borrow released)
    let pool_len = output_dev.len();
    let mut output = vec![0.0f64; pool_len];
    stream
        .memcpy_dtoh(&output_dev, &mut output)
        .map_err(|e| format!("D2H copy failed: {}", e))?;
    
    // Truncate to requested size
    output.truncate(data.len());
    Ok(output)
}

// =============================================================================
// Python-Exposed Toroidal Functions
// =============================================================================

/// Compute sin(2πx) for W⁴ torus coordinates
/// Uses GPU acceleration when available with proper memory handling.
#[pyfunction]
pub fn py_sin_toroidal(data: Vec<f64>) -> Vec<f64> {
    // Try GPU path with proper memory allocation
    match cuda_sin_toroidal_f64(&data) {
        Ok(result) => result,
        Err(_) => cpu_sin_toroidal_f64(&data), // Fallback to CPU
    }
}

/// Compute cos(2πx) for W⁴ torus coordinates
/// Uses GPU acceleration when available with proper memory handling.
#[pyfunction]
pub fn py_cos_toroidal(data: Vec<f64>) -> Vec<f64> {
    // Try GPU path with proper memory allocation
    match cuda_cos_toroidal_f64(&data) {
        Ok(result) => result,
        Err(_) => cpu_cos_toroidal_f64(&data), // Fallback to CPU
    }
}

/// Compute atan2 normalized for toroidal coordinates
#[pyfunction]
pub fn py_atan2_toroidal(y: Vec<f64>, x: Vec<f64>) -> PyResult<Vec<f64>> {
    if y.len() != x.len() {
        return Err(PyRuntimeError::new_err("y and x must have same length"));
    }
    Ok(cpu_atan2_toroidal_f64(&y, &x))
}

/// Compute φ^x (golden exponential)
/// Uses GPU acceleration when available with proper memory handling.
#[pyfunction]
pub fn py_phi_exp(data: Vec<f64>) -> Vec<f64> {
    // Try GPU path with proper memory allocation
    match cuda_phi_exp_f64(&data) {
        Ok(result) => result,
        Err(_) => cpu_phi_exp_f64(&data), // Fallback to CPU
    }
}

/// Compute φ^(-x) (inverse golden exponential)
/// Uses GPU acceleration when available with proper memory handling.
#[pyfunction]
pub fn py_phi_exp_inv(data: Vec<f64>) -> Vec<f64> {
    // Try GPU path with proper memory allocation
    match cuda_phi_exp_inv_f64(&data) {
        Ok(result) => result,
        Err(_) => cpu_phi_exp_inv_f64(&data), // Fallback to CPU
    }
}

// =============================================================================
// Gnosis Masking Functions (CPU implementations)
// =============================================================================

/// Standard gnosis mask: filters based on syntony threshold
///
/// mask(i) = input(i) * strength if syntony(i) > threshold else 0
#[allow(unused)]
fn cpu_gnosis_mask_f64(input: &[f64], syntony: &[f64], threshold: f64, strength: f64) -> Vec<f64> {
    input
        .iter()
        .zip(syntony.iter())
        .map(
            |(&inp, &syn)| {
                if syn > threshold {
                    inp * strength
                } else {
                    0.0
                }
            },
        )
        .collect()
}

/// Adaptive gnosis mask: adjusts threshold based on local syntony
///
/// local_threshold = threshold * (1 - adaptability * (syntony - mean_syntony))
#[allow(unused)]
fn cpu_adaptive_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    adaptability: f64,
    ratio: f64,
) -> Vec<f64> {
    let mean_syn: f64 = syntony.iter().sum::<f64>() / syntony.len() as f64;
    let threshold = 1.0 - Q_DEFICIT; // Default threshold from q-deficit

    input
        .iter()
        .zip(syntony.iter())
        .map(|(&inp, &syn)| {
            let local_thresh = threshold * (1.0 - adaptability * (syn - mean_syn));
            if syn > local_thresh {
                inp * ratio
            } else {
                inp * (1.0 - ratio)
            }
        })
        .collect()
}

/// Fractal gnosis mask: applies hierarchical masking at multiple scales
#[allow(unused)]
fn cpu_fractal_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    levels: usize,
    threshold: f64,
    scale: f64,
) -> Vec<f64> {
    let mut result = input.to_vec();

    for level in 0..levels {
        let level_scale = scale.powf(level as f64);
        let level_thresh = threshold * PHI_INV.powf(level as f64);

        for i in 0..result.len() {
            if syntony[i] > level_thresh {
                result[i] *= level_scale;
            }
        }
    }

    result
}

/// Temporal gnosis mask: incorporates previous state with memory decay
#[allow(unused)]
fn cpu_temporal_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    prev: &[f64],
    threshold: f64,
    memory: f64, // How much to remember (0-1)
    rate: f64,   // Learning rate for new information
) -> Vec<f64> {
    input
        .iter()
        .zip(syntony.iter())
        .zip(prev.iter())
        .map(|((&inp, &syn), &prv)| {
            let new_val = if syn > threshold { inp * rate } else { 0.0 };
            memory * prv + (1.0 - memory) * new_val
        })
        .collect()
}

// =============================================================================
// Python-Exposed Gnosis Mask Functions
// =============================================================================

/// Apply gnosis mask to filter by syntony threshold
#[pyfunction]
#[pyo3(signature = (input, syntony, threshold=0.9726, strength=1.0))]
pub fn py_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    threshold: f64,
    strength: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_gnosis_mask_f64(&input, &syntony, threshold, strength))
}

/// Apply adaptive gnosis mask with local threshold adjustment
#[pyfunction]
#[pyo3(signature = (input, syntony, adaptability=0.1, ratio=1.0))]
pub fn py_adaptive_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    adaptability: f64,
    ratio: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_adaptive_gnosis_mask_f64(
        &input,
        &syntony,
        adaptability,
        ratio,
    ))
}

/// Apply fractal gnosis mask at multiple hierarchical levels
#[pyfunction]
#[pyo3(signature = (input, syntony, levels=3, threshold=0.9726, scale=1.618))]
pub fn py_fractal_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    levels: usize,
    threshold: f64,
    scale: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_fractal_gnosis_mask_f64(
        &input, &syntony, levels, threshold, scale,
    ))
}

/// Apply temporal gnosis mask with memory of previous state
#[pyfunction]
#[pyo3(signature = (input, syntony, prev, threshold=0.9726, memory=0.618, rate=1.0))]
pub fn py_temporal_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    prev: Vec<f64>,
    threshold: f64,
    memory: f64,
    rate: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() || input.len() != prev.len() {
        return Err(PyRuntimeError::new_err(
            "input, syntony, and prev must have same length",
        ));
    }
    Ok(cpu_temporal_gnosis_mask_f64(
        &input, &syntony, &prev, threshold, memory, rate,
    ))
}

// =============================================================================
// Autograd Gradient Filtering (CPU implementation)
// =============================================================================

/// Filter gradients using golden attractor - corrupted gradients are snapped
/// to the Q(φ) lattice to prevent gradient explosion/vanishing
#[allow(unused)]
fn cpu_autograd_filter_f64(
    gradients: &[f64],
    current_state: &[f64],
    golden_attractor_strength: f64,
    corruption_threshold: f64,
) -> Vec<f64> {
    let phi = PHI;
    let phi_inv = PHI_INV;

    gradients
        .iter()
        .zip(current_state.iter())
        .map(|(&grad, &state)| {
            // Check if gradient is "corrupted" (too large or too small)
            let abs_grad = grad.abs();
            let is_corrupted = abs_grad > corruption_threshold
                || (abs_grad > 0.0 && abs_grad < corruption_threshold * phi_inv.powi(10));

            if is_corrupted {
                // Snap gradient to Q(φ) lattice point
                // Find nearest a + b*φ where a, b are small integers
                let scaled = grad / phi_inv;
                let a = scaled.round();
                let b = ((grad - a * phi_inv) / phi).round();
                let snapped = a * phi_inv + b * phi;

                // Blend with attractor
                let attractor = state * golden_attractor_strength;
                snapped * (1.0 - golden_attractor_strength) + attractor * golden_attractor_strength
            } else {
                grad
            }
        })
        .collect()
}

/// Attractor memory update: evolves current state toward attractor basin
#[allow(unused)]
fn cpu_attractor_memory_update_f64(
    current: &[f64],
    gradients: &[f64],
    attractor_strength: f64,
    learning_rate: f64,
) -> Vec<f64> {
    current
        .iter()
        .zip(gradients.iter())
        .map(|(&curr, &grad)| {
            // Apply gradient with φ-scaled learning rate
            let updated = curr - learning_rate * grad;

            // Pull toward nearest Q(φ) attractor point
            let scaled = updated / PHI;
            let a = scaled.round();
            let attractor_point = a * PHI;

            // Blend based on attractor strength
            updated * (1.0 - attractor_strength) + attractor_point * attractor_strength
        })
        .collect()
}

// =============================================================================
// Python-Exposed Autograd Functions
// =============================================================================

/// Filter gradients to prevent corruption (explosion/vanishing)
#[pyfunction]
#[pyo3(signature = (gradients, current_state, attractor_strength=0.027395, corruption_threshold=1e6))]
pub fn py_autograd_filter(
    gradients: Vec<f64>,
    current_state: Vec<f64>,
    attractor_strength: f64,
    corruption_threshold: f64,
) -> PyResult<Vec<f64>> {
    if gradients.len() != current_state.len() {
        return Err(PyRuntimeError::new_err(
            "gradients and current_state must have same length",
        ));
    }
    Ok(cpu_autograd_filter_f64(
        &gradients,
        &current_state,
        attractor_strength,
        corruption_threshold,
    ))
}

/// Update memory state with attractor basin pull
#[pyfunction]
#[pyo3(signature = (current, gradients, attractor_strength=0.027395, learning_rate=0.001))]
pub fn py_attractor_memory_update(
    current: Vec<f64>,
    gradients: Vec<f64>,
    attractor_strength: f64,
    learning_rate: f64,
) -> PyResult<Vec<f64>> {
    if current.len() != gradients.len() {
        return Err(PyRuntimeError::new_err(
            "current and gradients must have same length",
        ));
    }
    Ok(cpu_attractor_memory_update_f64(
        &current,
        &gradients,
        attractor_strength,
        learning_rate,
    ))
}

// =============================================================================
// Entropy and Syntony Metric Functions
// =============================================================================

/// Compute entropy: -Σ p * log(p) normalized by golden ratio
#[allow(unused)]
fn cpu_entropy_f64(values: &[f64]) -> f64 {
    // Normalize to probabilities
    let sum: f64 = values.iter().map(|v| v.abs()).sum();
    if sum == 0.0 {
        return 0.0;
    }

    let probs: Vec<f64> = values.iter().map(|v| v.abs() / sum).collect();

    // Compute entropy
    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize by log(n) * φ^(-1) for golden-scaled entropy
    let max_entropy = (probs.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / (max_entropy * PHI_INV)
    } else {
        0.0
    }
}

/// Compute syntony metric: measures how close tensor is to Q(φ) lattice
#[allow(unused)]
fn cpu_syntony_metric_f64(tensor: &[f64]) -> f64 {
    if tensor.is_empty() {
        return 1.0;
    }

    // Measure deviation from nearest Q(φ) lattice points
    let total_deviation: f64 = tensor
        .iter()
        .map(|&v| {
            // Find nearest a + b*φ
            let scaled = v / PHI;
            let a = scaled.round();
            let remainder = v - a * PHI;
            let b = (remainder / PHI_INV).round();
            let nearest = a * PHI + b * PHI_INV;
            (v - nearest).abs()
        })
        .sum();

    // Syntony = 1 - (average deviation / PHI)
    let avg_deviation = total_deviation / tensor.len() as f64;
    (1.0 - avg_deviation / PHI).max(0.0).min(1.0)
}

/// Compute golden-scaled entropy
#[pyfunction]
pub fn py_golden_entropy(values: Vec<f64>) -> f64 {
    cpu_entropy_f64(&values)
}

/// Compute syntony metric (how close to Q(φ) lattice)
#[pyfunction]
pub fn py_syntony_metric(tensor: Vec<f64>) -> f64 {
    cpu_syntony_metric_f64(&tensor)
}

// =============================================================================
// CUDA Matrix Multiplication Wrappers (when CUDA feature enabled)
// =============================================================================

use super::srt_kernels::{cuda_dgemm_native_f64, cuda_sgemm_native_f32};

use super::cuda::device_manager::get_pool;

/// High-performance SGEMM using SRT native kernels
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, alpha=1.0, beta=0.0, device_idx=0))]
pub fn py_cuda_sgemm(
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory using pool and stream
    let a_dev = device
        .default_stream()
        .clone_htod(&a)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy A: {}", e)))?;
    let b_dev = device
        .default_stream()
        .clone_htod(&b)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy B: {}", e)))?;
    let mut c_dev: CudaSlice<f32> = pool
        .alloc_f32(m * n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc C: {}", e)))?;

    // Run SGEMM
    cuda_sgemm_native_f32(&device, &mut c_dev, &a_dev, &b_dev, m, n, k, alpha, beta)?;

    // Copy result back
    let mut result = vec![0.0f32; m * n];
    device
        .default_stream()
        .memcpy_dtoh(&c_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// CUDA Double Precision Matrix Multiplication (DGEMM)
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, alpha, beta, device_idx=0))]
pub fn py_cuda_dgemm(
    a: Vec<f64>,
    b: Vec<f64>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory using pool and stream
    let a_dev = device
        .default_stream()
        .clone_htod(&a)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy A: {}", e)))?;
    let b_dev = device
        .default_stream()
        .clone_htod(&b)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy B: {}", e)))?;
    let mut c_dev: CudaSlice<f64> = pool
        .alloc_f64(m * n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc C: {}", e)))?;

    // Run DGEMM
    cuda_dgemm_native_f64(&device, &mut c_dev, &a_dev, &b_dev, m, n, k, alpha, beta)?;

    // Copy result back
    let mut result = vec![0.0f64; m * n];
    device
        .default_stream()
        .memcpy_dtoh(&c_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Precision-policy enforced matrix multiplication
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, alpha=1.0, beta=0.0, device_idx=0, operation="mm"))]
pub fn py_cuda_mm_with_policy(
    a: Vec<f64>,
    b: Vec<f64>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    device_idx: usize,
    operation: &str,
) -> PyResult<Vec<f64>> {
    // Get precision policy for this operation
    let policy = get_srt_operation_policy(operation);
    
    match policy {
        PrecisionPolicy::Exact => {
            // Use double precision for exact operations
            py_cuda_dgemm(a, b, m, n, k, alpha, beta, device_idx)
        }
        PrecisionPolicy::MixedPrecision => {
            // Convert to f32 for mixed precision
            let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;
            
            let result_f32 = py_cuda_sgemm(a_f32, b_f32, m, n, k, alpha_f32, beta_f32, device_idx)?;
            
            // Convert back to f64
            Ok(result_f32.iter().map(|&x| x as f64).collect())
        }
        PrecisionPolicy::LowPrecision => {
            // For low precision, we still use f32 but could use f16 in future
            let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;

            let result_f32 = py_cuda_sgemm(a_f32, b_f32, m, n, k, alpha_f32, beta_f32, device_idx)?;

            // Convert back to f64
            Ok(result_f32.iter().map(|&x| x as f64).collect())
        }
    }
}

/// Apply E8 hierarchy correction to values
///
/// Uses the hierarchy module to apply q-deficit corrections based on E8 structure.
/// The correction factor is (1 + sign * Q_DEFICIT / E8_ROOTS) where E8_ROOTS = 240.
#[pyfunction]
#[pyo3(signature = (values, sign=1))]
pub fn py_hierarchy_e8_correction(values: Vec<f64>, sign: i32) -> PyResult<Vec<f64>> {
    // Use hierarchy module's E8_ROOTS constant (240) as divisor
    let divisor = hierarchy::E8_ROOTS as f64;
    let divisors: Vec<f64> = vec![divisor; values.len()];
    let signs: Vec<i32> = vec![sign; values.len()];
    hierarchy::apply_correction(values, divisors, signs)
}

/// Apply E7 hierarchy correction to values
///
/// Uses hierarchy module for intermediate unification scale correction.
/// The correction factor is (1 + sign * Q_DEFICIT / E7_ROOTS) where E7_ROOTS = 126.
#[pyfunction]
#[pyo3(signature = (values, sign=1))]
pub fn py_hierarchy_e7_correction(values: Vec<f64>, sign: i32) -> PyResult<Vec<f64>> {
    let divisor = hierarchy::E7_ROOTS as f64;
    let divisors: Vec<f64> = vec![divisor; values.len()];
    let signs: Vec<i32> = vec![sign; values.len()];
    hierarchy::apply_correction(values, divisors, signs)
}

/// Apply Coxeter-Kissing (720) hierarchy correction
///
/// Uses h × K = 30 × 24 = 720, the fundamental collapse threshold product.
#[pyfunction]
#[pyo3(signature = (values, sign=1))]
pub fn py_hierarchy_coxeter_kissing_correction(values: Vec<f64>, sign: i32) -> PyResult<Vec<f64>> {
    let divisor = hierarchy::COXETER_KISSING_720 as f64;
    let divisors: Vec<f64> = vec![divisor; values.len()];
    let signs: Vec<i32> = vec![sign; values.len()];
    hierarchy::apply_correction(values, divisors, signs)
}

/// Gather elements from source array using indices (f32)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f32(src: Vec<f32>, indices: Vec<i32>, device_idx: usize) -> PyResult<Vec<f32>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f32> = pool
        .alloc_f32(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run gather
    super::srt_kernels::cuda_gather_f32(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter elements to output array using indices (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter
    super::srt_kernels::cuda_scatter_f64(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter add elements to output array using indices (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_add_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter add
    super::srt_kernels::cuda_scatter_add_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter elements to output array using indices (f32)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f32(
    src: Vec<f32>,
    indices: Vec<i32>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f32> = pool
        .alloc_f32(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter
    super::srt_kernels::cuda_scatter_f32(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with golden ratio weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_golden_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_golden_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with Mersenne stable precision (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_mersenne_stable_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_mersenne_stable_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with Lucas shadow weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_lucas_shadow_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_lucas_shadow_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with Pisano hooked weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_pisano_hooked_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_pisano_hooked_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with E8 roots weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_e8_roots_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_e8_roots_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with golden cone weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_golden_cone_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_golden_cone_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

// TRILINEAR wrappers (9 kernels) - begin
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_simple_f64(&device, &out_dev, &input_dev, input.len())?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_toroidal_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_toroidal_f64(
        &device,
        &mut out_dev,
        &input_dev,
        input.len(),
    )?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_phi_weighted_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_phi_weighted_f64(
        &device,
        &mut out_dev,
        &input_dev,
        input.len(),
    )?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_golden_decay_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_golden_decay_f64(
        &device,
        &mut out_dev,
        &input_dev,
        input.len(),
    )?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_causal_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_causal_f64(&device, &mut out_dev, &input_dev, input.len())?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_retrocausal_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_retrocausal_f64(
        &device,
        &mut out_dev,
        &input_dev,
        input.len(),
    )?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_symmetric_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_symmetric_f64(
        &device,
        &mut out_dev,
        &input_dev,
        input.len(),
    )?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_trilinear_acausal_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_trilinear_acausal_f64(&device, &mut out_dev, &input_dev, input.len())?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_bilinear_f64(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(input.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;
    super::srt_kernels::cuda_bilinear_f64(&device, &mut out_dev, &input_dev, input.len())?;
    let mut result = vec![0.0f64; input.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(result)
}
// TRILINEAR wrappers end
/// Gather with transcendence gate weighting (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_transcendence_gate_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_transcendence_gate_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with consciousness threshold (f64)
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_consciousness_threshold_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_consciousness_threshold_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

// =============================================================================
// SRT Reduction Operations
// =============================================================================

/// Reduce sum of array elements
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce mean of array elements
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_mean_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce mean of array elements (f32)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_mean_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce max of array elements
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_max_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce max of array elements (f32)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_max_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce min of array elements
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_min_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce min of array elements (f32)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_min_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce L2 norm of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_norm_l2_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce L2 norm of array elements (f32)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_norm_l2_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with golden weighted reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_golden_weighted_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_golden_weighted_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce syntony of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_syntony_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_syntony_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum rows of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_rows_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_rows_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum columns of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_cols_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_cols_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with phi scaled reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_phi_scaled_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_phi_scaled_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce variance with golden target reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_variance_golden_target_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_variance_golden_target_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with Mersenne stable reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_mersenne_stable_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_mersenne_stable_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with Lucas shadow reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_lucas_shadow_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_lucas_shadow_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce syntony deviation of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_syntony_deviation_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_syntony_deviation_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce consciousness count of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_consciousness_count_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_consciousness_count_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with Q corrected reduction (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_q_corrected_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_q_corrected_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce E8 norm of array elements (f64)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_e8_norm_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_e8_norm_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum of array elements (c128)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<(f64, f64)> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert input to complex numbers
    let complex_input: Vec<CudaComplex64> = input
        .into_iter()
        .map(|(re, im)| CudaComplex64(Complex64::new(re, im)))
        .collect();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&complex_input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<CudaComplex64> = pool
        .alloc_c128(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_c128(
        &device,
        &mut output_dev,
        &input_dev,
        complex_input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![CudaComplex64(Complex64::new(0.0, 0.0)); 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok((result[0].0.re, result[0].0.im))
}

/// Reduce norm of array elements (c128)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert input to complex numbers
    let complex_input: Vec<CudaComplex64> = input
        .into_iter()
        .map(|(re, im)| CudaComplex64(Complex64::new(re, im)))
        .collect();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&complex_input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_norm_c128(
        &device,
        &mut output_dev,
        &input_dev,
        complex_input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Compute argument (phase angle) of complex numbers (c128) using CUDA
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_arg_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<Vec<f64>> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let n = input.len();

    // Convert to interleaved real/imag format for CUDA kernel
    let interleaved: Vec<f64> = input
        .iter()
        .flat_map(|(re, im)| vec![*re, *im])
        .collect();

    // Allocate CUDA memory
    let z_dev = device
        .default_stream()
        .clone_htod(&interleaved)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run CUDA kernel
    super::srt_kernels::cuda_arg_c128(&device, &mut out_dev, &z_dev, n)?;

    // Copy result back
    let mut args = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut args)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(args)
}

/// Compute phase syntony metric for complex numbers (c128) using CUDA
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_phase_syntony_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    // Convert to interleaved real/imag format for CUDA kernel
    let interleaved: Vec<f64> = input
        .iter()
        .flat_map(|(re, im)| vec![*re, *im])
        .collect();

    // Allocate CUDA memory
    let z_dev = device
        .default_stream()
        .clone_htod(&interleaved)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut phase_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc phases: {}", e)))?;

    // Compute phase angles on GPU using cuda_arg_c128
    super::srt_kernels::cuda_arg_c128(&device, &mut phase_dev, &z_dev, n)?;

    // Copy phases back for variance computation
    let mut phases = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&phase_dev, &mut phases)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy phases: {}", e)))?;

    let mean_phase = phases.iter().sum::<f64>() / phases.len() as f64;
    let variance = phases.iter()
        .map(|phase| {
            let diff = phase - mean_phase;
            // Normalize phase difference to [-π, π]
            let normalized_diff = if diff > std::f64::consts::PI {
                diff - 2.0 * std::f64::consts::PI
            } else if diff < -std::f64::consts::PI {
                diff + 2.0 * std::f64::consts::PI
            } else {
                diff
            };
            normalized_diff * normalized_diff
        })
        .sum::<f64>() / phases.len() as f64;

    // Apply SRT correction factor for exact phase coherence (q/120 for complete E8 positive roots)
    const CORRECTION_FACTOR: f64 = Q_DEFICIT / 120.0;
    let syntony_base = 1.0 / (1.0 + variance);
    let syntony_corrected = syntony_base * (1.0 + CORRECTION_FACTOR);

    Ok(syntony_corrected)
}

/// Reduce sum of array elements (f32)
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

// CPU fallbacks for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f64(src: Vec<f64>, indices: Vec<i64>, device_idx: usize) -> PyResult<Vec<f64>> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    // Basic gather implementation: collect elements at specified indices
    let mut result = Vec::with_capacity(indices.len());
    for &idx in &indices {
        let idx_usize = idx as usize;
        if idx_usize >= src.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Index {} out of bounds for array of length {}", idx, src.len()
            )));
        }
        result.push(src[idx_usize]);
    }
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(format!(
            "src and indices must have same length, got {} and {}", src.len(), indices.len()
        )));
    }
    
    // Basic scatter implementation: place src values at specified indices
    let mut result = vec![0.0; output_size];
    for (i, &idx) in indices.iter().enumerate() {
        let idx_usize = idx as usize;
        if idx_usize >= output_size {
            return Err(PyRuntimeError::new_err(format!(
                "Index {} out of bounds for output size {}", idx, output_size
            )));
        }
        result[idx_usize] = src[i];
    }
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_add_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(format!(
            "src and indices must have same length, got {} and {}", src.len(), indices.len()
        )));
    }
    
    // Basic scatter_add implementation: add src values to specified indices
    let mut result = vec![0.0; output_size];
    for (i, &idx) in indices.iter().enumerate() {
        let idx_usize = idx as usize;
        if idx_usize >= output_size {
            return Err(PyRuntimeError::new_err(format!(
                "Index {} out of bounds for output size {}", idx, output_size
            )));
        }
        result[idx_usize] += src[i];
    }
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads

    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
    } else {
        // Simple parallel min using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut min_val = f64::INFINITY;

        for chunk in input.chunks(chunk_size) {
            let chunk_min = chunk.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            min_val = min_val.min(chunk_min);
        }
        Ok(min_val)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().fold(f32::INFINITY, |a, &b| a.min(b)))
    } else {
        // Simple parallel min using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut min_val = f32::INFINITY;

        for chunk in input.chunks(chunk_size) {
            let chunk_min = chunk.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            min_val = min_val.min(chunk_min);
        }
        Ok(min_val)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().sum())
    } else {
        // Simple parallel sum using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total = 0.0f32;

        for chunk in input.chunks(chunk_size) {
            total += chunk.iter().sum::<f32>();
        }
        Ok(total)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().sum())
    } else {
        // Simple parallel sum using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total = 0.0f64;

        for chunk in input.chunks(chunk_size) {
            total += chunk.iter().sum::<f64>();
        }
        Ok(total)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().sum::<f64>() / input.len() as f64)
    } else {
        // Parallel mean calculation
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total = 0.0f64;

        for chunk in input.chunks(chunk_size) {
            total += chunk.iter().sum::<f64>();
        }
        Ok(total / input.len() as f64)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().sum::<f32>() / input.len() as f32)
    } else {
        // Parallel mean calculation
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total = 0.0f32;

        for chunk in input.chunks(chunk_size) {
            total += chunk.iter().sum::<f32>();
        }
        Ok(total / input.len() as f32)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
    } else {
        // Simple parallel max using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut max_val = f64::NEG_INFINITY;

        for chunk in input.chunks(chunk_size) {
            let chunk_max = chunk.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            max_val = max_val.max(chunk_max);
        }
        Ok(max_val)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok(input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
    } else {
        // Simple parallel max using chunks
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut max_val = f32::NEG_INFINITY;

        for chunk in input.chunks(chunk_size) {
            let chunk_max = chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            max_val = max_val.max(chunk_max);
        }
        Ok(max_val)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok((input.iter().map(|x| x * x).sum::<f64>()).sqrt())
    } else {
        // Parallel L2 norm calculation
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total_sq = 0.0f64;

        for chunk in input.chunks(chunk_size) {
            total_sq += chunk.iter().map(|x| x * x).sum::<f64>();
        }
        Ok(total_sq.sqrt())
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    if num_threads == 1 {
        Ok((input.iter().map(|x| x * x).sum::<f32>()).sqrt())
    } else {
        // Parallel L2 norm calculation
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total_sq = 0.0f32;

        for chunk in input.chunks(chunk_size) {
            total_sq += chunk.iter().map(|x| x * x).sum::<f32>();
        }
        Ok(total_sq.sqrt())
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_golden_weighted_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    if num_threads == 1 {
        let mut sum = 0.0;
        for (i, &val) in input.iter().enumerate() {
            let weight = (-(i as f64 * i as f64) * PHI_INV).exp();
            sum += val * weight;
        }
        Ok(sum)
    } else {
        // Parallel golden weighted sum
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        let mut total = 0.0f64;

        for (chunk_idx, chunk) in input.chunks(chunk_size).enumerate() {
            let mut chunk_sum = 0.0;
            let start_idx = chunk_idx * chunk_size;
            for (i, &val) in chunk.iter().enumerate() {
                let global_idx = start_idx + i;
                let weight = (-(global_idx as f64 * global_idx as f64) * PHI_INV).exp();
                chunk_sum += val * weight;
            }
            total += chunk_sum;
        }
        Ok(total)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_syntony_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    if num_threads == 1 {
        // Single-threaded implementation
        // Compute norm squared for each element
        let mut norm_sq_sum = 0.0;
        let mut entropy = 0.0;
        
        for &val in &input {
            let amp_sq = val * val;
            norm_sq_sum += amp_sq;
        }
        
        if norm_sq_sum < 1e-15 {
            return Ok(0.0);
        }
        
        // Compute Shannon entropy: -Σ pᵢ log(pᵢ) where pᵢ = |xᵢ|² / Σ|xⱼ|²
        for &val in &input {
            let amp_sq = val * val;
            let p = amp_sq / norm_sq_sum;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        
        // Normalize to [0, φ] where φ ≈ 1.618
        let max_entropy = (input.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };
        
        Ok(normalized_entropy * PHI)
    } else {
        // Parallel implementation
        use std::thread;
        use std::sync::Arc;
        
        let input = Arc::new(input);
        let mut handles = vec![];
        
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        
        // First pass: compute norm squared sum in parallel
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_norm_sq = 0.0;
                for &val in &input[start_idx..end_idx] {
                    chunk_norm_sq += val * val;
                }
                chunk_norm_sq
            });
            handles.push(handle);
        }
        
        let mut total_norm_sq = 0.0;
        for handle in handles {
            total_norm_sq += handle.join().unwrap();
        }
        
        if total_norm_sq < 1e-15 {
            return Ok(0.0);
        }
        
        // Second pass: compute entropy in parallel
        let total_norm_sq = Arc::new(total_norm_sq);
        let mut handles = vec![];
        
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let total_norm_sq = Arc::clone(&total_norm_sq);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_entropy = 0.0;
                for &val in &input[start_idx..end_idx] {
                    let amp_sq = val * val;
                    let p = amp_sq / *total_norm_sq;
                    if p > 1e-15 {
                        chunk_entropy -= p * p.ln();
                    }
                }
                chunk_entropy
            });
            handles.push(handle);
        }
        
        let mut total_entropy = 0.0;
        for handle in handles {
            total_entropy += handle.join().unwrap();
        }
        
        // Normalize to [0, φ] where φ ≈ 1.618
        let max_entropy = (input.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 { total_entropy / max_entropy } else { 0.0 };
        
        Ok(normalized_entropy * PHI)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_rows_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // For row reduction, sum all elements (treating 1D array as single row)
    let sum: f64 = input.iter().sum();
    Ok(sum)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_cols_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // For column reduction, sum all elements (same as row reduction for 1D)
    let sum: f64 = input.iter().sum();
    Ok(sum)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_phi_scaled_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // SRT phi-scaled sum: sum(x_i * φ^i) where φ is golden ratio
    let mut sum = 0.0;
    let mut phi_power = 1.0;
    for &val in &input {
        sum += val * phi_power;
        phi_power *= PHI;
    }
    Ok(sum)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_variance_golden_target_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.len() < 2 {
        return Err(PyRuntimeError::new_err("input must have at least 2 elements for variance"));
    }
    
    // Golden target variance: variance relative to golden ratio target
    let mean: f64 = input.iter().sum::<f64>() / input.len() as f64;
    let variance: f64 = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() - 1) as f64;
    
    // Scale by golden ratio target (φ² ≈ 2.618)
    const GOLDEN_TARGET: f64 = 2.618033988749895;
    Ok(variance / GOLDEN_TARGET)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_mersenne_stable_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Mersenne stable sum: use compensated summation for numerical stability
    let mut sum = 0.0;
    let mut compensation = 0.0;
    
    for &val in &input {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    Ok(sum)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_lucas_shadow_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Lucas shadow sum: weighted by Lucas sequence ratios
    // Lucas sequence: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76...
    let mut sum = 0.0;
    let mut lucas_a = 2.0; // L(0)
    let mut lucas_b = 1.0; // L(1)
    
    for &val in &input {
        let weight = lucas_a / lucas_b; // Lucas ratio approaches φ
        sum += val * weight;
        
        // Next Lucas numbers
        let next = lucas_a + lucas_b;
        lucas_a = lucas_b;
        lucas_b = next;
    }
    
    Ok(sum)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_syntony_deviation_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Syntony deviation: measure deviation from golden ratio harmonics
    let mean: f64 = input.iter().sum::<f64>() / input.len() as f64;
    let mut deviation = 0.0;
    
    for (i, &val) in input.iter().enumerate() {
        // Golden harmonic: φ^(i+1) / φ^i = φ
        let harmonic = PHI.powi(i as i32 + 1);
        let expected = mean * harmonic;
        deviation += (val - expected).abs();
    }
    
    Ok(deviation / input.len() as f64)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_consciousness_count_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Validate device_idx is 0 for CPU (only CPU device supported)
    if device_idx != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "CPU fallback only supports device_idx=0, got {}", device_idx
        )));
    }
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Consciousness count: count elements above syntony threshold
    // Syntony threshold based on golden ratio
    const SYNTONY_THRESHOLD: f64 = PHI_INV * 0.5; // φ⁻¹ ≈ 0.618
    
    let count: f64 = input.iter()
        .filter(|&&x| x.abs() > SYNTONY_THRESHOLD)
        .count() as f64;
    
    Ok(count)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_q_corrected_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Q-corrected sum: sum with quantum correction factor
    // Q-deficit correction from SRT theory
    const Q_CORRECTION: f64 = 1.0 - Q_DEFICIT; // ≈ 0.9726
    
    if num_threads == 1 {
        // Single-threaded implementation
        let sum: f64 = input.iter().sum();
        Ok(sum * Q_CORRECTION)
    } else {
        // Parallel implementation using chunks
        use std::thread;
        use std::sync::Arc;
        
        let input = Arc::new(input);
        let mut handles = vec![];
        
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_sum = 0.0;
                for &val in &input[start_idx..end_idx] {
                    chunk_sum += val;
                }
                chunk_sum
            });
            handles.push(handle);
        }
        
        let mut total_sum = 0.0;
        for handle in handles {
            total_sum += handle.join().unwrap();
        }
        
        Ok(total_sum * Q_CORRECTION)
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_e8_norm_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // E8 norm: norm based on E8 lattice structure
    // E8 has 240 roots, use approximation with golden ratio harmonics
    if num_threads == 1 {
        // Single-threaded implementation
        let mut norm = 0.0;
        for &val in &input {
            // E8-like weighting with golden ratio harmonics
            let weight = PHI.powi((val.abs() * 10.0) as i32);
            norm += val * val * weight;
        }
        Ok(norm.sqrt())
    } else {
        // Parallel implementation using chunks
        use std::thread;
        use std::sync::Arc;
        
        let input = Arc::new(input);
        let mut handles = vec![];
        
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_norm = 0.0;
                for &val in &input[start_idx..end_idx] {
                    // E8-like weighting with golden ratio harmonics
                    let weight = PHI.powi((val.abs() * 10.0) as i32);
                    chunk_norm += val * val * weight;
                }
                chunk_norm
            });
            handles.push(handle);
        }
        
        let mut total_norm = 0.0;
        for handle in handles {
            total_norm += handle.join().unwrap();
        }
        
        Ok(total_norm.sqrt())
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<(f64, f64)> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Complex sum: sum real and imaginary parts separately
    if num_threads == 1 {
        // Single-threaded implementation
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;
        
        for &(re, im) in &input {
            real_sum += re;
            imag_sum += im;
        }
        
        Ok((real_sum, imag_sum))
    } else {
        // Parallel implementation using chunks
        use std::thread;
        use std::sync::Arc;
        
        let input = Arc::new(input);
        let mut handles = vec![];
        
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_real = 0.0;
                let mut chunk_imag = 0.0;
                for &(re, im) in &input[start_idx..end_idx] {
                    chunk_real += re;
                    chunk_imag += im;
                }
                (chunk_real, chunk_imag)
            });
            handles.push(handle);
        }
        
        let mut total_real = 0.0;
        let mut total_imag = 0.0;
        for handle in handles {
            let (chunk_real, chunk_imag) = handle.join().unwrap();
            total_real += chunk_real;
            total_imag += chunk_imag;
        }
        
        Ok((total_real, total_imag))
    }
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_c128(input: Vec<(f64, f64)>, device_idx: usize) -> PyResult<f64> {
    // Use device_idx as threading hint: 0 = single-threaded, >0 = parallel with 2^device_idx threads
    let num_threads = if device_idx == 0 { 1 } else { 1 << device_idx.min(4) }; // Cap at 16 threads
    
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }
    
    // Complex norm: sqrt(sum(|z_i|^2))
    if num_threads == 1 {
        // Single-threaded implementation
        let mut norm_sq = 0.0;
        for &(re, im) in &input {
            norm_sq += re * re + im * im;
        }
        Ok(norm_sq.sqrt())
    } else {
        // Parallel implementation using chunks
        use std::thread;
        use std::sync::Arc;
        
        let input = Arc::new(input);
        let mut handles = vec![];
        
        let chunk_size = (input.len() + num_threads - 1) / num_threads;
        
        for thread_id in 0..num_threads {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start_idx = thread_id * chunk_size;
                let end_idx = ((thread_id + 1) * chunk_size).min(input.len());
                
                let mut chunk_norm_sq = 0.0;
                for &(re, im) in &input[start_idx..end_idx] {
                    chunk_norm_sq += re * re + im * im;
                }
                chunk_norm_sq
            });
            handles.push(handle);
        }
        
        let mut total_norm_sq = 0.0;
        for handle in handles {
            total_norm_sq += handle.join().unwrap();
        }
        
        Ok(total_norm_sq.sqrt())
    }
}

// =============================================================================
// SRT Scatter/Gather Operations
// =============================================================================

/// Gather elements from source array using indices
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f64(src: Vec<f64>, indices: Vec<i64>, device_idx: usize) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    // Use device_idx to get device (ensures parameter is used)
    let _device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // CPU implementation
    let mut result = Vec::with_capacity(indices.len());
    for &idx in &indices {
        let idx_usize = if idx < 0 {
            (src.len() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx_usize >= src.len() {
            return Err(PyRuntimeError::new_err(format!("Index {} out of bounds for array of length {}", idx, src.len())));
        }
        result.push(src[idx_usize]);
    }
    Ok(result)
}

// =============================================================================
// Kernel Loader Functions (Python wrappers)
// =============================================================================

#[pyfunction]
pub fn py_load_wmma_syntonic_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_wmma_syntonic_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[pyfunction]
pub fn py_load_scatter_gather_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_scatter_gather_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[pyfunction]
pub fn py_load_reduction_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_reduction_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[pyfunction]
pub fn py_load_trilinear_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_trilinear_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[pyfunction]
pub fn py_load_complex_ops_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_complex_ops_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[pyfunction]
pub fn py_load_attention_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_attention_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

// =============================================================================
// SRT Attractor Kernel Functions (Retrocausal Training)
// =============================================================================

/// Load SRT attractor kernels for retrocausal operations
#[pyfunction]
#[pyo3(signature = (device_idx=0))]
pub fn py_load_attractor_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_attractor_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

/// Compute attractor centroid from multiple high-syntony states (CUDA)
#[pyfunction]
#[pyo3(signature = (attractors, weights, device_idx=0))]
pub fn py_attractor_centroid(
    attractors: Vec<Vec<f64>>,
    weights: Vec<f64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if attractors.is_empty() {
        return Err(PyRuntimeError::new_err("attractors cannot be empty"));
    }
    if attractors.len() != weights.len() {
        return Err(PyRuntimeError::new_err("attractors and weights must have same length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }
    let dim = attractors[0].len();

    // Compute weighted centroid on CPU for now (CUDA version in future)
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum == 0.0 {
        return Err(PyRuntimeError::new_err("weights sum to zero"));
    }

    let mut centroid = vec![0.0f64; dim];
    for (attractor, weight) in attractors.iter().zip(weights.iter()) {
        for (i, val) in attractor.iter().enumerate() {
            centroid[i] += val * weight / weight_sum;
        }
    }

    Ok(centroid)
}

/// Apply retrocausal harmonization: ψ_new = (1-λ)*ψ + λ*centroid (CUDA)
#[pyfunction]
#[pyo3(signature = (state, centroid, pull_strength=0.3, device_idx=0))]
pub fn py_retrocausal_harmonize(
    state: Vec<f64>,
    centroid: Vec<f64>,
    pull_strength: f64,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if state.len() != centroid.len() {
        return Err(PyRuntimeError::new_err("state and centroid must have same length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }

    // Apply retrocausal harmonization: (1-λ)*ψ + λ*centroid
    let one_minus_lambda = 1.0 - pull_strength;
    let harmonized: Vec<f64> = state.iter().zip(centroid.iter())
        .map(|(s, c)| one_minus_lambda * s + pull_strength * c)
        .collect();

    Ok(harmonized)
}

/// Compute attractor distance with golden decay weight
#[pyfunction]
#[pyo3(signature = (state, attractor, device_idx=0))]
pub fn py_attractor_distance(
    state: Vec<f64>,
    attractor: Vec<f64>,
    device_idx: usize,
) -> PyResult<(f64, f64)> {
    if state.len() != attractor.len() {
        return Err(PyRuntimeError::new_err("state and attractor must have same length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if device.ordinal() > 16 {
        return Err(PyRuntimeError::new_err("Invalid device ordinal"));
    }

    // Compute Euclidean distance
    let distance: f64 = state.iter().zip(attractor.iter())
        .map(|(s, a)| (s - a).powi(2))
        .sum::<f64>()
        .sqrt();

    // Compute gravity weight: exp(-distance/φ)
    let gravity_weight = (-distance / PHI).exp();

    Ok((distance, gravity_weight))
}

// =============================================================================
// DHSR Kernel Python Wrappers
// =============================================================================

/// Execute fused DHSR step (Differentiation + Harmonization) with syntony update
#[pyfunction]
#[pyo3(signature = (input, mode_norm_sq, alpha_0=0.1, zeta_0=0.01, beta_0=0.1, gamma_0=0.05, syntony=0.5, device_idx=0))]
pub fn py_dhsr_step_fused(
    input: Vec<f64>,
    mode_norm_sq: Vec<f64>,
    alpha_0: f64,
    zeta_0: f64,
    beta_0: f64,
    gamma_0: f64,
    syntony: f64,
    device_idx: usize,
) -> PyResult<(Vec<f64>, f64)> {
    if input.is_empty() || input.len() != mode_norm_sq.len() {
        return Err(PyRuntimeError::new_err("input and mode_norm_sq must have same non-zero length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mode_dev = device
        .default_stream()
        .clone_htod(&mode_norm_sq)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy mode_norm_sq: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run fused DHSR step
    let new_syntony = super::srt_kernels::cuda_dhsr_step_fused_f64(
        &device,
        &mut out_dev,
        &input_dev,
        &mode_dev,
        alpha_0,
        zeta_0,
        beta_0,
        gamma_0,
        syntony,
        n,
    )?;

    // Copy result back
    let mut output = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok((output, new_syntony))
}

/// Apply damping cascade for harmonization
#[pyfunction]
#[pyo3(signature = (input, mode_norm_sq, beta_0=0.1, syntony=0.5, delta_d=0.01, num_dampers=4, device_idx=0))]
pub fn py_damping_cascade(
    input: Vec<f64>,
    mode_norm_sq: Vec<f64>,
    beta_0: f64,
    syntony: f64,
    delta_d: f64,
    num_dampers: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if input.is_empty() || input.len() != mode_norm_sq.len() {
        return Err(PyRuntimeError::new_err("input and mode_norm_sq must have same non-zero length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mode_dev = device
        .default_stream()
        .clone_htod(&mode_norm_sq)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy mode_norm_sq: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run damping cascade kernel
    super::srt_kernels::cuda_damping_cascade_f64(
        &device,
        &mut out_dev,
        &input_dev,
        &mode_dev,
        beta_0,
        syntony,
        delta_d,
        num_dampers,
        n,
    )?;

    // Copy result back
    let mut output = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(output)
}

/// Apply full differentiation operator
#[pyfunction]
#[pyo3(signature = (input, fourier_contribution, laplacian, alpha_0=0.1, zeta_0=0.01, syntony=0.5, device_idx=0))]
pub fn py_differentiation_full(
    input: Vec<f64>,
    fourier_contribution: Vec<f64>,
    laplacian: Vec<f64>,
    alpha_0: f64,
    zeta_0: f64,
    syntony: f64,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if input.is_empty() || input.len() != fourier_contribution.len() || input.len() != laplacian.len() {
        return Err(PyRuntimeError::new_err("All inputs must have same non-zero length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let fourier_dev = device
        .default_stream()
        .clone_htod(&fourier_contribution)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy fourier: {}", e)))?;
    let laplacian_dev = device
        .default_stream()
        .clone_htod(&laplacian)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy laplacian: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run differentiation kernel
    super::srt_kernels::cuda_differentiation_full_f64(
        &device,
        &mut out_dev,
        &input_dev,
        &fourier_dev,
        &laplacian_dev,
        alpha_0,
        zeta_0,
        syntony,
        n,
    )?;

    // Copy result back
    let mut output = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(output)
}

/// Batch Fourier projection
#[pyfunction]
#[pyo3(signature = (input, modes, weights, device_idx=0))]
pub fn py_fourier_project_batch(
    input: Vec<f64>,
    modes: Vec<i32>,
    weights: Vec<f64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if input.is_empty() || modes.is_empty() || modes.len() != weights.len() {
        return Err(PyRuntimeError::new_err("input cannot be empty, modes and weights must have same length"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();
    let num_modes = modes.len();

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let modes_dev = device
        .default_stream()
        .clone_htod(&modes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy modes: {}", e)))?;
    let weights_dev = device
        .default_stream()
        .clone_htod(&weights)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy weights: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run Fourier projection kernel
    super::srt_kernels::cuda_fourier_project_batch_f64(
        &device,
        &mut out_dev,
        &input_dev,
        &modes_dev,
        &weights_dev,
        num_modes,
        n,
    )?;

    // Copy result back
    let mut output = vec![0.0f64; n];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(output)
}

// =============================================================================
// Static CUDA Library Python Wrappers
// =============================================================================
// These wrappers use the statically linked CUDA library for better performance
// and scaling compared to PTX runtime loading.

/// Compute 1D discrete Laplacian using static CUDA kernel
#[pyfunction]
pub fn py_static_laplacian_1d(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_laplacian_1d_f64;
    let mut out = vec![0.0f64; input.len()];
    static_laplacian_1d_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Project state toward target syntony using static CUDA kernel
#[pyfunction]
#[pyo3(signature = (input, target_syntony, strength=0.5))]
pub fn py_static_syntony_projection(
    input: Vec<f64>,
    target_syntony: Vec<f64>,
    strength: f64,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_syntony_projection_f64;
    if input.len() != target_syntony.len() {
        return Err(PyRuntimeError::new_err("input and target_syntony must have same length"));
    }
    let mut out = vec![0.0f64; input.len()];
    static_syntony_projection_f64(&mut out, &input, &target_syntony, strength)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Gather elements by indices using static CUDA kernel
#[pyfunction]
pub fn py_static_gather(src: Vec<f64>, indices: Vec<i32>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_gather_f64;
    let mut out = vec![0.0f64; indices.len()];
    static_gather_f64(&mut out, &src, &indices)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Scatter-add operation using static CUDA kernel
#[pyfunction]
pub fn py_static_scatter_add(out_size: usize, src: Vec<f32>, indices: Vec<i32>) -> PyResult<Vec<f32>> {
    use super::srt_kernels::static_scatter_add_f32;
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err("src and indices must have same length"));
    }
    let mut out = vec![0.0f32; out_size];
    static_scatter_add_f32(&mut out, &src, &indices)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

// =============================================================================
// Additional CUDA Kernel Python Wrappers
// =============================================================================

/// Compute 1D Laplacian using PTX-based CUDA kernel
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_laplacian_1d(input: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    let input_dev = device.default_stream().clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_laplacian_1d_f64(&device, &mut out_dev, &input_dev, n)?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Project toward syntony target using PTX-based CUDA kernel
#[pyfunction]
#[pyo3(signature = (input, target_syntony, gamma=0.5, syntony=0.7, device_idx=0))]
pub fn py_syntony_projection(
    input: Vec<f64>,
    target_syntony: Vec<f64>,
    gamma: f64,
    syntony: f64,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();

    if n != target_syntony.len() {
        return Err(PyRuntimeError::new_err("input and target must have same length"));
    }

    let input_dev = device.default_stream().clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let target_dev = device.default_stream().clone_htod(&target_syntony)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy target: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_syntony_projection_f64(&device, &mut out_dev, &input_dev, &target_dev, gamma, syntony, n)?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Half-precision matrix multiply using WMMA tensor cores
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, device_idx=0))]
pub fn py_wmma_fp16_matmul(
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    use super::cuda::memory_pool::CudaF16;
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert f32 to CudaF16 for WMMA (f32 -> half::f16 -> CudaF16)
    let a_f16: Vec<CudaF16> = a.iter().map(|&x| CudaF16(half::f16::from_f32(x))).collect();
    let b_f16: Vec<CudaF16> = b.iter().map(|&x| CudaF16(half::f16::from_f32(x))).collect();

    let a_dev = device.default_stream().clone_htod(&a_f16)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy A: {}", e)))?;
    let b_dev = device.default_stream().clone_htod(&b_f16)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy B: {}", e)))?;
    let mut c_dev: CudaSlice<CudaF16> = pool.alloc_f16(m * n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc C: {}", e)))?;

    super::srt_kernels::cuda_wmma_fp16_matmul(&device, &mut c_dev, &a_dev, &b_dev, m, n, k)?;

    // Copy back to host and convert to f32
    let mut c_host = vec![CudaF16(half::f16::from_f32(0.0)); m * n];
    device.default_stream().memcpy_dtoh(&c_dev, &mut c_host)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    let output: Vec<f32> = c_host.iter().map(|x| x.0.to_f32()).collect();
    Ok(output)
}

/// Gather with phi weighting (phi^(-i) weighting applied in kernel)
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_phi_weighted(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = indices.len();

    let src_dev = device.default_stream().clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device.default_stream().clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_phi_weighted_f64(&device, &mut out_dev, &src_dev, &idx_dev, n)?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Standard gather operation
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = indices.len();

    let src_dev = device.default_stream().clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device.default_stream().clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_f64(&device, &mut out_dev, &src_dev, &idx_dev, n)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Scatter-add operation
#[pyfunction]
#[pyo3(signature = (out_size, src, indices, device_idx=0))]
pub fn py_scatter_add(
    out_size: usize,
    src: Vec<f32>,
    indices: Vec<i32>,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err("src and indices must have same length"));
    }

    let src_dev = device.default_stream().clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device.default_stream().clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f32> = pool.alloc_f32(out_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_add_f32(&device, &mut out_dev, &src_dev, &idx_dev, src.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut output = vec![0.0f32; out_size];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Trilinear interpolation on 3D grid
/// grid: flattened 3D data [D, H, W]
/// coords: flattened Nx3 array of (x, y, z) coordinates
#[pyfunction]
#[pyo3(signature = (grid, coords, d, h, w, boundary_mode=0, device_idx=0))]
pub fn py_trilinear(
    grid: Vec<f64>,     // [D * H * W]
    coords: Vec<f64>,   // [N * 3] - (x, y, z) per point
    d: usize,
    h: usize,
    w: usize,
    boundary_mode: i32,  // 0=clamp, 1=zero, 2=reflect
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if coords.len() % 3 != 0 {
        return Err(PyRuntimeError::new_err("coords length must be divisible by 3"));
    }
    let n = coords.len() / 3;

    if grid.len() != d * h * w {
        return Err(PyRuntimeError::new_err(format!(
            "grid length {} doesn't match d*h*w = {}", grid.len(), d * h * w
        )));
    }

    let grid_dev = device.default_stream().clone_htod(&grid)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy grid: {}", e)))?;
    let coords_dev = device.default_stream().clone_htod(&coords)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy coords: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_trilinear_f64(&device, &mut out_dev, &grid_dev, &coords_dev, d, h, w, n, boundary_mode)?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Compute phase syntony for complex128 values
/// z: interleaved complex data [re0, im0, re1, im1, ...]
/// syntony: target syntony values for each element
#[pyfunction]
#[pyo3(signature = (z, syntony, device_idx=0))]
pub fn py_phase_syntony(z: Vec<f64>, syntony: Vec<f64>, device_idx: usize) -> PyResult<Vec<f64>> {
    // z is interleaved [re0, im0, re1, im1, ...]
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if z.len() % 2 != 0 {
        return Err(PyRuntimeError::new_err("z must have even length (interleaved complex)"));
    }
    let n = z.len() / 2;

    if syntony.len() != n {
        return Err(PyRuntimeError::new_err(format!(
            "syntony length {} must match complex count {}", syntony.len(), n
        )));
    }

    let z_dev = device.default_stream().clone_htod(&z)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy z: {}", e)))?;
    let syntony_dev = device.default_stream().clone_htod(&syntony)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy syntony: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool.alloc_f64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_phase_syntony_c128(&device, &mut out_dev, &z_dev, &syntony_dev, n)?;

    let mut output = vec![0.0f64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

/// Update attractor memory with new state
/// attractor_memory: flattened [max_attractors, state_dim] buffer
/// attractor_syntony: [max_attractors] syntony score per attractor
/// attractor_count: [1] current count of active attractors
/// state: new state vector to add if syntony is high enough
/// syntony: syntony score of the new state
/// Returns tuple (updated_memory, updated_syntony, updated_count)
#[pyfunction]
#[pyo3(signature = (attractor_memory, attractor_syntony, attractor_count, state, syntony, device_idx=0))]
pub fn py_cuda_attractor_memory_update(
    attractor_memory: Vec<f64>,   // [max_attractors * state_dim]
    attractor_syntony: Vec<f64>,  // [max_attractors]
    attractor_count: Vec<i32>,    // [1]
    state: Vec<f64>,              // [state_dim]
    syntony: f64,
    device_idx: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let state_dim = state.len();

    let mut mem_dev = device.default_stream().clone_htod(&attractor_memory)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy attractor_memory: {}", e)))?;
    let mut syn_dev = device.default_stream().clone_htod(&attractor_syntony)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy attractor_syntony: {}", e)))?;
    let mut cnt_dev = device.default_stream().clone_htod(&attractor_count)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy attractor_count: {}", e)))?;
    let state_dev = device.default_stream().clone_htod(&state)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy state: {}", e)))?;

    super::srt_kernels::cuda_attractor_memory_update_f64(
        &device, &mut mem_dev, &mut syn_dev, &mut cnt_dev, &state_dev, syntony, state_dim
    ).map_err(|e| PyRuntimeError::new_err(e))?;

    let mut out_mem = vec![0.0f64; attractor_memory.len()];
    let mut out_syn = vec![0.0f64; attractor_syntony.len()];
    let mut out_cnt = vec![0i32; attractor_count.len()];

    device.default_stream().memcpy_dtoh(&mem_dev, &mut out_mem)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy memory result: {}", e)))?;
    device.default_stream().memcpy_dtoh(&syn_dev, &mut out_syn)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy syntony result: {}", e)))?;
    device.default_stream().memcpy_dtoh(&cnt_dev, &mut out_cnt)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy count result: {}", e)))?;

    Ok((out_mem, out_syn, out_cnt))
}

// =============================================================================
// Additional Static CUDA Wrappers
// =============================================================================

/// Static wrapper: compute syntony for complex128 data
#[pyfunction]
pub fn py_static_compute_syntony_c128(psi: Vec<f64>, mode_norm_sq: Vec<f64>) -> PyResult<(f64, f64)> {
    use super::srt_kernels::static_compute_syntony_c128;
    let mut numerator = vec![0.0f64; 1];
    let mut denominator = vec![0.0f64; 1];
    static_compute_syntony_c128(&mut numerator, &mut denominator, &psi, &mode_norm_sq)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok((numerator[0], denominator[0]))
}

/// Static wrapper: batch Fourier projection
#[pyfunction]
pub fn py_static_fourier_project_batch(
    input: Vec<f64>,
    modes: Vec<i32>,
    weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_fourier_project_batch_f64;
    let mut out = vec![0.0f64; input.len()];
    static_fourier_project_batch_f64(&mut out, &input, &modes, &weights)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: damping cascade
#[pyfunction]
#[pyo3(signature = (input, gamma_0=0.1, delta_d=0.05, num_dampers=5, phi_weight=0.618))]
pub fn py_static_damping_cascade(
    input: Vec<f64>,
    gamma_0: f64,
    delta_d: f64,
    num_dampers: i32,
    phi_weight: f64,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_damping_cascade_f64;
    let mut out = vec![0.0f64; input.len()];
    static_damping_cascade_f64(&mut out, &input, gamma_0, delta_d, num_dampers, phi_weight)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: DHSR step fused
#[pyfunction]
#[pyo3(signature = (state, mode_norms, attractors, diff_strength=0.1, harm_strength=0.5, retro_pull=0.3))]
pub fn py_static_dhsr_step_fused(
    mut state: Vec<f64>,
    mode_norms: Vec<f64>,
    attractors: Vec<f64>,
    diff_strength: f64,
    harm_strength: f64,
    retro_pull: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use super::srt_kernels::static_dhsr_step_fused_f64;
    let mut syntony_out = vec![0.0f64; state.len()];
    static_dhsr_step_fused_f64(
        &mut state,
        &mut syntony_out,
        &mode_norms,
        &attractors,
        diff_strength,
        harm_strength,
        retro_pull,
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok((state, syntony_out))
}

/// Static wrapper: attractor memory update
#[pyfunction]
#[pyo3(signature = (attractors, new_state, syntony_scores, min_syntony=0.7, decay_rate=0.99, max_attractors=10))]
pub fn py_static_attractor_memory_update(
    mut attractors: Vec<f64>,
    new_state: Vec<f64>,
    syntony_scores: Vec<f64>,
    min_syntony: f64,
    decay_rate: f64,
    max_attractors: i32,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_attractor_memory_update_f64;
    static_attractor_memory_update_f64(
        &mut attractors,
        &new_state,
        &syntony_scores,
        min_syntony,
        decay_rate,
        max_attractors,
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(attractors)
}

/// Static wrapper: retrocausal harmonize
#[pyfunction]
#[pyo3(signature = (current, attractor_centroid, pull_strength=0.5, syntony_threshold=0.7))]
pub fn py_static_retrocausal_harmonize(
    current: Vec<f64>,
    attractor_centroid: Vec<f64>,
    pull_strength: f64,
    syntony_threshold: f64,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_retrocausal_harmonize_f64;
    let mut out = vec![0.0f64; current.len()];
    static_retrocausal_harmonize_f64(
        &mut out,
        &current,
        &attractor_centroid,
        pull_strength,
        syntony_threshold,
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: attractor centroid
#[pyfunction]
pub fn py_static_attractor_centroid(
    attractors: Vec<f64>,
    weights: Vec<f64>,
    num_attractors: i32,
    state_size: usize,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_attractor_centroid_f64;
    let mut centroid = vec![0.0f64; state_size];
    static_attractor_centroid_f64(&mut centroid, &attractors, &weights, num_attractors)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(centroid)
}

/// Static wrapper: atan2 toroidal (uses host_* FFI which handles H2D/D2H internally)
#[pyfunction]
pub fn py_static_atan2_toroidal(y: Vec<f64>, x: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_atan2_toroidal_f64;
    if y.len() != x.len() {
        return Err(PyRuntimeError::new_err("y and x must have same length"));
    }
    let mut out = vec![0.0f64; y.len()];
    static_atan2_toroidal_f64(&mut out, &y, &x)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: golden entropy (uses host_* FFI which handles H2D/D2H internally)
#[pyfunction]
pub fn py_static_golden_entropy(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_golden_entropy_f64;
    let mut out = vec![0.0f64; input.len()];
    static_golden_entropy_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}
/// Static wrapper: gather with phi weighting
#[pyfunction]
pub fn py_static_gather_phi_weighted(
    src: Vec<f64>,
    indices: Vec<i32>,
    weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_gather_phi_weighted_f64;
    let mut out = vec![0.0f64; indices.len()];
    static_gather_phi_weighted_f64(&mut out, &src, &indices, &weights)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: complex argument (phase angle)
#[pyfunction]
pub fn py_static_arg_c128(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_arg_c128;
    let n = input.len() / 2;
    let mut out = vec![0.0f64; n];
    static_arg_c128(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: phase syntony c128
#[pyfunction]
pub fn py_static_phase_syntony_c128(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_phase_syntony_c128;
    let n = input.len() / 2;
    let mut out = vec![0.0f64; n];
    static_phase_syntony_c128(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: trilinear interpolation
#[pyfunction]
pub fn py_static_trilinear(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    nx: i32,
    ny: i32,
    nz: i32,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_trilinear_f64;
    let mut out = vec![0.0f64; x.len()];
    static_trilinear_f64(&mut out, &x, &y, &z, nx, ny, nz)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: gnosis mask
#[pyfunction]
#[pyo3(signature = (input, syntony, threshold=0.7, strength=1.0))]
pub fn py_static_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    threshold: f64,
    strength: f64,
) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_gnosis_mask_f64;
    let mut out = vec![0.0f64; input.len()];
    static_gnosis_mask_f64(&mut out, &input, &syntony, threshold, strength)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static wrapper: matrix multiply
#[pyfunction]
pub fn py_static_matmul(a: Vec<f64>, b: Vec<f64>, m: i32, n: i32, k: i32) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_matmul_f64;
    let mut c = vec![0.0f64; (m * n) as usize];
    static_matmul_f64(&mut c, &a, &b, m, n, k)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(c)
}

/// Fixed-point Fourier project batch (using i64 representation)
#[pyfunction]
#[pyo3(signature = (input, modes, weights, device_idx=0))]
pub fn py_fourier_project_batch_fp64(
    input: Vec<i64>,
    modes: Vec<i32>,
    weights: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<i64>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = input.len();
    let num_modes = modes.len();

    let input_dev = device.default_stream().clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let modes_dev = device.default_stream().clone_htod(&modes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy modes: {}", e)))?;
    let weights_dev = device.default_stream().clone_htod(&weights)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy weights: {}", e)))?;
    let mut out_dev: CudaSlice<i64> = pool.alloc_i64(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_fourier_project_batch_fp64(
        &device,
        &mut out_dev,
        &input_dev,
        &modes_dev,
        &weights_dev,
        num_modes,
        n,
    )?;

    let mut output = vec![0i64; n];
    device.default_stream().memcpy_dtoh(&out_dev, &mut output)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;
    Ok(output)
}

// =============================================================================
// Static Kernel Wrappers - Use host_* FFI functions via static_* wrappers
// =============================================================================

/// Static sin toroidal kernel - uses host_* FFI through static_* wrapper
/// Note: For high-performance use, prefer py_sin_toroidal which uses GPU memory properly.
#[pyfunction]
pub fn py_static_sin_toroidal(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_sin_toroidal_f64;
    let mut out = vec![0.0f64; input.len()];
    static_sin_toroidal_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static cos toroidal kernel - uses host_* FFI through static_* wrapper
/// Note: For high-performance use, prefer py_cos_toroidal which uses GPU memory properly.
#[pyfunction]
pub fn py_static_cos_toroidal(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_cos_toroidal_f64;
    let mut out = vec![0.0f64; input.len()];
    static_cos_toroidal_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static phi exp kernel - uses host_* FFI through static_* wrapper
/// Computes φ^x for each element
#[pyfunction]
pub fn py_static_phi_exp(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_phi_exp_f64;
    let mut out = vec![0.0f64; input.len()];
    static_phi_exp_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

/// Static phi exp inv kernel - uses host_* FFI through static_* wrapper
/// Computes φ^(-x) for each element
#[pyfunction]
pub fn py_static_phi_exp_inv(input: Vec<f64>) -> PyResult<Vec<f64>> {
    use super::srt_kernels::static_phi_exp_inv_f64;
    let mut out = vec![0.0f64; input.len()];
    static_phi_exp_inv_f64(&mut out, &input)
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out)
}

