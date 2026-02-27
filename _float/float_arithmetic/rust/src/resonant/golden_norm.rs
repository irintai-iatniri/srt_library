// Golden Batch Normalization
//
// This module implements batch normalization with golden ratio target variance,
// aligning with SRT theory that natural systems exhibit golden ratio structure
// at equilibrium.
//
// Theory:
// -------
// Standard BatchNorm targets variance = 1.0
// Golden BatchNorm targets variance = 1/φ ≈ 0.618
//
// This aligns with the syntonic equilibrium S* = φ - q ≈ 1.591
// where the natural variance at equilibrium is σ² = 1/φ
//
// Three normalization modes:
// 1. Golden: Normalize to var = 1/φ (theory-aligned)
// 2. Standard: Normalize to var = 1.0 (baseline)
// 3. Custom: User-specified target variance
//
// Process:
// 1. Compute batch statistics: mean, variance
// 2. Normalize to zero mean, unit variance
// 3. Scale to target variance (multiply by sqrt(target_var))
// 4. Apply affine transform (optional γ, β parameters)

use crate::exact::golden::GoldenExact;
use crate::resonant::tensor::{ResonantPhase, ResonantTensor};
use crate::resonant::ResonantError;

use pyo3::prelude::*;

/// Mode for golden batch normalization
#[derive(Clone, Copy, Debug, PartialEq)]
#[pyclass]
pub enum GoldenNormMode {
    /// Golden ratio target variance: 1/φ ≈ 0.618 (theory-aligned)
    Golden {},

    /// Standard normalization: variance = 1.0 (baseline)
    Standard {},

    /// Custom target variance (user-specified)
    Custom { target_var: f64 },
}

impl GoldenNormMode {
    /// Get the target variance for this mode
    pub fn target_variance(&self) -> f64 {
        match self {
            GoldenNormMode::Golden {} => {
                let phi = GoldenExact::phi();
                1.0 / phi.to_f64()
            }
            GoldenNormMode::Standard {} => 1.0,
            GoldenNormMode::Custom { target_var } => *target_var,
        }
    }

    /// Get the scale factor: sqrt(target_variance)
    pub fn scale_factor(&self) -> f64 {
        self.target_variance().sqrt()
    }
}

#[pymethods]
impl GoldenNormMode {
    #[new]
    fn new(mode: &str) -> Result<Self, ResonantError> {
        match mode {
            "golden" => Ok(GoldenNormMode::Golden {}),
            "standard" => Ok(GoldenNormMode::Standard {}),
            _ => Err(ResonantError::InvalidPhaseTransition(format!(
                "Unknown mode: {}. Use 'golden' or 'standard'",
                mode
            ))),
        }
    }

    #[staticmethod]
    fn custom(target_var: f64) -> Result<Self, ResonantError> {
        if target_var <= 0.0 {
            return Err(ResonantError::InvalidPhaseTransition(
                "Target variance must be positive".to_string(),
            ));
        }
        Ok(GoldenNormMode::Custom { target_var })
    }

    fn __repr__(&self) -> String {
        match self {
            GoldenNormMode::Golden {} => "GoldenNormMode('golden')".to_string(),
            GoldenNormMode::Standard {} => "GoldenNormMode('standard')".to_string(),
            GoldenNormMode::Custom { target_var } => {
                format!("GoldenNormMode.custom({})", target_var)
            }
        }
    }
}

/// Golden batch normalization for 1D tensors (batch, features)
///
/// Normalizes input to:
/// - mean = 0
/// - variance = 1/φ (or custom target)
///
/// # Arguments
/// * `input` - Input tensor (batch_size, num_features)
/// * `mode` - Normalization mode (golden, standard, or custom)
/// * `eps` - Epsilon for numerical stability (default: 1e-5)
/// * `affine_gamma` - Optional affine scale parameters (num_features,)
/// * `affine_beta` - Optional affine shift parameters (num_features,)
///
/// # Returns
/// Normalized tensor with shape (batch_size, num_features)
pub fn golden_batch_norm_1d(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    // Validate input shape
    if input.shape().len() != 2 {
        return Err(ResonantError::ShapeMismatch(format!(
            "Expected 2D tensor (batch, features), got shape {:?}",
            input.shape()
        )));
    }

    let batch_size = input.shape()[0];
    let num_features = input.shape()[1];

    // Validate batch size and features
    if batch_size == 0 {
        return Err(ResonantError::ShapeMismatch(
            "Batch size must be > 0 for golden batch norm".to_string(),
        ));
    }
    if num_features == 0 {
        return Err(ResonantError::ShapeMismatch(
            "Number of features must be > 0 for golden batch norm".to_string(),
        ));
    }

    // Validate affine parameters if provided
    if let Some(gamma) = affine_gamma {
        if gamma.shape() != &[num_features] {
            return Err(ResonantError::ShapeMismatch(format!(
                "Gamma shape {:?} doesn't match num_features {}",
                gamma.shape(),
                num_features
            )));
        }
    }

    if let Some(beta) = affine_beta {
        if beta.shape() != &[num_features] {
            return Err(ResonantError::ShapeMismatch(format!(
                "Beta shape {:?} doesn't match num_features {}",
                beta.shape(),
                num_features
            )));
        }
    }

    if input.phase() == ResonantPhase::Flux {
        // Tensor is on GPU - use CUDA implementation
        return cuda_golden_batch_norm_1d_dispatch(input, mode, eps, affine_gamma, affine_beta);
    }

    // CPU implementation (tensor is Crystallized)
    cpu_golden_batch_norm_1d(input, mode, eps, affine_gamma, affine_beta)
}

/// CPU implementation of golden batch norm 1D
fn cpu_golden_batch_norm_1d(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    let batch_size = input.shape()[0];
    let num_features = input.shape()[1];

    let input_data = input.to_floats_core();
    let mut output_data = vec![0.0; batch_size * num_features];

    // Process each feature independently
    for feat_idx in 0..num_features {
        // Compute mean and variance across batch
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for batch_idx in 0..batch_size {
            let idx = batch_idx * num_features + feat_idx;
            let val = input_data[idx];
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / (batch_size as f64);
        let variance = (sum_sq / (batch_size as f64)) - (mean * mean);

        // Normalize to zero mean, unit variance
        let std = (variance + eps).sqrt();
        let scale = mode.scale_factor();

        // Get affine parameters
        let gamma = affine_gamma
            .map(|g| g.to_floats_core()[feat_idx])
            .unwrap_or(1.0);

        let beta = affine_beta
            .map(|b| b.to_floats_core()[feat_idx])
            .unwrap_or(0.0);

        // Apply normalization
        for batch_idx in 0..batch_size {
            let idx = batch_idx * num_features + feat_idx;
            let val = input_data[idx];

            // Normalize: (x - mean) / std
            let normalized = (val - mean) / std;

            // Scale to target variance: multiply by sqrt(target_var)
            let scaled = normalized * scale;

            // Apply affine transform: gamma * scaled + beta
            output_data[idx] = gamma * scaled + beta;
        }
    }

    // Create output tensor with same mode norms and precision as input
    ResonantTensor::from_floats(
        &output_data,
        input.shape().to_vec(),
        input.mode_norm_sq().to_vec(),
        input.precision(),
    )
}

/// Golden batch normalization for 2D tensors (batch, channels, height, width)
///
/// Normalizes across batch, height, and width dimensions per channel.
///
/// # Arguments
/// * `input` - Input tensor (batch_size, channels, height, width)
/// * `mode` - Normalization mode (golden, standard, or custom)
/// * `eps` - Epsilon for numerical stability (default: 1e-5)
/// * `affine_gamma` - Optional affine scale parameters (channels,)
/// * `affine_beta` - Optional affine shift parameters (channels,)
///
/// # Returns
/// Normalized tensor with shape (batch_size, channels, height, width)
pub fn golden_batch_norm_2d(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    // Validate input shape
    if input.shape().len() != 4 {
        return Err(ResonantError::ShapeMismatch(format!(
            "Expected 4D tensor (batch, channels, height, width), got shape {:?}",
            input.shape()
        )));
    }

    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let height = input.shape()[2];
    let width = input.shape()[3];

    // Validate tensor dimensions
    if batch_size == 0 {
        return Err(ResonantError::ShapeMismatch(
            "Batch size must be > 0 for 2D golden batch norm".to_string(),
        ));
    }
    if channels == 0 {
        return Err(ResonantError::ShapeMismatch(
            "Number of channels must be > 0 for 2D golden batch norm".to_string(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(ResonantError::ShapeMismatch(format!(
            "Spatial dimensions must be > 0, got height={}, width={}",
            height, width
        )));
    }

    // Validate affine parameters if provided
    if let Some(gamma) = affine_gamma {
        if gamma.shape() != &[channels] {
            return Err(ResonantError::ShapeMismatch(format!(
                "Gamma shape {:?} doesn't match channels {}",
                gamma.shape(),
                channels
            )));
        }
    }

    if let Some(beta) = affine_beta {
        if beta.shape() != &[channels] {
            return Err(ResonantError::ShapeMismatch(format!(
                "Beta shape {:?} doesn't match channels {}",
                beta.shape(),
                channels
            )));
        }
    }

    if input.phase() == ResonantPhase::Flux {
        // Tensor is on GPU - use CUDA implementation
        return cuda_golden_batch_norm_2d_dispatch(input, mode, eps, affine_gamma, affine_beta);
    }

    // CPU implementation (tensor is Crystallized)
    cpu_golden_batch_norm_2d(input, mode, eps, affine_gamma, affine_beta)
}

/// CPU implementation of golden batch norm 2D
fn cpu_golden_batch_norm_2d(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let height = input.shape()[2];
    let width = input.shape()[3];
    let spatial_size = height * width;

    let input_data = input.to_floats_core();
    let mut output_data = vec![0.0; batch_size * channels * height * width];

    // Process each channel independently
    for chan_idx in 0..channels {
        // Compute mean and variance across batch, height, width
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let count = (batch_size * spatial_size) as f64;

        for batch_idx in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let idx = ((batch_idx * channels + chan_idx) * height + h) * width + w;
                    let val = input_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        let mean = sum / count;
        let variance = (sum_sq / count) - (mean * mean);

        // Normalize to zero mean, unit variance
        let std = (variance + eps).sqrt();
        let scale = mode.scale_factor();

        // Get affine parameters
        let gamma = affine_gamma
            .map(|g| g.to_floats_core()[chan_idx])
            .unwrap_or(1.0);

        let beta = affine_beta
            .map(|b| b.to_floats_core()[chan_idx])
            .unwrap_or(0.0);

        // Apply normalization
        for batch_idx in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let idx = ((batch_idx * channels + chan_idx) * height + h) * width + w;
                    let val = input_data[idx];

                    // Normalize: (x - mean) / std
                    let normalized = (val - mean) / std;

                    // Scale to target variance: multiply by sqrt(target_var)
                    let scaled = normalized * scale;

                    // Apply affine transform: gamma * scaled + beta
                    output_data[idx] = gamma * scaled + beta;
                }
            }
        }
    }

    // Create output tensor with same mode norms and precision as input
    ResonantTensor::from_floats(
        &output_data,
        vec![batch_size, channels, height, width],
        input.mode_norm_sq().to_vec(),
        input.precision(),
    )
}

fn cuda_golden_batch_norm_1d_dispatch(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    use crate::tensor::cuda::device_manager::get_device;
    use crate::tensor::srt_kernels::cuda_golden_bn_1d_f64;
    use cudarc::driver::CudaSlice;

    // Only use CUDA for Golden mode (the kernel is hardcoded for golden ratio)
    if !matches!(mode, GoldenNormMode::Golden {}) {
        return cpu_golden_batch_norm_1d(input, mode, eps, affine_gamma, affine_beta);
    }

    // Get the device from the tensor
    let device_idx = input.device_idx().unwrap_or(0);
    let device = get_device(device_idx)
        .map_err(|e| ResonantError::CudaError(format!("Failed to get CUDA device: {}", e)))?;

    let batch_size = input.shape()[0] as i32;
    let features = input.shape()[1] as i32;

    // Get flux data from GPU
    let flux = input.flux_ref().ok_or(ResonantError::NoFluxPresent)?;

    // Prepare affine parameters
    let gpu_gamma = if let Some(gamma) = affine_gamma {
        let gamma_data = gamma.to_floats_core();
        Some(
            device
                .default_stream()
                .clone_htod(&gamma_data)
                .map_err(|e| ResonantError::CudaError(e.to_string()))?,
        )
    } else {
        None
    };

    let gpu_beta = if let Some(beta) = affine_beta {
        let beta_data = beta.to_floats_core();
        Some(
            device
                .default_stream()
                .clone_htod(&beta_data)
                .map_err(|e| ResonantError::CudaError(e.to_string()))?,
        )
    } else {
        None
    };

    // Allocate output buffer
    let total_elements = (batch_size * features) as usize;
    let mut gpu_output: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(total_elements)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

    // Run CUDA kernel
    cuda_golden_bn_1d_f64(
        &device,
        &mut gpu_output,
        flux,
        gpu_gamma.as_ref(),
        gpu_beta.as_ref(),
        eps,
        batch_size,
        features,
    )
    .map_err(|e| ResonantError::CudaError(format!("CUDA golden_bn_1d failed: {}", e)))?;

    // Copy result back to host
    let mut output_host = vec![0.0f64; total_elements];
    device
        .default_stream()
        .memcpy_dtoh(&gpu_output, &mut output_host)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

    // Create output ResonantTensor
    ResonantTensor::from_floats(
        &output_host,
        input.shape().to_vec(),
        input.mode_norm_sq().to_vec(),
        input.precision(),
    )
}

fn cuda_golden_batch_norm_2d_dispatch(
    input: &ResonantTensor,
    mode: GoldenNormMode,
    eps: f64,
    affine_gamma: Option<&ResonantTensor>,
    affine_beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    use crate::tensor::cuda::device_manager::get_device;
    use crate::tensor::srt_kernels::cuda_golden_bn_2d_f64;
    use cudarc::driver::CudaSlice;

    // Only use CUDA for Golden mode (the kernel is hardcoded for golden ratio)
    if !matches!(mode, GoldenNormMode::Golden {}) {
        return cpu_golden_batch_norm_2d(input, mode, eps, affine_gamma, affine_beta);
    }

    // Get the device from the tensor
    let device_idx = input.device_idx().unwrap_or(0);
    let device = get_device(device_idx)
        .map_err(|e| ResonantError::CudaError(format!("Failed to get CUDA device: {}", e)))?;

    let batch_size = input.shape()[0] as i32;
    let channels = input.shape()[1] as i32;
    let height = input.shape()[2] as i32;
    let width = input.shape()[3] as i32;

    // Get flux data from GPU
    let flux = input.flux_ref().ok_or(ResonantError::NoFluxPresent)?;

    // Compute batch statistics (mean and variance per channel)
    // This is a simplified version - in practice you'd want to use CUDA kernels for this too
    let input_host = input.to_floats_core();
    let mut mean_data = vec![0.0f64; channels as usize];
    let mut var_data = vec![0.0f64; channels as usize];

    let spatial_size = (height * width) as usize;
    let batch_spatial = (batch_size as usize) * spatial_size;

    for c in 0..channels as usize {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for b in 0..batch_size as usize {
            for s in 0..spatial_size {
                let idx = (b * channels as usize + c) * spatial_size + s;
                let val = input_host[idx];
                sum += val;
                sum_sq += val * val;
            }
        }

        let count = batch_spatial as f64;
        mean_data[c] = sum / count;
        var_data[c] = (sum_sq / count) - (mean_data[c] * mean_data[c]);
    }

    // Upload statistics to GPU
    let gpu_mean = device
        .default_stream()
        .clone_htod(&mean_data)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;
    let gpu_var = device
        .default_stream()
        .clone_htod(&var_data)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

    // Prepare affine parameters
    let gpu_gamma = if let Some(gamma) = affine_gamma {
        let gamma_data = gamma.to_floats_core();
        Some(
            device
                .default_stream()
                .clone_htod(&gamma_data)
                .map_err(|e| ResonantError::CudaError(e.to_string()))?,
        )
    } else {
        None
    };

    let gpu_beta = if let Some(beta) = affine_beta {
        let beta_data = beta.to_floats_core();
        Some(
            device
                .default_stream()
                .clone_htod(&beta_data)
                .map_err(|e| ResonantError::CudaError(e.to_string()))?,
        )
    } else {
        None
    };

    // Allocate output buffer
    let total_elements = (batch_size * channels * height * width) as usize;
    let mut gpu_output: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(total_elements)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

    // Run CUDA kernel
    cuda_golden_bn_2d_f64(
        &device,
        &mut gpu_output,
        flux,
        &gpu_mean,
        &gpu_var,
        gpu_gamma.as_ref(),
        gpu_beta.as_ref(),
        eps,
        batch_size,
        channels,
        height,
        width,
    )
    .map_err(|e| ResonantError::CudaError(e))?;

    // Download result
    let mut output_data = vec![0.0f64; total_elements];
    device
        .default_stream()
        .memcpy_dtoh(&gpu_output, &mut output_data)
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

    // Create output tensor
    ResonantTensor::from_floats(
        &output_data,
        input.shape().to_vec(),
        input.mode_norm_sq().to_vec(),
        input.precision(),
    )
}

// ============================================================================
// Python API
// ============================================================================

#[pyfunction]
#[pyo3(signature = (input, mode, eps=1e-5, gamma=None, beta=None))]
pub fn golden_batch_norm_1d_py(
    input: &ResonantTensor,
    mode: &GoldenNormMode,
    eps: f64,
    gamma: Option<&ResonantTensor>,
    beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    golden_batch_norm_1d(input, *mode, eps, gamma, beta)
}

#[pyfunction]
#[pyo3(signature = (input, mode, eps=1e-5, gamma=None, beta=None))]
pub fn golden_batch_norm_2d_py(
    input: &ResonantTensor,
    mode: &GoldenNormMode,
    eps: f64,
    gamma: Option<&ResonantTensor>,
    beta: Option<&ResonantTensor>,
) -> Result<ResonantTensor, ResonantError> {
    golden_batch_norm_2d(input, *mode, eps, gamma, beta)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_golden_norm_mode_variance() {
        let golden = GoldenNormMode::Golden {};
        let target_var = golden.target_variance();

        // Should be 1/φ ≈ 0.618
        assert_relative_eq!(target_var, 0.618034, epsilon = 0.001);
    }

    #[test]
    fn test_golden_batch_norm_1d_golden_mode() {
        // Create test input: (batch=4, features=2)
        let input_data = vec![
            1.0, 2.0, // batch 0
            3.0, 4.0, // batch 1
            5.0, 6.0, // batch 2
            7.0, 8.0, // batch 3
        ];
        let mode_norms = vec![0.0; 8]; // Default mode norms
        let input = ResonantTensor::from_floats(&input_data, vec![4, 2], mode_norms, 100).unwrap();

        let mode = GoldenNormMode::Golden {};
        let output = golden_batch_norm_1d(&input, mode, 1e-5, None, None).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[4, 2]);

        // Check that each feature has approximately zero mean
        let output_data = output.to_floats_core();

        // Feature 0: values at indices 0, 2, 4, 6
        let feat0_mean = (output_data[0] + output_data[2] + output_data[4] + output_data[6]) / 4.0;
        assert_relative_eq!(feat0_mean, 0.0, epsilon = 0.1);

        // Feature 1: values at indices 1, 3, 5, 7
        let feat1_mean = (output_data[1] + output_data[3] + output_data[5] + output_data[7]) / 4.0;
        assert_relative_eq!(feat1_mean, 0.0, epsilon = 0.1);

        // Check that variance is approximately 1/φ
        let feat0_var = ((output_data[0] - feat0_mean).powi(2)
            + (output_data[2] - feat0_mean).powi(2)
            + (output_data[4] - feat0_mean).powi(2)
            + (output_data[6] - feat0_mean).powi(2))
            / 4.0;

        let target_var = 1.0 / 1.618034;
        assert_relative_eq!(feat0_var, target_var, epsilon = 0.1);
    }

    #[test]
    fn test_golden_batch_norm_1d_standard_mode() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mode_norms = vec![0.0; 6];
        let input = ResonantTensor::from_floats(&input_data, vec![3, 2], mode_norms, 100).unwrap();

        let mode = GoldenNormMode::Standard {};
        let output = golden_batch_norm_1d(&input, mode, 1e-5, None, None).unwrap();

        assert_eq!(output.shape(), &[3, 2]);

        // For standard mode, variance should be ≈ 1.0
        let output_data = output.to_floats_core();
        let mean = (output_data[0] + output_data[2] + output_data[4]) / 3.0;
        let variance = ((output_data[0] - mean).powi(2)
            + (output_data[2] - mean).powi(2)
            + (output_data[4] - mean).powi(2))
            / 3.0;

        assert_relative_eq!(variance, 1.0, epsilon = 0.2);
    }

    #[test]
    fn test_golden_batch_norm_1d_with_affine() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0; 4];
        let input =
            ResonantTensor::from_floats(&input_data, vec![2, 2], mode_norms.clone(), 100).unwrap();

        // Affine parameters: gamma=[2.0, 3.0], beta=[1.0, -1.0]
        let gamma = ResonantTensor::from_floats(&[2.0, 3.0], vec![2], vec![0.0; 2], 100).unwrap();
        let beta = ResonantTensor::from_floats(&[1.0, -1.0], vec![2], vec![0.0; 2], 100).unwrap();

        let mode = GoldenNormMode::Golden {};
        let output = golden_batch_norm_1d(&input, mode, 1e-5, Some(&gamma), Some(&beta)).unwrap();

        assert_eq!(output.shape(), &[2, 2]);

        // Verify affine transform was applied
        // (normalized values should be scaled by gamma and shifted by beta)
    }

    #[test]
    fn test_golden_batch_norm_2d_shape() {
        // Test 2D batch norm with shape (batch, channels, height, width)
        let batch_size = 2;
        let channels = 3;
        let height = 4;
        let width = 4;
        let size = batch_size * channels * height * width;

        let input_data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let mode_norms = vec![0.0; size];
        let input = ResonantTensor::from_floats(
            &input_data,
            vec![batch_size, channels, height, width],
            mode_norms,
            100,
        )
        .unwrap();

        let mode = GoldenNormMode::Golden {};
        let output = golden_batch_norm_2d(&input, mode, 1e-5, None, None).unwrap();

        assert_eq!(output.shape(), &[batch_size, channels, height, width]);
    }

    #[test]
    fn test_golden_batch_norm_2d_per_channel() {
        // Verify that normalization is done per channel
        let input_data = vec![
            // Batch 0, Channel 0 (2x2)
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1 (2x2)
            10.0, 20.0, 30.0, 40.0, // Batch 1, Channel 0 (2x2)
            5.0, 6.0, 7.0, 8.0, // Batch 1, Channel 1 (2x2)
            50.0, 60.0, 70.0, 80.0,
        ];

        let mode_norms = vec![0.0; input_data.len()];
        let input =
            ResonantTensor::from_floats(&input_data, vec![2, 2, 2, 2], mode_norms, 100).unwrap();

        let mode = GoldenNormMode::Golden {};
        let output = golden_batch_norm_2d(&input, mode, 1e-5, None, None).unwrap();

        assert_eq!(output.shape(), &[2, 2, 2, 2]);

        // Each channel should have approximately zero mean across batch, H, W
        let output_data = output.to_floats_core();

        // Channel 0: indices 0-7 (batch 0: 0-3, batch 1: 8-11)
        let chan0_indices = [0, 1, 2, 3, 8, 9, 10, 11];
        let chan0_mean: f64 = chan0_indices.iter().map(|&i| output_data[i]).sum::<f64>() / 8.0;
        assert_relative_eq!(chan0_mean, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_golden_vs_standard_variance() {
        // Verify that golden mode gives smaller variance than standard
        let input_data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mode_norms = vec![0.0; 20];
        let input =
            ResonantTensor::from_floats(&input_data, vec![10, 2], mode_norms.clone(), 100).unwrap();

        let golden_output =
            golden_batch_norm_1d(&input, GoldenNormMode::Golden {}, 1e-5, None, None).unwrap();

        let standard_output =
            golden_batch_norm_1d(&input, GoldenNormMode::Standard {}, 1e-5, None, None).unwrap();

        let golden_data = golden_output.to_floats_core();
        let standard_data = standard_output.to_floats_core();

        // Compute variance for feature 0
        let golden_var: f64 = (0..10).map(|i| golden_data[i * 2].powi(2)).sum::<f64>() / 10.0;
        let standard_var: f64 = (0..10).map(|i| standard_data[i * 2].powi(2)).sum::<f64>() / 10.0;

        // Golden variance should be smaller (≈ 0.618 vs ≈ 1.0)
        assert!(golden_var < standard_var);
        assert_relative_eq!(golden_var / standard_var, 0.618, epsilon = 0.15);
    }
}
