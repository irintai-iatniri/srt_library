//! Phi-scaled residual operations for theory-aligned neural networks.
//!
//! Implements three residual modes:
//! - phi: output = identity + residual/φ (default, recommended)
//! - phi_symmetric: output = (identity + residual)/φ
//! - standard: output = identity + residual (for ablation)

use super::PHI_INV;
use crate::resonant::{ResonantError, ResonantPhase, ResonantTensor, PHI};
use pyo3::prelude::*;

/// Phi-residual modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum PhiResidualMode {
    /// output = identity + residual/φ (theory-aligned, default)
    Phi,
    /// output = (identity + residual)/φ (symmetric scaling)
    PhiSymmetric,
    /// output = identity + residual (standard ResNet)
    Standard,
}

#[pymethods]
impl PhiResidualMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "phi" => Ok(PhiResidualMode::Phi),
            "phi_symmetric" => Ok(PhiResidualMode::PhiSymmetric),
            "standard" => Ok(PhiResidualMode::Standard),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown mode: {}. Use 'phi', 'phi_symmetric', or 'standard'",
                mode_str
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PhiResidualMode::Phi => "PhiResidualMode('phi')".to_string(),
            PhiResidualMode::PhiSymmetric => "PhiResidualMode('phi_symmetric')".to_string(),
            PhiResidualMode::Standard => "PhiResidualMode('standard')".to_string(),
        }
    }
}

impl ResonantTensor {
    /// Apply phi-residual connection: combines identity with residual using golden ratio
    pub fn phi_residual(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        // Validate shapes match
        if identity.shape() != residual.shape() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Identity shape {:?} does not match residual shape {:?}",
                identity.shape(),
                residual.shape()
            )));
        }

        // Convert to floats for computation
        let identity_floats = identity.to_floats_core();
        let residual_floats = residual.to_floats_core();

        let output_floats: Vec<f64> = match mode {
            PhiResidualMode::Phi => identity_floats
                .iter()
                .zip(residual_floats.iter())
                .map(|(i, r)| i + r * PHI_INV)
                .collect(),
            PhiResidualMode::PhiSymmetric => identity_floats
                .iter()
                .zip(residual_floats.iter())
                .map(|(i, r)| (i + r) * PHI_INV)
                .collect(),
            PhiResidualMode::Standard => identity_floats
                .iter()
                .zip(residual_floats.iter())
                .map(|(i, r)| i + r)
                .collect(),
        };

        // Create output tensor
        ResonantTensor::from_floats(
            &output_floats,
            identity.shape().to_vec(),
            identity.mode_norm_sq().to_vec(),
            identity.precision(),
        )
    }

    /// Fused phi-residual + ReLU activation
    pub fn phi_residual_relu(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        // Try CUDA acceleration for all modes
        {
            if let Some(device_idx) = identity.device_idx().or(residual.device_idx()) {
                if let Ok(device) = crate::tensor::cuda::device_manager::get_device(device_idx) {
                    // Validate device ordinal matches requested index
                    if device.ordinal() as usize == device_idx {
                        // Use optimized fused kernel for Phi mode
                        if mode == PhiResidualMode::Phi {
                            return Self::phi_residual_relu_cuda(identity, residual);
                        } else {
                            // For other modes, use phi_residual_cuda followed by ReLU
                            let combined = Self::phi_residual_cuda(identity, residual, mode)?;

                            // Copy GPU flux to host, apply ReLU on CPU, and return new tensor.
                            let combined_flux =
                                combined.flux_ref().ok_or(ResonantError::NoFluxPresent)?;
                            let n = combined.len();

                            // Allocate host buffer and copy from device
                            let mut combined_host = vec![0.0f64; n];
                            device
                                .default_stream()
                                .memcpy_dtoh(combined_flux, &mut combined_host)
                                .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                            // Apply ReLU on host
                            let relu_host: Vec<f64> = combined_host
                                .iter()
                                .map(|&x| if x > 0.0 { x } else { 0.0 })
                                .collect();

                            return ResonantTensor::from_floats(
                                &relu_host,
                                combined.shape().to_vec(),
                                combined.mode_norm_sq().to_vec(),
                                combined.precision(),
                            );
                        }
                    }
                }
            }
        }

        // Fallback to CPU
        let combined = Self::phi_residual(identity, residual, mode)?;
        let combined_floats = combined.to_floats_core();
        let relu_floats: Vec<f64> = combined_floats
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        ResonantTensor::from_floats(
            &relu_floats,
            combined.shape().to_vec(),
            combined.mode_norm_sq().to_vec(),
            combined.precision(),
        )
    }
}

// Python-accessible wrapper functions
#[pyfunction]
pub fn phi_residual(
    identity: &ResonantTensor,
    residual: &ResonantTensor,
    mode: PhiResidualMode,
) -> PyResult<ResonantTensor> {
    ResonantTensor::phi_residual(identity, residual, mode).map_err(|e| PyErr::from(e))
}

#[pyfunction]
pub fn phi_residual_relu(
    identity: &ResonantTensor,
    residual: &ResonantTensor,
    mode: PhiResidualMode,
) -> PyResult<ResonantTensor> {
    ResonantTensor::phi_residual_relu(identity, residual, mode).map_err(|e| PyErr::from(e))
}

// =============================================================================
// CUDA Implementation (when feature enabled)
// =============================================================================

impl ResonantTensor {
    /// GPU-accelerated phi-residual operation
    pub fn phi_residual_cuda(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        use crate::tensor::srt_kernels::cuda_phi_residual_f64;

        let device_idx = identity.device_idx().or(residual.device_idx()).unwrap_or(0);
        let device = crate::tensor::cuda::device_manager::get_device(device_idx)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Ensure both tensors are in flux phase
        let mut identity = identity.clone();
        if identity.phase() != ResonantPhase::Flux {
            identity.wake_flux(device.clone())?;
        }

        let mut residual = residual.clone();
        if residual.phase() != ResonantPhase::Flux {
            residual.wake_flux(device.clone())?;
        }

        // Allocate output
        let n = identity.len();
        let mut out_flux = device
            .default_stream()
            .alloc_zeros::<f64>(n)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Launch kernel
        let identity_flux = identity.flux_ref().ok_or(ResonantError::NoFluxPresent)?;
        let residual_flux = residual.flux_ref().ok_or(ResonantError::NoFluxPresent)?;

        cuda_phi_residual_f64(&device, &mut out_flux, identity_flux, residual_flux, mode)
            .map_err(|e| ResonantError::CudaError(e))?;

        // Create output tensor (flux phase)
        let mut output = identity.clone();
        output.set_flux(out_flux);
        output.set_device_idx(device_idx);

        Ok(output)
    }

    /// GPU-accelerated fused phi-residual + ReLU operation
    pub fn phi_residual_relu_cuda(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        use crate::tensor::srt_kernels::cuda_phi_residual_relu_f64;

        let device_idx = identity.device_idx().or(residual.device_idx()).unwrap_or(0);
        let device = crate::tensor::cuda::device_manager::get_device(device_idx)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Ensure both tensors are in flux phase
        let mut identity = identity.clone();
        if identity.phase() != ResonantPhase::Flux {
            identity.wake_flux(device.clone())?;
        }

        let mut residual = residual.clone();
        if residual.phase() != ResonantPhase::Flux {
            residual.wake_flux(device.clone())?;
        }

        // Allocate output
        let n = identity.len();
        let mut out_flux = device
            .default_stream()
            .alloc_zeros::<f64>(n)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Launch kernel (note: cuda_phi_residual_relu_f64 doesn't take mode, assumes phi mode)
        let identity_flux = identity.flux_ref().ok_or(ResonantError::NoFluxPresent)?;
        let residual_flux = residual.flux_ref().ok_or(ResonantError::NoFluxPresent)?;

        cuda_phi_residual_relu_f64(&device, &mut out_flux, identity_flux, residual_flux)
            .map_err(|e| ResonantError::CudaError(e))?;

        // Create output tensor (flux phase)
        let mut output = identity.clone();
        output.set_flux(out_flux);
        output.set_device_idx(device_idx);

        Ok(output)
    }
}

/// ============================================================================
/// SGC Extensions: Numerically Stable Operations
/// ============================================================================

/// Extension trait for SGC-specific ResonantTensor operations
pub trait ResonantTensorExt {
    fn sgc_stable_cycle(&mut self, noise_scale: f64, base_precision: i64) -> PyResult<f64>;
    fn sgc_generate_phi_seed(&self, target_stats: &[f64]) -> PyResult<ResonantTensor>;
    fn sgc_phi_evolution(&self, scales: &[f64], steps_per_scale: usize) -> PyResult<Vec<Vec<Vec<f64>>>>;
    fn sgc_phi_converged(&self, previous_values: &[f64], tolerance: f64) -> bool;
}

impl ResonantTensorExt for ResonantTensor {
    /// Stable DHSR cycle with adaptive phi-scaling to prevent collapse
    ///
    /// Prevents evolution collapse by:
    /// 1. Detecting small values before crystallization
    /// 2. Applying phi-scaled amplification for small values
    /// 3. Using adaptive precision based on magnitude
    fn sgc_stable_cycle(&mut self, noise_scale: f64, base_precision: i64) -> PyResult<f64> {
        // Pre-cycle analysis
        let values = self.to_floats_core();
        let max_val = values.iter().cloned().fold(0.0f64, f64::max).abs();

        // Adaptive phi-scaling to prevent collapse
        let scale_factor = if max_val < 1e-6 {
            PHI  // Amplify tiny values by φ
        } else if max_val < 1e-3 {
            PHI_INV.sqrt()  // Moderate amplification
        } else if max_val > 10.0 {
            PHI_INV  // Dampen large values
        } else {
            1.0  // No scaling needed
        };

        // Apply phi-scaled amplification if needed
        if (scale_factor - 1.0).abs() > 1e-10 {
            let scaled_values: Vec<f64> = values.iter().map(|&v| v * scale_factor).collect();
            self.crystallize_cpu(&scaled_values, base_precision)?;
        }

        // Adaptive precision prevents division issues
        let precision = if max_val < 1e-6 {
            (base_precision * 10).max(1000)  // High precision for tiny values
        } else if max_val < 1e-3 {
            (base_precision * 5).max(500)
        } else {
            base_precision.max(10)
        };

        // Run cycle with safe parameters
        self.run_cpu_cycle(noise_scale.min(0.01), precision)
            .map_err(|e| e.into())
    }

    /// Phi-scaled seed generation for SGC structure regeneration
    ///
    /// Creates seeds using phi-weighted statistics to ensure generative stability
    fn sgc_generate_phi_seed(&self, target_stats: &[f64]) -> PyResult<Self> {
        if target_stats.len() < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need at least 3 statistics for phi-seed generation"
            ));
        }

        let mean = target_stats[0];
        let std = target_stats[1];
        let max_val = target_stats[2];

        // Phi-weighted seed generation
        let seed_values = vec![
            mean,
            std * PHI_INV,
            max_val * PHI_INV.sqrt(),
            mean + std,
            std - mean * PHI_INV,
            max_val * PHI_INV,
            mean * PHI,
            std * PHI.sqrt(),
        ];

        // Safe scaling to prevent overflow
        let max_seed = seed_values.iter().cloned().fold(0.0f64, f64::max).abs();
        let scale = if max_seed > 1.0 { PHI_INV } else { 1.0 };
        let safe_values: Vec<f64> = seed_values.iter().map(|&v| v * scale).collect();

        ResonantTensor::from_floats(
            &safe_values,
            vec![8],  // 8D seed
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],  // Mode norms
            100,
        ).map_err(|e| e.into())
    }

    /// Multi-scale phi evolution for robust SGC regeneration
    ///
    /// Evolves at multiple phi-scaled magnitudes to find stable generative patterns
    fn sgc_phi_evolution(&self, scales: &[f64], steps_per_scale: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let mut results = Vec::new();

        for &scale in scales {
            // Create scaled version
            let values = self.to_floats_core();
            let scaled_values: Vec<f64> = values.iter().map(|&v| v * scale).collect();
            let mut scaled_tensor = ResonantTensor::from_floats(
                &scaled_values,
                self.shape().to_vec(),
                self.mode_norm_sq().to_vec(),
                self.precision(),
            )?;

            let mut scale_history = vec![scaled_values];

            // Evolve at this scale
            for _ in 0..steps_per_scale {
                scaled_tensor.sgc_stable_cycle(0.001, 50)?;
                scale_history.push(scaled_tensor.to_floats_core());
            }

            results.push(scale_history);
        }

        Ok(results)
    }

    /// Phi-convergence detection for SGC evolution
    ///
    /// Uses phi-scaled thresholds to detect when evolution has stabilized
    fn sgc_phi_converged(&self, previous_values: &[f64], tolerance: f64) -> bool {
        let current_values = self.to_floats_core();

        if current_values.len() != previous_values.len() {
            return false;
        }

        // Phi-scaled convergence criteria
        let phi_tolerance = tolerance * PHI_INV;

        // Check absolute convergence
        let abs_diff = current_values.iter()
            .zip(previous_values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        if abs_diff < phi_tolerance {
            return true;
        }

        // Check relative convergence with phi-scaling
        let max_val = current_values.iter()
            .chain(previous_values.iter())
            .cloned()
            .fold(0.0f64, f64::max)
            .abs();

        if max_val > phi_tolerance {
            let rel_diff = abs_diff / max_val;
            if rel_diff < phi_tolerance {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_residual_mode_phi() {
        let identity =
            ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual =
            ResonantTensor::from_floats(&vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0.0; 4], 100)
                .unwrap();

        let result =
            ResonantTensor::phi_residual(&identity, &residual, PhiResidualMode::Phi).unwrap();

        let expected = vec![
            1.0 + 1.0 * PHI_INV,
            1.0 + 2.0 * PHI_INV,
            1.0 + 3.0 * PHI_INV,
            1.0 + 4.0 * PHI_INV,
        ];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_mode_symmetric() {
        let identity =
            ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual =
            ResonantTensor::from_floats(&vec![2.0, 4.0, 6.0, 8.0], vec![4], vec![0.0; 4], 100)
                .unwrap();

        let result =
            ResonantTensor::phi_residual(&identity, &residual, PhiResidualMode::PhiSymmetric)
                .unwrap();

        let expected = vec![
            (1.0 + 2.0) * PHI_INV,
            (1.0 + 4.0) * PHI_INV,
            (1.0 + 6.0) * PHI_INV,
            (1.0 + 8.0) * PHI_INV,
        ];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_mode_standard() {
        let identity =
            ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual =
            ResonantTensor::from_floats(&vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0.0; 4], 100)
                .unwrap();

        let result =
            ResonantTensor::phi_residual(&identity, &residual, PhiResidualMode::Standard).unwrap();

        let expected = vec![2.0, 3.0, 4.0, 5.0];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_shape_mismatch() {
        let identity =
            ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual =
            ResonantTensor::from_floats(&vec![1.0; 8], vec![8], vec![0.0; 8], 100).unwrap();

        let result = ResonantTensor::phi_residual(&identity, &residual, PhiResidualMode::Phi);

        assert!(result.is_err());
    }
}
