//! Crystallization Operations
//!
//! Functions for snapping floating-point values to the golden lattice Q(φ)
//! with Ĥ (harmonization) attenuation and φ-dwell timing enforcement.
//!
//! The crystallization process implements the H-phase of the DHSR cycle:
//! 1. Apply Ĥ attenuation: Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
//! 2. Snap to Q(φ) lattice
//! 3. Enforce φ-dwell: t_H ≥ φ × t_D

use super::{PHI, PHI_INV};
use crate::exact::GoldenExact;
use std::time::{Duration, Instant};

use crate::tensor::srt_kernels::cuda_resonant_snap_gradient_f64;
use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

/// Apply Ĥ (harmonization) operator and crystallize to Q(φ) lattice.
///
/// The Ĥ operator attenuates modes with low golden weight:
///   Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
///
/// Where:
///   β(S) = φ⁻¹ × S ≈ 0.618 × S
///   w(n) = exp(-|n|²/φ)
///
/// # Arguments
/// * `flux` - Floating-point values from D-phase
/// * `mode_norm_sq` - Mode norm squared |n|² for each element
/// * `syntony` - Current syntony value S ∈ [0, 1]
/// * `precision` - Max coefficient for lattice snap
///
/// # Returns
/// * Vector of GoldenExact lattice points after Ĥ + snap
pub fn harmonize_and_crystallize(
    flux: &[f64],
    mode_norm_sq: &[f64],
    syntony: f64,
    precision: i64,
) -> Vec<GoldenExact> {
    // β(S) = φ⁻¹ × S
    let beta = PHI_INV * syntony;

    flux.iter()
        .zip(mode_norm_sq.iter())
        .map(|(&val, &norm_sq)| {
            // Golden weight: w(n) = exp(-|n|²/φ)
            let golden_weight = (-norm_sq / PHI).exp();

            // Ĥ attenuation: scale = 1 - β × (1 - w)
            let h_scale = 1.0 - beta * (1.0 - golden_weight);
            let harmonized = val * h_scale;

            // Snap to Q(φ) lattice
            GoldenExact::find_nearest(harmonized, precision)
        })
        .collect()
}

/// Crystallize with Ĥ attenuation and φ-dwell timing enforcement.
///
/// This is the full H-phase implementation:
/// 1. Apply Ĥ operator to flux values
/// 2. Snap to Q(φ) lattice
/// 3. If time remains before φ × t_D, deepen precision
///
/// # Arguments
/// * `flux` - Floating-point values from D-phase
/// * `mode_norm_sq` - Mode norm squared |n|² for each element
/// * `syntony` - Current syntony value S ∈ [0, 1]
/// * `base_precision` - Initial precision (max coefficient bound)
/// * `target_duration` - Target duration for H-phase (φ × D-phase duration)
///
/// # Returns
/// * Tuple of (crystallized lattice points, final precision used, actual duration)
pub fn crystallize_with_dwell(
    flux: &[f64],
    mode_norm_sq: &[f64],
    syntony: f64,
    base_precision: i64,
    target_duration: Duration,
) -> (Vec<GoldenExact>, i64, Duration) {
    let start = Instant::now();

    // First pass: Ĥ attenuation + snap at base precision
    let mut precision = base_precision;
    let mut lattice = harmonize_and_crystallize(flux, mode_norm_sq, syntony, precision);

    // φ-DWELL ENFORCEMENT
    // If we finished early, deepen the lattice precision productively
    let max_precision = ((base_precision as f64) * PHI * 5.0).ceil() as i64;

    while start.elapsed() < target_duration && precision < max_precision {
        // Increase precision using φ scaling; ensure forward progress
        let candidate_precision = ((precision as f64) * PHI).ceil() as i64;
        precision = candidate_precision.max(precision + 1);

        // Re-snap with higher precision (Ĥ already applied, just re-snap)
        let float_values: Vec<f64> = lattice.iter().map(|g| g.to_f64()).collect();

        lattice = float_values
            .iter()
            .map(|&x| GoldenExact::find_nearest(x, precision))
            .collect();
    }

    let actual_duration = start.elapsed();
    (lattice, precision, actual_duration)
}

/// Legacy: Crystallize a vector of f64 values to GoldenExact with φ-dwell timing.
///
/// WARNING: This version does NOT apply Ĥ attenuation. Use harmonize_and_crystallize
/// or crystallize_with_dwell for proper H-phase behavior.
///
/// # Arguments
/// * `values` - Floating-point values to crystallize
/// * `base_precision` - Initial precision (max coefficient bound)
/// * `target_duration` - Target duration for H-phase (φ × D-phase duration)
///
/// # Returns
/// * Tuple of (crystallized lattice points, final precision used, actual duration)
pub fn crystallize_with_dwell_legacy(
    values: &[f64],
    base_precision: i64,
    target_duration: Duration,
) -> (Vec<GoldenExact>, i64, Duration) {
    let start = Instant::now();

    // First pass at base precision
    let mut precision = base_precision;
    let mut lattice = snap_to_lattice(values, precision);

    // If we have time remaining, deepen the search (meditation)
    let max_precision = base_precision * 10; // Upper bound to prevent runaway

    while start.elapsed() < target_duration && precision < max_precision {
        // Increase precision
        let new_precision = (precision as f64 * PHI) as i64;
        if new_precision <= precision {
            break;
        }
        precision = new_precision;

        // Re-snap with higher precision
        // Get float values from current lattice, then re-snap
        let float_values: Vec<f64> = lattice.iter().map(|g| g.to_f64()).collect();

        lattice = snap_to_lattice(&float_values, precision);

        // Check if we've reached diminishing returns
        // (snapping to same values means we've converged)
    }

    let actual_duration = start.elapsed();
    (lattice, precision, actual_duration)
}

/// Snap a vector of f64 values to the golden lattice Q(φ).
///
/// This is a wrapper around GoldenExact::snap_vector that discards residuals.
#[inline]
pub fn snap_to_lattice(values: &[f64], max_coeff: i64) -> Vec<GoldenExact> {
    let (lattice, _residuals) = GoldenExact::snap_vector(values, max_coeff);
    lattice
}

/// Compute the total snap distance (L2 norm of residuals).
///
/// This measures how far the floating-point values were from the lattice.
pub fn snap_distance(values: &[f64], max_coeff: i64) -> f64 {
    let (_lattice, residuals) = GoldenExact::snap_vector(values, max_coeff);
    residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// Compute syntony directly from lattice values without GPU.
///
/// S(Ψ) = Σ |ψ_n|² exp(-|n|²/φ) / Σ |ψ_n|²
///
/// # Arguments
/// * `lattice` - Vector of GoldenExact lattice values
/// * `mode_norm_sq` - Precomputed |n|² for each mode
///
/// # Returns
/// Syntony value in [0, 1]
pub fn compute_lattice_syntony(lattice: &[GoldenExact], mode_norm_sq: &[f64]) -> f64 {
    if lattice.len() != mode_norm_sq.len() {
        return 0.0;
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (g, &norm_sq) in lattice.iter().zip(mode_norm_sq.iter()) {
        let val = g.to_f64();
        let amp_sq = val * val; // For real-valued lattice
        let weight = (-norm_sq * PHI_INV).exp();

        numerator += amp_sq * weight;
        denominator += amp_sq;
    }

    if denominator < 1e-15 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute the snap gradient: direction from pre-snap to post-snap values.
///
/// This gradient can be used to bias mutations in evolutionary strategies.
///
/// # Arguments
/// * `pre_snap` - Values before crystallization
/// * `post_snap` - Lattice values after crystallization
///
/// # Returns
/// Vector of gradients (post - pre) for each element
pub fn compute_snap_gradient(pre_snap: &[f64], post_snap: &[GoldenExact]) -> Vec<f64> {
    pre_snap
        .iter()
        .zip(post_snap.iter())
        .map(|(&pre, post)| {
            let post_f64 = post.to_f64();
            post_f64 - pre
        })
        .collect()
}

/// Compute snap gradient using CUDA acceleration.
///
/// This is the GPU-accelerated version of compute_snap_gradient.
/// Falls back to CPU implementation if CUDA is not available.
///
/// # Arguments
/// * `pre_snap` - Values before crystallization
/// * `post_snap` - Lattice values after crystallization
/// * `mode_norm_sq` - Mode norm squared |n|² for golden weighting
/// * `device` - CUDA device for computation (optional)
///
/// # Returns
/// Vector of gradients (post - pre) weighted by golden ratio decay
pub fn compute_snap_gradient_cuda(
    pre_snap: &[f64],
    post_snap: &[GoldenExact],
    mode_norm_sq: &[f64],
    device: Option<&Arc<CudaDevice>>,
) -> Result<Vec<f64>, String> {
    if pre_snap.len() != post_snap.len() || pre_snap.len() != mode_norm_sq.len() {
        return Err("All input arrays must have the same length".to_string());
    }

    let n = pre_snap.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Convert post_snap to f64 values
    let post_snap_f64: Vec<f64> = post_snap.iter().map(|g| g.to_f64()).collect();

    // Use CUDA if device is provided
    if let Some(device) = device {
        // Upload data to GPU
        let gpu_pre_snap = device
            .default_stream()
            .clone_htod(pre_snap)
            .map_err(|e| format!("Failed to upload pre_snap to GPU: {}", e))?;
        let gpu_post_snap = device
            .default_stream()
            .clone_htod(&post_snap_f64)
            .map_err(|e| format!("Failed to upload post_snap to GPU: {}", e))?;
        let gpu_mode_norm_sq = device
            .default_stream()
            .clone_htod(mode_norm_sq)
            .map_err(|e| format!("Failed to upload mode_norm_sq to GPU: {}", e))?;

        // Allocate output buffer
        let mut gpu_gradient: CudaSlice<f64> = device
            .default_stream()
            .alloc_zeros(n)
            .map_err(|e| format!("Failed to allocate gradient buffer: {}", e))?;

        // Run CUDA kernel
        cuda_resonant_snap_gradient_f64(
            device,
            &mut gpu_gradient,
            &gpu_pre_snap,
            &gpu_post_snap,
            &gpu_mode_norm_sq,
            n,
        )
        .map_err(|e| format!("CUDA snap gradient failed: {}", e))?;

        // Download result
        let mut gradient = vec![0.0f64; n];
        device
            .default_stream()
            .memcpy_dtoh(&gpu_gradient, &mut gradient)
            .map_err(|e| format!("Failed to download gradient from GPU: {}", e))?;

        Ok(gradient)
    } else {
        // Fall back to CPU implementation (unweighted)
        Ok(compute_snap_gradient(pre_snap, post_snap))
    }
}

/// Compute snap gradient with automatic CUDA dispatch.
///
/// Uses CUDA acceleration if available, otherwise falls back to CPU.
/// This is the recommended function for general use.
///
/// # Arguments
/// * `pre_snap` - Values before crystallization
/// * `post_snap` - Lattice values after crystallization
///
/// # Returns
/// Vector of gradients (post - pre) for each element
pub fn compute_snap_gradient_dispatch(
    pre_snap: &[f64],
    post_snap: &[GoldenExact],
    mode_norm_sq: &[f64],
) -> Vec<f64> {
    {
        // Try to get default CUDA device
        if let Ok(device) = crate::tensor::cuda::device_manager::get_device(0) {
            match compute_snap_gradient_cuda(pre_snap, post_snap, mode_norm_sq, Some(&device)) {
                Ok(result) => return result,
                Err(_) => {} // Fall back to CPU
            }
        }
    }

    // Fall back to CPU implementation (unweighted)
    compute_snap_gradient(pre_snap, post_snap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_to_lattice() {
        let values = vec![1.0, PHI, PHI * PHI, 3.0];
        let lattice = snap_to_lattice(&values, 100);

        assert_eq!(lattice.len(), 4);

        // Check that values are close to originals
        for (orig, snapped) in values.iter().zip(lattice.iter()) {
            let error = (orig - snapped.to_f64()).abs();
            assert!(error < 0.01, "Error {} too large", error);
        }
    }

    #[test]
    fn test_lattice_syntony() {
        // Constant mode should have high syntony
        let lattice = vec![
            GoldenExact::from_ints(1, 0),
            GoldenExact::from_ints(1, 0),
            GoldenExact::from_ints(1, 0),
            GoldenExact::from_ints(1, 0),
        ];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];

        let syntony = compute_lattice_syntony(&lattice, &mode_norms);

        // Syntony should be positive
        assert!(syntony > 0.0);
        assert!(syntony <= 1.0);
    }

    #[test]
    fn test_snap_gradient() {
        let pre = vec![1.5, 2.5, 3.5];
        let post = vec![
            GoldenExact::from_ints(2, 0), // 2.0
            GoldenExact::from_ints(2, 0), // 2.0
            GoldenExact::from_ints(4, 0), // 4.0
        ];

        let gradient = compute_snap_gradient(&pre, &post);

        assert_eq!(gradient.len(), 3);
        assert!((gradient[0] - 0.5).abs() < 1e-10); // 2.0 - 1.5
        assert!((gradient[1] - (-0.5)).abs() < 1e-10); // 2.0 - 2.5
        assert!((gradient[2] - 0.5).abs() < 1e-10); // 4.0 - 3.5
    }

    #[test]
    fn test_crystallize_with_dwell_precision_ramp() {
        let flux = vec![0.5, 1.0, 1.5];
        let mode_norms = vec![0.0, 1.0, 4.0];
        let syntony = 0.4;
        let base_precision = 50;
        let target = Duration::from_millis(5);

        let (_lat, final_precision, actual_duration) =
            crystallize_with_dwell(&flux, &mode_norms, syntony, base_precision, target);

        if actual_duration < target {
            assert!(final_precision > base_precision);
        } else {
            assert!(final_precision >= base_precision);
        }
    }
}
