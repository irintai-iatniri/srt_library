//! Spectral operations for SRT theta series and heat kernels.
//!
//! This module provides high-performance implementations of:
//! - Theta series evaluation: Θ(t) = Σ_n w(n) exp(-π|n|²/t)
//! - Heat kernel trace: K(t) = Σ_n exp(-λ_n t)
//! - Spectral zeta function: ζ(s) = Σ_n λ_n^(-s)
//! - Eigenvalue computations

use pyo3::prelude::*;
use crate::constants::SRT_PI as PI;

use crate::tensor::srt_kernels::PHI;
use crate::winding::WindingState;
use crate::exact::{FixedPoint64, ExactScalar, rational::Rational};

const PHI_FP: FixedPoint64 = FixedPoint64(6949826725); // φ × 2³² (derived from crate::constants::PHI)
const PI_FP: FixedPoint64 = FixedPoint64(13493037704); // π × 2³² (derived from std::f64::consts::PI)

// =============================================================================
// Theta Series Operations
// =============================================================================

/// Compute theta series: Θ(t) = Σ_n w(n) exp(-π|n|²/t)
///
/// Args:
///     windings: List of WindingState instances
///     t: Temperature parameter
///
/// Returns:
///     Theta series value at t
#[pyfunction]
pub fn theta_series_evaluate(windings: Vec<WindingState>, t: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let mut sum_fp = FixedPoint64::zero();
    
    // Safety check for t close to zero to avoid division by zero
    if t_fp.0 == 0 { return 0.0; }

    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        
        // Golden weight: exp(-|n|²/φ)
        // arg = -norm_sq / PHI
        let weight = (-(norm_sq_fp / PHI_FP)).exp();
        
        // Theta term: exp(-π|n|²/t)
        // arg = -PI * norm_sq / t
        let theta_term = (-(PI_FP * norm_sq_fp / t_fp)).exp();
        
        sum_fp = sum_fp + weight * theta_term;
    }

    sum_fp.to_f64()
}

/// Compute theta series with custom weights.
///
/// Args:
///     windings: List of WindingState instances
///     weights: List of weights corresponding to each winding
///     t: Temperature parameter
///
/// Returns:
///     Weighted theta series value at t
#[pyfunction]
pub fn theta_series_weighted(windings: Vec<WindingState>, weights: Vec<f64>, t: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let mut sum_fp = FixedPoint64::zero();
    if t_fp.0 == 0 { return 0.0; }

    for (w, weight_val) in windings.iter().zip(weights.iter()) {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        let weight_fp = FixedPoint64::from_f64(*weight_val);
        
        let theta_term = (-(PI_FP * norm_sq_fp / t_fp)).exp();
        sum_fp = sum_fp + weight_fp * theta_term;
    }

    sum_fp.to_f64()
}

/// Compute theta series derivative: dΘ/dt = (π/t²) Σ_n w(n) |n|² exp(-π|n|²/t)
#[pyfunction]
pub fn theta_series_derivative(windings: Vec<WindingState>, t: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    if t_fp.0 == 0 { return 0.0; }

    let t_sq_fp = t_fp * t_fp;
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        
        // weight = exp(-|n|²/φ)
        let weight = (-(norm_sq_fp / PHI_FP)).exp();
        
        // theta_term = exp(-π|n|²/t)
        let theta_term = (-(PI_FP * norm_sq_fp / t_fp)).exp();
        
        sum_fp = sum_fp + weight * norm_sq_fp * theta_term;
    }

    let factor = PI_FP / t_sq_fp;
    (factor * sum_fp).to_f64()
}

// =============================================================================
// Heat Kernel Operations
// =============================================================================

/// Compute heat kernel trace: K(t) = Σ_n exp(-λ_n t)
///
/// Args:
///     windings: List of WindingState instances
///     t: Time parameter
///     base_eigenvalue: Base eigenvalue scale (λ_n = base * |n|²)
///
/// Returns:
///     Heat kernel trace at time t
#[pyfunction]
pub fn heat_kernel_trace(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        let eigenvalue = base_fp * norm_sq_fp;
        
        sum_fp = sum_fp + (-(eigenvalue * t_fp)).exp();
    }

    sum_fp.to_f64()
}

/// Compute weighted heat kernel trace: K(t) = Σ_n w(n) exp(-λ_n t)
#[pyfunction]
pub fn heat_kernel_weighted(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        
        // weight = exp(-|n|²/φ)
        let weight = (-(norm_sq_fp / PHI_FP)).exp();
        
        let eigenvalue = base_fp * norm_sq_fp;
        sum_fp = sum_fp + weight * (-(eigenvalue * t_fp)).exp();
    }

    sum_fp.to_f64()
}

/// Compute heat kernel derivative: dK/dt = -Σ_n λ_n exp(-λ_n t)
#[pyfunction]
pub fn heat_kernel_derivative(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        let eigenvalue = base_fp * norm_sq_fp;
        
        sum_fp = sum_fp + eigenvalue * (-(eigenvalue * t_fp)).exp();
    }

    (-sum_fp).to_f64()
}

// =============================================================================
// Eigenvalue Operations
// =============================================================================

/// Batch compute eigenvalues for all windings: λ_n = base * |n|²
///
/// Args:
///     windings: List of WindingState instances
///     base: Base eigenvalue scale
///
/// Returns:
///     List of eigenvalues
#[pyfunction]
pub fn compute_eigenvalues(windings: Vec<WindingState>, base: f64) -> Vec<f64> {
    let base_fp = FixedPoint64::from_f64(base);
    windings
        .iter()
        .map(|w| {
            let norm_sq = w.norm_squared();
            let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
            (base_fp * norm_sq_fp).to_f64()
        })
        .collect()
}

/// Batch compute golden weights for all windings: w(n) = exp(-|n|²/φ)
#[pyfunction]
pub fn compute_golden_weights(windings: Vec<WindingState>) -> Vec<f64> {
    windings
        .iter()
        .map(|w| {
            let norm_sq = w.norm_squared();
            let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
            (-(norm_sq_fp / PHI_FP)).exp().to_f64()
        })
        .collect()
}

/// Batch compute norm squared for all windings
#[pyfunction]
pub fn compute_norm_squared(windings: Vec<WindingState>) -> Vec<i64> {
    windings.iter().map(|w| w.norm_squared()).collect()
}

// =============================================================================
// Spectral Zeta Function
// =============================================================================

/// Compute spectral zeta function: ζ(s) = Σ_{n≠0} λ_n^(-s)
///
/// Args:
///     windings: List of WindingState instances
///     s: Complex exponent (real part)
///     base_eigenvalue: Base eigenvalue scale
///
/// Returns:
///     Spectral zeta function value
#[pyfunction]
pub fn spectral_zeta(windings: Vec<WindingState>, s: f64, base_eigenvalue: f64) -> f64 {
    let s_fp = FixedPoint64::from_f64(s);
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    // Validate inputs using fixed-point representations
    if base_fp.0 == 0 {
        return 0.0; // Zero base eigenvalue means zero zeta
    }

    // Check if s is close to an integer for potential exact computation
    let s_rounded = s.round();
    let s_is_integer = (s - s_rounded).abs() < 1e-10 && s_rounded.abs() < 64.0;

    for w in &windings {
        let norm_sq = w.norm_squared();
        if norm_sq > 0 {
            let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
            // Compute eigenvalue in fixed-point: λ = base * |n|²
            let eigenvalue_fp = base_fp * norm_sq_fp;

            // λ^(-s) computation:
            // For integer exponents, use exact fixed-point powi
            // For fractional exponents, use hybrid f64 for the power
            let pow_term = if s_is_integer {
                // Use exact integer power via fixed-point
                let s_int = s_rounded as i64;
                if s_int > 0 {
                    // λ^(-s) = 1/λ^s
                    FixedPoint64::one() / eigenvalue_fp.powi(s_int)
                } else {
                    // λ^(-s) = λ^|s|
                    eigenvalue_fp.powi(-s_int)
                }
            } else {
                // Fractional exponent: use f64 for powf, then convert back
                // Validate s_fp is reasonable (prevents extreme values)
                if s_fp.0.abs() > (100i64 << 32) {
                    // Exponent too large - skip this term to avoid overflow
                    continue;
                }
                let eigenvalue_f64 = eigenvalue_fp.to_f64();
                FixedPoint64::from_f64(eigenvalue_f64.powf(-s))
            };

            sum_fp = sum_fp + pow_term;
        }
    }
    sum_fp.to_f64()
}

/// Compute weighted spectral zeta: ζ_w(s) = Σ_{n≠0} w(n) λ_n^(-s)
#[pyfunction]
pub fn spectral_zeta_weighted(windings: Vec<WindingState>, s: f64, base_eigenvalue: f64) -> f64 {
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let norm_sq = w.norm_squared();
        if norm_sq > 0 {
            let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
            let weight = (-(norm_sq_fp / PHI_FP)).exp();
            
            // Hybrid fallback for powf
            let eigenvalue = base_eigenvalue * (norm_sq as f64);
            let pow_term = FixedPoint64::from_f64(eigenvalue.powf(-s));
            
            sum_fp = sum_fp + weight * pow_term;
        }
    }
    sum_fp.to_f64()
}

// =============================================================================
// Partition Function
// =============================================================================

/// Compute partition function: Z = Σ_n exp(-|n|²/φ)
#[pyfunction]
pub fn partition_function(windings: Vec<WindingState>) -> f64 {
    let mut sum_fp = FixedPoint64::zero();
    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        sum_fp = sum_fp + (-(norm_sq_fp / PHI_FP)).exp();
    }
    sum_fp.to_f64()
}

/// Compute combined theta sum: Θ_c(t) = Σ_n exp(-|n|² * (1/φ + π/t))
///
/// This efficiently computes the product of golden measure and theta series.
#[pyfunction]
pub fn theta_sum_combined(windings: Vec<WindingState>, t: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    if t_fp.0 == 0 { return 0.0; }
    
    // (1/phi + pi/t)
    let combined_factor = (FixedPoint64::one() / PHI_FP) + (PI_FP / t_fp);
    
    let mut sum_fp = FixedPoint64::zero();
    for w in &windings {
        let norm_sq = w.norm_squared();
        let norm_sq_fp = FixedPoint64::new(norm_sq << 32);
        sum_fp = sum_fp + (-(norm_sq_fp * combined_factor)).exp();
    }
    sum_fp.to_f64()
}

// =============================================================================
// Knot Laplacian Operations
// =============================================================================

/// Compute knot Laplacian eigenvalue: λ_n = base * |n|² * (1 + φ^(-|n|²))
///
/// This includes the golden-weighted knot potential correction.
#[pyfunction]
pub fn knot_eigenvalue(norm_squared: i64, base: f64) -> f64 {
    let base_fp = FixedPoint64::from_f64(base);
    let n_sq_fp = FixedPoint64::new(norm_squared << 32);
    
    // correction = PHI^(-n_sq)
    let correction = PHI_FP.powi(-norm_squared);
    
    // base * n_sq * (1 + correction)
    let result = base_fp * n_sq_fp * (FixedPoint64::one() + correction);
    result.to_f64()
}

/// Batch compute knot eigenvalues for all windings
#[pyfunction]
pub fn compute_knot_eigenvalues(windings: Vec<WindingState>, base: f64) -> Vec<f64> {
    let base_fp = FixedPoint64::from_f64(base);
    windings
        .iter()
        .map(|w| {
            let n_sq = w.norm_squared();
            let n_sq_fp = FixedPoint64::new(n_sq << 32);
            let correction = PHI_FP.powi(-n_sq);
            let result = base_fp * n_sq_fp * (FixedPoint64::one() + correction);
            result.to_f64()
        })
        .collect()
}

/// Compute knot heat kernel trace: K(t) = Σ_n exp(-λ_knot_n * t)
///
/// Uses knot eigenvalues λ_n = base * |n|² * (1 + φ^(-|n|²))
#[pyfunction]
pub fn knot_heat_kernel_trace(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let t_fp = FixedPoint64::from_f64(t);
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let n_sq = w.norm_squared();
        let n_sq_fp = FixedPoint64::new(n_sq << 32);
        
        // eigenvalue calculation
        let correction = PHI_FP.powi(-n_sq);
        let eigenvalue = base_fp * n_sq_fp * (FixedPoint64::one() + correction);
        
        sum_fp = sum_fp + (-(eigenvalue * t_fp)).exp();
    }

    sum_fp.to_f64()
}

/// Compute knot spectral zeta: ζ(s) = Σ_{n≠0} λ_knot_n^(-s)
///
/// Uses knot eigenvalues λ_n = base * |n|² * (1 + φ^(-|n|²))
#[pyfunction]
pub fn knot_spectral_zeta(windings: Vec<WindingState>, s: f64, base_eigenvalue: f64) -> f64 {
    let base_fp = FixedPoint64::from_f64(base_eigenvalue);
    let mut sum_fp = FixedPoint64::zero();

    for w in &windings {
        let n_sq = w.norm_squared();
        if n_sq > 0 {
            // Calculate eigenvalue in FixedPoint to be consistent with kernel
            let n_sq_fp = FixedPoint64::new(n_sq << 32);
            let correction = PHI_FP.powi(-n_sq);
            let eigenvalue_fp = base_fp * n_sq_fp * (FixedPoint64::one() + correction);
            
            // Fallback to f64 for powf(-s) until ln() is available
            let val_f64 = eigenvalue_fp.to_f64();
            let term = val_f64.powf(-s);
            sum_fp = sum_fp + FixedPoint64::from_f64(term);
        }
    }

    sum_fp.to_f64()
}

/// Compute knot spectral zeta with complex s parameter
#[pyfunction]
pub fn knot_spectral_zeta_complex(
    windings: Vec<WindingState>,
    s_real: f64,
    s_imag: f64,
    base_eigenvalue: f64,
) -> (f64, f64) {
    let mut sum_real = Rational::from_integer(0);
    let mut sum_imag = Rational::from_integer(0);

    for w in &windings {
        let n_sq = w.norm_squared();
        if n_sq > 0 {
            let n_sq_f = n_sq as f64;
            let correction = PHI.powf(-n_sq_f);
            let eigenvalue = base_eigenvalue * n_sq_f * (Rational::from_integer(1).eval_f64() + correction);

            // λ^(-s) = exp(-s * ln(λ)) = exp(-(s_r + i*s_i) * ln(λ))
            let ln_lambda = eigenvalue.ln();
            let exp_arg_real = -s_real * ln_lambda;
            let exp_arg_imag = -s_imag * ln_lambda;

            // Normalize imaginary argument to [-π, π] for numerical stability
            let normalized_arg = exp_arg_imag - (Rational::from_integer(2).eval_f64() * PI) * (exp_arg_imag / (Rational::from_integer(2).eval_f64() * PI)).round();

            let magnitude = exp_arg_real.exp();
            sum_real = sum_real.add(&Rational::from_f64_approx(magnitude * normalized_arg.cos()));
            sum_imag = sum_imag.add(&Rational::from_f64_approx(magnitude * normalized_arg.sin()));
        }
    }

    (sum_real.eval_f64(), sum_imag.eval_f64())
}

// =============================================================================
// Generation Statistics
// =============================================================================

/// Count windings by generation.
///
/// Returns a HashMap mapping generation number to count.
#[pyfunction]
pub fn count_by_generation(windings: Vec<WindingState>) -> std::collections::HashMap<i64, usize> {
    let mut result = std::collections::HashMap::new();

    for w in &windings {
        let gen = w.generation();
        *result.entry(gen).or_insert(0) += 1;
    }

    result
}

/// Filter windings by generation.
#[pyfunction]
pub fn filter_by_generation(windings: Vec<WindingState>, generation: i64) -> Vec<WindingState> {
    windings
        .into_iter()
        .filter(|w| w.generation() == generation)
        .collect()
}
