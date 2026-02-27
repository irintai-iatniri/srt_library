//! Python wrappers for number theory and syntony functions.
//!
//! Exposes Rust implementations to Python via PyO3.

use super::number_theory;
use super::syntony;
use pyo3::prelude::*;
use std::time::Duration;

/// Compute the Möbius function μ(n).
///
/// μ(n) = 1 if n is a product of an even number of distinct primes
/// μ(n) = -1 if n is a product of an odd number of distinct primes
/// μ(n) = 0 if n has a squared prime factor
#[pyfunction]
pub fn py_mobius(n: i64) -> i64 {
    number_theory::mobius(n)
}

/// Check if n is square-free (has no squared prime factors).
#[pyfunction]
pub fn py_is_square_free(n: i64) -> bool {
    number_theory::is_square_free(n)
}

/// Compute the Mertens function M(n) = Σ_{k=1}^n μ(k).
#[pyfunction]
pub fn py_mertens(n: usize) -> i64 {
    number_theory::mertens(n)
}

/// Compute golden weight w(n) = exp(-|n|²/φ).
#[pyfunction]
pub fn py_golden_weight(mode_norm_sq: f64) -> f64 {
    number_theory::golden_weight(mode_norm_sq)
}

/// Compute golden weights w(n) = exp(-|n|²/φ) for a vector of mode norms.
#[pyfunction]
pub fn py_golden_weights(mode_norms: Vec<f64>) -> Vec<f64> {
    number_theory::golden_weights(&mode_norms)
}

/// Compute E* = e^π - π ≈ 20.1408...
#[pyfunction]
pub fn py_e_star() -> f64 {
    number_theory::e_star()
}

/// Compute winding syntony for a tensor.
///
/// S(Ψ) = Σ |ψᵢ|² w(nᵢ) / Σ |ψᵢ|²
/// where w(n) = exp(-|n|²/φ) is the golden weight.
#[pyfunction]
pub fn py_compute_winding_syntony(values: Vec<f64>, mode_norms: Vec<f64>) -> f64 {
    syntony::compute_winding_syntony(&values, &mode_norms)
}

/// Compute batch winding syntony for a 2D tensor.
#[pyfunction]
pub fn py_batch_winding_syntony(
    values: Vec<f64>,
    batch_size: usize,
    dim: usize,
    mode_norms: Vec<f64>,
) -> Vec<f64> {
    syntony::batch_winding_syntony(&values, batch_size, dim, &mode_norms)
}

/// Aggregate syntony values using different methods.
#[pyfunction]
pub fn py_aggregate_syntony(syntonies: Vec<f64>, method: &str) -> f64 {
    syntony::aggregate_syntony(&syntonies, method)
}

/// Generate standard mode norms based on index squared.
#[pyfunction]
pub fn py_standard_mode_norms(dim: usize) -> Vec<f64> {
    syntony::standard_mode_norms(dim)
}

// === Crystallization Functions ===

/// Legacy crystallization with dwell time enforcement.
///
/// This function provides the original crystallization algorithm
/// with explicit dwell timing for φ-resonance enforcement.
#[pyfunction]
pub fn py_crystallize_with_dwell_legacy(
    values: Vec<f64>,
    base_precision: i64,
    target_duration_ms: u64,
) -> (Vec<f64>, i64, u64) {
    let target_duration = Duration::from_millis(target_duration_ms);
    let (lattice, precision, actual_duration) =
        super::crystallize::crystallize_with_dwell_legacy(&values, base_precision, target_duration);

    // Convert GoldenExact to f64 for Python
    let lattice_f64 = lattice.iter().map(|g| g.to_f64()).collect();
    let actual_duration_ms = actual_duration.as_millis() as u64;

    (lattice_f64, precision, actual_duration_ms)
}

/// Compute snap distance for crystallization.
///
/// Measures how far values are from their nearest lattice points.
#[pyfunction]
pub fn py_snap_distance(values: Vec<f64>, max_coeff: i64) -> f64 {
    super::crystallize::snap_distance(&values, max_coeff)
}

/// Compute snap gradient for crystallization.
///
/// Computes the gradient towards lattice points for optimization.
#[pyfunction]
pub fn py_compute_snap_gradient(pre_snap: Vec<f64>, post_snap_lattice: Vec<f64>) -> Vec<f64> {
    // Convert f64 post_snap to GoldenExact for the function
    let post_snap: Vec<crate::GoldenExact> = post_snap_lattice
        .iter()
        .map(|&x| crate::GoldenExact::find_nearest(x, 1000))
        .collect();

    super::crystallize::compute_snap_gradient(&pre_snap, &post_snap)
}

/// Compute snap gradient with CUDA acceleration.
///
/// Uses GPU acceleration when available for computing gradients towards lattice points.
#[pyfunction]
pub fn py_compute_snap_gradient_cuda(
    pre_snap: Vec<f64>,
    post_snap_lattice: Vec<f64>,
    mode_norm_sq: Vec<f64>,
) -> Vec<f64> {
    // Convert f64 post_snap to GoldenExact for the function
    let post_snap: Vec<crate::GoldenExact> = post_snap_lattice
        .iter()
        .map(|&x| crate::GoldenExact::find_nearest(x, 1000))
        .collect();

    super::crystallize::compute_snap_gradient_dispatch(&pre_snap, &post_snap, &mode_norm_sq)
}

// === Loss Functions ===

/// Compute Mean Squared Error loss.
#[pyfunction]
pub fn py_mse_loss(pred: Vec<f64>, target: Vec<f64>) -> f64 {
    super::loss::mse_loss(&pred, &target)
}

/// Compute softmax probabilities.
#[pyfunction]
pub fn py_softmax(logits: Vec<f64>) -> Vec<f64> {
    super::loss::softmax(&logits)
}

/// Compute cross-entropy loss.
#[pyfunction]
pub fn py_cross_entropy_loss(logits: Vec<f64>, target: Vec<f64>) -> f64 {
    super::loss::cross_entropy_loss(&logits, &target)
}

/// Compute batch cross-entropy loss.
#[pyfunction]
pub fn py_batch_cross_entropy_loss(
    logits: Vec<f64>,
    targets: Vec<f64>,
    batch_size: usize,
    num_classes: usize,
) -> f64 {
    super::loss::batch_cross_entropy_loss(&logits, &targets, batch_size, num_classes)
}

/// Compute syntony loss component.
#[pyfunction]
pub fn py_syntony_loss(model_syntony: f64, lambda_syntony: f64) -> f64 {
    super::loss::syntony_loss(model_syntony, lambda_syntony)
}

/// Compute SRT-grounded syntony loss using target S* = φ - q.
#[pyfunction]
pub fn py_syntony_loss_srt(model_syntony: f64, lambda_syntony: f64) -> f64 {
    super::loss::syntony_loss_srt(model_syntony, lambda_syntony)
}

/// Get the SRT target syntony S* = φ - q ≈ 1.5906.
#[pyfunction]
pub fn py_get_target_syntony() -> f64 {
    super::loss::get_target_syntony()
}

/// Get the Q-deficit constant q ≈ 0.027395.
#[pyfunction]
pub fn py_get_q_deficit() -> f64 {
    super::loss::get_q_deficit()
}

/// Compute phase alignment loss.
#[pyfunction]
pub fn py_phase_alignment_loss(values: Vec<f64>, target_phase: f64, mu_phase: f64) -> f64 {
    super::loss::phase_alignment_loss(&values, target_phase, mu_phase)
}

/// Compute combined syntonic loss.
/// Returns (total_loss, loss_task, loss_syntony, loss_phase).
#[pyfunction]
pub fn py_syntonic_loss(
    task_loss: f64,
    model_syntony: f64,
    phase_cost: f64,
    lambda_syntony: f64,
    mu_phase: f64,
) -> (f64, f64, f64, f64) {
    super::loss::syntonic_loss(
        task_loss,
        model_syntony,
        phase_cost,
        lambda_syntony,
        mu_phase,
    )
}

/// Estimate syntony from probability distribution.
#[pyfunction]
pub fn py_estimate_syntony_from_probs(probs: Vec<f64>) -> f64 {
    super::loss::estimate_syntony_from_probs(&probs)
}

/// Compute golden-ratio weighted decay loss.
#[pyfunction]
pub fn py_golden_decay_loss(weight_norms: Vec<f64>, lambda_decay: f64) -> f64 {
    super::loss::golden_decay_loss(&weight_norms, lambda_decay)
}

// =============================================================================
// SRT/CRT Prime Theory Functions (New Python Bindings)
// =============================================================================

/// Check if a Mersenne number 2^p - 1 is prime.
/// According to Axiom 6: Stable matter exists iff M_p is prime.
#[pyfunction]
pub fn py_is_mersenne_prime(p: u32) -> bool {
    number_theory::is_mersenne_prime(p)
}

/// Check if a Fermat number 2^(2^n) + 1 is prime.
/// According to CRT: Forces exist iff F_n is prime.
#[pyfunction]
pub fn py_is_fermat_prime(n: u32) -> bool {
    number_theory::is_fermat_prime(n)
}

/// Check if a Lucas number L_n is prime.
/// According to CRT: Dark sectors stabilize iff L_n is prime.
#[pyfunction]
pub fn py_is_lucas_prime(n: u64) -> bool {
    number_theory::is_lucas_prime(n)
}

/// Compute the nth Lucas number L_n = φ^n + (1-φ)^n.
#[pyfunction]
pub fn py_lucas_number(n: u64) -> u128 {
    number_theory::lucas_number(n)
}

/// Compute the Pisano period π(p) for prime p.
/// Determines the "hooking cycle" of prime windings.
#[pyfunction]
pub fn py_pisano_period(p: u64) -> u64 {
    number_theory::pisano_period(p)
}

/// Check if a winding index p generates a stable Mersenne geometry.
/// According to Axiom 6: Stable iff 2^p - 1 is prime.
#[pyfunction]
pub fn py_is_stable_winding(p: u32) -> bool {
    number_theory::is_stable_winding(p)
}

/// Returns the stability barrier where physics changes phase.
/// Currently p=11 where M11 = 23 × 89 (composite).
#[pyfunction]
pub fn py_get_stability_barrier() -> u32 {
    number_theory::get_stability_barrier()
}

/// Check if a number corresponds to a "transcendence gate".
#[pyfunction]
pub fn py_is_transcendence_gate(n: u64) -> bool {
    number_theory::is_transcendence_gate(n)
}

/// Calculate the "versal grip" strength of a prime.
#[pyfunction]
pub fn py_versal_grip_strength(p: u64) -> f64 {
    number_theory::versal_grip_strength(p)
}

/// Generate Mersenne primes up to maximum exponent.
#[pyfunction]
pub fn py_mersenne_sequence(max_p: u32) -> Vec<u64> {
    number_theory::mersenne_sequence(max_p)
}

/// Generate Fermat primes up to maximum index.
#[pyfunction]
pub fn py_fermat_sequence(max_n: u32) -> Vec<u64> {
    number_theory::fermat_sequence(max_n)
}

/// Generate Lucas primes up to maximum index.
#[pyfunction]
pub fn py_lucas_primes(max_n: u64) -> Vec<u64> {
    number_theory::lucas_primes(max_n)
}

/// Compute the Lucas boost factor L_{17}/L_{13} ≈ 6.854.
/// Used for dark matter mass predictions.
#[pyfunction]
pub fn py_lucas_dark_boost() -> f64 {
    number_theory::lucas_dark_boost()
}

/// Predict dark matter mass using Lucas boost.
/// M_dark = M_anchor × (L_{17}/L_{13})
#[pyfunction]
pub fn py_predict_dark_matter_mass(anchor_mass_gev: f64) -> f64 {
    number_theory::predict_dark_matter_mass(anchor_mass_gev)
}

// === Broadcasting Operations ===

use crate::tensor::broadcast;

/// Compute broadcast shape for two shapes.
/// Returns None if incompatible.
#[pyfunction]
pub fn py_broadcast_shape(a: Vec<usize>, b: Vec<usize>) -> Option<Vec<usize>> {
    broadcast::broadcast_shape(&a, &b)
}

/// Check if two shapes are broadcastable.
#[pyfunction]
pub fn py_are_broadcastable(a: Vec<usize>, b: Vec<usize>) -> bool {
    broadcast::are_broadcastable(&a, &b)
}

/// Broadcast addition.
#[pyfunction]
pub fn py_broadcast_add(
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast::broadcast_add(&a, &a_shape, &b, &b_shape)
}

/// Broadcast multiplication.
#[pyfunction]
pub fn py_broadcast_mul(
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast::broadcast_mul(&a, &a_shape, &b, &b_shape)
}

/// Broadcast subtraction.
#[pyfunction]
pub fn py_broadcast_sub(
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast::broadcast_sub(&a, &a_shape, &b, &b_shape)
}

/// Broadcast division.
#[pyfunction]
pub fn py_broadcast_div(
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast::broadcast_div(&a, &a_shape, &b, &b_shape)
}

// === In-place Operations ===

/// In-place add scalar.
#[pyfunction]
pub fn py_inplace_add_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    broadcast::inplace_add_scalar(&mut data, scalar);
    data
}

/// In-place multiply scalar.
#[pyfunction]
pub fn py_inplace_mul_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    broadcast::inplace_mul_scalar(&mut data, scalar);
    data
}

/// In-place subtract scalar.
#[pyfunction]
pub fn py_inplace_sub_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    broadcast::inplace_sub_scalar(&mut data, scalar);
    data
}

/// In-place divide scalar.
#[pyfunction]
pub fn py_inplace_div_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    broadcast::inplace_div_scalar(&mut data, scalar);
    data
}

/// In-place negate.
#[pyfunction]
pub fn py_inplace_negate(mut data: Vec<f64>) -> Vec<f64> {
    broadcast::inplace_negate(&mut data);
    data
}

/// In-place absolute value.
#[pyfunction]
pub fn py_inplace_abs(mut data: Vec<f64>) -> Vec<f64> {
    broadcast::inplace_abs(&mut data);
    data
}

/// In-place clamp.
#[pyfunction]
pub fn py_inplace_clamp(mut data: Vec<f64>, min: f64, max: f64) -> Vec<f64> {
    broadcast::inplace_clamp(&mut data, min, max);
    data
}

/// In-place golden weight: x = exp(-|n|²/φ)
#[pyfunction]
pub fn py_inplace_golden_weight(mut data: Vec<f64>, phi: f64) -> Vec<f64> {
    broadcast::inplace_golden_weight(&mut data, phi);
    data
}

/// Compute linear index from indices and strides.
#[pyfunction]
pub fn py_linear_index(indices: Vec<usize>, strides: Vec<usize>) -> usize {
    broadcast::linear_index(&indices, &strides)
}

// === Convolution Operations ===

use crate::tensor::conv;

/// 2D Convolution.
///
/// Args:
///     input: Flattened input [batch, height, width, in_channels]
///     input_shape: [batch, height, width, in_channels]
///     kernel: Flattened kernel [out_channels, kernel_h, kernel_w, in_channels]
///     kernel_shape: [out_channels, kernel_h, kernel_w, in_channels]
///     stride: (stride_h, stride_w)
///     padding: (pad_h, pad_w)
///
/// Returns:
///     (output_data, output_shape)
#[pyfunction]
pub fn py_conv2d(
    input: Vec<f64>,
    input_shape: Vec<usize>,
    kernel: Vec<f64>,
    kernel_shape: Vec<usize>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f64>, Vec<usize>) {
    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let k_shape: [usize; 4] = [
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        kernel_shape[3],
    ];
    let (output, out_shape) = conv::conv2d(&input, &in_shape, &kernel, &k_shape, stride, padding);
    (output, out_shape.to_vec())
}

/// 2D Max Pooling.
#[pyfunction]
pub fn py_max_pool2d(
    input: Vec<f64>,
    input_shape: Vec<usize>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> (Vec<f64>, Vec<usize>) {
    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output, out_shape) = conv::max_pool2d(&input, &in_shape, pool_size, stride);
    (output, out_shape.to_vec())
}

/// 2D Average Pooling.
#[pyfunction]
pub fn py_avg_pool2d(
    input: Vec<f64>,
    input_shape: Vec<usize>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> (Vec<f64>, Vec<usize>) {
    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output, out_shape) = conv::avg_pool2d(&input, &in_shape, pool_size, stride);
    (output, out_shape.to_vec())
}

/// Global Average Pooling (spatial -> per-channel average).
#[pyfunction]
pub fn py_global_avg_pool2d(input: Vec<f64>, input_shape: Vec<usize>) -> (Vec<f64>, Vec<usize>) {
    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output, out_shape) = conv::global_avg_pool2d(&input, &in_shape);
    (output, out_shape.to_vec())
}
