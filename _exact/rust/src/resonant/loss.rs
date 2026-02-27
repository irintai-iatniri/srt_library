//! Loss functions for SRT neural networks.
//!
//! Provides efficient Rust implementations of:
//! - MSE (Mean Squared Error)
//! - Cross-entropy loss
//! - Syntony loss (task + syntony penalty)
//! - Phase alignment loss

use super::number_theory::PHI;
use crate::constants::SRT_PI as PI;

// Q deficit 2014 canonical import
pub use crate::constants::Q_DEFICIT_CONST as Q_DEFICIT;

/// Target syntony S* = φ - q
pub const S_TARGET: f64 = PHI - Q_DEFICIT;

/// Compute Mean Squared Error loss.
///
/// L = (1/n) Σ (pred_i - target_i)²
///
/// # Arguments
/// * `pred` - Predicted values
/// * `target` - Ground truth values
///
/// # Returns
/// MSE loss value
pub fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    if pred.is_empty() || target.is_empty() {
        return 0.0;
    }

    let n = pred.len().min(target.len());
    let sum_sq: f64 = pred
        .iter()
        .zip(target.iter())
        .take(n)
        .map(|(p, t)| (p - t).powi(2))
        .sum();

    sum_sq / n as f64
}

/// Compute softmax probabilities with numerical stability.
///
/// softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    // Numerical stability: subtract max
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_vals.iter().sum();

    exp_vals.iter().map(|&e| e / sum_exp).collect()
}

/// Compute cross-entropy loss for classification.
///
/// L = -Σ target_i · log(softmax(pred)_i)
///
/// # Arguments
/// * `logits` - Raw logits (before softmax)
/// * `target` - One-hot encoded targets (same length as logits)
///
/// # Returns
/// Cross-entropy loss value
pub fn cross_entropy_loss(logits: &[f64], target: &[f64]) -> f64 {
    if logits.is_empty() || target.is_empty() {
        return 0.0;
    }

    let probs = softmax(logits);
    let epsilon = 1e-10;

    let loss: f64 = target
        .iter()
        .zip(probs.iter())
        .map(|(&t, &p)| -t * (p + epsilon).ln())
        .sum();

    loss
}

/// Compute cross-entropy loss for a batch.
///
/// # Arguments
/// * `logits` - Batch of logits (flattened, row-major)
/// * `targets` - Batch of one-hot targets (flattened, row-major)
/// * `batch_size` - Number of samples
/// * `num_classes` - Number of classes
///
/// # Returns
/// Mean cross-entropy loss over batch
pub fn batch_cross_entropy_loss(
    logits: &[f64],
    targets: &[f64],
    batch_size: usize,
    num_classes: usize,
) -> f64 {
    if batch_size == 0 || num_classes == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    for b in 0..batch_size {
        let start = b * num_classes;
        let end = start + num_classes;

        if end > logits.len() || end > targets.len() {
            break;
        }

        let sample_logits = &logits[start..end];
        let sample_target = &targets[start..end];

        total_loss += cross_entropy_loss(sample_logits, sample_target);
    }

    total_loss / batch_size as f64
}

/// Compute syntony loss component.
///
/// L_syntony = λ × (1 - S_model)
///
/// # Arguments
/// * `model_syntony` - Current model syntony
/// * `lambda_syntony` - Weight for syntony term
///
/// # Returns
/// Syntony loss value
pub fn syntony_loss(model_syntony: f64, lambda_syntony: f64) -> f64 {
    lambda_syntony * (1.0 - model_syntony)
}

/// Compute SRT-grounded syntony loss using target syntony S* = φ - q.
///
/// L_syntony = λ × |S_model - S*|²
///
/// where S* = φ - q ≈ 1.5906 is the theoretical target syntony.
///
/// # Arguments
/// * `model_syntony` - Current model syntony
/// * `lambda_syntony` - Weight for syntony term
///
/// # Returns
/// SRT syntony loss value
pub fn syntony_loss_srt(model_syntony: f64, lambda_syntony: f64) -> f64 {
    let deviation = model_syntony - S_TARGET;
    lambda_syntony * deviation * deviation
}

/// Get the SRT target syntony value S* = φ - q.
///
/// This is the theoretically optimal syntony for converged states.
pub fn get_target_syntony() -> f64 {
    S_TARGET
}

/// Get the Q-deficit constant q ≈ 0.027395.
///
/// This is the universal syntony deficit derived from φ, π, e.
pub fn get_q_deficit() -> f64 {
    Q_DEFICIT
}

/// Compute phase alignment loss.
///
/// C_{iπ} = |estimated_phase - target_phase|²
///
/// Uses variance as proxy for phase (spectral concentration).
///
/// # Arguments
/// * `values` - Output values
/// * `target_phase` - Target phase (default: π/2)
/// * `mu_phase` - Loss weight
///
/// # Returns
/// Phase alignment loss
pub fn phase_alignment_loss(values: &[f64], target_phase: f64, mu_phase: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // Compute variance
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let var: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    // Map variance to phase estimate
    // Low variance → concentrated → phase near 0
    // High variance → spread → phase near π/2
    let normalized_var = var / (1.0 + var);
    let estimated_phase = normalized_var * (PI / 2.0);

    let phase_deviation = (estimated_phase - target_phase).powi(2);
    mu_phase * phase_deviation
}

/// Compute combined syntonic loss.
///
/// L_total = L_task + λ_syntony(1 - S_model) + μ_{iπ}·C_{iπ}
///
/// # Arguments
/// * `task_loss` - Base task loss (MSE or cross-entropy)
/// * `model_syntony` - Current model syntony
/// * `phase_cost` - Phase alignment cost C_{iπ}
/// * `lambda_syntony` - Weight for syntony term
/// * `mu_phase` - Weight for phase term
///
/// # Returns
/// (total_loss, loss_task, loss_syntony, loss_phase)
pub fn syntonic_loss(
    task_loss: f64,
    model_syntony: f64,
    phase_cost: f64,
    lambda_syntony: f64,
    mu_phase: f64,
) -> (f64, f64, f64, f64) {
    let loss_syntony = lambda_syntony * (1.0 - model_syntony);
    let loss_phase = mu_phase * phase_cost;
    let total = task_loss + loss_syntony + loss_phase;

    (total, task_loss, loss_syntony, loss_phase)
}

/// Estimate syntony from output statistics.
///
/// Uses entropy as inverse syntony proxy for classification outputs.
///
/// # Arguments
/// * `probs` - Probability distribution (softmax outputs)
///
/// # Returns
/// Estimated syntony [0, 1]
pub fn estimate_syntony_from_probs(probs: &[f64]) -> f64 {
    if probs.is_empty() {
        return 0.5;
    }

    let epsilon = 1e-10;

    // Shannon entropy: -Σ p_i log(p_i)
    let entropy: f64 = probs
        .iter()
        .map(|&p| if p > epsilon { -p * p.ln() } else { 0.0 })
        .sum();

    let max_entropy = (probs.len() as f64).ln();
    if max_entropy < epsilon {
        return 0.5;
    }

    let normalized_entropy = entropy / max_entropy;

    // Syntony = 1 - normalized_entropy
    (1.0 - normalized_entropy).clamp(0.0, 1.0)
}

/// Compute golden-ratio weighted decay loss.
///
/// L_decay = λ Σ_l (φ^{-l}) ||W_l||²
///
/// # Arguments
/// * `weight_norms` - L2 norms squared of weight matrices per layer
/// * `lambda_decay` - Base decay rate
///
/// # Returns
/// Golden decay loss
pub fn golden_decay_loss(weight_norms: &[f64], lambda_decay: f64) -> f64 {
    let mut total = 0.0;

    for (i, &norm) in weight_norms.iter().enumerate() {
        let scale = PHI.powi(-(i as i32));
        total += scale * norm;
    }

    lambda_decay * total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 4.0];
        let loss = mse_loss(&pred, &target);
        // (0 + 0 + 1) / 3 = 0.333...
        assert!((loss - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Should be increasing
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_cross_entropy() {
        // Perfect prediction
        let logits = vec![10.0, 0.0, 0.0]; // Will softmax to ~[1, 0, 0]
        let target = vec![1.0, 0.0, 0.0];
        let loss = cross_entropy_loss(&logits, &target);
        assert!(loss < 0.01, "Perfect prediction should have near-zero loss");

        // Wrong prediction
        let logits2 = vec![0.0, 0.0, 10.0]; // Will softmax to ~[0, 0, 1]
        let loss2 = cross_entropy_loss(&logits2, &target);
        assert!(loss2 > 5.0, "Wrong prediction should have high loss");
    }

    #[test]
    fn test_syntony_loss() {
        // High syntony → low loss
        let loss_high = syntony_loss(0.9, 0.1);
        // Low syntony → high loss
        let loss_low = syntony_loss(0.3, 0.1);

        assert!(loss_high < loss_low);
    }

    #[test]
    fn test_syntonic_loss_combined() {
        let (total, task, syntony, phase) = syntonic_loss(
            1.0,  // task_loss
            0.8,  // model_syntony
            0.1,  // phase_cost
            0.1,  // lambda_syntony
            0.01, // mu_phase
        );

        // Verify task loss is preserved unchanged
        assert!((task - 1.0).abs() < 1e-10, "Task loss should be preserved as 1.0");
        assert!((syntony - 0.02).abs() < 1e-10); // 0.1 * (1 - 0.8) = 0.02
        assert!((phase - 0.001).abs() < 1e-10); // 0.01 * 0.1 = 0.001
        assert!((total - (task + syntony + phase)).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_syntony() {
        // Uniform distribution → low syntony (high entropy)
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let s_uniform = estimate_syntony_from_probs(&uniform);

        // Concentrated distribution → high syntony (low entropy)
        let concentrated = vec![0.97, 0.01, 0.01, 0.01];
        let s_concentrated = estimate_syntony_from_probs(&concentrated);

        assert!(
            s_concentrated > s_uniform,
            "Concentrated should have higher syntony"
        );
    }

    #[test]
    fn test_golden_decay() {
        let weights = vec![1.0, 1.0, 1.0]; // Same norm per layer
        let decay = golden_decay_loss(&weights, 0.01);

        // Decay: φ^0 + φ^-1 + φ^-2 = 1 + 0.618 + 0.382 ≈ 2.0
        assert!(decay > 0.019 && decay < 0.021);
    }
}
