//! Syntony computation for winding states.
//!
//! Computes syntony S(Ψ) using winding-aware mode norms:
//!
//!     S(Ψ) = Σᵢ |ψᵢ|² × exp(-|nᵢ|²/φ) / Σᵢ |ψᵢ|²
//!
//! where |nᵢ|² is the mode norm squared for each feature.

use super::number_theory::golden_weight;

/// Compute winding syntony for a 1D tensor.
///
/// S(Ψ) = Σ |ψᵢ|² w(nᵢ) / Σ |ψᵢ|²
///
/// where w(n) = exp(-|n|²/φ) is the golden weight.
///
/// # Arguments
/// * `values` - Tensor values (flattened)
/// * `mode_norms` - Mode norm squared |n|² for each feature
///
/// # Returns
/// Syntony S ∈ [0, 1]
pub fn compute_winding_syntony(values: &[f64], mode_norms: &[f64]) -> f64 {
    if values.is_empty() || mode_norms.is_empty() {
        return 0.0;
    }

    let mut weighted_energy = 0.0;
    let mut total_energy = 0.0;

    for (i, &val) in values.iter().enumerate() {
        let norm = if i < mode_norms.len() {
            mode_norms[i]
        } else {
            // Extrapolate mode norm if fewer norms than values
            mode_norms[mode_norms.len() - 1]
        };

        let energy = val * val;
        let weight = golden_weight(norm);

        weighted_energy += energy * weight;
        total_energy += energy;
    }

    if total_energy < 1e-10 {
        return 0.0;
    }

    let syntony = weighted_energy / total_energy;
    syntony.clamp(0.0, 1.0)
}

/// Compute batch syntony for a 2D tensor.
///
/// Returns syntony for each sample in the batch.
///
/// # Arguments
/// * `values` - Tensor values (flattened, row-major)
/// * `batch_size` - Number of samples in batch
/// * `dim` - Feature dimension
/// * `mode_norms` - Mode norm squared |n|² for each feature (length = dim)
///
/// # Returns
/// Vector of syntony values, one per sample
pub fn batch_winding_syntony(
    values: &[f64],
    batch_size: usize,
    dim: usize,
    mode_norms: &[f64],
) -> Vec<f64> {
    let mut syntonies = Vec::with_capacity(batch_size);

    // Precompute golden weights
    let weights: Vec<f64> = (0..dim)
        .map(|i| {
            let norm = if i < mode_norms.len() {
                mode_norms[i]
            } else {
                0.0
            };
            golden_weight(norm)
        })
        .collect();

    for b in 0..batch_size {
        let mut weighted_energy = 0.0;
        let mut total_energy = 0.0;

        for d in 0..dim {
            let idx = b * dim + d;
            if idx >= values.len() {
                break;
            }

            let val = values[idx];
            let energy = val * val;

            weighted_energy += energy * weights[d];
            total_energy += energy;
        }

        let syntony = if total_energy < 1e-10 {
            0.0
        } else {
            (weighted_energy / total_energy).clamp(0.0, 1.0)
        };

        syntonies.push(syntony);
    }

    syntonies
}

/// Aggregate syntony using different methods.
///
/// # Arguments
/// * `syntonies` - Per-sample syntony values
/// * `method` - Aggregation method: "mean", "min", "geometric"
///
/// # Returns
/// Aggregated syntony value
pub fn aggregate_syntony(syntonies: &[f64], method: &str) -> f64 {
    if syntonies.is_empty() {
        return 0.5;
    }

    match method {
        "mean" => syntonies.iter().sum::<f64>() / syntonies.len() as f64,
        "min" => syntonies.iter().cloned().fold(f64::INFINITY, f64::min),
        "geometric" => {
            let product: f64 = syntonies.iter().map(|&s| s.max(1e-10)).product();
            product.powf(1.0 / syntonies.len() as f64)
        }
        _ => syntonies.iter().sum::<f64>() / syntonies.len() as f64,
    }
}

/// Generate standard mode norms based on index.
///
/// Returns |n|² = i² for each feature index i.
///
/// # Arguments
/// * `dim` - Feature dimension
///
/// # Returns
/// Vector of mode norm squared values
pub fn standard_mode_norms(dim: usize) -> Vec<f64> {
    (0..dim).map(|i| (i * i) as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_winding_syntony() {
        // Concentrated energy in low modes → high syntony
        let values = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mode_norms = standard_mode_norms(8);
        let s = compute_winding_syntony(&values, &mode_norms);
        assert!(
            s > 0.8,
            "Expected high syntony for concentrated energy, got {}",
            s
        );
    }

    #[test]
    fn test_scattered_energy() {
        // Scattered energy in high modes → low syntony
        let values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let mode_norms = standard_mode_norms(8);
        let s = compute_winding_syntony(&values, &mode_norms);
        // High mode indices (6, 7) have weights exp(-36/φ) and exp(-49/φ) ≈ 0
        assert!(
            s < 0.1,
            "Expected low syntony for scattered energy, got {}",
            s
        );
    }

    #[test]
    fn test_batch_syntony() {
        // Two samples: one concentrated, one scattered
        let values = vec![
            // Sample 0: concentrated
            1.0, 1.0, 0.0, 0.0, // Sample 1: scattered
            0.0, 0.0, 1.0, 1.0,
        ];
        let mode_norms = standard_mode_norms(4);
        let syntonies = batch_winding_syntony(&values, 2, 4, &mode_norms);

        assert_eq!(syntonies.len(), 2);
        assert!(
            syntonies[0] > syntonies[1],
            "Concentrated should have higher syntony"
        );
    }

    #[test]
    fn test_aggregate_methods() {
        let syntonies = vec![0.6, 0.8, 0.7];

        let mean = aggregate_syntony(&syntonies, "mean");
        assert!((mean - 0.7).abs() < 0.001);

        let min = aggregate_syntony(&syntonies, "min");
        assert!((min - 0.6).abs() < 0.001);

        let geo = aggregate_syntony(&syntonies, "geometric");
        // geometric mean < arithmetic mean
        assert!(geo < mean);
    }
}
