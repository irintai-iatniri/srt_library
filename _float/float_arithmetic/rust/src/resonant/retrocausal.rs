/*!
Retrocausal Harmonization for RES.

Extends standard harmonization (Ĥ) with attractor-guided influence. Instead
of just damping non-golden modes, harmonization also "pulls" toward proven
high-syntony configurations stored in attractor memory.

From CRT_Altruxa_Bridge.md §17: Future high-syntony states exert retrocausal
influence on parameter evolution through biased harmonization.

# Mathematical Formulation

Standard harmonization:
```text
Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
```

Retrocausal harmonization:
```text
Ĥ_retro[ψ]ₙ = (1 - λ_retro) × Ĥ[ψ]ₙ + λ_retro × Σᵢ wᵢ × (Aᵢ,ₙ - ψₙ)
```

Where:
- `Aᵢ` = attractor i lattice values
- `wᵢ` = weight of attractor i (syntony² × temporal_decay)
- `λ_retro` = retrocausal pull strength
*/

use super::attractor::AttractorMemory;
use super::tensor::ResonantTensor;
use super::ResonantError;
use crate::exact::golden::GoldenExact;

/// Apply harmonization with attractor-guided retrocausal influence.
///
/// # Arguments
/// * `tensor` - Tensor to harmonize (modified in place)
/// * `attractor_memory` - Memory of high-syntony attractors
/// * `pull_strength` - λ_retro (0.0 = standard Ĥ, 1.0 = full attractor pull)
///
/// # Returns
/// Updated syntony after harmonization
///
/// # Example
/// ```ignore
/// use syntonic::resonant::{harmonize_with_attractor_pull, AttractorMemory};
///
/// let mut tensor = ResonantTensor::from_floats_default_modes(...);
/// let memory = AttractorMemory::new(32, 0.7, 0.98);
///
/// let syntony = harmonize_with_attractor_pull(
///     &mut tensor,
///     &memory,
///     0.3  // Moderate pull
/// )?;
/// ```
pub fn harmonize_with_attractor_pull(
    tensor: &mut ResonantTensor,
    attractor_memory: &AttractorMemory,
    pull_strength: f64,
) -> Result<f64, ResonantError> {
    // Validate pull strength
    let lambda_retro = pull_strength.clamp(0.0, 1.0);

    // If no attractors or zero pull, use standard harmonization
    if attractor_memory.is_empty() || lambda_retro < 1e-10 {
        return standard_harmonization(tensor);
    }

    // Step 1: Compute standard harmonization target
    let h_standard = compute_standard_harmonization_target(tensor)?;

    // Step 2: Compute attractor pull
    let attractor_pull = attractor_memory.compute_attractor_pull(tensor)?;

    // Step 3: Blend harmonization with attractor pull
    //         H_retro = (1 - λ) × H_standard + λ × attractor_pull
    let blended = blend_harmonization(&h_standard, &attractor_pull, lambda_retro);

    // Step 4: Apply blended harmonization
    apply_harmonization_values(tensor, &blended)?;

    // Step 5: Crystallize to snap back to Q(φ) lattice
    // Convert blended values to floats for crystallization
    let blended_floats: Vec<f64> = blended.iter().map(|g| g.to_f64()).collect();
    let precision = tensor.precision();
    tensor.crystallize_cpu(&blended_floats, precision)?;

    Ok(tensor.syntony())
}

/// Compute standard harmonization target (without attractor influence).
///
/// Returns the values that standard Ĥ would produce.
fn compute_standard_harmonization_target(
    tensor: &ResonantTensor,
) -> Result<Vec<GoldenExact>, ResonantError> {
    // Standard Ĥ formula:
    // Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
    //
    // Where:
    // - β(S) = syntony-dependent attenuation factor
    // - w(n) = golden weight for mode n

    let syntony = tensor.syntony();
    let beta = compute_beta(syntony);

    let lattice = tensor.lattice();
    let mode_norm_sq = tensor.mode_norm_sq();
    let mut harmonized = Vec::with_capacity(lattice.len());

    for (i, &lattice_val) in lattice.iter().enumerate() {
        // Get mode norm for this index
        let mode_norm = if i < mode_norm_sq.len() {
            mode_norm_sq[i]
        } else {
            0.0
        };

        // Golden weight: w(n) = exp(-|n|²/φ)
        let golden_weight = compute_golden_weight(mode_norm);

        // Attenuation factor: (1 - β × (1 - w))
        let attenuation = 1.0 - beta * (1.0 - golden_weight);

        // Convert attenuation to GoldenExact
        let attenuation_golden = GoldenExact::find_nearest(attenuation, 1000);

        // Apply: ψ × attenuation
        let harmonized_val = lattice_val * attenuation_golden;
        harmonized.push(harmonized_val);
    }

    Ok(harmonized)
}

/// Blend standard harmonization with attractor pull.
///
/// Returns: (1 - λ) × H_standard + λ × attractor_pull
fn blend_harmonization(
    h_standard: &[GoldenExact],
    attractor_pull: &[GoldenExact],
    lambda_retro: f64,
) -> Vec<GoldenExact> {
    let one_minus_lambda = 1.0 - lambda_retro;

    // Convert scalars to GoldenExact with sufficient precision
    let lambda_golden = GoldenExact::find_nearest(lambda_retro, 1000);
    let one_minus_lambda_golden = GoldenExact::find_nearest(one_minus_lambda, 1000);

    h_standard
        .iter()
        .zip(attractor_pull.iter())
        .map(|(&h, &pull)| h * one_minus_lambda_golden + pull * lambda_golden)
        .collect()
}

/// Apply harmonization values to tensor.
fn apply_harmonization_values(
    tensor: &mut ResonantTensor,
    values: &[GoldenExact],
) -> Result<(), ResonantError> {
    tensor.set_lattice(values)
}

/// Standard harmonization (no attractor influence).
fn standard_harmonization(tensor: &mut ResonantTensor) -> Result<f64, ResonantError> {
    let h_standard = compute_standard_harmonization_target(tensor)?;

    // Convert to floats for crystallization
    let h_floats: Vec<f64> = h_standard.iter().map(|g| g.to_f64()).collect();
    let precision = tensor.precision();
    tensor.crystallize_cpu(&h_floats, precision)?;

    Ok(tensor.syntony())
}

/// Compute β(S) - syntony-dependent attenuation factor.
///
/// Higher syntony → more attenuation of non-golden modes.
fn compute_beta(syntony: f64) -> f64 {
    // Standard RES uses β = λ (universal syntony deficit)
    // For retrocausal, we use syntony-dependent β
    //
    // β(S) = λ × (1 + S) / 2
    //
    // This makes harmonization stronger as syntony increases.
    const LAMBDA: f64 = 0.027395146920; // Q_DEFICIT
    LAMBDA * (1.0 + syntony) / 2.0
}

/// Compute golden weight for a mode.
///
/// w(n) = exp(-|n|²/φ)
fn compute_golden_weight(mode_norm_sq: f64) -> f64 {
    const PHI: f64 = 1.6180339887498948482;
    (-mode_norm_sq / PHI).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Rational;

    #[test]
    fn test_standard_harmonization() {
        let mut tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0], vec![3], 100).unwrap();

        let initial_syntony = tensor.syntony();
        let syntony = standard_harmonization(&mut tensor).unwrap();

        // Syntony should change after harmonization
        assert_ne!(syntony, initial_syntony);
    }

    #[test]
    fn test_retrocausal_harmonization_no_attractors() {
        let mut tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0], vec![3], 100).unwrap();

        let memory = AttractorMemory::new(10, 0.7, 0.95);

        // Should behave like standard harmonization
        let syntony = harmonize_with_attractor_pull(&mut tensor, &memory, 0.3).unwrap();
        assert!(syntony >= 0.0 && syntony <= 1.0);
    }

    #[test]
    fn test_retrocausal_harmonization_with_attractors() {
        let mut tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0], vec![3], 100).unwrap();

        // Create attractor memory with one high-syntony attractor
        let mut memory = AttractorMemory::new(10, 0.5, 0.95);
        let attractor =
            ResonantTensor::from_floats_default_modes(&[2.0, 3.0, 4.0], vec![3], 100).unwrap();
        memory.maybe_add(&attractor, 0.9, 0);

        // Apply retrocausal harmonization
        let syntony = harmonize_with_attractor_pull(&mut tensor, &memory, 0.5).unwrap();
        assert!(syntony >= 0.0 && syntony <= 1.0);
    }

    #[test]
    fn test_compute_golden_weight() {
        // Mode 0 (DC): weight should be 1.0
        assert!((compute_golden_weight(0.0) - 1.0).abs() < 1e-10);

        // High-frequency modes should have lower weight
        let w_low = compute_golden_weight(1.0);
        let w_high = compute_golden_weight(10.0);
        assert!(w_high < w_low);
    }

    #[test]
    fn test_compute_beta() {
        // β should increase with syntony
        let beta_low = compute_beta(0.0);
        let beta_high = compute_beta(1.0);
        assert!(beta_high > beta_low);
    }

    #[test]
    fn test_blend_harmonization() {
        let h_std = vec![
            GoldenExact::find_nearest(1.0, 100),
            GoldenExact::find_nearest(2.0, 100),
        ];
        let pull = vec![
            GoldenExact::find_nearest(3.0, 100),
            GoldenExact::find_nearest(4.0, 100),
        ];

        // With lambda=0, should get standard
        let result = blend_harmonization(&h_std, &pull, 0.0);
        assert_eq!(result[0], h_std[0]);
        assert_eq!(result[1], h_std[1]);

        // With lambda=1, should get pull
        let result = blend_harmonization(&h_std, &pull, 1.0);
        assert_eq!(result[0], pull[0]);
        assert_eq!(result[1], pull[1]);

        // With lambda=0.5, should get average
        let result = blend_harmonization(&h_std, &pull, 0.5);
        let expected_0 = h_std[0] * GoldenExact::from_rational(Rational::new(1, 2))
            + pull[0] * GoldenExact::from_rational(Rational::new(1, 2));
        assert_eq!(result[0], expected_0);
    }
}
