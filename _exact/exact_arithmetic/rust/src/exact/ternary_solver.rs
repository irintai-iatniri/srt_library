//! Ternary Rational Fourier Transform (TRFT)
//!
//! Core implementation of geometric wave decomposition using:
//! - Ternary gradient space {-1, 0, 1} for signal representation
//! - Pythagorean triples for exact rational rotation angles
//! - Dual-basis projection for simultaneous amplitude/phase solving
//! - Zero-drift wave generation via RationalRotator
//!
//! # Architecture
//!
//! TernarySolver decomposes signals into sum of Pythagorean waves:
//!   signal(x) ≈ Σᵢ Aᵢ·sin(θᵢ·x + φᵢ)
//!
//! Where each wave is generated exactly via rotation matrix from
//! Pythagorean triple (aᵢ, bᵢ, hᵢ) with a² + b² = h².

use super::pythagorean::{generate_resonance_ladder_adaptive, PythagoreanTriple};
use super::rational::Rational;
use super::rotator::RationalRotator;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Fixed-point scale for integer-based projection
/// 2^30 provides ~9 decimal digits precision with reasonable headroom for i128
/// (avoids overflow in Rational multiplication with typical signal values)
pub const BASIS_SCALE: i128 = 1 << 30;

/// Wave layer parameters
#[derive(Clone, Debug)]
pub struct WaveLayer {
    /// Pythagorean triple (a, b, h)
    pub triple: PythagoreanTriple,
    /// Gradient amplitude at position 0 (scaled by BASIS_SCALE)
    pub gx0: i128,
    /// Gradient amplitude at position 1 (scaled by BASIS_SCALE)
    pub gy0: i128,
    /// Phase offset (position where wave starts)
    pub phase_offset: i128,
    /// Detection threshold used
    pub threshold: i128,
    /// Energy ratio captured by this layer [0, 1]
    pub energy_ratio: f64,
}

/// Ternary Rational Fourier Transform solver
pub struct TernarySolver {
    /// Resonance ladder of Pythagorean triples
    resonance_ladder: Vec<PythagoreanTriple>,
}

impl TernarySolver {
    /// Create new solver with resonance ladder
    ///
    /// # Arguments
    /// * `target_count` - Number of triples in ladder (~400 recommended)
    ///
    /// # Returns
    /// Initialized TernarySolver
    pub fn new(target_count: usize) -> Self {
        let resonance_ladder = generate_resonance_ladder_adaptive(target_count);
        TernarySolver { resonance_ladder }
    }

    /// Extract ternary gradient from position signal
    ///
    /// Converts amplitude sequence [a₀, a₁, a₂, ...] to gradient {-1, 0, 1}
    /// based on motion: gradient[i] = sign(signal[i+1] - signal[i])
    ///
    /// # Arguments
    /// * `signal` - Position signal (scaled integers)
    ///
    /// # Returns
    /// Ternary gradient sequence
    fn extract_gradient(&self, signal: &[i128]) -> Vec<i8> {
        if signal.is_empty() {
            return Vec::new();
        }

        let mut gradient = Vec::with_capacity(signal.len());

        for i in 0..signal.len() - 1 {
            let delta = signal[i + 1] - signal[i];
            gradient.push(delta.signum() as i8);
        }

        // Last gradient same as previous (or 0 if only one sample)
        gradient.push(*gradient.last().unwrap_or(&0));

        gradient
    }

    /// Compute Median Absolute Deviation for adaptive thresholding
    ///
    /// # Arguments
    /// * `gradient` - Ternary gradient signal
    ///
    /// # Returns
    /// MAD value (integer)
    fn compute_mad(&self, gradient: &[i8]) -> i128 {
        if gradient.is_empty() {
            return 0;
        }

        // Compute median
        let mut sorted: Vec<i128> = gradient.iter().map(|&g| g as i128).collect();
        sorted.sort_unstable();
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2
        } else {
            sorted[sorted.len() / 2]
        };

        // Compute absolute deviations
        let mut deviations: Vec<i128> = gradient
            .iter()
            .map(|&g| (g as i128 - median).abs())
            .collect();
        deviations.sort_unstable();

        // Return median of deviations
        if deviations.len() % 2 == 0 {
            (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2
        } else {
            deviations[deviations.len() / 2]
        }
    }

    /// Generate cosine and sine basis vectors for projection
    ///
    /// # Arguments
    /// * `triple` - Pythagorean triple
    /// * `length` - Signal length
    ///
    /// # Returns
    /// (cosine_basis, sine_basis) as integer vectors scaled by BASIS_SCALE
    fn generate_gradient_basis(
        &self,
        triple: &PythagoreanTriple,
        length: usize,
    ) -> (Vec<i128>, Vec<i128>) {
        // For computational efficiency, work directly with scaled integers
        // instead of full Rational to avoid overflow

        let h = triple.h;
        let a = triple.a;
        let b = triple.b;

        // Start positions for cosine: (h, 0)
        let mut cos_x = h;
        let mut cos_y: i128 = 0;

        // Start positions for sine: (0, h)
        let mut sin_x: i128 = 0;
        let mut sin_y = h;

        let mut cos_basis = Vec::with_capacity(length);
        let mut sin_basis = Vec::with_capacity(length);

        // Generate cosine positions (x-component)
        let mut cos_positions = Vec::with_capacity(length);
        for _ in 0..length {
            // Apply rotation: x' = (a*x - b*y) / h, y' = (b*x + a*y) / h
            let new_x = (a * cos_x - b * cos_y) / h;
            let new_y = (b * cos_x + a * cos_y) / h;
            cos_x = new_x;
            cos_y = new_y;
            cos_positions.push(cos_x);
        }

        // Generate sine positions (x-component with 90° offset)
        let mut sin_positions = Vec::with_capacity(length);
        for _ in 0..length {
            let new_x = (a * sin_x - b * sin_y) / h;
            let new_y = (b * sin_x + a * sin_y) / h;
            sin_x = new_x;
            sin_y = new_y;
            sin_positions.push(sin_x);
        }

        // Convert positions to gradients
        for i in 0..length - 1 {
            let delta_cos = cos_positions[i + 1] - cos_positions[i];
            let delta_sin = sin_positions[i + 1] - sin_positions[i];
            cos_basis.push(delta_cos);
            sin_basis.push(delta_sin);
        }

        // Duplicate last gradient
        cos_basis.push(*cos_basis.last().unwrap_or(&0));
        sin_basis.push(*sin_basis.last().unwrap_or(&0));

        (cos_basis, sin_basis)
    }

    /// Project ternary gradient onto basis with normalization
    ///
    /// Multiplication-free accumulation using ternary {-1, 0, 1}.
    ///
    /// # Arguments
    /// * `gradient` - Ternary gradient signal
    /// * `basis` - Basis vector (integer)
    ///
    /// # Returns
    /// Projection value (unnormalized)
    fn project_ternary(&self, gradient: &[i8], basis: &[i128]) -> i128 {
        let mut acc: i128 = 0;

        for i in 0..gradient.len().min(basis.len()) {
            match gradient[i] {
                1 => acc += basis[i],
                -1 => acc -= basis[i],
                0 => {} // No contribution
                _ => unreachable!("Invalid ternary value"),
            }
        }

        acc
    }

    /// Compute self-dot product of basis (for normalization)
    fn basis_self_dot(&self, basis: &[i128]) -> i128 {
        basis.iter().map(|&b| b * b).sum()
    }

    /// Decompose signal with hint ring buffer
    ///
    /// # Arguments
    /// * `signal` - Position signal (integer)
    /// * `hints` - Ring buffer of recent triples to check first
    /// * `max_layers` - Maximum layers to extract
    /// * `energy_threshold_dominant` - Tier 1 threshold (>20% = 0.20)
    /// * `energy_threshold_subtle` - Tier 2 threshold (>5% = 0.05)
    ///
    /// # Returns
    /// Vec of WaveLayer (empty if no geometric structure found)
    pub fn decompose_with_hints(
        &self,
        signal: &[i128],
        hints: &[(i128, i128, i128)],
        max_layers: usize,
        energy_threshold_dominant: f64,
        energy_threshold_subtle: f64,
    ) -> Vec<WaveLayer> {
        if signal.is_empty() {
            return Vec::new();
        }

        let mut layers = Vec::with_capacity(max_layers);
        let mut residual = signal.to_vec();

        for layer_idx in 0..max_layers {
            // Extract gradient
            let gradient = self.extract_gradient(&residual);
            let mad = self.compute_mad(&gradient);
            let threshold = (mad / 2).max(1); // Ensure minimum threshold of 1

            if mad == 0 {
                break; // No structure remaining (all gradients identical)
            }

            // Check hints first (O(5) fast path)
            let mut best_triple: Option<PythagoreanTriple> = None;
            let mut best_energy = 0i128;
            let mut best_gx0 = 0i128;
            let mut best_gy0 = 0i128;

            for &(a, b, h) in hints {
                let triple = PythagoreanTriple::new(a, b, h);
                let (cos_basis, sin_basis) =
                    self.generate_gradient_basis(&triple, gradient.len());

                let proj_cos = self.project_ternary(&gradient, &cos_basis);
                let proj_sin = self.project_ternary(&gradient, &sin_basis);

                let energy = proj_cos * proj_cos + proj_sin * proj_sin;

                if energy > best_energy {
                    best_energy = energy;
                    best_triple = Some(triple);
                    best_gx0 = proj_cos;
                    best_gy0 = proj_sin;
                }
            }

            // If hint gives >80% of max possible energy, accept immediately
            let signal_energy: i128 = gradient.iter().map(|&g| (g as i128) * (g as i128)).sum();
            let energy_ratio = if signal_energy > 0 {
                (best_energy as f64) / (signal_energy as f64)
            } else {
                0.0
            };

            let hint_accepted = energy_ratio > 0.8;

            // If hint not good enough, search full ladder
            if !hint_accepted {
                #[cfg(feature = "rayon")]
                let search_iter = self.resonance_ladder.par_iter();
                #[cfg(not(feature = "rayon"))]
                let search_iter = self.resonance_ladder.iter();

                let results: Vec<_> = search_iter
                    .map(|triple| {
                        let (cos_basis, sin_basis) =
                            self.generate_gradient_basis(triple, gradient.len());
                        let proj_cos = self.project_ternary(&gradient, &cos_basis);
                        let proj_sin = self.project_ternary(&gradient, &sin_basis);
                        let energy = proj_cos * proj_cos + proj_sin * proj_sin;
                        (*triple, energy, proj_cos, proj_sin)
                    })
                    .collect();

                // Find best
                for (triple, energy, proj_cos, proj_sin) in results {
                    if energy > best_energy {
                        best_energy = energy;
                        best_triple = Some(triple);
                        best_gx0 = proj_cos;
                        best_gy0 = proj_sin;
                    }
                }
            }

            // Check tier thresholds
            let energy_ratio = if signal_energy > 0 {
                (best_energy as f64) / (signal_energy as f64)
            } else {
                0.0
            };

            if layer_idx == 0 && energy_ratio < energy_threshold_dominant {
                break; // First layer must be dominant
            }

            if energy_ratio < energy_threshold_subtle {
                break; // Subsequent layers must exceed subtle threshold
            }

            // Accept layer
            if let Some(triple) = best_triple {
                let layer = WaveLayer {
                    triple,
                    gx0: best_gx0,
                    gy0: best_gy0,
                    phase_offset: 0, // Phase encoded in gx0/gy0
                    threshold,
                    energy_ratio,
                };

                // Subtract reconstructed layer from residual BEFORE pushing to layers
                let reconstructed = self.reconstruct_layer(&layer, residual.len());
                for i in 0..residual.len() {
                    residual[i] -= reconstructed[i];
                }

                layers.push(layer);
            } else {
                break; // No valid triple found
            }
        }

        layers
    }

    /// Reconstruct single wave layer
    ///
    /// # Arguments
    /// * `layer` - Wave parameters
    /// * `length` - Output length
    ///
    /// # Returns
    /// Position signal (integer)
    fn reconstruct_layer(&self, layer: &WaveLayer, length: usize) -> Vec<i128> {
        let (cos_basis, sin_basis) =
            self.generate_gradient_basis(&layer.triple, length);

        // Reconstruct as: gx0 * cos_basis + gy0 * sin_basis
        // Since basis values are already integers (not scaled), we don't divide
        let mut gradient = vec![0i128; length];
        for i in 0..length {
            gradient[i] = layer.gx0 * cos_basis[i] + layer.gy0 * sin_basis[i];
        }

        // Integrate gradient to position
        let mut position = vec![0i128; length];
        position[0] = gradient[0];
        for i in 1..length {
            position[i] = position[i - 1] + gradient[i];
        }

        position
    }

    /// Synthesize composite wave from multiple layers (parallel)
    ///
    /// # Arguments
    /// * `layers` - Wave parameters
    /// * `length` - Output length
    ///
    /// # Returns
    /// Position signal (sum of all layers)
    pub fn synthesize_composite(&self, layers: &[WaveLayer], length: usize) -> Vec<i128> {
        if layers.is_empty() {
            return vec![0; length];
        }

        #[cfg(feature = "rayon")]
        let layer_signals: Vec<Vec<i128>> = layers
            .par_iter()
            .map(|layer| self.reconstruct_layer(layer, length))
            .collect();

        #[cfg(not(feature = "rayon"))]
        let layer_signals: Vec<Vec<i128>> = layers
            .iter()
            .map(|layer| self.reconstruct_layer(layer, length))
            .collect();

        // Sequential summation to prevent race conditions
        let mut composite = vec![0i128; length];
        for signal in layer_signals {
            for i in 0..length {
                composite[i] += signal[i];
            }
        }

        composite
    }

    /// Get resonance ladder size
    pub fn ladder_size(&self) -> usize {
        self.resonance_ladder.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = TernarySolver::new(400);
        assert!(solver.ladder_size() >= 360);
        assert!(solver.ladder_size() <= 440);
    }

    #[test]
    fn test_gradient_extraction() {
        let solver = TernarySolver::new(100);
        let signal = vec![0, 1, 3, 4, 4, 3, 1, 0];
        let gradient = solver.extract_gradient(&signal);

        assert_eq!(gradient.len(), signal.len());
        assert_eq!(gradient[0], 1); // 0→1: increasing
        assert_eq!(gradient[1], 1); // 1→3: increasing
        assert_eq!(gradient[4], -1); // 4→3: decreasing
    }

    #[test]
    fn test_projection_basic() {
        let solver = TernarySolver::new(100);
        let gradient = vec![1, 1, 0, -1, -1];
        let basis = vec![10, 20, 30, 40, 50];

        let projection = solver.project_ternary(&gradient, &basis);
        // 1*10 + 1*20 + 0*30 + (-1)*40 + (-1)*50 = 10 + 20 - 40 - 50 = -60
        assert_eq!(projection, -60);
    }

    #[test]
    fn test_pure_sine_decomposition() {
        let solver = TernarySolver::new(400);

        // Generate pure sine wave using (3, 4, 5) triple
        let triple = PythagoreanTriple::new(3, 4, 5);
        let length = 100;

        let mut rotator = RationalRotator::new(
            Rational::from_int(5),
            Rational::from_int(0),
            triple.a,
            triple.b,
            triple.h,
        );

        let signal: Vec<i128> = (0..length)
            .map(|_| {
                let val = rotator.next();
                val.to_integer_approx(BASIS_SCALE / 100)
            })
            .collect();

        // Decompose
        let hints = vec![(3, 4, 5)];
        let layers = solver.decompose_with_hints(&signal, &hints, 1, 0.20, 0.05);

        // Should find exactly one layer
        assert_eq!(layers.len(), 1);
        assert!(layers[0].energy_ratio > 0.5);
    }
}
