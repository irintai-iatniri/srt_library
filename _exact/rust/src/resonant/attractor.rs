/*!
Attractor Memory for Retrocausal RES.

Stores high-syntony states discovered during evolution as "temporal memory"
of optimal geometries. These attractors exert retrocausal influence on
harmonization, pulling evolution toward proven high-S configurations.

From CRT_Altruxa_Bridge.md §17: Future high-syntony states reach backward
through the DHSR cycle to guide parameter evolution.
*/

use super::tensor::ResonantTensor;
use super::ResonantError;
use crate::exact::golden::GoldenExact;

/// Attractor memory structure storing high-syntony states.
///
/// Acts as temporal memory of optimal geometries discovered during evolution.
/// Older attractors gradually fade via temporal decay.
#[derive(Clone, Debug)]
pub struct AttractorMemory {
    /// Stored high-syntony tensors (lattice snapshots)
    attractors: Vec<ResonantTensor>,

    /// Syntony value for each attractor
    syntony_values: Vec<f64>,

    /// Generation when each attractor was added
    generations: Vec<usize>,

    /// Current generation (for temporal decay)
    current_generation: usize,

    /// Maximum number of attractors to retain
    capacity: usize,

    /// Minimum syntony threshold for storage
    min_syntony: f64,

    /// Temporal decay rate (older attractors fade)
    decay_rate: f64,
}

impl AttractorMemory {
    /// Create new attractor memory.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of attractors to store
    /// * `min_syntony` - Minimum syntony threshold for attractor storage
    /// * `decay_rate` - Temporal fade rate per generation (e.g., 0.98)
    pub fn new(capacity: usize, min_syntony: f64, decay_rate: f64) -> Self {
        Self {
            attractors: Vec::with_capacity(capacity),
            syntony_values: Vec::with_capacity(capacity),
            generations: Vec::with_capacity(capacity),
            current_generation: 0,
            capacity,
            min_syntony,
            decay_rate,
        }
    }

    /// Add a new attractor if it exceeds syntony threshold.
    ///
    /// If capacity is exceeded, removes the weakest attractor
    /// (considering both syntony and temporal weight).
    pub fn maybe_add(&mut self, tensor: &ResonantTensor, syntony: f64, generation: usize) {
        // Check threshold
        if syntony < self.min_syntony {
            return;
        }

        // If at capacity, check if new attractor is better than weakest
        if self.attractors.len() >= self.capacity {
            // Find weakest attractor (lowest effective weight)
            let weakest_idx = self.find_weakest_index();
            let weakest_weight = self.compute_effective_weight(weakest_idx);
            let new_weight = syntony * syntony; // Quadratic preference

            if new_weight <= weakest_weight {
                return; // New attractor not strong enough
            }

            // Replace weakest
            self.attractors[weakest_idx] = tensor.clone();
            self.syntony_values[weakest_idx] = syntony;
            self.generations[weakest_idx] = generation;
        } else {
            // Room available, add directly
            self.attractors.push(tensor.clone());
            self.syntony_values.push(syntony);
            self.generations.push(generation);
        }
    }

    /// Compute weighted influence vector toward attractors.
    ///
    /// Returns a pull vector in Q(φ) lattice space, representing the
    /// weighted centroid of all attractors. Recent high-syntony attractors
    /// have the strongest influence.
    pub fn compute_attractor_pull(
        &self,
        current: &ResonantTensor,
    ) -> Result<Vec<GoldenExact>, ResonantError> {
        if self.attractors.is_empty() {
            // No attractors, return zero pull
            let lattice_len = current.lattice().len();
            return Ok(vec![GoldenExact::zero(); lattice_len]);
        }

        let lattice_len = current.lattice().len();
        let mut weighted_pull = vec![GoldenExact::zero(); lattice_len];
        let mut total_weight = 0.0;

        // Compute weighted pull from each attractor
        for (i, attractor) in self.attractors.iter().enumerate() {
            let syntony = self.syntony_values[i];
            let gen = self.generations[i];

            // Temporal weight: decay^(age)
            let age = if self.current_generation >= gen {
                (self.current_generation - gen) as i32
            } else {
                0
            };
            let temporal_weight = self.decay_rate.powi(age);

            // Syntony weight: prefer high-syntony attractors (quadratic)
            let syntony_weight = syntony * syntony;

            // Combined weight
            let weight = temporal_weight * syntony_weight;
            let weight_golden = GoldenExact::find_nearest(weight, 1000);

            // Compute direction: attractor - current (in Q(φ) lattice)
            let attractor_lattice = attractor.lattice();
            let current_lattice = current.lattice();
            for j in 0..current_lattice.len() {
                let delta = attractor_lattice[j] - current_lattice[j];
                weighted_pull[j] = weighted_pull[j] + delta * weight_golden;
            }

            total_weight += weight;
        }

        // Normalize by total weight
        if total_weight > 1e-15 {
            let scale = 1.0 / total_weight;
            let scale_golden = GoldenExact::find_nearest(scale, 1000);
            for val in &mut weighted_pull {
                *val = *val * scale_golden;
            }
        }

        Ok(weighted_pull)
    }

    /// Apply temporal decay to all attractors.
    ///
    /// Called each generation to fade older attractors.
    pub fn apply_decay(&mut self) {
        self.current_generation += 1;
        // Decay is implicit in compute_attractor_pull (uses age)
    }

    /// Get current number of stored attractors.
    pub fn len(&self) -> usize {
        self.attractors.len()
    }

    /// Check if attractor memory is empty.
    pub fn is_empty(&self) -> bool {
        self.attractors.is_empty()
    }

    /// Get top-k attractors by current effective weight.
    pub fn get_top_attractors(&self, k: usize) -> Vec<&ResonantTensor> {
        let mut indices_weights: Vec<(usize, f64)> = (0..self.attractors.len())
            .map(|i| (i, self.compute_effective_weight(i)))
            .collect();

        // Sort by weight descending
        indices_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        indices_weights
            .into_iter()
            .take(k.min(self.attractors.len()))
            .map(|(i, _)| &self.attractors[i])
            .collect()
    }

    /// Compute effective weight of attractor at index.
    fn compute_effective_weight(&self, idx: usize) -> f64 {
        if idx >= self.attractors.len() {
            return 0.0;
        }

        let syntony = self.syntony_values[idx];
        let gen = self.generations[idx];

        let age = if self.current_generation >= gen {
            (self.current_generation - gen) as i32
        } else {
            0
        };
        let temporal_weight = self.decay_rate.powi(age);
        let syntony_weight = syntony * syntony;

        temporal_weight * syntony_weight
    }

    /// Find index of weakest attractor (lowest effective weight).
    fn find_weakest_index(&self) -> usize {
        let mut weakest_idx = 0;
        let mut weakest_weight = self.compute_effective_weight(0);

        for i in 1..self.attractors.len() {
            let weight = self.compute_effective_weight(i);
            if weight < weakest_weight {
                weakest_weight = weight;
                weakest_idx = i;
            }
        }

        weakest_idx
    }

    /// Clear all attractors.
    pub fn clear(&mut self) {
        self.attractors.clear();
        self.syntony_values.clear();
        self.generations.clear();
        self.current_generation = 0;
    }

    /// Get syntony values for all attractors.
    pub fn get_syntony_values(&self) -> &[f64] {
        &self.syntony_values
    }

    /// Get generations for all attractors.
    pub fn get_generations(&self) -> &[usize] {
        &self.generations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attractor_memory_creation() {
        let memory = AttractorMemory::new(10, 0.7, 0.95);
        assert_eq!(memory.len(), 0);
        assert!(memory.is_empty());
        assert_eq!(memory.capacity, 10);
        assert_eq!(memory.min_syntony, 0.7);
    }

    #[test]
    fn test_maybe_add_above_threshold() {
        let mut memory = AttractorMemory::new(10, 0.7, 0.95);
        let tensor = ResonantTensor::from_floats_default_modes(&[1.0, 2.0], vec![2], 100).unwrap();

        memory.maybe_add(&tensor, 0.8, 0);
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_maybe_add_below_threshold() {
        let mut memory = AttractorMemory::new(10, 0.7, 0.95);
        let tensor = ResonantTensor::from_floats_default_modes(&[1.0, 2.0], vec![2], 100).unwrap();

        memory.maybe_add(&tensor, 0.5, 0);
        assert_eq!(memory.len(), 0); // Not added
    }

    #[test]
    fn test_capacity_enforcement() {
        let mut memory = AttractorMemory::new(3, 0.5, 0.95);

        // Add 4 attractors
        for i in 0..4 {
            let tensor = ResonantTensor::from_floats_default_modes(
                &[i as f64, (i + 1) as f64],
                vec![2],
                100,
            )
            .unwrap();
            let syntony = 0.6 + i as f64 * 0.1;
            memory.maybe_add(&tensor, syntony, i);
        }

        // Should only have 3 (capacity)
        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_compute_attractor_pull_empty() {
        let memory = AttractorMemory::new(10, 0.7, 0.95);
        let current = ResonantTensor::from_floats_default_modes(&[1.0, 2.0], vec![2], 100).unwrap();

        let pull = memory.compute_attractor_pull(&current).unwrap();
        assert_eq!(pull.len(), current.len());
        // All zeros
        for val in pull {
            assert_eq!(val, GoldenExact::zero());
        }
    }

    #[test]
    fn test_temporal_decay() {
        let mut memory = AttractorMemory::new(10, 0.5, 0.95);

        let tensor = ResonantTensor::from_floats_default_modes(&[1.0, 2.0], vec![2], 100).unwrap();
        memory.maybe_add(&tensor, 0.8, 0);

        let initial_weight = memory.compute_effective_weight(0);

        // Advance 10 generations
        for _ in 0..10 {
            memory.apply_decay();
        }

        let decayed_weight = memory.compute_effective_weight(0);
        assert!(decayed_weight < initial_weight);
    }
}
