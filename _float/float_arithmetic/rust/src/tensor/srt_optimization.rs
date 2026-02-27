//! SRT-Based Optimization Algorithms
//!
//! Implements the Golden Momentum optimizer as described in The Grand Synthesis.
//! The key insight: beta = 1/φ provides natural temporal decay aligned with SRT.

use pyo3::prelude::*;

/// Golden ratio constant
const PHI: f64 = 1.618033988749895;

/// GoldenMomentum optimizer state.
///
/// This optimizer uses phi-based momentum where beta = 1/φ ≈ 0.618.
/// The system retains ~61.8% of its past velocity at every step,
/// making it resistant to short-term noise while responsive to persistent gradients.
#[pyclass]
#[derive(Debug, Clone)]
pub struct GoldenMomentum {
    /// Velocity buffer (momentum accumulator)
    velocity: Vec<f64>,
    /// Momentum coefficient: 1/φ ≈ 0.618
    beta: f64,
    /// Learning rate
    lr: f64,
    /// Size of parameter vector
    size: usize,
}

#[pymethods]
impl GoldenMomentum {
    /// Create a new GoldenMomentum optimizer.
    ///
    /// # Arguments
    /// * `size` - Number of parameters to optimize
    /// * `lr` - Learning rate (default: 0.01)
    #[new]
    #[pyo3(signature = (size, lr=0.01))]
    pub fn new(size: usize, lr: f64) -> Self {
        // beta = 1/φ - the golden inertia
        // This means the system retains ~61.8% of its past identity at every step
        let beta = 1.0 / PHI;

        GoldenMomentum {
            velocity: vec![0.0; size],
            beta,
            lr,
            size,
        }
    }

    /// Perform a single optimization step.
    ///
    /// Updates weights in-place using the golden momentum rule:
    /// v(t+1) = (1/φ) * v(t) + gradient
    /// w(t+1) = w(t) - lr * v(t+1)
    ///
    /// # Arguments
    /// * `weights` - Mutable weight vector (updated in-place)
    /// * `gradients` - Gradient vector
    ///
    /// # Returns
    /// Updated weights as a new vector
    pub fn step(&mut self, weights: Vec<f64>, gradients: Vec<f64>) -> PyResult<Vec<f64>> {
        if weights.len() != self.size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Weight size {} doesn't match optimizer size {}",
                weights.len(),
                self.size
            )));
        }
        if gradients.len() != self.size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Gradient size {} doesn't match optimizer size {}",
                gradients.len(),
                self.size
            )));
        }

        let mut new_weights = weights;

        for i in 0..self.size {
            // 1. Update Velocity (The History of the Path)
            // v(t+1) = (1/φ) * v(t) + gradient
            self.velocity[i] = (self.beta * self.velocity[i]) + gradients[i];

            // 2. Apply Update (The Choice)
            // w(t+1) = w(t) - lr * v(t+1)
            new_weights[i] -= self.lr * self.velocity[i];
        }

        Ok(new_weights)
    }

    /// Reset velocity to zero.
    pub fn reset(&mut self) {
        self.velocity = vec![0.0; self.size];
    }

    /// Get the current learning rate.
    #[getter]
    pub fn get_lr(&self) -> f64 {
        self.lr
    }

    /// Set the learning rate.
    #[setter]
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Get the momentum coefficient (1/φ).
    #[getter]
    pub fn get_beta(&self) -> f64 {
        self.beta
    }

    /// Get the current velocity buffer.
    #[getter]
    pub fn get_velocity(&self) -> Vec<f64> {
        self.velocity.clone()
    }

    /// Get the optimizer size.
    #[getter]
    pub fn get_size(&self) -> usize {
        self.size
    }
}

/// Pure Rust implementation for batch optimization.
impl GoldenMomentum {
    /// Step with SIMD-friendly layout (for future CUDA extension).
    pub fn step_batch(&mut self, weights: &mut [f64], gradients: &[f64]) {
        debug_assert_eq!(weights.len(), self.size);
        debug_assert_eq!(gradients.len(), self.size);

        for i in 0..self.size {
            self.velocity[i] = self.beta * self.velocity[i] + gradients[i];
            weights[i] -= self.lr * self.velocity[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_momentum_creation() {
        let optim = GoldenMomentum::new(10, 0.01);
        assert_eq!(optim.size, 10);
        assert!((optim.beta - (1.0 / PHI)).abs() < 1e-10);
        assert_eq!(optim.velocity.len(), 10);
    }

    #[test]
    fn test_golden_momentum_step() {
        let mut optim = GoldenMomentum::new(3, 0.1);
        let weights = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];

        let new_weights = optim.step(weights, gradients).unwrap();

        // After first step:
        // v = 0 * beta + grad = grad
        // w_new = w - lr * v = w - lr * grad
        assert!((new_weights[0] - 0.99).abs() < 1e-10); // 1.0 - 0.1 * 0.1
        assert!((new_weights[1] - 1.98).abs() < 1e-10); // 2.0 - 0.1 * 0.2
        assert!((new_weights[2] - 2.97).abs() < 1e-10); // 3.0 - 0.1 * 0.3
    }

    #[test]
    fn test_momentum_accumulation() {
        let mut optim = GoldenMomentum::new(1, 0.1);

        // Step 1
        let w1 = optim.step(vec![1.0], vec![1.0]).unwrap();
        // v = 1.0, w = 1.0 - 0.1 * 1.0 = 0.9
        assert!((w1[0] - 0.9).abs() < 1e-10);

        // Step 2 - momentum should accumulate
        let w2 = optim.step(w1, vec![1.0]).unwrap();
        // v = beta * 1.0 + 1.0 = 1/φ + 1 ≈ 1.618
        // w = 0.9 - 0.1 * 1.618 ≈ 0.738
        let expected_v = (1.0 / PHI) + 1.0;
        let expected_w = 0.9 - 0.1 * expected_v;
        assert!((w2[0] - expected_w).abs() < 1e-10);
    }
}
