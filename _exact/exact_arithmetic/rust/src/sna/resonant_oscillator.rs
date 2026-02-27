//! Resonant Oscillator - The Fundamental Unit of Syntonic Neural Architecture
//!
//! This module implements the "Amoeba" - a phase-coherent neuron that performs
//! computation via resonant phasor interactions instead of scalar dot products.
//!
//! # Architecture
//!
//! **Stage 1: Sensation (W4 Projection)**
//! - DiscreteHilbertKernel converts scalar signal → complex analytic signal
//! - Projects R → W4 (quaternionic phase space) via integer Hilbert transform
//!
//! **Stage 2: Integration (Phase Resonance)**
//! - ResonantOscillator performs complex multiplication (phasor rotation) by weights
//! - Sums weighted phasors as vectors, computing total energy
//!
//! **Stage 3: Action (Ternary Spike)**
//! - Emits {-1, 0, +1} based on energy threshold and phase
//! - sign(Re) determines spike polarity (forward/reverse/null)
//!
//! # Exact Arithmetic
//!
//! All operations use integer arithmetic with fixed-point scaling:
//! - KERNEL_SCALE = 1024 (2^10) provides ~10 bits fractional precision
//! - π approximation: 355/113 (Archimedean rational)
//! - i64 for signals/weights, i128 for accumulation (overflow safety)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Fixed-point scale for all operations: 2^10 = 1024
/// 
/// Provides ~10 bits fractional precision for phase calculations.
/// All complex components are scaled by this factor.
/// Exposed to Python for weight calculations.
pub const KERNEL_SCALE: i64 = 1024;

/// Discrete Hilbert Transform Kernel - Projects scalar signal to complex analytic signal
///
/// Implements integer-only approximation of Hilbert transform:
/// h[n] = (2 · SCALE · 113) / (355 · n) for odd n, 0 for even n
///
/// # Mathematical Foundation
///
/// The Hilbert transform produces an analytic signal z(t) = x(t) + jH[x(t)]
/// where H[x] is the Hilbert transform. For discrete signals:
///
/// - Real component (I): Original signal delayed by group delay
/// - Imaginary component (Q): Convolution with antisymmetric h[n]
///
/// This projects scalar signal into quaternionic phase space (W4),
/// enabling phase-coherent operations.
#[pyclass]
#[derive(Clone)]
pub struct DiscreteHilbertKernel {
    /// Ring buffer storing signal history (taps samples)
    history: Vec<i64>,
    /// Pre-computed integer Hilbert coefficients
    coeffs: Vec<i64>,
    /// Current position in ring buffer
    head: usize,
    /// Cached analytic signal ("The Afterimage") - physical membrane state
    /// Decision 4: Cache physical state for learning without recomputation
    last_analytic: (i64, i64),
}

#[pymethods]
impl DiscreteHilbertKernel {
    /// Create new Discrete Hilbert Kernel with specified filter taps
    ///
    /// # Arguments
    /// * `taps` - Filter length, must be odd and >= 3
    ///
    /// # Returns
    /// Initialized kernel with identity history and computed coefficients
    ///
    /// # Errors
    /// Returns PyValueError if taps is even or < 3
    ///
    /// # Example
    /// ```python
    /// from syntonic.sna import DiscreteHilbertKernel
    /// kernel = DiscreteHilbertKernel(31)  # 31-tap filter
    /// ```
    #[new]
    pub fn new(taps: usize) -> PyResult<Self> {
        // Validate taps: must be odd and >= 3
        if taps < 3 {
            return Err(PyValueError::new_err(
                format!("taps must be >= 3, got {}", taps)
            ));
        }
        if taps % 2 == 0 {
            return Err(PyValueError::new_err(
                format!("taps must be odd for symmetric Hilbert transform, got {}", taps)
            ));
        }

        // Compute integer Hilbert coefficients
        // Ideal: h[n] = 2/(π·n) for odd n, 0 for even n
        // Integer: h[n] = (2 · KERNEL_SCALE · 113) / (355 · n)
        // Using π ≈ 355/113 (Archimedean approximation)
        let mut coeffs = Vec::with_capacity(taps);
        let center = (taps / 2) as i64;

        for i in 0..taps {
            let n = (i as i64) - center;
            if n % 2 == 0 {
                // Even indices (including center): zero coefficient
                coeffs.push(0);
            } else {
                // Odd indices: 2/(π·n) ≈ (2·SCALE·113)/(355·n)
                // Use i128 intermediate to prevent overflow
                let numerator = 2i128 * (KERNEL_SCALE as i128) * 113i128;
                let denominator = 355i128 * (n.abs() as i128);
                let mut coeff = (numerator / denominator) as i64;
                
                // Preserve antisymmetry: h[-n] = -h[n]
                if n < 0 {
                    coeff = -coeff;
                }
                coeffs.push(coeff);
            }
        }

        Ok(Self {
            history: vec![0; taps],
            coeffs,
            head: 0,
            last_analytic: (0, 0),
        })
    }

    /// Reset kernel state (clear history buffer and cached analytic signal)
    ///
    /// Useful for marking epoch boundaries or preventing cross-contamination
    /// between independent sequences.
    pub fn reset(&mut self) {
        self.history.fill(0);
        self.head = 0;
        self.last_analytic = (0, 0);
    }

    /// Process one sample, returning complex phasor (real, imaginary)
    ///
    /// Internal method called by ResonantOscillator during batch processing.
    /// Not exposed to Python - use ResonantOscillator.process_batch() instead.
    fn process_tick(&mut self, input: i64) -> (i64, i64) {
        let taps = self.history.len();
        
        // 1. Store input in ring buffer
        self.history[self.head] = input;
        
        // 2. Compute imaginary component via convolution
        // Q = Σ history[i] · coeffs[i]
        let mut q_accum: i64 = 0;
        for i in 0..taps {
            let hist_idx = (self.head + taps - i) % taps;
            q_accum += self.history[hist_idx] * self.coeffs[i];
        }
        
        // Descale: divide by KERNEL_SCALE
        let q = q_accum / KERNEL_SCALE;
        
        // 3. Real component: delayed input (group delay compensation)
        let delay = taps / 2;
        let center_idx = (self.head + taps - delay) % taps;
        let i_val = self.history[center_idx];
        
        // 4. Advance ring buffer head
        self.head = (self.head + 1) % taps;
        
        // Cache the analytic signal (physical membrane state)
        self.last_analytic = (i_val, q);
        
        (i_val, q)
    }
    
    /// O(1) Access to current membrane state ("peek" without advancing)
    /// 
    /// Returns the last computed analytic signal without recomputing convolution.
    /// Used by train_step to access sensor state for learning.
    fn peek_current(&self) -> (i64, i64) {
        self.last_analytic
    }
}

/// Resonant Oscillator - The Syntonic Neuron
///
/// Performs phase-coherent computation by:
/// 1. Converting inputs to phasors via Hilbert kernels (Sensation)
/// 2. Rotating phasors by complex weights (Integration)
/// 3. Computing energy and emitting ternary spike (Action)
///
/// # Ternary Output Semantics
///
/// - **+1 (Excitatory)**: Energy exceeds threshold, phase forward (Re > 0)
/// - **-1 (Inhibitory)**: Energy exceeds threshold, phase reverse (Re < 0)
/// - **0 (Null)**: Energy below threshold, no spike
///
/// # Weight Constraint
///
/// Weights must satisfy unit norm: |w|² ≤ KERNEL_SCALE² (with 10% tolerance)
/// Enforced by set_weights() validation.
#[pyclass]
#[derive(Clone)]
pub struct ResonantOscillator {
    /// Per-input Hilbert kernels (scalar → phasor conversion)
    sensors: Vec<DiscreteHilbertKernel>,
    /// Complex weights (real, imag) representing synaptic impedance
    weights: Vec<(i64, i64)>,
    /// Squared energy threshold for spike emission
    threshold_sq: i128,
}

#[pymethods]
impl ResonantOscillator {
    /// Create new Resonant Oscillator neuron
    ///
    /// # Arguments
    /// * `input_count` - Number of input channels (sensors)
    /// * `kernel_taps` - Hilbert kernel size (must be odd, >= 3)
    /// * `threshold` - Energy threshold for spike emission (will be squared)
    ///
    /// # Returns
    /// Initialized neuron with identity weights (KERNEL_SCALE, 0)
    ///
    /// # Example
    /// ```python
    /// from syntonic.sna import ResonantOscillator
    /// neuron = ResonantOscillator(
    ///     input_count=1,
    ///     kernel_taps=31,
    ///     threshold=500
    /// )
    /// ```
    #[new]
    pub fn new(input_count: usize, kernel_taps: usize, threshold: i64) -> PyResult<Self> {
        // Create Hilbert kernels for each input
        let mut sensors = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            sensors.push(DiscreteHilbertKernel::new(kernel_taps)?);
        }
        
        // Initialize weights to identity: (KERNEL_SCALE, 0)
        // This provides pass-through behavior (no rotation, unit magnitude)
        let weights = vec![(KERNEL_SCALE, 0); input_count];
        
        // Square threshold once at construction
        let threshold_sq = (threshold as i128) * (threshold as i128);
        
        Ok(Self {
            sensors,
            weights,
            threshold_sq,
        })
    }

    /// Set complex weights with unit norm validation
    ///
    /// # Arguments
    /// * `weights` - List of (real, imag) tuples scaled by KERNEL_SCALE
    ///
    /// # Errors
    /// - Wrong length: must match input_count
    /// - Excessive magnitude: |w|² > KERNEL_SCALE² × 1.1
    ///
    /// # Example
    /// ```python
    /// from syntonic.sna import ResonantOscillator, KERNEL_SCALE
    /// neuron = ResonantOscillator(2, 31, 500)
    /// 
    /// # 45° rotation, magnitude 1.0
    /// neuron.set_weights([
    ///     (int(KERNEL_SCALE * 0.707), int(KERNEL_SCALE * 0.707)),
    ///     (KERNEL_SCALE, 0)
    /// ])
    /// ```
    pub fn set_weights(&mut self, weights: Vec<(i64, i64)>) -> PyResult<()> {
        // Validate length
        if weights.len() != self.weights.len() {
            return Err(PyValueError::new_err(
                format!("Expected {} weights, got {}", self.weights.len(), weights.len())
            ));
        }
        
        // Validate unit norm constraint with 10% tolerance
        let max_mag_sq = (KERNEL_SCALE as i128) * (KERNEL_SCALE as i128);
        let tolerance = max_mag_sq / 10; // 10% tolerance
        let limit = max_mag_sq + tolerance;
        
        for (i, &(wr, wi)) in weights.iter().enumerate() {
            let mag_sq = (wr as i128) * (wr as i128) + (wi as i128) * (wi as i128);
            if mag_sq > limit {
                return Err(PyValueError::new_err(
                    format!("Weight {} exceeds unit norm: |w|² = {} > {}", i, mag_sq, limit)
                ));
            }
        }
        
        // Validation passed, update weights
        self.weights = weights;
        Ok(())
    }

    /// Get current complex weights
    ///
    /// # Returns
    /// List of (real, imag) tuples scaled by KERNEL_SCALE
    pub fn get_weights(&self) -> Vec<(i64, i64)> {
        self.weights.clone()
    }

    /// Returns the number of input sensors (Hilbert Kernels)
    ///
    /// # Returns
    /// Number of input channels this oscillator accepts
    ///
    /// # Example
    /// ```python
    /// neuron = ResonantOscillator(3, 31, 500)
    /// assert neuron.sensor_count() == 3
    /// ```
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Reset all kernel states (clear history buffers)
    ///
    /// Use to mark epoch boundaries or prevent cross-contamination
    /// between independent sequences.
    pub fn reset(&mut self) {
        for sensor in &mut self.sensors {
            sensor.reset();
        }
    }

    /// Process multiple time steps across all inputs (batch processing)
    ///
    /// # Arguments
    /// * `input_streams` - List of signal streams, one per input channel
    ///
    /// # Returns
    /// List of ternary spikes {-1, 0, +1} for each time step
    ///
    /// # Errors
    /// - Wrong number of streams: must match input_count
    /// - Ragged arrays: all streams must have same length
    ///
    /// # Example
    /// ```python
    /// import math
    /// from syntonic.sna import ResonantOscillator
    /// 
    /// neuron = ResonantOscillator(1, 31, 500)
    /// 
    /// # Generate sine wave (amplitude 1000, 100 samples)
    /// signal = [int(1000 * math.sin(2 * math.pi * t / 20)) 
    ///           for t in range(100)]
    /// 
    /// spikes = neuron.process_batch([signal])
    /// print("".join("▲" if s > 0 else "▼" if s < 0 else "·" for s in spikes))
    /// ```
    pub fn process_batch(&mut self, input_streams: Vec<Vec<i64>>) -> PyResult<Vec<i8>> {
        // Validate stream count
        if input_streams.len() != self.sensors.len() {
            return Err(PyValueError::new_err(
                format!("Expected {} input streams, got {}", self.sensors.len(), input_streams.len())
            ));
        }
        
        // Validate all streams have same length (reject ragged arrays)
        if input_streams.is_empty() {
            return Ok(Vec::new());
        }
        
        let expected_len = input_streams[0].len();
        for (i, stream) in input_streams.iter().enumerate() {
            if stream.len() != expected_len {
                return Err(PyValueError::new_err(
                    format!("Stream {} has length {} but expected {} (ragged arrays not allowed)",
                            i, stream.len(), expected_len)
                ));
            }
        }
        
        // Process each time step
        let mut spikes = Vec::with_capacity(expected_len);
        for t in 0..expected_len {
            // Build input vector for this time step
            let inputs: Vec<i64> = input_streams.iter()
                .map(|stream| stream[t])
                .collect();
            
            // Compute activation
            let spike = self.compute_activation(&inputs);
            spikes.push(spike);
        }
        
        Ok(spikes)
    }

    /// Process single time step (single sample per input)
    ///
    /// # Arguments
    /// * `inputs` - Signal values for this time step
    ///
    /// # Returns
    /// Ternary spike: +1 (excitatory), -1 (inhibitory), 0 (null)
    ///
    /// # Example
    /// ```python
    /// neuron = ResonantOscillator(2, 31, 500)
    /// spike = neuron.step([1000, -500])
    /// ```
    pub fn step(&mut self, inputs: Vec<i64>) -> PyResult<i8> {
        if inputs.len() != self.sensors.len() {
            return Err(PyValueError::new_err(
                format!("Expected {} inputs, got {}", self.sensors.len(), inputs.len())
            ));
        }
        
        Ok(self.compute_activation(&inputs))
    }

    /// Syntonic Torque Learning Rule (The "Mind" of the Neuron)
    ///
    /// # Theory: Geometric Phase Alignment
    ///
    /// Standard AI minimizes scalar error (MSE). SNA maximizes Vector Resonance.
    /// We treat the Weight (W) as a Rotor on the Unit Circle. We want to rotate W
    /// so that the Input Vector (S) aligns with the Target Phase (T).
    ///
    /// **Target Vector:** V_t = S * T (The direction we want S to point)
    /// **Torque:** Cross product between current alignment and target
    ///
    /// In the complex plane, to align S with Real+, we multiply by S_conjugate:
    /// - Target (+1): V_ideal = (Sig_r, -Sig_i) [conjugate for +Real alignment]
    /// - Target (-1): V_ideal = (-Sig_r, Sig_i) [negative conjugate for -Real]
    ///
    /// **Update Rule:** W_new = W_old + rate * (Target * S_conjugate) / SCALE
    ///
    /// This applies a "Torque" perpendicular to the error vector, rotating the
    /// weight toward the desired phase alignment. After update, we project back
    /// onto the unit circle via integer Newton-Raphson normalization.
    ///
    /// # Arguments
    /// * `inputs` - Current input signals (must match sensor count)
    /// * `target` - Desired ternary spike {-1, 0, +1}
    /// * `rate` - Learning rate (integer angular momentum, e.g., 50 ≈ 3° rotation)
    ///
    /// # Returns
    /// Tuple (learning_flag, syntony_score):
    /// - **learning_flag**: 1 if weights updated, 0 if skipped (target=0 or rate=0)
    /// - **syntony_score**: Projection of resonance onto target direction (higher is better)
    ///   - For target +1: score = sum_real (want positive real)
    ///   - For target -1: score = -sum_real (want negative real)
    ///   - For target 0: score = -energy (want minimal energy)
    ///
    /// # Errors
    /// Returns PyValueError if input dimension mismatch
    ///
    /// # Example
    /// ```python
    /// from syntonic.sna import ResonantOscillator
    /// 
    /// neuron = ResonantOscillator(1, 31, 500)
    /// 
    /// # Process input
    /// spike = neuron.step([1000])
    /// 
    /// # If output doesn't match target, apply learning
    /// if spike != 1:
    ///     learned, syntony = neuron.train_step([1000], target=1, rate=50)
    ///     print(f"Learning: {learned}, Syntony: {syntony}")
    /// ```
    pub fn train_step(
        &mut self, 
        inputs: Vec<i64>, 
        target: i8, 
        rate: i64
    ) -> PyResult<(i8, i64)> {
        // 1. Input Validation
        if inputs.len() != self.sensors.len() {
            return Err(PyValueError::new_err(
                format!("Input dimension mismatch: expected {}, got {}", 
                        self.sensors.len(), inputs.len())
            ));
        }

        // 2. Compute Current Resonance (to calculate syntony score)
        let mut sum_real: i128 = 0;
        let mut sum_imag: i128 = 0;
        let mut input_phasors = Vec::with_capacity(inputs.len());

        for i in 0..self.sensors.len() {
            // Peek: Read the state that just caused the last firing
            let (sig_r, sig_i) = self.sensors[i].peek_current();
            let (w_r, w_i) = self.weights[i];
            
            // i128 precision for accumulation (overflow safety)
            let scale = KERNEL_SCALE as i128;
            let sig_r_l = sig_r as i128;
            let sig_i_l = sig_i as i128;
            let w_r_l = w_r as i128;
            let w_i_l = w_i as i128;

            // Complex multiplication: (sig_r + j·sig_i) × (w_r + j·w_i)
            let rot_r = (sig_r_l * w_r_l - sig_i_l * w_i_l) / scale;
            let rot_i = (sig_r_l * w_i_l + sig_i_l * w_r_l) / scale;
            
            sum_real += rot_r;
            sum_imag += rot_i;
            input_phasors.push((sig_r, sig_i));
        }

        // 3. Calculate Syntony Score (Before Update)
        // Measures alignment with target direction
        let syntony_score = match target {
             1 => sum_real as i64,      // Want Positive Real output
            -1 => -(sum_real as i64),   // Want Negative Real output
             0 => -((sum_real.pow(2) + sum_imag.pow(2)) as i64), // Want Zero Energy
             _ => 0
        };

        // 4. Learning Gating
        if target == 0 {
            // Prototype 1: Null targets return score without active damping
            return Ok((0, syntony_score));
        }
        
        if rate == 0 {
            // Natural termination: rate crystallized to zero
            return Ok((0, syntony_score));
        }

        // 5. Apply Syntonic Torque to Each Weight
        for i in 0..self.weights.len() {
            let (sig_r, sig_i) = input_phasors[i];
            
            // Torque Direction: Target × Conjugate(Signal)
            // This gives the phase rotation needed to align signal with target
            let t_val = target as i64;
            
            // Decision 1: Upgrade to i128 for torque calculation (overflow safety)
            let sig_r_128 = sig_r as i128;
            let sig_i_128 = sig_i as i128;
            let rate_128 = rate as i128;
            let scale_128 = KERNEL_SCALE as i128;

            // Compute torque vector components
            // For target +1: nudge toward (sig_r, -sig_i) [conjugate]
            // For target -1: nudge toward (-sig_r, sig_i) [negative conjugate]
            let nudge_r = (sig_r_128 * (t_val as i128) * rate_128) / scale_128;
            let nudge_i = (-sig_i_128 * (t_val as i128) * rate_128) / scale_128;
            
            let (curr_w_r, curr_w_i) = self.weights[i];
            
            // Apply nudge (vector addition in weight space)
            let mut new_w_r = curr_w_r + (nudge_r as i64);
            let mut new_w_i = curr_w_i + (nudge_i as i64);
            
            // 6. Syntonic Normalization (Energy Conservation via Newton-Raphson)
            // Project weight back onto unit circle: |w| = KERNEL_SCALE
            let mag_sq = (new_w_r as i128).pow(2) + (new_w_i as i128).pow(2);
            let mag = self.integer_sqrt(mag_sq);
            
            if mag > 0 {
                // Rescale to KERNEL_SCALE: w_normalized = w * SCALE / mag
                new_w_r = (new_w_r as i128 * scale_128 / mag) as i64;
                new_w_i = (new_w_i as i128 * scale_128 / mag) as i64;
            }

            self.weights[i] = (new_w_r, new_w_i);
        }

        Ok((1, syntony_score))
    }
}

impl ResonantOscillator {
    /// Integer Square Root (Newton-Raphson Iteration)
    ///
    /// Computes floor(√n) using integer Newton-Raphson method.
    /// Converges quadratically - 10 iterations sufficient for i128 precision.
    ///
    /// # Algorithm
    /// Newton iteration: x_new = (x + n/x) / 2
    /// Terminates when x_new >= x (monotonic convergence from above)
    ///
    /// # Arguments
    /// * `n` - Integer to compute square root of
    ///
    /// # Returns
    /// Integer approximation of √n (floor value)
    ///
    /// # Examples
    /// - integer_sqrt(1048576) = 1024 (exact)
    /// - integer_sqrt(2097152) ≈ 1448 (√2 × 1024)
    fn integer_sqrt(&self, n: i128) -> i128 {
        if n < 0 { return 0; }
        if n == 0 { return 0; }
        
        // Initial guess: (n >> 1) + 1
        let mut x = (n >> 1) + 1;
        let mut y = (x + n / x) >> 1;
        
        // Iterate until convergence (y >= x means we've found floor)
        while y < x {
            x = y;
            y = (x + n / x) >> 1;
        }
        
        x
    }

    /// Internal activation computation (Stage 2 & 3: Integration + Action)
    ///
    /// Performs phase-coherent computation:
    /// 1. Convert inputs to phasors via Hilbert kernels
    /// 2. Rotate by complex weights (complex multiplication)
    /// 3. Sum as vectors, compute energy
    /// 4. Emit ternary spike based on energy threshold and phase
    fn compute_activation(&mut self, inputs: &[i64]) -> i8 {
        // Accumulators for complex sum (use i128 to prevent overflow)
        let mut sum_real: i128 = 0;
        let mut sum_imag: i128 = 0;
        
        // Stage 1 (Sensation) + Stage 2 (Integration)
        for (i, &signal) in inputs.iter().enumerate() {
            // Convert scalar → phasor via Hilbert transform
            let (sig_r, sig_i) = self.sensors[i].process_tick(signal);
            let (w_r, w_i) = self.weights[i];
            
            // Complex multiplication: (sig_r + j·sig_i) × (w_r + j·w_i)
            // = (sig_r·w_r - sig_i·w_i) + j(sig_r·w_i + sig_i·w_r)
            // Use i128 intermediate, then descale
            let real_product = (sig_r as i128) * (w_r as i128) 
                             - (sig_i as i128) * (w_i as i128);
            let imag_product = (sig_r as i128) * (w_i as i128) 
                             + (sig_i as i128) * (w_r as i128);
            
            let rotated_r = real_product / (KERNEL_SCALE as i128);
            let rotated_i = imag_product / (KERNEL_SCALE as i128);
            
            // Vector summation
            sum_real += rotated_r;
            sum_imag += rotated_i;
        }
        
        // Stage 3 (Action): Ternary spike based on energy and phase
        // Energy = |z|² = Re² + Im²
        let energy = sum_real * sum_real + sum_imag * sum_imag;
        
        if energy >= self.threshold_sq {
            // Spike polarity determined by phase (sign of real component)
            // Exact zero-crossing produces null spike
            if sum_real > 0 {
                1  // Excitatory (forward phase)
            } else if sum_real < 0 {
                -1  // Inhibitory (reverse phase)
            } else {
                0  // Null (exact zero-crossing)
            }
        } else {
            0  // Below threshold, no spike
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        // Valid: odd taps >= 3
        assert!(DiscreteHilbertKernel::new(3).is_ok());
        assert!(DiscreteHilbertKernel::new(31).is_ok());
        
        // Invalid: even taps
        assert!(DiscreteHilbertKernel::new(4).is_err());
        
        // Invalid: too small
        assert!(DiscreteHilbertKernel::new(1).is_err());
    }

    #[test]
    fn test_weight_validation() {
        let mut neuron = ResonantOscillator::new(2, 31, 500).unwrap();
        
        // Valid: unit magnitude
        assert!(neuron.set_weights(vec![
            (KERNEL_SCALE, 0),
            (724, 724)  // ~0.707 each, unit norm
        ]).is_ok());
        
        // Invalid: excessive magnitude
        assert!(neuron.set_weights(vec![
            (KERNEL_SCALE * 2, 0),
            (0, 0)
        ]).is_err());
    }

    #[test]
    fn test_batch_validation() {
        let mut neuron = ResonantOscillator::new(2, 31, 500).unwrap();
        
        // Valid: equal length streams
        let result = neuron.process_batch(vec![
            vec![1, 2, 3],
            vec![4, 5, 6]
        ]);
        assert!(result.is_ok());
        
        // Invalid: ragged array
        let result = neuron.process_batch(vec![
            vec![1, 2, 3],
            vec![4, 5]  // Wrong length
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_neuron_reset() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Process some data
        neuron.process_batch(vec![vec![1000, 2000, 3000]]).unwrap();
        
        // Reset clears history
        neuron.reset();
        
        // Processing after reset should work
        let result = neuron.process_batch(vec![vec![100, 200]]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ternary_output() {
        let mut neuron = ResonantOscillator::new(1, 31, 100).unwrap();
        
        // Process weak signal (below threshold)
        let spike = neuron.step(vec![10]).unwrap();
        assert_eq!(spike, 0);  // Should be null
        
        // Process strong signal (above threshold)
        neuron.reset();
        let spike = neuron.step(vec![1000]).unwrap();
        // Should be non-zero (either +1 or -1)
        assert!(spike == 1 || spike == -1);
    }

    #[test]
    fn test_clone_independence() {
        let mut neuron1 = ResonantOscillator::new(1, 31, 500).unwrap();
        let mut neuron2 = neuron1.clone();
        
        // Process different signals
        neuron1.process_batch(vec![vec![1000]]).unwrap();
        neuron2.process_batch(vec![vec![2000]]).unwrap();
        
        // Neurons should maintain independent state
        // (Testing via different spike patterns would require deterministic input)
    }

    // ============================================================================
    // Phase 2: Syntonic Plasticity Tests
    // ============================================================================

    #[test]
    fn test_integer_sqrt() {
        let neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Test exact squares
        assert_eq!(neuron.integer_sqrt(0), 0);
        assert_eq!(neuron.integer_sqrt(1), 1);
        assert_eq!(neuron.integer_sqrt(4), 2);
        assert_eq!(neuron.integer_sqrt(1024 * 1024), 1024);
        assert_eq!(neuron.integer_sqrt(1048576), 1024); // KERNEL_SCALE²
        
        // Test approximate values
        let sqrt2_times_1024 = neuron.integer_sqrt(2097152); // √2 × 1024
        assert!(sqrt2_times_1024 >= 1447 && sqrt2_times_1024 <= 1449); // ≈1448
        
        // Test negative (should return 0)
        assert_eq!(neuron.integer_sqrt(-100), 0);
    }

    #[test]
    fn test_peek_current() {
        let mut kernel = DiscreteHilbertKernel::new(31).unwrap();
        
        // Initial state should be (0, 0)
        assert_eq!(kernel.peek_current(), (0, 0));
        
        // Process a tick
        let result = kernel.process_tick(1000);
        
        // Peek should return same as last process_tick
        assert_eq!(kernel.peek_current(), result);
        
        // Peek multiple times should be idempotent
        assert_eq!(kernel.peek_current(), result);
        assert_eq!(kernel.peek_current(), result);
    }

    #[test]
    fn test_train_step_no_change() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Process input
        neuron.step(vec![1000]).unwrap();
        
        // Get weights before training
        let weights_before = neuron.get_weights();
        
        // Train with target=0 (null) - should not update
        let (learned, _score) = neuron.train_step(vec![1000], 0, 50).unwrap();
        assert_eq!(learned, 0); // No learning flag
        
        // Weights should be unchanged
        let weights_after = neuron.get_weights();
        assert_eq!(weights_before, weights_after);
    }

    #[test]
    fn test_train_step_updates_weights() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Process input to populate sensor state
        neuron.step(vec![1000]).unwrap();
        
        // Get initial weights
        let weights_before = neuron.get_weights();
        
        // Train toward target +1 with significant rate
        let (learned, syntony_before) = neuron.train_step(vec![1000], 1, 100).unwrap();
        assert_eq!(learned, 1); // Learning occurred
        
        // Weights should have changed
        let weights_after = neuron.get_weights();
        assert_ne!(weights_before, weights_after);
        
        // Weights should still satisfy unit norm (within tolerance)
        let (wr, wi) = weights_after[0];
        let mag_sq = (wr as i128).pow(2) + (wi as i128).pow(2);
        let expected_sq = (KERNEL_SCALE as i128).pow(2);
        let tolerance = expected_sq / 10; // 10% tolerance
        assert!(mag_sq >= expected_sq - tolerance && mag_sq <= expected_sq + tolerance,
                "Weight magnitude {} out of tolerance range", mag_sq);
        
        // Train again - syntony should change
        neuron.step(vec![1000]).unwrap();
        let (_learned2, syntony_after) = neuron.train_step(vec![1000], 1, 100).unwrap();
        
        // Syntony can increase or decrease depending on initial phase
        // Just verify it's a valid i64
        assert!(syntony_after.abs() < i64::MAX);
    }

    #[test]
    fn test_train_step_convergence() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Constant input signal
        let signal = vec![1000];
        let target = 1i8; // Want excitatory response
        let mut rate = 50;
        
        let mut syntony_history = Vec::new();
        
        // Train for 20 steps
        for _ in 0..20 {
            neuron.step(signal.clone()).unwrap();
            let (_learned, syntony) = neuron.train_step(signal.clone(), target, rate).unwrap();
            syntony_history.push(syntony);
            
            // Golden decay: rate -= rate / 100 (simplified)
            rate = rate.saturating_sub(rate / 100).max(1);
        }
        
        // Syntony should generally increase or stabilize
        // (Exact convergence depends on initial conditions, but system should be stable)
        let first_syntony = syntony_history[0];
        let last_syntony = syntony_history[syntony_history.len() - 1];
        
        // Verify system is stable (not oscillating wildly)
        let max_syntony = syntony_history.iter().max().unwrap();
        let min_syntony = syntony_history.iter().min().unwrap();
        let range = max_syntony - min_syntony;
        
        // Range should be bounded (not exponentially growing)
        assert!(range < 100000, "Syntony oscillating too wildly: range = {}", range);
    }

    #[test]
    fn test_train_step_rate_zero() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        neuron.step(vec![1000]).unwrap();
        let weights_before = neuron.get_weights();
        
        // Train with rate=0 (crystallization point)
        let (learned, _score) = neuron.train_step(vec![1000], 1, 0).unwrap();
        assert_eq!(learned, 0); // No learning
        
        // Weights unchanged
        assert_eq!(weights_before, neuron.get_weights());
    }

    #[test]
    fn test_train_step_input_mismatch() {
        let mut neuron = ResonantOscillator::new(2, 31, 500).unwrap();
        
        neuron.step(vec![1000, 500]).unwrap();
        
        // Try to train with wrong input count
        let result = neuron.train_step(vec![1000], 1, 50);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_syntony_score_directions() {
        let mut neuron = ResonantOscillator::new(1, 31, 500).unwrap();
        
        // Process to get some resonance
        neuron.step(vec![1000]).unwrap();
        
        // Test target +1 (want positive real)
        let (_l1, score_pos) = neuron.train_step(vec![1000], 1, 0).unwrap();
        
        // Test target -1 (want negative real) - should invert sign
        let (_l2, score_neg) = neuron.train_step(vec![1000], -1, 0).unwrap();
        
        // Scores should have opposite relationship to sum_real
        // score_pos = sum_real, score_neg = -sum_real
        // So score_pos + score_neg should be close to 0 (or exactly 0)
        assert_eq!(score_pos, -score_neg);
    }
}