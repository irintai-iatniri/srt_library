//! ResonantTensor: Dual-state tensor for the Resonant Engine
//!
//! The ResonantTensor exists in two complementary representations:
//! - **Lattice** (CPU): Exact GoldenExact values in Q(φ) - the "truth"
//! - **Flux** (GPU): Ephemeral floating-point values - the "shadow"
//!
//! Phase transitions:
//! - `wake_flux()`: Project lattice → GPU floats (enter D-phase)
//! - `crystallize()`: Snap GPU floats → lattice (enter H-phase)
//! - `destroy_shadow()`: Free GPU memory without crystallizing

use pyo3::prelude::*;
use std::fmt;
use std::time::Duration;

use super::crystallize::{
    compute_lattice_syntony, crystallize_with_dwell, harmonize_and_crystallize, snap_to_lattice,
};
use super::{PHI, PHI_INV, PHI_INV_SQ};
use crate::exact::{GoldenExact, Rational};

use crate::tensor::srt_kernels::{
    cuda_bmm_nt_f64, cuda_matmul_nt_f64, cuda_resonant_compute_syntony_f64,
    cuda_resonant_d_phase_f64,
};
use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

/// Error type for resonant operations
#[derive(Debug, Clone)]
pub enum ResonantError {
    /// Invalid phase transition attempted
    InvalidPhaseTransition(String),
    /// No flux present when expected
    NoFluxPresent,
    /// No device present when expected
    NoDevicePresent,
    /// CUDA error
    CudaError(String),
    /// Shape mismatch
    ShapeMismatch(String),
}

impl fmt::Display for ResonantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResonantError::InvalidPhaseTransition(msg) => {
                write!(f, "Invalid phase transition: {}", msg)
            }
            ResonantError::NoFluxPresent => write!(f, "No flux present"),
            ResonantError::NoDevicePresent => write!(f, "No CUDA device present"),
            ResonantError::CudaError(msg) => write!(f, "CUDA error: {}", msg),
            ResonantError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
        }
    }
}

impl std::error::Error for ResonantError {}

impl From<ResonantError> for PyErr {
    fn from(err: ResonantError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// Phase of the resonant tensor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResonantPhase {
    /// Crystallized on CPU lattice (exact arithmetic)
    /// No GPU memory allocated
    Crystallized,

    /// Active flux on GPU (floating point)
    /// GPU memory allocated, lattice is stale
    Flux,

    /// Transitioning between phases
    /// Should not persist
    Transitioning,
}

impl fmt::Display for ResonantPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResonantPhase::Crystallized => write!(f, "crystallized"),
            ResonantPhase::Flux => write!(f, "flux"),
            ResonantPhase::Transitioning => write!(f, "transitioning"),
        }
    }
}

/// A tensor that exists in two complementary representations.
///
/// The ResonantTensor is the core data structure of the Resonant Engine.
/// It maintains:
/// - An exact lattice representation in Q(φ) (always valid)
/// - An optional GPU flux representation (valid only during D-phase)
///
/// # Invariants
/// - `lattice.len() == shape.product()`
/// - `mode_norm_sq.len() == lattice.len()`
/// - `flux.is_some()` implies `phase == Flux`
/// - `phase == Crystallized` implies `flux.is_none()`
#[pyclass]
#[derive(Debug)]
pub struct ResonantTensor {
    /// Exact lattice representation: a + b·φ for each element
    lattice: Vec<GoldenExact>,

    /// GPU flux representation (active during D-phase only)
    flux: Option<CudaSlice<f64>>,

    /// GPU flux representation in f32 (experimental/performance)
    flux_f32: Option<CudaSlice<f32>>,

    /// CPU flux representation (active during D-phase fallback or pure CPU mode)
    cpu_flux: Option<Vec<f64>>,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Current syntony value S ∈ [0, 1]
    syntony: Rational,

    /// Precomputed mode norm squared |n|² for each element
    /// Used for syntony computation and D̂/Ĥ operators
    mode_norm_sq: Vec<Rational>,

    /// Current phase
    phase: ResonantPhase,

    /// CUDA device reference
    device: Option<Arc<CudaDevice>>,

    /// Device index
    device_idx: usize,

    /// Last D-phase duration (nanoseconds)
    last_d_duration_ns: u64,

    /// Precision used for last crystallization
    precision: i64,
}

impl ResonantTensor {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a ResonantTensor from an existing GoldenExact lattice.
    pub fn from_lattice(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Vec<Rational>,
    ) -> Result<Self, ResonantError> {
        let expected_size: usize = shape.iter().product();
        if lattice.len() != expected_size {
            return Err(ResonantError::ShapeMismatch(format!(
                "Lattice size {} doesn't match shape {:?}",
                lattice.len(),
                shape
            )));
        }
        if mode_norm_sq.len() != lattice.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "mode_norm_sq size {} doesn't match lattice size {}",
                mode_norm_sq.len(),
                lattice.len()
            )));
        }

        let syntony = compute_lattice_syntony(&lattice, &mode_norm_sq);

        Ok(ResonantTensor {
            lattice,
            flux: None,
            flux_f32: None,
            cpu_flux: None,
            shape,
            syntony,
            mode_norm_sq,
            phase: ResonantPhase::Crystallized,
            device: None,
            device_idx: 0,
            last_d_duration_ns: 0,
            precision: 100, // Default precision
        })
    }

    /// Create from f64 data by snapping to nearest golden lattice points.
    pub fn from_floats(
        data: &[f64],
        shape: Vec<usize>,
        mode_norm_sq: Vec<f64>,
        precision: i64,
    ) -> Result<Self, ResonantError> {
        let lattice = snap_to_lattice(data, precision);
        let mode_norm_sq_exact: Vec<Rational> = mode_norm_sq.iter().map(|&x| Rational::from_f64_nearest(x, precision)).collect();
        let mut tensor = Self::from_lattice(lattice, shape, mode_norm_sq_exact)?;
        tensor.precision = precision;
        Ok(tensor)
    }

    /// Create from f32 data by snapping to nearest golden lattice points.
    /// Converts f32 to f64 internally for lattice representation.
    pub fn from_floats_f32(
        data: &[f32],
        shape: Vec<usize>,
        mode_norm_sq: Vec<f32>,
        precision: i64,
    ) -> Result<Self, ResonantError> {
        // Convert f32 to f64 for lattice snapping
        let data_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
        let mode_norm_sq_exact: Vec<Rational> = mode_norm_sq.iter().map(|&v| Rational::from_f64_nearest(v as f64, precision)).collect();
        let lattice = snap_to_lattice(&data_f64, precision);
        let mut tensor = Self::from_lattice(lattice, shape, mode_norm_sq_exact)?;
        tensor.precision = precision;
        Ok(tensor)
    }
    /// Create a tensor of zeros with exact GoldenExact values.
    ///
    /// Initializes a Crystallized tensor with zero-valued lattice points.
    pub fn zeros(shape: Vec<usize>, precision: i64) -> Self {
        let size: usize = shape.iter().product();
        let lattice = vec![GoldenExact::zero(); size];
        // Mode norms for zero tensor are zero (0^2 + ... = 0)
        let mode_norm_sq = vec![0.0; size];

        ResonantTensor {
            lattice,
            flux: None,
            flux_f32: None,
            cpu_flux: None,
            shape,
            syntony: 0.0,
            mode_norm_sq,
            phase: ResonantPhase::Crystallized,
            device: None,
            device_idx: 0,
            last_d_duration_ns: 0,
            precision,
        }
    }

    /// Create with default mode norms (|n|² = index²).
    pub fn from_floats_default_modes(
        data: &[f64],
        shape: Vec<usize>,
        precision: i64,
    ) -> Result<Self, ResonantError> {
        let mode_norm_sq: Vec<f64> = (0..data.len()).map(|i| (i as f64).powi(2)).collect();
        Self::from_floats(data, shape, mode_norm_sq, precision)
    }

    // =========================================================================
    // Phase Transitions
    // =========================================================================

    /// PROJECT: Lattice → Flux (exact → approximate)
    ///
    /// This is the "wake_flux" operation that enters D-phase.
    /// The exact lattice values are projected to f64 and transferred to GPU.
    pub fn wake_flux(&mut self, device: Arc<CudaDevice>) -> Result<(), ResonantError> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();

        // Upload to GPU
        let gpu_slice = device
            .default_stream()
            .clone_htod(&floats)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        self.flux = Some(gpu_slice);
        self.device = Some(device);
        self.phase = ResonantPhase::Flux;

        Ok(())
    }

    /// CPU-only version of wake_flux for testing without GPU
    #[cfg(not(feature = "cuda"))]
    pub fn wake_flux_cpu(&mut self) -> Result<Vec<f64>, ResonantError> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();

        // Store CPU-side flux shadow for later CPU D-phase operations
        self.cpu_flux = Some(floats.clone());

        self.phase = ResonantPhase::Flux;
        Ok(floats)
    }

    /// CRYSTALLIZE: Flux → Lattice (approximate → exact)
    ///
    /// This is the core H-phase operation. GPU floats are snapped to the
    /// nearest golden lattice points using LLL-based approximation.
    pub fn crystallize(&mut self, precision: i64) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Flux {
            return Err(ResonantError::InvalidPhaseTransition(
                "crystallize requires Flux phase".to_string(),
            ));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device
            .default_stream()
            .memcpy_dtoh(flux, &mut host_data)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Golden snap: find nearest exact lattice point for each element
        self.lattice = snap_to_lattice(&host_data, precision);
        self.precision = precision;

        // Recompute syntony on crystallized lattice
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);

        // Destroy the shadow
        self.flux = None;
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// Crystallize with Ĥ attenuation and φ-dwell timing enforcement.
    ///
    /// This implements the full H-phase:
    /// 1. Download flux from GPU
    /// 2. Apply Ĥ attenuation: Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
    /// 3. Snap to Q(φ) lattice
    /// 4. If time remains before φ × t_D, deepen precision
    /// 5. Recompute syntony
    ///
    /// # Arguments
    /// * `base_precision` - Initial precision (max coefficient bound)
    /// * `target_duration` - Target duration for H-phase (φ × D-phase duration)
    pub fn crystallize_with_dwell_time(
        &mut self,
        base_precision: i64,
        target_duration: Duration,
    ) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Flux {
            return Err(ResonantError::InvalidPhaseTransition(
                "crystallize requires Flux phase".to_string(),
            ));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device
            .default_stream()
            .memcpy_dtoh(flux, &mut host_data)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Crystallize with Ĥ attenuation and φ-dwell timing
        // Pass mode_norm_sq and syntony for Ĥ operator
        let (new_lattice, final_precision, _actual_duration) = crystallize_with_dwell(
            &host_data,
            &self.mode_norm_sq,
            self.syntony,
            base_precision,
            target_duration,
        );

        self.lattice = new_lattice;
        self.precision = final_precision;
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);

        // Destroy the shadow
        self.flux = None;
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// CPU-only crystallization with Ĥ attenuation.
    ///
    /// Applies the Ĥ (harmonization) operator then snaps to Q(φ) lattice:
    ///   Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
    ///
    /// Where β(S) = φ⁻¹ × S and w(n) = exp(-|n|²/φ)
    pub fn crystallize_cpu(
        &mut self,
        values: &[f64],
        precision: i64,
    ) -> Result<f64, ResonantError> {
        if values.len() != self.lattice.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Values length {} doesn't match lattice length {}",
                values.len(),
                self.lattice.len()
            )));
        }

        // Apply Ĥ attenuation + snap to lattice
        self.lattice =
            harmonize_and_crystallize(values, &self.mode_norm_sq, self.syntony, precision);
        self.precision = precision;
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// Explicitly destroy flux without crystallizing (discard D-phase work)
    pub fn destroy_shadow(&mut self) {
        {
            self.flux = None;
            self.flux_f32 = None;
        }
        if self.phase == ResonantPhase::Flux {
            self.phase = ResonantPhase::Crystallized;
        }
    }

    // =========================================================================
    // CUDA-Accelerated D-Phase
    // =========================================================================

    /// Wake flux and apply D-phase transformation with CUDA kernel.
    ///
    /// This applies the D̂ operator with stochastic noise on GPU:
    /// flux[i] = lattice[i] * (1 + α(S) * √|n_i|²) + noise_scale * (1 - S) * noise[i]
    pub fn wake_flux_with_d_phase(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
    ) -> Result<(), ResonantError> {
        use rand::Rng;

        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Ensure kernels are loaded
        // ensure_srt_kernels_loaded(&device).map_err(|e| ResonantError::CudaError(e.to_string()))?; // Function not available

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();
        let n = floats.len();

        // Upload lattice and mode norms to GPU
        let gpu_lattice = device
            .default_stream()
            .clone_htod(&floats)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_mode_norms = device
            .default_stream()
            .clone_htod(&self.mode_norm_sq)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Generate Gaussian noise on CPU and upload
        let mut rng = rand::thread_rng();
        let noise: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let gpu_noise = device
            .default_stream()
            .clone_htod(&noise)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Allocate output buffer
        let mut gpu_flux: CudaSlice<f64> = device
            .default_stream()
            .alloc_zeros(n)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Run D-phase kernel
        let start = std::time::Instant::now();
        cuda_resonant_d_phase_f64(
            &device,
            &mut gpu_flux,
            &gpu_lattice,
            &gpu_mode_norms,
            &gpu_noise,
            self.syntony,
            noise_scale,
            n,
        )
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        self.last_d_duration_ns = start.elapsed().as_nanos() as u64;

        self.flux = Some(gpu_flux);
        self.device = Some(device);
        self.phase = ResonantPhase::Flux;

        Ok(())
    }

    /// Complete a full D→H cycle using CUDA for D-phase.
    ///
    /// 1. Wake flux with D-phase kernel (GPU)
    /// 2. Crystallize to lattice (CPU with LLL)
    ///
    /// Returns the new syntony value.
    pub fn cuda_cycle(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
        precision: i64,
    ) -> Result<f64, ResonantError> {
        // D-phase: GPU differentiation with noise
        self.wake_flux_with_d_phase(device, noise_scale)?;

        // H-phase: CPU crystallization
        self.crystallize(precision)
    }

    /// Complete a full D→H cycle with φ-dwell timing.
    ///
    /// H-phase duration is scaled by φ^syntony relative to D-phase duration.
    pub fn cuda_cycle_with_phi_dwell(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
        base_precision: i64,
    ) -> Result<f64, ResonantError> {
        // D-phase: GPU differentiation with noise
        self.wake_flux_with_d_phase(device.clone(), noise_scale)?;

        // Compute target H-phase duration: φ^S × D_duration
        let phi_golden = GoldenExact::new(Rational::zero(), Rational::one()); // φ
        let phi_power = phi_golden.power_rational(&self.syntony, self.precision);
        let phi_factor = phi_power.to_f64();
        let target_h_ns = (self.last_d_duration_ns as f64 * phi_factor) as u64;
        let target_duration = Duration::from_nanos(target_h_ns.max(1000)); // Min 1μs

        // H-phase: CPU crystallization with dwell
        self.crystallize_with_dwell_time(base_precision, target_duration)
    }

    /// Get the duration of the last D-phase in nanoseconds.
    pub fn last_d_duration_ns(&self) -> u64 {
        self.last_d_duration_ns
    }

    /// Compute syntony on current GPU flux (without crystallizing).
    pub fn compute_flux_syntony(&self) -> Result<f64, ResonantError> {
        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        // Upload mode norms if needed (they may already be there, but simpler to re-upload)
        let gpu_mode_norms = device
            .default_stream()
            .clone_htod(&self.mode_norm_sq)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        cuda_resonant_compute_syntony_f64(device, flux, &gpu_mode_norms, self.len())
            .map_err(|e| ResonantError::CudaError(e.to_string()))
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the current syntony value.
    pub fn syntony(&self) -> f64 {
        self.syntony
    }

    /// Get the current phase.
    pub fn phase(&self) -> ResonantPhase {
        self.phase
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the lattice values (read-only).
    pub fn lattice(&self) -> &[GoldenExact] {
        &self.lattice
    }

    /// Get the mode norm squared values.
    pub fn mode_norm_sq(&self) -> &[f64] {
        &self.mode_norm_sq
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.lattice.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.lattice.is_empty()
    }

    /// Convert lattice to f64 vector (for export/inspection).
    pub fn to_floats_core(&self) -> Vec<f64> {
        self.lattice.iter().map(|g| g.to_f64()).collect()
    }

    /// Convert to list of floats (Rust-accessible).
    pub fn to_floats_rust(&self) -> Vec<f64> {
        // If a CPU-side flux shadow exists and we're in Flux phase, prefer it
        if self.phase == ResonantPhase::Flux {
            if let Some(ref cpu) = self.cpu_flux {
                return cpu.clone();
            }

            // Download from GPU flux if available
            if let Some(device) = &self.device {
                // Check f32 flux
                if let Some(flux_f32) = &self.flux_f32 {
                    let mut host_data = vec![0.0f32; self.len()];
                    if device
                        .default_stream()
                        .memcpy_dtoh(flux_f32, &mut host_data)
                        .is_ok()
                    {
                        return host_data.into_iter().map(|v| v as f64).collect();
                    }
                }

                // Check f64 flux
                if let Some(flux) = &self.flux {
                    let mut host_data = vec![0.0f64; self.len()];
                    if device
                        .default_stream()
                        .memcpy_dtoh(flux, &mut host_data)
                        .is_ok()
                    {
                        return host_data;
                    }
                }
            }
        }

        self.to_floats_core()
    }

    /// Get the precision used for last crystallization.
    pub fn precision(&self) -> i64 {
        self.precision
    }

    pub fn set_device_idx(&mut self, idx: usize) {
        self.device_idx = idx;
    }

    pub fn device_idx(&self) -> Option<usize> {
        Some(self.device_idx)
    }

    /// Access the flux (GPU buffer) if present.
    pub fn flux_ref(&self) -> Option<&cudarc::driver::CudaSlice<f64>> {
        self.flux.as_ref()
    }

    /// Set the flux (GPU buffer) and update phase.
    pub fn set_flux(&mut self, flux: cudarc::driver::CudaSlice<f64>) {
        self.flux = Some(flux);
        self.flux_f32 = None;
        self.phase = ResonantPhase::Flux;
    }

    /// Access the f32 flux (GPU buffer) if present.
    pub fn flux_f32_ref(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.flux_f32.as_ref()
    }

    /// Set the f32 flux (GPU buffer) and update phase.
    pub fn set_flux_f32(&mut self, flux: cudarc::driver::CudaSlice<f32>) {
        self.flux = None;
        self.flux_f32 = Some(flux);
        self.phase = ResonantPhase::Flux;
    }

    /// Set the CUDA device reference.
    pub fn set_device(&mut self, device: Arc<CudaDevice>) {
        self.device = Some(device);
    }

    /// Recompute syntony from current lattice state.
    pub fn recompute_syntony(&mut self) {
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
    }

    /// Set lattice values (module-private, for retrocausal harmonization).
    pub(super) fn set_lattice(&mut self, values: &[GoldenExact]) -> Result<(), ResonantError> {
        if self.lattice.len() != values.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Lattice size mismatch: expected {}, got {}",
                self.lattice.len(),
                values.len()
            )));
        }
        self.lattice.copy_from_slice(values);
        Ok(())
    }

    /// Complete a full D→H cycle in CPU mode (Rust-accessible).
    ///
    /// This simulates what would happen with GPU differentiation
    /// by applying noise to the values and then crystallizing.
    ///
    /// Returns the new syntony value.
    pub fn run_cpu_cycle(
        &mut self,
        noise_scale: f64,
        precision: i64,
    ) -> Result<f64, ResonantError> {
        use rand::Rng;

        // Wake flux
        let mut values = self.to_floats_core();
        self.phase = ResonantPhase::Flux;

        // Apply simulated D-phase noise
        let mut rng = rand::thread_rng();
        for (i, v) in values.iter_mut().enumerate() {
            let norm_sq = &self.mode_norm_sq[i];
            let s = &self.syntony;

            // D̂ coefficient: α(S) = φ⁻² × (1 - S)  
            let phi_inv_sq = GoldenExact::new(Rational::new(1, 1), Rational::new(-1, 1)); // 1 - φ = φ⁻²
            let one_minus_s = Rational::one() - s;
            let alpha = phi_inv_sq.to_f64() * one_minus_s.to_f64();
            let scale = 1.0 + alpha * norm_sq.to_f64().sqrt();

            // Stochastic noise (f64 for compatibility)
            let noise: f64 = rng.gen::<f64>() - 0.5;
            let noise_scale_factor = noise_scale * one_minus_s.to_f64();
            *v = *v * scale + noise * noise_scale_factor;
        }

        // Crystallize
        self.crystallize_cpu(&values, precision)
    }

    /// Native matrix-vector multiplication for GoldenExact lattice with BMM support.
    ///
    /// Performs Y = XW^T.
    /// Supports broadcasting and BMM (Batched Matrix Multiplication).
    ///
    /// Dispatch Logic:
    /// - Rank 2 x Rank 2 ([M, K], [N, K]): Standard MatMul (A @ B^T)
    /// - Rank 3+ x Rank 2: Broadcast B against A's batch dimensions.
    /// - Rank 3+ x Rank 3+: True Batched MatMul (BMM), requires matching batch dimensions.
    ///   e.g. [B, M, K] @ [B, N, K] -> [B, M, N]
    ///
    /// Resulting tensor is in Crystallized phase.
    pub fn matmul_core(&self, weights: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Input tensor cannot be empty".to_string(),
            ));
        }

        // --- 1. Shape Analysis & Dispatch Strategy ---
        let rank_a = self.shape.len();
        let rank_b = weights.shape.len();

        if rank_a < 2 {
            return Err(ResonantError::ShapeMismatch(format!(
                "Input tensor must be at least 2D for matmul, got {:?}",
                self.shape
            )));
        }

        let k_a = *self.shape.last().unwrap();
        // Weights is arguably [Out, In] (Rank 2) or [Batch, ..., Out, In] (Rank > 2)
        // If weights is Rank < 2, fail.
        if rank_b < 2 {
            return Err(ResonantError::ShapeMismatch(format!(
                "Weights must be at least 2D, got {:?}",
                weights.shape
            )));
        }
        let k_b = weights.shape[rank_b - 1]; // In_features is last dim?
                                             // Wait, standard convention in this codebase seems to be:
                                             // weights is [out_features, in_features] for Linear.
                                             // So K is dimension 1.
                                             // But for BMM (A @ B^T), if B is [Batch, N, K], then K is last.
                                             // matmul normally is A @ B.
                                             // This function explicitly says "Performs Y = XW^T".
                                             // If W is [N, K], then W^T is [K, N].
                                             // So A=[M, K] @ [K, N] -> [M, N].
                                             // Match dim is K.
                                             // So for W, the matching dimension is index 1 (last).

        if k_a != k_b {
            return Err(ResonantError::ShapeMismatch(format!(
                "Inner dimension mismatch: input ...{} vs weights ...{}",
                k_a, k_b
            )));
        }

        let batch_dims_a = &self.shape[0..rank_a - 2];
        let batch_dims_b = &weights.shape[0..rank_b - 2];

        // Check for BMM vs Broadcast
        let is_bmm = rank_b > 2;

        // --- 2. BMM Execution (Rank 3+ x Rank 3+) ---
        if is_bmm && rank_a >= 3 {
            // simplified BMM: Requires exactly matching batch dimensions for now
            // or standard broadcasting rules.
            // Let's implement strict BMM first: [Batch, M, K] @ [Batch, N, K] -> [Batch, M, N]

            // Flatten generic batch dims to single batch dim
            let batch_size_a: usize = batch_dims_a.iter().product();
            let batch_size_b: usize = batch_dims_b.iter().product();

            if batch_size_a != batch_size_b {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Batch size mismatch for BMM: {} vs {}",
                    batch_size_a, batch_size_b
                )));
            }

            let m = self.shape[rank_a - 2]; // Valid for rank >= 2
            let n = weights.shape[rank_b - 2];
            let k = k_a;

            // Check full batch shape strict equality for now
            if batch_dims_a != batch_dims_b {
                // Warning or Error? Let's error for safety until generic broadcast BMM is added
                return Err(ResonantError::ShapeMismatch(format!(
                    "Strict BMM requires matching batch shapes. Got {:?} vs {:?}",
                    batch_dims_a, batch_dims_b
                )));
            }

            // Handle CUDA path for BMM
            if self.phase == ResonantPhase::Flux || weights.phase == ResonantPhase::Flux {
                // Ensure device availability
                if let Some(device) = self.device.clone().or_else(|| weights.device.clone()) {
                    let a_slice = if let Some(flux) = &self.flux {
                        flux.clone()
                    } else {
                        device
                            .default_stream()
                            .clone_htod(&self.to_floats_core())
                            .map_err(|e| ResonantError::CudaError(e.to_string()))?
                    };

                    let b_slice = if let Some(flux) = &weights.flux {
                        flux.clone()
                    } else {
                        device
                            .default_stream()
                            .clone_htod(&weights.to_floats_core())
                            .map_err(|e| ResonantError::CudaError(e.to_string()))?
                    };

                    let mut c_slice = device
                        .default_stream()
                        .alloc_zeros::<f64>(batch_size_a * m * n)
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                    // BMM Kernel: [Batch, M, K] @ [Batch, N, K]^T -> [Batch, M, N]
                    cuda_bmm_nt_f64(
                        &device,
                        &mut c_slice,
                        &a_slice,
                        &b_slice,
                        batch_size_a,
                        m,
                        n,
                        k,
                    )
                    .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                    let mut host_data = vec![0.0f64; batch_size_a * m * n];
                    device
                        .default_stream()
                        .memcpy_dtoh(&c_slice, &mut host_data)
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                    let mut result_shape = self.shape.clone();
                    *result_shape.last_mut().unwrap() = n;

                    let mode_norms: Vec<f64> = (0..host_data.len())
                        .map(|i| ((i % n) as f64).powi(2))
                        .collect();
                    return Self::from_floats(&host_data, result_shape, mode_norms, self.precision);
                }
            }

            // Fallback BMM CPU - Perform exact Q(φ) multiplication and addition
            // Just loop over batch
            let mut result_lattice = Vec::with_capacity(batch_size_a * m * n);
            for b in 0..batch_size_a {
                let offset_a = b * m * k;
                let offset_b = b * n * k;

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = GoldenExact::zero();
                        for l in 0..k {
                            let val_a = self.lattice[offset_a + i * k + l];
                            // B is [Batch, N, K]
                            // We act as if we are doing A @ B^T.
                            // B^T would have shape [Batch, K, N].
                            // (B^T)[k, j] = B[j, k].
                            // So we access B at [Batch, row=j, col=l]
                            let val_b = weights.lattice[offset_b + j * k + l];
                            sum = sum + (val_a * val_b);
                        }
                        result_lattice.push(sum);
                    }
                }
            }

            let mut result_shape = self.shape.clone();
            *result_shape.last_mut().unwrap() = n;
            let mode_norms = (0..result_lattice.len())
                .map(|i| ((i % n) as f64).powi(2))
                .collect();

            return Self::from_lattice(result_lattice, result_shape, mode_norms);
        }

        // --- 3. Standard Broadcast Matmul (Rank 3+ x Rank 2) or (Rank 2 x Rank 2) ---

        // Calculate flattened batch dimension (M_total) of A
        // A is [Batch..., M_local, K]
        // Flatten [Batch..., M_local] -> M_total
        let m_total: usize = self.shape.iter().take(rank_a - 1).product();
        let k = k_a;
        let n = weights.shape[rank_b - 2]; // weights is [N, K]

        // Handle Empty
        if m_total == 0 {
            let mut result_shape = self.shape.clone();
            *result_shape.last_mut().unwrap() = n;
            return Ok(Self::zeros(result_shape, self.precision));
        }

        // CUDA Path (Standard or Broadcast)
        if self.phase == ResonantPhase::Flux || weights.phase == ResonantPhase::Flux {
            if let Some(device) = self.device.clone().or_else(|| weights.device.clone()) {
                let a_slice = if let Some(flux) = &self.flux {
                    flux.clone()
                } else {
                    device
                        .default_stream()
                        .clone_htod(&self.to_floats_core())
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?
                };

                let b_slice = if let Some(flux) = &weights.flux {
                    flux.clone()
                } else {
                    device
                        .default_stream()
                        .clone_htod(&weights.to_floats_core())
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?
                };

                let mut c_slice = device
                    .default_stream()
                    .alloc_zeros::<f64>(m_total * n)
                    .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                // Perform [M_total, K] @ [N, K]^T -> [M_total, N]
                // This handles Rank 2x2 and Broadcast Rank Nx2 equivalently
                cuda_matmul_nt_f64(&device, &mut c_slice, &a_slice, &b_slice, m_total, n, k)
                    .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                let mut host_data = vec![0.0f64; m_total * n];
                device
                    .default_stream()
                    .memcpy_dtoh(&c_slice, &mut host_data)
                    .map_err(|e| ResonantError::CudaError(e.to_string()))?;

                let mut result_shape = self.shape.clone();
                *result_shape.last_mut().unwrap() = n;

                let mode_norms: Vec<f64> =
                    (0..m_total * n).map(|i| ((i % n) as f64).powi(2)).collect();

                return Self::from_floats(&host_data, result_shape, mode_norms, self.precision);
            }
        }

        // CPU Path (Standard or Broadcast - GoldenExact)
        let mut result_lattice = Vec::with_capacity(m_total * n);

        // Perform exact Q(φ) multiplication and addition
        for m_idx in 0..m_total {
            for n_idx in 0..n {
                let mut sum = GoldenExact::zero();
                for k_idx in 0..k {
                    let x_val = self.lattice[m_idx * k + k_idx];
                    let w_val = weights.lattice[n_idx * k + k_idx];
                    sum = sum + (x_val * w_val);
                }
                result_lattice.push(sum);
            }
        }

        let mut result_norms = Vec::with_capacity(m_total * n);
        for _ in 0..m_total {
            for n_idx in 0..n {
                result_norms.push((n_idx as f64).powi(2));
            }
        }

        let mut result_shape = self.shape.clone();
        *result_shape.last_mut().unwrap() = n;

        Self::from_lattice(result_lattice, result_shape, result_norms)
    }

    /// Native bias addition for GoldenExact lattice.
    ///
    /// Performs self + bias where self is (batch, out) and bias is (out).
    pub fn add_bias_core(&mut self, bias: &ResonantTensor) -> Result<(), ResonantError> {
        if self.shape.len() != 2 {
            return Err(ResonantError::ShapeMismatch(
                "self must be 2D [batch, out]".to_string(),
            ));
        }

        let batch_size = self.shape[0];
        let out_features = self.shape[1];

        if bias.len() != out_features {
            return Err(ResonantError::ShapeMismatch(format!(
                "Bias dim {} must match layer dim {}",
                bias.len(),
                out_features
            )));
        }

        for b in 0..batch_size {
            for o in 0..out_features {
                self.lattice[b * out_features + o] =
                    self.lattice[b * out_features + o] + bias.lattice[o];
            }
        }

        Ok(())
    }

    /// Native ReLU activation on GoldenExact lattice.
    ///
    /// Snaps all negative lattice points to zero.
    pub fn relu_core(&mut self) {
        for val in self.lattice.iter_mut() {
            if val.to_f64() < 0.0 {
                *val = GoldenExact::zero();
            }
        }
    }

    /// Apply the Golden Recursion Map R(n) = floor(phi * n) to all lattice values.
    ///
    /// This scales all lattice values by phi, implementing one "recursion step".
    /// Used for building recursive layers without depth explosion.
    pub fn apply_recursion_core(&mut self) {
        // GoldenExact multiplication by phi is exact: (a, b) -> (b, a+b)
        // This is the Fibonacci scaling property of the golden lattice.
        for val in self.lattice.iter_mut() {
            *val = val.mul_phi();
        }
    }

    /// Apply the inverse recursion map R^{-1}(n) = floor(n / phi).
    ///
    /// This scales all lattice values by 1/phi.
    pub fn apply_inverse_recursion_core(&mut self) {
        for val in self.lattice.iter_mut() {
            *val = val.div_phi();
        }
    }

    /// Apply hierarchical pruning: snap values below threshold to zero.
    ///
    /// Threshold is specified as a fraction of q (e.g., 248 for q/248).
    /// Values with |v| < q/divisor are snapped to zero.
    pub fn prune_hierarchy_core(&mut self, q: f64, divisor: f64) {
        let q_rational = Rational::from_f64_nearest(q, self.precision);
        let divisor_rational = Rational::from_f64_nearest(divisor, self.precision);
        let threshold_rational = q_rational / divisor_rational;
        for val in self.lattice.iter_mut() {
            if val.magnitude_sq() < threshold_rational * threshold_rational {
                *val = GoldenExact::zero();
            }
        }
    }

    /// Apply sigmoid activation: σ(x) = 1 / (1 + e^(-x)) to all lattice values.
    ///
    /// This converts lattice values to floats, applies sigmoid, and snaps back to Q(φ).
    pub fn sigmoid_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self
            .lattice
            .iter()
            .map(|g| {
                let x = g.to_f64();
                1.0 / (1.0 + (-x).exp())
            })
            .collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Apply tanh activation to all lattice values.
    pub fn tanh_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64().tanh()).collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Apply GELU activation (erf-based) and snap to lattice.
    pub fn gelu_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self
            .lattice
            .iter()
            .map(|g| {
                let x = g.to_f64();
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x3 = x * x * x;
                let tanh_arg = 0.7978845608 * (x + 0.044715 * x3);
                0.5 * x * (1.0 + tanh_arg.tanh())
            })
            .collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Element-wise multiplication: self * other (Hadamard product).
    ///
    /// Both tensors must have the same shape.
    /// Result is in exact Q(φ) arithmetic.
    pub fn elementwise_mul_core(
        &self,
        other: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result_lattice: Vec<GoldenExact> = self
            .lattice
            .iter()
            .zip(other.lattice.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        // Mode norms: for element-wise ops, preserve the mode structure
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise addition: self + other.
    ///
    /// Both tensors must have the same shape.
    /// Result is in exact Q(φ) arithmetic.
    pub fn elementwise_add_core(
        &self,
        other: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result_lattice: Vec<GoldenExact> = self
            .lattice
            .iter()
            .zip(other.lattice.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Scalar multiplication: self * scalar.
    ///
    /// Multiplies every element by a scalar value.
    /// Result is in exact Q(φ) arithmetic.
    pub fn scalar_mul_core(&self, scalar: f64) -> Result<ResonantTensor, ResonantError> {
        // Convert scalar to Q(φ) with precision from self
        let scalar_golden = GoldenExact::find_nearest(scalar, self.precision);

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| *g * scalar_golden).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Scalar addition: self + scalar.
    ///
    /// Adds a scalar value to every element.
    /// Result is in exact Q(φ) arithmetic.
    pub fn scalar_add_core(&self, scalar: f64) -> Result<ResonantTensor, ResonantError> {
        // Convert scalar to Q(φ) with precision from self
        let scalar_golden = GoldenExact::find_nearest(scalar, self.precision);

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| *g + scalar_golden).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Negate: -self.
    ///
    /// Multiplies every element by -1.
    /// Equivalent to scalar_mul(-1.0) but more explicit.
    pub fn negate_core(&self) -> Result<ResonantTensor, ResonantError> {
        // -1 = -1 + 0·φ
        let neg_one = GoldenExact::new(Rational::new(-1, 1), Rational::new(0, 1));

        let result_lattice: Vec<GoldenExact> = self.lattice.iter().map(|g| *g * neg_one).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// One minus: 1 - self.
    ///
    /// Computes 1 - x for every element.
    /// Common pattern in gating mechanisms.
    pub fn one_minus_core(&self) -> Result<ResonantTensor, ResonantError> {
        // 1 = 1 + 0·φ
        let one = GoldenExact::new(Rational::new(1, 1), Rational::new(0, 1));
        // -1 = -1 + 0·φ
        let neg_one = GoldenExact::new(Rational::new(-1, 1), Rational::new(0, 1));

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| one + (*g * neg_one)).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Natural logarithm: ln(x).
    ///
    /// Converts to floats, applies ln, snaps back to Q(φ) lattice.
    pub fn log_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats
            .iter()
            .map(|&x| {
                let val = x.ln();
                // Handle ln(0) = -inf by clamping to a large negative number
                // This prevents snap_to_lattice from seeing -inf (which it improperly snaps to 0)
                if val.is_infinite() && val.is_sign_negative() {
                    -1000.0 // Sufficiently large negative number relative to expected log norms
                } else {
                    val
                }
            })
            .collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Natural exponential: e^x.
    ///
    /// Converts to floats, applies exp, snaps back to Q(φ) lattice.
    pub fn exp_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.exp()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Sine: sin(x).
    ///
    /// Converts to floats, applies sin, snaps back to Q(φ) lattice.
    pub fn sin_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.sin()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Cosine: cos(x).
    ///
    /// Converts to floats, applies cos, snaps back to Q(φ) lattice.
    pub fn cos_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.cos()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Power: x^exponent.
    ///
    /// Converts to floats, applies pow, snaps back to Q(φ) lattice.
    pub fn pow_core(&self, exponent: f64, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.powf(exponent)).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Square root: sqrt(x).
    ///
    /// Converts to floats, applies sqrt, snaps back to Q(φ) lattice.
    pub fn sqrt_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.sqrt()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Softmax activation along a specified dimension.
    ///
    /// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in that dimension.
    ///
    /// # Arguments
    /// * `dim` - Dimension to apply softmax along. Defaults to last dimension (-1).
    /// * `precision` - Lattice precision for snapping back to Q(φ).
    pub fn softmax_core(
        &mut self,
        dim: Option<isize>,
        precision: i64,
    ) -> Result<(), ResonantError> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return Err(ResonantError::ShapeMismatch(
                "Cannot softmax scalar/empty tensor".to_string(),
            ));
        }

        // Resolve dimension
        let dim_val = dim.unwrap_or(-1);
        let dim_idx = if dim_val < 0 {
            ndim as isize + dim_val
        } else {
            dim_val
        };

        if dim_idx < 0 || dim_idx >= ndim as isize {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-D tensor",
                dim_val, ndim
            )));
        }
        let dim_idx = dim_idx as usize;

        let floats = self.to_floats();
        let mut result_floats = floats.clone(); // Clone to use as output buffer

        // Generalized strided softmax
        let dim_size = self.shape[dim_idx];
        let outer_size: usize = self.shape[..dim_idx].iter().product();
        let inner_size: usize = self.shape[dim_idx + 1..].iter().product();
        let stride = inner_size; // indices along dim are separated by stride
        let jump = dim_size * inner_size; // moving to next outer index jumps this much

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * jump + inner;

                // For contiguous case (inner_size == 1), we can use the optimized helper
                if inner_size == 1 {
                    // Extract contiguous slice for this softmax group
                    let start = base_idx;
                    let end = start + dim_size;
                    let slice = &floats[start..end];
                    let softmax_result = self.softmax_1d(slice, dim_size);
                    for (i, &val) in softmax_result.iter().enumerate() {
                        result_floats[start + i] = val;
                    }
                } else {
                    // Strided case: elements are not contiguous
                    // 1. Find max for numerical stability
                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = base_idx + i * stride;
                        let val = floats[idx];
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    // 2. Compute exp and sum
                    let mut sum = 0.0;
                    let mut exps = Vec::with_capacity(dim_size);

                    for i in 0..dim_size {
                        let idx = base_idx + i * stride;
                        let val = (floats[idx] - max_val).exp();
                        exps.push(val);
                        sum += val;
                    }

                    // 3. Normalize
                    let inv_sum = 1.0 / sum;
                    for i in 0..dim_size {
                        let idx = base_idx + i * stride;
                        result_floats[idx] = exps[i] * inv_sum;
                    }
                }
            }
        }

        // Snap back to Q(φ) lattice
        self.lattice = snap_to_lattice(&result_floats, precision);
        self.precision = precision;
        self.recompute_syntony();

        Ok(())
    }

    /// Helper: Softmax for 1D slice (numerically stable).
    ///
    /// This is a utility function for computing softmax on a slice of floats.
    /// Uses the max-subtraction trick for numerical stability.
    ///
    /// # Arguments
    /// * `x` - Input slice of f64 values
    /// * `n` - Number of elements to process (uses first n elements of x)
    ///
    /// # Returns
    /// Vector of softmax probabilities that sum to 1.0
    pub fn softmax_1d(&self, x: &[f64], n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x_i - max) and sum
        let mut exp_vals = Vec::with_capacity(n);
        let mut sum = 0.0;

        for &val in x.iter().take(n) {
            let exp_val = (val - max_val).exp();
            exp_vals.push(exp_val);
            sum += exp_val;
        }

        // Normalize: exp(x_i - max) / sum
        exp_vals.iter().map(|&e| e / sum).collect()
    }

    /// Mean reduction along an optional dimension.
    pub fn mean_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Mean on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global mean
        if axis.is_none() {
            let total: f64 = self.lattice.iter().map(|g| g.to_f64()).sum();
            let mean = total / self.len() as f64;
            let lattice = vec![GoldenExact::find_nearest(mean, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64();
                }
                outputs.push(acc / axis_len as f64);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();

        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Variance reduction along an optional dimension (population variance).
    pub fn var_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Var on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global variance
        if axis.is_none() {
            let mean: f64 =
                self.lattice.iter().map(|g| g.to_f64()).sum::<f64>() / self.len() as f64;
            let var: f64 = self
                .lattice
                .iter()
                .map(|g| {
                    let v = g.to_f64() - mean;
                    v * v
                })
                .sum::<f64>()
                / self.len() as f64;
            let lattice = vec![GoldenExact::find_nearest(var, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64();
                }
                let mean = acc / axis_len as f64;
                let mut var_acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    let v = self.lattice[idx].to_f64() - mean;
                    var_acc += v * v;
                }
                outputs.push(var_acc / axis_len as f64);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();

        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Sum reduction along a dimension.
    ///
    /// Computes sum of elements along specified dimension, preserving Q(φ) lattice.
    pub fn sum_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Sum on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global sum
        if axis.is_none() {
            let total: f64 = self.lattice.iter().map(|g| g.to_f64()).sum();
            let lattice = vec![GoldenExact::find_nearest(total, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64();
                }
                outputs.push(acc);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();
        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Max reduction along a dimension.
    ///
    /// Finds maximum element along specified dimension.
    pub fn max_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Max on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global max
        if axis.is_none() {
            let max_val = self
                .lattice
                .iter()
                .map(|g| g.to_f64())
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            let lattice = vec![GoldenExact::find_nearest(max_val, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut max_val = f64::NEG_INFINITY;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    max_val = max_val.max(self.lattice[idx].to_f64());
                }
                outputs.push(max_val);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();
        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Min reduction along a dimension.
    ///
    /// Finds minimum element along specified dimension.
    pub fn min_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Min on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global min
        if axis.is_none() {
            let min_val = self
                .lattice
                .iter()
                .map(|g| g.to_f64())
                .fold(f64::INFINITY, |a, b| a.min(b));
            let lattice = vec![GoldenExact::find_nearest(min_val, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut min_val = f64::INFINITY;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    min_val = min_val.min(self.lattice[idx].to_f64());
                }
                outputs.push(min_val);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();
        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Product reduction along a dimension.
    ///
    /// Computes product of elements along specified dimension.
    pub fn prod_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Prod on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global product
        if axis.is_none() {
            let product: f64 = self.lattice.iter().map(|g| g.to_f64()).product();
            let lattice = vec![GoldenExact::find_nearest(product, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 1.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc *= self.lattice[idx].to_f64();
                }
                outputs.push(acc);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();
        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Norm reduction along a dimension.
    ///
    /// Computes p-norm along specified dimension.
    pub fn norm_core(
        &self,
        p: f64,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Norm on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global norm
        if axis.is_none() {
            let norm_val: f64 = self
                .lattice
                .iter()
                .map(|g| g.to_f64().abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p);
            let lattice = vec![GoldenExact::find_nearest(norm_val, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64().abs().powf(p);
                }
                outputs.push(acc.powf(1.0 / p));
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();
        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Masked select using a boolean mask.
    ///
    /// Selects elements where mask is true, flattening the result.
    pub fn masked_select_core(
        &self,
        mask: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape != mask.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Mask shape {:?} doesn't match tensor shape {:?}",
                mask.shape, self.shape
            )));
        }

        let mut selected_lattice = Vec::new();
        let mut selected_norms = Vec::new();

        for i in 0..self.lattice.len() {
            // Check if mask value is "true" (non-zero)
            if mask.lattice[i].to_f64().abs() > 1e-10 {
                selected_lattice.push(self.lattice[i]);
                selected_norms.push(self.mode_norm_sq[i]);
            }
        }

        let result_shape = vec![selected_lattice.len()];
        Self::from_lattice(selected_lattice, result_shape, selected_norms)
    }

    /// Gather elements along a dimension using indices.
    ///
    /// This is more general than index_select - allows arbitrary indexing.
    pub fn gather_core(
        &self,
        dim: usize,
        index: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        let ndim = self.shape.len();
        if dim >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-D tensor",
                dim, ndim
            )));
        }

        if index.shape.len() != self.shape.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Index tensor must have same number of dimensions as source tensor"
            )));
        }

        // For now, implement a simple case where index has same shape as self
        if index.shape != self.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Index tensor shape {:?} doesn't match source shape {:?}",
                index.shape, self.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        let mut result_norms = Vec::with_capacity(self.mode_norm_sq.len());

        // Calculate strides for source tensor
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        for i in 0..index.lattice.len() {
            let idx_val = index.lattice[i].to_f64() as usize;
            if idx_val >= self.shape[dim] {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx_val, dim, self.shape[dim]
                )));
            }

            // Calculate coordinates for this element
            let mut coords = vec![0; ndim];
            let mut remaining = i;
            for d in 0..ndim {
                coords[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            // Replace the target dimension with the index value
            coords[dim] = idx_val;

            // Calculate flat index in source
            let flat_idx = coords
                .iter()
                .zip(strides.iter())
                .map(|(c, s)| c * s)
                .sum::<usize>();
            result_lattice.push(self.lattice[flat_idx]);
            result_norms.push(self.mode_norm_sq[flat_idx]);
        }

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Scatter values into a tensor along a dimension.
    ///
    /// This is the inverse of gather.
    pub fn scatter_core(
        &mut self,
        dim: usize,
        index: &ResonantTensor,
        src: &ResonantTensor,
    ) -> Result<(), ResonantError> {
        let ndim = self.shape.len();
        if dim >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-D tensor",
                dim, ndim
            )));
        }

        if index.shape != src.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Index shape {:?} doesn't match source shape {:?}",
                index.shape, src.shape
            )));
        }

        // Calculate strides for destination tensor
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        for i in 0..index.lattice.len() {
            let idx_val = index.lattice[i].to_f64() as usize;
            if idx_val >= self.shape[dim] {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx_val, dim, self.shape[dim]
                )));
            }

            // Calculate coordinates for this element
            let mut coords = vec![0; ndim];
            let mut remaining = i;
            for d in 0..ndim {
                coords[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            // Replace the target dimension with the index value
            coords[dim] = idx_val;

            // Calculate flat index in destination
            let flat_idx = coords
                .iter()
                .zip(strides.iter())
                .map(|(c, s)| c * s)
                .sum::<usize>();
            self.lattice[flat_idx] = src.lattice[i];
        }

        Ok(())
    }

    /// Element-wise comparison: equal.
    pub fn eq_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let eq = if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(eq, 10)); // Low precision for boolean
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise comparison: not equal.
    pub fn ne_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let ne = if (a - b).abs() >= 1e-10 { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(ne, 10));
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise comparison: less than.
    pub fn lt_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let lt = if a < b { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(lt, 10));
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise comparison: less than or equal.
    pub fn le_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let le = if a <= b { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(le, 10));
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise comparison: greater than.
    pub fn gt_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let gt = if a > b { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(gt, 10));
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise comparison: greater than or equal.
    pub fn ge_core(&self, other: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let mut result_lattice = Vec::with_capacity(self.lattice.len());
        for i in 0..self.lattice.len() {
            let a = self.lattice[i].to_f64();
            let b = other.lattice[i].to_f64();
            let ge = if a >= b { 1.0 } else { 0.0 };
            result_lattice.push(GoldenExact::find_nearest(ge, 10));
        }

        let result_norms = vec![0.0; result_lattice.len()];
        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Concatenate tensors along a specified dimension.
    ///
    /// Preserves exact Q(φ) lattice arithmetic by directly combining GoldenExact values.
    /// All tensors must have the same shape except along the concatenation dimension.
    ///
    /// # Arguments
    /// * `tensors` - Slice of ResonantTensor references to concatenate
    /// * `dim` - Dimension to concatenate along (0-indexed)
    ///
    /// # Returns
    /// New ResonantTensor with tensors concatenated along dimension `dim`
    ///
    /// # Example
    /// ```ignore
    /// let a = ResonantTensor::from_floats(&[1.0, 2.0], vec![2], ...);
    /// let b = ResonantTensor::from_floats(&[3.0, 4.0], vec![2], ...);
    /// let c = ResonantTensor::concat(&[&a, &b], 0)?; // Shape: [4]
    /// ```
    pub fn concat_core(
        tensors: &[&ResonantTensor],
        dim: usize,
    ) -> Result<ResonantTensor, ResonantError> {
        if tensors.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Cannot concat empty tensor list".to_string(),
            ));
        }

        let first = tensors[0];
        let ndim = first.shape.len();

        if dim >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-dimensional tensor",
                dim, ndim
            )));
        }

        // Validate all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape.len() != ndim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    tensor.shape.len(),
                    ndim
                )));
            }
            for (d, (&s1, &s2)) in first.shape.iter().zip(tensor.shape.iter()).enumerate() {
                if d != dim && s1 != s2 {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Tensor {} has incompatible shape {:?}, expected {:?} along dim {}",
                        i, tensor.shape, first.shape, d
                    )));
                }
            }
        }

        // Compute result shape
        let mut result_shape = first.shape.clone();
        result_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        let outer: usize = first.shape[..dim].iter().product::<usize>().max(1);
        let inner: usize = first.shape[dim + 1..].iter().product::<usize>().max(1);
        let total_size = result_shape.iter().product();

        let mut result_lattice = Vec::with_capacity(total_size);
        let mut result_norms = Vec::with_capacity(total_size);

        // Copy contiguous slices for each outer index, preserving per-element mode norms
        for outer_idx in 0..outer {
            for tensor in tensors.iter() {
                let axis = tensor.shape[dim];
                for axis_idx in 0..axis {
                    let start = outer_idx * axis * inner + axis_idx * inner;
                    let end = start + inner;
                    result_lattice.extend_from_slice(&tensor.lattice[start..end]);
                    result_norms.extend_from_slice(&tensor.mode_norm_sq[start..end]);
                }
            }
        }

        Self::from_lattice(result_lattice, result_shape, result_norms)
    }

    /// View/Reshape tensor.
    ///
    /// Preserves exact Q(φ) lattice values. Data is not copied if possible, but
    /// ResonantTensor currently owns its data, so this effectively creates a new
    /// tensor sharing nothing but values (cloned).
    ///
    /// # Arguments
    /// * `shape` - New shape
    pub fn view_core(&self, shape: Vec<usize>) -> Result<ResonantTensor, ResonantError> {
        let current_size: usize = self.len();
        let new_size: usize = shape.iter().product();
        if current_size != new_size {
            return Err(ResonantError::ShapeMismatch(format!(
                "Cannot view tensor of size {} as {:?}",
                current_size, shape
            )));
        }

        // Mode norms are flat and element-wise, so they are preserved
        let lattice = self.lattice.clone();
        let mode_norm_sq = self.mode_norm_sq.clone();

        ResonantTensor::from_lattice(lattice, shape, mode_norm_sq)
    }

    /// Transpose tensor dimensions.
    ///
    /// Preserves exact Q(φ) lattice values.
    ///
    /// # Arguments
    /// * `dim0` - First dimension
    /// * `dim1` - Second dimension
    pub fn transpose_core(
        &self,
        dim0: usize,
        dim1: usize,
    ) -> Result<ResonantTensor, ResonantError> {
        let ndim = self.shape.len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension out of bounds: {} or {} >= {}",
                dim0, dim1, ndim
            )));
        }

        if dim0 == dim1 {
            return self.view_core(self.shape.clone());
        }

        let mut permutation: Vec<usize> = (0..ndim).collect();
        permutation.swap(dim0, dim1);

        self.permute_core(&permutation)
    }

    /// Permute tensor dimensions.
    ///
    /// # Arguments
    /// * `dims` - New order of dimensions
    pub fn permute_core(&self, dims: &[usize]) -> Result<ResonantTensor, ResonantError> {
        let ndim = self.shape.len();
        if dims.len() != ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Permutation length {} does not match tensor dimension {}",
                dims.len(),
                ndim
            )));
        }

        // Validate permutation
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();
        if sorted_dims != (0..ndim).collect::<Vec<_>>() {
            return Err(ResonantError::ShapeMismatch(
                "Invalid permutation indices".to_string(),
            ));
        }

        // Compute new shape
        let new_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();

        // Compute strides for the original tensor
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Permute
        // We iterate over the *new* tensor layout sequentially
        // and map back to the *old* tensor index.

        let len = self.len();
        let mut result_lattice = Vec::with_capacity(len);
        let mut result_norms = Vec::with_capacity(len);

        // Strides for the new tensor
        let mut new_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        for i in 0..len {
            // Convert flat index 'i' (in new layout) to coords in new layout
            let mut new_coords = vec![0; ndim];
            let mut temp = i;
            for d in 0..ndim {
                new_coords[d] = temp / new_strides[d];
                temp %= new_strides[d];
            }

            // Map to old coords
            // new_coords[k] is the coordinate for dimension k in new tensor.
            // dimension k in new tensor corresponds to dimension dims[k] in old tensor.
            // So old_coords[dims[k]] = new_coords[k].

            // Calculating flat index in old tensor:
            // index = sum(old_coords[d] * strides[d])
            let mut old_idx = 0;
            for (k, &dim_idx) in dims.iter().enumerate() {
                old_idx += new_coords[k] * strides[dim_idx];
            }

            result_lattice.push(self.lattice[old_idx]);
            result_norms.push(self.mode_norm_sq[old_idx]);
        }

        ResonantTensor::from_lattice(result_lattice, new_shape, result_norms)
    }

    /// Lattice-aware dropout.
    ///
    /// Sets coefficients to zero with probability p.
    /// Scales remaining coefficients by 1/(1-p) to maintain magnitude expectation.
    /// scaling uses Rational approximation if necessary, keeping values in Q(phi).
    pub fn dropout_core(&mut self, p: f64) {
        use rand::Rng;
        if p <= 0.0 {
            return;
        }
        if p >= 1.0 {
            for v in self.lattice.iter_mut() {
                *v = GoldenExact::zero();
            }
            self.recompute_syntony();
            return;
        }

        let scale = 1.0 / (1.0 - p);
        // Find nearest GoldenExact for scale
        let scale_golden = GoldenExact::find_nearest(scale, self.precision);

        let mut rng = rand::thread_rng();

        for v in self.lattice.iter_mut() {
            if rng.gen::<f64>() < p {
                *v = GoldenExact::zero();
            } else {
                *v = *v * scale_golden;
            }
        }
        self.recompute_syntony();
    }

    /// Index select (gather) along a dimension.
    ///
    /// Preserves exact Q(φ) lattice values.
    ///
    /// # Arguments
    /// * `indices` - Indices to select
    /// * `dim` - Dimension to select along
    pub fn index_select_core(
        &self,
        indices: &[usize],
        dim: usize,
    ) -> Result<ResonantTensor, ResonantError> {
        let ndim = self.shape.len();
        if dim >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-dimensional tensor",
                dim, ndim
            )));
        }

        let mut result_shape = self.shape.clone();
        result_shape[dim] = indices.len();

        let outer: usize = self.shape[..dim].iter().product::<usize>().max(1);
        let inner: usize = self.shape[dim + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[dim];
        let total = result_shape.iter().product();

        let mut result_lattice = Vec::with_capacity(total);
        let mut result_norms = Vec::with_capacity(total);

        for outer_idx in 0..outer {
            for &idx in indices {
                if idx >= axis_len {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Index {} out of bounds for dim {} size {}",
                        idx, dim, axis_len
                    )));
                }

                let start = outer_idx * axis_len * inner + idx * inner;
                let end = start + inner;

                result_lattice.extend_from_slice(&self.lattice[start..end]);
                result_norms.extend_from_slice(&self.mode_norm_sq[start..end]);
            }
        }

        Self::from_lattice(result_lattice, result_shape, result_norms)
    }

    /// Layer normalization with optional golden target variance.
    ///
    /// Normalizes across the last dimension (features).
    /// For 2D tensors [batch, features], normalizes each batch sample independently.
    ///
    /// # Arguments
    /// * `gamma` - Optional scale parameter (broadcast to features)
    /// * `beta` - Optional shift parameter (broadcast to features)
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    /// * `golden_target` - If true, scale to target variance = 1/φ
    pub fn layer_norm_core(
        &self,
        gamma: Option<&ResonantTensor>,
        beta: Option<&ResonantTensor>,
        eps: f64,
        golden_target: bool,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape.len() < 1 {
            return Err(ResonantError::ShapeMismatch(
                "LayerNorm requires at least 1D tensor".to_string(),
            ));
        }

        let feature_dim = self.shape[self.shape.len() - 1];
        let batch_size: usize = self.shape[..self.shape.len() - 1].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        // Validate gamma/beta shapes if provided
        if let Some(g) = gamma {
            if g.len() != feature_dim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Gamma length {} must match feature_dim {}",
                    g.len(),
                    feature_dim
                )));
            }
        }
        if let Some(b) = beta {
            if b.len() != feature_dim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Beta length {} must match feature_dim {}",
                    b.len(),
                    feature_dim
                )));
            }
        }

        let mut result_floats = Vec::with_capacity(self.len());

        // Process each sample
        for b in 0..batch_size {
            let offset = b * feature_dim;
            let sample: Vec<f64> = (0..feature_dim)
                .map(|i| self.lattice[offset + i].to_f64())
                .collect();

            // Compute mean
            let mean: f64 = sample.iter().sum::<f64>() / feature_dim as f64;

            // Compute variance
            let var: f64 =
                sample.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / feature_dim as f64;

            // Compute reciprocal standard deviation
            let mut rstd = 1.0 / (var + eps).sqrt();

            // Golden scaling: scale to variance = 1/φ
            if golden_target {
                rstd *= PHI_INV.sqrt();
            }

            // Normalize and apply affine transform
            for i in 0..feature_dim {
                let mut val = (sample[i] - mean) * rstd;

                // Apply gamma (scale)
                if let Some(g) = gamma {
                    val *= g.lattice[i].to_f64();
                }

                // Apply beta (shift)
                if let Some(b) = beta {
                    val += b.lattice[i].to_f64();
                }

                result_floats.push(val);
            }
        }

        // Snap to lattice and create result tensor
        let result_lattice = snap_to_lattice(&result_floats, self.precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Complete a full D→H cycle for a batch of samples.
    ///
    /// Assumes the first dimension of the shape is the batch dimension.
    /// Applies the same mode_norm_sq sequence to each sample in the batch.
    ///
    /// Returns a vector of new syntony values for each sample.
    pub fn run_batch_cpu_cycle(
        &mut self,
        noise_scale: f64,
        precision: i64,
    ) -> Result<Vec<f64>, ResonantError> {
        use rand::Rng;

        if self.shape.len() < 2 {
            return Err(ResonantError::ShapeMismatch(
                "batch_cpu_cycle requires at least 2 dimensions [batch, dim, ...]".to_string(),
            ));
        }

        let batch_size = self.shape[0];
        let sample_dim: usize = self.shape[1..].iter().product();

        if self.mode_norm_sq.len() != sample_dim && self.mode_norm_sq.len() != self.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "mode_norm_sq length {} must match sample dimension {} or total length {}",
                self.mode_norm_sq.len(),
                sample_dim,
                self.len()
            )));
        }

        // Wake flux
        let mut values = self.to_floats_core();
        self.phase = ResonantPhase::Flux;

        let mut rng = rand::thread_rng();
        let mut batch_syntony = Vec::with_capacity(batch_size);

        // Process each sample in the batch
        for b in 0..batch_size {
            let offset = b * sample_dim;
            let sample_slice = &mut values[offset..offset + sample_dim];

            // Current syntony (approximate for the whole batch or per-sample?)
            // For now, use the tensor's global syntony to drive the D-phase
            let s = self.syntony;
            let alpha = PHI_INV_SQ * (1.0 - s);

            for (i, v) in sample_slice.iter_mut().enumerate() {
                let norm_sq = self.mode_norm_sq[i];
                let scale = 1.0 + alpha * norm_sq.sqrt();
                let noise: f64 = rng.gen::<f64>() - 0.5;
                *v = *v * scale + noise * noise_scale * (1.0 - s);
            }
        }

        // Crystallize: returns exact lattice
        self.lattice =
            harmonize_and_crystallize(&values, &vec![0.0; values.len()], self.syntony, precision);
        // Note: harmonize_and_crystallize above uses dummy mode_norms because we already applied DH logic?
        // Wait, harmonize_and_crystallize applies H-phase (attenuation).
        // I should probably implement a batch version of harmonize_and_crystallize too.

        // Actually, let's just use the logic directly here for clarity
        let beta = PHI_INV * self.syntony;
        self.lattice = values
            .iter()
            .enumerate()
            .map(|(idx, &val)| {
                let i = idx % sample_dim;
                let norm_sq = self.mode_norm_sq[i];
                let golden_weight = (-norm_sq / PHI).exp();
                let h_scale = 1.0 - beta * (1.0 - golden_weight);
                GoldenExact::find_nearest(val * h_scale, precision)
            })
            .collect();

        self.precision = precision;
        self.phase = ResonantPhase::Crystallized;

        // Compute per-sample syntony for return values
        // Note: each sample shares the same mode structure, so we use the first sample_dim norms
        let sample_norms = &self.mode_norm_sq[0..sample_dim];
        for b in 0..batch_size {
            let offset = b * sample_dim;
            let sample_lattice = &self.lattice[offset..offset + sample_dim];
            batch_syntony.push(compute_lattice_syntony(sample_lattice, sample_norms));
        }

        // Update global syntony as average
        let sum_syntony: Rational = batch_syntony.iter().map(|&x| Rational::from_f64_nearest(x, self.precision)).sum();
        let batch_size_rational = Rational::from_i64(batch_size as i64);
        self.syntony = sum_syntony / batch_size_rational;

        Ok(batch_syntony)
    }
}

// =========================================================================
// PyO3 Methods
// =========================================================================

#[pymethods]
impl ResonantTensor {
    /// Create a ResonantTensor from a list of floats.
    ///
    /// Args:
    ///     data: List of floating-point values
    ///     shape: Shape of the tensor
    ///     mode_norm_sq: Precomputed |n|² for each mode (optional)
    ///     precision: Maximum coefficient for golden lattice (default: 100)
    ///
    /// Returns:
    ///     A new ResonantTensor in Crystallized phase
    #[new]
    #[pyo3(signature = (data, shape, mode_norm_sq=None, precision=100))]
    fn py_new(
        data: Vec<f64>,
        shape: Vec<usize>,
        mode_norm_sq: Option<Vec<f64>>,
        precision: i64,
    ) -> PyResult<Self> {
        let mode_norms =
            mode_norm_sq.unwrap_or_else(|| (0..data.len()).map(|i| (i as f64).powi(2)).collect());
        Self::from_floats(&data, shape, mode_norms, precision).map_err(|e| e.into())
    }

    /// Create from a list of GoldenExact values.
    #[staticmethod]
    #[pyo3(signature = (lattice, shape, mode_norm_sq=None))]
    fn from_golden_exact(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mode_norms_f64 = mode_norm_sq
            .unwrap_or_else(|| (0..lattice.len()).map(|i| (i as f64).powi(2)).collect());
        let mode_norms_exact: Vec<Rational> = mode_norms_f64.iter().map(|&x| Rational::from_f64_nearest(x, 100)).collect();
        Self::from_lattice(lattice, shape, mode_norms_exact).map_err(|e| e.into())
    }

    /// Create a zero-initialized tensor (using exact GoldenExact::zero).
    #[staticmethod]
    #[pyo3(name = "zeros", signature = (shape, precision=100))]
    fn py_zeros(shape: Vec<usize>, precision: i64) -> Self {
        Self::zeros(shape, precision)
    }

    /// Create a ResonantTensor directly on GPU from floats.
    ///
    /// This bypasses the CPU creation → GPU transfer for better performance.
    ///
    /// Args:
    ///     data: List of floating-point values
    ///     shape: Shape of the tensor
    ///     device_idx: CUDA device index (default: 0)
    ///     mode_norm_sq: Precomputed |n|² for each mode (optional)
    ///     precision: Maximum coefficient for golden lattice (default: 100)
    ///
    /// Returns:
    ///     A new ResonantTensor in Crystallized phase on GPU
    #[staticmethod]
    #[pyo3(signature = (data, shape, device_idx=0, mode_norm_sq=None, precision=100))]
    fn from_floats_cuda(
        data: Vec<f64>,
        shape: Vec<usize>,
        device_idx: usize,
        mode_norm_sq: Option<Vec<f64>>,
        precision: i64,
    ) -> PyResult<Self> {
        // Get CUDA device
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA device error: {}", e))
        })?;

        // Create on CPU first (we need the lattice for GPU operations)
        let mode_norms =
            mode_norm_sq.unwrap_or_else(|| (0..data.len()).map(|i| (i as f64).powi(2)).collect());
        let mut tensor = Self::from_floats(&data, shape, mode_norms, precision).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor creation error: {:?}", e))
        })?;

        // Move directly to GPU by initializing flux
        tensor.wake_flux(device).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("GPU initialization error: {:?}", e))
        })?;

        Ok(tensor)
    }

    /// Create a ResonantTensor directly on GPU from floats (fallback for non-CUDA builds).
    #[cfg(not(feature = "cuda"))]
    #[staticmethod]
    #[pyo3(signature = (_data, _shape, _device_idx=0, _mode_norm_sq=None, _precision=100))]
    fn from_floats_cuda(
        _data: Vec<f64>,
        _shape: Vec<usize>,
        _device_idx: usize,
        _mode_norm_sq: Option<Vec<f64>>,
        _precision: i64,
    ) -> PyResult<Self> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not compiled",
        ))
    }

    /// Matrix multiplication: self @ weights.
    ///
    /// Args:
    ///     weights: Another ResonantTensor (W)
    ///
    /// Returns:
    ///     New ResonantTensor representing self * weights^T
    fn matmul(&self, weights: &ResonantTensor) -> PyResult<Self> {
        self.matmul_core(weights).map_err(|e| e.into())
    }

    /// Add bias to the tensor.
    fn add_bias(&mut self, bias: &ResonantTensor) -> PyResult<()> {
        self.add_bias_core(bias).map_err(|e| e.into())
    }

    /// Apply ReLU activation.
    fn relu(&mut self) {
        self.relu_core();
    }

    /// Complete a full D→H cycle for a batch of samples.
    fn batch_cpu_cycle(&mut self, noise_scale: f64, precision: i64) -> PyResult<Vec<f64>> {
        self.run_batch_cpu_cycle(noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Convert lattice to a list of GoldenExact objects.
    fn to_lattice_list(&self) -> Vec<GoldenExact> {
        self.lattice.clone()
    }

    /// Get the current syntony value.
    #[getter]
    fn get_syntony(&self) -> f64 {
        self.syntony.to_f64()
    }

    /// Get the current phase as a string.
    #[getter]
    fn get_phase(&self) -> String {
        self.phase.to_string()
    }

    /// Get the device index (if any).
    #[pyo3(name = "device_idx")]
    fn py_device_idx(&self) -> Option<usize> {
        self.device_idx()
    }

    /// Get the tensor shape.
    #[getter]
    fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get the number of elements.
    fn __len__(&self) -> usize {
        self.len()
    }

    /// Get the precision used for last crystallization.
    #[getter]
    fn get_precision(&self) -> i64 {
        self.precision()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "ResonantTensor(shape={:?}, phase={}, syntony={:.4}, precision={})",
            self.shape, self.phase, self.syntony, self.precision
        )
    }

    /// Convert to list of floats.
    fn to_list(&self) -> Vec<f64> {
        self.to_floats()
    }

    /// Alias for to_list()
    fn to_floats(&self) -> Vec<f64> {
        self.to_floats_rust()
    }

    /// Get the lattice as a list of GoldenExact values.
    fn get_lattice(&self) -> Vec<GoldenExact> {
        self.lattice.clone()
    }

    /// Get the mode norm squared values.
    fn get_mode_norm_sq(&self) -> Vec<f64> {
        self.mode_norm_sq.iter().map(|x| x.to_f64()).collect()
    }

    /// Crystallize CPU flux values (for testing without GPU).
    fn crystallize_from_values(&mut self, values: Vec<f64>, precision: i64) -> PyResult<f64> {
        self.crystallize_cpu(&values, precision)
            .map_err(|e| e.into())
    }

    /// Wake flux in CPU mode (for testing without GPU).
    /// Returns the projected float values.
    fn wake_flux_values(&mut self) -> PyResult<Vec<f64>> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            )
            .into());
        }

        let floats = self.to_floats_core();
        self.phase = ResonantPhase::Flux;
        Ok(floats)
    }

    /// Complete a full D→H cycle in CPU mode.
    ///
    /// This simulates what would happen with GPU differentiation
    /// by applying noise to the values and then crystallizing.
    fn cpu_cycle(&mut self, noise_scale: f64, precision: i64) -> PyResult<f64> {
        self.run_cpu_cycle(noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Apply the Golden Recursion Map R(n) = floor(phi * n).
    ///
    /// Scales all lattice values by phi (exactly).
    fn apply_recursion(&mut self) {
        self.apply_recursion_core();
    }

    /// Apply the inverse Golden Recursion Map R^{-1}(n).
    ///
    /// Scales all lattice values by 1/phi (exactly).
    fn apply_inverse_recursion(&mut self) {
        self.apply_inverse_recursion_core();
    }

    /// Apply hierarchical pruning to the lattice.
    ///
    /// Values with magnitude < q/divisor are snapped to zero.
    #[pyo3(signature = (q, divisor=248.0))]
    fn prune_hierarchy(&mut self, q: f64, divisor: f64) {
        self.prune_hierarchy_core(q, divisor);
    }

    /// Apply sigmoid activation: σ(x) = 1 / (1 + e^(-x)).
    ///
    /// Converts to floats, applies sigmoid, snaps back to Q(φ) lattice.
    ///
    /// Args:
    ///     precision: Lattice precision for crystallization (default: 100)
    #[pyo3(signature = (precision=100))]
    fn sigmoid(&mut self, precision: i64) {
        self.sigmoid_core(precision);
    }

    /// Apply tanh activation.
    ///
    /// Args:
    ///     precision: Lattice precision for crystallization (default: 100)
    #[pyo3(signature = (precision=100))]
    fn tanh(&mut self, precision: i64) {
        self.tanh_core(precision);
    }

    /// Apply GELU activation (erf-based) and snap to Q(φ) lattice.
    #[pyo3(signature = (precision=100))]
    fn gelu(&mut self, precision: i64) {
        self.gelu_core(precision);
    }

    /// Element-wise multiplication (Hadamard product): self * other.
    ///
    /// Args:
    ///     other: Another ResonantTensor with the same shape
    ///
    /// Returns:
    ///     New ResonantTensor with element-wise product
    fn elementwise_mul(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.elementwise_mul_core(other).map_err(|e| e.into())
    }

    /// Element-wise addition: self + other.
    ///
    /// Args:
    ///     other: Another ResonantTensor with the same shape
    ///
    /// Returns:
    ///     New ResonantTensor with element-wise sum
    fn elementwise_add(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.elementwise_add_core(other).map_err(|e| e.into())
    }

    /// Scalar multiplication: self * scalar.
    ///
    /// Multiplies every element by a scalar value.
    ///
    /// Args:
    ///     scalar: Scalar value to multiply by
    ///
    /// Returns:
    ///     New ResonantTensor with all elements multiplied by scalar
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> y = x.scalar_mul(2.0)
    ///     >>> y.to_floats()
    ///     [2.0, 4.0, 6.0]
    fn scalar_mul(&self, scalar: f64) -> PyResult<Self> {
        self.scalar_mul_core(scalar).map_err(|e| e.into())
    }

    /// Scalar addition: self + scalar.
    ///
    /// Adds a scalar value to every element.
    ///
    /// Args:
    ///     scalar: Scalar value to add
    ///
    /// Returns:
    ///     New ResonantTensor with scalar added to all elements
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> y = x.scalar_add(10.0)
    ///     >>> y.to_floats()
    ///     [11.0, 12.0, 13.0]
    fn scalar_add(&self, scalar: f64) -> PyResult<Self> {
        self.scalar_add_core(scalar).map_err(|e| e.into())
    }

    /// Negate: -self.
    ///
    /// Multiplies every element by -1.
    ///
    /// Returns:
    ///     New ResonantTensor with all elements negated
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, -2.0, 3.0], [3])
    ///     >>> y = x.negate()
    ///     >>> y.to_floats()
    ///     [-1.0, 2.0, -3.0]
    fn negate(&self) -> PyResult<Self> {
        self.negate_core().map_err(|e| e.into())
    }

    /// One minus: 1 - self.
    ///
    /// Computes 1 - x for every element.
    /// Common pattern in gating mechanisms.
    ///
    /// Returns:
    ///     New ResonantTensor with 1 - x for each element
    ///
    /// Example:
    ///     >>> x = ResonantTensor([0.2, 0.5, 0.8], [3])
    ///     >>> y = x.one_minus()
    ///     >>> y.to_floats()
    ///     [0.8, 0.5, 0.2]
    fn one_minus(&self) -> PyResult<Self> {
        self.one_minus_core().map_err(|e| e.into())
    }

    /// Natural logarithm.
    #[pyo3(signature = (precision=None))]
    fn log(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.log_core(prec).map_err(|e| e.into())
    }

    /// Natural exponential.
    #[pyo3(signature = (precision=None))]
    fn exp(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.exp_core(prec).map_err(|e| e.into())
    }

    /// Sine function.
    #[pyo3(signature = (precision=None))]
    fn sin(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.sin_core(prec).map_err(|e| e.into())
    }

    /// Cosine function.
    #[pyo3(signature = (precision=None))]
    fn cos(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.cos_core(prec).map_err(|e| e.into())
    }

    /// Power function.
    #[pyo3(signature = (exponent, precision=None))]
    fn pow(&self, exponent: f64, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.pow_core(exponent, prec).map_err(|e| e.into())
    }

    /// Square root.
    #[pyo3(signature = (precision=None))]
    fn sqrt(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.sqrt_core(prec).map_err(|e| e.into())
    }

    /// Applies softmax normalization along a specified dimension.
    ///
    /// Uses numerically stable computation: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    /// Snaps result back to Q(φ) lattice with specified precision.
    ///
    /// Args:
    ///     dim: Dimension to apply softmax along (default: -1, last dimension)
    ///     precision: Precision for lattice snapping (default: 32)
    ///
    /// Returns:
    ///     Self (mutated in-place)
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> x.softmax()
    ///     >>> x.to_floats()
    ///     [0.09003057, 0.24472847, 0.66524096]
    #[pyo3(signature = (dim=None, precision=32))]
    fn softmax(&mut self, dim: Option<isize>, precision: i64) -> PyResult<()> {
        self.softmax_core(dim, precision).map_err(|e| e.into())
    }

    /// Concatenate tensors along a specified dimension.
    ///
    /// Preserves exact Q(φ) lattice arithmetic. All tensors must have compatible shapes.
    ///
    /// Args:
    ///     tensors: List of ResonantTensor objects to concatenate
    ///     dim: Dimension along which to concatenate (default: -1, last dimension)
    ///
    /// Returns:
    ///     New ResonantTensor with tensors concatenated
    ///
    /// Example:
    ///     >>> a = ResonantTensor([1.0, 2.0], [2])
    ///     >>> b = ResonantTensor([3.0, 4.0], [2])
    ///     >>> c = ResonantTensor.concat([a, b], dim=0)
    ///     >>> c.shape
    ///     [4]
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=-1))]
    fn concat(py: Python, tensors: Vec<Py<ResonantTensor>>, dim: i32) -> PyResult<Self> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty tensor list",
            ));
        }

        // Borrow all tensors and collect PyRef objects
        let borrowed_tensors: Vec<pyo3::PyRef<ResonantTensor>> =
            tensors.iter().map(|t| t.borrow(py)).collect();

        // Convert to slice of references
        let tensor_refs: Vec<&ResonantTensor> = borrowed_tensors.iter().map(|t| &**t).collect();

        // Handle negative dimension indexing
        let ndim = tensor_refs[0].shape.len() as i32;
        let actual_dim = if dim < 0 {
            (ndim + dim) as usize
        } else {
            dim as usize
        };

        Self::concat_core(&tensor_refs, actual_dim).map_err(|e| e.into())
    }

    /// Select slices along a dimension.
    ///
    /// Args:
    ///     indices: List of indices to selected
    ///     dim: Dimension to select along (default: 0)
    ///
    /// Returns:
    ///     New ResonantTensor with selected slices
    #[pyo3(signature = (indices, dim=0))]
    fn index_select(&self, indices: Vec<usize>, dim: i32) -> PyResult<Self> {
        let ndim = self.shape.len() as i32;
        let actual_dim = if dim < 0 {
            (ndim + dim) as usize
        } else {
            dim as usize
        };

        self.index_select_core(&indices, actual_dim)
            .map_err(|e| e.into())
    }

    /// View/Reshape tensor.
    #[pyo3(signature = (shape))]
    fn view(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.view_core(shape).map_err(|e| e.into())
    }

    /// Alias for view.
    #[pyo3(signature = (shape))]
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.view_core(shape).map_err(|e| e.into())
    }

    /// Transpose dimensions.
    #[pyo3(signature = (dim0, dim1))]
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        self.transpose_core(dim0, dim1).map_err(|e| e.into())
    }

    /// Permute dimensions.
    #[pyo3(signature = (dims))]
    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        self.permute_core(&dims).map_err(|e| e.into())
    }

    /// Apply dropout.
    #[pyo3(signature = (p))]
    fn dropout(&mut self, p: f64) {
        self.dropout_core(p);
    }

    /// Layer normalization with optional golden target variance.
    ///
    /// Normalizes across the last dimension (features).
    ///
    /// Args:
    ///     gamma: Optional scale parameter (ResonantTensor of shape [features])
    ///     beta: Optional shift parameter (ResonantTensor of shape [features])
    ///     eps: Small constant for numerical stability (default: 1e-8)
    ///     golden_target: If True, scale to target variance = 1/φ (default: True)
    ///
    /// Returns:
    ///     New ResonantTensor with normalized values
    #[pyo3(signature = (gamma=None, beta=None, eps=1e-8, golden_target=true))]
    fn layer_norm(
        &self,
        gamma: Option<&ResonantTensor>,
        beta: Option<&ResonantTensor>,
        eps: f64,
        golden_target: bool,
    ) -> PyResult<Self> {
        self.layer_norm_core(gamma, beta, eps, golden_target)
            .map_err(|e| e.into())
    }

    /// Mean reduction along an optional dimension.
    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn mean(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let ndim = self.shape.len() as i32;
        let axis = match dim {
            Some(d) if d < 0 => {
                let adj = ndim + d;
                if adj < 0 {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Dimension {} out of bounds for {}-D tensor",
                        d, ndim
                    ))
                    .into());
                }
                Some(adj as usize)
            }
            Some(d) => Some(d as usize),
            None => None,
        };
        let prec = precision.unwrap_or(self.precision);
        self.mean_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    /// Variance reduction along an optional dimension (population variance).
    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn var(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.var_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn sum(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.sum_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn max(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.max_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn min(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.min_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn prod(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.prod_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    #[pyo3(signature = (p=2.0, dim=None, keepdim=false, precision=None))]
    fn norm(
        &self,
        p: f64,
        dim: Option<i32>,
        keepdim: bool,
        precision: Option<i64>,
    ) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        let axis = dim.map(|d| d as usize);
        self.norm_core(p, axis, keepdim, prec).map_err(|e| e.into())
    }

    fn masked_select(&self, mask: &ResonantTensor) -> PyResult<Self> {
        self.masked_select_core(mask).map_err(|e| e.into())
    }

    fn gather(&self, dim: usize, index: &ResonantTensor) -> PyResult<Self> {
        self.gather_core(dim, index).map_err(|e| e.into())
    }

    fn scatter(
        &mut self,
        dim: usize,
        index: &ResonantTensor,
        src: &ResonantTensor,
    ) -> PyResult<()> {
        self.scatter_core(dim, index, src).map_err(|e| e.into())
    }

    fn __eq__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.eq_core(other).map_err(|e| e.into())
    }

    fn __ne__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.ne_core(other).map_err(|e| e.into())
    }

    fn __lt__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.lt_core(other).map_err(|e| e.into())
    }

    fn __le__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.le_core(other).map_err(|e| e.into())
    }

    fn __gt__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.gt_core(other).map_err(|e| e.into())
    }

    fn __ge__(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.ge_core(other).map_err(|e| e.into())
    }

    // =========================================================================
    // CUDA-specific Methods (compiled only with cuda feature)
    // =========================================================================

    /// Run a full D→H cycle using CUDA for the D-phase and CPU crystallization for H-phase.
    ///
    /// Args:
    ///     device_idx: CUDA device index (usize)
    ///     noise_scale: stochastic noise scale for D-phase
    ///     precision: precision for crystallization
    #[pyo3(signature = (device_idx, noise_scale=0.1, precision=100))]
    fn cuda_cycle_gpu(
        &mut self,
        device_idx: usize,
        noise_scale: f64,
        precision: i64,
    ) -> PyResult<f64> {
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("cuda device error: {}", e))
        })?;

        self.cuda_cycle(device, noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Wake flux with D-phase on the specified CUDA device (GPU-only helper).
    #[pyo3(signature = (device_idx, noise_scale=0.1))]
    fn wake_flux_with_d_phase_py(&mut self, device_idx: usize, noise_scale: f64) -> PyResult<()> {
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("cuda device error: {}", e))
        })?;

        self.wake_flux_with_d_phase(device, noise_scale)
            .map_err(|e| e.into())
    }

    /// Move tensor to CUDA device.
    ///
    /// Creates a new tensor on the specified GPU.
    /// If the current tensor has active flux (is on another GPU), it is downloaded first.
    #[pyo3(signature = (device_idx))]
    pub fn to_device(&self, device_idx: usize) -> PyResult<Self> {
        // 1. Resolve current values (download if on GPU)
        let values = self.resolve_values_core()?;

        // 2. Create new tensor (snaps to lattice)
        let mut new_tensor = Self::from_floats(
            &values,
            self.shape.clone(),
            self.mode_norm_sq.clone(),
            self.precision,
        )
        .map_err(|e| PyErr::from(e))?;

        // 3. Move to target device (use F32 for GPU operations)
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("cuda device error: {}", e))
        })?;

        // Create F32 flux for GPU operations
        let floats_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let gpu_slice = device
            .default_stream()
            .clone_htod(&floats_f32)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        new_tensor.flux = None;
        new_tensor.flux_f32 = Some(gpu_slice);
        new_tensor.device = Some(device);
        new_tensor.phase = ResonantPhase::Flux;
        new_tensor.set_device_idx(device_idx);

        Ok(new_tensor)
    }

    /// Move tensor to CPU.
    ///
    /// If the tensor is on GPU (Flux phase), downloads the values.
    /// Returns a new tensor on CPU (Crystallized phase).
    #[pyo3(signature = ())]
    pub fn to_cpu(&self) -> PyResult<Self> {
        // 1. Resolve current values
        let values = self.resolve_values_core()?;

        // 2. Create new tensor (snaps to lattice)
        Self::from_floats(
            &values,
            self.shape.clone(),
            self.mode_norm_sq.clone(),
            self.precision,
        )
        .map_err(|e| PyErr::from(e))
    }

    // ============================================================================
    // SGC Extensions: Numerically Stable Operations
    // ============================================================================

    /// Stable DHSR cycle with adaptive phi-scaling to prevent collapse.
    ///
    /// Prevents evolution collapse by detecting small values and applying
    /// phi-scaled amplification before crystallization.
    ///
    /// Args:
    ///     noise_scale: Stochastic noise amplitude (default: 0.01)
    ///     base_precision: Base lattice precision (default: 100)
    ///
    /// Returns:
    ///     New syntony value after cycle
    #[pyo3(signature = (noise_scale=0.01, base_precision=100))]
    fn sgc_stable_cycle(&mut self, noise_scale: f64, base_precision: i64) -> PyResult<f64> {
        use crate::resonant::phi_ops::ResonantTensorExt;
        ResonantTensorExt::sgc_stable_cycle(self, noise_scale, base_precision)
    }

    /// Generate phi-weighted seed for SGC structure regeneration.
    ///
    /// Creates seeds using phi-scaled statistics to ensure generative stability.
    ///
    /// Args:
    ///     target_stats: [mean, std, max_val, ...] statistics from target data
    ///
    /// Returns:
    ///     New ResonantTensor seed optimized for SGC
    fn sgc_generate_phi_seed(&self, target_stats: Vec<f64>) -> PyResult<Self> {
        use crate::resonant::phi_ops::ResonantTensorExt;
        ResonantTensorExt::sgc_generate_phi_seed(self, &target_stats)
    }

    /// Multi-scale phi evolution for robust SGC regeneration.
    ///
    /// Evolves at multiple phi-scaled magnitudes to find stable generative patterns.
    ///
    /// Args:
    ///     scales: List of scaling factors to try
    ///     steps_per_scale: Evolution steps at each scale
    ///
    /// Returns:
    ///     Evolution histories for each scale
    fn sgc_phi_evolution(&self, scales: Vec<f64>, steps_per_scale: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
        use crate::resonant::phi_ops::ResonantTensorExt;
        ResonantTensorExt::sgc_phi_evolution(self, &scales, steps_per_scale)
    }

    /// Phi-convergence detection for SGC evolution.
    ///
    /// Uses phi-scaled thresholds to detect when evolution has stabilized.
    ///
    /// Args:
    ///     previous_values: Values from previous evolution step
    ///     tolerance: Convergence tolerance (default: 1e-8)
    ///
    /// Returns:
    ///     True if evolution has converged
    #[pyo3(signature = (previous_values, tolerance=1e-8))]
    fn sgc_phi_converged(&self, previous_values: Vec<f64>, tolerance: f64) -> bool {
        use crate::resonant::phi_ops::ResonantTensorExt;
        ResonantTensorExt::sgc_phi_converged(self, &previous_values, tolerance)
    }
}

impl ResonantTensor {
    /// Helper to resolve values to Vec<f64>, downloading from GPU if necessary.
    fn resolve_values_core(&self) -> PyResult<Vec<f64>> {
        if self.phase == ResonantPhase::Flux {
            if let Some(device) = &self.device {
                // Check f32 flux
                if let Some(flux_f32) = &self.flux_f32 {
                    let mut host_data = vec![0.0f32; self.len()];
                    device
                        .default_stream()
                        .memcpy_dtoh(flux_f32, &mut host_data)
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?;
                    return Ok(host_data.into_iter().map(|v| v as f64).collect());
                }

                // Check f64 flux
                if let Some(flux) = &self.flux {
                    let mut host_data = vec![0.0f64; self.len()];
                    device
                        .default_stream()
                        .memcpy_dtoh(flux, &mut host_data)
                        .map_err(|e| ResonantError::CudaError(e.to_string()))?;
                    return Ok(host_data);
                }
            }
        }

        // Fallback to CPU methods
        Ok(self.to_floats_rust())
    }
}

impl Clone for ResonantTensor {
    fn clone(&self) -> Self {
        ResonantTensor {
            lattice: self.lattice.clone(),
            flux: None, // Don't clone GPU memory
            flux_f32: None,
            cpu_flux: None,
            shape: self.shape.clone(),
            syntony: self.syntony,
            mode_norm_sq: self.mode_norm_sq.clone(),
            phase: ResonantPhase::Crystallized, // Clone always starts crystallized
            device: None,
            device_idx: self.device_idx,
            last_d_duration_ns: 0,
            precision: self.precision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.phase(), ResonantPhase::Crystallized);
        assert!(tensor.syntony() >= 0.0 && tensor.syntony() <= 1.0);
    }

    #[test]
    fn test_to_floats() {
        let data = vec![1.0, PHI, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let recovered = tensor.to_floats_core();
        assert_eq!(recovered.len(), 4);

        // Values should be close to originals
        for (orig, recovered) in data.iter().zip(recovered.iter()) {
            assert!((orig - recovered).abs() < 0.1);
        }
    }

    #[test]
    fn test_cpu_cycle() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let initial_syntony = tensor.syntony();

        // Run a CPU cycle
        let new_syntony = tensor.cpu_cycle(0.1, 100).expect("Should complete cycle");

        assert!(new_syntony >= 0.0 && new_syntony <= 1.0);
        assert_eq!(tensor.phase(), ResonantPhase::Crystallized);
    }

    #[test]
    fn test_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let cloned = tensor.clone();

        assert_eq!(cloned.len(), tensor.len());
        assert_eq!(cloned.syntony(), tensor.syntony());
        assert_eq!(cloned.phase(), ResonantPhase::Crystallized);
    }

    #[test]
    fn test_concat_semantics() {
        let a = ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], 100)
            .expect("create a");
        let b = ResonantTensor::from_floats_default_modes(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], 100)
            .expect("create b");

        let c = ResonantTensor::concat_core(&[&a, &b], 1).expect("concat");
        assert_eq!(c.shape(), &[2, 4]);

        let floats = c.to_floats_core();
        let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        for (x, y) in floats.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mean_var_axis() {
        let tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 5.0], vec![2, 2], 200)
                .expect("create tensor");

        let mean = tensor.mean_core(Some(1), false, 200).expect("mean");
        assert_eq!(mean.shape(), &[2]);
        let m = mean.to_floats_core();
        assert!((m[0] - 1.5).abs() < 1e-6);
        assert!((m[1] - 4.0).abs() < 1e-6);

        let var = tensor.var_core(Some(1), false, 200).expect("var");
        assert_eq!(var.shape(), &[2]);
        let v = var.to_floats_core();
        assert!((v[0] - 0.25).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_activation() {
        let mut tensor = ResonantTensor::from_floats_default_modes(&[-1.0, 0.0, 1.0], vec![3], 150)
            .expect("create tensor");
        tensor.gelu_core(200);
        let vals = tensor.to_floats_core();
        assert!(vals[0] < 0.0);
        assert!(vals[1].abs() < 1e-6);
        assert!(vals[2] > 0.0);
    }

    #[test]
    fn test_layer_norm_golden_target() {
        let tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], 300)
                .expect("create tensor");
        let out = tensor
            .layer_norm_core(None, None, 1e-8, true)
            .expect("layer norm");
        let vals = out.to_floats_core();

        // Per-sample mean ≈ 0 and variance ≈ 1/φ
        for sample in 0..2 {
            let start = sample * 2;
            let sample_vals = &vals[start..start + 2];
            let mean: f64 = sample_vals.iter().sum::<f64>() / 2.0;
            let var: f64 = sample_vals
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f64>()
                / 2.0;

            assert!(mean.abs() < 1e-6);
            assert!((var - PHI_INV).abs() < 1e-3);
        }
    }
}
