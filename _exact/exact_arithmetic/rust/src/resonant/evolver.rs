//! Resonant Evolution Strategy (RES) for discrete lattice-based optimization.
//!
//! The RES is a population-based evolution algorithm that exploits the insight that
//! lattice syntony (geometric beauty) correlates with task fitness. Key principle:
//! 80% of candidates die cheaply on CPU (lattice syntony check); only geometrically
//! promising candidates get expensive GPU evaluation.
//!
//! # Algorithm Overview
//!
//! 4-Step Generation Loop:
//! 1. **Spawn mutants** (CPU): Generate population_size mutants via lattice perturbation
//! 2. **Filter by lattice syntony** (CPU): Keep top survivor_count (~25%)
//! 3. **GPU evaluation** (GPU→CPU): D-phase + crystallize survivors only
//! 4. **Select winner** (CPU): score = fitness + λ × flux_syntony
//!
//! # Example
//!
//! ```ignore
//! use syntonic::resonant::{ResonantEvolver, RESConfig, ResonantTensor};
//!
//! let config = RESConfig::default();
//! let template = ResonantTensor::from_floats(&[1.0, 2.0, 3.0, 4.0], vec![4], 100)?;
//! let mut evolver = ResonantEvolver::from_template(&template, config);
//!
//! // Run evolution
//! for _ in 0..100 {
//!     let best_syntony = evolver.step()?;
//!     println!("Best syntony: {:.4}", best_syntony);
//! }
//!
//! let result = evolver.result();
//! ```

use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;

use super::attractor::AttractorMemory;
use super::retrocausal::harmonize_with_attractor_pull;
use super::tensor::{ResonantError, ResonantTensor};
use super::PHI;
use crate::exact::GoldenExact;

use crate::tensor::srt_kernels::{cuda_resonant_box_muller_f64, cuda_resonant_d_phase_batch_f64};
use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

/// Universal syntony deficit q - NOT a hyperparameter!
/// This is a fundamental constant from SRT.
pub use crate::constants::Q_DEFICIT_CONST as Q_DEFICIT;

/// Configuration for the Resonant Evolution Strategy.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RESConfig {
    /// Population size (number of mutants per generation)
    #[pyo3(get, set)]
    pub population_size: usize,

    /// Number of survivors after lattice syntony filtering (~25% of population)
    #[pyo3(get, set)]
    pub survivor_count: usize,

    /// Syntony weight in scoring: score = fitness + lambda * syntony
    /// Default: Q_DEFICIT = 0.027395146920 (NOT a hyperparameter!)
    #[pyo3(get, set)]
    pub lambda_val: f64,

    /// Scale of mutations in Q(φ) space
    #[pyo3(get, set)]
    pub mutation_scale: f64,

    /// Precision for crystallization (max coefficient)
    #[pyo3(get, set)]
    pub precision: i64,

    /// Noise scale for D-phase
    #[pyo3(get, set)]
    pub noise_scale: f64,

    /// Maximum generations before stopping
    #[pyo3(get, set)]
    pub max_generations: usize,

    /// Convergence threshold (stop if improvement < threshold)
    #[pyo3(get, set)]
    pub convergence_threshold: f64,

    // === Retrocausal RES Parameters ===
    /// Enable retrocausal attractor-guided harmonization
    #[pyo3(get, set)]
    pub enable_retrocausal: bool,

    /// Maximum number of attractors to store
    #[pyo3(get, set)]
    pub attractor_capacity: usize,

    /// Retrocausal pull strength (0.0 = disabled, 1.0 = full pull)
    #[pyo3(get, set)]
    pub attractor_pull_strength: f64,

    /// Minimum syntony threshold for attractor storage
    #[pyo3(get, set)]
    pub attractor_min_syntony: f64,

    /// Temporal decay rate for attractors (per generation)
    #[pyo3(get, set)]
    pub attractor_decay_rate: f64,

    /// CUDA device indices for multi-GPU acceleration (empty to disable)
    #[pyo3(get, set)]
    pub cuda_device_indices: Vec<i32>,
}

impl Default for RESConfig {
    fn default() -> Self {
        RESConfig {
            population_size: 64,
            survivor_count: 16,
            lambda_val: Q_DEFICIT,
            mutation_scale: 0.1,
            precision: 100,
            noise_scale: 0.01,
            max_generations: 1000,
            convergence_threshold: 1e-6,
            // Retrocausal defaults (opt-in)
            enable_retrocausal: false,
            attractor_capacity: 32,
            attractor_pull_strength: 0.3,
            attractor_min_syntony: 0.7,
            attractor_decay_rate: 0.98,
            cuda_device_indices: vec![], // empty = disabled
        }
    }
}

#[pymethods]
impl RESConfig {
    /// Create a new RESConfig with default values.
    #[new]
    #[pyo3(signature = (
        population_size=64,
        survivor_count=16,
        lambda_val=None,
        mutation_scale=0.1,
        precision=100,
        noise_scale=0.01,
        max_generations=1000,
        convergence_threshold=1e-6,
        enable_retrocausal=false,
        attractor_capacity=32,
        attractor_pull_strength=0.3,
        attractor_min_syntony=0.7,
        attractor_decay_rate=0.98,
        cuda_device_idx=-1
    ))]
    fn py_new(
        population_size: usize,
        survivor_count: usize,
        lambda_val: Option<f64>,
        mutation_scale: f64,
        precision: i64,
        noise_scale: f64,
        max_generations: usize,
        convergence_threshold: f64,
        enable_retrocausal: bool,
        attractor_capacity: usize,
        attractor_pull_strength: f64,
        attractor_min_syntony: f64,
        attractor_decay_rate: f64,
        cuda_device_idx: i32,
    ) -> Self {
        let cuda_device_indices = if cuda_device_idx >= 0 {
            vec![cuda_device_idx]
        } else {
            vec![]
        };

        RESConfig {
            population_size,
            survivor_count,
            lambda_val: lambda_val.unwrap_or(Q_DEFICIT),
            mutation_scale,
            precision,
            noise_scale,
            max_generations,
            convergence_threshold,
            enable_retrocausal,
            attractor_capacity,
            attractor_pull_strength,
            attractor_min_syntony,
            attractor_decay_rate,
            cuda_device_indices,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RESConfig(pop={}, survivors={}, λ={:.6}, mut_scale={}, prec={}, cuda_devices={:?})",
            self.population_size,
            self.survivor_count,
            self.lambda_val,
            self.mutation_scale,
            self.precision,
            self.cuda_device_indices
        )
    }
}

/// Result of a RES evolution run.
#[pyclass]
#[derive(Clone)]
pub struct RESResult {
    /// The winning tensor with highest syntony
    #[pyo3(get)]
    pub winner: ResonantTensor,

    /// Final syntony of the winner
    #[pyo3(get)]
    pub final_syntony: f64,

    /// Number of generations run
    #[pyo3(get)]
    pub generations: usize,

    /// Whether the evolution converged
    #[pyo3(get)]
    pub converged: bool,

    /// History of best syntony per generation
    #[pyo3(get)]
    pub syntony_history: Vec<f64>,
}

#[pymethods]
impl RESResult {
    fn __repr__(&self) -> String {
        format!(
            "RESResult(syntony={:.4}, generations={}, converged={})",
            self.final_syntony, self.generations, self.converged
        )
    }
}

/// Resonant Evolution Strategy (RES) evolver.
///
/// A population-based evolution algorithm that uses lattice syntony
/// as a cheap proxy for fitness, filtering 75-80% of candidates on CPU
/// before expensive GPU evaluation.
#[pyclass]
pub struct ResonantEvolver {
    /// Configuration
    config: RESConfig,

    /// Current best tensor (parent for next generation)
    best_tensor: Option<ResonantTensor>,

    /// Best syntony achieved so far
    best_syntony: f64,

    /// Current generation number
    generation: usize,

    /// History of best syntony per generation
    syntony_history: Vec<f64>,

    /// Template tensor (for shape and mode norms)
    template: Option<ResonantTensor>,

    /// Convergence window for checking stagnation
    convergence_window: Vec<f64>,

    /// Attractor memory for retrocausal influence
    attractor_memory: AttractorMemory,
}

impl ResonantEvolver {
    /// Create a new evolver with configuration.
    pub fn new(config: RESConfig) -> Self {
        let attractor_memory = AttractorMemory::new(
            config.attractor_capacity,
            config.attractor_min_syntony,
            config.attractor_decay_rate,
        );

        ResonantEvolver {
            config,
            best_tensor: None,
            best_syntony: 0.0,
            generation: 0,
            syntony_history: Vec::new(),
            template: None,
            convergence_window: Vec::new(),
            attractor_memory,
        }
    }

    /// Create an evolver from a template tensor.
    pub fn from_template(template: &ResonantTensor, config: RESConfig) -> Self {
        let syntony = template.syntony();
        let attractor_memory = AttractorMemory::new(
            config.attractor_capacity,
            config.attractor_min_syntony,
            config.attractor_decay_rate,
        );

        ResonantEvolver {
            config,
            best_tensor: Some(template.clone()),
            best_syntony: syntony,
            generation: 0,
            syntony_history: vec![syntony],
            template: Some(template.clone()),
            convergence_window: vec![syntony],
            attractor_memory,
        }
    }

    /// Spawn mutants from a parent tensor.
    ///
    /// Each mutant is created by adding random perturbations in Q(φ) space
    /// to each lattice element.
    pub fn spawn_mutants(&self, parent: &ResonantTensor) -> Vec<ResonantTensor> {
        let mut rng = rand::thread_rng();
        let mut mutants = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let mutant = self.mutate_lattice(parent, &mut rng);
            mutants.push(mutant);
        }

        mutants
    }

    /// Mutate a single tensor's lattice.
    fn mutate_lattice(&self, parent: &ResonantTensor, rng: &mut impl Rng) -> ResonantTensor {
        let lattice = parent.lattice();
        let mode_norm_sq = parent.mode_norm_sq();
        let shape = parent.shape().to_vec();

        let mutated: Vec<GoldenExact> = lattice
            .iter()
            .map(|g| {
                // Random perturbation in Q(φ) space
                // delta = delta_a + delta_b * φ
                let delta_a_f64 = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;
                let delta_b_f64 = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;

                // Convert to exact GoldenExact perturbation directly
                let perturbation = GoldenExact::find_nearest(delta_a_f64 + delta_b_f64 * PHI, self.config.precision);

                // Add perturbation to original
                *g + perturbation
            })
            .collect();

        // Create new tensor from mutated lattice
        ResonantTensor::from_lattice(mutated, shape, mode_norm_sq.to_vec())
            .unwrap_or_else(|_| parent.clone())
    }

    /// Filter candidates by lattice syntony (CPU, cheap).
    ///
    /// Returns the top `survivor_count` candidates sorted by descending syntony.
    pub fn filter_by_lattice_syntony(
        &self,
        candidates: Vec<ResonantTensor>,
    ) -> Vec<ResonantTensor> {
        let mut scored: Vec<(ResonantTensor, f64)> = candidates
            .into_iter()
            .map(|t| {
                let syntony = t.syntony();
                (t, syntony)
            })
            .collect();

        // Sort by syntony descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Take top survivors
        scored
            .into_iter()
            .take(self.config.survivor_count)
            .map(|(t, _)| t)
            .collect()
    }

    /// Evaluate survivors using CPU D-phase cycle.
    ///
    /// For each survivor:
    /// 1. Wake flux with noise (simulated D-phase)
    /// 2. Crystallize back to lattice
    /// 3. Compute final syntony
    ///
    /// Returns survivors with their scores (syntony after D→H cycle).
    pub fn evaluate_survivors_cpu(
        &self,
        survivors: Vec<ResonantTensor>,
    ) -> Vec<(ResonantTensor, f64)> {
        survivors
            .into_iter()
            .map(|mut tensor| {
                // CPU D→H cycle
                let syntony = tensor
                    .run_cpu_cycle(self.config.noise_scale, self.config.precision)
                    .unwrap_or(tensor.syntony());
                (tensor, syntony)
            })
            .collect()
    }

    /// Evaluate survivors using CUDA batch D-phase cycle.
    ///
    /// Processes all survivors in a single batch using cuda_resonant_d_phase_batch_f64:
    /// 1. Batch wake flux with noise (GPU D-phase)
    /// 2. Batch crystallize back to lattice (CPU)
    /// 3. Compute final syntony for each survivor
    ///
    /// Returns survivors with their scores (syntony after D→H cycle).
    pub fn evaluate_survivors_cuda(
        &self,
        survivors: &[ResonantTensor],
        device: Arc<CudaDevice>,
    ) -> Result<Vec<(ResonantTensor, f64)>, ResonantError> {
        if survivors.is_empty() {
            return Ok(Vec::new());
        }

        let pop_size = survivors.len();
        let n = survivors[0].len();

        // Verify all survivors have the same shape
        for (i, survivor) in survivors.iter().enumerate() {
            if survivor.len() != n {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Survivor {} has length {}, expected {}",
                    i,
                    survivor.len(),
                    n
                )));
            }
        }

        // Prepare batch data
        let mut lattice_batch = Vec::with_capacity(n * pop_size);
        let mut mode_norm_sq_batch = Vec::with_capacity(n * pop_size);

        // Collect lattice and mode norms from all survivors
        for survivor in survivors {
            let lattice_floats: Vec<f64> = survivor.lattice().iter().map(|g| g.to_f64()).collect();
            lattice_batch.extend(lattice_floats);
            mode_norm_sq_batch.extend(survivor.mode_norm_sq());
        }

        // Generate Gaussian noise for all survivors using Box-Muller transform
        let noise_batch: Vec<f64> = self
            .generate_gaussian_noise(n * pop_size)
            .into_iter()
            .map(|x| x * self.config.noise_scale)
            .collect();

        // Upload to GPU
        let gpu_lattice_batch = device
            .default_stream()
            .clone_htod(&lattice_batch)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_mode_norm_sq = device
            .default_stream()
            .clone_htod(&mode_norm_sq_batch)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_noise_batch = device
            .default_stream()
            .clone_htod(&noise_batch)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Allocate output buffers
        let mut gpu_flux_batch: CudaSlice<f64> = device
            .default_stream()
            .alloc_zeros(n * pop_size)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_syntonies: CudaSlice<f64> = device
            .default_stream()
            .alloc_zeros(pop_size)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Run batch D-phase kernel
        cuda_resonant_d_phase_batch_f64(
            &device,
            &mut gpu_flux_batch,
            &gpu_lattice_batch,
            &gpu_mode_norm_sq,
            &gpu_noise_batch,
            &gpu_syntonies,
            self.config.noise_scale,
            n,
            pop_size,
        )
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Download results
        let mut host_flux_batch = vec![0.0f64; n * pop_size];
        let mut host_syntonies = vec![0.0f64; pop_size];

        device
            .default_stream()
            .memcpy_dtoh(&gpu_flux_batch, &mut host_flux_batch)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        device
            .default_stream()
            .memcpy_dtoh(&gpu_syntonies, &mut host_syntonies)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Process each survivor: crystallize and update syntony
        let mut results = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let start = i * n;
            let end = start + n;
            let flux_slice = &host_flux_batch[start..end];

            // Create a copy of the survivor and crystallize it
            let mut survivor = survivors[i].clone();
            survivor
                .crystallize_cpu(flux_slice, self.config.precision)
                .map_err(|e| ResonantError::CudaError(format!("Crystallization failed: {}", e)))?;

            let final_syntony = survivor.syntony();
            results.push((survivor, final_syntony));
        }

        Ok(results)
    }

    /// Evaluate survivors using multiple CUDA devices for distributed computation.
    pub fn evaluate_survivors_multi_cuda(
        &self,
        survivors: &[ResonantTensor],
    ) -> Result<Vec<(ResonantTensor, f64)>, ResonantError> {
        if survivors.is_empty() {
            return Ok(Vec::new());
        }

        let num_devices = self.config.cuda_device_indices.len();
        if num_devices == 1 {
            // Single device case
            let device_idx = self.config.cuda_device_indices[0] as usize;
            let device =
                crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
                    ResonantError::CudaError(format!("Failed to get device {}: {}", device_idx, e))
                })?;
            return self.evaluate_survivors_cuda(&survivors, device);
        }

        // Multi-device case: distribute survivors across devices
        let survivors_per_device = (survivors.len() + num_devices - 1) / num_devices; // Ceiling division
        let mut results = Vec::with_capacity(survivors.len());

        for (i, device_idx) in self.config.cuda_device_indices.iter().enumerate() {
            let start = i * survivors_per_device;
            let end = (start + survivors_per_device).min(survivors.len());
            if start >= end {
                break;
            }

            let chunk = &survivors[start..end];
            let device = crate::tensor::cuda::device_manager::get_device(*device_idx as usize)
                .map_err(|e| {
                    ResonantError::CudaError(format!("Failed to get device {}: {}", device_idx, e))
                })?;

            let mut chunk_results = self.evaluate_survivors_cuda(chunk, device)?;
            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    /// Generate Gaussian noise using Box-Muller transform.
    ///
    /// Uses CUDA acceleration if available, otherwise falls back to CPU.
    /// The Box-Muller transform converts uniform random numbers to Gaussian.
    ///
    /// # Arguments
    /// * `count` - Number of Gaussian samples to generate
    /// * `device` - Optional CUDA device for acceleration
    ///
    /// # Returns
    /// Vector of Gaussian random numbers with mean 0, variance 1
    pub fn generate_gaussian_noise_cuda(
        &self,
        count: usize,
        device: Option<&Arc<CudaDevice>>,
    ) -> Result<Vec<f64>, ResonantError> {
        if count == 0 {
            return Ok(Vec::new());
        }

        // Generate uniform random numbers (2 per Gaussian sample for Box-Muller)
        let mut uniform_noise = Vec::with_capacity(2 * count);
        let mut rng = rand::thread_rng();
        for _ in 0..(2 * count) {
            uniform_noise.push(rng.gen::<f64>()); // Uniform [0, 1)
        }

        if let Some(device) = device {
            // Upload uniform noise to GPU
            let gpu_uniform = device
                .default_stream()
                .clone_htod(&uniform_noise)
                .map_err(|e| {
                    ResonantError::CudaError(format!("Failed to upload uniform noise: {}", e))
                })?;

            // Allocate output buffer for Gaussian noise
            let mut gpu_gaussian: CudaSlice<f64> =
                device.default_stream().alloc_zeros(count).map_err(|e| {
                    ResonantError::CudaError(format!("Failed to allocate Gaussian buffer: {}", e))
                })?;

            // Run Box-Muller transform on GPU
            cuda_resonant_box_muller_f64(device, &mut gpu_gaussian, &gpu_uniform, count)
                .map_err(|e| ResonantError::CudaError(format!("CUDA Box-Muller failed: {}", e)))?;

            // Download Gaussian noise
            let mut gaussian_noise = vec![0.0f64; count];
            device
                .default_stream()
                .memcpy_dtoh(&gpu_gaussian, &mut gaussian_noise)
                .map_err(|e| {
                    ResonantError::CudaError(format!("Failed to download Gaussian noise: {}", e))
                })?;

            Ok(gaussian_noise)
        } else {
            // CPU fallback: simple Box-Muller implementation
            let mut gaussian_noise = Vec::with_capacity(count);
            let mut rng = rand::thread_rng();

            // Ensure we have enough uniform noise for Box-Muller (needs 2 per Gaussian)
            let mut local_uniform = uniform_noise.clone();
            if local_uniform.len() < 2 * count {
                // Generate additional uniform noise if needed
                use rand::Rng;
                while local_uniform.len() < 2 * count {
                    local_uniform.push(rng.gen::<f64>());
                }
            }

            for i in (0..(2 * count)).step_by(2) {
                // Box-Muller transform: convert two uniform [0,1) to one Gaussian
                let u1 = local_uniform[i];
                let u2 = local_uniform[i + 1];
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * crate::constants::SRT_PI * u2;
                let z0 = r * theta.cos();
                gaussian_noise.push(z0);
            }

            Ok(gaussian_noise)
        }
    }

    /// Generate Gaussian noise with automatic CUDA dispatch.
    ///
    /// Uses CUDA acceleration if available, otherwise falls back to CPU.
    ///
    /// # Arguments
    /// * `count` - Number of Gaussian samples to generate
    ///
    /// # Returns
    /// Vector of Gaussian random numbers with mean 0, variance 1
    pub fn generate_gaussian_noise(&self, count: usize) -> Vec<f64> {
        {
            if !self.config.cuda_device_indices.is_empty() {
                if let Ok(device) = crate::tensor::cuda::device_manager::get_device(
                    self.config.cuda_device_indices[0] as usize,
                ) {
                    match self.generate_gaussian_noise_cuda(count, Some(&device)) {
                        Ok(noise) => return noise,
                        Err(_) => {} // Fall back to CPU
                    }
                }
            }
        }

        // CPU fallback
        let mut gaussian_noise = Vec::with_capacity(count);
        let mut rng = rand::thread_rng();

        for _ in 0..count {
            // Box-Muller transform
            let u1 = rng.gen::<f64>();
            let u2 = rng.gen::<f64>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * crate::constants::SRT_PI * u2;
            let z0 = r * theta.cos();
            gaussian_noise.push(z0);
        }

        gaussian_noise
    }

    /// Apply retrocausal harmonization to survivors.
    ///
    /// Uses attractor memory to bias harmonization toward proven high-syntony states.
    fn apply_retrocausal_harmonization(
        &self,
        survivors: Vec<ResonantTensor>,
    ) -> Result<Vec<ResonantTensor>, ResonantError> {
        survivors
            .into_iter()
            .map(|mut t| {
                harmonize_with_attractor_pull(
                    &mut t,
                    &self.attractor_memory,
                    self.config.attractor_pull_strength,
                )?;
                Ok(t)
            })
            .collect()
    }

    /// Select the winner from evaluated candidates.
    ///
    /// Winner is selected by: score = syntony (since we don't have external fitness yet)
    pub fn select_winner(&self, evaluated: Vec<(ResonantTensor, f64)>) -> Option<ResonantTensor> {
        evaluated
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(t, _)| t)
    }

    /// Select the winner from evaluated candidates, returning both tensor and score.
    ///
    /// Winner is selected by: score = syntony (since we don't have external fitness yet)
    fn select_winner_with_score(
        &self,
        evaluated: &[(ResonantTensor, f64)],
    ) -> Option<(ResonantTensor, f64)> {
        evaluated
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(t, s)| (t.clone(), *s))
    }

    /// Run one generation of evolution.
    ///
    /// Returns the best syntony achieved this generation.
    pub fn step(&mut self) -> Result<f64, ResonantError> {
        // Get parent (or create random if none)
        let parent = match &self.best_tensor {
            Some(t) => t.clone(),
            None => {
                return Err(ResonantError::InvalidPhaseTransition(
                    "No parent tensor set. Use from_template() to initialize.".to_string(),
                ))
            }
        };

        // Step 1: Spawn mutants (CPU)
        let mutants = self.spawn_mutants(&parent);

        // Step 2: Filter by lattice syntony (CPU, cheap)
        let mut survivors = self.filter_by_lattice_syntony(mutants);

        // Step 3: RETROCAUSAL HARMONIZATION (if enabled)
        if self.config.enable_retrocausal && !self.attractor_memory.is_empty() {
            survivors = self.apply_retrocausal_harmonization(survivors)?;
        }

        // Step 4: Evaluate survivors (CUDA if available, else CPU)
        let evaluated = if !self.config.cuda_device_indices.is_empty() {
            {
                self.evaluate_survivors_multi_cuda(&survivors)
                    .unwrap_or_else(|_| self.evaluate_survivors_cpu(survivors.clone()))
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.evaluate_survivors_cpu(survivors)
            }
        } else {
            self.evaluate_survivors_cpu(survivors)
        };

        // Step 5: Select winner and update attractors
        if let Some((winner, score)) = self.select_winner_with_score(&evaluated) {
            let winner_syntony = score; // Use the returned score directly

            // Store high-syntony candidates as attractors
            if self.config.enable_retrocausal {
                for (tensor, _) in &evaluated {
                    let syntony = tensor.syntony();
                    self.attractor_memory
                        .maybe_add(tensor, syntony, self.generation);
                }
            }

            // Update best if improved
            if winner_syntony > self.best_syntony {
                self.best_syntony = winner_syntony;
                self.best_tensor = Some(winner);
            }
        }

        // Apply temporal decay to attractors
        if self.config.enable_retrocausal {
            self.attractor_memory.apply_decay();
        }

        // Update tracking
        self.generation += 1;
        self.syntony_history.push(self.best_syntony);
        self.update_convergence_window(self.best_syntony);

        Ok(self.best_syntony)
    }

    /// Update convergence tracking window.
    fn update_convergence_window(&mut self, syntony: f64) {
        self.convergence_window.push(syntony);
        if self.convergence_window.len() > 10 {
            self.convergence_window.remove(0);
        }
    }

    /// Check if evolution has converged.
    pub fn is_converged(&self) -> bool {
        if self.convergence_window.len() < 10 {
            return false;
        }

        let min = self
            .convergence_window
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .convergence_window
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        (max - min) < self.config.convergence_threshold
    }

    /// Run evolution until convergence or max generations.
    pub fn run(&mut self) -> Result<RESResult, ResonantError> {
        while self.generation < self.config.max_generations && !self.is_converged() {
            self.step()?;
        }

        let winner = self.best_tensor.clone().ok_or_else(|| {
            ResonantError::InvalidPhaseTransition("No winner found after evolution".to_string())
        })?;

        Ok(RESResult {
            winner,
            final_syntony: self.best_syntony,
            generations: self.generation,
            converged: self.is_converged(),
            syntony_history: self.syntony_history.clone(),
        })
    }

    /// Get the current best tensor.
    pub fn best(&self) -> Option<&ResonantTensor> {
        self.best_tensor.as_ref()
    }

    /// Get the current generation number.
    pub fn current_generation(&self) -> usize {
        self.generation
    }

    /// Get the best syntony achieved.
    pub fn best_syntony(&self) -> f64 {
        self.best_syntony
    }
}

// =========================================================================
// PyO3 Methods
// =========================================================================

#[pymethods]
impl ResonantEvolver {
    /// Create a new evolver from a template tensor.
    #[new]
    #[pyo3(signature = (template, config=None))]
    fn py_new(template: &ResonantTensor, config: Option<RESConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        ResonantEvolver::from_template(template, cfg)
    }

    /// Run one generation. Returns best syntony.
    #[pyo3(name = "step")]
    fn py_step(&mut self) -> PyResult<f64> {
        self.step().map_err(|e| e.into())
    }

    /// Run evolution until convergence. Returns RESResult.
    #[pyo3(name = "run")]
    fn py_run(&mut self) -> PyResult<RESResult> {
        self.run().map_err(|e| e.into())
    }

    /// Get the current best tensor.
    #[getter]
    fn get_best(&self) -> Option<ResonantTensor> {
        self.best_tensor.clone()
    }

    /// Get the template tensor used for initialization.
    #[getter]
    fn get_template(&self) -> Option<ResonantTensor> {
        self.template.clone()
    }

    /// Get the current generation number.
    #[getter]
    fn get_generation(&self) -> usize {
        self.generation
    }

    /// Get the best syntony achieved.
    #[getter]
    fn get_best_syntony(&self) -> f64 {
        self.best_syntony
    }

    /// Check if evolution has converged.
    #[getter]
    fn get_is_converged(&self) -> bool {
        self.is_converged()
    }

    /// Get the syntony history.
    #[getter]
    fn get_syntony_history(&self) -> Vec<f64> {
        self.syntony_history.clone()
    }

    /// Get the configuration.
    #[getter]
    fn get_config(&self) -> RESConfig {
        self.config.clone()
    }

    /// Get the top-k attractors currently stored in memory.
    fn get_top_attractors(&self, k: usize) -> Vec<ResonantTensor> {
        self.attractor_memory
            .get_top_attractors(k)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get the number of attractors currently stored.
    #[getter]
    fn attractor_count(&self) -> usize {
        self.attractor_memory.len()
    }

    /// Retrieve syntony values for all attractors.
    #[getter]
    fn get_attractor_syntony_values(&self) -> Vec<f64> {
        self.attractor_memory.get_syntony_values().to_vec()
    }

    /// Retrieve generations when each attractor was recorded.
    #[getter]
    fn get_attractor_generations(&self) -> Vec<usize> {
        self.attractor_memory.get_generations().to_vec()
    }

    /// Clear all attractors from memory.
    fn clear_attractors(&mut self) {
        self.attractor_memory.clear();
    }

    /// Store a tensor as an attractor if syntony exceeds threshold.
    ///
    /// Returns true if the tensor was stored (syntony >= min_syntony threshold).
    fn store_attractor(&mut self, tensor: &ResonantTensor) -> bool {
        let syntony = tensor.syntony();
        self.attractor_memory
            .maybe_add(tensor, syntony, self.generation);
        syntony >= self.config.attractor_min_syntony
    }

    /// Apply temporal decay to all stored attractors.
    ///
    /// Called each generation to fade older attractors.
    fn apply_decay(&mut self) {
        self.attractor_memory.apply_decay();
    }

    /// Harmonize tensor with retrocausal attractor pull.
    ///
    /// Applies standard harmonization blended with attractor-guided influence.
    fn harmonize(&self, tensor: &ResonantTensor) -> PyResult<ResonantTensor> {
        use super::retrocausal::harmonize_with_attractor_pull;
        let mut result = tensor.clone();
        harmonize_with_attractor_pull(
            &mut result,
            &self.attractor_memory,
            self.config.attractor_pull_strength,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    /// Pull tensor toward stored attractors.
    ///
    /// Computes weighted pull vector from all attractors and applies to tensor.
    fn pull(&self, tensor: &ResonantTensor) -> PyResult<ResonantTensor> {
        let pull_vec = self
            .attractor_memory
            .compute_attractor_pull(tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut result = tensor.clone();
        result
            .set_lattice(&pull_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    /// Unlock attractors for maximum influence (consciousness emergence).
    ///
    /// Sets pull_strength to 1.0 for full attractor influence.
    fn unlock(&mut self) {
        self.config.attractor_pull_strength = 1.0;
    }

    fn __repr__(&self) -> String {
        format!(
            "ResonantEvolver(generation={}, best_syntony={:.4}, converged={})",
            self.generation,
            self.best_syntony,
            self.is_converged()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RESConfig::default();
        assert_eq!(config.population_size, 64);
        assert_eq!(config.survivor_count, 16);
        assert!((config.lambda_val - Q_DEFICIT).abs() < 1e-10);
    }

    #[test]
    fn test_evolver_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, RESConfig::default());
        assert_eq!(evolver.current_generation(), 0);
        assert!(evolver.best().is_some());
    }

    #[test]
    fn test_spawn_mutants() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, RESConfig::default());
        let mutants = evolver.spawn_mutants(&tensor);

        assert_eq!(mutants.len(), 64);
    }

    #[test]
    fn test_filter_by_syntony() {
        let config = RESConfig {
            population_size: 10,
            survivor_count: 3,
            ..Default::default()
        };

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, config);
        let mutants = evolver.spawn_mutants(&tensor);
        let survivors = evolver.filter_by_lattice_syntony(mutants);

        assert_eq!(survivors.len(), 3);
    }

    #[test]
    fn test_single_step() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let config = RESConfig {
            population_size: 10,
            survivor_count: 3,
            ..Default::default()
        };

        let mut evolver = ResonantEvolver::from_template(&tensor, config);
        let syntony = evolver.step().unwrap();

        assert!(syntony >= 0.0 && syntony <= 1.0);
        assert_eq!(evolver.current_generation(), 1);
    }
}
