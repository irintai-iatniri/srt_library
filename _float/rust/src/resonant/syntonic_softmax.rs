//! Syntony-weighted softmax for theory-aligned classification.
//!
//! Standard softmax: p_i = exp(x_i) / Σ exp(x_j)
//! Syntonic softmax: p_i = w_i * exp(x_i) / Σ w_j * exp(x_j)
//!
//! where w(n) = exp(-|n|²/φ) is the golden measure weight.

use crate::resonant::{
    crystallize::snap_to_lattice,
    e8_lattice::{compute_8d_weight, E8Lattice, GoldenProjector},
    ResonantPhase, ResonantTensor, PHI,
};
use pyo3::prelude::*;

/// Syntonic softmax modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum SyntonicSoftmaxMode {
    /// Learn mode norms (8D E8 roots)
    Learned,
    /// Accept pre-computed syntony values
    Provided,
    /// Standard softmax (ablation baseline)
    Identity,
}

#[pymethods]
impl SyntonicSoftmaxMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "learned" => Ok(SyntonicSoftmaxMode::Learned),
            "provided" => Ok(SyntonicSoftmaxMode::Provided),
            "identity" => Ok(SyntonicSoftmaxMode::Identity),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown mode: {}. Use 'learned', 'provided', or 'identity'",
                mode_str
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            SyntonicSoftmaxMode::Learned => "SyntonicSoftmaxMode::Learned".to_string(),
            SyntonicSoftmaxMode::Provided => "SyntonicSoftmaxMode::Provided".to_string(),
            SyntonicSoftmaxMode::Identity => "SyntonicSoftmaxMode::Identity".to_string(),
        }
    }
}

/// Syntonic softmax state (learnable mode norms)
#[pyclass]
#[derive(Clone, Debug)]
pub struct SyntonicSoftmaxState {
    /// Mode norms |n|² for each feature OR 8D E8 vectors [num_features, 8]
    pub mode_norms: Option<ResonantTensor>,

    /// Syntony weighting scale
    pub syntony_scale: f64,

    /// Softmax dimension (typically -1 for last dim)
    pub dim: isize,

    /// Mode
    pub mode: SyntonicSoftmaxMode,

    /// Number of features (for learned mode)
    pub num_features: Option<usize>,
}

#[pymethods]
impl SyntonicSoftmaxState {
    #[new]
    pub fn new(
        mode: SyntonicSoftmaxMode,
        dim: Option<isize>,
        num_features: Option<usize>,
        syntony_scale: Option<f64>,
    ) -> PyResult<Self> {
        // Validate
        if mode == SyntonicSoftmaxMode::Learned && num_features.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_features required for learned mode",
            ));
        }

        // Initialize mode norms if learned
        let mode_norms = if mode == SyntonicSoftmaxMode::Learned {
            let n = num_features.unwrap();

            // Generate E8 roots
            let roots = E8Lattice::generate_roots();
            let projector = GoldenProjector::new();

            // Filter to Golden Cone using 4 null vectors: ⟨c_a, α⟩ > 0 for all a
            // This selects exactly 36 roots = Φ⁺(E₆)
            let cone_roots: Vec<Vec<f64>> = roots
                .into_iter()
                .filter(|root| {
                    let weight = compute_8d_weight(root, &projector);
                    weight > 1e-8 // Reject 1e-9 (outside cone) but accept valid exp(-x/φ)
                })
                .collect();

            if cone_roots.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "No E8 roots found in Golden Cone",
                ));
            }

            // Cycle through cone roots to fill num_features
            let mut data = Vec::with_capacity(n * 8);
            for i in 0..n {
                let root = &cone_roots[i % cone_roots.len()];
                data.extend_from_slice(root);
            }

            // Create [n, 8] tensor
            // We use from_floats_default_modes which calculates trivial mode norms for data values
            Some(
                ResonantTensor::from_floats_default_modes(&data, vec![n, 8], 100)
                    .map_err(|e| PyErr::from(e))?,
            )
        } else {
            None
        };

        Ok(SyntonicSoftmaxState {
            mode_norms,
            syntony_scale: syntony_scale.unwrap_or(1.0),
            dim: dim.unwrap_or(-1),
            mode,
            num_features,
        })
    }

    /// Forward pass: apply syntony-weighted softmax
    pub fn forward(
        &self,
        x: &ResonantTensor,
        syntony: Option<&ResonantTensor>,
    ) -> PyResult<ResonantTensor> {
        // Numerical stability check: Check for NaN/Inf in input
        // Note: This forces a CPU sync if data is on GPU. For maximum performance in production
        // this might be optional, but for "Robustness" phase it is required.
        // We only check if not on GPU or if explicitly requested (omitted for now to avoid sync overhead on GPU path)
        if x.device_idx().is_none() {
            let floats = x.to_floats_rust();
            if floats.iter().any(|v| !v.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Input contains NaN or Inf values",
                ));
            }
        }

        match self.mode {
            SyntonicSoftmaxMode::Identity => {
                // Dispatch to GPU if available
                {
                    if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
                        return self.forward_cuda(x, None);
                    }
                }

                // CPU fallback: Standard softmax (no syntony weighting)
                let mut output = x.clone();
                output
                    .softmax_core(Some(self.dim), 1000)
                    .map_err(|e| PyErr::from(e))?;
                Ok(output)
            }
            SyntonicSoftmaxMode::Learned => {
                // w(n) = exp(-|n|²/φ)
                let mode_norms = self.mode_norms.as_ref().unwrap();
                let weights = self.compute_golden_weights(mode_norms)?;

                // weighted_logits = logits + scale * log(weights)
                let log_weights = weights.log_core(1000)?;
                let scaled_log_weights = log_weights
                    .scalar_mul_core(self.syntony_scale)
                    .map_err(|e| PyErr::from(e))?;

                // Dispatch to CUDA or CPU
                {
                    if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
                        return self.forward_cuda(x, Some(&scaled_log_weights));
                    }
                }

                let final_weights = if x.shape().len() > 1 && scaled_log_weights.shape().len() == 1
                {
                    let w_shape = scaled_log_weights.shape();
                    let x_shape = x.shape();
                    let ndim = x_shape.len();

                    // Calculate actual dimension index (handle negative dims)
                    let dim_idx = if self.dim < 0 {
                        (ndim as isize + self.dim) as usize
                    } else {
                        self.dim as usize
                    };

                    let softmax_dim_size = x_shape[dim_idx];
                    let weight_dim = w_shape[0];

                    if weight_dim != softmax_dim_size {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch in Syntonic Softmax: x shape {:?} dim {} (size {}) incompatible with weights shape {:?}",
                            x_shape, self.dim, softmax_dim_size, w_shape
                        )));
                    }

                    // Broadcast weights to match x shape along softmax dimension
                    // For shape [3, 4] with dim=0: replicate each weight 4 times (inner), no outer
                    // For shape [3, 4] with dim=-1: replicate pattern 3 times (outer)
                    let outer: usize = x_shape[..dim_idx].iter().product();
                    let inner: usize = if dim_idx + 1 < ndim {
                        x_shape[dim_idx + 1..].iter().product()
                    } else {
                        1
                    };

                    let w_lattice = scaled_log_weights.lattice();
                    let w_norms = scaled_log_weights.mode_norm_sq();
                    let total_elements: usize = x_shape.iter().product();

                    let mut new_lattice = Vec::with_capacity(total_elements);
                    let mut new_norms = Vec::with_capacity(total_elements);

                    // Broadcast: [outer, weight_dim, inner]
                    for _ in 0..outer {
                        for w_idx in 0..weight_dim {
                            for _ in 0..inner {
                                new_lattice.push(w_lattice[w_idx].clone());
                                new_norms.push(w_norms[w_idx]);
                            }
                        }
                    }

                    ResonantTensor::from_lattice(new_lattice, x_shape.to_vec(), new_norms)
                        .map_err(|e| PyErr::from(e))?
                } else {
                    scaled_log_weights
                };

                let weighted_logits = x
                    .elementwise_add_core(&final_weights)
                    .map_err(|e| PyErr::from(e))?;
                let mut output = weighted_logits.clone();
                output
                    .softmax_core(Some(self.dim), 1000)
                    .map_err(|e| PyErr::from(e))?;
                Ok(output)
            }
            SyntonicSoftmaxMode::Provided => {
                // Use provided syntony values
                let syntony = syntony.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "syntony values required for 'provided' mode",
                    )
                })?;

                // Validate shape match
                if syntony.shape() != x.shape() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Shape mismatch: x {:?} vs syntony {:?}",
                        x.shape(),
                        syntony.shape()
                    )));
                }

                // Dispatch to CUDA or CPU
                {
                    if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
                        return self.forward_cuda(x, Some(syntony));
                    }
                }

                // weighted_logits = logits + scale * log(syntony)
                let log_syntony = syntony.log_core(1000)?;
                let scaled_log_syntony = log_syntony
                    .scalar_mul_core(self.syntony_scale)
                    .map_err(|e| PyErr::from(e))?;
                let weighted_logits = x
                    .elementwise_add_core(&scaled_log_syntony)
                    .map_err(|e| PyErr::from(e))?;

                let mut output = weighted_logits.clone();
                output
                    .softmax_core(Some(self.dim), 1000)
                    .map_err(|e| PyErr::from(e))?;
                Ok(output)
            }
        }
    }

    /// CUDA-accelerated forward pass
    fn forward_cuda(
        &self,
        x: &ResonantTensor,
        weights: Option<&ResonantTensor>,
    ) -> PyResult<ResonantTensor> {
        use crate::tensor::srt_kernels::{
            cuda_softmax_identity_f32,
            // Identity kernels
            cuda_softmax_identity_f64,
            cuda_softmax_identity_strided_f32,
            cuda_softmax_identity_strided_f64,
            // F32 kernels
            cuda_syntonic_softmax_learned_f32,
            // F64 kernels
            cuda_syntonic_softmax_learned_f64,
            cuda_syntonic_softmax_learned_strided_f32,
            cuda_syntonic_softmax_learned_strided_f64,
            cuda_syntonic_softmax_provided_f32,
            cuda_syntonic_softmax_provided_f64,
            cuda_syntonic_softmax_provided_strided_f32,
            cuda_syntonic_softmax_provided_strided_f64,
        };

        let device_idx = x.device_idx().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Tensor must be on GPU for CUDA softmax")
        })?;

        let device = crate::tensor::cuda::device_manager::get_device(device_idx)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let shape = x.shape();
        let ndim = shape.len();

        // Check if using F32 mode (flux_f32 exists)
        let use_f32 = x.flux_f32_ref().is_some();

        // Handle negative dim
        let dim_idx = if self.dim < 0 {
            if (ndim as isize + self.dim) < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Dimension out of range: dim={} for ndim={}",
                    self.dim, ndim
                )));
            }
            (ndim as isize + self.dim) as usize
        } else {
            if self.dim as usize >= ndim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Dimension out of range: dim={} for ndim={}",
                    self.dim, ndim
                )));
            }
            self.dim as usize
        };

        // Determine dims
        let num_classes = shape[dim_idx]; // dim_size
        let is_last_dim = dim_idx == ndim - 1;

        let batch_size = if is_last_dim {
            x.len() / num_classes
        } else {
            0 // unused in strided
        };

        // Strided parameters calculation
        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let dim_size = num_classes;

        // Identity mode uses dedicated GPU kernels
        if self.mode == SyntonicSoftmaxMode::Identity {
            if use_f32 {
                // F32 identity mode
                let x_flux = x.flux_f32_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("F32 Flux missing for CUDA operation")
                })?;
                let mut out_flux = device
                    .default_stream()
                    .alloc_zeros::<f32>(x.len())
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to allocate GPU memory for output (f32 identity): {}",
                            e
                        ))
                    })?;

                if is_last_dim {
                    cuda_softmax_identity_f32(
                        &device,
                        &mut out_flux,
                        x_flux,
                        batch_size as i32,
                        num_classes as i32,
                    )
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                } else {
                    cuda_softmax_identity_strided_f32(
                        &device,
                        &mut out_flux,
                        x_flux,
                        outer_size as i32,
                        dim_size as i32,
                        inner_size as i32,
                    )
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                }

                let mut output = x.clone();
                output.set_flux_f32(out_flux);
                output.set_device(device.clone());
                output.set_device_idx(device_idx);
                return Ok(output);
            } else {
                // F64 identity mode
                let x_flux = x.flux_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Flux missing for CUDA operation")
                })?;
                let mut out_flux = device
                    .default_stream()
                    .alloc_zeros::<f64>(x.len())
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to allocate GPU memory for output (f64 identity): {}",
                            e
                        ))
                    })?;

                if is_last_dim {
                    cuda_softmax_identity_f64(
                        &device,
                        &mut out_flux,
                        x_flux,
                        batch_size as i32,
                        num_classes as i32,
                    )
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                } else {
                    cuda_softmax_identity_strided_f64(
                        &device,
                        &mut out_flux,
                        x_flux,
                        outer_size as i32,
                        dim_size as i32,
                        inner_size as i32,
                    )
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                }

                let mut output = x.clone();
                output.set_flux(out_flux);

                // Download results and create new tensor for immediate access
                let mut host_data = vec![0.0f64; output.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(output.flux_ref().unwrap(), &mut host_data)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to download GPU results: {}",
                            e
                        ))
                    })?;

                // Create new tensor from downloaded data
                let result_lattice = snap_to_lattice(&host_data, x.precision());
                let result_norms = x.mode_norm_sq().to_vec();
                return ResonantTensor::from_lattice(
                    result_lattice,
                    x.shape().to_vec(),
                    result_norms,
                )
                .map_err(|e| PyErr::from(e));
            }
        }

        // Learned and Provided modes with F32/F64 dispatch
        if use_f32 {
            // F32 path
            let x_flux = x.flux_f32_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("F32 Flux missing for CUDA operation")
            })?;
            let mut out_flux = device
                .default_stream()
                .alloc_zeros::<f32>(x.len())
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to allocate GPU memory for output (f32): {}",
                        e
                    ))
                })?;

            match self.mode {
                SyntonicSoftmaxMode::Learned => {
                    let mode_norms = self.mode_norms.as_ref().unwrap();

                    let norms_flux = mode_norms.flux_f32_ref().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "mode_norms must be on GPU (F32) for CUDA softmax",
                        )
                    })?;

                    if is_last_dim {
                        cuda_syntonic_softmax_learned_f32(
                            &device,
                            &mut out_flux,
                            x_flux,
                            norms_flux,
                            self.syntony_scale as f32,
                            batch_size as i32,
                            num_classes as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    } else {
                        cuda_syntonic_softmax_learned_strided_f32(
                            &device,
                            &mut out_flux,
                            x_flux,
                            norms_flux,
                            self.syntony_scale as f32,
                            outer_size as i32,
                            dim_size as i32,
                            inner_size as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    }
                }
                SyntonicSoftmaxMode::Provided => {
                    let w = weights.unwrap();
                    let w_flux = w.flux_f32_ref().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "syntony weights must be on GPU (F32) for CUDA softmax",
                        )
                    })?;

                    if is_last_dim {
                        cuda_syntonic_softmax_provided_f32(
                            &device,
                            &mut out_flux,
                            x_flux,
                            w_flux,
                            self.syntony_scale as f32,
                            batch_size as i32,
                            num_classes as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    } else {
                        cuda_syntonic_softmax_provided_strided_f32(
                            &device,
                            &mut out_flux,
                            x_flux,
                            w_flux,
                            self.syntony_scale as f32,
                            outer_size as i32,
                            dim_size as i32,
                            inner_size as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    }
                }
                SyntonicSoftmaxMode::Identity => unreachable!(), // Handled above
            }

            let mut output = x.clone();
            output.set_flux_f32(out_flux);
            output.set_device(device.clone());
            output.set_device_idx(device_idx);
            Ok(output)
        } else {
            // F64 path (original implementation)
            let x_flux = x.flux_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Flux missing for CUDA operation")
            })?;
            let mut out_flux = device
                .default_stream()
                .alloc_zeros::<f64>(x.len())
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to allocate GPU memory for output (f64): {}",
                        e
                    ))
                })?;

            match self.mode {
                SyntonicSoftmaxMode::Learned => {
                    let mode_norms = self.mode_norms.as_ref().unwrap();

                    let norms_flux = mode_norms.flux_ref().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "mode_norms must be on GPU for CUDA softmax",
                        )
                    })?;

                    if is_last_dim {
                        cuda_syntonic_softmax_learned_f64(
                            &device,
                            &mut out_flux,
                            x_flux,
                            norms_flux,
                            self.syntony_scale,
                            batch_size as i32,
                            num_classes as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    } else {
                        cuda_syntonic_softmax_learned_strided_f64(
                            &device,
                            &mut out_flux,
                            x_flux,
                            norms_flux,
                            self.syntony_scale,
                            outer_size as i32,
                            dim_size as i32,
                            inner_size as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    }
                }
                SyntonicSoftmaxMode::Provided => {
                    let w = weights.unwrap();
                    let w_flux = w.flux_ref().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "syntony weights must be on GPU for CUDA softmax",
                        )
                    })?;

                    if is_last_dim {
                        cuda_syntonic_softmax_provided_f64(
                            &device,
                            &mut out_flux,
                            x_flux,
                            w_flux,
                            self.syntony_scale,
                            batch_size as i32,
                            num_classes as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    } else {
                        cuda_syntonic_softmax_provided_strided_f64(
                            &device,
                            &mut out_flux,
                            x_flux,
                            w_flux,
                            self.syntony_scale,
                            outer_size as i32,
                            dim_size as i32,
                            inner_size as i32,
                        )
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    }
                }
                SyntonicSoftmaxMode::Identity => unreachable!(), // Handled above
            }

            let mut output = x.clone();
            output.set_flux(out_flux);
            output.set_device(device.clone());
            output.set_device_idx(device_idx);
            Ok(output)
        }
    }

    /// Get current mode norms (for learned mode)
    fn get_mode_norms(&self) -> PyResult<Option<ResonantTensor>> {
        Ok(self.mode_norms.clone())
    }

    /// Transfer state to GPU device
    #[pyo3(signature = (device))]
    fn to_device(&mut self, device: usize) -> PyResult<()> {
        self.to_device_inner(device)
    }

    /// Transfer state to GPU device
    pub fn to_device_inner(&mut self, device: usize) -> PyResult<()> {
        if let Some(ref mut mode_norms) = self.mode_norms {
            if mode_norms.phase() != ResonantPhase::Flux || mode_norms.device_idx() != Some(device)
            {
                *mode_norms = mode_norms.to_device(device)?;
            }
        }
        Ok(())
    }
}

impl SyntonicSoftmaxState {
    /// Compute golden measure weights: w(n) = exp(-|n|²/φ)
    ///
    /// If mode_norms is [N, 8], computes Golden Cone weights.
    /// If mode_norms is [N], assumes inputs are already squared norms (legacy/simple).
    fn compute_golden_weights(&self, mode_norms: &ResonantTensor) -> PyResult<ResonantTensor> {
        let shape = mode_norms.shape();

        if shape.len() == 2 && shape[1] == 8 {
            // 8D E8 Lattice weights
            let data = mode_norms.lattice(); // Returns &Vec<GoldenExact>
            let n = shape[0];
            let projector = GoldenProjector::new();
            let mut weights = Vec::with_capacity(n);

            for chunk in data.chunks(8) {
                if chunk.len() == 8 {
                    // Convert slice to Vec<f64> for compute_8d_weight
                    let vec: Vec<f64> = chunk.iter().map(|g| g.to_f64()).collect();
                    // Weight w = exp(-|P_parallel|^2 / phi)
                    // Note: compute_8d_weight returns the weight directly
                    let w = compute_8d_weight(&vec, &projector);
                    weights.push(w);
                }
            }

            // Return as 1D tensor [N]
            ResonantTensor::from_floats_default_modes(&weights, vec![n], 100)
                .map_err(|e| PyErr::from(e))
        } else {
            // Fallback for scalar/1D norms (legacy)
            // w = exp(-mode_norms / φ)
            let neg_inv_phi = -1.0 / PHI;
            let scaled_norms = mode_norms
                .scalar_mul_core(neg_inv_phi)
                .map_err(|e| PyErr::from(e))?;
            Ok(scaled_norms.exp_core(32).map_err(|e| PyErr::from(e))?)
        }
    }
}

// Python-accessible function
#[pyfunction]
#[pyo3(name = "syntonic_softmax")]
pub fn syntonic_softmax_py(
    x: &ResonantTensor,
    dim: Option<isize>,
    mode: Option<&str>,
    mode_norms: Option<&ResonantTensor>,
    syntony: Option<&ResonantTensor>,
    syntony_scale: Option<f64>,
) -> PyResult<ResonantTensor> {
    // Parse mode string
    let mode_enum = match mode {
        Some(s) => match s {
            "learned" => SyntonicSoftmaxMode::Learned,
            "provided" => SyntonicSoftmaxMode::Provided,
            "identity" => SyntonicSoftmaxMode::Identity,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown mode: {}. Use 'learned', 'provided' or 'identity'",
                    s
                )))
            }
        },
        None => {
            // Default: learned if mode_norms supplied, else identity
            if mode_norms.is_some() {
                SyntonicSoftmaxMode::Learned
            } else {
                SyntonicSoftmaxMode::Identity
            }
        }
    };

    // Input validation
    let x_shape = x.shape();
    let ndim = x_shape.len();

    // Validate input has at least 1 dimension
    if ndim < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input must have at least 1 dimension",
        ));
    }

    // Validate dimension range
    let dim_val = dim.unwrap_or(-1);
    if dim_val < -(ndim as isize) || dim_val >= ndim as isize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dimension {} out of range for tensor with {} dimensions",
            dim_val, ndim
        )));
    }

    // Validate mode_norms if provided
    if let Some(norms) = mode_norms {
        let norms_ndim = norms.shape().len();
        if norms_ndim != 1 && norms_ndim != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode_norms must be 1D (scalar norms) or 2D [N, 8] (E8 vectors)",
            ));
        }
        if norms_ndim == 2 && norms.shape()[1] != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "2D mode_norms must have shape [N, 8], got [{}, {}]",
                norms.shape()[0],
                norms.shape()[1]
            )));
        }
    }

    if let Some(syn) = syntony {
        if syn.shape() != x.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "syntony shape {:?} must match input shape {:?}",
                syn.shape(),
                x.shape()
            )));
        }
    }

    let num_features = if let Some(norms) = mode_norms {
        Some(norms.shape()[0])
    } else {
        None
    };

    let mut state = SyntonicSoftmaxState::new(mode_enum, dim, num_features, syntony_scale)?;

    // If mode_norms provided for learned mode, attach them
    if mode_enum == SyntonicSoftmaxMode::Learned {
        if let Some(norms) = mode_norms {
            state.mode_norms = Some(norms.clone());
        }
    }

    // For provided mode, require syntony
    if mode_enum == SyntonicSoftmaxMode::Provided && syntony.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "syntony tensor required for 'provided' mode",
        ));
    }

    state.forward(x, syntony)
}

/// Compute per-feature syntonic weights from mode_norms (1D norms or 2D [N,8] E8 vectors).
#[pyfunction]
#[pyo3(name = "compute_syntonic_weights")]
pub fn compute_syntonic_weights_py(mode_norms: &ResonantTensor) -> PyResult<ResonantTensor> {
    let shape = mode_norms.shape();
    // 2D [N,8] case
    if shape.len() == 2 && shape[1] == 8 {
        let data = mode_norms.lattice();
        let n = shape[0];
        let projector = GoldenProjector::new();
        let mut weights: Vec<f64> = Vec::with_capacity(n);

        for chunk in data.chunks(8) {
            if chunk.len() == 8 {
                let vec: Vec<f64> = chunk.iter().map(|g| g.to_f64()).collect();
                let w = compute_8d_weight(&vec, &projector);
                weights.push(w);
            }
        }
        ResonantTensor::from_floats_default_modes(&weights, vec![n], 100)
            .map_err(|e| PyErr::from(e))
    } else {
        // Fallback: assume mode_norms are scalar norms and compute w = exp(-norm / phi)
        let neg_inv_phi = -1.0 / PHI;
        let scaled = mode_norms
            .scalar_mul_core(neg_inv_phi)
            .map_err(|e| PyErr::from(e))?;
        Ok(scaled.exp_core(32).map_err(|e| PyErr::from(e))?)
    }
}
