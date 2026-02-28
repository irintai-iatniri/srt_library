//! SRT-specific CUDA kernels for Syntonic Resonance Theory computations.
//!
//! This module provides GPU-accelerated operations for:
//! - Golden ratio operations (φ scaling, gaussian weights)
//! - E₈ lattice projections (P_φ, P_⊥, quadratic forms)
//! - Heat kernel / theta series summation
//! - DHSR cycle operations (differentiation, harmonization, syntony)
//! - SRT correction factors (1 ± q/N)

use crate::tensor::cuda::device_manager::get_device;
use crate::tensor::cuda::memory_pool::CudaComplex64;
use crate::tensor::precision_policy::{srt_operation_policy, PrecisionPolicy};
use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// PTX Kernel Sources (Pre-compiled)
// =============================================================================

const PTX_GOLDEN_SM75: &str = include_str!("../../kernels/ptx/golden_ops_sm75.ptx");
const PTX_GOLDEN_SM80: &str = include_str!("../../kernels/ptx/golden_ops_sm80.ptx");
const PTX_GOLDEN_SM86: &str = include_str!("../../kernels/ptx/golden_ops_sm86.ptx");
const PTX_GOLDEN_SM90: &str = include_str!("../../kernels/ptx/golden_ops_sm90.ptx");

const PTX_E8_SM75: &str = include_str!("../../kernels/ptx/e8_projection_sm75.ptx");
const PTX_E8_SM80: &str = include_str!("../../kernels/ptx/e8_projection_sm80.ptx");
const PTX_E8_SM86: &str = include_str!("../../kernels/ptx/e8_projection_sm86.ptx");
const PTX_E8_SM90: &str = include_str!("../../kernels/ptx/e8_projection_sm90.ptx");

const PTX_HEAT_SM75: &str = include_str!("../../kernels/ptx/heat_kernel_sm75.ptx");
const PTX_HEAT_SM80: &str = include_str!("../../kernels/ptx/heat_kernel_sm80.ptx");
const PTX_HEAT_SM86: &str = include_str!("../../kernels/ptx/heat_kernel_sm86.ptx");
const PTX_HEAT_SM90: &str = include_str!("../../kernels/ptx/heat_kernel_sm90.ptx");

const PTX_DHSR_SM75: &str = include_str!("../../kernels/ptx/dhsr_sm75.ptx");
const PTX_DHSR_SM80: &str = include_str!("../../kernels/ptx/dhsr_sm80.ptx");
const PTX_DHSR_SM86: &str = include_str!("../../kernels/ptx/dhsr_sm86.ptx");
const PTX_DHSR_SM90: &str = include_str!("../../kernels/ptx/dhsr_sm90.ptx");

const PTX_CORR_SM75: &str = include_str!("../../kernels/ptx/corrections_sm75.ptx");
const PTX_CORR_SM80: &str = include_str!("../../kernels/ptx/corrections_sm80.ptx");
const PTX_CORR_SM86: &str = include_str!("../../kernels/ptx/corrections_sm86.ptx");
const PTX_CORR_SM90: &str = include_str!("../../kernels/ptx/corrections_sm90.ptx");

const PTX_RESONANT_SM75: &str = include_str!("../../kernels/ptx/resonant_d_sm75.ptx");
const PTX_RESONANT_SM80: &str = include_str!("../../kernels/ptx/resonant_d_sm80.ptx");
const PTX_RESONANT_SM86: &str = include_str!("../../kernels/ptx/resonant_d_sm86.ptx");
const PTX_RESONANT_SM90: &str = include_str!("../../kernels/ptx/resonant_d_sm90.ptx");

const PTX_PHI_RESIDUAL_SM75: &str = include_str!("../../kernels/ptx/phi_residual_sm75.ptx");
const PTX_PHI_RESIDUAL_SM80: &str = include_str!("../../kernels/ptx/phi_residual_sm80.ptx");
const PTX_PHI_RESIDUAL_SM86: &str = include_str!("../../kernels/ptx/phi_residual_sm86.ptx");
const PTX_PHI_RESIDUAL_SM90: &str = include_str!("../../kernels/ptx/phi_residual_sm90.ptx");

// Matmul PTX (4 compute capabilities)
const PTX_MATMUL_SM75: &str = include_str!("../../kernels/ptx/matmul_sm75.ptx");
const PTX_MATMUL_SM80: &str = include_str!("../../kernels/ptx/matmul_sm80.ptx");
const PTX_MATMUL_SM86: &str = include_str!("../../kernels/ptx/matmul_sm86.ptx");
const PTX_MATMUL_SM90: &str = include_str!("../../kernels/ptx/matmul_sm90.ptx");

const PTX_GOLDEN_BATCH_NORM_SM75: &str =
    include_str!("../../kernels/ptx/golden_batch_norm_sm75.ptx");
const PTX_GOLDEN_BATCH_NORM_SM80: &str =
    include_str!("../../kernels/ptx/golden_batch_norm_sm80.ptx");
const PTX_GOLDEN_BATCH_NORM_SM86: &str =
    include_str!("../../kernels/ptx/golden_batch_norm_sm86.ptx");
const PTX_GOLDEN_BATCH_NORM_SM90: &str =
    include_str!("../../kernels/ptx/golden_batch_norm_sm90.ptx");

// Syntonic Softmax PTX (4 compute capabilities)
const PTX_SYNTONIC_SOFTMAX_SM75: &str = include_str!("../../kernels/ptx/syntonic_softmax_sm75.ptx");
const PTX_SYNTONIC_SOFTMAX_SM80: &str = include_str!("../../kernels/ptx/syntonic_softmax_sm80.ptx");
const PTX_SYNTONIC_SOFTMAX_SM86: &str = include_str!("../../kernels/ptx/syntonic_softmax_sm86.ptx");
const PTX_SYNTONIC_SOFTMAX_SM90: &str = include_str!("../../kernels/ptx/syntonic_softmax_sm90.ptx");

// Hierarchy PTX (4 compute capabilities)
const PTX_HIERARCHY_SM75: &str = include_str!("../../kernels/ptx/hierarchy_sm75.ptx");
const PTX_HIERARCHY_SM80: &str = include_str!("../../kernels/ptx/hierarchy_sm80.ptx");
const PTX_HIERARCHY_SM86: &str = include_str!("../../kernels/ptx/hierarchy_sm86.ptx");
const PTX_HIERARCHY_SM90: &str = include_str!("../../kernels/ptx/hierarchy_sm90.ptx");

// GoldenGELU PTX (4 compute capabilities)
const PTX_GOLDEN_GELU_SM75: &str = include_str!("../../kernels/ptx/golden_gelu_sm75.ptx");
const PTX_GOLDEN_GELU_SM80: &str = include_str!("../../kernels/ptx/golden_gelu_sm80.ptx");
const PTX_GOLDEN_GELU_SM86: &str = include_str!("../../kernels/ptx/golden_gelu_sm86.ptx");
const PTX_GOLDEN_GELU_SM90: &str = include_str!("../../kernels/ptx/golden_gelu_sm90.ptx");

// Prime Selection PTX (4 compute capabilities) - Fermat/Mersenne/Lucas number theory
const PTX_PRIME_SELECTION_SM75: &str = include_str!("../../kernels/ptx/prime_selection_sm75.ptx");
const PTX_PRIME_SELECTION_SM80: &str = include_str!("../../kernels/ptx/prime_selection_sm80.ptx");
const PTX_PRIME_SELECTION_SM86: &str = include_str!("../../kernels/ptx/prime_selection_sm86.ptx");
const PTX_PRIME_SELECTION_SM90: &str = include_str!("../../kernels/ptx/prime_selection_sm90.ptx");

// Prime Ops PTX (4 compute capabilities) - Additional prime computations
const PTX_PRIME_OPS_SM75: &str = include_str!("../../kernels/ptx/prime_ops_sm75.ptx");
const PTX_PRIME_OPS_SM80: &str = include_str!("../../kernels/ptx/prime_ops_sm80.ptx");
const PTX_PRIME_OPS_SM86: &str = include_str!("../../kernels/ptx/prime_ops_sm86.ptx");
const PTX_PRIME_OPS_SM90: &str = include_str!("../../kernels/ptx/prime_ops_sm90.ptx");

// Gnosis PTX (4 compute capabilities) - Consciousness/Gnosis metrics
const PTX_GNOSIS_SM75: &str = include_str!("../../kernels/ptx/gnosis_sm75.ptx");
const PTX_GNOSIS_SM80: &str = include_str!("../../kernels/ptx/gnosis_sm80.ptx");
const PTX_GNOSIS_SM86: &str = include_str!("../../kernels/ptx/gnosis_sm86.ptx");
const PTX_GNOSIS_SM90: &str = include_str!("../../kernels/ptx/gnosis_sm90.ptx");

// Winding Ops PTX (4 compute capabilities)
const PTX_WINDING_OPS_SM75: &str = include_str!("../../kernels/ptx/winding_ops_sm75.ptx");
const PTX_WINDING_OPS_SM80: &str = include_str!("../../kernels/ptx/winding_ops_sm80.ptx");
const PTX_WINDING_OPS_SM86: &str = include_str!("../../kernels/ptx/winding_ops_sm86.ptx");
const PTX_WINDING_OPS_SM90: &str = include_str!("../../kernels/ptx/winding_ops_sm90.ptx");

// Elementwise PTX (4 compute capabilities)
const PTX_ELEMENTWISE_SM75: &str = include_str!("../../kernels/ptx/elementwise_sm75.ptx");
const PTX_ELEMENTWISE_SM80: &str = include_str!("../../kernels/ptx/elementwise_sm80.ptx");
const PTX_ELEMENTWISE_SM86: &str = include_str!("../../kernels/ptx/elementwise_sm86.ptx");
const PTX_ELEMENTWISE_SM90: &str = include_str!("../../kernels/ptx/elementwise_sm90.ptx");

// Core Ops PTX (4 compute capabilities)
const PTX_CORE_OPS_SM75: &str = include_str!("../../kernels/ptx/core_ops_sm75.ptx");
const PTX_CORE_OPS_SM80: &str = include_str!("../../kernels/ptx/core_ops_sm80.ptx");
const PTX_CORE_OPS_SM86: &str = include_str!("../../kernels/ptx/core_ops_sm86.ptx");
const PTX_CORE_OPS_SM90: &str = include_str!("../../kernels/ptx/core_ops_sm90.ptx");

// Conv Ops PTX (4 compute capabilities)
const PTX_CONV_OPS_SM75: &str = include_str!("../../kernels/ptx/conv_ops_sm75.ptx");
const PTX_CONV_OPS_SM80: &str = include_str!("../../kernels/ptx/conv_ops_sm80.ptx");
const PTX_CONV_OPS_SM86: &str = include_str!("../../kernels/ptx/conv_ops_sm86.ptx");
const PTX_CONV_OPS_SM90: &str = include_str!("../../kernels/ptx/conv_ops_sm90.ptx");

const PTX_WMMA_SYNTONIC_SM75: &str = include_str!("../../kernels/ptx/wmma_syntonic_sm75.ptx");
const PTX_WMMA_SYNTONIC_SM80: &str = include_str!("../../kernels/ptx/wmma_syntonic_sm80.ptx");
const PTX_WMMA_SYNTONIC_SM86: &str = include_str!("../../kernels/ptx/wmma_syntonic_sm86.ptx");
const PTX_WMMA_SYNTONIC_SM90: &str = include_str!("../../kernels/ptx/wmma_syntonic_sm90.ptx");

const PTX_SGEMM_NATIVE_SM75: &str = include_str!("../../kernels/ptx/sgemm_native_sm75.ptx");
const PTX_SGEMM_NATIVE_SM80: &str = include_str!("../../kernels/ptx/sgemm_native_sm80.ptx");
const PTX_SGEMM_NATIVE_SM86: &str = include_str!("../../kernels/ptx/sgemm_native_sm86.ptx");
const PTX_SGEMM_NATIVE_SM90: &str = include_str!("../../kernels/ptx/sgemm_native_sm90.ptx");

const PTX_DGEMM_NATIVE_SM75: &str = include_str!("../../kernels/ptx/dgemm_native_sm75.ptx");
const PTX_DGEMM_NATIVE_SM80: &str = include_str!("../../kernels/ptx/dgemm_native_sm80.ptx");
const PTX_DGEMM_NATIVE_SM86: &str = include_str!("../../kernels/ptx/dgemm_native_sm86.ptx");
const PTX_DGEMM_NATIVE_SM90: &str = include_str!("../../kernels/ptx/dgemm_native_sm90.ptx");

const PTX_SCATTER_GATHER_SRT_SM75: &str =
    include_str!("../../kernels/ptx/scatter_gather_srt_sm75.ptx");
const PTX_SCATTER_GATHER_SRT_SM80: &str =
    include_str!("../../kernels/ptx/scatter_gather_srt_sm80.ptx");
const PTX_SCATTER_GATHER_SRT_SM86: &str =
    include_str!("../../kernels/ptx/scatter_gather_srt_sm86.ptx");
const PTX_SCATTER_GATHER_SRT_SM90: &str =
    include_str!("../../kernels/ptx/scatter_gather_srt_sm90.ptx");

const PTX_REDUCTION_SM75: &str = include_str!("../../kernels/ptx/reduction_sm75.ptx");
const PTX_REDUCTION_SM80: &str = include_str!("../../kernels/ptx/reduction_sm80.ptx");
const PTX_REDUCTION_SM86: &str = include_str!("../../kernels/ptx/reduction_sm86.ptx");
const PTX_REDUCTION_SM90: &str = include_str!("../../kernels/ptx/reduction_sm90.ptx");

const PTX_TRILINEAR_SM75: &str = include_str!("../../kernels/ptx/trilinear_sm75.ptx");
const PTX_TRILINEAR_SM80: &str = include_str!("../../kernels/ptx/trilinear_sm80.ptx");
const PTX_TRILINEAR_SM86: &str = include_str!("../../kernels/ptx/trilinear_sm86.ptx");
const PTX_TRILINEAR_SM90: &str = include_str!("../../kernels/ptx/trilinear_sm90.ptx");

// TRILINEAR wrappers (complete set of 9)
pub fn cuda_trilinear_simple_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_toroidal_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_toroidal_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

// Additional TRILINEAR wrappers (7 more)
pub fn cuda_trilinear_phi_weighted_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_phi_weighted_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_golden_decay_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_golden_decay_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_causal_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_causal_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_retrocausal_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_retrocausal_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_symmetric_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_symmetric_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_trilinear_acausal_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("trilinear_acausal_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_bilinear_f64(
    device: &Arc<CudaDevice>,
    out: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load trilinear: {}",
                e
            ))
        })?;
    let func = module
        .load_function("bilinear_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

const PTX_COMPLEX_OPS_SM75: &str = include_str!("../../kernels/ptx/complex_ops_sm75.ptx");
const PTX_COMPLEX_OPS_SM80: &str = include_str!("../../kernels/ptx/complex_ops_sm80.ptx");
const PTX_COMPLEX_OPS_SM86: &str = include_str!("../../kernels/ptx/complex_ops_sm86.ptx");
const PTX_COMPLEX_OPS_SM90: &str = include_str!("../../kernels/ptx/complex_ops_sm90.ptx");

const PTX_ATTENTION_SM75: &str = include_str!("../../kernels/ptx/attention_sm75.ptx");
const PTX_ATTENTION_SM80: &str = include_str!("../../kernels/ptx/attention_sm80.ptx");
const PTX_ATTENTION_SM86: &str = include_str!("../../kernels/ptx/attention_sm86.ptx");
const PTX_ATTENTION_SM90: &str = include_str!("../../kernels/ptx/attention_sm90.ptx");

const PTX_AUTOGRAD_SM75: &str = include_str!("../../kernels/ptx/autograd_sm75.ptx");
const PTX_AUTOGRAD_SM80: &str = include_str!("../../kernels/ptx/autograd_sm80.ptx");
const PTX_AUTOGRAD_SM86: &str = include_str!("../../kernels/ptx/autograd_sm86.ptx");
const PTX_AUTOGRAD_SM90: &str = include_str!("../../kernels/ptx/autograd_sm90.ptx");

const PTX_ATTRACTOR_SM75: &str = include_str!("../../kernels/ptx/attractor_sm75.ptx");
const PTX_ATTRACTOR_SM80: &str = include_str!("../../kernels/ptx/attractor_sm80.ptx");
const PTX_ATTRACTOR_SM86: &str = include_str!("../../kernels/ptx/attractor_sm86.ptx");
const PTX_ATTRACTOR_SM90: &str = include_str!("../../kernels/ptx/attractor_sm90.ptx");

// =============================================================================
// Kernel Function Lists
// =============================================================================

/// Golden operations kernel functions
const GOLDEN_FUNCS: &[&str] = &[
    "scale_phi_f64",
    "scale_phi_f32",
    "scale_phi_inv_f64",
    "scale_phi_inv_f32",
    "fma_phi_kernel_f64",
    "fma_phi_kernel_f32",
    "fma_phi_inv_kernel_f64",
    "fma_phi_inv_kernel_f32",
    "golden_gaussian_weight_scalar_f64",
    "golden_gaussian_weight_scalar_f32",
    "golden_gaussian_weight_4d_int",
    "golden_gaussian_weight_4d_f32",
    "golden_gaussian_weight_8d_f32",
    "golden_gaussian_weight_8d_f64",
    "golden_recursion_4d_int",
    "golden_recursion_f32",
    "golden_recursion_f64",
    "golden_recursion_inv_4d_int",
    "fibonacci_binet_f64",
    "lucas_binet_f64",
    "compute_generation_4d_int",
    "weighted_inner_product_golden_f32",
    "golden_normalize_f32",
    "golden_norm_factor_f32",
    "scale_phi_c128",
    "golden_gaussian_weight_c128",
];

/// E₈ projection kernel functions
const E8_FUNCS: &[&str] = &[
    "project_parallel_f32",
    "project_parallel_f64",
    "project_perpendicular_f32",
    "project_perpendicular_f64",
    "quadratic_form_f32",
    "quadratic_form_f64",
    "golden_cone_test_f32",
    "golden_cone_test_f64",
    "e8_batch_projection_f32",
    "e8_batch_projection_f64",
    "count_cone_roots",
    "norm_squared_8d_f32",
    "norm_squared_8d_f64",
    "weighted_e8_contribution_f32",
    "weighted_e8_contribution_f64",
];

/// Heat kernel / theta series functions
const HEAT_FUNCS: &[&str] = &[
    "theta_series_sum_f32",
    "theta_series_sum_f64",
    "heat_kernel_e8_f32",
    "heat_kernel_e8_f64",
    "theta_series_shells_f32",
    "golden_weighted_sum_f32",
    "golden_weighted_sum_f64",
    "modular_inversion_f32",
    "modular_inversion_f64",
    "spectral_zeta_f64",
    "winding_heat_kernel_f32",
    "knot_contribution_f32",
];

/// DHSR cycle functions
const DHSR_FUNCS: &[&str] = &[
    "compute_syntony_f32",
    "compute_syntony_c128",
    "differentiation_f32",
    "differentiation_c128",
    "harmonization_f32",
    "harmonization_c128",
    "dhsr_cycle_f32",
    "dhsr_cycle_c128",
    "dhsr_cycle_inplace_f32",
    "dhsr_cycle_inplace_c128",
    "compute_gnosis_f32",
    "verify_dh_partition_f32",
    "dhsr_multi_cycle_c128",
    "harmonize_history_kernel_f32",
    "harmonize_history_kernel_f64",
    "archonic_filter_kernel_f32",
    "archonic_filter_kernel_f64",
    "entropy_kernel_f32",
    "entropy_kernel_f64",
    "syntony_metric_kernel_f32",
    "syntony_metric_kernel_f64",
    "gnosis_mask_kernel_f32",
    "gnosis_mask_kernel_f64",
    "adaptive_gnosis_mask_kernel_f32",
    "fractal_gnosis_mask_kernel_f32",
    "temporal_gnosis_mask_kernel_f32",
    // New accelerated DHSR operators
    "fourier_project_single_f64",
    "fourier_project_batch_f64",
    "laplacian_1d_f64",
    "laplacian_2d_f64",
    "differentiation_full_f64",
    "damping_cascade_f64",
    "syntony_projection_f64",
    "harmonization_full_f64",
    "dhsr_step_fused_f64",
    // Fixed-Point (Exact) Kernels
    "compute_syntony_fp64",
    "differentiation_fp64",
    "harmonization_fp64",
    "dhsr_cycle_fp64",
    "fourier_project_batch_fp64",
    "differentiation_full_fp64",
    "laplacian_1d_fp64",
];

/// DHSR gravity functions
const DHSR_GRAVITY_FUNCS: &[&str] = &["apply_geodesic_gravity_f64"];

/// Correction factor functions
const CORR_FUNCS: &[&str] = &[
    "apply_correction_f64",
    "apply_correction_f32",
    "apply_corrections_batch_f64",
    "apply_corrections_batch_f32",
    "compound_correction_f64",
    "compound_correction_f32",
    "lepton_mass_correction_f64",
    "quark_mass_correction_f64",
    "coupling_correction_f64",
    "custom_correction_f64",
    "custom_correction_f32",
    "higgs_mass_correction_f64",
    "mixing_matrix_correction_f64",
    "compute_correction_factors_f64",
    "compute_correction_factors_f32",
    "get_q_deficit",
    "get_structure_dimension",
];

/// Resonant D-phase functions
const RESONANT_FUNCS: &[&str] = &[
    "resonant_d_phase_f64",
    "resonant_d_phase_f32",
    "resonant_d_phase_batch_f64",
    "resonant_compute_syntony_f64",
    "resonant_compute_syntony_f32",
    "resonant_weighted_snap_gradient_f64",
    "resonant_argmax_syntony_f64",
    "resonant_box_muller_f64",
    "resonant_box_muller_f32",
    "resonant_residual_modulated_noise_f64",
    "resonant_compute_dwell_f64",
];

/// Phi-residual connection functions
const PHI_RESIDUAL_FUNCS: &[&str] = &[
    "phi_residual_mode_phi_f64",
    "phi_residual_mode_phi_f32",
    "phi_residual_mode_symmetric_f64",
    "phi_residual_mode_symmetric_f32",
    "phi_residual_mode_standard_f64",
    "phi_residual_mode_standard_f32",
    "phi_residual_relu_f64",
    "phi_residual_relu_f32",
    "phi_residual_gelu_f64",
    "phi_residual_gelu_f32",
    "phi_residual_layernorm_f64",
    "phi_residual_layernorm_f32",
    "phi_residual_mode_phi_vec4_f32",
    "phi_residual_mode_phi_vec2_f64",
    "phi_residual_component_norm_f64",
];

/// Golden batch normalization functions
const GOLDEN_BATCH_NORM_FUNCS: &[&str] = &[
    "golden_bn_1d_compute_stats_f64",
    "golden_bn_1d_compute_stats_f32",
    "golden_bn_1d_normalize_f64",
    "golden_bn_1d_normalize_f32",
    "golden_bn_2d_compute_stats_f64",
    "golden_bn_2d_compute_stats_f32",
    "golden_bn_2d_normalize_f64",
    "golden_bn_2d_normalize_f32",
    "golden_bn_1d_fused_f64",
    "golden_bn_1d_fused_f32",
    "golden_layer_norm_f64",
    "golden_layer_norm_f32",
    "compute_output_stats_f64",
];

/// Syntonic softmax functions
const SYNTONIC_SOFTMAX_FUNCS: &[&str] = &[
    // Learned mode
    "cuda_syntonic_softmax_learned_f64",
    "cuda_syntonic_softmax_learned_f32",
    "cuda_syntonic_softmax_learned_strided_f64",
    "cuda_syntonic_softmax_learned_strided_f32",
    // Provided mode
    "cuda_syntonic_softmax_provided_f64",
    "cuda_syntonic_softmax_provided_f32",
    "cuda_syntonic_softmax_provided_strided_f64",
    "cuda_syntonic_softmax_provided_strided_f32",
    // Identity mode (standard softmax)
    "cuda_softmax_identity_f64",
    "cuda_softmax_identity_f32",
    "cuda_softmax_identity_strided_f64",
    "cuda_softmax_identity_strided_f32",
];

/// Matmul kernel functions
const MATMUL_FUNCS: &[&str] = &[
    // Standard matrix multiplication
    "matmul_f64",
    "matmul_f32",
    "matmul_c128",
    "matmul_tiled_f64",
    "matmul_tiled_f32",
    // Transposed variants
    "matmul_tn_f64",
    "matmul_tn_f32",
    "matmul_tn_c128",
    "matmul_nt_f64",
    "matmul_nt_f32",
    "matmul_tt_f64",
    "matmul_tt_f32",
    // Hermitian variants (complex)
    "matmul_hn_c128",
    "matmul_nh_c128",
    // GEMM operations
    "gemm_nn_f64",
    "gemm_tn_f64",
    "gemm_nt_f64",
    "gemm_tt_f64",
    // Batched operations
    "bmm_f64",
    "bmm_c128",
    "bmm_nt_f64",
    // SRT-specific operations
    "matmul_phi_scaled_f64",
    "golden_commutator_f64",
    "golden_anticommutator_f64",
    "matmul_golden_weighted_f64",
    "matmul_golden_weighted_c128",
    // Complex arithmetic
    "complex_div_c128",
    "complex_reciprocal_c128",
];

/// Hierarchy correction kernel functions
const HIERARCHY_FUNCS: &[&str] = &[
    "apply_correction_f64",
    "apply_correction_f32",
    "apply_correction_uniform_f64",
    "apply_special_correction_f64",
    "apply_winding_instability_f64",
    "apply_recursion_penalty_f64",
    "apply_double_inverse_f64",
    "apply_fixed_point_penalty_f64",
    "apply_correction_chain_f64",
    "compute_e_star_n_f64",
    "apply_correction_by_name_f64",
    "compute_phi_powers_f64",
];

/// Golden GELU activation kernel functions
const GOLDEN_GELU_FUNCS: &[&str] = &[
    "golden_gelu_f64",
    "golden_gelu_f32",
    "golden_gelu_backward_f64",
    "golden_gelu_backward_f32",
    "batched_golden_gelu_f64",
];

/// Prime selection kernel functions (Fermat/Mersenne/Lucas)
const PRIME_SELECTION_FUNCS: &[&str] = &[
    "fermat_numbers_kernel",
    "is_fermat_prime_kernel",
    "mersenne_numbers_kernel",
    "lucas_lehmer_kernel",
    "lucas_numbers_kernel",
    "shadow_phase_kernel",
    "lucas_boost_kernel",
    "lucas_extended_kernel",
    "fibonacci_lucas_kernel",
    "apply_hierarchy_correction_kernel",
    "apply_suppression_kernel",
];

/// Prime ops kernel functions
const PRIME_OPS_FUNCS: &[&str] = &[
    "mersenne_prime_check_kernel",
    "fermat_prime_check_kernel",
    "lucas_prime_check_kernel",
    "pisano_period_kernel",
    "fibonacci_resonance_boost_kernel",
    "lucas_gap_pressure_kernel",
];

/// Gnosis kernel functions (consciousness metrics)
const GNOSIS_FUNCS: &[&str] = &[
    "is_conscious_kernel",
    "gnosis_score_kernel",
    "creativity_kernel",
    "consciousness_probability_kernel",
    "dhsr_gnosis_step_kernel",
];

/// Winding ops kernel functions
const WINDING_OPS_FUNCS: &[&str] = &[
    "compute_winding_number_kernel",
    "winding_distance_kernel",
    "winding_norm_kernel",
];

/// Elementwise kernel functions
const ELEMENTWISE_FUNCS: &[&str] = &[
    "add_f64",
    "add_f32",
    "sub_f64",
    "sub_f32",
    "mul_f64",
    "mul_f32",
    "div_f64",
    "div_f32",
    "neg_f64",
    "neg_f32",
    "abs_f64",
    "abs_f32",
    "sqrt_f64",
    "sqrt_f32",
    "exp_f64",
    "exp_f32",
    "log_f64",
    "log_f32",
    "sin_f64",
    "cos_f64",
    "tanh_f64",
    "sigmoid_f64",
    "relu_f64",
    "relu_f32",
    // Toroidal math functions (W⁴ geometry)
    "sin_toroidal_f64",
    "sin_toroidal_f32",
    "cos_toroidal_f64",
    "cos_toroidal_f32",
    "atan2_toroidal_f64",
    "atan2_toroidal_f32",
    // Golden exponentials (consciousness growth)
    "phi_exp_f64",
    "phi_exp_f32",
    "phi_exp_inv_f64",
    "phi_exp_inv_f32",
];

/// Core ops kernel functions
const CORE_OPS_FUNCS: &[&str] = &[
    "sum_f64",
    "sum_f32",
    "mean_f64",
    "mean_f32",
    "max_f64",
    "min_f64",
    "reduce_sum_f64",
    "reduce_mean_f64",
    "layer_norm_f64",
    "dropout_f64",
];

/// Conv ops kernel functions  
const CONV_OPS_FUNCS: &[&str] = &[
    "conv1d_f64",
    "conv1d_f32",
    "conv2d_f64",
    "conv2d_f32",
    "conv1d_backward_f64",
    "conv2d_backward_f64",
];

/// Autograd kernel functions (standard backward pass)
const AUTOGRAD_FUNCS: &[&str] = &[
    "backward_add_f64",
    "backward_add_f32",
    "backward_mul_f64",
    "backward_mul_f32",
    "backward_matmul_f64",
    "backward_matmul_f32",
    "backward_softmax_f64",
    "backward_softmax_f32",
    "backward_layernorm_f64",
    "backward_layernorm_f32",
    "backward_elementwise_f64",
    "backward_elementwise_f32",
];

/// Attractor kernel functions (retrocausal backward pass)
const ATTRACTOR_FUNCS: &[&str] = &[
    "attractor_memory_update_f64",
    "hooking_coefficient_f64",
    "retrocausal_harmonize_f64",
    "attractor_distance_f64",
    "attractor_centroid_f64",
];

/// WMMA Syntonic kernel functions (tensor cores with golden weighting)
const WMMA_SYNTONIC_FUNCS: &[&str] = &["wmma_golden_weighted_fp16", "wmma_syntonic_fp16"];

/// Scatter/Gather SRT kernel functions (index-based operations)
const SCATTER_GATHER_FUNCS: &[&str] = &[
    "gather_f64",
    "gather_f32",
    "scatter_f64",
    "scatter_f32",
    "scatter_add_f64",
    "scatter_add_f32",
    "gather_phi_weighted_f64",
    "scatter_golden_f64",
    "scatter_mersenne_stable_f64",
    "gather_lucas_shadow_f64",
    "gather_pisano_hooked_f64",
    "gather_e8_roots_f64",
    "scatter_golden_cone_f64",
    "gather_transcendence_gate_f64",
    "scatter_consciousness_threshold_f64",
];

/// Reduction kernel functions (aggregation operations)
const REDUCTION_FUNCS: &[&str] = &[
    "reduce_sum_f64",
    "reduce_sum_f32",
    "reduce_mean_f64",
    "reduce_mean_f32",
    "reduce_max_f64",
    "reduce_max_f32",
    "reduce_min_f64",
    "reduce_min_f32",
    "reduce_norm_l2_f64",
    "reduce_norm_l2_f32",
    "reduce_sum_golden_weighted_f64",
    "reduce_syntony_f64",
    "reduce_sum_rows_f64",
    "reduce_sum_cols_f64",
    "reduce_sum_phi_scaled_f64",
    "reduce_variance_golden_target_f64",
    "reduce_sum_c128",
    "reduce_norm_c128",
    "reduce_sum_mersenne_stable_f64",
    "reduce_sum_lucas_shadow_f64",
    "reduce_syntony_deviation_f64",
    "reduce_consciousness_count_f64",
    "reduce_sum_q_corrected_f64",
    "reduce_e8_norm_f64",
];

/// Trilinear kernel functions (3D interpolation)
const TRILINEAR_FUNCS: &[&str] = &[
    "trilinear_f64",
    "trilinear_toroidal_f64",
    "trilinear_phi_weighted_f64",
    "trilinear_golden_decay_f64",
    "trilinear_causal_f64",
    "trilinear_retrocausal_f64",
    "trilinear_symmetric_f64",
    "trilinear_acausal_f64",
    "bilinear_f64",
];

/// Complex operations kernel functions
const COMPLEX_OPS_FUNCS: &[&str] = &[
    "arg_c128",
    "arg_c64",
    "normalize_phase_c128",
    "normalize_phase_c64",
    "rotate_phase_c128",
    "rotate_phase_c64",
    "quantize_phase_pi_c128",
    "conj_c128",
    "conj_c64",
    "hermitian_inner_c128",
    "phase_syntony_c128",
    "berry_phase_c128",
    "probability_c128",
    "probability_c64",
    "normalize_wavefunction_c128",
    "golden_rotate_c128",
    "phi_weighted_sum_c128",
];

/// Attention kernel functions (flash attention variants)
const ATTENTION_FUNCS: &[&str] = &[
    "flash_attention_f32",
    "flash_attention_syntony_f32",
    "flash_attention_golden_f32",
    "flash_attention_mersenne_127_f32",
    "flash_attention_causal_f32",
    "flash_attention_retrocausal_f32",
];

// =============================================================================
// PTX Selection Based on Compute Capability
// =============================================================================

fn select_golden_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_GOLDEN_SM90
    } else if cc >= 86 {
        PTX_GOLDEN_SM86
    } else if cc >= 80 {
        PTX_GOLDEN_SM80
    } else {
        PTX_GOLDEN_SM75
    }
}

fn select_e8_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_E8_SM90
    } else if cc >= 86 {
        PTX_E8_SM86
    } else if cc >= 80 {
        PTX_E8_SM80
    } else {
        PTX_E8_SM75
    }
}

fn select_heat_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_HEAT_SM90
    } else if cc >= 86 {
        PTX_HEAT_SM86
    } else if cc >= 80 {
        PTX_HEAT_SM80
    } else {
        PTX_HEAT_SM75
    }
}

fn select_dhsr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_DHSR_SM90
    } else if cc >= 86 {
        PTX_DHSR_SM86
    } else if cc >= 80 {
        PTX_DHSR_SM80
    } else {
        PTX_DHSR_SM75
    }
}

fn select_corr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_CORR_SM90
    } else if cc >= 86 {
        PTX_CORR_SM86
    } else if cc >= 80 {
        PTX_CORR_SM80
    } else {
        PTX_CORR_SM75
    }
}

fn select_resonant_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_RESONANT_SM90
    } else if cc >= 86 {
        PTX_RESONANT_SM86
    } else if cc >= 80 {
        PTX_RESONANT_SM80
    } else {
        PTX_RESONANT_SM75
    }
}

fn select_phi_residual_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_PHI_RESIDUAL_SM90
    } else if cc >= 86 {
        PTX_PHI_RESIDUAL_SM86
    } else if cc >= 80 {
        PTX_PHI_RESIDUAL_SM80
    } else {
        PTX_PHI_RESIDUAL_SM75
    }
}

fn select_golden_batch_norm_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_GOLDEN_BATCH_NORM_SM90
    } else if cc >= 86 {
        PTX_GOLDEN_BATCH_NORM_SM86
    } else if cc >= 80 {
        PTX_GOLDEN_BATCH_NORM_SM80
    } else {
        PTX_GOLDEN_BATCH_NORM_SM75
    }
}

fn select_syntonic_softmax_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_SYNTONIC_SOFTMAX_SM90
    } else if cc >= 86 {
        PTX_SYNTONIC_SOFTMAX_SM86
    } else if cc >= 80 {
        PTX_SYNTONIC_SOFTMAX_SM80
    } else {
        PTX_SYNTONIC_SOFTMAX_SM75
    }
}

fn select_matmul_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_MATMUL_SM90
    } else if cc >= 86 {
        PTX_MATMUL_SM86
    } else if cc >= 80 {
        PTX_MATMUL_SM80
    } else {
        PTX_MATMUL_SM75
    }
}

fn select_hierarchy_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_HIERARCHY_SM90
    } else if cc >= 86 {
        PTX_HIERARCHY_SM86
    } else if cc >= 80 {
        PTX_HIERARCHY_SM80
    } else {
        PTX_HIERARCHY_SM75
    }
}

fn select_golden_gelu_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_GOLDEN_GELU_SM90
    } else if cc >= 86 {
        PTX_GOLDEN_GELU_SM86
    } else if cc >= 80 {
        PTX_GOLDEN_GELU_SM80
    } else {
        PTX_GOLDEN_GELU_SM75
    }
}

fn select_prime_selection_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_PRIME_SELECTION_SM90
    } else if cc >= 86 {
        PTX_PRIME_SELECTION_SM86
    } else if cc >= 80 {
        PTX_PRIME_SELECTION_SM80
    } else {
        PTX_PRIME_SELECTION_SM75
    }
}

fn select_prime_ops_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_PRIME_OPS_SM90
    } else if cc >= 86 {
        PTX_PRIME_OPS_SM86
    } else if cc >= 80 {
        PTX_PRIME_OPS_SM80
    } else {
        PTX_PRIME_OPS_SM75
    }
}

fn select_gnosis_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_GNOSIS_SM90
    } else if cc >= 86 {
        PTX_GNOSIS_SM86
    } else if cc >= 80 {
        PTX_GNOSIS_SM80
    } else {
        PTX_GNOSIS_SM75
    }
}

fn select_winding_ops_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_WINDING_OPS_SM90
    } else if cc >= 86 {
        PTX_WINDING_OPS_SM86
    } else if cc >= 80 {
        PTX_WINDING_OPS_SM80
    } else {
        PTX_WINDING_OPS_SM75
    }
}

fn select_elementwise_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_ELEMENTWISE_SM90
    } else if cc >= 86 {
        PTX_ELEMENTWISE_SM86
    } else if cc >= 80 {
        PTX_ELEMENTWISE_SM80
    } else {
        PTX_ELEMENTWISE_SM75
    }
}

fn select_core_ops_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_CORE_OPS_SM90
    } else if cc >= 86 {
        PTX_CORE_OPS_SM86
    } else if cc >= 80 {
        PTX_CORE_OPS_SM80
    } else {
        PTX_CORE_OPS_SM75
    }
}

fn select_conv_ops_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_CONV_OPS_SM90
    } else if cc >= 86 {
        PTX_CONV_OPS_SM86
    } else if cc >= 80 {
        PTX_CONV_OPS_SM80
    } else {
        PTX_CONV_OPS_SM75
    }
}

fn select_autograd_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_AUTOGRAD_SM90
    } else if cc >= 86 {
        PTX_AUTOGRAD_SM86
    } else if cc >= 80 {
        PTX_AUTOGRAD_SM80
    } else {
        PTX_AUTOGRAD_SM75
    }
}

fn select_attractor_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_ATTRACTOR_SM90
    } else if cc >= 86 {
        PTX_ATTRACTOR_SM86
    } else if cc >= 80 {
        PTX_ATTRACTOR_SM80
    } else {
        PTX_ATTRACTOR_SM75
    }
}

// =============================================================================
// Kernel Loading
// =============================================================================

/// Get compute capability from device
fn get_compute_capability(device: &Arc<CudaDevice>) -> (i32, i32) {
    use cudarc::driver::result;
    use cudarc::driver::sys::CUdevice_attribute_enum;

    let ordinal = device.ordinal() as i32;

    let major = unsafe {
        result::device::get_attribute(
            ordinal,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap_or(7)
    };
    let minor = unsafe {
        result::device::get_attribute(
            ordinal,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap_or(0)
    };

    (major, minor)
}

// ============================================================================
// Static CUDA Library FFI Declarations
// ============================================================================
// These functions are compiled from host_wrappers.cu and linked statically.
// They provide an alternative to PTX runtime loading with better scaling.

#[cfg(feature = "cuda")]
#[link(name = "srt_cuda_kernels", kind = "static")]
extern "C" {
    // DHSR Operations
    fn host_compute_syntony_c128(
        numerator: *mut f64,
        denominator: *mut f64,
        psi: *const f64,
        mode_norm_sq: *const f64,
        n: i32,
    ) -> i32;

    fn host_laplacian_1d_f64(out: *mut f64, in_data: *const f64, size: i32) -> i32;

    fn host_fourier_project_batch_f64(
        out: *mut f64,
        in_data: *const f64,
        modes: *const i32,
        weights: *const f64,
        num_modes: i32,
        size: i32,
    ) -> i32;

    fn host_damping_cascade_f64(
        out: *mut f64,
        in_data: *const f64,
        gamma_0: f64,
        delta_d: f64,
        num_dampers: i32,
        size: i32,
        phi_weight: f64,
    ) -> i32;

    fn host_syntony_projection_f64(
        out: *mut f64,
        in_data: *const f64,
        target_syntony: *const f64,
        strength: f64,
        size: i32,
    ) -> i32;

    fn host_dhsr_step_fused_f64(
        state: *mut f64,
        syntony_out: *mut f64,
        mode_norms: *const f64,
        attractors: *const f64,
        diff_strength: f64,
        harm_strength: f64,
        retro_pull: f64,
        size: i32,
    ) -> i32;

    // Attractor Operations
    fn host_attractor_memory_update_f64(
        attractors: *mut f64,
        new_state: *const f64,
        syntony_scores: *const f64,
        min_syntony: f64,
        decay_rate: f64,
        state_size: i32,
        max_attractors: i32,
    ) -> i32;

    fn host_retrocausal_harmonize_f64(
        out: *mut f64,
        current: *const f64,
        attractor_centroid: *const f64,
        pull_strength: f64,
        syntony_threshold: f64,
        size: i32,
    ) -> i32;

    fn host_attractor_centroid_f64(
        centroid: *mut f64,
        attractors: *const f64,
        weights: *const f64,
        num_attractors: i32,
        state_size: i32,
    ) -> i32;

    // Golden Operations
    fn host_sin_toroidal_f64(out: *mut f64, in_data: *const f64, n: i32) -> i32;
    fn host_cos_toroidal_f64(out: *mut f64, in_data: *const f64, n: i32) -> i32;
    fn host_atan2_toroidal_f64(out: *mut f64, y: *const f64, x: *const f64, n: i32) -> i32;
    fn host_phi_exp_f64(out: *mut f64, in_data: *const f64, n: i32) -> i32;
    fn host_phi_exp_inv_f64(out: *mut f64, in_data: *const f64, n: i32) -> i32;
    fn host_golden_entropy_f64(out: *mut f64, in_data: *const f64, n: i32) -> i32;

    // Scatter/Gather Operations
    fn host_gather_f64(out: *mut f64, src: *const f64, indices: *const i32, n: i32) -> i32;
    fn host_scatter_add_f32(out: *mut f32, src: *const f32, indices: *const i32, n: i32) -> i32;
    fn host_gather_phi_weighted_f64(
        out: *mut f64,
        src: *const f64,
        indices: *const i32,
        weights: *const f64,
        n: i32,
    ) -> i32;

    // Complex Operations
    fn host_arg_c128(out: *mut f64, in_data: *const f64, n: i32) -> i32;
    fn host_phase_syntony_c128(out: *mut f64, in_data: *const f64, n: i32) -> i32;

    // Trilinear
    fn host_trilinear_f64(
        out: *mut f64,
        x: *const f64,
        y: *const f64,
        z: *const f64,
        nx: i32,
        ny: i32,
        nz: i32,
        count: i32,
    ) -> i32;

    // Gnosis
    fn host_gnosis_mask_f64(
        out: *mut f64,
        input: *const f64,
        syntony: *const f64,
        size: i32,
        threshold: f64,
        strength: f64,
    ) -> i32;

    // Matrix Operations
    fn host_matmul_f64(
        c: *mut f64,
        a: *const f64,
        b: *const f64,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    // Phase-State Compiler Operations (int8 exact-integer pipeline)
    fn host_phase_state_run_2d(
        m4: *mut i8,
        t4: *mut i8,
        is_source: *const i8,
        stale_cycles: *mut i32,
        recursive_depth: *mut i32,
        rows: i32,
        cols: i32,
        stale_threshold: i32,
        k_threshold: i32,
        allow_novelty: i32,
        max_cycles: i32,
        out_gnosis: *mut i32,
        out_cycles_used: *mut i32,
    ) -> i32;
}

// ============================================================================
// Static Library Wrapper Functions
// ============================================================================
// These call the FFI functions when static linking is available.

/// Call statically linked laplacian kernel
#[cfg(feature = "cuda")]
pub fn static_laplacian_1d_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_laplacian_1d_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked syntony projection kernel
#[cfg(feature = "cuda")]
pub fn static_syntony_projection_f64(
    out: &mut [f64],
    input: &[f64],
    target: &[f64],
    strength: f64,
) -> Result<(), String> {
    if out.len() != input.len() || out.len() != target.len() {
        return Err("All arrays must have same length".to_string());
    }
    let result = unsafe {
        host_syntony_projection_f64(
            out.as_mut_ptr(),
            input.as_ptr(),
            target.as_ptr(),
            strength,
            input.len() as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked sin toroidal kernel
#[cfg(feature = "cuda")]
pub fn static_sin_toroidal_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_sin_toroidal_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked cos toroidal kernel
#[cfg(feature = "cuda")]
pub fn static_cos_toroidal_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_cos_toroidal_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked phi exp kernel
#[cfg(feature = "cuda")]
pub fn static_phi_exp_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_phi_exp_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked phi exp inv kernel
#[cfg(feature = "cuda")]
pub fn static_phi_exp_inv_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_phi_exp_inv_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked gather kernel
#[cfg(feature = "cuda")]
pub fn static_gather_f64(out: &mut [f64], src: &[f64], indices: &[i32]) -> Result<(), String> {
    if out.len() != indices.len() {
        return Err("Output and indices must have same length".to_string());
    }
    let result = unsafe {
        host_gather_f64(out.as_mut_ptr(), src.as_ptr(), indices.as_ptr(), indices.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked scatter add kernel
#[cfg(feature = "cuda")]
pub fn static_scatter_add_f32(out: &mut [f32], src: &[f32], indices: &[i32]) -> Result<(), String> {
    if src.len() != indices.len() {
        return Err("Source and indices must have same length".to_string());
    }
    let result = unsafe {
        host_scatter_add_f32(out.as_mut_ptr(), src.as_ptr(), indices.as_ptr(), indices.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked compute_syntony kernel for c128
#[cfg(feature = "cuda")]
pub fn static_compute_syntony_c128(
    numerator: &mut [f64],
    denominator: &mut [f64],
    psi: &[f64],
    mode_norm_sq: &[f64],
) -> Result<(), String> {
    let n = psi.len() / 2; // complex values
    if numerator.len() < 1 || denominator.len() < 1 {
        return Err("Output buffers too small".to_string());
    }
    let result = unsafe {
        host_compute_syntony_c128(
            numerator.as_mut_ptr(),
            denominator.as_mut_ptr(),
            psi.as_ptr(),
            mode_norm_sq.as_ptr(),
            n as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked fourier project batch kernel
#[cfg(feature = "cuda")]
pub fn static_fourier_project_batch_f64(
    out: &mut [f64],
    input: &[f64],
    modes: &[i32],
    weights: &[f64],
) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    if modes.len() != weights.len() {
        return Err("Modes and weights must have same length".to_string());
    }
    let result = unsafe {
        host_fourier_project_batch_f64(
            out.as_mut_ptr(),
            input.as_ptr(),
            modes.as_ptr(),
            weights.as_ptr(),
            modes.len() as i32,
            input.len() as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked damping cascade kernel
#[cfg(feature = "cuda")]
pub fn static_damping_cascade_f64(
    out: &mut [f64],
    input: &[f64],
    gamma_0: f64,
    delta_d: f64,
    num_dampers: i32,
    phi_weight: f64,
) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_damping_cascade_f64(
            out.as_mut_ptr(),
            input.as_ptr(),
            gamma_0,
            delta_d,
            num_dampers,
            input.len() as i32,
            phi_weight,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked DHSR step fused kernel
#[cfg(feature = "cuda")]
pub fn static_dhsr_step_fused_f64(
    state: &mut [f64],
    syntony_out: &mut [f64],
    mode_norms: &[f64],
    attractors: &[f64],
    diff_strength: f64,
    harm_strength: f64,
    retro_pull: f64,
) -> Result<(), String> {
    let result = unsafe {
        host_dhsr_step_fused_f64(
            state.as_mut_ptr(),
            syntony_out.as_mut_ptr(),
            mode_norms.as_ptr(),
            attractors.as_ptr(),
            diff_strength,
            harm_strength,
            retro_pull,
            state.len() as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked attractor memory update kernel
#[cfg(feature = "cuda")]
pub fn static_attractor_memory_update_f64(
    attractors: &mut [f64],
    new_state: &[f64],
    syntony_scores: &[f64],
    min_syntony: f64,
    decay_rate: f64,
    max_attractors: i32,
) -> Result<(), String> {
    let state_size = new_state.len() as i32;
    let result = unsafe {
        host_attractor_memory_update_f64(
            attractors.as_mut_ptr(),
            new_state.as_ptr(),
            syntony_scores.as_ptr(),
            min_syntony,
            decay_rate,
            state_size,
            max_attractors,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked retrocausal harmonize kernel
#[cfg(feature = "cuda")]
pub fn static_retrocausal_harmonize_f64(
    out: &mut [f64],
    current: &[f64],
    attractor_centroid: &[f64],
    pull_strength: f64,
    syntony_threshold: f64,
) -> Result<(), String> {
    if out.len() != current.len() || current.len() != attractor_centroid.len() {
        return Err("All arrays must have same length".to_string());
    }
    let result = unsafe {
        host_retrocausal_harmonize_f64(
            out.as_mut_ptr(),
            current.as_ptr(),
            attractor_centroid.as_ptr(),
            pull_strength,
            syntony_threshold,
            current.len() as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked attractor centroid kernel
#[cfg(feature = "cuda")]
pub fn static_attractor_centroid_f64(
    centroid: &mut [f64],
    attractors: &[f64],
    weights: &[f64],
    num_attractors: i32,
) -> Result<(), String> {
    let state_size = centroid.len() as i32;
    let result = unsafe {
        host_attractor_centroid_f64(
            centroid.as_mut_ptr(),
            attractors.as_ptr(),
            weights.as_ptr(),
            num_attractors,
            state_size,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked atan2 toroidal kernel
#[cfg(feature = "cuda")]
pub fn static_atan2_toroidal_f64(out: &mut [f64], y: &[f64], x: &[f64]) -> Result<(), String> {
    if out.len() != y.len() || y.len() != x.len() {
        return Err("All arrays must have same length".to_string());
    }
    let result = unsafe {
        host_atan2_toroidal_f64(out.as_mut_ptr(), y.as_ptr(), x.as_ptr(), out.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked golden entropy kernel
#[cfg(feature = "cuda")]
pub fn static_golden_entropy_f64(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if out.len() != input.len() {
        return Err("Output and input must have same length".to_string());
    }
    let result = unsafe {
        host_golden_entropy_f64(out.as_mut_ptr(), input.as_ptr(), input.len() as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked gather phi weighted kernel
#[cfg(feature = "cuda")]
pub fn static_gather_phi_weighted_f64(
    out: &mut [f64],
    src: &[f64],
    indices: &[i32],
    weights: &[f64],
) -> Result<(), String> {
    if out.len() != indices.len() || indices.len() != weights.len() {
        return Err("Output, indices, and weights must have same length".to_string());
    }
    let result = unsafe {
        host_gather_phi_weighted_f64(
            out.as_mut_ptr(),
            src.as_ptr(),
            indices.as_ptr(),
            weights.as_ptr(),
            indices.len() as i32,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked arg_c128 kernel (complex argument)
#[cfg(feature = "cuda")]
pub fn static_arg_c128(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if input.len() % 2 != 0 {
        return Err("Input must have even length (interleaved complex)".to_string());
    }
    let n = input.len() / 2;
    if out.len() != n {
        return Err("Output must be half the length of input".to_string());
    }
    let result = unsafe { host_arg_c128(out.as_mut_ptr(), input.as_ptr(), n as i32) };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked phase syntony kernel for c128
#[cfg(feature = "cuda")]
pub fn static_phase_syntony_c128(out: &mut [f64], input: &[f64]) -> Result<(), String> {
    if input.len() % 2 != 0 {
        return Err("Input must have even length (interleaved complex)".to_string());
    }
    let n = input.len() / 2;
    if out.len() != n {
        return Err("Output must be half the length of input".to_string());
    }
    let result = unsafe { host_phase_syntony_c128(out.as_mut_ptr(), input.as_ptr(), n as i32) };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked trilinear interpolation kernel
#[cfg(feature = "cuda")]
pub fn static_trilinear_f64(
    out: &mut [f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    nx: i32,
    ny: i32,
    nz: i32,
) -> Result<(), String> {
    let count = x.len();
    if count != y.len() || count != z.len() || count != out.len() {
        return Err("All arrays must have same length".to_string());
    }
    let result = unsafe {
        host_trilinear_f64(out.as_mut_ptr(), x.as_ptr(), y.as_ptr(), z.as_ptr(), nx, ny, nz, count as i32)
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked gnosis mask kernel
#[cfg(feature = "cuda")]
pub fn static_gnosis_mask_f64(
    out: &mut [f64],
    input: &[f64],
    syntony: &[f64],
    threshold: f64,
    strength: f64,
) -> Result<(), String> {
    if out.len() != input.len() || input.len() != syntony.len() {
        return Err("All arrays must have same length".to_string());
    }
    let result = unsafe {
        host_gnosis_mask_f64(
            out.as_mut_ptr(),
            input.as_ptr(),
            syntony.as_ptr(),
            input.len() as i32,
            threshold,
            strength,
        )
    };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Call statically linked matrix multiply kernel
#[cfg(feature = "cuda")]
pub fn static_matmul_f64(c: &mut [f64], a: &[f64], b: &[f64], m: i32, n: i32, k: i32) -> Result<(), String> {
    if a.len() != (m * k) as usize {
        return Err(format!("A has wrong size: {} expected {}", a.len(), m * k));
    }
    if b.len() != (k * n) as usize {
        return Err(format!("B has wrong size: {} expected {}", b.len(), k * n));
    }
    if c.len() != (m * n) as usize {
        return Err(format!("C has wrong size: {} expected {}", c.len(), m * n));
    }
    let result = unsafe { host_matmul_f64(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, n, k) };
    if result != 0 {
        return Err(format!("CUDA kernel failed with code {}", result));
    }
    Ok(())
}

/// Run the Phase-State compile cycle loop on GPU.
///
/// All data stays in VRAM between cycles. Returns (gnosis_reached, cycles_used).
///
/// # Arguments
/// * `m4` - Mutable i8 slice for real components (-1, 0, 1)
/// * `t4` - Mutable i8 slice for imaginary components (-1, 0, 1)
/// * `is_source` - Immutable i8 slice marking source nodes
/// * `stale_cycles` - Mutable i32 slice for per-node staleness
/// * `recursive_depth` - Mutable i32 slice for per-node recursion depth
/// * `rows`, `cols` - Grid dimensions (n = rows * cols)
/// * `stale_threshold` - Cycles before recursion triggers
/// * `k_threshold` - Kissing number for gnosis (typically 24)
/// * `allow_novelty` - Enable majority-vote novelty injection
/// * `max_cycles` - Maximum compile cycles
#[cfg(feature = "cuda")]
pub fn static_phase_state_run_2d(
    m4: &mut [i8],
    t4: &mut [i8],
    is_source: &[i8],
    stale_cycles: &mut [i32],
    recursive_depth: &mut [i32],
    rows: i32,
    cols: i32,
    stale_threshold: i32,
    k_threshold: i32,
    allow_novelty: bool,
    max_cycles: i32,
) -> Result<(bool, i32), String> {
    let n = (rows * cols) as usize;
    if m4.len() != n || t4.len() != n || is_source.len() != n
        || stale_cycles.len() != n || recursive_depth.len() != n
    {
        return Err(format!(
            "All arrays must have length {} (rows={} * cols={})",
            n, rows, cols
        ));
    }
    let mut gnosis_out: i32 = 0;
    let mut cycles_used: i32 = 0;
    let result = unsafe {
        host_phase_state_run_2d(
            m4.as_mut_ptr(),
            t4.as_mut_ptr(),
            is_source.as_ptr(),
            stale_cycles.as_mut_ptr(),
            recursive_depth.as_mut_ptr(),
            rows,
            cols,
            stale_threshold,
            k_threshold,
            if allow_novelty { 1 } else { 0 },
            max_cycles,
            &mut gnosis_out,
            &mut cycles_used,
        )
    };
    if result != 0 {
        return Err(format!("CUDA phase-state kernel failed with code {}", result));
    }
    Ok((gnosis_out != 0, cycles_used))
}

/// Validate that listed SRT kernels are present in the PTX modules for a device.
#[pyfunction]
pub fn validate_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    let device = get_device(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    validate_kernels_for_device(&device)
}

fn validate_kernels_for_device(device: &Arc<CudaDevice>) -> PyResult<Vec<String>> {
    use cudarc::nvrtc::Ptx;

    let (major, minor) = get_compute_capability(device);
    let mut missing = Vec::new();

    let mut check_module = |ptx_src: &str, label: &str, funcs: &[&str]| -> PyResult<()> {
        let module = device.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load {} PTX: {}",
                label, e
            ))
        })?;

        for &func_name in funcs {
            if module.load_function(func_name).is_err() {
                missing.push(format!("{}: {}", label, func_name));
            }
        }

        Ok(())
    };

    check_module(select_golden_ptx(major, minor), "golden", GOLDEN_FUNCS)?;
    check_module(select_e8_ptx(major, minor), "e8", E8_FUNCS)?;
    check_module(select_heat_ptx(major, minor), "heat", HEAT_FUNCS)?;
    check_module(select_dhsr_ptx(major, minor), "dhsr", DHSR_FUNCS)?;
    check_module(
        select_dhsr_ptx(major, minor),
        "dhsr_gravity",
        DHSR_GRAVITY_FUNCS,
    )?;
    check_module(select_corr_ptx(major, minor), "correction", CORR_FUNCS)?;
    check_module(
        select_resonant_ptx(major, minor),
        "resonant",
        RESONANT_FUNCS,
    )?;
    check_module(
        select_phi_residual_ptx(major, minor),
        "phi_residual",
        PHI_RESIDUAL_FUNCS,
    )?;
    check_module(
        select_golden_batch_norm_ptx(major, minor),
        "golden_batch_norm",
        GOLDEN_BATCH_NORM_FUNCS,
    )?;
    check_module(
        select_syntonic_softmax_ptx(major, minor),
        "syntonic_softmax",
        SYNTONIC_SOFTMAX_FUNCS,
    )?;
    check_module(select_matmul_ptx(major, minor), "matmul", MATMUL_FUNCS)?;
    check_module(
        select_hierarchy_ptx(major, minor),
        "hierarchy",
        HIERARCHY_FUNCS,
    )?;
    check_module(
        select_golden_gelu_ptx(major, minor),
        "golden_gelu",
        GOLDEN_GELU_FUNCS,
    )?;
    check_module(
        select_prime_selection_ptx(major, minor),
        "prime_selection",
        PRIME_SELECTION_FUNCS,
    )?;
    check_module(
        select_prime_ops_ptx(major, minor),
        "prime_ops",
        PRIME_OPS_FUNCS,
    )?;
    check_module(select_gnosis_ptx(major, minor), "gnosis", GNOSIS_FUNCS)?;
    check_module(
        select_winding_ops_ptx(major, minor),
        "winding_ops",
        WINDING_OPS_FUNCS,
    )?;
    check_module(
        select_elementwise_ptx(major, minor),
        "elementwise",
        ELEMENTWISE_FUNCS,
    )?;
    check_module(
        select_core_ops_ptx(major, minor),
        "core_ops",
        CORE_OPS_FUNCS,
    )?;
    check_module(
        select_conv_ops_ptx(major, minor),
        "conv_ops",
        CONV_OPS_FUNCS,
    )?;

    Ok(missing)
}

// =============================================================================
// SRT Constants
// =============================================================================

// Canonical constants — derived from exact math in crate::constants
pub use crate::constants::PHI;
pub use crate::constants::PHI_INV;
pub use crate::constants::Q_DEFICIT_CONST as Q_DEFICIT;

/// Structure indices for correction factors
pub mod structure {
    pub const E8_DIM: i32 = 0; // 248
    pub const E8_ROOTS: i32 = 1; // 240
    pub const E8_POS: i32 = 2; // 120
    pub const E6_DIM: i32 = 3; // 78
    pub const E6_CONE: i32 = 4; // 36
    pub const E6_27: i32 = 5; // 27
    pub const D4_KISSING: i32 = 6; // 24
    pub const G2_DIM: i32 = 7; // 14
}

// =============================================================================
// Launch Configuration Helpers
// =============================================================================

/// Standard launch configuration for n elements (block size 256)
pub fn launch_cfg_256(n: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch configuration for E₈ operations (block size 240)
pub fn launch_cfg_e8(n: usize) -> LaunchConfig {
    let block_size = 240u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch configuration with shared memory for reductions
pub fn launch_cfg_reduce(n: usize, elem_size: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: (block_size as u32) * (elem_size as u32),
    }
}

// =============================================================================
// High-Level Operation Wrappers
// =============================================================================

/// Scale array by golden ratio φ
pub fn cuda_scale_phi_f64(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load golden_ops kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("scale_phi_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute golden gaussian weights for 8D vectors: w(λ) = exp(-|λ|²/φ)
pub fn cuda_golden_gaussian_8d_f64(
    device: &Arc<CudaDevice>,
    vectors: &CudaSlice<f64>, // count × 8 flattened
    weights: &mut CudaSlice<f64>,
    count: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load golden_ops kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("golden_gaussian_weight_8d_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(weights)
            .arg(vectors)
            .arg(&(count as i32))
            .launch(launch_cfg_256(count))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batch E₈ projection with Q values and cone test
pub fn cuda_e8_batch_projection_f64(
    device: &Arc<CudaDevice>,
    roots: &CudaSlice<f64>,             // count × 8
    proj_parallel: &mut CudaSlice<f64>, // count × 4
    proj_perp: &mut CudaSlice<f64>,     // count × 4
    q_values: &mut CudaSlice<f64>,      // count
    in_cone: &mut CudaSlice<i32>,       // count
    count: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_e8_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load e8_projection kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("e8_batch_projection_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(proj_parallel)
            .arg(proj_perp)
            .arg(q_values)
            .arg(in_cone)
            .arg(roots)
            .arg(&(count as i32))
            .launch(launch_cfg_e8(count))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute theta series sum: Θ(t) = Σ_λ w(λ) exp(-π Q(λ) / t)
pub fn cuda_theta_series_f64(
    device: &Arc<CudaDevice>,
    q_values: &CudaSlice<f64>,
    in_cone: &CudaSlice<i32>,
    weights: Option<&CudaSlice<f64>>,
    t: f64,
    count: usize,
) -> PyResult<f64> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_heat_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load heat_kernel kernels: {}",
                e
            ))
        })?;

    // Allocate result on device
    let mut result: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("theta_series_sum_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(count, std::mem::size_of::<f64>());

    // Handle optional weights pointer
    let weights_ptr: u64 = match weights {
        Some(w) => w.device_ptr(&device.default_stream()).0,
        None => 0u64, // null pointer
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut result)
            .arg(q_values)
            .arg(in_cone)
            .arg(&weights_ptr)
            .arg(&t)
            .arg(&(count as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Copy result back
    let mut host_result = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&result, &mut host_result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(host_result[0])
}

/// Compute syntony metric S(ψ)
pub fn cuda_compute_syntony_c128(
    device: &Arc<CudaDevice>,
    psi: &CudaSlice<CudaComplex64>, // Interleaved complex [re, im, ...]
    mode_norm_sq: &CudaSlice<f64>,  // |n|² for each mode
    n: usize,
) -> PyResult<f64> {
    // Verify precision policy - syntony requires Exact precision
    let policy = srt_operation_policy("syntony");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "Syntony is an SRT operation requiring Exact precision"
    );

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let mut numerator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("compute_syntony_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut numerator)
            .arg(&mut denominator)
            .arg(psi)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(0.0)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

/// Apply DHSR cycle in-place
pub fn cuda_dhsr_cycle_inplace_c128(
    device: &Arc<CudaDevice>,
    psi: &mut CudaSlice<CudaComplex64>, // In/out: interleaved complex
    mode_norm_sq: &CudaSlice<f64>,
    syntony: f64,
    n: usize,
) -> PyResult<f64> {
    // Verify precision policy - DHSR operations require Exact precision
    let policy = srt_operation_policy("recurse");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "DHSR cycle is an SRT operation requiring Exact precision"
    );

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let mut new_num: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut new_den: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("dhsr_cycle_inplace_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(psi)
            .arg(mode_norm_sq)
            .arg(&syntony)
            .arg(&mut new_num)
            .arg(&mut new_den)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&new_num, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&new_den, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(syntony)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

// =============================================================================
// Accelerated DHSR Operators (NEW)
// =============================================================================

/// Batch Fourier projection with weights: Σᵢ αᵢ P̂ᵢ[Ψ]
/// - out: output complex array (interleaved [re0, im0, re1, im1, ...])
/// - in_data: input complex array
/// - modes: array of mode indices
/// - weights: αᵢ(S) weights for each mode
/// - num_modes: number of modes to project
/// - size: number of complex elements
pub fn cuda_fourier_project_batch_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    modes: &CudaSlice<i32>,
    weights: &CudaSlice<f64>,
    num_modes: usize,
    size: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("fourier_project_batch_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(modes)
            .arg(weights)
            .arg(&(num_modes as i32))
            .arg(&(size as i32))
            .launch(launch_cfg_256(size))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// 1D Discrete Laplacian with periodic boundary: ∇²[Ψ]ᵢ = Ψᵢ₋₁ - 2Ψᵢ + Ψᵢ₊₁
/// - out: output complex array
/// - in_data: input complex array
/// - size: number of complex elements
pub fn cuda_laplacian_1d_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    size: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("laplacian_1d_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(&(size as i32))
            .launch(launch_cfg_256(size))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Full Differentiation Operator: D̂[Ψ] = Ψ + α(S) Σᵢ P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]
/// - out: output complex array
/// - in_data: input complex array
/// - fourier_contribution: pre-computed Σᵢ αᵢ P̂ᵢ[Ψ]
/// - laplacian: pre-computed ∇²[Ψ]
/// - alpha_0, zeta_0: base coefficients
/// - syntony: current syntony value [0, 1]
/// - size: number of complex elements
pub fn cuda_differentiation_full_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    fourier_contribution: &CudaSlice<f64>,
    laplacian: &CudaSlice<f64>,
    alpha_0: f64,
    zeta_0: f64,
    syntony: f64,
    size: usize,
) -> PyResult<()> {
    // Verify precision policy - differentiation requires Exact precision
    let policy = srt_operation_policy("differentiate");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "Differentiation is an SRT operation requiring Exact precision"
    );

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("differentiation_full_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(fourier_contribution)
            .arg(laplacian)
            .arg(&alpha_0)
            .arg(&zeta_0)
            .arg(&syntony)
            .arg(&(size as i32))
            .launch(launch_cfg_256(size))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Damping Cascade: Ĥ high-frequency damping with syntony-dependent decay
/// - out: output complex array
/// - in_data: input complex array
/// - mode_norm_sq: |n|² for each mode
/// - beta_0: base damping coefficient
/// - syntony: current syntony value
/// - delta_d: differentiation magnitude
/// - num_dampers: number of damping levels
/// - size: number of complex elements
pub fn cuda_damping_cascade_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    beta_0: f64,
    syntony: f64,
    delta_d: f64,
    num_dampers: usize,
    size: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("damping_cascade_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(mode_norm_sq)
            .arg(&beta_0)
            .arg(&syntony)
            .arg(&delta_d)
            .arg(&(num_dampers as i32))
            .arg(&(size as i32))
            .launch(launch_cfg_256(size))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Syntony Projection: Project toward syntony-promoting target
/// - out: output complex array
/// - in_data: input complex array
/// - target: normalized mean (syntony-promoting target)
/// - gamma: projection strength coefficient
/// - syntony: current syntony value
/// - size: number of complex elements
pub fn cuda_syntony_projection_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    target: &CudaSlice<f64>,
    gamma: f64,
    syntony: f64,
    size: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("syntony_projection_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(target)
            .arg(&gamma)
            .arg(&syntony)
            .arg(&(size as i32))
            .launch(launch_cfg_256(size))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Fused DHSR Step: Single kernel for D̂→Ĥ with syntony recomputation
/// Returns the new syntony value after the step
/// - out: output complex array
/// - in_data: input complex array
/// - mode_norm_sq: |n|² for each mode
/// - alpha_0, zeta_0: D̂ parameters
/// - beta_0, gamma_0: Ĥ parameters
/// - syntony: current syntony value
/// - size: number of complex elements
pub fn cuda_dhsr_step_fused_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    in_data: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    alpha_0: f64,
    zeta_0: f64,
    beta_0: f64,
    gamma_0: f64,
    syntony: f64,
    size: usize,
) -> PyResult<f64> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    // Allocate syntony accumulators
    let mut new_num: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut new_den: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("dhsr_step_fused_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    // Shared memory for reduction: 2 * blockDim.x * sizeof(double)
    let cfg = launch_cfg_reduce(size, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(in_data)
            .arg(mode_norm_sq)
            .arg(&alpha_0)
            .arg(&zeta_0)
            .arg(&beta_0)
            .arg(&gamma_0)
            .arg(&syntony)
            .arg(&mut new_num)
            .arg(&mut new_den)
            .arg(&(size as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Read back syntony
    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&new_num, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&new_den, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(syntony)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

/// Apply SRT correction factor: value × (1 + sign × q / N)
/// structure_idx should be one of the structure:: constants:
///   E8_DIM(0), E8_ROOTS(1), E8_POS(2), E6_DIM(3), E6_CONE(4), E6_27(5), D4_KISSING(6), G2_DIM(7)
pub fn cuda_apply_correction_f64(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
    structure_idx: i32,
    sign: i32,
    n: usize,
) -> PyResult<()> {
    // Validate structure index using structure constants
    let valid_idx = matches!(
        structure_idx,
        structure::E8_DIM
            | structure::E8_ROOTS
            | structure::E8_POS
            | structure::E6_DIM
            | structure::E6_CONE
            | structure::E6_27
            | structure::D4_KISSING
            | structure::G2_DIM
    );
    if !valid_idx {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid structure index {}. Use structure::E8_DIM, E8_ROOTS, etc.",
            structure_idx
        )));
    }

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_corr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load corrections kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("apply_correction_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&structure_idx)
            .arg(&sign)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Matrix Multiplication CUDA Operations
// =============================================================================

/// Standard matrix multiplication: C = A × B (f64)
pub fn cuda_matmul_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Tiled matrix multiplication for better performance: C = A × B (f64)
pub fn cuda_matmul_tiled_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    println!(
        "DEBUG: cuda_matmul_tiled_f64 called with m={}, n={}, k={}",
        m, n, k
    );

    let (major, minor) = get_compute_capability(device);
    println!("DEBUG: CUDA compute capability: {}.{}", major, minor);

    let ptx_src = select_matmul_ptx(major, minor);
    println!("DEBUG: Selected PTX length: {}", ptx_src.len());

    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_src))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;
    println!("DEBUG: Module loaded successfully");

    let func = module.load_function("matmul_tiled_f64").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel matmul_tiled_f64 not found")
    })?;
    println!("DEBUG: Function matmul_tiled_f64 loaded successfully");

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);
    println!(
        "DEBUG: Launch config: block_dim={:?}, grid_dim={:?}",
        block_dim, grid_dim
    );

    let launch_result = unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    };

    match &launch_result {
        Ok(_) => println!("DEBUG: Kernel launch succeeded"),
        Err(e) => println!("DEBUG: Kernel launch failed: {}", e),
    }

    launch_result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Synchronize to ensure kernel completes
    device.default_stream().synchronize().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Stream sync failed: {}", e))
    })?;
    println!("DEBUG: Stream synchronized successfully");

    Ok(())
}

/// Complex matrix multiplication: C = A × B (complex128)
pub fn cuda_matmul_c128(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<CudaComplex64>,
    a: &CudaSlice<CudaComplex64>,
    b: &CudaSlice<CudaComplex64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = Aᵀ × B (f64)
pub fn cuda_matmul_tn_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_tn_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = Aᵀ × B (f32)
pub fn cuda_matmul_tn_f32(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_tn_f32")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = Aᵀ × B (complex128)
pub fn cuda_matmul_tn_c128(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<CudaComplex64>,
    a: &CudaSlice<CudaComplex64>,
    b: &CudaSlice<CudaComplex64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_tn_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = A × Bᵀ (f64)
pub fn cuda_matmul_nt_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_nt_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
pub fn cuda_golden_commutator_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("golden_commutator_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
pub fn cuda_golden_anticommutator_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("golden_anticommutator_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// φ-scaled matrix multiplication: C = φⁿ × (A × B)
pub fn cuda_matmul_phi_scaled_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    n: i32, // Power of phi
    m: usize,
    k: usize,
    p: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_phi_scaled_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((p + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&n)
            .arg(&(m as i32))
            .arg(&(p as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batched matrix multiplication: C[i] = A[i] × B[i]
pub fn cuda_bmm_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("bmm_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (8, 8, 8);
    let grid_dim = (
        ((n + 7) / 8) as u32,
        ((m + 7) / 8) as u32,
        ((batch_size + 7) / 8) as u32,
    );

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(batch_size as i32))
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batched matrix multiplication with Transposed B: C[i] = A[i] × B[i]ᵀ
pub fn cuda_bmm_nt_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("bmm_nt_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (
        ((n + 15) / 16) as u32,
        ((m + 15) / 16) as u32,
        batch_size as u32,
    );

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(batch_size as i32))
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Complex division: C = A / B (element-wise)
pub fn cuda_complex_div_c128(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<CudaComplex64>,
    a: &CudaSlice<CudaComplex64>,
    b: &CudaSlice<CudaComplex64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("complex_div_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// WMMA matrix multiplication for fp16 tensors
pub fn cuda_wmma_fp16_matmul(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<crate::tensor::cuda::memory_pool::CudaF16>,
    a: &CudaSlice<crate::tensor::cuda::memory_pool::CudaF16>,
    b: &CudaSlice<crate::tensor::cuda::memory_pool::CudaF16>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_wmma_syntonic_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load WMMA kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_wmma_syntonic_fp16")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load WMMA function: {}",
                e
            ))
        })?;

    let cfg = LaunchConfig {
        grid_dim: (m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .arg(&(false as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Select WMMA syntonic PTX based on compute capability
fn select_wmma_syntonic_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_WMMA_SYNTONIC_SM90,
        8 => {
            if minor >= 6 {
                PTX_WMMA_SYNTONIC_SM86
            } else {
                PTX_WMMA_SYNTONIC_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_WMMA_SYNTONIC_SM75
            } else {
                PTX_WMMA_SYNTONIC_SM75 // fallback
            }
        }
        _ => PTX_WMMA_SYNTONIC_SM75, // fallback
    }
}

/// SRT Native SGEMM: High-performance register-blocked matrix multiplication
pub fn cuda_sgemm_native_f32(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_sgemm_native_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Native SGEMM kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_sgemm_native_f32")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Native SGEMM function: {}",
                e
            ))
        })?;

    // Grid: (N/BLOCK_N, M/BLOCK_M)
    let grid_x = (n + 127) / 128; // BLOCK_N = 128
    let grid_y = (m + 127) / 128; // BLOCK_M = 128

    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (16, 16, 1), // 256 threads
        shared_mem_bytes: 0,    // Shared memory allocated in kernel
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// SRT Native DGEMM: High-performance double-precision matrix multiplication
pub fn cuda_dgemm_native_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dgemm_native_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Native DGEMM kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_dgemm_native_f64")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Native DGEMM function: {}",
                e
            ))
        })?;

    // Grid: (N/BLOCK_N, M/BLOCK_M) with BLOCK_M=64, BLOCK_N=64
    let grid_x = (n + 63) / 64;
    let grid_y = (m + 63) / 64;

    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (8, 16, 1), // 128 threads
        shared_mem_bytes: 0,   // Shared memory allocated in kernel
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Select SRT Native SGEMM PTX based on compute capability
fn select_sgemm_native_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_SGEMM_NATIVE_SM90,
        8 => {
            if minor >= 6 {
                PTX_SGEMM_NATIVE_SM86
            } else {
                PTX_SGEMM_NATIVE_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_SGEMM_NATIVE_SM75
            } else {
                PTX_SGEMM_NATIVE_SM75 // fallback
            }
        }
        _ => PTX_SGEMM_NATIVE_SM75, // fallback
    }
}

/// Select SRT Native DGEMM PTX based on compute capability
fn select_dgemm_native_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_DGEMM_NATIVE_SM90,
        8 => {
            if minor >= 6 {
                PTX_DGEMM_NATIVE_SM86
            } else {
                PTX_DGEMM_NATIVE_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_DGEMM_NATIVE_SM75
            } else {
                PTX_DGEMM_NATIVE_SM75 // fallback
            }
        }
        _ => PTX_DGEMM_NATIVE_SM75, // fallback
    }
}

/// SRT Scatter/Gather operations: Theory-aligned index operations
pub fn cuda_gather_phi_weighted_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Scatter/Gather kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_gather_phi_weighted_f64")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load gather function: {}",
                e
            ))
        })?;

    let cfg = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// SRT Scatter/Gather: Basic gather operation (no weighting)
pub fn cuda_gather_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_f64")
        .map_err(|_| "Kernel gather_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Basic scatter operation
pub fn cuda_scatter_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_f64")
        .map_err(|_| "Kernel scatter_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter add operation (adds to existing values)
pub fn cuda_scatter_add_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_add_f64")
        .map_err(|_| "Kernel scatter_add_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter add operation (f32)
pub fn cuda_scatter_add_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    src: &CudaSlice<f32>,
    idx: &CudaSlice<i32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_add_f32")
        .map_err(|_| "Kernel scatter_add_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter with golden ratio weighting
pub fn cuda_scatter_golden_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_golden_f64")
        .map_err(|_| "Kernel scatter_golden_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter with Mersenne stable precision
pub fn cuda_scatter_mersenne_stable_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_mersenne_stable_f64")
        .map_err(|_| "Kernel scatter_mersenne_stable_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Gather with Lucas shadow weighting
pub fn cuda_gather_lucas_shadow_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_lucas_shadow_f64")
        .map_err(|_| "Kernel gather_lucas_shadow_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Gather with Pisano hooked weighting
pub fn cuda_gather_pisano_hooked_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_pisano_hooked_f64")
        .map_err(|_| "Kernel gather_pisano_hooked_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Gather with E8 roots weighting
pub fn cuda_gather_e8_roots_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_e8_roots_f64")
        .map_err(|_| "Kernel gather_e8_roots_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter with golden cone weighting
pub fn cuda_scatter_golden_cone_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_golden_cone_f64")
        .map_err(|_| "Kernel scatter_golden_cone_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Gather with transcendence gate weighting
pub fn cuda_gather_transcendence_gate_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_transcendence_gate_f64")
        .map_err(|_| "Kernel gather_transcendence_gate_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Scatter with consciousness threshold
pub fn cuda_scatter_consciousness_threshold_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    src: &CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_consciousness_threshold_f64")
        .map_err(|_| "Kernel scatter_consciousness_threshold_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Basic gather operation (f32)
pub fn cuda_gather_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    src: &CudaSlice<f32>,
    idx: &CudaSlice<i32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("gather_f32")
        .map_err(|_| "Kernel gather_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Scatter/Gather: Basic scatter operation (f32)
pub fn cuda_scatter_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    src: &CudaSlice<f32>,
    idx: &CudaSlice<i32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let func = module
        .load_function("scatter_f32")
        .map_err(|_| "Kernel scatter_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(idx)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Select SRT Scatter/Gather PTX based on compute capability
fn select_scatter_gather_srt_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_SCATTER_GATHER_SRT_SM90,
        8 => {
            if minor >= 6 {
                PTX_SCATTER_GATHER_SRT_SM86
            } else {
                PTX_SCATTER_GATHER_SRT_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_SCATTER_GATHER_SRT_SM75
            } else {
                PTX_SCATTER_GATHER_SRT_SM75 // fallback
            }
        }
        _ => PTX_SCATTER_GATHER_SRT_SM75, // fallback
    }
}

/// SRT Trilinear Interpolation: Theory-aligned grid sampling
pub fn cuda_trilinear_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    grid: &CudaSlice<f64>,   // [D, H, W]
    coords: &CudaSlice<f64>, // [N, 3] (x, y, z)
    d: usize,
    h: usize,
    w: usize,
    n: usize,
    boundary_mode: i32,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Trilinear kernel: {}",
                e
            ))
        })?;

    let func = module.load_function("launch_trilinear_f64").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load trilinear function: {}",
            e
        ))
    })?;

    let cfg = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(grid)
            .arg(coords)
            .arg(&(d as i32))
            .arg(&(h as i32))
            .arg(&(w as i32))
            .arg(&(n as i32))
            .arg(&boundary_mode)
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Select SRT Trilinear PTX based on compute capability
fn select_trilinear_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_TRILINEAR_SM90,
        8 => {
            if minor >= 6 {
                PTX_TRILINEAR_SM86
            } else {
                PTX_TRILINEAR_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_TRILINEAR_SM75
            } else {
                PTX_TRILINEAR_SM75 // fallback
            }
        }
        _ => PTX_TRILINEAR_SM75, // fallback
    }
}

/// SRT Complex Operations: Theory-aligned complex arithmetic
pub fn cuda_arg_c128(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    z: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_complex_ops_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Complex Ops kernel: {}",
                e
            ))
        })?;

    let func = module.load_function("launch_arg_c128").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load arg function: {}",
            e
        ))
    })?;

    let cfg = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(z)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// SRT Phase Syntony Operation: z * e^{iπS} (i≈π postulate)
pub fn cuda_phase_syntony_c128(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    z: &CudaSlice<f64>,
    syntony: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_complex_ops_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT Complex Ops kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_phase_syntony_c128")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load phase_syntony function: {}",
                e
            ))
        })?;

    let cfg = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(z)
            .arg(syntony)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Select SRT Complex Ops PTX based on compute capability
fn select_complex_ops_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_COMPLEX_OPS_SM90,
        8 => {
            if minor >= 6 {
                PTX_COMPLEX_OPS_SM86
            } else {
                PTX_COMPLEX_OPS_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_COMPLEX_OPS_SM75
            } else {
                PTX_COMPLEX_OPS_SM75 // fallback
            }
        }
        _ => PTX_COMPLEX_OPS_SM75, // fallback
    }
}

/// Select SRT Attention PTX based on compute capability
fn select_attention_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_ATTENTION_SM90,
        8 => {
            if minor >= 6 {
                PTX_ATTENTION_SM86
            } else {
                PTX_ATTENTION_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_ATTENTION_SM75
            } else {
                PTX_ATTENTION_SM75 // fallback
            }
        }
        _ => PTX_ATTENTION_SM75, // fallback
    }
}

/// SRT Theory-Correct Reductions: Mersenne-stable sum
pub fn cuda_reduce_sum_mersenne_stable_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load SRT reduction kernel: {}",
                e
            ))
        })?;

    let func = module
        .load_function("launch_reduce_sum_mersenne_stable_f64")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load reduction function: {}",
                e
            ))
        })?;

    let cfg = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// SRT Reduction: Sum reduction (basic)
pub fn cuda_reduce_sum_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_f64")
        .map_err(|_| "Kernel reduce_sum_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Mean reduction
pub fn cuda_reduce_mean_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_mean_f64")
        .map_err(|_| "Kernel reduce_mean_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Mean reduction (f32)
pub fn cuda_reduce_mean_f32(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_mean_f32")
        .map_err(|_| "Kernel reduce_mean_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Max reduction
pub fn cuda_reduce_max_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_max_f64")
        .map_err(|_| "Kernel reduce_max_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Max reduction (f32)
pub fn cuda_reduce_max_f32(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_max_f32")
        .map_err(|_| "Kernel reduce_max_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Min reduction
pub fn cuda_reduce_min_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_min_f64")
        .map_err(|_| "Kernel reduce_min_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Min reduction (f32)
pub fn cuda_reduce_min_f32(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_min_f32")
        .map_err(|_| "Kernel reduce_min_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum reduction (f32)
pub fn cuda_reduce_sum_f32(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_f32")
        .map_err(|_| "Kernel reduce_sum_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: L2 norm reduction (f64)
pub fn cuda_reduce_norm_l2_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_norm_l2_f64")
        .map_err(|_| "Kernel reduce_norm_l2_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: L2 norm reduction (f32)
pub fn cuda_reduce_norm_l2_f32(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_norm_l2_f32")
        .map_err(|_| "Kernel reduce_norm_l2_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum with golden weighted reduction (f64)
pub fn cuda_reduce_sum_golden_weighted_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_golden_weighted_f64")
        .map_err(|_| "Kernel reduce_sum_golden_weighted_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Syntony reduction (f64)
pub fn cuda_reduce_syntony_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_syntony_f64")
        .map_err(|_| "Kernel reduce_syntony_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum rows reduction (f64)
pub fn cuda_reduce_sum_rows_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_rows_f64")
        .map_err(|_| "Kernel reduce_sum_rows_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum columns reduction (f64)
pub fn cuda_reduce_sum_cols_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_cols_f64")
        .map_err(|_| "Kernel reduce_sum_cols_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum with phi scaled reduction (f64)
pub fn cuda_reduce_sum_phi_scaled_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_phi_scaled_f64")
        .map_err(|_| "Kernel reduce_sum_phi_scaled_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Variance with golden target reduction (f64)
pub fn cuda_reduce_variance_golden_target_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_variance_golden_target_f64")
        .map_err(|_| "Kernel reduce_variance_golden_target_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum with Lucas shadow reduction (f64)
pub fn cuda_reduce_sum_lucas_shadow_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_lucas_shadow_f64")
        .map_err(|_| "Kernel reduce_sum_lucas_shadow_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Syntony deviation reduction (f64)
pub fn cuda_reduce_syntony_deviation_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_syntony_deviation_f64")
        .map_err(|_| "Kernel reduce_syntony_deviation_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Consciousness count reduction (f64)
pub fn cuda_reduce_consciousness_count_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_consciousness_count_f64")
        .map_err(|_| "Kernel reduce_consciousness_count_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum with Q corrected reduction (f64)
pub fn cuda_reduce_sum_q_corrected_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_q_corrected_f64")
        .map_err(|_| "Kernel reduce_sum_q_corrected_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: E8 norm reduction (f64)
pub fn cuda_reduce_e8_norm_f64(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_e8_norm_f64")
        .map_err(|_| "Kernel reduce_e8_norm_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Sum reduction (c128)
pub fn cuda_reduce_sum_c128(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<CudaComplex64>,
    input: &CudaSlice<CudaComplex64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_sum_c128")
        .map_err(|_| "Kernel reduce_sum_c128 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// SRT Reduction: Norm reduction (c128)
pub fn cuda_reduce_norm_c128(
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<f64>,
    input: &CudaSlice<CudaComplex64>,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let func = module
        .load_function("reduce_norm_c128")
        .map_err(|_| "Kernel reduce_norm_c128 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Select SRT reduction PTX based on compute capability
fn select_reduction_ptx(major: i32, minor: i32) -> &'static str {
    match major {
        9 => PTX_REDUCTION_SM90,
        8 => {
            if minor >= 6 {
                PTX_REDUCTION_SM86
            } else {
                PTX_REDUCTION_SM80
            }
        }
        7 => {
            if minor >= 5 {
                PTX_REDUCTION_SM75
            } else {
                PTX_REDUCTION_SM75 // fallback
            }
        }
        _ => PTX_REDUCTION_SM75, // fallback
    }
}

// =============================================================================
// CPU Fallback Implementations (when CUDA not available)
// =============================================================================

/// CPU implementation of golden gaussian weight
pub fn cpu_golden_gaussian_8d_f64(vectors: &[f64], weights: &mut [f64]) {
    let count = weights.len();
    for i in 0..count {
        let idx = i * 8;
        let mut norm_sq = 0.0;
        for j in 0..8 {
            let v = vectors[idx + j];
            norm_sq += v * v;
        }
        weights[i] = (-norm_sq * PHI_INV).exp();
    }
}

/// CPU implementation of correction factor
pub fn cpu_correction_factor(structure_n: i32, sign: i32) -> f64 {
    1.0 + (sign as f64) * Q_DEFICIT / (structure_n as f64)
}

/// Get structure dimension by index
pub fn get_structure_dimension(idx: i32) -> i32 {
    match idx {
        0 => 248, // E₈ dim
        1 => 240, // E₈ roots
        2 => 120, // E₈ positive roots
        3 => 78,  // E₆ dim
        4 => 36,  // E₆ golden cone
        5 => 27,  // E₆ 27-rep
        6 => 24,  // D₄ kissing
        7 => 14,  // G₂ dim
        _ => 1,
    }
}

// =============================================================================
// Resonant Engine CUDA Operations
// =============================================================================

/// Execute D-phase: lattice → flux with stochastic noise
pub fn cuda_resonant_d_phase_f64(
    device: &Arc<CudaDevice>,
    flux: &mut CudaSlice<f64>,     // Output: ephemeral floats
    lattice: &CudaSlice<f64>,      // Input: crystallized values
    mode_norm_sq: &CudaSlice<f64>, // |n|² for each mode
    noise: &CudaSlice<f64>,        // Pre-generated Gaussian noise
    syntony: f64,                  // Current syntony S
    noise_scale: f64,              // Base noise amplitude
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_d_phase_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(flux)
            .arg(lattice)
            .arg(mode_norm_sq)
            .arg(noise)
            .arg(&syntony)
            .arg(&noise_scale)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Execute batch D-phase for RES population
pub fn cuda_resonant_d_phase_batch_f64(
    device: &Arc<CudaDevice>,
    flux_batch: &mut CudaSlice<f64>,
    lattice_batch: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    noise_batch: &CudaSlice<f64>,
    syntonies: &CudaSlice<f64>,
    noise_scale: f64,
    n: usize,
    pop_size: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_d_phase_batch_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let total = n * pop_size;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(flux_batch)
            .arg(lattice_batch)
            .arg(mode_norm_sq)
            .arg(noise_batch)
            .arg(syntonies)
            .arg(&noise_scale)
            .arg(&(n as i32))
            .arg(&(pop_size as i32))
            .launch(launch_cfg_256(total))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute syntony on flux values
pub fn cuda_resonant_compute_syntony_f64(
    device: &Arc<CudaDevice>,
    flux: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    n: usize,
) -> PyResult<f64> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let mut numerator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("resonant_compute_syntony_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut numerator)
            .arg(&mut denominator)
            .arg(flux)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(0.0)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

// =============================================================================
// Phi-Residual CUDA Kernels
// =============================================================================

/// Execute phi-residual connection on GPU
pub fn cuda_phi_residual_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    identity: &CudaSlice<f64>,
    residual: &CudaSlice<f64>,
    mode: crate::resonant::phi_ops::PhiResidualMode,
) -> Result<(), String> {
    use crate::resonant::phi_ops::PhiResidualMode;

    let n = out.len();

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_phi_residual_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load phi_residual kernels: {}", e))?;

    // Select kernel based on mode
    let kernel_name = match mode {
        PhiResidualMode::Phi => "phi_residual_mode_phi_f64",
        PhiResidualMode::PhiSymmetric => "phi_residual_mode_symmetric_f64",
        PhiResidualMode::Standard => "phi_residual_mode_standard_f64",
    };

    let func = module
        .load_function(kernel_name)
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = launch_cfg_256(n);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(identity)
            .arg(residual)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Execute fused phi-residual + ReLU on GPU
pub fn cuda_phi_residual_relu_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    identity: &CudaSlice<f64>,
    residual: &CudaSlice<f64>,
) -> Result<(), String> {
    let n = out.len();

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_phi_residual_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load phi_residual kernels: {}", e))?;

    let func = module
        .load_function("phi_residual_relu_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = launch_cfg_256(n);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(identity)
            .arg(residual)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Compute snap gradient for directed exploration
pub fn cuda_resonant_snap_gradient_f64(
    device: &Arc<CudaDevice>,
    gradient: &mut CudaSlice<f64>,
    flux: &CudaSlice<f64>,
    lattice: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_weighted_snap_gradient_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(gradient)
            .arg(flux)
            .arg(lattice)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Box-Muller transform: uniform → Gaussian noise
pub fn cuda_resonant_box_muller_f64(
    device: &Arc<CudaDevice>,
    gaussian: &mut CudaSlice<f64>,
    uniform: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_box_muller_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(gaussian)
            .arg(uniform)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Golden Batch Normalization CUDA Kernels
// =============================================================================

/// Golden batch normalization for 1D tensors (batch, features)
pub fn cuda_golden_bn_1d_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    gamma: Option<&CudaSlice<f64>>,
    beta: Option<&CudaSlice<f64>>,
    eps: f64,
    batch_size: i32,
    features: i32,
) -> Result<(), String> {
    let n = (batch_size * features) as usize;
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_batch_norm_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load golden_batch_norm kernels: {}", e))?;

    let func = module
        .load_function("golden_bn_1d_fused_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let gamma_ptr = gamma
        .map(|g| g.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);
    let beta_ptr = beta
        .map(|b| b.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&eps)
            .arg(&batch_size)
            .arg(&features)
            .launch(launch_cfg_256(n))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_golden_bn_2d_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    mean: &CudaSlice<f64>,
    var: &CudaSlice<f64>,
    gamma: Option<&CudaSlice<f64>>,
    beta: Option<&CudaSlice<f64>>,
    eps: f64,
    b: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), String> {
    let n = (b * c * h * w) as usize;
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_batch_norm_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load golden_batch_norm kernels: {}", e))?;

    let func = module
        .load_function("golden_bn_2d_normalize_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let gamma_ptr = gamma
        .map(|g| g.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);
    let beta_ptr = beta
        .map(|b| b.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(mean)
            .arg(var)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&eps)
            .arg(&b)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .launch(launch_cfg_256(n))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Syntonic Softmax CUDA Kernels
// =============================================================================

pub fn cuda_syntonic_softmax_learned_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    mode_norms: &CudaSlice<f64>,
    syntony_scale: f64,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_learned_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f64>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(mode_norms)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_provided_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    syntony: &CudaSlice<f64>,
    syntony_scale: f64,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_provided_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f64>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(syntony)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_learned_strided_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    mode_norms: &CudaSlice<f64>,
    syntony_scale: f64,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_learned_strided_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(mode_norms)
            .arg(&syntony_scale)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_provided_strided_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    syntony: &CudaSlice<f64>,
    syntony_scale: f64,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_provided_strided_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(syntony)
            .arg(&syntony_scale)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Syntonic Softmax CUDA Kernels - F32 Variants
// =============================================================================

pub fn cuda_syntonic_softmax_learned_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    mode_norms: &CudaSlice<f32>,
    syntony_scale: f32,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_learned_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(mode_norms)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_provided_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    syntony: &CudaSlice<f32>,
    syntony_scale: f32,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_provided_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(syntony)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_learned_strided_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    mode_norms: &CudaSlice<f32>,
    syntony_scale: f32,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_learned_strided_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(mode_norms)
            .arg(&syntony_scale)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_syntonic_softmax_provided_strided_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    syntony: &CudaSlice<f32>,
    syntony_scale: f32,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_syntonic_softmax_provided_strided_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(syntony)
            .arg(&syntony_scale)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Identity Softmax CUDA Kernels (Standard softmax, no golden weighting)
// =============================================================================

pub fn cuda_softmax_identity_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_softmax_identity_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8 + 256 * 8, // s_max (double) + s_sum (double)
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_softmax_identity_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_softmax_identity_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8 * 2, // s_max and s_sum (double) - parallel reduction for better precision
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_softmax_identity_strided_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_softmax_identity_strided_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn cuda_softmax_identity_strided_f32(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    outer_size: i32,
    dim_size: i32,
    inner_size: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("cuda_softmax_identity_strided_f32")
        .map_err(|_| "Kernel not found".to_string())?;

    let count = (outer_size * inner_size) as usize;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(&outer_size)
            .arg(&dim_size)
            .arg(&inner_size)
            .launch(launch_cfg_256(count))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Geodesic Gravity Operations
// =============================================================================

/// Apply Geodesic Gravity Slide to weights (Physical AI update)
pub fn apply_geodesic_gravity_f64(
    device: &Arc<CudaDevice>,
    weights: &CudaSlice<f64>, // Mutable reference in intent, but Arc shared in Python
    attractor: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    gravity: f64,
    temperature: f64,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| format!("Failed to load dhsr kernels: {}", e))?;

    let func = module
        .load_function("apply_geodesic_gravity_f64")
        .map_err(|_| "Kernel apply_geodesic_gravity_f64 not found".to_string())?;

    // IMPORTANT: The kernel processes chunks of 8 (E8 unit cells)
    // We launch threads for *blocks of 8*, not individual elements
    let num_threads = (n + 7) / 8;
    let cfg = launch_cfg_e8(num_threads);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(weights)
            .arg(attractor)
            .arg(mode_norm_sq)
            .arg(&gravity)
            .arg(&temperature)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// SRT Autograd Kernels (Standard Backward Pass)
// =============================================================================

/// Load SRT Autograd kernels for backward pass operations
pub fn load_autograd_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_autograd_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load autograd kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in AUTOGRAD_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

/// Backward pass for element-wise addition
pub fn cuda_backward_add_f32(
    device: &Arc<CudaDevice>,
    grad_output: &CudaSlice<f32>,
    grad_x: &mut CudaSlice<f32>,
    grad_y: &mut CudaSlice<f32>,
    size: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_autograd_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load autograd kernels: {}", e))?;

    let func = module
        .load_function("backward_add_f32")
        .map_err(|_| "Kernel backward_add_f32 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(size as u32);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(grad_output)
            .arg(grad_x)
            .arg(grad_y)
            .arg(&(size as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// SRT Attractor Kernels (Retrocausal Backward Pass)
// =============================================================================

/// Load SRT Attractor kernels for retrocausal operations
pub fn load_attractor_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_attractor_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load attractor kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in ATTRACTOR_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_wmma_syntonic_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_wmma_syntonic_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load wmma_syntonic kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in WMMA_SYNTONIC_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_scatter_gather_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_scatter_gather_srt_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load scatter_gather kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in SCATTER_GATHER_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_reduction_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_reduction_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load reduction kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in REDUCTION_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_trilinear_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_trilinear_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load trilinear kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in TRILINEAR_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_complex_ops_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_complex_ops_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load complex_ops kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in COMPLEX_OPS_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

pub fn load_attention_kernels(
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_attention_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load attention kernels: {}", e))?;

    let mut functions = HashMap::new();
    for &func_name in ATTENTION_FUNCS {
        let func = module
            .load_function(func_name)
            .map_err(|_| format!("Kernel {} not found", func_name))?;
        functions.insert(func_name.to_string(), func);
    }

    Ok(functions)
}

/// Store high-syntony states in attractor memory
pub fn cuda_attractor_memory_update_f64(
    device: &Arc<CudaDevice>,
    attractor_memory: &mut CudaSlice<f64>,
    attractor_syntony: &mut CudaSlice<f64>,
    attractor_count: &mut CudaSlice<i32>,
    state: &CudaSlice<f64>,
    syntony: f64,
    state_dim: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_attractor_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load attractor kernels: {}", e))?;

    let func = module
        .load_function("attractor_memory_update_f64")
        .map_err(|_| "Kernel attractor_memory_update_f64 not found".to_string())?;

    let cfg = LaunchConfig::for_num_elems(1);
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(attractor_memory)
            .arg(attractor_syntony)
            .arg(attractor_count)
            .arg(state)
            .arg(&syntony)
            .arg(&(state_dim as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Fixed-Point (fp64) Kernel Wrappers
// =============================================================================

/// Compute syntony metric S(ψ) using fixed-point arithmetic
pub fn cuda_compute_syntony_fp64(
    device: &Arc<CudaDevice>,
    psi_re: &CudaSlice<i64>,
    psi_im: &CudaSlice<i64>,
    mode_norm_sq: &CudaSlice<i64>,
    n: usize,
) -> PyResult<(i64, i64)> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let mut numerator: CudaSlice<i64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<i64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("compute_syntony_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    // Shared memory: 2 * 256 * sizeof(int64)
    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<i64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut numerator)
            .arg(&mut denominator)
            .arg(psi_re)
            .arg(psi_im)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0i64];
    let mut host_den = [0i64];
    device
        .default_stream()
        .memcpy_dtoh(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((host_num[0], host_den[0]))
}

pub fn cuda_differentiation_fp64(
    device: &Arc<CudaDevice>,
    out_re: &mut CudaSlice<i64>,
    out_im: &mut CudaSlice<i64>,
    in_re: &CudaSlice<i64>,
    in_im: &CudaSlice<i64>,
    mode_norm_sq: &CudaSlice<i64>,
    syntony: i64,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("differentiation_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out_re)
            .arg(out_im)
            .arg(in_re)
            .arg(in_im)
            .arg(mode_norm_sq)
            .arg(&syntony)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_harmonization_fp64(
    device: &Arc<CudaDevice>,
    out_re: &mut CudaSlice<i64>,
    out_im: &mut CudaSlice<i64>,
    in_re: &CudaSlice<i64>,
    in_im: &CudaSlice<i64>,
    mode_norm_sq: &CudaSlice<i64>,
    syntony: i64,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("harmonization_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out_re)
            .arg(out_im)
            .arg(in_re)
            .arg(in_im)
            .arg(mode_norm_sq)
            .arg(&syntony)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_dhsr_cycle_fp64(
    device: &Arc<CudaDevice>,
    out_re: &mut CudaSlice<i64>,
    out_im: &mut CudaSlice<i64>,
    in_re: &CudaSlice<i64>,
    in_im: &CudaSlice<i64>,
    mode_norm_sq: &CudaSlice<i64>,
    syntony: i64,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("dhsr_cycle_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out_re)
            .arg(out_im)
            .arg(in_re)
            .arg(in_im)
            .arg(mode_norm_sq)
            .arg(&syntony)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_fourier_project_batch_fp64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<i64>,
    input: &CudaSlice<i64>,
    modes: &CudaSlice<i32>,
    weights: &CudaSlice<i64>,
    num_modes: usize,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("fourier_project_batch_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(modes)
            .arg(weights)
            .arg(&(num_modes as i32))
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_laplacian_1d_fp64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<i64>,
    input: &CudaSlice<i64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("laplacian_1d_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

pub fn cuda_differentiation_full_fp64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<i64>,
    input: &CudaSlice<i64>,
    fourier: &CudaSlice<i64>,
    laplacian: &CudaSlice<i64>,
    alpha: i64,
    zeta: i64,
    syntony: i64,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("differentiation_full_fp64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(fourier)
            .arg(laplacian)
            .arg(&alpha)
            .arg(&zeta)
            .arg(&syntony)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}
