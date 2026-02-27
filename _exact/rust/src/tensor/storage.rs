//! Tensor storage with CPU and CUDA backends.
//!
//! This module provides NumPy-free tensor operations using the Python buffer protocol
//! for data transfer and optional CUDA acceleration via cudarc.

use ndarray::{ArrayD, Ix1, Ix2, IxDyn};
use ndarray_linalg::{Cholesky, Determinant, Eig, Eigh, Inverse, Solve, Trace, QR, SVD, UPLO};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng;

// Import exact types - all numerical constants derive from these
// Use re-export paths from exact/mod.rs for SymExpr
use crate::exact::constants::FundamentalConstant;
use crate::exact::golden::GoldenExact;
use crate::exact::rational::Rational;
use crate::exact::fixed::FixedPoint64;
use crate::exact::SymExpr;
use crate::exact::syntonic::SyntonicExact;
use crate::linalg::matmul;
use crate::constants::{PHI, PHI_INV, PHI_INV_SQ};

// SRT kernel constants and functions (non-CUDA)
#[cfg(not(feature = "cuda"))]
use super::srt_kernels;

use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::safe::PushKernelArg;
use cudarc::driver::{CudaSlice, LaunchConfig};
use std::sync::Arc;
// Use re-export paths from cuda/mod.rs for CUDA infrastructure
use super::cuda::async_transfer::{AsyncTensorOps, TransferComputeOverlap};
use super::cuda::device_manager::get_srt_protocol;
use super::cuda::device_manager::{
    create_stream, get_device, get_local_manager, get_pool, sync_device,
};
use super::cuda::multi_gpu::MultiGpuInfo;
use super::cuda::{gather, peer_copy, scatter, ReduceOp};
use super::cuda::{AsyncTensorTransfer, AsyncTransfer};
use super::cuda::{CudaComplex64, MemoryPool, PoolConfig, PoolStats, PooledSlice};
use super::cuda::{CudaError, DeviceManager, StreamKind};
use super::srt_kernels;

/// Pre-compiled PTX kernels for different compute capabilities
/// These are compiled offline to ensure driver compatibility
const PTX_SM75: &str = include_str!("../../kernels/ptx/elementwise_sm75.ptx");
const PTX_SM80: &str = include_str!("../../kernels/ptx/elementwise_sm80.ptx");
const PTX_SM86: &str = include_str!("../../kernels/ptx/elementwise_sm86.ptx");
const PTX_SM90: &str = include_str!("../../kernels/ptx/elementwise_sm90.ptx");

use cudarc::driver::CudaFunction;
/// Global cache for CUDA modules and functions to avoid loading PTX repeatedly
use std::collections::HashMap;
use std::sync::RwLock;

lazy_static::lazy_static! {
    /// Cache structure: device_ordinal -> (ptx_hash, module, functions)
    static ref KERNEL_CACHE: RwLock<HashMap<usize, (u64, Arc<cudarc::driver::CudaModule>, HashMap<String, Arc<CudaFunction>>)>> = RwLock::new(HashMap::new());
}

/// Compute hash of PTX source for cache invalidation
fn ptx_hash(ptx: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    ptx.hash(&mut hasher);
    hasher.finish()
}

/// Get or load cached CUDA module and functions for a device
fn get_cached_module_and_functions(
    device: &Arc<CudaDevice>,
    ptx_source: &str,
) -> Result<
    (
        Arc<cudarc::driver::CudaModule>,
        HashMap<String, Arc<CudaFunction>>,
    ),
    cudarc::driver::result::DriverError,
> {
    let device_idx = device.ordinal() as usize;
    let current_hash = ptx_hash(ptx_source);

    // Check cache first
    {
        let cache = KERNEL_CACHE.read().unwrap();
        if let Some((cached_hash, module, functions)) = cache.get(&device_idx) {
            if *cached_hash == current_hash {
                return Ok((module.clone(), functions.clone()));
            }
        }
    }

    // Cache miss - load new module
    let module = device.load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))?;

    // Load all elementwise functions
    let mut functions = HashMap::new();
    let function_names = [
        "add_f32",
        "add_f64",
        "add_c128",
        "sub_f32",
        "sub_f64",
        "sub_c128",
        "mul_f32",
        "mul_f64",
        "mul_c128",
        "div_f32",
        "div_f64",
        "div_c128",
        "neg_f32",
        "neg_f64",
        "neg_c128",
        "abs_f32",
        "abs_f64",
        "exp_f32",
        "exp_f64",
        "exp_golden_f32",
        "exp_golden_f64",
        "log_f32",
        "log_f64",
        "sin_f32",
        "sin_f64",
        "cos_f32",
        "cos_f64",
        "sqrt_f32",
        "sqrt_f64",
        "tanh_f32",
        "tanh_f64",
        "sigmoid_f32",
        "sigmoid_f64",
        "relu_f32",
        "relu_f64",
    ];

    for name in &function_names {
        if let Ok(func) = module.load_function(name) {
            functions.insert(name.to_string(), Arc::new(func));
        }
    }

    let functions_clone = functions.clone();

    // Update cache
    {
        let mut cache = KERNEL_CACHE.write().unwrap();
        cache.insert(
            device_idx,
            (current_hash, module.clone(), functions.clone()),
        );
    }

    Ok((module, functions_clone))
}

/// Get the compute capability for the current device
fn get_device_compute_capability(device: &Arc<CudaDevice>) -> (i32, i32) {
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

/// Select appropriate pre-compiled PTX based on device compute capability
fn select_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_SM90
    } else if cc >= 86 {
        PTX_SM86
    } else if cc >= 80 {
        PTX_SM80
    } else {
        PTX_SM75
    } // Minimum supported (Turing and above)
}

/// Ensure CUDA kernels are loaded for the given device
fn ensure_kernels_loaded(device: &Arc<CudaDevice>, device_idx: usize) -> PyResult<()> {
    // In cudarc 0.18.2, modules are not cached by name
    // We load the module on demand in each operation
    // This function is kept for API compatibility but does nothing

    // Sanity check
    debug_assert_eq!(
        device.ordinal() as usize,
        device_idx,
        "Device ordinal mismatch in ensure_kernels_loaded"
    );
    // Ensure variables are effectively used to silence warnings
    let _ = (device, device_idx);
    Ok(())
}

/// Get optimal launch configuration for n elements
fn launch_cfg(n: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device type for tensor storage
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceType {
    Cpu,
    Cuda(usize), // Device index
}

impl DeviceType {
    pub fn from_str(s: &str) -> PyResult<Self> {
        if s == "cpu" {
            Ok(DeviceType::Cpu)
        } else if s.starts_with("cuda") {
            {
                let idx = if s.contains(':') {
                    s.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0)
                } else {
                    0
                };
                Ok(DeviceType::Cuda(idx))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "CUDA not available - compile with cuda feature",
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown device: {}",
                s
            )))
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            DeviceType::Cpu => "cpu".to_string(),
            DeviceType::Cuda(idx) => format!("cuda:{}", idx),
        }
    }
}

/// CPU tensor data storage
#[derive(Clone, Debug)]
pub enum CpuData {
    // Existing float types (backward compatibility)
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex128(ArrayD<Complex64>),
    Int64(ArrayD<i64>),

    // NEW: Exact arithmetic types (primary for SGC)
    GoldenExact(ArrayD<GoldenExact>),   // Q(φ) field - exact golden ratio arithmetic
    Rational(ArrayD<Rational>),          // Q field - exact rational arithmetic
    FixedPoint64(ArrayD<i64>),           // Q32.32 fixed-point - deterministic
    Syntonic(ArrayD<SyntonicExact>),
}

/// CUDA tensor data storage
#[derive(Clone)]
pub enum CudaData {
    // Existing float types (backward compatibility)
    Float32(Arc<PooledSlice<f32>>),
    Float64(Arc<PooledSlice<f64>>),
    Int64(Arc<PooledSlice<i64>>),
    /// Complex128 stored as valid C-layout complex numbers
    Complex128(Arc<PooledSlice<CudaComplex64>>),

    // NEW: Exact arithmetic type for GPU (Q32.32 fixed-point)
    FixedPoint64(Arc<PooledSlice<i64>>),  // Q32.32 - GPU exact, deterministic
}

/// Unified tensor data enum
#[derive(Clone)]
pub enum TensorData {
    Cpu(CpuData),
    Cuda {
        data: Arc<CudaData>,
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: String,
    },
}

/// Core tensor storage
#[pyclass]
pub struct TensorStorage {
    pub(crate) data: TensorData,
    pub(crate) shape: Vec<usize>,
    pub(crate) device: DeviceType,
}

#[pymethods]
impl TensorStorage {
    /// Create tensor from a flat Python list with shape and dtype
    #[staticmethod]
    pub fn from_list(
        data: &Bound<'_, PyList>,
        shape: Vec<usize>,
        dtype: &str,
        device: &str,
    ) -> PyResult<Self> {
        let device_type = DeviceType::from_str(device)?;
        let total_size: usize = shape.iter().product();

        let cpu_data =
            match dtype {
                "float32" | "f32" => {
                    let values: Vec<f32> = data
                        .iter()
                        .map(|x| x.extract::<f32>())
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Float32(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(
                        |e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
                    )?)
                }
                "float64" | "f64" => {
                    let values: Vec<f64> = data
                        .iter()
                        .map(|x| x.extract::<f64>())
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Float64(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(
                        |e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
                    )?)
                }
                "complex128" | "c128" => {
                    // Accept complex numbers directly from Python
                    let values: Vec<Complex64> = data
                        .iter()
                        .map(|x| {
                            // Try extracting as Python complex first
                            if let Ok(c) = x.extract::<num_complex::Complex<f64>>() {
                                Ok(Complex64::new(c.re, c.im))
                            } else if let Ok(f) = x.extract::<f64>() {
                                // Fall back to real number (imaginary = 0)
                                Ok(Complex64::new(f, 0.0))
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                    "Expected complex or float number",
                                ))
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Complex data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Complex128(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(
                        |e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
                    )?)
                }
                "int64" | "i64" => {
                    let values: Vec<i64> = data
                        .iter()
                        .map(|x| x.extract::<i64>())
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Int64(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?)
                }
                "golden_exact" | "golden" => {
                    // Convert Python ints/floats to GoldenExact via approximation
                    let values: Vec<GoldenExact> = data
                        .iter()
                        .map(|x| {
                            if let Ok(i) = x.extract::<i32>() {
                                Ok(GoldenExact::from_int(i))
                            } else if let Ok(f) = x.extract::<f64>() {
                                Ok(GoldenExact::find_nearest(f, 1 << 30))
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                    "Expected int or float for golden_exact",
                                ))
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::GoldenExact(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?)
                }
                "rational" => {
                    // Convert Python ints/floats to Rational
                    let values: Vec<Rational> = data
                        .iter()
                        .map(|x| {
                            if let Ok(i) = x.extract::<i64>() {
                                Ok(Rational::from_int(i as i128))
                            } else if let Ok(f) = x.extract::<f64>() {
                                Rational::from_f64_approx(f, 1 << 30).ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "Could not approximate float as rational",
                                    )
                                })
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                    "Expected int or float for rational",
                                ))
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Rational(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?)
                }
                "fixed_point64" | "fp64" | "fixed" => {
                    // Convert Python ints/floats to Q32.32 fixed-point
                    let values: Vec<i64> = data
                        .iter()
                        .map(|x| {
                            if let Ok(i) = x.extract::<i64>() {
                                Ok(FixedPoint64::from_int(i).0)
                            } else if let Ok(f) = x.extract::<f64>() {
                                Ok(FixedPoint64::from_f64(f).0)
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                    "Expected int or float for fixed_point64",
                                ))
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::FixedPoint64(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?)
                }
                "syntonic" | "syntonic_exact" => {
                   let values: Vec<SyntonicExact> = data
                        .iter()
                        .map(|x| x.extract::<SyntonicExact>())
                        .collect::<PyResult<Vec<_>>>()?;
                    if values.len() != total_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Data length {} doesn't match shape {:?}",
                            values.len(),
                            shape
                        )));
                    }
                    CpuData::Syntonic(ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?)
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported dtype: {}",
                        dtype
                    )))
                }
            };

        // Transfer to CUDA if requested
        if let DeviceType::Cuda(idx) = &device_type {
            return Self::cpu_to_cuda(cpu_data, shape, *idx);
        }

        Ok(TensorStorage {
            data: TensorData::Cpu(cpu_data),
            shape,
            device: device_type,
        })
    }

    #[staticmethod]
    pub fn zeros(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let device_type = DeviceType::from_str(device)?;
        let dim = IxDyn(&shape);

        let cpu_data = match dtype {
            "float32" | "f32" => CpuData::Float32(ArrayD::zeros(dim)),
            "float64" | "f64" => CpuData::Float64(ArrayD::zeros(dim)),
            "complex128" | "c128" | "complex" => CpuData::Complex128(ArrayD::zeros(dim)),
            "int64" | "i64" | "int" => CpuData::Int64(ArrayD::zeros(dim)),
            "golden_exact" => CpuData::GoldenExact(ArrayD::from_elem(dim, GoldenExact::zero())),
            "rational" => CpuData::Rational(ArrayD::from_elem(dim, Rational::zero())),
            "fixed_point64" => CpuData::FixedPoint64(ArrayD::zeros(dim)), // Q32.32: 0 is represented as i64 0
            "syntonic" | "syntonic_exact" => CpuData::Syntonic(ArrayD::from_elem(dim, SyntonicExact::new(GoldenExact::zero()))),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported dtype: {}",
                    dtype
                )))
            }
        };

        if let DeviceType::Cuda(idx) = &device_type {
            return Self::cpu_to_cuda(cpu_data, shape, *idx);
        }

        Ok(TensorStorage {
            data: TensorData::Cpu(cpu_data),
            shape,
            device: device_type,
        })
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        match &self.data {
            TensorData::Cpu(cpu) => match cpu {
                CpuData::Float32(_) => "float32".to_string(),
                CpuData::Float64(_) => "float64".to_string(),
                CpuData::Complex128(_) => "complex128".to_string(),
                CpuData::Int64(_) => "int64".to_string(),
                CpuData::GoldenExact(_) => "golden_exact".to_string(),
                CpuData::Rational(_) => "rational".to_string(),
                CpuData::FixedPoint64(_) => "fixed_point64".to_string(),
                CpuData::Syntonic(_) => "syntonic".to_string(),
            },
            TensorData::Cuda { dtype, .. } => dtype.clone(),
        }
    }

    #[getter]
    pub fn device_name(&self) -> String {
        self.device.to_string()
    }

    /// Convert to flat Python list
    pub fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cpu_data = self.ensure_cpu()?;

        match cpu_data {
            CpuData::Float32(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            }
            CpuData::Float64(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            }
            CpuData::Complex128(arr) => {
                // Return complex numbers directly to Python
                let list = PyList::new_bound(
                    py,
                    arr.iter()
                        .map(|c| pyo3::types::PyComplex::from_doubles_bound(py, c.re, c.im)),
                );
                Ok(list.into())
            }
            CpuData::Int64(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            }
            CpuData::GoldenExact(arr) => {
                // Convert GoldenExact to float approximations for Python
                let list = PyList::new_bound(py, arr.iter().map(|x| x.to_f64()));
                Ok(list.into())
            }
            CpuData::Rational(arr) => {
                // Convert Rational to float approximations for Python
                let list = PyList::new_bound(py, arr.iter().map(|x| x.to_f64()));
                Ok(list.into())
            }
            CpuData::FixedPoint64(arr) => {
                // Convert Q32.32 fixed-point to float for Python
                let list = PyList::new_bound(py, arr.iter().map(|&x| (x as f64) / (1i64 << 32) as f64));
                Ok(list.into())
            }
            CpuData::Syntonic(arr) => {
                // Return exact objects, no float degradation
                let list = PyList::new_bound(py, arr.iter().map(|x| x.clone().into_py(py)));
                Ok(list.into())
            }
        }
    }

    /// Move tensor to specified device
    pub fn to_device(&self, device: &str) -> PyResult<TensorStorage> {
        let target_device = DeviceType::from_str(device)?;

        if self.device == target_device {
            return Ok(self.clone_storage());
        }

        {
            match (&self.device, &target_device) {
                (DeviceType::Cpu, DeviceType::Cpu) => Ok(self.clone_storage()),

                (DeviceType::Cpu, DeviceType::Cuda(idx)) => {
                    let cpu_data = self.ensure_cpu()?;
                    Self::cpu_to_cuda(cpu_data, self.shape.clone(), *idx)
                }

                (DeviceType::Cuda(_), DeviceType::Cpu) => {
                    let cpu_data = self.ensure_cpu()?;
                    Ok(TensorStorage {
                        data: TensorData::Cpu(cpu_data),
                        shape: self.shape.clone(),
                        device: DeviceType::Cpu,
                    })
                }

                (DeviceType::Cuda(from), DeviceType::Cuda(to)) if from != to => {
                    // Cross-device transfer: CUDA -> CPU -> CUDA
                    let cpu_data = self.ensure_cpu()?;
                    Self::cpu_to_cuda(cpu_data, self.shape.clone(), *to)
                }

                (DeviceType::Cuda(_), DeviceType::Cuda(_)) => Ok(self.clone_storage()),
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, only CPU devices exist
            Ok(self.clone_storage())
        }
    }

    // ===== DType Conversion Operations (Exact ↔ Float) =====

    /// Convert tensor to GoldenExact type (CPU only)
    /// For float inputs, uses rational approximation with max denominator 2^30
    pub fn to_golden_exact(&self) -> PyResult<TensorStorage> {
        let cpu_data = self.ensure_cpu()?;

        let converted = match cpu_data {
            CpuData::GoldenExact(arr) => CpuData::GoldenExact(arr.clone()),
            CpuData::Rational(arr) => {
                // Rational → GoldenExact: Convert each Rational to GoldenExact(a, 0)
                CpuData::GoldenExact(arr.mapv(|r| GoldenExact::from_rational(r)))
            },
            CpuData::FixedPoint64(arr) => {
                // Q32.32 → GoldenExact via float approximation
                CpuData::GoldenExact(arr.mapv(|fp| {
                    let fp_val = FixedPoint64(fp);
                    GoldenExact::find_nearest(fp_val.to_f64(), 1 << 30)
                }))
            },
            CpuData::Float64(arr) => {
                CpuData::GoldenExact(arr.mapv(|f| GoldenExact::find_nearest(f, 1 << 30)))
            },
            CpuData::Float32(arr) => {
                CpuData::GoldenExact(arr.mapv(|f| GoldenExact::find_nearest(f as f64, 1 << 30)))
            },
            CpuData::Int64(arr) => {
                CpuData::GoldenExact(arr.mapv(|i| GoldenExact::from_int(i as i32)))
            },
            CpuData::Complex128(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot convert complex numbers to GoldenExact"
                ))
            }
            CpuData::Syntonic(arr) => {
                // Syntonic → GoldenExact (extract base)
                CpuData::GoldenExact(arr.mapv(|s| s.base))
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cpu(converted),
            shape: self.shape.clone(),
            device: DeviceType::Cpu,
        })
    }

    /// Convert tensor to Rational type (CPU only)
    pub fn to_rational(&self) -> PyResult<TensorStorage> {
        let cpu_data = self.ensure_cpu()?;

        let converted = match cpu_data {
            CpuData::Rational(arr) => CpuData::Rational(arr.clone()),
            CpuData::GoldenExact(arr) => {
                // GoldenExact → Rational: Evaluate and approximate
                CpuData::Rational(arr.mapv(|ge| {
                    Rational::from_f64_approx(ge.to_f64(), 1 << 30).unwrap_or_else(|| Rational::zero())
                }))
            },
            CpuData::FixedPoint64(arr) => {
                // Q32.32 → Rational
                CpuData::Rational(arr.mapv(|fp| {
                    let fp_val = FixedPoint64(fp);
                    Rational::from_f64_approx(fp_val.to_f64(), 1 << 30).unwrap_or_else(|| Rational::zero())
                }))
            },
            CpuData::Float64(arr) => {
                CpuData::Rational(arr.mapv(|f| Rational::from_f64_approx(f, 1 << 30).unwrap_or_else(|| Rational::zero())))
            },
            CpuData::Float32(arr) => {
                CpuData::Rational(arr.mapv(|f| Rational::from_f64_approx(f as f64, 1 << 30).unwrap_or_else(|| Rational::zero())))
            },
            CpuData::Int64(arr) => {
                CpuData::Rational(arr.mapv(|i| Rational::from_int(i as i128)))
            },
            CpuData::Complex128(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot convert complex numbers to Rational"
                ))
            }
            CpuData::Syntonic(arr) => {
                // Syntonic → Rational (Project to base, then approximate)
                // We avoid full eval() to prevent E* and q pollution
                CpuData::Rational(arr.mapv(|s| {
                     // a + b*phi approximation
                     let base = s.base;
                     Rational::from_f64_approx(base.to_f64(), 1 << 30).unwrap_or_else(|| Rational::zero())
                }))
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cpu(converted),
            shape: self.shape.clone(),
            device: DeviceType::Cpu,
        })
    }

    /// Convert tensor to FixedPoint64 (Q32.32) type
    /// Works on both CPU and GPU
    pub fn to_fixed_point(&self) -> PyResult<TensorStorage> {
        let cpu_data = self.ensure_cpu()?;

        let converted = match cpu_data {
            CpuData::FixedPoint64(arr) => CpuData::FixedPoint64(arr.clone()),
            CpuData::GoldenExact(arr) => {
                // GoldenExact → Q32.32
                CpuData::FixedPoint64(arr.mapv(|ge| FixedPoint64::from_f64(ge.to_f64()).0))
            },
            CpuData::Rational(arr) => {
                // Rational → Q32.32
                CpuData::FixedPoint64(arr.mapv(|r| FixedPoint64::from_f64(r.to_f64()).0))
            },
            CpuData::Float64(arr) => {
                CpuData::FixedPoint64(arr.mapv(|f| FixedPoint64::from_f64(f).0))
            },
            CpuData::Float32(arr) => {
                CpuData::FixedPoint64(arr.mapv(|f| FixedPoint64::from_f64(f as f64).0))
            },
            CpuData::Int64(arr) => {
                CpuData::FixedPoint64(arr.mapv(|i| FixedPoint64::from_int(i).0))
            },
            CpuData::Complex128(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot convert complex numbers to FixedPoint64"
                ))
            }
            CpuData::Syntonic(arr) => {
                // Syntonic → FixedPoint (via integer arithmetic)
                CpuData::FixedPoint64(arr.mapv(|s| s.to_fixed_point().0))
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cpu(converted),
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    /// Convert tensor to Float64 type (lossy for exact types)
    /// This is the primary way to opt into float mode
    pub fn to_float64(&self) -> PyResult<TensorStorage> {
        let cpu_data = self.ensure_cpu()?;

        let converted = match cpu_data {
            CpuData::Float64(arr) => CpuData::Float64(arr.clone()),
            CpuData::Float32(arr) => CpuData::Float64(arr.mapv(|f| f as f64)),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|i| i as f64)),
            // Exact → Float (LOSSY conversions)
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|ge| ge.to_f64()))
            },
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|r| r.to_f64()))
            },
            CpuData::FixedPoint64(arr) => {
                CpuData::Float64(arr.mapv(|fp| FixedPoint64(fp).to_f64()))
            },
            CpuData::Complex128(arr) => {
                // Take real part only
                CpuData::Float64(arr.mapv(|c| c.re))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Strict Exactness Policy: Cannot convert SyntonicExact to Float64. Use Exact types only."
                ))
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cpu(converted),
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    // ===== Arithmetic Operations =====

    pub fn add(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Check shapes match for element-wise operations
        if self.shape != other.shape {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        // Try GPU-native operation if both tensors are on the same CUDA device index
        if let (
            TensorData::Cuda {
                data: a,
                device: dev_a,
                ..
            },
            TensorData::Cuda {
                data: b, device: _, ..
            },
        ) = (&self.data, &other.data)
        {
            // Compare device indices, not Arc pointers
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) =
                (&self.device, &other.device)
            {
                if idx_a == idx_b {
                    let cuda_success = ensure_kernels_loaded(dev_a, *idx_a)
                        .and_then(|_| self.binary_cuda_op(a, b, dev_a, "add"));

                    if let Ok(res) = cuda_success {
                        return Ok(res);
                    }
                    return self.binary_cpu_fallback(a, b, dev_a, "add");
                }
            }
        }

        // CPU fallback
        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a + &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a + &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a + &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a + &b),
            (CpuData::GoldenExact(a), CpuData::GoldenExact(b)) => {
                CpuData::GoldenExact(a.mapv(|x| x) + b.mapv(|x| x))
            }
            (CpuData::Rational(a), CpuData::Rational(b)) => {
                CpuData::Rational(a.mapv(|x| x) + b.mapv(|x| x))
            }
            (CpuData::FixedPoint64(a), CpuData::FixedPoint64(b)) => {
                CpuData::FixedPoint64(&a + &b)
            }
            (CpuData::Syntonic(a), CpuData::Syntonic(b)) => {
                CpuData::Syntonic(a.mapv(|x| x) + b.mapv(|x| x))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch",
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn add_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr + scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr + scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr + Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr + scalar as i64),
            CpuData::GoldenExact(arr) => {
                // Approximate float scalar as GoldenExact
                let scalar_exact = GoldenExact::nearest(scalar, 1 << 30);
                CpuData::GoldenExact(arr.mapv(|x| x + scalar_exact))
            }
            CpuData::Rational(arr) => {
                // Approximate float scalar as rational
                let scalar_rat = Rational::from_f64_approx(scalar, 1 << 30)
                    .unwrap_or_else(|| Rational::from_int(scalar as i128));
                CpuData::Rational(arr.mapv(|x| x + scalar_rat))
            }
            CpuData::FixedPoint64(arr) => {
                // Convert scalar to Q32.32 format
                let scalar_fixed = (scalar * (1i64 << 32) as f64) as i64;
                CpuData::FixedPoint64(&arr + scalar_fixed)
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot add float scalar to SyntonicExact tensor. Use Exact types."
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn sub(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Check shapes match for element-wise operations
        if self.shape != other.shape {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Shape mismatch for subtraction: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        if let (
            TensorData::Cuda {
                data: a,
                device: dev_a,
                ..
            },
            TensorData::Cuda {
                data: b, device: _, ..
            },
        ) = (&self.data, &other.data)
        {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) =
                (&self.device, &other.device)
            {
                if idx_a == idx_b {
                    let cuda_success = ensure_kernels_loaded(dev_a, *idx_a)
                        .and_then(|_| self.binary_cuda_op(a, b, dev_a, "sub"));

                    if let Ok(res) = cuda_success {
                        return Ok(res);
                    }
                    return self.binary_cpu_fallback(a, b, dev_a, "sub");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a - &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a - &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a - &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a - &b),
            (CpuData::GoldenExact(a), CpuData::GoldenExact(b)) => {
                CpuData::GoldenExact(a.mapv(|x| x) - b.mapv(|x| x))
            }
            (CpuData::Rational(a), CpuData::Rational(b)) => {
                CpuData::Rational(a.mapv(|x| x) - b.mapv(|x| x))
            }
            (CpuData::FixedPoint64(a), CpuData::FixedPoint64(b)) => {
                CpuData::FixedPoint64(&a - &b)
            }
            (CpuData::Syntonic(a), CpuData::Syntonic(b)) => {
                CpuData::Syntonic(a.mapv(|x| x) - b.mapv(|x| x))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch",
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn sub_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr - scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr - scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr - Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr - scalar as i64),
            CpuData::GoldenExact(arr) => {
                // Approximate float scalar as GoldenExact
                let scalar_exact = GoldenExact::nearest(scalar, 1 << 30);
                CpuData::GoldenExact(arr.mapv(|x| x - scalar_exact))
            }
            CpuData::Rational(arr) => {
                // Approximate float scalar as rational
                let scalar_rat = Rational::from_f64_approx(scalar, 1 << 30)
                    .unwrap_or_else(|| Rational::from_int(scalar as i128));
                CpuData::Rational(arr.mapv(|x| x - scalar_rat))
            }
            CpuData::FixedPoint64(arr) => {
                let scalar_fixed = (scalar * (1i64 << 32) as f64) as i64;
                CpuData::FixedPoint64(&arr - scalar_fixed)
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot subtract float scalar from SyntonicExact tensor. Use Exact types."
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn mul(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Check shapes match for element-wise operations
        if self.shape != other.shape {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        if let (
            TensorData::Cuda {
                data: a,
                device: dev_a,
                ..
            },
            TensorData::Cuda {
                data: b, device: _, ..
            },
        ) = (&self.data, &other.data)
        {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) =
                (&self.device, &other.device)
            {
                if idx_a == idx_b {
                    let cuda_success = ensure_kernels_loaded(dev_a, *idx_a)
                        .and_then(|_| self.binary_cuda_op(a, b, dev_a, "mul"));

                    if let Ok(res) = cuda_success {
                        return Ok(res);
                    }
                    return self.binary_cpu_fallback(a, b, dev_a, "mul");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a * &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a * &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a * &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a * &b),
            (CpuData::GoldenExact(a), CpuData::GoldenExact(b)) => {
                CpuData::GoldenExact(a.mapv(|x| x) * b.mapv(|x| x))
            }
            (CpuData::Rational(a), CpuData::Rational(b)) => {
                CpuData::Rational(a.mapv(|x| x) * b.mapv(|x| x))
            }
            (CpuData::FixedPoint64(a), CpuData::FixedPoint64(b)) => {
                // Q32.32 multiplication needs special handling
                use ndarray::Zip;
                let mut result = ArrayD::<i64>::zeros(a.raw_dim());
                Zip::from(&mut result)
                    .and(a.view())
                    .and(b.view())
                    .for_each(|r, &x, &y| {
                        *r = (FixedPoint64::new(x) * FixedPoint64::new(y)).0;
                    });
                CpuData::FixedPoint64(result)
            }
            (CpuData::Syntonic(a), CpuData::Syntonic(b)) => {
                CpuData::Syntonic(a.mapv(|x| x) * b.mapv(|x| x))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch",
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn mul_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr * scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr * scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr * Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr * scalar as i64),
            CpuData::GoldenExact(arr) => {
                // Approximate float scalar as GoldenExact (a + b·φ)
                let scalar_exact = GoldenExact::nearest(scalar, 1 << 30);
                CpuData::GoldenExact(arr.mapv(|x| x * scalar_exact))
            }
            CpuData::Rational(arr) => {
                // Approximate float scalar as rational p/q
                // Use continued fraction approximation with max denominator 2^30
                let scalar_rat = Rational::from_f64_approx(scalar, 1 << 30)
                    .unwrap_or_else(|| Rational::from_int(scalar as i128));
                CpuData::Rational(arr.mapv(|x| x * scalar_rat))
            }
            CpuData::FixedPoint64(arr) => {
                // Q32.32 multiplication: need to scale properly
                let scalar_fixed = FixedPoint64::new((scalar * (1i64 << 32) as f64) as i64);
                CpuData::FixedPoint64(arr.mapv(|x| (FixedPoint64::new(x) * scalar_fixed).0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot multiply SyntonicExact tensor by float scalar. Use mul_scalar_golden."
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Multiply tensor by a GoldenExact scalar using exact coefficients.
    /// 
    /// GoldenExact represents values as a + b·φ where φ is the golden ratio.
    /// This method takes the exact rational coefficients to avoid float approximation.
    /// 
    /// # Arguments
    /// * `a_num`, `a_denom` - Numerator and denominator of the rational 'a' coefficient
    /// * `b_num`, `b_denom` - Numerator and denominator of the rational 'b' coefficient (φ coefficient)
    /// 
    /// # Example
    /// To multiply by 1/φ = 0 + 1·φ⁻¹ = -1 + 1·φ (since φ⁻¹ = φ - 1):
    /// ```ignore
    /// let result = tensor.mul_scalar_golden(-1, 1, 1, 1)?;  // -1 + 1·φ = 1/φ
    /// ```
    pub fn mul_scalar_golden(
        &self,
        a_num: i64,
        a_denom: i64,
        b_num: i64,
        b_denom: i64,
    ) -> PyResult<TensorStorage> {
        // Construct exact GoldenExact scalar from rational coefficients
        let a_rat = Rational::new(a_num as i128, a_denom as i128);
        let b_rat = Rational::new(b_num as i128, b_denom as i128);
        let scalar_exact = GoldenExact::new(a_rat, b_rat);

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::GoldenExact(arr) => {
                // Exact GoldenExact × GoldenExact multiplication
                CpuData::GoldenExact(arr.mapv(|x| x * scalar_exact))
            }
            CpuData::Rational(arr) => {
                // Rational × GoldenExact → GoldenExact
                // Convert Rational to GoldenExact first (a + 0·φ)
                let result_arr = arr.mapv(|r| {
                    let r_as_ge = GoldenExact::from_rational(r);
                    r_as_ge * scalar_exact
                });
                CpuData::GoldenExact(result_arr)
            }
            CpuData::Float64(arr) => {
                // Float64 × GoldenExact: evaluate GoldenExact to float then multiply
                // (This is the fallback for non-exact types)
                let scalar_f64 = scalar_exact.to_f64();
                CpuData::Float64(&arr * scalar_f64)
            }
            CpuData::Float32(arr) => {
                let scalar_f32 = scalar_exact.to_f64() as f32;
                CpuData::Float32(&arr * scalar_f32)
            }
            CpuData::Complex128(arr) => {
                let scalar_f64 = scalar_exact.to_f64();
                CpuData::Complex128(&arr * Complex64::new(scalar_f64, 0.0))
            }
            CpuData::Int64(arr) => {
                // Int64 × GoldenExact: convert to GoldenExact array first
                let result_arr = arr.mapv(|i| {
                    let i_as_ge = GoldenExact::from_int(i as i32);
                    i_as_ge * scalar_exact
                });
                CpuData::GoldenExact(result_arr)
            }
            CpuData::FixedPoint64(arr) => {
                // FixedPoint64 × GoldenExact: evaluate to fixed-point
                // Scale the GoldenExact to Q32.32 format
                let scalar_fixed = FixedPoint64::from_f64(scalar_exact.to_f64());
                CpuData::FixedPoint64(arr.mapv(|x| (FixedPoint64::new(x) * scalar_fixed).0))
            }
            CpuData::Syntonic(arr) => {
                // Syntonic × GoldenExact: convert scalar to Syntonic and multiply
                let scalar_syntonic = SyntonicExact::new(scalar_exact);
                CpuData::Syntonic(arr.mapv(|x| x * scalar_syntonic))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn div(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Check shapes match for element-wise operations
        if self.shape != other.shape {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Shape mismatch for division: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        if let (
            TensorData::Cuda {
                data: a,
                device: dev_a,
                ..
            },
            TensorData::Cuda {
                data: b, device: _, ..
            },
        ) = (&self.data, &other.data)
        {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) =
                (&self.device, &other.device)
            {
                if idx_a == idx_b {
                    let cuda_success = ensure_kernels_loaded(dev_a, *idx_a)
                        .and_then(|_| self.binary_cuda_op(a, b, dev_a, "div"));

                    if let Ok(res) = cuda_success {
                        return Ok(res);
                    }
                    return self.binary_cpu_fallback(a, b, dev_a, "div");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a / &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a / &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a / &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a / &b),
            (CpuData::GoldenExact(a), CpuData::GoldenExact(b)) => {
                CpuData::GoldenExact(a.mapv(|x| x) / b.mapv(|x| x))
            }
            (CpuData::Rational(a), CpuData::Rational(b)) => {
                CpuData::Rational(a.mapv(|x| x) / b.mapv(|x| x))
            }
            (CpuData::FixedPoint64(a), CpuData::FixedPoint64(b)) => {
                // Q32.32 division needs special handling
                use ndarray::Zip;
                let mut result = ArrayD::<i64>::zeros(a.raw_dim());
                Zip::from(&mut result)
                    .and(a.view())
                    .and(b.view())
                    .for_each(|r, &x, &y| {
                        *r = (FixedPoint64::new(x) / FixedPoint64::new(y)).0;
                    });
                CpuData::FixedPoint64(result)
            }
            (CpuData::Syntonic(a), CpuData::Syntonic(b)) => {
                CpuData::Syntonic(a.mapv(|x| x) / b.mapv(|x| x))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch",
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn div_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr / scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr / scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr / Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr / scalar as i64),
            CpuData::GoldenExact(arr) => {
                // Approximate float scalar as GoldenExact
                let scalar_exact = GoldenExact::nearest(scalar, 1 << 30);
                CpuData::GoldenExact(arr.mapv(|x| x / scalar_exact))
            }
            CpuData::Rational(arr) => {
                // Approximate float scalar as rational
                let scalar_rat = Rational::from_f64_approx(scalar, 1 << 30)
                    .unwrap_or_else(|| Rational::from_int(scalar as i128));
                CpuData::Rational(arr.mapv(|x| x / scalar_rat))
            }
            CpuData::FixedPoint64(arr) => {
                // Q32.32 division: need to scale properly
                let scalar_fixed = FixedPoint64::new((scalar * (1i64 << 32) as f64) as i64);
                CpuData::FixedPoint64(arr.mapv(|x| (FixedPoint64::new(x) / scalar_fixed).0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Cannot divide SyntonicExact tensor by float scalar."
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn neg(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "neg");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(-&arr),
            CpuData::Float32(arr) => CpuData::Float32(-&arr),
            CpuData::Complex128(arr) => CpuData::Complex128(-&arr),
            CpuData::Int64(arr) => CpuData::Int64(-&arr),
            CpuData::GoldenExact(arr) => CpuData::GoldenExact(arr.mapv(|x| -x)),
            CpuData::Rational(arr) => CpuData::Rational(arr.mapv(|x| -x)),
            CpuData::FixedPoint64(arr) => CpuData::FixedPoint64(-&arr),
            CpuData::Syntonic(arr) => CpuData::Syntonic(arr.mapv(|x| -x)),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn abs(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "abs");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.abs())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.abs())),
            CpuData::Complex128(arr) => CpuData::Float64(arr.mapv(|x| x.norm())),
            CpuData::Int64(arr) => CpuData::Int64(arr.mapv(|x| x.abs())),
            CpuData::GoldenExact(arr) => {
                // Abs of GoldenExact: convert to float approximation
                CpuData::Float64(arr.mapv(|x| x.to_f64().abs()))
            }
            CpuData::Rational(arr) => {
                CpuData::Rational(arr.mapv(|x| x.abs()))
            }
            CpuData::FixedPoint64(arr) => CpuData::FixedPoint64(arr.mapv(|x| x.abs())),
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Absolute value not supported for SyntonicExact (use norm)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "exp");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.exp())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.exp())),
            CpuData::Complex128(arr) => {
                // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
                CpuData::Complex128(arr.mapv(|c| {
                    let exp_re = c.re.exp();
                    Complex64::new(exp_re * c.im.cos(), exp_re * c.im.sin())
                }))
            }
            CpuData::Int64(arr) => {
                // Convert to float for exp
                CpuData::Float64(arr.mapv(|x| (x as f64).exp()))
            }
            CpuData::GoldenExact(arr) => {
                // Convert to float for exp (transcendental function)
                CpuData::Float64(arr.mapv(|x| x.to_f64().exp()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().exp()))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Exponential function not supported for SyntonicExact (transcendental float leak)"
                ))
            }
            CpuData::FixedPoint64(arr) => {
                // Use exact fixed-point exp for deterministic computation
                CpuData::FixedPoint64(arr.mapv(|x| FixedPoint64::new(x).exp().0))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise golden exponential: exp(-x/φ)
    /// Used for computing golden measure weights w(n) = exp(-|n|²/φ)
    pub fn exp_golden(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "exp_golden");
        }


        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| (-x * PHI_INV).exp())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| (-x * PHI_INV as f32).exp())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (-(x as f64) * PHI_INV).exp())),
            CpuData::Complex128(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "exp_golden not supported for complex types",
                ));
            }
            CpuData::GoldenExact(arr) => {
                // Exact golden exponential using φ̂ = 1/φ
                let phi_hat = GoldenExact::phi_hat();
                CpuData::Float64(arr.mapv(|x| (-(x * phi_hat)).to_f64().exp()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| (-(x.to_f64()) * PHI_INV).exp()))
            }
            CpuData::FixedPoint64(arr) => {
                // Use fixed-point arithmetic with 1/φ in Q32.32
                let phi_inv_fp = FixedPoint64::new((PHI_INV * (1i64 << 32) as f64) as i64);
                CpuData::FixedPoint64(arr.mapv(|x| (-(FixedPoint64::new(x) * phi_inv_fp).exp()).0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "exp_golden not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "log");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.ln())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.ln())),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| c.ln())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (x as f64).ln())),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().ln()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().ln()))
            }
            CpuData::FixedPoint64(arr) => {
                // log not implemented in fixed-point, convert to float
                CpuData::Float64(arr.mapv(|x| (( x as f64) / (1i64 << 32) as f64).ln()))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "log not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise sine
    pub fn sin(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "sin");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.sin())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.sin())),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| c.sin())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (x as f64).sin())),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().sin()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().sin()))
            }
            CpuData::FixedPoint64(arr) => {
                // Use exact CORDIC sin for deterministic computation
                CpuData::FixedPoint64(arr.mapv(|x| FixedPoint64::new(x).sin().0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "sin not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise cosine
    pub fn cos(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "cos");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.cos())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.cos())),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| c.cos())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (x as f64).cos())),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().cos()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().cos()))
            }
            CpuData::FixedPoint64(arr) => {
                // Use exact CORDIC cos for deterministic computation
                CpuData::FixedPoint64(arr.mapv(|x| FixedPoint64::new(x).cos().0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "cos not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "sqrt");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.sqrt())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.sqrt())),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| c.sqrt())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (x as f64).sqrt())),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().sqrt()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().sqrt()))
            }
            CpuData::FixedPoint64(arr) => {
                // Use exact fixed-point sqrt for deterministic computation
                CpuData::FixedPoint64(arr.mapv(|x| FixedPoint64::new(x).sqrt().0))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "sqrt not supported for SyntonicExact (potential irrational leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise hyperbolic tangent
    pub fn tanh(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "tanh");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.tanh())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.tanh())),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| c.tanh())),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| (x as f64).tanh())),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().tanh()))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| x.to_f64().tanh()))
            }
            CpuData::FixedPoint64(arr) => {
                // tanh not implemented in fixed-point, convert to float
                CpuData::Float64(arr.mapv(|x| (( x as f64) / (1i64 << 32) as f64).tanh()))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "tanh not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise sigmoid: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "sigmoid");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| {
                let exp_neg_c = (-c).exp();
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + exp_neg_c)
            })),
            CpuData::Int64(arr) => CpuData::Float64(arr.mapv(|x| {
                let x_f64 = x as f64;
                1.0 / (1.0 + (-x_f64).exp())
            })),
            CpuData::GoldenExact(arr) => {
                CpuData::Float64(arr.mapv(|x| {
                    let x_f64 = x.to_f64();
                    1.0 / (1.0 + (-x_f64).exp())
                }))
            }
            CpuData::Rational(arr) => {
                CpuData::Float64(arr.mapv(|x| {
                    let x_f64 = x.to_f64();
                    1.0 / (1.0 + (-x_f64).exp())
                }))
            }
            CpuData::FixedPoint64(arr) => {
                CpuData::Float64(arr.mapv(|x| {
                    let x_f64 = ( x as f64) / (1i64 << 32) as f64;
                    1.0 / (1.0 + (-x_f64).exp())
                }))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "sigmoid not supported for SyntonicExact (transcendental float leak)"
                ))
            }
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Element-wise ReLU: max(0, x)
    pub fn relu(&self) -> PyResult<TensorStorage> {
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device {
                DeviceType::Cuda(idx) => *idx,
                _ => 0,
            };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "relu");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.max(0.0))),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.max(0.0))),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|c| {
                if c.re > 0.0 {
                    c
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })),
            CpuData::Int64(arr) => CpuData::Int64(arr.mapv(|x| x.max(0))),
            CpuData::GoldenExact(arr) => {
                // Use approximate comparison for ReLU
                CpuData::GoldenExact(arr.mapv(|x| if x.to_f64() > 0.0 { x } else { GoldenExact::zero() }))
            }
            CpuData::Rational(arr) => {
                CpuData::Rational(arr.mapv(|x| if x.to_f64() > 0.0 { x } else { Rational::zero() }))
            }
            CpuData::FixedPoint64(arr) => {
                CpuData::FixedPoint64(arr.mapv(|x| if x > 0 { x } else { 0 }))
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "relu not supported for SyntonicExact"
                ))
            }

        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn norm(&self, ord: Option<i32>) -> PyResult<f64> {
        if let Some(o) = ord {
            if o != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    format!(
                        "Only L2 norm (ord=2 or None) is currently supported, got {}",
                        o
                    ),
                ));
            }
        }
        let cpu = self.ensure_cpu()?;
        Ok(match cpu {
            CpuData::Float64(arr) => arr.iter().map(|x| x * x).sum::<f64>().sqrt(),
            CpuData::Float32(arr) => arr.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
            CpuData::Complex128(arr) => arr.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt(),
            CpuData::Int64(arr) => arr.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
            CpuData::GoldenExact(arr) => arr.iter().map(|x| x.to_f64().powi(2)).sum::<f64>().sqrt(),
            CpuData::Rational(arr) => arr.iter().map(|x| x.to_f64().powi(2)).sum::<f64>().sqrt(),
            CpuData::FixedPoint64(arr) => {
                arr.iter().map(|&x| {
                    let x_f64 = (x as f64) / (1i64 << 32) as f64;
                    x_f64.powi(2)
                }).sum::<f64>().sqrt()
            }
            CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "norm (L2 float) not supported for SyntonicExact (use exact_norm or norm_sqr)"
                ))
            }
        })
    }

    pub fn conj(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|x| x.conj())),
            other => other.clone(),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn matmul(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        matmul::mm(self, other)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    pub fn transpose(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.t().to_owned()),
            CpuData::Float32(arr) => CpuData::Float32(arr.t().to_owned()),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.t().to_owned()),
            CpuData::Int64(arr) => CpuData::Int64(arr.t().to_owned()),
            CpuData::GoldenExact(arr) => CpuData::GoldenExact(arr.t().to_owned()),
            CpuData::Rational(arr) => CpuData::Rational(arr.t().to_owned()),
            CpuData::FixedPoint64(arr) => CpuData::FixedPoint64(arr.t().to_owned()),
            CpuData::Syntonic(arr) => CpuData::Syntonic(arr.t().to_owned()),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    // ===== Linear Algebra =====

    pub fn eig(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (e, v) = arr_2d.eig().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device),
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (e, v) = arr_2d.eig().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Eig only for f64/c128",
            )),
        }
    }

    pub fn eigh(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (e, v) = arr_2d.eigh(UPLO::Lower).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(v.into_dyn()), &self.device),
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (e, v) = arr_2d.eigh(UPLO::Lower).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Eigh only for f64/c128",
            )),
        }
    }

    pub fn svd(
        &self,
        full_matrices: bool,
    ) -> PyResult<(TensorStorage, TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (u, s, vt) = arr_2d.svd(true, full_matrices).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(u.unwrap().into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(s.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(vt.unwrap().into_dyn()), &self.device),
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (u, s, vt) = arr_2d.svd(true, full_matrices).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(u.unwrap().into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(s.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(vt.unwrap().into_dyn()), &self.device),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "SVD only for f64/c128",
            )),
        }
    }

    pub fn qr(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (q, r) = arr_2d.qr().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(q.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(r.into_dyn()), &self.device),
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let (q, r) = arr_2d.qr().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(q.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(r.into_dyn()), &self.device),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "QR only for f64/c128",
            )),
        }
    }

    pub fn inv(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.inv().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Self::wrap_cpu(
                    CpuData::Float64(res.into_dyn()),
                    &self.device,
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.inv().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Self::wrap_cpu(
                    CpuData::Complex128(res.into_dyn()),
                    &self.device,
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Inv only for f64/c128",
            )),
        }
    }

    pub fn solve(&self, b: &TensorStorage) -> PyResult<TensorStorage> {
        let a_cpu = self.ensure_cpu()?;
        let b_cpu = b.ensure_cpu()?;

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a), CpuData::Float64(rhs)) => {
                let a_2d = a
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("A must be 2D"))?;
                if let Ok(rhs_1d) = rhs.clone().into_dimensionality::<Ix1>() {
                    let res = a_2d.solve(&rhs_1d).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Self::wrap_cpu(
                        CpuData::Float64(res.into_dyn()),
                        &self.device,
                    ))
                } else {
                    let rhs_2d = rhs.clone().into_dimensionality::<Ix2>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("B must be 1D or 2D")
                    })?;
                    let a_inv = a_2d.inv().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Self::wrap_cpu(
                        CpuData::Float64(a_inv.dot(&rhs_2d).into_dyn()),
                        &self.device,
                    ))
                }
            }
            (CpuData::Complex128(a), CpuData::Complex128(rhs)) => {
                let a_2d = a
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("A must be 2D"))?;
                if let Ok(rhs_1d) = rhs.clone().into_dimensionality::<Ix1>() {
                    let res = a_2d.solve(&rhs_1d).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Self::wrap_cpu(
                        CpuData::Complex128(res.into_dyn()),
                        &self.device,
                    ))
                } else {
                    let rhs_2d = rhs.clone().into_dimensionality::<Ix2>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("B must be 1D or 2D")
                    })?;
                    let a_inv = a_2d.inv().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Self::wrap_cpu(
                        CpuData::Complex128(a_inv.dot(&rhs_2d).into_dyn()),
                        &self.device,
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Solve only for f64/c128",
            )),
        }
    }

    pub fn det(&self) -> PyResult<Complex64> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.det().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Complex64::new(res, 0.0))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                arr_2d
                    .det()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Det only for f64/c128",
            )),
        }
    }

    pub fn trace(&self) -> PyResult<Complex64> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.trace().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Complex64::new(res, 0.0))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                arr_2d
                    .trace()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Trace only for f64/c128",
            )),
        }
    }

    /// Matrix exponential exp(A)
    ///
    /// Computes the matrix exponential using eigenvalue decomposition:
    /// If A = V * D * V^{-1}, then exp(A) = V * exp(D) * V^{-1}
    /// where exp(D) is diagonal with exp(λᵢ) entries.
    ///
    /// For CRT: evolution operators U(t) = exp(-iHt).
    pub fn expm(&self) -> PyResult<TensorStorage> {
        use ndarray::Array2;
        use num_complex::Complex64;

        let cpu = self.ensure_cpu()?;

        // Convert to complex for eigenvalue decomposition
        let arr_complex = match cpu {
            CpuData::Float64(a) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                a_2d.mapv(|x| Complex64::new(x, 0.0))
            }
            CpuData::Complex128(a) => a.clone().into_dimensionality::<Ix2>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
            })?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Expm only for f64/c128",
                ));
            }
        };

        let n = arr_complex.nrows();
        if n != arr_complex.ncols() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix must be square",
            ));
        }

        // Compute eigenvalue decomposition
        let (eigenvalues, eigenvectors) = arr_complex
            .eig()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Compute exp(D) where D is diagonal matrix of eigenvalues
        let mut exp_eigenvalues = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            exp_eigenvalues[[i, i]] = eigenvalues[i].exp();
        }

        // Compute inverse of eigenvectors
        let eigenvectors_inv = eigenvectors.inv().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to invert eigenvector matrix: {}",
                e
            ))
        })?;

        // exp(A) = V * exp(D) * V^{-1}
        let result = eigenvectors.dot(&exp_eigenvalues).dot(&eigenvectors_inv);

        Ok(Self::wrap_cpu(
            CpuData::Complex128(result.into_dyn()),
            &self.device,
        ))
    }

    /// Matrix logarithm log(A)
    ///
    /// Computes the principal matrix logarithm using eigenvalue decomposition:
    /// If A = V * D * V^{-1}, then log(A) = V * log(D) * V^{-1}
    /// where log(D) is diagonal with log(λᵢ) entries.
    ///
    /// Returns complex result even for real input.
    pub fn logm(&self) -> PyResult<TensorStorage> {
        use ndarray::Array2;
        use num_complex::Complex64;

        let cpu = self.ensure_cpu()?;

        // Convert to complex for eigenvalue decomposition
        let arr_complex = match cpu {
            CpuData::Float64(a) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                a_2d.mapv(|x| Complex64::new(x, 0.0))
            }
            CpuData::Complex128(a) => a.clone().into_dimensionality::<Ix2>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
            })?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Logm only for f64/c128",
                ));
            }
        };

        let n = arr_complex.nrows();
        if n != arr_complex.ncols() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix must be square",
            ));
        }

        // Compute eigenvalue decomposition
        let (eigenvalues, eigenvectors) = arr_complex
            .eig()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Compute log(D) where D is diagonal matrix of eigenvalues
        let mut log_eigenvalues = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            // Check for zero or near-zero eigenvalues
            if eigenvalues[i].norm() < 1e-15 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Matrix has zero or near-zero eigenvalue - logarithm undefined",
                ));
            }
            log_eigenvalues[[i, i]] = eigenvalues[i].ln();
        }

        // Compute inverse of eigenvectors
        let eigenvectors_inv = eigenvectors.inv().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to invert eigenvector matrix: {}",
                e
            ))
        })?;

        // log(A) = V * log(D) * V^{-1}
        let result = eigenvectors.dot(&log_eigenvalues).dot(&eigenvectors_inv);

        Ok(Self::wrap_cpu(
            CpuData::Complex128(result.into_dyn()),
            &self.device,
        ))
    }

    pub fn cholesky(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.cholesky(UPLO::Lower).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Self::wrap_cpu(
                    CpuData::Float64(res.into_dyn()),
                    &self.device,
                ))
            }
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D")
                })?;
                let res = arr_2d.cholesky(UPLO::Lower).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                Ok(Self::wrap_cpu(
                    CpuData::Complex128(res.into_dyn()),
                    &self.device,
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Cholesky only for f64/c128",
            )),
        }
    }

    // ===== SRT φ-Algebra Operations =====

    /// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
    /// This is the fundamental bracket for SRT Lie algebra representations.
    pub fn phi_bracket(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Compute AB
        let ab = self.matmul(other)?;
        // Compute BA
        let ba = other.matmul(self)?;
        // Compute φ⁻¹BA
        let phi_inv_ba = ba.mul_scalar(Self::phi_inv())?;
        // Result: AB - φ⁻¹BA
        ab.sub(&phi_inv_ba)
    }

    /// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
    /// Symmetric counterpart to the φ-bracket.
    pub fn phi_antibracket(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Compute AB
        let ab = self.matmul(other)?;
        // Compute BA
        let ba = other.matmul(self)?;
        // Compute φ⁻¹BA
        let phi_inv_ba = ba.mul_scalar(Self::phi_inv())?;
        // Result: AB + φ⁻¹BA
        ab.add(&phi_inv_ba)
    }

    /// Golden-scaled matmul: φ^k × (A @ B)
    /// Used for hierarchical scale operations in SRT.
    pub fn mm_phi(&self, other: &TensorStorage, k: i32) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        let scale = Self::phi_power(k);
        result.mul_scalar(scale)
    }

    /// Corrected matmul: (1 + sign × q/N) × (A @ B)
    /// Applies SRT correction factor based on algebraic structure dimension.
    ///
    /// The correction factor is derived symbolically as (1 ± q/N) then evaluated.
    pub fn mm_corrected(&self, other: &TensorStorage, n: u32, sign: i8) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        let correction = Self::correction_factor(n, sign);
        result.mul_scalar(correction)
    }

    /// Matmul with additive term: α(A @ B) + βC
    /// Used for iterative refinement and accumulation.
    pub fn mm_add(
        &self,
        other: &TensorStorage,
        c: &TensorStorage,
        alpha: f64,
        beta: f64,
    ) -> PyResult<TensorStorage> {
        let ab = self.matmul(other)?;
        let scaled_ab = ab.mul_scalar(alpha)?;
        let scaled_c = c.mul_scalar(beta)?;
        scaled_ab.add(&scaled_c)
    }

    /// Transposed matmul: Aᵀ @ B
    pub fn mm_tn(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_t = self.transpose()?;
        a_t.matmul(other)
    }

    /// Matmul with transposed right: A @ Bᵀ
    pub fn mm_nt(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let b_t = other.transpose()?;
        self.matmul(&b_t)
    }

    /// Double transposed matmul: Aᵀ @ Bᵀ
    pub fn mm_tt(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_t = self.transpose()?;
        let b_t = other.transpose()?;
        a_t.matmul(&b_t)
    }

    /// Hermitian-None matmul: A† @ B (conjugate transpose of A times B)
    /// Critical for quantum/complex operations in SRT.
    pub fn mm_hn(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_h = self.transpose()?.conj()?;
        a_h.matmul(other)
    }

    /// None-Hermitian matmul: A @ B† (A times conjugate transpose of B)
    pub fn mm_nh(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let b_h = other.transpose()?.conj()?;
        self.matmul(&b_h)
    }

    /// Batched matrix multiplication: C[i] = A[i] @ B[i]
    /// For 3D tensors, applies matmul along the first (batch) dimension.
    pub fn bmm(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Ensure both are 3D
        if self.shape.len() != 3 || other.shape.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "bmm requires 3D tensors, got shapes {:?} and {:?}",
                self.shape, other.shape
            )));
        }
        if self.shape[0] != other.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Batch sizes must match: {} vs {}",
                self.shape[0], other.shape[0]
            )));
        }

        let batch_size = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let n = other.shape[2];

        if k != other.shape[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Inner dimensions must match: {} vs {}",
                k, other.shape[1]
            )));
        }

        let a_cpu = self.ensure_cpu()?;
        let b_cpu = other.ensure_cpu()?;

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                let mut result = ndarray::Array3::<f64>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Float64(result.into_dyn())))
            }
            (CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                let mut result = ndarray::Array3::<f32>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Float32(result.into_dyn())))
            }
            (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                let mut result = ndarray::Array3::<Complex64>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Complex128(result.into_dyn())))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported dtype combination for bmm",
            )),
        }
    }

    /// Golden phase matmul: e^{iπn/φ} × (A @ B)
    /// Applies SRT phase rotation based on golden ratio.
    pub fn mm_golden_phase(&self, other: &TensorStorage, n: i32) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        // Compute phase: e^{iπn/φ} = cos(πn/φ) + i·sin(πn/φ)
        let phi = Self::phi();
        let angle = crate::constants::SRT_PI * (n as f64) / phi;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Apply phase rotation
        let cpu = result.ensure_cpu()?;
        let rotated = match cpu {
            CpuData::Float64(arr) => {
                // For real, just multiply by cos(angle) (imaginary part would be sin*arr)
                CpuData::Float64(arr.mapv(|x| x * cos_a))
            }
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x * (cos_a as f32))),
            CpuData::Complex128(arr) => {
                let phase = Complex64::new(cos_a, sin_a);
                CpuData::Complex128(arr.mapv(|x| x * phase))
            }
            CpuData::Int64(_) | CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::FixedPoint64(_) | CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Golden phase not supported for int64 or exact types",
                ));
            }
        };
        Ok(Self::wrap_cpu_data(rotated))
    }

    /// Golden-weighted matmul: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(-k²/φ)
    /// Applies Golden Gaussian weights during matrix multiplication.
    pub fn mm_golden_weighted(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_cpu = self.ensure_cpu()?;
        let b_cpu = other.ensure_cpu()?;
        let phi = Self::phi();

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                let a_2d = a_arr.clone().into_dimensionality::<Ix2>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("A must be 2D: {}", e))
                })?;
                let b_2d = b_arr.clone().into_dimensionality::<Ix2>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("B must be 2D: {}", e))
                })?;

                let (m, k_a) = (a_2d.nrows(), a_2d.ncols());
                let (k_b, n) = (b_2d.nrows(), b_2d.ncols());

                if k_a != k_b {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Dimension mismatch: {} vs {}",
                        k_a, k_b
                    )));
                }

                // Precompute golden weights: w[k] = exp(-k²/φ)
                let weights: Vec<f64> = (0..k_a)
                    .map(|k| (-(k as f64).powi(2) / phi).exp())
                    .collect();

                // Weighted matmul
                let mut result = ndarray::Array2::<f64>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for ki in 0..k_a {
                            sum += a_2d[[i, ki]] * b_2d[[ki, j]] * weights[ki];
                        }
                        result[[i, j]] = sum;
                    }
                }
                Ok(Self::wrap_cpu_data(CpuData::Float64(result.into_dyn())))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "mm_golden_weighted currently only supports float64",
            )),
        }
    }

    /// Weighted sum of tensors: Σ w_i × T_i
    /// Used for DHSR projection summation over lattice points.
    #[staticmethod]
    pub fn projection_sum(
        weights: &Bound<'_, PyList>,
        tensors: &Bound<'_, PyList>,
    ) -> PyResult<TensorStorage> {
        let n = weights.len();
        if n == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Empty weights list",
            ));
        }
        if n != tensors.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Weights ({}) and tensors ({}) must have same length",
                n,
                tensors.len()
            )));
        }

        // Extract first tensor to get shape and dtype
        let first_tensor: PyRef<TensorStorage> = tensors.get_item(0)?.extract()?;
        let first_weight: f64 = weights.get_item(0)?.extract()?;
        let mut result = first_tensor.mul_scalar(first_weight)?;

        // Accumulate remaining terms
        for i in 1..n {
            let w: f64 = weights.get_item(i)?.extract()?;
            let t: PyRef<TensorStorage> = tensors.get_item(i)?.extract()?;
            let scaled = t.mul_scalar(w)?;
            result = result.add(&scaled)?;
        }

        Ok(result)
    }

    /// Syntony-scaled matmul: σ(ψ) × (A @ B)
    /// Applies syntony measure as scaling factor.
    pub fn mm_syntony(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        // Compute syntony from self
        let syntony = self.compute_syntony_basic();
        // Scale by (1 - q × syntony) for coherent states
        let scale = 1.0 - Self::q_deficit() * syntony;
        result.mul_scalar(scale)
    }

    // ===== DHSR Operations =====

    pub fn compute_syntony_basic(&self) -> f64 {
        const EPSILON: f64 = 1e-10;
        const ALPHA_0: f64 = 0.1;

        let d_psi = match self.differentiate(ALPHA_0) {
            Ok(s) => s,
            Err(_) => return 0.5,
        };

        let h_d_psi = match d_psi.harmonize(Self::phi_inv(), 0.0) {
            Ok(s) => s,
            Err(_) => return 0.5,
        };

        let self_cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };
        let d_cpu = match d_psi.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };
        let hd_cpu = match h_d_psi.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };

        let (numerator, denominator) = match (self_cpu, d_cpu, hd_cpu) {
            (CpuData::Float64(psi), CpuData::Float64(d), CpuData::Float64(hd)) => {
                let num: f64 = hd
                    .iter()
                    .zip(d.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let den: f64 = d
                    .iter()
                    .zip(psi.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (num, den)
            }
            (CpuData::Float32(psi), CpuData::Float32(d), CpuData::Float32(hd)) => {
                let num: f64 = hd
                    .iter()
                    .zip(d.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let den: f64 = d
                    .iter()
                    .zip(psi.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (num, den)
            }
            (CpuData::Complex128(psi), CpuData::Complex128(d), CpuData::Complex128(hd)) => {
                let num: f64 = hd
                    .iter()
                    .zip(d.iter())
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let den: f64 = d
                    .iter()
                    .zip(psi.iter())
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                (num, den)
            }
            _ => return 0.5,
        };

        (numerator / (denominator + EPSILON)).clamp(0.0, 1.0)
    }

    pub fn free_energy(&self) -> f64 {
        let cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        match cpu {
            CpuData::Float64(arr) => {
                Self::compute_free_energy(&arr.iter().cloned().collect::<Vec<_>>())
            }
            CpuData::Float32(arr) => {
                Self::compute_free_energy(&arr.iter().map(|x| *x as f64).collect::<Vec<_>>())
            }
            CpuData::Complex128(arr) => {
                Self::compute_free_energy(&arr.iter().map(|x| x.norm()).collect::<Vec<_>>())
            }
            CpuData::Int64(arr) => {
                Self::compute_free_energy(&arr.iter().map(|x| *x as f64).collect::<Vec<_>>())
            }
            CpuData::GoldenExact(arr) => {
                Self::compute_free_energy(&arr.iter().map(|x| x.to_f64()).collect::<Vec<_>>())
            }
            CpuData::Rational(arr) => {
                Self::compute_free_energy(&arr.iter().map(|x| x.to_f64()).collect::<Vec<_>>())
            }
            CpuData::FixedPoint64(arr) => {
                Self::compute_free_energy(&arr.iter().map(|&x| (x as f64) / (1i64 << 32) as f64).collect::<Vec<_>>())
            }
            CpuData::Syntonic(_) => {
                // Free energy returns float metric, strict no-float policy
                return 0.0;
            }
        }
    }

    pub fn compute_tv_sum(&self) -> f64 {
        let cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        match cpu {
            CpuData::Float64(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
            }
            CpuData::Float32(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum()
            }
            CpuData::Complex128(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
            }
            CpuData::Int64(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum()
            }
            CpuData::GoldenExact(arr) => {
                let flat: Vec<_> = arr.iter().map(|x| x.to_f64()).collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
            }
            CpuData::Rational(arr) => {
                let flat: Vec<_> = arr.iter().map(|x| x.to_f64()).collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
            }
            CpuData::FixedPoint64(arr) => {
                let flat: Vec<_> = arr.iter().map(|&x| (x as f64) / (1i64 << 32) as f64).collect();
                if flat.len() < 2 {
                    return 0.0;
                }
                flat.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
            }
            CpuData::Syntonic(_) => {
                // TV sum returns float metric
                return 0.0;
            }
        }
    }

    pub fn differentiate(&self, alpha: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;

        match cpu {
            CpuData::Float64(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let values: Vec<f64> = arr.iter().cloned().collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| x * x).sum();
                if original_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let mut result = arr.clone();
                let mut rng = rand::thread_rng();
                let mean_amp = (original_energy / n as f64).sqrt();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    *x = *x + effective_alpha * noise * mode_weight * mean_amp;
                }

                let result_energy: f64 = result.iter().map(|x| x * x).sum();
                if result_energy > 1e-15 {
                    let scale = (original_energy / result_energy).sqrt();
                    for x in result.iter_mut() {
                        *x *= scale;
                    }
                }

                Ok(Self::wrap_cpu(CpuData::Float64(result), &self.device))
            }
            CpuData::Float32(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let values: Vec<f64> = arr.iter().map(|x| *x as f64).collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| (*x as f64).powi(2)).sum();
                if original_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let mut result = arr.clone();
                let mut rng = rand::thread_rng();
                let mean_amp = (original_energy / n as f64).sqrt();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    *x = *x + (effective_alpha * noise * mode_weight * mean_amp) as f32;
                }

                let result_energy: f64 = result.iter().map(|x| (*x as f64).powi(2)).sum();
                if result_energy > 1e-15 {
                    let scale = ((original_energy / result_energy).sqrt()) as f32;
                    for x in result.iter_mut() {
                        *x *= scale;
                    }
                }

                Ok(Self::wrap_cpu(CpuData::Float32(result), &self.device))
            }
            CpuData::Complex128(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let values: Vec<f64> = arr.iter().map(|x| x.norm()).collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| x.norm_sqr()).sum();
                if original_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let mean_amp = (original_energy / n as f64).sqrt();
                let mut result = arr.clone();
                let mut rng = rand::thread_rng();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise_real: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    let noise_imag: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    let noise = Complex64::new(noise_real, noise_imag) * mode_weight * mean_amp;
                    *x = *x + noise * effective_alpha;
                }

                let result_energy: f64 = result.iter().map(|x| x.norm_sqr()).sum();
                if result_energy > 1e-15 {
                    let scale = (original_energy / result_energy).sqrt();
                    for x in result.iter_mut() {
                        *x *= scale;
                    }
                }

                Ok(Self::wrap_cpu(CpuData::Complex128(result), &self.device))
            }
            CpuData::Int64(_) | CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::FixedPoint64(_) | CpuData::Syntonic(_) => {
                Ok(self.clone_storage())
            }
        }
    }

    pub fn harmonize(&self, strength: f64, legacy_gamma: f64) -> PyResult<TensorStorage> {
        // Enforce that if legacy_gamma is provided (non-zero), it matches strength or we warn
        if legacy_gamma != 0.0 && (legacy_gamma - strength).abs() > 1e-6 {
            // In a perfect world we'd warn, but for now we'll just prioritize strength (the first arg)
            // as it drives the logic below.
            // eprintln!("Warning: harmonize called with mismatching strength={} and gamma={}", strength, legacy_gamma);
        }
        let gamma = if (strength - Self::phi_inv()).abs() < 0.001 {
            Self::phi_inv()
        } else {
            strength
        };
        let cpu = self.ensure_cpu()?;

        match cpu {
            CpuData::Float64(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let total_energy: f64 = arr.iter().map(|x| x * x).sum();
                if total_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target: Vec<f64> = golden_weights
                    .iter()
                    .map(|w| (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt())
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let sign = if *x >= 0.0 { 1.0 } else { -1.0 };
                    let target_val = sign * target[i];
                    *x = (1.0 - gamma) * (*x) + gamma * target_val;
                }

                Ok(Self::wrap_cpu(CpuData::Float64(result), &self.device))
            }
            CpuData::Float32(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let total_energy: f64 = arr.iter().map(|x| (*x as f64).powi(2)).sum();
                if total_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target: Vec<f32> = golden_weights
                    .iter()
                    .map(|w| {
                        (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt() as f32
                    })
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let sign = if *x >= 0.0 { 1.0f32 } else { -1.0f32 };
                    let target_val = sign * target[i];
                    *x = (1.0 - gamma as f32) * (*x) + (gamma as f32) * target_val;
                }

                Ok(Self::wrap_cpu(CpuData::Float32(result), &self.device))
            }
            CpuData::Complex128(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                let total_energy: f64 = arr.iter().map(|x| x.norm_sqr()).sum();
                if total_energy < 1e-15 {
                    return Ok(self.clone_storage());
                }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target_amplitudes: Vec<f64> = golden_weights
                    .iter()
                    .map(|w| (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt())
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let phase = x.arg();
                    let target_val = Complex64::from_polar(target_amplitudes[i], phase);
                    *x = (*x) * (1.0 - gamma) + target_val * gamma;
                }

                Ok(Self::wrap_cpu(CpuData::Complex128(result), &self.device))
            }
            CpuData::Int64(_) | CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::FixedPoint64(_) | CpuData::Syntonic(_) => {
                Ok(self.clone_storage())
            }
        }
    }

    // ===== Core Reduction Operations =====

    /// Sum reduction over all elements or along an axis
    /// Returns scalar for full reduction, tensor for axis reduction
    #[pyo3(signature = (axis = None, keepdim = false))]
    pub fn sum_reduce(&self, axis: Option<i32>, keepdim: bool) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;

        if axis.is_none() {
            // Full reduction to scalar
            let sum = match &cpu {
                CpuData::Float64(arr) => arr.iter().sum::<f64>(),
                CpuData::Float32(arr) => arr.iter().map(|x| *x as f64).sum(),
                CpuData::Complex128(arr) => arr.iter().map(|x| x.re + x.im).sum(),
                CpuData::Int64(arr) => arr.iter().map(|x| *x as f64).sum(),
                CpuData::GoldenExact(arr) => arr.iter().map(|x| x.to_f64()).sum(),
                CpuData::Rational(arr) => arr.iter().map(|x| x.to_f64()).sum(),
                CpuData::FixedPoint64(arr) => arr.iter().map(|&x| (x as f64) / (1i64 << 32) as f64).sum(),
                CpuData::Syntonic(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "sum_reduce (float) not supported for SyntonicExact (use exact reduction)",
                    ))
                }
            };
            let shape = if keepdim { vec![1] } else { vec![1] };
            return Ok(TensorStorage {
                data: TensorData::Cpu(CpuData::Float64(ArrayD::from_elem(IxDyn(&shape), sum))),
                shape,
                device: self.device.clone(),
            });
        }

        // Axis reduction (simplified - full along outer dim)
        let axis_val = axis.unwrap() as usize;
        if axis_val >= self.shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Axis {} out of bounds for shape {:?}",
                axis_val, self.shape
            )));
        }

        match &cpu {
            CpuData::Float64(arr) => {
                let result = arr.sum_axis(ndarray::Axis(axis_val));
                let mut new_shape: Vec<usize> = result.shape().to_vec();
                if keepdim {
                    new_shape.insert(axis_val, 1);
                    let reshaped = result.into_shape(IxDyn(&new_shape)).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Self::wrap_cpu(CpuData::Float64(reshaped), &self.device))
                } else {
                    Ok(Self::wrap_cpu(
                        CpuData::Float64(result.into_dyn()),
                        &self.device,
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "sum_reduce axis not implemented for this dtype",
            )),
        }
    }

    /// Mean reduction over all elements or along an axis
    #[pyo3(signature = (axis = None, keepdim = false))]
    pub fn mean_reduce(&self, axis: Option<i32>, keepdim: bool) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;

        if axis.is_none() {
            let (sum, count) = match &cpu {
                CpuData::Float64(arr) => (arr.iter().sum::<f64>(), arr.len()),
                CpuData::Float32(arr) => (arr.iter().map(|x| *x as f64).sum(), arr.len()),
                CpuData::Complex128(arr) => (arr.iter().map(|x| x.re).sum(), arr.len()),
                CpuData::Int64(arr) => (arr.iter().map(|x| *x as f64).sum(), arr.len()),
                CpuData::GoldenExact(arr) => (arr.iter().map(|x| x.to_f64()).sum(), arr.len()),
                CpuData::Rational(arr) => (arr.iter().map(|x| x.to_f64()).sum(), arr.len()),
                CpuData::FixedPoint64(arr) => (arr.iter().map(|&x| (x as f64) / (1i64 << 32) as f64).sum(), arr.len()),
                CpuData::Syntonic(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "mean_reduce (float) not supported for SyntonicExact (use exact reduction)",
                    ))
                }
            };
            let mean = sum / count as f64;
            let shape = if keepdim { vec![1] } else { vec![1] };
            return Ok(TensorStorage {
                data: TensorData::Cpu(CpuData::Float64(ArrayD::from_elem(IxDyn(&shape), mean))),
                shape,
                device: self.device.clone(),
            });
        }

        // For axis mean, use sum then divide
        let sum_result = self.sum_reduce(axis, keepdim)?;
        let axis_size = self.shape[axis.unwrap() as usize] as f64;
        sum_result.div_scalar(axis_size)
    }

    // ===== Layer Normalization (SRT-aligned) =====

    /// Layer normalization with optional golden target variance
    ///
    /// Standard: normalize to mean=0, variance=1
    /// Golden target (golden_target=true): normalize to variance = 1/φ ≈ 0.618
    ///
    /// This aligns with the syntonic equilibrium where S* = 1/φ
    #[pyo3(signature = (weight = None, bias = None, eps = 1e-5, golden_target = true))]
    pub fn layer_norm(
        &self,
        weight: Option<&TensorStorage>,
        bias: Option<&TensorStorage>,
        eps: f64,
        golden_target: bool,
    ) -> PyResult<TensorStorage> {


        let cpu = self.ensure_cpu()?;

        match &cpu {
            CpuData::Float64(arr) => {
                let n = arr.len();
                if n == 0 {
                    return Ok(self.clone_storage());
                }

                // Compute mean
                let mean: f64 = arr.iter().sum::<f64>() / n as f64;

                // Compute variance
                let var: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

                // Compute reciprocal std
                let mut rstd = 1.0 / (var + eps).sqrt();

                // Golden scaling: target variance = 1/φ
                if golden_target {
                    rstd *= PHI_INV.sqrt();
                }

                // Normalize
                let normalized: Vec<f64> = arr.iter().map(|x| (x - mean) * rstd).collect();

                // Apply weight and bias if provided
                let result = match (weight, bias) {
                    (Some(w), Some(b)) => {
                        let w_cpu = w.ensure_cpu()?;
                        let b_cpu = b.ensure_cpu()?;
                        match (&w_cpu, &b_cpu) {
                            (CpuData::Float64(w_arr), CpuData::Float64(b_arr)) => normalized
                                .iter()
                                .zip(w_arr.iter())
                                .zip(b_arr.iter())
                                .map(|((x, w), b)| x * w + b)
                                .collect(),
                            _ => normalized,
                        }
                    }
                    (Some(w), None) => {
                        let w_cpu = w.ensure_cpu()?;
                        match &w_cpu {
                            CpuData::Float64(w_arr) => normalized
                                .iter()
                                .zip(w_arr.iter())
                                .map(|(x, w)| x * w)
                                .collect(),
                            _ => normalized,
                        }
                    }
                    (None, Some(b)) => {
                        let b_cpu = b.ensure_cpu()?;
                        match &b_cpu {
                            CpuData::Float64(b_arr) => normalized
                                .iter()
                                .zip(b_arr.iter())
                                .map(|(x, b)| x + b)
                                .collect(),
                            _ => normalized,
                        }
                    }
                    (None, None) => normalized,
                };

                let result_arr =
                    ArrayD::from_shape_vec(IxDyn(&self.shape), result).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(Self::wrap_cpu(CpuData::Float64(result_arr), &self.device))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "layer_norm currently only supports float64",
            )),
        }
    }

    // ===== Dropout =====

    /// Dropout: randomly zero elements during training
    /// At inference (training=false), returns identity
    /// Uses inverted dropout: active units scaled by 1/(1-p)
    #[pyo3(signature = (p = 0.5, training = true, seed = None))]
    pub fn dropout(&self, p: f64, training: bool, seed: Option<u64>) -> PyResult<TensorStorage> {
        if !training || p == 0.0 {
            return Ok(self.clone_storage());
        }

        if p < 0.0 || p >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout probability must be in [0, 1)",
            ));
        }

        let scale = 1.0 / (1.0 - p);
        let cpu = self.ensure_cpu()?;

        // Use seed or generate random
        let mut rng = match seed {
            Some(s) => {
                use rand::SeedableRng;
                rand::rngs::StdRng::seed_from_u64(s)
            }
            None => {
                use rand::SeedableRng;
                rand::rngs::StdRng::from_entropy()
            }
        };

        match &cpu {
            CpuData::Float64(arr) => {
                let result: Vec<f64> = arr
                    .iter()
                    .map(|x| {
                        let u: f64 = rng.gen();
                        if u < p {
                            0.0
                        } else {
                            x * scale
                        }
                    })
                    .collect();
                let result_arr =
                    ArrayD::from_shape_vec(IxDyn(&self.shape), result).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(Self::wrap_cpu(CpuData::Float64(result_arr), &self.device))
            }
            CpuData::Float32(arr) => {
                let result: Vec<f32> = arr
                    .iter()
                    .map(|x| {
                        let u: f64 = rng.gen();
                        if u < p {
                            0.0
                        } else {
                            x * scale as f32
                        }
                    })
                    .collect();
                let result_arr =
                    ArrayD::from_shape_vec(IxDyn(&self.shape), result).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(Self::wrap_cpu(CpuData::Float32(result_arr), &self.device))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dropout only supports float32/float64",
            )),
        }
    }

    // ========== Multi-GPU Support Methods ==========

    /// Scatter tensor across multiple GPUs
    /// Returns a list of tensors, one per specified device
    pub fn scatter_to_devices(&self, device_ids: Vec<usize>) -> PyResult<Vec<TensorStorage>> {
        scatter(self, &device_ids)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Gather tensors from multiple GPUs to a single device
    #[staticmethod]
    pub fn gather_from_devices(
        tensors: Vec<PyRef<'_, TensorStorage>>,
        target_device: usize,
    ) -> PyResult<TensorStorage> {
        let tensor_refs: Vec<TensorStorage> =
            tensors.iter().map(|t| (*t).clone_storage()).collect();
        gather(&tensor_refs, target_device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Copy tensor directly between GPUs (P2P if available)
    pub fn copy_to_gpu(&self, target_device: usize) -> PyResult<TensorStorage> {
        match &self.data {
            TensorData::Cuda { data, device, .. } => match data.as_ref() {
                CudaData::Float64(slice) => {
                    let src_device = device.ordinal();
                    let target_pool = get_pool(target_device).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    let new_slice = peer_copy(src_device, slice, target_device).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    let pooled = PooledSlice::new(new_slice, target_pool);

                    let target_dev = get_device(target_device).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(TensorStorage::new_from_cuda(
                        CudaData::Float64(Arc::new(pooled)),
                        target_dev,
                        self.shape.clone(),
                        target_device,
                    ))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "copy_to_gpu currently only supports float64 tensors",
                )),
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "copy_to_gpu requires source tensor to be on a CUDA device",
            )),
        }
    }

    /// Get multi-GPU topology information
    #[staticmethod]
    pub fn multi_gpu_info() -> PyResult<(usize,)> {
        let info = MultiGpuInfo::query()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok((info.device_count,))
    }

    // ========== Memory Pool Methods ==========

    /// Get memory pool statistics for a CUDA device
    #[staticmethod]
    pub fn pool_stats(device_idx: usize) -> PyResult<(usize, usize, usize, usize, usize)> {
        let pool = get_pool(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let stats = pool.stats();
        Ok((
            stats.cached_bytes,
            stats.cache_hits,
            stats.cache_misses,
            stats.total_allocations,
            stats.total_bytes_allocated,
        ))
    }

    /// Trim the memory pool for a CUDA device (release cached memory)
    #[staticmethod]
    pub fn pool_trim(device_idx: usize) -> PyResult<()> {
        let pool = get_pool(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        pool.trim();
        Ok(())
    }

    // ========== Async Transfer Methods ==========

    /// Start an async transfer to the specified CUDA device and return the transfer handle.
    /// The returned handle exposes diagnostics such as readiness, shape, and dtype.
    pub fn transfer_async(
        &self,
        py: Python<'_>,
        device: &str,
    ) -> PyResult<Py<AsyncTensorTransfer>> {
        let transfer = self
            .to_device_async(device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Py::new(py, transfer)
    }

    /// Synchronize a CUDA device (wait for all pending operations)
    #[staticmethod]
    pub fn sync_cuda_device(device_idx: usize) -> PyResult<()> {
        sync_device(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create a new CUDA stream for the specified device
    #[staticmethod]
    pub fn create_cuda_stream(device_idx: usize) -> PyResult<String> {
        let stream = create_stream(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(format!("cuda:{}/stream ({:?})", device_idx, stream))
    }

    /// Get stream kind names for documentation
    #[staticmethod]
    pub fn stream_kinds() -> Vec<String> {
        vec![
            format!("{:?}", StreamKind::Default),
            format!("{:?}", StreamKind::Transfer),
            format!("{:?}", StreamKind::Compute),
            format!("{:?}", StreamKind::Auxiliary),
        ]
    }

    /// Create a transfer-compute overlap helper for efficient pipelining
    #[staticmethod]
    pub fn create_overlap_helper(device_idx: usize) -> PyResult<String> {
        let overlap = TransferComputeOverlap::new(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(format!(
            "cuda:{}/overlap (device ordinal={})",
            device_idx,
            overlap.device().ordinal()
        ))
    }

    /// Allocate a pooled tensor (uses memory pool for efficiency)
    #[staticmethod]
    pub fn alloc_pooled_f64(count: usize, device_idx: usize) -> PyResult<TensorStorage> {
        let pool = get_pool(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let device = get_device(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let slice = PooledSlice::alloc(pool.clone(), count)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(TensorStorage::new_from_cuda(
            CudaData::Float64(Arc::new(slice)),
            device,
            vec![count],
            device_idx,
        ))
    }

    /// Get pool configuration info
    #[staticmethod]
    pub fn pool_config_info() -> (usize, usize, bool, usize) {
        let config = PoolConfig::default();
        (
            config.min_block_size,
            config.max_cached_bytes,
            config.round_to_power_of_2,
            config.max_cacheable_size,
        )
    }

    // ========== Multi-GPU Collective Operations ==========

    /// All-reduce operation across multiple GPUs
    ///
    /// Applies a reduction operation to tensors across GPUs and broadcasts
    /// the result back to all devices.
    ///
    /// Operations: "sum", "mean", "max", "min", "product"
    #[staticmethod]
    pub fn all_reduce_across_gpus(
        tensors: &Bound<'_, PyList>,
        op: &str,
    ) -> PyResult<Vec<TensorStorage>> {
        let reduce_op = match op.to_lowercase().as_str() {
            "sum" => ReduceOp::Sum,
            "mean" | "avg" | "average" => ReduceOp::Mean,
            "max" => ReduceOp::Max,
            "min" => ReduceOp::Min,
            "product" | "prod" | "mul" => ReduceOp::Product,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown reduce op '{}': use 'sum', 'mean', 'max', 'min', or 'product'",
                    op
                )))
            }
        };

        // Extract tensors
        let n = tensors.len();
        let mut tensor_vec: Vec<TensorStorage> = Vec::with_capacity(n);

        for i in 0..n {
            let item = tensors.get_item(i)?;
            let tensor_ref: pyo3::PyRef<'_, TensorStorage> = item.extract()?;
            tensor_vec.push(tensor_ref.clone_storage_internal());
        }

        // Perform all-reduce (modifies in place)
        super::cuda::multi_gpu::all_reduce(&mut tensor_vec, reduce_op)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(tensor_vec)
    }

    /// Print multi-GPU topology information
    #[staticmethod]
    pub fn print_gpu_topology() -> PyResult<String> {
        let info = MultiGpuInfo::query()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        info.print_topology(); // Prints to stdout
        Ok(format!("{} GPU(s) available", info.device_count))
    }

    /// Clear CUDA caches (memory pools, streams) for memory management
    ///
    /// Useful for:
    /// - Releasing memory under pressure
    /// - Testing clean state scenarios
    /// - Forcing fresh allocations
    #[staticmethod]
    pub fn clear_cuda_caches() -> PyResult<()> {
        get_local_manager().with(|manager| {
            manager.clear_caches();
        });
        Ok(())
    }

    // ========== Direct CUDA Infrastructure Methods ==========

    /// Copy tensor directly between GPUs using peer_copy
    pub fn copy_between_gpus(&self, target_device: usize) -> PyResult<TensorStorage> {
        match &self.data {
            TensorData::Cuda { data, device, .. } => match data.as_ref() {
                CudaData::Float64(src_slice) => {
                    let src_device = device.ordinal();
                    let target_pool = get_pool(target_device).map_err(|e: CudaError| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    let dst_slice = peer_copy(src_device, src_slice, target_device).map_err(
                        |e: CudaError| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        },
                    )?;

                    let pooled = PooledSlice::new(dst_slice, target_pool);

                    let target_dev = get_device(target_device).map_err(|e: CudaError| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(TensorStorage::new_from_cuda(
                        CudaData::Float64(Arc::new(pooled)),
                        target_dev,
                        self.shape.clone(),
                        target_device,
                    ))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "copy_between_gpus only supports f64 CUDA tensors",
                )),
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Source tensor must be on CUDA device",
            )),
        }
    }

    /// Scatter tensor to multiple GPUs using scatter function
    pub fn scatter_multi_gpu(&self, device_ids: Vec<usize>) -> PyResult<Vec<TensorStorage>> {
        scatter(self, &device_ids).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// Gather tensors from multiple GPUs to target device using gather function
    #[staticmethod]
    pub fn gather_multi_gpu(
        tensors: &Bound<'_, PyList>,
        target_device: usize,
    ) -> PyResult<TensorStorage> {
        let n = tensors.len();
        let mut tensor_vec: Vec<TensorStorage> = Vec::with_capacity(n);
        for i in 0..n {
            let item = tensors.get_item(i)?;
            let tensor_ref: pyo3::PyRef<'_, TensorStorage> = item.extract()?;
            tensor_vec.push(tensor_ref.clone_storage_internal());
        }
        gather(&tensor_vec, target_device).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// Allocate using PooledSlice and return a TensorStorage
    /// Demonstrates explicit use of PooledSlice construction
    #[staticmethod]
    pub fn alloc_with_pooled_slice(count: usize, device_idx: usize) -> PyResult<TensorStorage> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let device = get_device(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Allocate raw slice and wrap in PooledSlice
        let raw_slice: CudaSlice<f64> = pool.alloc_f64(count).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let pooled: PooledSlice<f64> = PooledSlice::new(raw_slice, pool.clone());

        // Use PooledSlice methods
        let len = pooled.len();
        // Validate pooled slice properties (retain for potential diagnostics)
        let is_empty = pooled.is_empty();
        let slice_ref = pooled.as_slice();
        debug_assert_eq!(slice_ref.len(), len);
        let pool_ref = pooled.pool(); // Access the pool field
        if is_empty {
            // unlikely for a freshly allocated pooled slice; keep as sanity check
            eprintln!(
                "alloc_with_pooled_slice: allocated pooled slice is empty on device {}",
                device_idx
            );
        }
        // Clone & drop to explicitly release an owned Arc rather than dropping a reference
        let pool_owned = pool_ref.clone();
        drop(pool_owned);

        Ok(TensorStorage::new_from_cuda(
            CudaData::Float64(Arc::new(pooled)),
            device,
            vec![len],
            device_idx,
        ))
    }

    /// Get detailed pool statistics using PoolStats
    #[staticmethod]
    pub fn detailed_pool_stats(device_idx: usize) -> PyResult<String> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let stats: PoolStats = pool.stats();
        Ok(format!(
            "PoolStats {{ cached_bytes: {}, cache_hits: {}, cache_misses: {}, total_allocations: {}, total_bytes: {} }}",
            stats.cached_bytes, stats.cache_hits, stats.cache_misses,
            stats.total_allocations, stats.total_bytes_allocated
        ))
    }

    /// Get device count using DeviceManager
    #[staticmethod]
    pub fn cuda_device_count_from_manager() -> usize {
        DeviceManager::device_count()
    }

    /// Transfer to device asynchronously, returning full AsyncTensorTransfer info
    pub fn transfer_async_full(&self, device_str: &str) -> PyResult<(String, Vec<usize>, String)> {
        let transfer: AsyncTensorTransfer =
            self.to_device_async(device_str).map_err(|e: CudaError| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;
        Ok((
            format!("cuda:{}", transfer.device_index()),
            transfer.tensor_shape().to_vec(),
            transfer.dtype_str().to_string(),
        ))
    }

    /// Create an async transfer handle for H2D transfer
    #[staticmethod]
    pub fn create_async_transfer(data: Vec<f64>, device_idx: usize) -> PyResult<(usize, String)> {
        // Create an AsyncTransfer handle to track device operations
        let transfer = AsyncTransfer::new(device_idx);
        let len = data.len();

        // Actually do the transfer via device
        let device = get_device(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let cuda_slice = device
            .default_stream()
            .clone_htod(&data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Wait for transfer to complete
        transfer.wait().map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Drop the device slice after the transfer completes to release device memory
        drop(cuda_slice);

        Ok((len, format!("cuda:{}", device_idx)))
    }

    /// Compute symbolic q-correction factor using SymExpr
    /// Returns (symbolic_string, numeric_value) tuple
    #[staticmethod]
    pub fn symbolic_q_correction(n: u32, sign: i8) -> (String, f64) {
        let one = SymExpr::from_int(1);
        let q = SymExpr::q();
        let n_expr = SymExpr::from_int(n as i128);
        let q_over_n = q.div(n_expr);

        let result = if sign >= 0 {
            one.add(q_over_n)
        } else {
            one.sub(q_over_n)
        };

        // Use Display trait (to_string) for formatting
        (result.to_string(), result.eval_f64())
    }

    /// Get the StreamKind names as documentation
    #[staticmethod]
    pub fn available_stream_kinds() -> Vec<String> {
        vec![
            format!("StreamKind::{:?}", StreamKind::Default),
            format!("StreamKind::{:?}", StreamKind::Transfer),
            format!("StreamKind::{:?}", StreamKind::Compute),
            format!("StreamKind::{:?}", StreamKind::Auxiliary),
        ]
    }

    /// Prefetch tensor data to a device (async start)
    /// Uses AsyncTensorOps trait method
    pub fn prefetch_to_device(&self, device: &str) -> PyResult<()> {
        self.prefetch(device).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// Transfer with compute overlap example
    /// Demonstrates using TransferComputeOverlap
    #[staticmethod]
    pub fn transfer_with_overlap(data: Vec<f64>, device_idx: usize) -> PyResult<String> {
        let overlap = TransferComputeOverlap::new(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Use transfer method
        let slice = overlap.transfer(&data).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Use sync method
        overlap.sync().map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Use device method
        let device = overlap.device();

        // Drop the temporary device slice after sync to release memory
        drop(slice);

        Ok(format!(
            "Transferred {} f64s to device {}",
            data.len(),
            device.ordinal()
        ))
    }

    /// Transfer with async tracking - waits for completion
    /// Demonstrates AsyncTensorTransfer.is_ready() and wait()
    pub fn transfer_async_with_wait(&self, device_str: &str) -> PyResult<(String, bool)> {
        let mut transfer: AsyncTensorTransfer =
            self.to_device_async(device_str).map_err(|e: CudaError| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

        // Check if ready before waiting
        let was_ready = transfer.is_ready();

        // Wait for completion
        transfer.wait().map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        Ok((format!("cuda:{}", transfer.device_index()), was_ready))
    }

    /// SRT-optimized transfer to CUDA (async kernel path with device sync)
    /// Uses golden ratio batching and pinned memory pooling for improved throughput
    pub fn to_cuda_async_srt(&self, device_idx: usize) -> PyResult<TensorStorage> {
        // Ensure CPU data available
        let cpu_data = self
            .ensure_cpu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Create overlap manager which provides SRT transfer helpers
        let overlap = TransferComputeOverlap::new(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Get pool for wrapping allocations
        let pool = get_pool(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Execute SRT-optimized transfer based on dtype
        let (cuda_data, _) = match cpu_data {
            CpuData::Float32(arr) => {
                let slice = overlap.transfer_f32(arr.as_slice().unwrap()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let pooled = Arc::new(PooledSlice::new(slice, pool.clone()));
                (CudaData::Float32(pooled), "float32".to_string())
            }
            CpuData::Float64(arr) => {
                let slice = overlap.transfer_f64(arr.as_slice().unwrap()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let pooled = Arc::new(PooledSlice::new(slice, pool.clone()));
                (CudaData::Float64(pooled), "float64".to_string())
            }
            CpuData::Complex128(arr) => {
                let complex_slice = arr.as_slice().unwrap();
                // Cast Complex64 to CudaComplex64 (safe wrapper)
                let cuda_complex_slice: &[CudaComplex64] = unsafe {
                    std::slice::from_raw_parts(
                        complex_slice.as_ptr() as *const CudaComplex64,
                        complex_slice.len(),
                    )
                };

                // Generic async H2D transfer (bypass SRT opt used for floats)
                use super::cuda::async_transfer::htod_async;
                let slice = htod_async(overlap.device(), cuda_complex_slice).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;

                let pooled = Arc::new(PooledSlice::new(slice, pool.clone()));
                (CudaData::Complex128(pooled), "complex128".to_string())
            }
            CpuData::Int64(_) | CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Int64 and exact types not supported on CUDA",
                ))
            }
            CpuData::FixedPoint64(arr) => {
                // Transfer Q32.32 fixed-point as i64 to GPU - use simple H2D transfer
                let device = get_device(device_idx)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let mut slice = PooledSlice::alloc(pool.clone(), arr.len()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                device
                    .default_stream()
                    .memcpy_htod(arr.as_slice().unwrap(), slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                let pooled = Arc::new(slice);
                (CudaData::FixedPoint64(pooled), "fixed_point64".to_string())
            }
        };

        // Ensure transfer completion
        overlap
            .sync()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Build TensorStorage on CUDA device
        let device = get_device(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(TensorStorage::new_from_cuda(
            cuda_data,
            device,
            self.shape.clone(),
            device_idx,
        ))
    }

    /// SRT-optimized transfer to CPU (async path with device sync)
    /// Uses golden ratio batching and pinned memory pooling for improved throughput
    pub fn to_cpu_async_srt(&self, device_idx: usize) -> PyResult<TensorStorage> {
        let cuda_data = match &self.data {
            TensorData::Cuda { data, .. } => data.as_ref(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Not on CUDA device",
                ))
            }
        };

        // Create overlap manager which provides SRT transfer helpers
        let overlap = TransferComputeOverlap::new(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Execute SRT-optimized D2H transfer based on dtype
        let cpu_data = match cuda_data {
            CudaData::Float32(slice) => {
                let host_vec = overlap.receive_f32(slice).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                CpuData::Float32(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), host_vec)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?,
                )
            }
            CudaData::Float64(slice) => {
                let host_vec = overlap.receive_f64(slice).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                CpuData::Float64(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), host_vec)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?,
                )
            }
            CudaData::Complex128(slice) => {
                // Use generic dtoh_async since we don't have srt_d2h_transfer_c128 yet
                use super::cuda::async_transfer::dtoh_async;

                let mut host_vec_c: Vec<CudaComplex64> =
                    vec![CudaComplex64::default(); slice.len()];

                dtoh_async(overlap.device(), &**slice, &mut host_vec_c).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;

                // Convert Vec<CudaComplex64> to Vec<Complex64> safely
                // They are layout compatible (transparent wrapper)
                let complex_vec = unsafe {
                    let ptr = host_vec_c.as_ptr() as *mut num_complex::Complex64;
                    let len = host_vec_c.len();
                    let cap = host_vec_c.capacity();
                    std::mem::forget(host_vec_c);
                    Vec::from_raw_parts(ptr, len, cap)
                };

                CpuData::Complex128(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), complex_vec)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?,
                )
            }
            CudaData::Int64(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Async D2H for Int64 not supported yet",
                ));
            }
            CudaData::FixedPoint64(slice) => {
                // Transfer Q32.32 fixed-point from GPU as i64 - use simple D2H transfer
                let device = get_device(device_idx)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let mut host_vec = vec![0i64; slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_vec)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                // Truncate to actual shape size
                host_vec.truncate(self.shape.iter().product());
                CpuData::FixedPoint64(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), host_vec)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?,
                )
            }
        };

        // Ensure transfer completion
        overlap
            .sync()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(TensorStorage {
            data: TensorData::Cpu(cpu_data),
            shape: self.shape.clone(),
            device: DeviceType::Cpu,
        })
    }

    /// Copy data back from device to host asynchronously
    /// Uses dtoh_async function
    pub fn copy_to_host_async(&self) -> PyResult<Vec<f64>> {
        use super::cuda::async_transfer::dtoh_async;

        match &self.data {
            TensorData::Cuda { data, device, .. } => match data.as_ref() {
                CudaData::Float64(slice) => {
                    let mut host_buf = vec![0.0f64; slice.len()];
                    dtoh_async(device, slice, &mut host_buf).map_err(|e: CudaError| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(host_buf)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Only f64 CUDA tensors supported for async D2H",
                )),
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Tensor must be on CUDA device",
            )),
        }
    }

    /// Allocate with pooled slice using mutable operations
    /// Uses PooledSlice.as_slice_mut() and pool field
    #[staticmethod]
    pub fn alloc_and_fill_pooled(
        count: usize,
        fill_value: f64,
        device_idx: usize,
    ) -> PyResult<TensorStorage> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let device = get_device(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Allocate PooledSlice
        let mut pooled = PooledSlice::alloc(pool, count)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Fill with values on host then copy
        let host_data: Vec<f64> = vec![fill_value; count];
        device
            .default_stream()
            .memcpy_htod(&host_data, pooled.as_slice_mut())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(TensorStorage::new_from_cuda(
            CudaData::Float64(Arc::new(pooled)),
            device,
            vec![count],
            device_idx,
        ))
    }

    /// Get memory pool info including internal details
    /// Uses MemoryPool's device_idx and config fields
    #[staticmethod]
    pub fn pool_full_info(device_idx: usize) -> PyResult<String> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Use device_idx() method
        let dev_idx = pool.device_idx();
        // Use device() method
        let device = pool.device();
        // Use stats()
        let stats = pool.stats();

        Ok(format!(
            "MemoryPool {{ device_idx: {}, device_ordinal: {}, cached_bytes: {}, hits: {}, misses: {} }}",
            dev_idx, device.ordinal(), stats.cached_bytes, stats.cache_hits, stats.cache_misses
        ))
    }

    /// Trim pool to release memory
    /// Uses MemoryPool.trim()
    #[staticmethod]
    pub fn pool_trim_all(device_idx: usize) -> PyResult<()> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        pool.trim();
        Ok(())
    }

    /// Allocate raw bytes and return to pool
    /// Uses MemoryPool.alloc_bytes() and free_bytes()
    #[staticmethod]
    pub fn pool_alloc_free_test(size: usize, device_idx: usize) -> PyResult<String> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Allocate bytes
        let slice = pool.alloc_bytes(size).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let allocated_len = slice.len();

        // Free bytes (returns to pool)
        pool.free_bytes(slice);

        Ok(format!(
            "Allocated {} bytes, freed back to pool",
            allocated_len
        ))
    }

    /// Allocate f32 memory from pool
    /// Uses MemoryPool.alloc_f32()
    #[staticmethod]
    pub fn alloc_pooled_f32(count: usize, device_idx: usize) -> PyResult<TensorStorage> {
        let pool: Arc<MemoryPool> = get_pool(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;
        let device = get_device(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let slice = PooledSlice::alloc(pool, count)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(TensorStorage::new_from_cuda(
            CudaData::Float32(Arc::new(slice)),
            device,
            vec![count],
            device_idx,
        ))
    }

    /// Create a pool with custom configuration
    /// Uses MemoryPool::with_config() and config field
    #[staticmethod]
    pub fn create_custom_pool_info(device_idx: usize) -> PyResult<String> {
        let device = get_device(device_idx).map_err(|e: CudaError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        // Create custom config
        let config = PoolConfig {
            min_block_size: 512,
            max_cached_bytes: 1024 * 1024, // 1MB
            round_to_power_of_2: true,
            max_cacheable_size: 256 * 1024, // 256KB
        };

        // Create pool with custom config
        let pool = MemoryPool::with_config(device_idx, device.clone(), config);

        // Use pool methods to verify it works
        let stats = pool.stats();
        let dev_idx = pool.device_idx();

        Ok(format!(
            "Custom pool created: device_idx={}, cached_bytes={}",
            dev_idx, stats.cached_bytes
        ))
    }
}

// Private implementation
impl TensorStorage {
    // All constants derived from exact symbolic infrastructure
    // No hardcoded floating-point values

    /// φ - derived from GoldenExact::phi()
    #[inline]
    fn phi() -> f64 {
        GoldenExact::phi().to_f64()
    }

    /// φ⁻¹ = φ - 1 - derived from GoldenExact::phi_hat()
    #[inline]
    fn phi_inv() -> f64 {
        GoldenExact::phi_hat().to_f64()
    }

    /// q - the universal syntony deficit - derived from FundamentalConstant::Q
    #[inline]
    fn q_deficit() -> f64 {
        FundamentalConstant::Q.approx_f64()
    }

    /// E* = e^π - π - derived from FundamentalConstant::EStar
    #[inline]
    fn e_star() -> f64 {
        FundamentalConstant::EStar.approx_f64()
    }

    /// φ^k - derived from exact Fibonacci formula via GoldenExact::phi_power(k)
    #[inline]
    fn phi_power(k: i32) -> f64 {
        GoldenExact::phi_power(k).to_f64()
    }

    /// Symbolic correction (1 + sign × q/N) evaluated to f64
    #[inline]
    fn correction_factor(n: u32, sign: i8) -> f64 {
        let one = SymExpr::from_int(1);
        let q = SymExpr::q();
        let n_expr = SymExpr::from_int(n as i128);
        let q_over_n = q.div(n_expr);

        if sign >= 0 {
            one.add(q_over_n).eval_f64()
        } else {
            one.sub(q_over_n).eval_f64()
        }
    }

    /// Generic binary CUDA operation (add, sub, mul, div)
    fn binary_cuda_op(
        &self,
        a: &Arc<CudaData>,
        b: &Arc<CudaData>,
        device: &Arc<CudaDevice>,
        op: &str,
    ) -> PyResult<TensorStorage> {
        let n = self.shape.iter().product::<usize>();
        let cfg = launch_cfg(n);

        // Get pool for allocations
        let pool = get_pool(device.ordinal() as usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Get cached module and functions
        let (major, minor) = get_device_compute_capability(device);
        let ptx_source = select_ptx(major, minor);
        let (module, functions) =
            get_cached_module_and_functions(device, ptx_source).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "CUDA kernel loading failed: {}",
                    e
                ))
            })?;
        // Keep module alive for the duration of the kernel launch
        let _ = module;

        let (out_data, out_dtype) = match (a.as_ref(), b.as_ref()) {
            (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
                let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let func_name = format!("{}_f64", op);
                let func = functions.get(&func_name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Function {} not found in cached module",
                        func_name
                    ))
                })?;
                unsafe {
                    device
                        .default_stream()
                        .launch_builder(&func)
                        .arg(out.as_slice_mut())
                        .arg(a_slice.as_slice())
                        .arg(b_slice.as_slice())
                        .arg(&(n as i32))
                        .launch(cfg)
                }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float64(Arc::new(out)), "float64".to_string())
            }
            (CudaData::Float32(a_slice), CudaData::Float32(b_slice)) => {
                let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let func_name = format!("{}_f32", op);
                let func = functions.get(&func_name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Function {} not found in cached module",
                        func_name
                    ))
                })?;
                unsafe {
                    device
                        .default_stream()
                        .launch_builder(&func)
                        .arg(out.as_slice_mut())
                        .arg(a_slice.as_slice())
                        .arg(b_slice.as_slice())
                        .arg(&(n as i32))
                        .launch(cfg)
                }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float32(Arc::new(out)), "float32".to_string())
            }
            (CudaData::Complex128(a_slice), CudaData::Complex128(b_slice)) => {
                // Special case: use dedicated kernel for complex division
                if op == "div" {
                    let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    crate::tensor::srt_kernels::cuda_complex_div_c128(
                        device,
                        out.as_slice_mut(),
                        a_slice.as_slice(),
                        b_slice.as_slice(),
                        n,
                    )?;

                    (
                        CudaData::Complex128(Arc::new(out)),
                        "complex128".to_string(),
                    )
                } else {
                    // Generic path for other operations
                    let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    let func_name = format!("{}_c128", op);
                    let func = functions.get(&func_name).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Function {} not found in cached module",
                            func_name
                        ))
                    })?;
                    unsafe {
                        device
                            .default_stream()
                            .launch_builder(&func)
                            .arg(out.as_slice_mut())
                            .arg(a_slice.as_slice())
                            .arg(b_slice.as_slice())
                            .arg(&(n as i32))
                            .launch(cfg)
                    }
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    (
                        CudaData::Complex128(Arc::new(out)),
                        "complex128".to_string(),
                    )
                }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch on CUDA",
                ))
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(out_data),
                device: device.clone(),
                shape: self.shape.clone(),
                dtype: out_dtype,
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    /// Generic unary CUDA operation (neg, abs)
    fn unary_cuda_op(
        &self,
        a: &Arc<CudaData>,
        device: &Arc<CudaDevice>,
        op: &str,
    ) -> PyResult<TensorStorage> {
        let n = self.shape.iter().product::<usize>();
        let cfg = launch_cfg(n);

        // Get pool for allocations
        let pool = get_pool(device.ordinal() as usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Get cached module and functions
        let (major, minor) = get_device_compute_capability(device);
        let ptx_source = select_ptx(major, minor);
        let (module, functions) =
            get_cached_module_and_functions(device, ptx_source).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "CUDA kernel loading failed: {}",
                    e
                ))
            })?;
        // Keep module alive for the duration of the kernel launch
        let _ = module;

        let (out_data, out_dtype) = match a.as_ref() {
            CudaData::Float64(a_slice) => {
                let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let func_name = format!("{}_f64", op);
                let func = functions.get(&func_name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Function {} not found in cached module",
                        func_name
                    ))
                })?;
                unsafe {
                    device
                        .default_stream()
                        .launch_builder(&func)
                        .arg(out.as_slice_mut())
                        .arg(a_slice.as_slice())
                        .arg(&(n as i32))
                        .launch(cfg)
                }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float64(Arc::new(out)), "float64".to_string())
            }
            CudaData::Float32(a_slice) => {
                let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let func_name = format!("{}_f32", op);
                let func = functions.get(&func_name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Function {} not found in cached module",
                        func_name
                    ))
                })?;
                unsafe {
                    device
                        .default_stream()
                        .launch_builder(&func)
                        .arg(out.as_slice_mut())
                        .arg(a_slice.as_slice())
                        .arg(&(n as i32))
                        .launch(cfg)
                }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float32(Arc::new(out)), "float32".to_string())
            }
            CudaData::Complex128(a_slice) => {
                // For neg_c128, output is complex. For abs, would need different handling
                if op == "neg" {
                    let mut out = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    let func_name = "neg_c128".to_string();
                    let func = functions.get(&func_name).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Function {} not found in cached module",
                            func_name
                        ))
                    })?;
                    unsafe {
                        device
                            .default_stream()
                            .launch_builder(&func)
                            .arg(out.as_slice_mut())
                            .arg(a_slice.as_slice())
                            .arg(&(n as i32))
                            .launch(cfg)
                    }
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    (
                        CudaData::Complex128(Arc::new(out)),
                        "complex128".to_string(),
                    )
                } else {
                    // abs on complex: fall back to CPU (complex abs not implemented in CUDA yet)
                    return self.unary_cpu_fallback(op);
                }
            }
            CudaData::Int64(_) => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "CUDA Int64 unary op not supported yet",
                ))?
            }
            CudaData::FixedPoint64(_) => {
                // Fall back to CPU for fixed-point operations
                return self.unary_cpu_fallback(op);
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(out_data),
                device: device.clone(),
                shape: self.shape.clone(),
                dtype: out_dtype,
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    fn unary_cpu_fallback(&self, op: &str) -> PyResult<TensorStorage> {
        // Transfer to CPU and perform operation there
        let cpu_data = self.ensure_cpu()?;
        let result = match (op, &cpu_data) {
            ("exp", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.exp())),
            ("exp", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.exp())),
            ("exp", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| {
                let exp_re = c.re.exp();
                Complex64::new(exp_re * c.im.cos(), exp_re * c.im.sin())
            })),
            ("exp", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).exp())),

            ("log", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.ln())),
            ("log", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.ln())),
            ("log", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| c.ln())),
            ("log", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).ln())),

            ("sin", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.sin())),
            ("sin", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.sin())),
            ("sin", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| c.sin())),
            ("sin", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).sin())),

            ("cos", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.cos())),
            ("cos", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.cos())),
            ("cos", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| c.cos())),
            ("cos", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).cos())),

            ("sqrt", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.sqrt())),
            ("sqrt", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.sqrt())),
            ("sqrt", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| c.sqrt())),
            ("sqrt", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).sqrt())),

            ("tanh", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.tanh())),
            ("tanh", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.tanh())),
            ("tanh", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| c.tanh())),
            ("tanh", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| (x as f64).tanh())),

            ("sigmoid", CpuData::Float64(arr)) => {
                CpuData::Float64(arr.mapv(|x| 1.0 / (1.0 + (-x).exp())))
            }
            ("sigmoid", CpuData::Float32(arr)) => {
                CpuData::Float32(arr.mapv(|x| 1.0 / (1.0 + (-x).exp())))
            }
            ("sigmoid", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| {
                let exp_neg_c = (-c).exp();
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + exp_neg_c)
            })),
            ("sigmoid", CpuData::Int64(arr)) => CpuData::Float64(arr.mapv(|x| {
                let x_f64 = x as f64;
                1.0 / (1.0 + (-x_f64).exp())
            })),

            ("relu", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.max(0.0))),
            ("relu", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.max(0.0))),
            ("relu", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| {
                if c.re > 0.0 {
                    c
                } else {
                    Complex64::new(0.0, c.im)
                }
            })),
            ("relu", CpuData::Int64(arr)) => CpuData::Int64(arr.mapv(|x| x.max(0))),

            ("neg", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| -x)),
            ("neg", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| -x)),
            ("neg", CpuData::Complex128(arr)) => CpuData::Complex128(arr.mapv(|c| -c)),
            ("neg", CpuData::Int64(arr)) => CpuData::Int64(arr.mapv(|x| -x)),

            ("abs", CpuData::Float64(arr)) => CpuData::Float64(arr.mapv(|x| x.abs())),
            ("abs", CpuData::Float32(arr)) => CpuData::Float32(arr.mapv(|x| x.abs())),
            ("abs", CpuData::Complex128(arr)) => CpuData::Float64(arr.mapv(|x| x.norm())),
            ("abs", CpuData::Int64(arr)) => CpuData::Int64(arr.mapv(|x| x.abs())),

            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    format!("CPU fallback not implemented for operation: {}", op),
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    fn binary_cpu_fallback(
        &self,
        a: &Arc<CudaData>,
        b: &Arc<CudaData>,
        device: &Arc<CudaDevice>,
        op: &str,
    ) -> PyResult<TensorStorage> {
        // Transfer both tensors to CPU and perform operation there
        let a_cpu = TensorStorage::cuda_to_cpu(a, device, &self.shape)?;
        let b_cpu = TensorStorage::cuda_to_cpu(b, device, &self.shape)?;

        let result = match (op, &a_cpu, &b_cpu) {
            ("add", CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                CpuData::Float64(a_arr + b_arr)
            }
            ("add", CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                CpuData::Float32(a_arr + b_arr)
            }
            ("add", CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                CpuData::Complex128(a_arr + b_arr)
            }
            ("add", CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(a_arr + b_arr),

            ("sub", CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                CpuData::Float64(a_arr - b_arr)
            }
            ("sub", CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                CpuData::Float32(a_arr - b_arr)
            }
            ("sub", CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                CpuData::Complex128(a_arr - b_arr)
            }
            ("sub", CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(a_arr - b_arr),

            ("mul", CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                CpuData::Float64(a_arr * b_arr)
            }
            ("mul", CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                CpuData::Float32(a_arr * b_arr)
            }
            ("mul", CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                CpuData::Complex128(a_arr * b_arr)
            }
            ("mul", CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(a_arr * b_arr),

            ("div", CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                CpuData::Float64(a_arr / b_arr)
            }
            ("div", CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                CpuData::Float32(a_arr / b_arr)
            }
            ("div", CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                CpuData::Complex128(a_arr / b_arr)
            }
            ("div", CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(a_arr / b_arr),

            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Dtype mismatch in CPU fallback",
                ))
            }
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    /// Wrap CPU data with specified device
    fn wrap_cpu(data: CpuData, device: &DeviceType) -> Self {
        let shape = match &data {
            CpuData::Float32(a) => a.shape().to_vec(),
            CpuData::Float64(a) => a.shape().to_vec(),
            CpuData::Complex128(a) => a.shape().to_vec(),
            CpuData::Int64(a) => a.shape().to_vec(),
            CpuData::GoldenExact(a) => a.shape().to_vec(),
            CpuData::Rational(a) => a.shape().to_vec(),
            CpuData::FixedPoint64(a) => a.shape().to_vec(),
            CpuData::Syntonic(a) => a.shape().to_vec(),
        };
        Self::new_from_cpu(data, shape, device.clone())
    }

    /// Wrap CPU data with default CPU device
    fn wrap_cpu_data(data: CpuData) -> Self {
        Self::wrap_cpu(data, &DeviceType::Cpu)
    }

    fn clone_storage(&self) -> TensorStorage {
        TensorStorage {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }

    fn ensure_cpu(&self) -> PyResult<CpuData> {
        match &self.data {
            TensorData::Cpu(cpu) => Ok(cpu.clone()),
            TensorData::Cuda {
                data,
                device,
                shape,
                dtype: _,
            } => {
                // Transfer from GPU to CPU
                Self::cuda_to_cpu(data, device, shape)
            }
        }
    }

    fn cpu_to_cuda(
        cpu_data: CpuData,
        shape: Vec<usize>,
        device_idx: usize,
    ) -> PyResult<TensorStorage> {
        // Use DeviceManager for cached device handles
        let device = get_device(device_idx)?;
        let pool = get_pool(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let (cuda_data, dtype) = match cpu_data {
            CpuData::Float32(arr) => {
                // Simple synchronous H2D transfer via default stream
                let mut slice = PooledSlice::alloc(pool.clone(), arr.len()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                device
                    .default_stream()
                    .memcpy_htod(arr.as_slice().unwrap(), slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                (CudaData::Float32(Arc::new(slice)), "float32".to_string())
            }
            CpuData::Float64(arr) => {
                // Simple synchronous H2D transfer via default stream
                let mut slice = PooledSlice::alloc(pool.clone(), arr.len()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                device
                    .default_stream()
                    .memcpy_htod(arr.as_slice().unwrap(), slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                (CudaData::Float64(Arc::new(slice)), "float64".to_string())
            }
            CpuData::Complex128(arr) => {
                let complex_slice = arr.as_slice().unwrap();
                // Cast Complex64 to CudaComplex64 (safe wrapper)
                let cuda_complex_slice: &[CudaComplex64] = unsafe {
                    std::slice::from_raw_parts(
                        complex_slice.as_ptr() as *const CudaComplex64,
                        complex_slice.len(),
                    )
                };

                let mut slice =
                    PooledSlice::alloc(pool.clone(), complex_slice.len()).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                device
                    .default_stream()
                    .memcpy_htod(cuda_complex_slice, slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                (
                    CudaData::Complex128(Arc::new(slice)),
                    "complex128".to_string(),
                )
            }
            CpuData::Int64(arr) => {
                let mut slice = PooledSlice::alloc(pool.clone(), arr.len()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                device
                    .default_stream()
                    .memcpy_htod(arr.as_slice().unwrap(), slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                (CudaData::Int64(Arc::new(slice)), "int64".to_string())
            }
            CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::Syntonic(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "GoldenExact, Rational, and Syntonic types not supported on CUDA",
                ))
            }
            CpuData::FixedPoint64(arr) => {
                // Transfer Q32.32 fixed-point as i64 to GPU
                let mut slice = PooledSlice::alloc(pool.clone(), arr.len()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                device
                    .default_stream()
                    .memcpy_htod(arr.as_slice().unwrap(), slice.as_slice_mut())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                (CudaData::FixedPoint64(Arc::new(slice)), "fixed_point64".to_string())
            }
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(cuda_data),
                device,
                shape: shape.clone(),
                dtype,
            },
            shape,
            device: DeviceType::Cuda(device_idx),
        })
    }

    fn cuda_to_cpu(
        data: &Arc<CudaData>,
        device: &Arc<CudaDevice>,
        shape: &[usize],
    ) -> PyResult<CpuData> {
        let dim = IxDyn(shape);
        let shape_product: usize = shape.iter().product();

        match data.as_ref() {
            CudaData::Float32(slice) => {
                // Simple synchronous D2H transfer via default stream
                // Note: slice.len() may be larger than shape_product due to pool bucket rounding
                let mut host_data = vec![0.0f32; slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_data)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                // Truncate to actual shape size (pool may allocate larger buffers)
                host_data.truncate(shape_product);
                Ok(CpuData::Float32(
                    ArrayD::from_shape_vec(dim, host_data).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?,
                ))
            }
            CudaData::Float64(slice) => {
                // Simple synchronous D2H transfer via default stream
                // Note: slice.len() may be larger than shape_product due to pool bucket rounding
                let mut host_data = vec![0.0f64; slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_data)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                // Truncate to actual shape size (pool may allocate larger buffers)
                host_data.truncate(shape_product);
                Ok(CpuData::Float64(
                    ArrayD::from_shape_vec(dim, host_data).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?,
                ))
            }
            CudaData::Complex128(slice) => {
                // Simple synchronous D2H transfer via default stream
                // Note: slice.len() may be larger than shape_product due to pool bucket rounding
                let mut host_data = vec![CudaComplex64::default(); slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_data)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                // Truncate to actual shape size (pool may allocate larger buffers)
                host_data.truncate(shape_product);

                // Convert the Vec<CudaComplex64> to Vec<Complex64> via reinterpret
                let complex_data: Vec<Complex64> = unsafe {
                    let ptr = host_data.as_mut_ptr() as *mut Complex64;
                    let len = host_data.len();
                    let cap = host_data.capacity();
                    std::mem::forget(host_data);
                    Vec::from_raw_parts(ptr, len, cap)
                };

                Ok(CpuData::Complex128(
                    ArrayD::from_shape_vec(dim, complex_data).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?,
                ))
            }
            CudaData::Int64(slice) => {
                // Note: slice.len() may be larger than shape_product due to pool bucket rounding
                let mut host_data = vec![0i64; slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_data)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                // Truncate to actual shape size (pool may allocate larger buffers)
                host_data.truncate(shape_product);
                Ok(CpuData::Int64(
                    ArrayD::from_shape_vec(dim, host_data).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?,
                ))
            }
            CudaData::FixedPoint64(slice) => {
                // Q32.32 fixed-point: transfer from GPU as i64
                let mut host_data = vec![0i64; slice.len()];
                device
                    .default_stream()
                    .memcpy_dtoh(slice.as_slice(), &mut host_data)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                host_data.truncate(shape_product);
                Ok(CpuData::FixedPoint64(
                    ArrayD::from_shape_vec(dim, host_data).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?,
                ))
            }
        }
    }

    fn compute_shannon_entropy(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        if sum_sq < 1e-15 {
            return 0.0;
        }
        let entropy: f64 = values
            .iter()
            .map(|x| {
                let p = (x * x) / sum_sq;
                if p > 1e-15 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        let max_entropy = (values.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    fn compute_free_energy(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        if sum_sq < 1e-15 {
            return 0.0;
        }
        let n = values.len() as f64;
        let mut free_energy = 0.0;
        for (i, x) in values.iter().enumerate() {
            let rho = (x * x) / sum_sq;
            if rho > 1e-15 {
                let norm_i = (i as f64) / n;
                let entropy_term = rho * rho.ln();
                let potential_term = rho * norm_i * norm_i / Self::phi();
                free_energy += entropy_term + potential_term;
            }
        }
        free_energy / Self::e_star()
    }

    // ===== Internal Methods for linalg module =====

    /// Internal access to CPU data (for linalg module use)
    pub(crate) fn ensure_cpu_internal(&self) -> Result<CpuData, pyo3::PyErr> {
        self.ensure_cpu()
    }

    /// Internal access to device reference
    pub(crate) fn device_ref(&self) -> &DeviceType {
        &self.device
    }

    /// Internal clone
    pub(crate) fn clone_storage_internal(&self) -> TensorStorage {
        self.clone_storage()
    }

    /// Create TensorStorage from CPU data (internal constructor for linalg)
    pub(crate) fn new_from_cpu(
        data: CpuData,
        shape: Vec<usize>,
        device: DeviceType,
    ) -> TensorStorage {
        TensorStorage {
            data: TensorData::Cpu(data),
            shape,
            device,
        }
    }

    /// Create TensorStorage from CUDA data (internal constructor for cuda module)
    pub(crate) fn new_from_cuda(
        cuda_data: CudaData,
        device: std::sync::Arc<CudaDevice>,
        shape: Vec<usize>,
        device_idx: usize,
    ) -> TensorStorage {
        let dtype = match &cuda_data {
            CudaData::Float32(_) => "float32".to_string(),
            CudaData::Float64(_) => "float64".to_string(),
            CudaData::Int64(_) => "int64".to_string(),
            CudaData::Complex128(_) => "complex128".to_string(),
            CudaData::FixedPoint64(_) => "fixed_point64".to_string(),
        };

        TensorStorage {
            data: TensorData::Cuda {
                data: std::sync::Arc::new(cuda_data),
                device,
                shape: shape.clone(),
                dtype,
            },
            shape,
            device: DeviceType::Cuda(device_idx),
        }
    }
}

/// Check if CUDA is available
#[pyfunction]
pub fn cuda_is_available() -> bool {
    {
        DeviceManager::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA device count
#[pyfunction]
pub fn cuda_device_count() -> usize {
    {
        DeviceManager::device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Get SRT Memory Transfer Protocol statistics for a device
///
/// Returns a dict with:
/// - total_transfers: Number of SRT-optimized transfers
/// - total_bytes: Total bytes transferred
/// - avg_transfer_time_us: Average transfer latency in microseconds
/// - resonance_efficiency: Cache efficiency (0.0-1.0, target > 0.618)
/// - q_correction_applied: q-deficit correction factor (~1.0034)
#[pyfunction]
pub fn srt_transfer_stats(device_idx: usize) -> PyResult<std::collections::HashMap<String, f64>> {
    let protocol = get_srt_protocol(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let stats = protocol.get_stats();

    let mut result = std::collections::HashMap::new();
    result.insert("total_transfers".to_string(), stats.total_transfers as f64);
    result.insert("total_bytes".to_string(), stats.total_bytes as f64);
    result.insert(
        "avg_transfer_time_us".to_string(),
        stats.avg_transfer_time_us,
    );
    result.insert(
        "resonance_efficiency".to_string(),
        stats.resonance_efficiency,
    );
    result.insert(
        "q_correction_applied".to_string(),
        stats.q_correction_applied,
    );
    Ok(result)
}

/// Reserve pinned memory in the SRT pool (for manual management)
/// returns the size of the allocated block
#[pyfunction]
pub fn srt_reserve_memory(device_idx: usize, size: usize) -> PyResult<usize> {
    let protocol = get_srt_protocol(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Allocate and immediately drop (returning to pool if logic allows, or just testing allocation)
    // Since 'take' returns a Vec<u8> which is pinned, dropping it might unregister or free it.
    // The current implementation of SRTPinnedPool DOES handle dropping by unregistering.
    // However, if we want to "reserve" it in the pool, we should probably return it to the pool explicitly
    // or keep it alive. But since we can't easily pass the Vec<u8> to Python without converting,
    // this function primarily serves to exercise the 'take' path and verify allocation potential.
    // For a real reservation system, we'd need a Python object wrapping the allocation.
    // Given the constraints, we'll just exercise the method.
    let block = protocol
        .take(size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Keep the block alive briefly to ensure allocation succeeded, then drop it.
    // Returning the block to the pool or exposing it to Python would require a wrapper type.
    drop(block);

    // In a real scenario, we might want to keep this block alive or return a handle.
    // For now, this confirms we can take from the pool.

    Ok(size)
}

/// Wait for the next resonant window for transfer optimization
#[pyfunction]
pub fn srt_wait_for_resonance(device_idx: usize) -> PyResult<()> {
    let protocol = get_srt_protocol(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    protocol.wait_for_resonance();
    Ok(())
}

/// Get SRT pinned pool statistics
/// Returns (total_pinned_bytes, pooled_blocks_count, unique_sizes_count)
#[pyfunction]
pub fn srt_pool_stats(device_idx: usize) -> PyResult<(usize, usize, usize)> {
    let protocol = get_srt_protocol(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(protocol.stats())
}

/// Get resonance score for a specific memory block ID
#[pyfunction]
pub fn srt_memory_resonance(device_idx: usize, block_id: usize) -> PyResult<f64> {
    let protocol = get_srt_protocol(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(protocol.get_resonance(block_id))
}

/// Helper to force usage of PooledSlice::take to avoid dead code warnings
/// (Used in internal stress tests)
#[pyfunction]
pub fn _debug_stress_pool_take(device_idx: usize) -> PyResult<()> {
    // This function simply allocates a small slice and takes it,
    // ensuring the method is compiled and linked.
    let pool = get_pool(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Allocate 1 element
    let slice: PooledSlice<f32> = PooledSlice::alloc(pool, 1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Take ownership (removing from RAII pool management)
    let raw_cuda_slice = slice.take();

    // raw_cuda_slice will be dropped here, freeing the memory via normal CudaSlice Drop
    drop(raw_cuda_slice);
    // but bypassing the pool's recycle logic.
    Ok(())
}

// =============================================================================
// SRT CUDA Operation Wrappers
// =============================================================================

/// Scale a tensor by the golden ratio φ (GPU-accelerated when on CUDA)
#[pyfunction]
pub fn srt_scale_phi(tensor: &TensorStorage) -> PyResult<TensorStorage> {
    let n: usize = tensor.shape.iter().product();

    match &tensor.data {
        TensorData::Cpu(cpu_data) => {
            // CPU implementation
            match cpu_data {
                CpuData::Float64(arr) => {
                    let phi = srt_kernels::PHI;
                    let scaled: Vec<f64> = arr.iter().map(|x| x * phi).collect();
                    let new_arr =
                        ArrayD::from_shape_vec(IxDyn(&tensor.shape), scaled).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    Ok(TensorStorage {
                        data: TensorData::Cpu(CpuData::Float64(new_arr)),
                        shape: tensor.shape.clone(),
                        device: DeviceType::Cpu,
                    })
                }
                CpuData::Float32(arr) => {
                    let phi = srt_kernels::PHI as f32;
                    let scaled: Vec<f32> = arr.iter().map(|x| x * phi).collect();
                    let new_arr =
                        ArrayD::from_shape_vec(IxDyn(&tensor.shape), scaled).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    Ok(TensorStorage {
                        data: TensorData::Cpu(CpuData::Float32(new_arr)),
                        shape: tensor.shape.clone(),
                        device: DeviceType::Cpu,
                    })
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_scale_phi requires float32 or float64 tensor",
                )),
            }
        }
        TensorData::Cuda {
            data,
            device,
            shape,
            dtype: _,
        } => match data.as_ref() {
            CudaData::Float64(input_slice) => {
                let pool = get_pool(device.ordinal() as usize).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let mut output_slice = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                srt_kernels::cuda_scale_phi_f64(
                    device,
                    input_slice.as_slice(),
                    output_slice.as_slice_mut(),
                    n,
                )?;
                Ok(TensorStorage::new_from_cuda(
                    CudaData::Float64(Arc::new(output_slice)),
                    device.clone(),
                    shape.clone(),
                    device.ordinal(),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "CUDA srt_scale_phi currently only supports float64",
            )),
        },
    }
}

/// Compute golden gaussian weights for 8D vectors: w(λ) = exp(-|λ|²/φ)
/// Input tensor must have shape [..., 8] (last dimension is 8)
#[pyfunction]
pub fn srt_golden_gaussian_weights(vectors: &TensorStorage) -> PyResult<TensorStorage> {
    // Validate shape - last dimension must be 8
    if vectors.shape.is_empty() || *vectors.shape.last().unwrap() != 8 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input tensor must have last dimension of 8 for 8D golden gaussian weights",
        ));
    }

    let count: usize = vectors.shape.iter().take(vectors.shape.len() - 1).product();
    let count = if count == 0 { 1 } else { count }; // Handle scalar case
    let output_shape: Vec<usize> = vectors.shape[..vectors.shape.len() - 1].to_vec();
    let output_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    match &vectors.data {
        TensorData::Cpu(cpu_data) => match cpu_data {
            CpuData::Float64(arr) => {
                let mut weights = vec![0.0f64; count];
                srt_kernels::cpu_golden_gaussian_8d_f64(arr.as_slice().unwrap(), &mut weights);
                let new_arr =
                    ArrayD::from_shape_vec(IxDyn(&output_shape), weights).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(TensorStorage {
                    data: TensorData::Cpu(CpuData::Float64(new_arr)),
                    shape: output_shape,
                    device: DeviceType::Cpu,
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "srt_golden_gaussian_weights requires float64 tensor",
            )),
        },
        TensorData::Cuda {
            data,
            device,
            shape: _,
            dtype: _,
        } => match data.as_ref() {
            CudaData::Float64(input_slice) => {
                let pool = get_pool(device.ordinal() as usize).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let mut output_slice = PooledSlice::alloc(pool.clone(), count).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                srt_kernels::cuda_golden_gaussian_8d_f64(
                    device,
                    input_slice.as_slice(),
                    output_slice.as_slice_mut(),
                    count,
                )?;
                Ok(TensorStorage::new_from_cuda(
                    CudaData::Float64(Arc::new(output_slice)),
                    device.clone(),
                    output_shape.clone(),
                    device.ordinal(),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "CUDA srt_golden_gaussian_weights currently only supports float64",
            )),
        },
    }
}

/// Apply SRT correction factor: value × (1 + sign × q / N)
/// structure_idx: 0=E8_DIM(248), 1=E8_ROOTS(240), 2=E8_POS(120), etc.
/// sign: +1 or -1
#[pyfunction]
pub fn srt_apply_correction(
    tensor: &TensorStorage,
    structure_idx: i32,
    sign: i32,
) -> PyResult<TensorStorage> {
    let n: usize = tensor.shape.iter().product();
    let factor = srt_kernels::cpu_correction_factor(
        srt_kernels::get_structure_dimension(structure_idx),
        sign,
    );

    match &tensor.data {
        TensorData::Cpu(cpu_data) => match cpu_data {
            CpuData::Float64(arr) => {
                let corrected: Vec<f64> = arr.iter().map(|x| x * factor).collect();
                let new_arr =
                    ArrayD::from_shape_vec(IxDyn(&tensor.shape), corrected).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(TensorStorage {
                    data: TensorData::Cpu(CpuData::Float64(new_arr)),
                    shape: tensor.shape.clone(),
                    device: DeviceType::Cpu,
                })
            }
            CpuData::Float32(arr) => {
                let factor_f32 = factor as f32;
                let corrected: Vec<f32> = arr.iter().map(|x| x * factor_f32).collect();
                let new_arr =
                    ArrayD::from_shape_vec(IxDyn(&tensor.shape), corrected).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                Ok(TensorStorage {
                    data: TensorData::Cpu(CpuData::Float32(new_arr)),
                    shape: tensor.shape.clone(),
                    device: DeviceType::Cpu,
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "srt_apply_correction requires float32 or float64 tensor",
            )),
        },
        TensorData::Cuda {
            data,
            device,
            shape,
            dtype: _,
        } => match data.as_ref() {
            CudaData::Float64(input_slice) => {
                let pool = get_pool(device.ordinal() as usize).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                let mut output_slice = PooledSlice::alloc(pool.clone(), n).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                srt_kernels::cuda_apply_correction_f64(
                    device,
                    input_slice.as_slice(),
                    output_slice.as_slice_mut(),
                    structure_idx,
                    sign,
                    n,
                )?;
                Ok(TensorStorage::new_from_cuda(
                    CudaData::Float64(Arc::new(output_slice)),
                    device.clone(),
                    shape.clone(),
                    device.ordinal(),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "CUDA srt_apply_correction currently only supports float64",
            )),
        },
    }
}

/// Compute E8 batch projection for 8D root vectors
/// Returns tuple: (proj_parallel [N×4], proj_perp [N×4], q_values [N], in_cone [N])
/// proj_parallel: projection onto physical subspace
/// proj_perp: projection onto internal subspace
/// q_values: quadratic form Q = ||p_par||² - ||p_perp||²
/// in_cone: 1 if root is in golden cone, 0 otherwise
#[pyfunction]
pub fn srt_e8_batch_projection(
    roots: &TensorStorage,
) -> PyResult<(TensorStorage, TensorStorage, TensorStorage, TensorStorage)> {
    // Validate shape - must be [N, 8]
    if roots.shape.len() != 2 || roots.shape[1] != 8 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input tensor must have shape [N, 8] for E8 batch projection",
        ));
    }
    let count = roots.shape[0];

    // Golden ratio constant

    // Projection normalization: 1/sqrt(2*PHI + 2) = 1/sqrt(2*(phi+1))
    const PROJ_NORM: f64 = 0.3717480344601846;

    match &roots.data {
        TensorData::Cpu(cpu_data) => {
            match cpu_data {
                CpuData::Float64(arr) => {
                    let root_data = arr.as_slice().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "Cannot access array as contiguous slice",
                        )
                    })?;

                    // Allocate output arrays
                    let mut proj_parallel_data = vec![0.0f64; count * 4];
                    let mut proj_perp_data = vec![0.0f64; count * 4];
                    let mut q_values_data = vec![0.0f64; count];
                    let mut in_cone_data = vec![0i64; count];

                    for i in 0..count {
                        let root = &root_data[i * 8..(i + 1) * 8];

                        // Project to parallel (physical) space: P_φ · root
                        let p_par = [
                            PROJ_NORM * (PHI * root[0] + root[1]),
                            PROJ_NORM * (PHI * root[2] + root[3]),
                            PROJ_NORM * (PHI * root[4] + root[5]),
                            PROJ_NORM * (PHI * root[6] + root[7]),
                        ];

                        // Project to perpendicular (internal) space: P_⊥ · root
                        let p_perp = [
                            PROJ_NORM * (root[0] - PHI * root[1]),
                            PROJ_NORM * (root[2] - PHI * root[3]),
                            PROJ_NORM * (root[4] - PHI * root[5]),
                            PROJ_NORM * (root[6] - PHI * root[7]),
                        ];

                        // Quadratic form Q = ||p_par||² - ||p_perp||²
                        let par_sq: f64 = p_par.iter().map(|x| x * x).sum();
                        let perp_sq: f64 = p_perp.iter().map(|x| x * x).sum();
                        q_values_data[i] = par_sq - perp_sq;

                        // Golden cone test: all B_a >= 0 where B_a = root[2a] - PHI * root[2a+1]
                        let in_cone = (0..4).all(|a| root[2 * a] - PHI * root[2 * a + 1] >= -1e-10);
                        in_cone_data[i] = if in_cone { 1 } else { 0 };

                        // Store projections
                        proj_parallel_data[i * 4..(i + 1) * 4].copy_from_slice(&p_par);
                        proj_perp_data[i * 4..(i + 1) * 4].copy_from_slice(&p_perp);
                    }

                    // Create output tensors
                    let proj_parallel = TensorStorage {
                        data: TensorData::Cpu(CpuData::Float64(
                            ArrayD::from_shape_vec(IxDyn(&[count, 4]), proj_parallel_data)
                                .map_err(|e| {
                                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                                })?,
                        )),
                        shape: vec![count, 4],
                        device: DeviceType::Cpu,
                    };
                    let proj_perp = TensorStorage {
                        data: TensorData::Cpu(CpuData::Float64(
                            ArrayD::from_shape_vec(IxDyn(&[count, 4]), proj_perp_data).map_err(
                                |e| {
                                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                                },
                            )?,
                        )),
                        shape: vec![count, 4],
                        device: DeviceType::Cpu,
                    };
                    let q_values = TensorStorage {
                        data: TensorData::Cpu(CpuData::Float64(
                            ArrayD::from_shape_vec(IxDyn(&[count]), q_values_data).map_err(
                                |e| {
                                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                                },
                            )?,
                        )),
                        shape: vec![count],
                        device: DeviceType::Cpu,
                    };
                    let in_cone = TensorStorage {
                        data: TensorData::Cpu(CpuData::Int64(
                            ArrayD::from_shape_vec(IxDyn(&[count]), in_cone_data).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                            })?,
                        )),
                        shape: vec![count],
                        device: DeviceType::Cpu,
                    };

                    Ok((proj_parallel, proj_perp, q_values, in_cone))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_e8_batch_projection requires float64 tensor",
                )),
            }
        }
        TensorData::Cuda {
            data,
            device,
            shape: _,
            dtype: _,
        } => {
            match data.as_ref() {
                CudaData::Float64(input_slice) => {
                    let pool = get_pool(device.ordinal() as usize).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    // Allocate output buffers
                    let mut proj_parallel =
                        PooledSlice::alloc(pool.clone(), count * 4).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    let mut proj_perp =
                        PooledSlice::alloc(pool.clone(), count * 4).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    let mut q_values = PooledSlice::alloc(pool.clone(), count).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    let mut in_cone = PooledSlice::alloc(pool.clone(), count).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    srt_kernels::cuda_e8_batch_projection_f64(
                        device,
                        input_slice.as_slice(),
                        proj_parallel.as_slice_mut(),
                        proj_perp.as_slice_mut(),
                        q_values.as_slice_mut(),
                        in_cone.as_slice_mut(),
                        count,
                    )?;

                    // Convert in_cone i32 to i64 on CPU for consistency
                    let in_cone_i32: Vec<i32> = device
                        .default_stream()
                        .clone_dtoh(in_cone.as_slice())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    let in_cone_i64: Vec<i64> = in_cone_i32.iter().map(|&x| x as i64).collect();

                    Ok((
                        TensorStorage::new_from_cuda(
                            CudaData::Float64(Arc::new(proj_parallel)),
                            device.clone(),
                            vec![count, 4],
                            device.ordinal(),
                        ),
                        TensorStorage::new_from_cuda(
                            CudaData::Float64(Arc::new(proj_perp)),
                            device.clone(),
                            vec![count, 4],
                            device.ordinal(),
                        ),
                        TensorStorage::new_from_cuda(
                            CudaData::Float64(Arc::new(q_values)),
                            device.clone(),
                            vec![count],
                            device.ordinal(),
                        ),
                        TensorStorage {
                            data: TensorData::Cpu(CpuData::Int64(
                                ArrayD::from_shape_vec(IxDyn(&[count]), in_cone_i64).map_err(
                                    |e| {
                                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                            e.to_string(),
                                        )
                                    },
                                )?,
                            )),
                            shape: vec![count],
                            device: DeviceType::Cpu,
                        },
                    ))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_e8_batch_projection requires float64 tensor",
                )),
            }
        }
    }
}

/// Compute theta series sum: Θ(t) = Σ_λ w(λ) exp(-π Q(λ) / t)
/// q_values: quadratic form values Q(λ) for each lattice point
/// in_cone: binary mask (1 = include, 0 = exclude) for golden cone filtering
/// t: modular parameter for the theta function
/// Returns the scalar theta series value
#[pyfunction]
pub fn srt_theta_series(
    q_values: &TensorStorage,
    in_cone: &TensorStorage,
    t: f64,
) -> PyResult<f64> {
    let count: usize = q_values.shape.iter().product();
    let cone_count: usize = in_cone.shape.iter().product();

    if count != cone_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "q_values length {} must match in_cone length {}",
            count, cone_count
        )));
    }

    match (&q_values.data, &in_cone.data) {
        (TensorData::Cpu(q_cpu), TensorData::Cpu(cone_cpu)) => {
            // CPU implementation
            match (q_cpu, cone_cpu) {
                (CpuData::Float64(q_arr), CpuData::Int64(cone_arr)) => {
                    let pi = crate::constants::SRT_PI;
                    let sum: f64 = q_arr
                        .iter()
                        .zip(cone_arr.iter())
                        .filter(|(_, &c)| c != 0)
                        .map(|(&q, _)| (-pi * q / t).exp())
                        .sum();
                    Ok(sum)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_theta_series requires float64 q_values and int64 in_cone",
                )),
            }
        }
        (
            TensorData::Cuda {
                data: q_data,
                device,
                ..
            },
            TensorData::Cpu(cone_cpu),
        ) => {
            // CUDA q_values with CPU in_cone - need to copy in_cone to GPU
            match (q_data.as_ref(), cone_cpu) {
                (CudaData::Float64(q_slice), CpuData::Int64(cone_arr)) => {
                    let cone_i32: Vec<i32> = cone_arr.iter().map(|&x| x as i32).collect();
                    let cone_cuda: CudaSlice<i32> =
                        device.default_stream().clone_htod(&cone_i32).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    srt_kernels::cuda_theta_series_f64(device, q_slice, &cone_cuda, None, t, count)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_theta_series requires float64 q_values and int64 in_cone",
                )),
            }
        }
        (
            TensorData::Cuda {
                data: q_data,
                device,
                ..
            },
            TensorData::Cuda {
                data: cone_data, ..
            },
        ) => {
            // Both on CUDA - directly use GPU data
            match q_data.as_ref() {
                CudaData::Float64(q_slice) => {
                    // in_cone should be Int64 stored as CudaData - need to convert
                    // For now, copy to CPU and back since our CUDA kernel expects i32
                    let cone_cpu: Vec<i64> = match cone_data.as_ref() {
                        CudaData::Float64(s) => {
                            let mut f: Vec<f64> = vec![0.0; s.len()];
                            device
                                .default_stream()
                                .memcpy_dtoh(s.as_slice(), &mut f)
                                .map_err(|e| {
                                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                                })?;
                            f.iter().map(|&x| x as i64).collect()
                        }
                        _ => {
                            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Unexpected CUDA data type for in_cone",
                            ))
                        }
                    };
                    let cone_i32: Vec<i32> = cone_cpu.iter().map(|&x| x as i32).collect();
                    let cone_cuda: CudaSlice<i32> =
                        device.default_stream().clone_htod(&cone_i32).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    srt_kernels::cuda_theta_series_f64(device, q_slice, &cone_cuda, None, t, count)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_theta_series requires float64 q_values",
                )),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "srt_theta_series: q_values and in_cone must both be on CPU or q_values on CUDA",
        )),
    }
}

/// Compute syntony metric S(ψ) for complex128 state vectors
/// S(ψ) = Σ|ψ|² exp(-|n|²/φ) / Σ|ψ|²
/// Returns the syntony value in [0, 1]
#[pyfunction]
pub fn srt_compute_syntony(psi: &TensorStorage, mode_norm_sq: &TensorStorage) -> PyResult<f64> {
    let n: usize = psi.shape.iter().product();
    let norm_len: usize = mode_norm_sq.shape.iter().product();

    if n != norm_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "psi length {} must match mode_norm_sq length {}",
            n, norm_len
        )));
    }

    // 1/φ = φ - 1


    match (&psi.data, &mode_norm_sq.data) {
        (TensorData::Cpu(psi_cpu), TensorData::Cpu(norm_cpu)) => {
            // CPU implementation
            match (psi_cpu, norm_cpu) {
                (CpuData::Complex128(psi_arr), CpuData::Float64(norm_arr)) => {
                    let mut numerator = 0.0f64;
                    let mut denominator = 0.0f64;

                    for (psi_val, &norm_sq) in psi_arr.iter().zip(norm_arr.iter()) {
                        let amp_sq = psi_val.norm_sqr(); // |re|² + |im|²
                        let weight = (-norm_sq * PHI_INV).exp();
                        numerator += amp_sq * weight;
                        denominator += amp_sq;
                    }

                    if denominator < 1e-15 {
                        Ok(0.0)
                    } else {
                        Ok(numerator / denominator)
                    }
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_compute_syntony requires complex128 psi and float64 mode_norm_sq",
                )),
            }
        }
        (
            TensorData::Cuda {
                data: psi_data,
                device,
                ..
            },
            TensorData::Cuda {
                data: norm_data, ..
            },
        ) => match (psi_data.as_ref(), norm_data.as_ref()) {
            (CudaData::Complex128(psi_slice), CudaData::Float64(norm_slice)) => {
                srt_kernels::cuda_compute_syntony_c128(device, psi_slice, norm_slice, n)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "srt_compute_syntony requires complex128 psi and float64 mode_norm_sq",
            )),
        },
        _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "srt_compute_syntony requires both tensors on same device (CPU or CUDA)",
        )),
    }
}

/// Apply DHSR (Differentiate-Harmonize-Syntony-Recursion) cycle to complex128 state vector
/// Differentiation: amplifies high-frequency modes based on (1 - syntony)
/// Harmonization: attenuates non-golden modes based on syntony
/// Returns tuple: (new_psi, new_syntony) where new_psi is the evolved state
#[pyfunction]
pub fn srt_dhsr_cycle(
    psi: &TensorStorage,
    mode_norm_sq: &TensorStorage,
    syntony: f64,
) -> PyResult<(TensorStorage, f64)> {
    let n: usize = psi.shape.iter().product();
    let norm_len: usize = mode_norm_sq.shape.iter().product();

    if n != norm_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "psi length {} must match mode_norm_sq length {}",
            n, norm_len
        )));
    }

    // Golden ratio constants
 // 1/φ = φ - 1


    match (&psi.data, &mode_norm_sq.data) {
        (TensorData::Cpu(psi_cpu), TensorData::Cpu(norm_cpu)) => {
            // CPU implementation
            match (psi_cpu, norm_cpu) {
                (CpuData::Complex128(psi_arr), CpuData::Float64(norm_arr)) => {
                    let alpha = PHI_INV_SQ * (1.0 - syntony); // Differentiation strength
                    let beta = PHI_INV * syntony; // Harmonization strength

                    let mut new_psi_data: Vec<Complex64> = Vec::with_capacity(n);
                    let mut new_num = 0.0f64;
                    let mut new_den = 0.0f64;

                    for (psi_val, &norm_sq) in psi_arr.iter().zip(norm_arr.iter()) {
                        // Differentiation: scale = 1 + α√|n|²
                        let d_scale = 1.0 + alpha * norm_sq.sqrt();

                        // Harmonization: scale = 1 - β(1 - exp(-|n|²/φ))
                        let golden_weight = (-norm_sq * PHI_INV).exp();
                        let h_scale = 1.0 - beta * (1.0 - golden_weight);

                        // Combined DHSR scale
                        let total_scale = d_scale * h_scale;

                        // Apply to psi (complex)
                        let new_psi_val = psi_val * total_scale;
                        new_psi_data.push(new_psi_val);

                        // Accumulate new syntony
                        let new_amp_sq = new_psi_val.norm_sqr();
                        new_num += new_amp_sq * golden_weight;
                        new_den += new_amp_sq;
                    }

                    let new_syntony = if new_den < 1e-15 {
                        syntony
                    } else {
                        new_num / new_den
                    };

                    let new_psi = TensorStorage {
                        data: TensorData::Cpu(CpuData::Complex128(
                            ArrayD::from_shape_vec(IxDyn(&psi.shape), new_psi_data).map_err(
                                |e| {
                                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                                },
                            )?,
                        )),
                        shape: psi.shape.clone(),
                        device: DeviceType::Cpu,
                    };

                    Ok((new_psi, new_syntony))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_dhsr_cycle requires complex128 psi and float64 mode_norm_sq",
                )),
            }
        }
        (
            TensorData::Cuda {
                data: psi_data,
                device,
                shape,
                ..
            },
            TensorData::Cuda {
                data: norm_data, ..
            },
        ) => {
            match (psi_data.as_ref(), norm_data.as_ref()) {
                (CudaData::Complex128(psi_slice), CudaData::Float64(norm_slice)) => {
                    // Get memory pool
                    let pool = get_pool(device.ordinal()).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    // Copy psi to a new buffer for in-place modification
                    let mut new_psi_slice = PooledSlice::alloc(pool, n).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

                    device
                        .default_stream()
                        .memcpy_dtod(psi_slice.as_slice(), new_psi_slice.as_slice_mut())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;

                    let new_syntony = srt_kernels::cuda_dhsr_cycle_inplace_c128(
                        device,
                        new_psi_slice.as_slice_mut(),
                        norm_slice,
                        syntony,
                        n,
                    )?;

                    let new_psi = TensorStorage::new_from_cuda(
                        CudaData::Complex128(Arc::new(new_psi_slice)),
                        device.clone(),
                        shape.clone(),
                        device.ordinal(),
                    );

                    Ok((new_psi, new_syntony))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "srt_dhsr_cycle requires complex128 psi and float64 mode_norm_sq",
                )),
            }
        }

        _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "srt_dhsr_cycle requires both tensors on same device (CPU or CUDA)",
        )),
    }
}

// =============================================================================
// Fixed-Point (fp64) High-Level Wrappers
// =============================================================================

/// Compute syntony S(ψ) using fixed-point arithmetic (bit-exact)
/// Returns (numerator, denominator) as (i64, i64)
#[pyfunction]
pub fn srt_compute_syntony_fp64(
    psi_re: &TensorStorage,
    psi_im: &TensorStorage,
    mode_norm_sq: &TensorStorage,
) -> PyResult<(i64, i64)> {
    let n: usize = psi_re.shape.iter().product();
    if psi_im.shape.iter().product::<usize>() != n || mode_norm_sq.shape.iter().product::<usize>() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Shape mismatch"));
    }

    match (&psi_re.data, &psi_im.data, &mode_norm_sq.data) {
        (
            TensorData::Cuda { data: re_data, device: dev, .. },
            TensorData::Cuda { data: im_data, .. },
            TensorData::Cuda { data: norm_data, .. },
        ) => {
             match (re_data.as_ref(), im_data.as_ref(), norm_data.as_ref()) {
                (CudaData::Int64(re), CudaData::Int64(im), CudaData::Int64(norm)) => {
                    srt_kernels::cuda_compute_syntony_fp64(dev, re, im, norm, n)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 tensors")),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}

#[pyfunction]
pub fn srt_differentiation_fp64(
    in_re: &TensorStorage,
    in_im: &TensorStorage,
    mode_norm_sq: &TensorStorage,
    syntony: i64,
) -> PyResult<(TensorStorage, TensorStorage)> {
    let n: usize = in_re.shape.iter().product();
    
    match (&in_re.data, &in_im.data, &mode_norm_sq.data) {
        (
            TensorData::Cuda { data: re_in, device: dev, shape, .. },
            TensorData::Cuda { data: im_in, .. },
            TensorData::Cuda { data: norm_data, .. },
        ) => {
             match (re_in.as_ref(), im_in.as_ref(), norm_data.as_ref()) {
                (CudaData::Int64(re), CudaData::Int64(im), CudaData::Int64(norm)) => {
                    let pool = get_pool(dev.ordinal()).unwrap();
                    let mut out_re_slice = PooledSlice::alloc(pool.clone(), n).unwrap();
                    let mut out_im_slice = PooledSlice::alloc(pool, n).unwrap();
                    
                    srt_kernels::cuda_differentiation_fp64(dev, &mut out_re_slice, &mut out_im_slice, re, im, norm, syntony, n)?;
                    
                    let out_re_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_re_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    let out_im_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_im_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    Ok((out_re_ts, out_im_ts))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 tensors")),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}

#[pyfunction]
pub fn srt_harmonization_fp64(
    in_re: &TensorStorage,
    in_im: &TensorStorage,
    mode_norm_sq: &TensorStorage,
    syntony: i64,
) -> PyResult<(TensorStorage, TensorStorage)> {
    let n: usize = in_re.shape.iter().product();
    
    match (&in_re.data, &in_im.data, &mode_norm_sq.data) {
        (
            TensorData::Cuda { data: re_in, device: dev, shape, .. },
            TensorData::Cuda { data: im_in, .. },
            TensorData::Cuda { data: norm_data, .. },
        ) => {
             match (re_in.as_ref(), im_in.as_ref(), norm_data.as_ref()) {
                (CudaData::Int64(re), CudaData::Int64(im), CudaData::Int64(norm)) => {
                    let pool = get_pool(dev.ordinal()).unwrap();
                    let mut out_re_slice = PooledSlice::alloc(pool.clone(), n).unwrap();
                    let mut out_im_slice = PooledSlice::alloc(pool, n).unwrap();
                    
                    srt_kernels::cuda_harmonization_fp64(dev, &mut out_re_slice, &mut out_im_slice, re, im, norm, syntony, n)?;
                    
                    let out_re_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_re_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    let out_im_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_im_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    Ok((out_re_ts, out_im_ts))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 tensors")),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}

#[pyfunction]
pub fn srt_dhsr_cycle_fp64(
    in_re: &TensorStorage,
    in_im: &TensorStorage,
    mode_norm_sq: &TensorStorage,
    syntony: i64,
) -> PyResult<(TensorStorage, TensorStorage)> {
    let n: usize = in_re.shape.iter().product();
    
    match (&in_re.data, &in_im.data, &mode_norm_sq.data) {
        (
            TensorData::Cuda { data: re_in, device: dev, shape, .. },
            TensorData::Cuda { data: im_in, .. },
            TensorData::Cuda { data: norm_data, .. },
        ) => {
             match (re_in.as_ref(), im_in.as_ref(), norm_data.as_ref()) {
                (CudaData::Int64(re), CudaData::Int64(im), CudaData::Int64(norm)) => {
                    let pool = get_pool(dev.ordinal()).unwrap();
                    let mut out_re_slice = PooledSlice::alloc(pool.clone(), n).unwrap();
                    let mut out_im_slice = PooledSlice::alloc(pool, n).unwrap();
                    
                    srt_kernels::cuda_dhsr_cycle_fp64(dev, &mut out_re_slice, &mut out_im_slice, re, im, norm, syntony, n)?;
                    
                    let out_re_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_re_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    let out_im_ts = TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_im_slice)), dev.clone(), shape.clone(), dev.ordinal());
                    Ok((out_re_ts, out_im_ts))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 tensors")),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}

#[pyfunction]
pub fn srt_laplacian_1d_fp64(
    input: &TensorStorage,
) -> PyResult<TensorStorage> {
    let n: usize = input.shape.iter().product(); 
    
    match &input.data {
        TensorData::Cuda { data: in_data, device: dev, shape, .. } => {
             match in_data.as_ref() {
                CudaData::Int64(inp) => {
                     let pool = get_pool(dev.ordinal()).unwrap();
                     let mut out_slice = PooledSlice::alloc(pool, n).unwrap();
                     srt_kernels::cuda_laplacian_1d_fp64(dev, &mut out_slice, inp, n)?;
                     Ok(TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_slice)), dev.clone(), shape.clone(), dev.ordinal()))
                }
                 _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 inputs")),
             }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}

#[pyfunction]
pub fn srt_differentiation_full_fp64(
    input: &TensorStorage,
    fourier_proj: &TensorStorage,
    laplacian: &TensorStorage,
    alpha: i64,
    zeta: i64,
    syntony: i64,
) -> PyResult<TensorStorage> {
    let n: usize = input.shape.iter().product();
     match (&input.data, &fourier_proj.data, &laplacian.data) {
        (
            TensorData::Cuda { data: in_data, device: dev, shape, .. },
            TensorData::Cuda { data: f_data, .. },
            TensorData::Cuda { data: lap_data, .. },
        ) => {
             match (in_data.as_ref(), f_data.as_ref(), lap_data.as_ref()) {
                (CudaData::Int64(inp), CudaData::Int64(f), CudaData::Int64(lap)) => {
                    let pool = get_pool(dev.ordinal()).unwrap();
                    let mut out_slice = PooledSlice::alloc(pool, n).unwrap();
                    srt_kernels::cuda_differentiation_full_fp64(dev, &mut out_slice, inp, f, lap, alpha, zeta, syntony, n)?;
                    Ok(TensorStorage::new_from_cuda(CudaData::Int64(Arc::new(out_slice)), dev.clone(), shape.clone(), dev.ordinal()))
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Requires int64 tensors")),
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CPU fixed-point not exposed yet")),
    }
}
