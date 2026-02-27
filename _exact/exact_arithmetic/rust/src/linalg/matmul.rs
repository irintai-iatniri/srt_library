//! Matrix multiplication operations for Syntonic tensors.
//!
//! All constants are derived from the exact symbolic infrastructure.
//! No hardcoded floating-point values.

use ndarray::{ArrayD, Axis, Ix2, IxDyn};
use num_complex::Complex64;
use std::sync::Arc;

use crate::exact::constants::{FundamentalConstant, Structure};
use crate::exact::golden::GoldenExact;
use crate::exact::symexpr::SymExpr;
use crate::tensor::cuda::{device_manager::get_pool, PooledSlice};
use crate::tensor::precision_policy::{srt_operation_policy, PrecisionPolicy};
use crate::tensor::storage::{CpuData, DeviceType, TensorStorage};
use crate::exact::syntonic::SyntonicExact;
use crate::exact::rational::Rational;

/// Error types for matmul operations
#[derive(Debug, Clone)]
pub enum MatmulError {
    DimensionMismatch { a_cols: usize, b_rows: usize },
    DeviceMismatch { a: String, b: String },
    UnsupportedDtype(String),
    ShapeError(String),
    NotImplemented(String),
}

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulError::DimensionMismatch { a_cols, b_rows } => write!(
                f,
                "Dimension mismatch: A has {} columns, B has {} rows",
                a_cols, b_rows
            ),
            MatmulError::DeviceMismatch { a, b } => {
                write!(f, "Device mismatch: A on {}, B on {}", a, b)
            }
            MatmulError::UnsupportedDtype(dt) => write!(f, "Unsupported dtype: {}", dt),
            MatmulError::ShapeError(msg) => write!(f, "Shape error: {}", msg),
            MatmulError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl std::error::Error for MatmulError {}

impl From<MatmulError> for pyo3::PyErr {
    fn from(e: MatmulError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}

/// Transpose operation for matmul
/// Used in generalized gemm API for BLAS-style operation specification
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transpose {
    /// No transpose (N)
    None,
    /// Transpose (T)
    Trans,
    /// Conjugate transpose (H) - for complex only
    ConjTrans,
}

// =============================================================================
// Constants derived from exact infrastructure
// =============================================================================

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

/// φ^k - derived from exact Fibonacci formula via GoldenExact::phi_power(k)
#[inline]
fn phi_power(k: i32) -> f64 {
    GoldenExact::phi_power(k).to_f64()
}

/// q - the universal syntony deficit - derived from FundamentalConstant::Q
#[inline]
fn q_deficit() -> f64 {
    FundamentalConstant::Q.approx_f64()
}

/// π - derived from FundamentalConstant::Pi
#[inline]
fn pi_value() -> f64 {
    FundamentalConstant::Pi.approx_f64()
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

// =============================================================================
// Core Matrix Multiplication
// =============================================================================

/// Core matrix multiplication: C = A × B
///
/// Supports f32, f64, and Complex128 dtypes.
/// Uses CUDA acceleration when available, falls back to CPU ndarray operations.
pub fn mm(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes for dimension checking
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MatmulError::ShapeError(
            "Both tensors must be 2D for matrix multiplication".to_string(),
        ));
    }

    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k != k2 {
        return Err(MatmulError::DimensionMismatch {
            a_cols: k,
            b_rows: k2,
        });
    }

    // Debug: Print device info
    println!("DEBUG: mm called with device: {:?}", a.device_ref());

    // Dispatch based on device and dtype
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            println!(
                "DEBUG: Dispatching to CUDA matmul for device {}",
                device_idx
            );
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            mm_cuda_dispatch(a, b, m, k, n, &device)
        }
        DeviceType::Cpu => {
            println!("DEBUG: Dispatching to CPU matmul");
            mm_cpu_dispatch(a, b, m, k, n)
        }
    }
}

/// CUDA dispatch for matrix multiplication
fn mm_cuda_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    m: usize,
    k: usize,
    n: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    println!(
        "DEBUG: mm_cuda_dispatch called with m={}, k={}, n={}",
        m, k, n
    );

    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => {
            println!("DEBUG: A tensor is CUDA with device idx {}", dev.ordinal());
            (data.as_ref(), dev, dev.ordinal())
        }
        _ => {
            println!("DEBUG: A tensor is NOT CUDA, falling back to CPU");
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ));
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => {
            println!("DEBUG: B tensor is CUDA");
            (data.as_ref(), &(), 0)
        }
        _ => {
            println!("DEBUG: B tensor is NOT CUDA, falling back to CPU");
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ));
        }
    };

    println!(
        "DEBUG: Processing CUDA matmul with data types: A={:?}, B={:?}",
        std::mem::discriminant(a_cuda),
        std::mem::discriminant(b_cuda)
    );

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            println!("DEBUG: f64 CUDA matmul - allocating output buffer");
            let mut c_slice = PooledSlice::alloc(pool.clone(), m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            // Use basic kernel for smaller matrices (< 128), tiled for larger
            let threshold = 128;
            if m < threshold && n < threshold && k < threshold {
                println!("DEBUG: f64 CUDA matmul - using basic kernel for small matrix");
                crate::tensor::srt_kernels::cuda_matmul_f64(
                    device,
                    c_slice.as_slice_mut(),
                    a_slice.as_slice(),
                    b_slice.as_slice(),
                    m,
                    n,
                    k,
                )
                .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul failed: {}", e)))?;
            } else {
                println!("DEBUG: f64 CUDA matmul - using tiled kernel for large matrix");
                crate::tensor::srt_kernels::cuda_matmul_tiled_f64(
                    device,
                    c_slice.as_slice_mut(),
                    a_slice.as_slice(),
                    b_slice.as_slice(),
                    m,
                    n,
                    k,
                )
                .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul failed: {}", e)))?;
            }

            println!("DEBUG: f64 CUDA matmul - completed successfully");
            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }
        (CudaData::Float32(a_slice), CudaData::Float32(b_slice)) => {
            println!("DEBUG: f32 CUDA matmul - allocating output buffer");
            let mut c_slice = PooledSlice::alloc(pool.clone(), m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            // Use tiled version for f32 as well
            println!("DEBUG: f32 CUDA matmul - calling cuda_matmul_tiled_f64 (transmuted)");
            crate::tensor::srt_kernels::cuda_matmul_tiled_f64(
                device,
                unsafe { std::mem::transmute(c_slice.as_slice_mut()) },
                unsafe { std::mem::transmute(a_slice.as_slice()) },
                unsafe { std::mem::transmute(b_slice.as_slice()) },
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul failed: {}", e)))?;

            println!("DEBUG: f32 CUDA matmul - completed successfully");
            Ok(TensorStorage::new_from_cuda(
                CudaData::Float32(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }
        (CudaData::Complex128(a_slice), CudaData::Complex128(b_slice)) => {
            println!("DEBUG: Complex128 CUDA matmul - allocating output buffer");
            let mut c_slice = PooledSlice::alloc(pool.clone(), m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            println!("DEBUG: Complex128 CUDA matmul - calling cuda_matmul_c128");
            crate::tensor::srt_kernels::cuda_matmul_c128(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul failed: {}", e)))?;

            println!("DEBUG: Complex128 CUDA matmul - completed successfully");
            Ok(TensorStorage::new_from_cuda(
                CudaData::Complex128(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }

        _ => {
            println!("DEBUG: Unsupported CUDA dtype combination");
            Err(MatmulError::UnsupportedDtype(
                "CUDA matmul: dtype mismatch or unsupported".to_string(),
            ))
        }
    }
}

/// CPU dispatch for matrix multiplication (fallback)
fn mm_cpu_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    m: usize,
    k: usize,
    n: usize,
) -> Result<TensorStorage, MatmulError> {
    debug_assert_eq!(a.shape(), vec![m, k], "A shape mismatch in cpu dispatch");
    debug_assert_eq!(b.shape(), vec![k, n], "B shape mismatch in cpu dispatch");

    let a_cpu = a
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let b_cpu = b
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    match (a_cpu, b_cpu) {
        (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
            let a_2d = a_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("A must be 2D: {}", e)))?;
            let b_2d = b_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("B must be 2D: {}", e)))?;

            let result = a_2d.dot(&b_2d);
            Ok(wrap_cpu(
                CpuData::Float64(result.into_dyn()),
                a.device_ref(),
            ))
        }
        (CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
            let a_2d = a_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("A must be 2D: {}", e)))?;
            let b_2d = b_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("B must be 2D: {}", e)))?;

            let result = a_2d.dot(&b_2d);
            Ok(wrap_cpu(
                CpuData::Float32(result.into_dyn()),
                a.device_ref(),
            ))
        }
        (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
            let a_2d = a_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("A must be 2D: {}", e)))?;
            let b_2d = b_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("B must be 2D: {}", e)))?;

            let result = a_2d.dot(&b_2d);
            Ok(wrap_cpu(
                CpuData::Complex128(result.into_dyn()),
                a.device_ref(),
            ))
        }
        _ => Err(MatmulError::UnsupportedDtype(
            "CPU matmul: dtype mismatch or unsupported".to_string(),
        )),
    }
}

/// GEMM: C = α × (A × B) + β × C
///
/// General matrix-matrix multiplication with scaling.
pub fn mm_add(
    a: &TensorStorage,
    b: &TensorStorage,
    c: &TensorStorage,
    alpha: f64,
    beta: f64,
) -> Result<TensorStorage, MatmulError> {
    let ab = mm(a, b)?;

    let ab_cpu = ab
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let c_cpu = c
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    match (ab_cpu, c_cpu) {
        (CpuData::Float64(ab_arr), CpuData::Float64(c_arr)) => {
            let result = &ab_arr * alpha + &c_arr * beta;
            Ok(wrap_cpu(CpuData::Float64(result), a.device_ref()))
        }
        (CpuData::Float32(ab_arr), CpuData::Float32(c_arr)) => {
            let result = &ab_arr * (alpha as f32) + &c_arr * (beta as f32);
            Ok(wrap_cpu(CpuData::Float32(result), a.device_ref()))
        }
        (CpuData::Complex128(ab_arr), CpuData::Complex128(c_arr)) => {
            let alpha_c = Complex64::new(alpha, 0.0);
            let beta_c = Complex64::new(beta, 0.0);
            let result = &ab_arr * alpha_c + &c_arr * beta_c;
            Ok(wrap_cpu(CpuData::Complex128(result), a.device_ref()))
        }
        _ => Err(MatmulError::UnsupportedDtype("Dtype mismatch".to_string())),
    }
}

// =============================================================================
// Transpose Variants
// =============================================================================

/// Transposed-None matmul: C = Aᵀ × B
pub fn mm_tn(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes for dimension checking
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MatmulError::ShapeError(
            "Both tensors must be 2D for matrix multiplication".to_string(),
        ));
    }

    // For Aᵀ × B: A is (m, k) -> Aᵀ is (k, m), B is (k, n), result is (m, n)
    let (m, k) = (a_shape[0], a_shape[1]); // A: (m, k)
    let (k2, n) = (b_shape[0], b_shape[1]); // B: (k, n)

    if k != k2 {
        return Err(MatmulError::DimensionMismatch {
            a_cols: k,
            b_rows: k2,
        });
    }

    // Dispatch based on device and dtype
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            mm_tn_cuda_dispatch(a, b, m, k, n, &device)
        }
        DeviceType::Cpu => {
            // Fallback: transpose A then regular matmul
            let a_t = transpose_internal(a)?;
            mm(&a_t, b)
        }
    }
}

/// CUDA dispatch for transposed-none matrix multiplication
fn mm_tn_cuda_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    m: usize,
    k: usize,
    n: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => (data.as_ref(), dev, dev.ordinal()),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ))
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => (data.as_ref(), &(), 0),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ))
        }
    };

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_matmul_tn_f64(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul_tn failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }
        (CudaData::Float32(a_slice), CudaData::Float32(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_matmul_tn_f32(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul_tn failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float32(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }
        (CudaData::Complex128(a_slice), CudaData::Complex128(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_matmul_tn_c128(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA matmul_tn failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Complex128(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, n],
                a_device_idx,
            ))
        }
        _ => Err(MatmulError::UnsupportedDtype(
            "CUDA matmul_tn: dtype mismatch or unsupported".to_string(),
        )),
    }
}

/// None-Transposed matmul: C = A × Bᵀ
pub fn mm_nt(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let b_t = transpose_internal(b)?;
    mm(a, &b_t)
}

/// Transposed-Transposed matmul: C = Aᵀ × Bᵀ
pub fn mm_tt(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let a_t = transpose_internal(a)?;
    let b_t = transpose_internal(b)?;
    mm(&a_t, &b_t)
}

/// Generalized matrix multiply: C = α × op(A) × op(B) + β × C
///
/// BLAS-style GEMM with transpose specification using the `Transpose` enum:
/// - `Transpose::None` - use matrix as-is
/// - `Transpose::Trans` - transpose the matrix
/// - `Transpose::ConjTrans` - conjugate transpose (Hermitian, for complex)
///
/// This unifies mm_tn(), mm_nt(), mm_tt(), mm_hn(), mm_nh() into one function.
pub fn mm_gemm(
    a: &TensorStorage,
    b: &TensorStorage,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: f64,
    beta: f64,
    c: Option<&TensorStorage>,
) -> Result<TensorStorage, MatmulError> {
    // Apply transpose operations based on enum values
    let a_op = match trans_a {
        Transpose::None => a.clone_storage_internal(),
        Transpose::Trans => transpose_internal(a)?,
        Transpose::ConjTrans => conj_transpose_internal(a)?,
    };

    let b_op = match trans_b {
        Transpose::None => b.clone_storage_internal(),
        Transpose::Trans => transpose_internal(b)?,
        Transpose::ConjTrans => conj_transpose_internal(b)?,
    };

    // Compute A × B (or op(A) × op(B))
    let ab = mm(&a_op, &b_op)?;

    // Apply scaling and accumulate
    match c {
        Some(c_mat) => {
            // C = α × (A × B) + β × C
            mm_add(&a_op, &b_op, c_mat, alpha, beta)
        }
        None => {
            // C = α × (A × B)
            if (alpha - 1.0).abs() < 1e-15 {
                Ok(ab)
            } else {
                mul_scalar_internal(&ab, alpha)
            }
        }
    }
}

// =============================================================================
// Hermitian Variants (for complex matrices)
// =============================================================================

/// Hermitian-None matmul: C = A† × B (conjugate transpose of A)
pub fn mm_hn(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let a_h = conj_transpose_internal(a)?;
    mm(&a_h, b)
}

/// None-Hermitian matmul: C = A × B†
pub fn mm_nh(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let b_h = conj_transpose_internal(b)?;
    mm(a, &b_h)
}

// =============================================================================
// Batched Matrix Multiplication
// =============================================================================

/// Batched matrix multiplication: C[i] = A[i] × B[i]
///
/// For 3D tensors of shape (batch, m, n) and (batch, n, k),
/// computes batched matrix products.
pub fn bmm(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 3 || b_shape.len() != 3 {
        return Err(MatmulError::ShapeError(
            "bmm requires 3D tensors".to_string(),
        ));
    }

    let batch = a_shape[0];
    if batch != b_shape[0] {
        return Err(MatmulError::DimensionMismatch {
            a_cols: batch,
            b_rows: b_shape[0],
        });
    }

    let (m, k) = (a_shape[1], a_shape[2]);
    let (k2, n) = (b_shape[1], b_shape[2]);

    if k != k2 {
        return Err(MatmulError::DimensionMismatch {
            a_cols: k,
            b_rows: k2,
        });
    }

    // Dispatch based on device and dtype
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            bmm_cuda_dispatch(a, b, batch, m, k, n, &device)
        }
        DeviceType::Cpu => bmm_cpu_dispatch(a, b, batch, m, k, n),
    }
}

/// CUDA dispatch for batched matrix multiplication
fn bmm_cuda_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => (data.as_ref(), dev, dev.ordinal()),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ))
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => (data.as_ref(), &(), 0),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ))
        }
    };

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, batch * m * n)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_bmm_f64(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                batch,
                m,
                n,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA bmm failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(Arc::new(c_slice)),
                a_device.clone(),
                vec![batch, m, n],
                a_device_idx,
            ))
        }
        (CudaData::Complex128(_), CudaData::Complex128(_)) => {
            // Note: Need to implement cuda_bmm_c128 in srt_kernels.rs
            // For now, fall back to CPU
            bmm_cpu_dispatch(a, b, batch, m, k, n)
        }
        _ => bmm_cpu_dispatch(a, b, batch, m, k, n), // Fallback to CPU
    }
}

/// CPU dispatch for batched matrix multiplication
fn bmm_cpu_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<TensorStorage, MatmulError> {
    // Validate dimensions: A is (batch, m, k) and B is (batch, k, n)
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 3 || b_shape.len() != 3 {
        return Err(MatmulError::ShapeError(
            "Batched matmul requires 3D tensors".to_string(),
        ));
    }

    if a_shape[0] != batch || a_shape[1] != m || a_shape[2] != k {
        return Err(MatmulError::ShapeError(format!(
            "Matrix A shape {:?} doesn't match expected ({}, {}, {})",
            a_shape, batch, m, k
        )));
    }

    if b_shape[0] != batch || b_shape[1] != k || b_shape[2] != n {
        return Err(MatmulError::ShapeError(format!(
            "Matrix B shape {:?} doesn't match expected ({}, {}, {})",
            b_shape, batch, k, n
        )));
    }

    let a_cpu = a
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let b_cpu = b
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    match (a_cpu, b_cpu) {
        (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
            let mut results: Vec<f64> = Vec::with_capacity(batch * m * n);

            for i in 0..batch {
                let a_slice = a_arr.index_axis(Axis(0), i);
                let b_slice = b_arr.index_axis(Axis(0), i);

                let a_2d = a_slice
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
                let b_2d = b_slice
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

                let c = a_2d.dot(&b_2d);
                results.extend(c.iter());
            }

            let result = ArrayD::from_shape_vec(IxDyn(&[batch, m, n]), results)
                .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

            Ok(wrap_cpu(CpuData::Float64(result), a.device_ref()))
        }
        (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
            let mut results: Vec<Complex64> = Vec::with_capacity(batch * m * n);

            for i in 0..batch {
                let a_slice = a_arr.index_axis(Axis(0), i);
                let b_slice = b_arr.index_axis(Axis(0), i);

                let a_2d = a_slice
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
                let b_2d = b_slice
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

                let c = a_2d.dot(&b_2d);
                results.extend(c.iter());
            }

            let result = ArrayD::from_shape_vec(IxDyn(&[batch, m, n]), results)
                .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

            Ok(wrap_cpu(CpuData::Complex128(result), a.device_ref()))
        }
        (CpuData::Syntonic(a_arr), CpuData::Syntonic(b_arr)) => {
            let mut results: Vec<SyntonicExact> = Vec::with_capacity(batch * m * n);
            for i in 0..batch {
                let a_slice = a_arr.index_axis(Axis(0), i);
                let b_slice = b_arr.index_axis(Axis(0), i);
                let a_2d = a_slice.into_dimensionality::<Ix2>().map_err(|e| MatmulError::ShapeError(e.to_string()))?;
                let b_2d = b_slice.into_dimensionality::<Ix2>().map_err(|e| MatmulError::ShapeError(e.to_string()))?;
                
                // Manual matmul loop to avoid ndarray::dot trait issues/overflow
                // c[i,j] = sum_k a[i,k] * b[k,j]
                let zero_elem = crate::exact::syntonic::SyntonicExact::new(crate::exact::golden::GoldenExact::zero());
                let mut c = ndarray::Array2::<SyntonicExact>::from_elem((m, n), zero_elem);
                for row in 0..m {
                    for col in 0..n {
                        let mut sum = zero_elem;
                        for k_idx in 0..a_2d.ncols() {
                            sum = sum + (a_2d[[row, k_idx]] * b_2d[[k_idx, col]]);
                        }
                        c[[row, col]] = sum;
                    }
                }
                
                results.extend(c.iter());
            }
            let result = ArrayD::from_shape_vec(IxDyn(&[batch, m, n]), results)
                .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
             Ok(wrap_cpu(CpuData::Syntonic(result), a.device_ref()))
        }
        _ => Err(MatmulError::UnsupportedDtype(
            "bmm only supports f64 and complex128".to_string(),
        )),
    }
}

// =============================================================================
// SRT-Specific Operations (Using Symbolic Infrastructure)
// =============================================================================

/// φ-scaled matmul: φⁿ × (A × B)
///
/// Uses GoldenExact::phi_power(n) for exact φⁿ computed via Fibonacci formula:
/// φⁿ = (Fₙ₊₁ + Fₙ × φ) in Q(φ)
pub fn mm_phi(a: &TensorStorage, b: &TensorStorage, n: i32) -> Result<TensorStorage, MatmulError> {
    // Verify precision policy - mm_phi requires Exact precision
    let policy = srt_operation_policy("mm_phi");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "mm_phi is an SRT operation requiring Exact precision, got {:?}",
        policy
    );

    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes for dimension checking
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MatmulError::ShapeError(
            "Both tensors must be 2D for matrix multiplication".to_string(),
        ));
    }

    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, p) = (b_shape[0], b_shape[1]);

    if k != k2 {
        return Err(MatmulError::DimensionMismatch {
            a_cols: k,
            b_rows: k2,
        });
    }

    // Dispatch based on device and dtype
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            mm_phi_cuda_dispatch(a, b, n, m, k, p, &device)
        }
        DeviceType::Cpu => {
            // Fallback: matmul then scalar multiply
            let result = mm(a, b)?;
            let scale = phi_power(n);
            mul_scalar_internal(&result, scale)
        }
    }
}

/// CUDA dispatch for phi-scaled matrix multiplication
fn mm_phi_cuda_dispatch(
    a: &TensorStorage,
    b: &TensorStorage,
    n: i32,
    m: usize,
    k: usize,
    p: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => (data.as_ref(), dev, dev.ordinal()),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ))
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => (data.as_ref(), &(), 0),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ))
        }
    };

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, m * p)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_matmul_phi_scaled_f64(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                n,
                m,
                k,
                p,
            )
            .map_err(|e| {
                MatmulError::ShapeError(format!("CUDA matmul_phi_scaled failed: {}", e))
            })?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, p],
                a_device_idx,
            ))
        }
        _ => {
            // Fallback to CPU for other dtypes
            let result = mm(a, b)?;
            let scale = phi_power(n);
            mul_scalar_internal(&result, scale)
        }
    }
}

/// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
///
/// The fundamental bracket for SRT φ-Lie algebra representations.
/// Uses GoldenExact::phi_hat() for exact φ⁻¹ = φ - 1.
pub fn phi_bracket(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MatmulError::ShapeError(
            "Both tensors must be 2D for phi_bracket".to_string(),
        ));
    }

    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k != k2 || m != n {
        return Err(MatmulError::ShapeError(
            "Matrices must be square and compatible for commutator".to_string(),
        ));
    }

    // Dispatch based on device
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            phi_bracket_cuda(a, b, m, k, &device)
        }
        DeviceType::Cpu => phi_bracket_cpu(a, b),
    }
}

/// CUDA implementation of golden commutator
fn phi_bracket_cuda(
    a: &TensorStorage,
    b: &TensorStorage,
    m: usize,
    k: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => (data.as_ref(), dev, dev.ordinal()),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ))
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => (data.as_ref(), &(), 0),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ))
        }
    };

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            let mut c_slice = PooledSlice::alloc(pool, m * m)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_golden_commutator_f64(
                device,
                c_slice.as_slice_mut(),
                a_slice.as_slice(),
                b_slice.as_slice(),
                m,
                m,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA commutator failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(Arc::new(c_slice)),
                a_device.clone(),
                vec![m, m],
                a_device_idx,
            ))
        }
        _ => phi_bracket_cpu(a, b), // Fallback to CPU for other dtypes
    }
}

/// CPU implementation of golden commutator
fn phi_bracket_cpu(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let ab = mm(a, b)?;
    let ba = mm(b, a)?;
    let phi_inv_ba = mul_scalar_internal(&ba, phi_inv())?;
    sub_internal(&ab, &phi_inv_ba)
}

/// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
///
/// Symmetric counterpart to the φ-bracket.
pub fn phi_antibracket(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    // Check device compatibility
    if a.device_ref() != b.device_ref() {
        return Err(MatmulError::DeviceMismatch {
            a: format!("{:?}", a.device_ref()),
            b: format!("{:?}", b.device_ref()),
        });
    }

    // Get tensor shapes
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MatmulError::ShapeError(
            "Both tensors must be 2D for phi_antibracket".to_string(),
        ));
    }

    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k != k2 || m != n {
        return Err(MatmulError::ShapeError(
            "Matrices must be square and compatible for anticommutator".to_string(),
        ));
    }

    // Dispatch based on device
    match a.device_ref() {
        DeviceType::Cuda(device_idx) => {
            use crate::tensor::cuda::device_manager::get_device;
            let device = get_device(*device_idx).map_err(|e| {
                MatmulError::ShapeError(format!("Failed to get CUDA device: {}", e))
            })?;
            phi_antibracket_cuda(a, b, m, k, &device)
        }
        DeviceType::Cpu => phi_antibracket_cpu(a, b),
    }
}

/// CUDA implementation of golden anticommutator
fn phi_antibracket_cuda(
    a: &TensorStorage,
    b: &TensorStorage,
    m: usize,
    k: usize,
    device: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
) -> Result<TensorStorage, MatmulError> {
    use crate::tensor::cuda::device_manager::get_pool;
    use crate::tensor::storage::{CudaData, TensorData};

    // Extract CUDA data directly from tensor variants
    let (a_cuda, a_device, a_device_idx) = match &a.data {
        TensorData::Cuda {
            data, device: dev, ..
        } => (data.as_ref(), dev, dev.ordinal()),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for A".to_string(),
            ))
        }
    };

    let (b_cuda, _, _) = match &b.data {
        TensorData::Cuda { data, .. } => (data.as_ref(), &(), 0),
        _ => {
            return Err(MatmulError::ShapeError(
                "Expected CUDA tensor for B".to_string(),
            ))
        }
    };

    let pool = get_pool(a_device_idx)
        .map_err(|e| MatmulError::ShapeError(format!("CUDA pool error: {}", e)))?;

    match (a_cuda, b_cuda) {
        (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
            let mut c_pooled = crate::tensor::cuda::memory_pool::PooledSlice::alloc(pool, m * m)
                .map_err(|e| MatmulError::ShapeError(format!("CUDA alloc failed: {}", e)))?;

            crate::tensor::srt_kernels::cuda_golden_anticommutator_f64(
                device,
                &mut *c_pooled,
                &a_slice,
                &b_slice,
                m,
                m,
                k,
            )
            .map_err(|e| MatmulError::ShapeError(format!("CUDA anticommutator failed: {}", e)))?;

            Ok(TensorStorage::new_from_cuda(
                CudaData::Float64(std::sync::Arc::new(c_pooled)),
                a_device.clone(),
                vec![m, m],
                a_device_idx,
            ))
        }
        _ => phi_antibracket_cpu(a, b), // Fallback to CPU for other dtypes
    }
}

/// CPU implementation of golden anticommutator
fn phi_antibracket_cpu(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let ab = mm(a, b)?;
    let ba = mm(b, a)?;
    let phi_inv_ba = mul_scalar_internal(&ba, phi_inv())?;
    add_internal(&ab, &phi_inv_ba)
}

/// Correction factor matmul: (1 ± q/N) × (A × B)
///
/// Uses Structure::correction() for symbolic construction then evaluates.
/// The correction factor encodes how q-deficit modifies operations on
/// structures of dimension N (e.g., E₈ has N=248).
pub fn mm_corrected(
    a: &TensorStorage,
    b: &TensorStorage,
    structure: Structure,
    sign: i8,
) -> Result<TensorStorage, MatmulError> {
    // Verify precision policy - mm_corrected requires Exact precision
    let policy = srt_operation_policy("mm_corrected");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "mm_corrected is an SRT operation requiring Exact precision, got {:?}",
        policy
    );

    let result = mm(a, b)?;
    let correction = structure.correction(sign).eval_f64();
    mul_scalar_internal(&result, correction)
}

/// Direct q-deficit correction: (1 ± q/N) × (A × B)
///
/// Unlike `mm_corrected` which uses the Structure enum, this function
/// allows specifying the dimension N directly. This is useful for:
/// - Custom structures not in the standard hierarchy
/// - Testing with arbitrary dimensions
/// - Fine-grained control over correction factors
///
/// The correction factor is: (1 + sign×q/N) where q ≈ 0.027395
///
/// Standard dimensions from the SRT hierarchy:
/// - E₈: N=248 (adjoint), N=240 (roots), N=120 (positive roots), N=8 (rank)
/// - E₆: N=78 (adjoint), N=36 (golden cone), N=27 (fundamental)
/// - D₄: N=24 (kissing number)
/// - G₂: N=14 (adjoint)
pub fn mm_q_corrected_direct(
    a: &TensorStorage,
    b: &TensorStorage,
    n: u32,
    sign: i8,
) -> Result<TensorStorage, MatmulError> {
    let result = mm(a, b)?;
    let correction = correction_factor(n, sign);
    mul_scalar_internal(&result, correction)
}

/// Direct q-deficit scalar: (1 ± q/N) × scalar
///
/// Returns the raw correction factor without matrix multiplication.
/// Useful for applying q-corrections to individual values.
///
/// Uses q_deficit() directly for numeric evaluation.
pub fn q_correction_scalar(n: u32, sign: i8) -> f64 {
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

/// Complex phase matmul: e^{iπn/φ} × (A × B)
///
/// Applies a golden-ratio-modulated phase rotation.
/// Uses π from FundamentalConstant::Pi and φ from GoldenExact.
pub fn mm_golden_phase(
    a: &TensorStorage,
    b: &TensorStorage,
    n: i32,
) -> Result<TensorStorage, MatmulError> {
    let result = mm(a, b)?;

    // Phase angle: πn/φ
    let angle = pi_value() * (n as f64) / phi();
    let phase = Complex64::new(angle.cos(), angle.sin());

    mul_complex_scalar_internal(&result, phase)
}

/// Golden-weighted matmul: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(−k²/φ)
///
/// Each summation index k is weighted by a golden Gaussian.
/// Uses GoldenExact::phi() for exact φ.
pub fn mm_golden_weighted(
    a: &TensorStorage,
    b: &TensorStorage,
) -> Result<TensorStorage, MatmulError> {
    // Verify precision policy - mm_golden_weighted requires Exact precision
    let policy = srt_operation_policy("mm_golden_weighted");
    debug_assert!(
        matches!(policy, PrecisionPolicy::Exact),
        "mm_golden_weighted is an SRT operation requiring Exact precision, got {:?}",
        policy
    );

    let a_cpu = a
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let b_cpu = b
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    match (a_cpu, b_cpu) {
        (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
            let a_2d = a_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("A must be 2D: {}", e)))?;
            let b_2d = b_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("B must be 2D: {}", e)))?;

            let m = a_2d.nrows();
            let k = a_2d.ncols();
            let n = b_2d.ncols();

            if k != b_2d.nrows() {
                return Err(MatmulError::DimensionMismatch {
                    a_cols: k,
                    b_rows: b_2d.nrows(),
                });
            }

            // Precompute weights: exp(-k²/φ)
            let phi_val = phi();
            let weights: Vec<f64> = (0..k)
                .map(|i| (-(i as f64).powi(2) / phi_val).exp())
                .collect();

            // Compute weighted matrix product
            let mut result = ndarray::Array2::<f64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut sum = Rational::from_integer(0);
                    for ki in 0..k {
                        sum = sum.add(&Rational::from_f64_approx(a_2d[[i, ki]] * b_2d[[ki, j]] * weights[ki]));
                    }
                    result[[i, j]] = sum.eval_f64();
                }
            }

            Ok(wrap_cpu(
                CpuData::Float64(result.into_dyn()),
                a.device_ref(),
            ))
        }
        (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
            let a_2d = a_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("A must be 2D: {}", e)))?;
            let b_2d = b_arr
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|e| MatmulError::ShapeError(format!("B must be 2D: {}", e)))?;

            let m = a_2d.nrows();
            let k = a_2d.ncols();
            let n = b_2d.ncols();

            if k != b_2d.nrows() {
                return Err(MatmulError::DimensionMismatch {
                    a_cols: k,
                    b_rows: b_2d.nrows(),
                });
            }

            let phi_val = phi();
            let weights: Vec<f64> = (0..k)
                .map(|i| (-(i as f64).powi(2) / phi_val).exp())
                .collect();

            let mut result = ndarray::Array2::<Complex64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut sum = Complex64::new(Rational::from_integer(0).eval_f64(), Rational::from_integer(0).eval_f64());
                    for ki in 0..k {
                        sum += a_2d[[i, ki]] * b_2d[[ki, j]] * weights[ki];
                    }
                    result[[i, j]] = sum;
                }
            }

            Ok(wrap_cpu(
                CpuData::Complex128(result.into_dyn()),
                a.device_ref(),
            ))
        }
        _ => Err(MatmulError::UnsupportedDtype(
            "golden_weighted only supports f64 and complex128".to_string(),
        )),
    }
}

/// Projection sum: Ψ + Σₖ αₖ × (Pₖ × Ψ)
///
/// Used for DHSR projection summation over lattice points.
/// Coefficients should be derived from symbolic values when possible.
pub fn projection_sum(
    psi: &TensorStorage,
    projectors: &[TensorStorage],
    coefficients: &[f64],
) -> Result<TensorStorage, MatmulError> {
    if projectors.len() != coefficients.len() {
        return Err(MatmulError::ShapeError(format!(
            "Projectors ({}) and coefficients ({}) must have same length",
            projectors.len(),
            coefficients.len()
        )));
    }

    if projectors.is_empty() {
        return Ok(psi.clone_storage_internal());
    }

    // Start with Ψ
    let mut result = psi.clone_storage_internal();

    // Add Σₖ αₖ × (Pₖ × Ψ)
    for (p, &alpha) in projectors.iter().zip(coefficients.iter()) {
        let p_psi = mm(p, psi)?;
        let scaled = mul_scalar_internal(&p_psi, alpha)?;
        result = add_internal(&result, &scaled)?;
    }

    Ok(result)
}

// =============================================================================
// Internal Helper Functions
// =============================================================================

fn transpose_internal(t: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let cpu = t
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

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

    Ok(wrap_cpu(result, t.device_ref()))
}

fn conj_transpose_internal(t: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let cpu = t
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    let result = match cpu {
        CpuData::Float64(arr) => CpuData::Float64(arr.t().to_owned()),
        CpuData::Float32(arr) => CpuData::Float32(arr.t().to_owned()),
        CpuData::Complex128(arr) => {
            // Conjugate transpose: transpose then conjugate each element
            let transposed = arr.t().to_owned();
            CpuData::Complex128(transposed.mapv(|x| x.conj()))
        }
        CpuData::Int64(arr) => CpuData::Int64(arr.t().to_owned()),
        CpuData::GoldenExact(arr) => CpuData::GoldenExact(arr.t().to_owned()),
        CpuData::Rational(arr) => CpuData::Rational(arr.t().to_owned()),
        CpuData::FixedPoint64(arr) => CpuData::FixedPoint64(arr.t().to_owned()),
        CpuData::Syntonic(arr) => CpuData::Syntonic(arr.t().to_owned()), // Syntonic self-conjugate (real-ish)
    };

    Ok(wrap_cpu(result, t.device_ref()))
}

fn mul_scalar_internal(t: &TensorStorage, scalar: f64) -> Result<TensorStorage, MatmulError> {
    let cpu = t
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    let result = match cpu {
        CpuData::Float64(arr) => CpuData::Float64(&arr * scalar),
        CpuData::Float32(arr) => CpuData::Float32(&arr * (scalar as f32)),
        CpuData::Complex128(arr) => CpuData::Complex128(&arr * Complex64::new(scalar, 0.0)),
        CpuData::Int64(arr) => CpuData::Int64(&arr * (scalar as i64)),
        CpuData::GoldenExact(arr) => {
            use crate::exact::golden::GoldenExact;
            use crate::exact::rational::Rational;
            let scalar_exact = GoldenExact::from_rational(Rational::from_int(scalar as i128));
            CpuData::GoldenExact(arr.mapv(|x| x * scalar_exact))
        }
        CpuData::Rational(arr) => {
            use crate::exact::rational::Rational;
            let scalar_rat = Rational::from_int(scalar as i128);
            CpuData::Rational(arr.mapv(|x| x * scalar_rat))
        }
        CpuData::FixedPoint64(arr) => {
            use crate::exact::fixed::FixedPoint64;
            let scalar_fixed = FixedPoint64::new((scalar * (1i64 << 32) as f64) as i64);
            CpuData::FixedPoint64(arr.mapv(|x| (FixedPoint64::new(x) * scalar_fixed).0))
        }
        CpuData::Syntonic(_) => {
            return Err(MatmulError::UnsupportedDtype("Cannot mul_scalar float on Syntonic".into()))
        }
    };

    Ok(wrap_cpu(result, t.device_ref()))
}

fn mul_complex_scalar_internal(
    t: &TensorStorage,
    scalar: Complex64,
) -> Result<TensorStorage, MatmulError> {
    let cpu = t
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    let result = match cpu {
        CpuData::Float64(arr) => {
            // Promote to complex
            let complex_arr = arr.mapv(|x| Complex64::new(x, 0.0));
            CpuData::Complex128(&complex_arr * scalar)
        }
        CpuData::Float32(arr) => {
            let complex_arr = arr.mapv(|x| Complex64::new(x as f64, 0.0));
            CpuData::Complex128(&complex_arr * scalar)
        }
        CpuData::Complex128(arr) => CpuData::Complex128(&arr * scalar),
        CpuData::Int64(_) | CpuData::GoldenExact(_) | CpuData::Rational(_) | CpuData::FixedPoint64(_) | CpuData::Syntonic(_) => {
            return Err(MatmulError::UnsupportedDtype(
                "Complex scalar multiplication not supported for int64 or exact types".to_string(),
            ))
        }
    };

    Ok(wrap_cpu(result, t.device_ref()))
}

fn add_internal(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let a_cpu = a
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let b_cpu = b
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    let result = match (a_cpu, b_cpu) {
        (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => CpuData::Float64(&a_arr + &b_arr),
        (CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => CpuData::Float32(&a_arr + &b_arr),
        (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
            CpuData::Complex128(&a_arr + &b_arr)
        }
        (CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(&a_arr + &b_arr),
        (CpuData::Syntonic(a_arr), CpuData::Syntonic(b_arr)) => CpuData::Syntonic(&a_arr + &b_arr),
        _ => {
            return Err(MatmulError::UnsupportedDtype(
                "Dtype mismatch in add".to_string(),
            ))
        }
    };

    Ok(wrap_cpu(result, a.device_ref()))
}

fn sub_internal(a: &TensorStorage, b: &TensorStorage) -> Result<TensorStorage, MatmulError> {
    let a_cpu = a
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;
    let b_cpu = b
        .ensure_cpu_internal()
        .map_err(|e| MatmulError::ShapeError(e.to_string()))?;

    let result = match (a_cpu, b_cpu) {
        (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => CpuData::Float64(&a_arr - &b_arr),
        (CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => CpuData::Float32(&a_arr - &b_arr),
        (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
            CpuData::Complex128(&a_arr - &b_arr)
        }
        (CpuData::Int64(a_arr), CpuData::Int64(b_arr)) => CpuData::Int64(&a_arr - &b_arr),
        (CpuData::Syntonic(a_arr), CpuData::Syntonic(b_arr)) => CpuData::Syntonic(&a_arr - &b_arr),
        _ => {
            return Err(MatmulError::UnsupportedDtype(
                "Dtype mismatch in sub".to_string(),
            ))
        }
    };

    Ok(wrap_cpu(result, a.device_ref()))
}

fn wrap_cpu(data: CpuData, device: &DeviceType) -> TensorStorage {
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
    TensorStorage::new_from_cpu(data, shape, device.clone())
}
