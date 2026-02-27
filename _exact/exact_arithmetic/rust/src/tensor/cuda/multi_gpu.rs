//! Multi-GPU support with P2P transfers and collective operations.
//!
//! Provides scatter/gather operations and GPU-to-GPU transfers.

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::device_manager::{get_device, get_pool, CudaError, DeviceManager};
use super::memory_pool::PooledSlice;
use crate::tensor::storage::{CpuData, CudaData, DeviceType, TensorStorage};

/// Reduction operations for multi-GPU collectives
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
}

/// P2P transfer between GPUs (staged via CPU for compatibility)
pub fn peer_copy(
    src_device: usize,
    src_data: &CudaSlice<f64>,
    dst_device: usize,
) -> Result<CudaSlice<f64>, CudaError> {
    if src_device == dst_device {
        // Same device: copy within device
        let device = get_device(src_device)?;

        let mut dst = device
            .default_stream()
            .alloc_zeros::<f64>(src_data.len())
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))?;

        device
            .default_stream()
            .memcpy_dtod(src_data, &mut dst)
            .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

        return Ok(dst);
    }

    // Cross-device: stage through CPU
    staged_copy(src_device, src_data, dst_device)
}

/// Staged transfer through CPU
fn staged_copy(
    src_device: usize,
    src_data: &CudaSlice<f64>,
    dst_device: usize,
) -> Result<CudaSlice<f64>, CudaError> {
    let src_dev = get_device(src_device)?;
    let dst_dev = get_device(dst_device)?;

    // Allocate host staging buffer
    let mut host_buf = vec![0.0f64; src_data.len()];

    // D2H from source
    src_dev
        .default_stream()
        .memcpy_dtoh(src_data, &mut host_buf)
        .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

    // H2D to destination
    let dst = dst_dev
        .default_stream()
        .clone_htod(&host_buf)
        .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

    Ok(dst)
}

/// Scatter a tensor across multiple GPUs
pub fn scatter(tensor: &TensorStorage, devices: &[usize]) -> Result<Vec<TensorStorage>, CudaError> {
    if devices.is_empty() {
        return Err(CudaError::TransferFailed(
            "No devices specified for scatter".to_string(),
        ));
    }

    let cpu_data = tensor
        .ensure_cpu_internal()
        .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

    let mut results = Vec::with_capacity(devices.len());

    match cpu_data {
        CpuData::Float64(arr) => {
            let data = arr.as_slice().unwrap();
            let chunk_size = (data.len() + devices.len() - 1) / devices.len();

            for (i, &dev_idx) in devices.iter().enumerate() {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(data.len());

                if start >= data.len() {
                    break;
                }

                let chunk = &data[start..end];
                let device = get_device(dev_idx)?;

                let gpu_data = device
                    .default_stream()
                    .clone_htod(chunk)
                    .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

                let chunk_shape = vec![end - start];
                let tensor = create_cuda_tensor_f64(gpu_data, chunk_shape, dev_idx)?;
                results.push(tensor);
            }
        }
        _ => {
            return Err(CudaError::TransferFailed(
                "Scatter only supports f64".to_string(),
            ))
        }
    }

    Ok(results)
}

/// Gather tensors from multiple GPUs to a single device
pub fn gather(tensors: &[TensorStorage], target_device: usize) -> Result<TensorStorage, CudaError> {
    if tensors.is_empty() {
        return Err(CudaError::TransferFailed(
            "No tensors to gather".to_string(),
        ));
    }

    let target_dev = get_device(target_device)?;

    // Collect all data to CPU first
    let mut all_data = Vec::new();

    for tensor in tensors {
        let cpu_data = tensor
            .ensure_cpu_internal()
            .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

        match cpu_data {
            CpuData::Float64(arr) => {
                all_data.extend(arr.iter().cloned());
            }
            _ => {
                return Err(CudaError::TransferFailed(
                    "Gather only supports f64".to_string(),
                ))
            }
        }
    }

    // Transfer to target device
    let gpu_data = target_dev
        .default_stream()
        .clone_htod(&all_data)
        .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

    let shape = vec![all_data.len()];
    create_cuda_tensor_f64(gpu_data, shape, target_device)
}

/// All-reduce operation across GPUs
pub fn all_reduce(tensors: &mut [TensorStorage], op: ReduceOp) -> Result<(), CudaError> {
    if tensors.len() <= 1 {
        return Ok(());
    }

    // Gather all data to CPU
    let mut all_cpu: Vec<Vec<f64>> = Vec::new();

    for tensor in tensors.iter() {
        let cpu_data = tensor
            .ensure_cpu_internal()
            .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

        match cpu_data {
            CpuData::Float64(arr) => {
                all_cpu.push(arr.iter().cloned().collect());
            }
            _ => {
                return Err(CudaError::TransferFailed(
                    "All-reduce only supports f64".to_string(),
                ))
            }
        }
    }

    // Check all have same length
    let len = all_cpu[0].len();
    if !all_cpu.iter().all(|v| v.len() == len) {
        return Err(CudaError::TransferFailed(
            "All tensors must have same size for all-reduce".to_string(),
        ));
    }

    // Perform reduction
    let reduced: Vec<f64> = (0..len)
        .map(|i| {
            let values: Vec<f64> = all_cpu.iter().map(|v| v[i]).collect();
            match op {
                ReduceOp::Sum => values.iter().sum(),
                ReduceOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
                ReduceOp::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ReduceOp::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
                ReduceOp::Product => values.iter().product(),
            }
        })
        .collect();

    // Broadcast result back to all devices
    for tensor in tensors.iter_mut() {
        let device_idx = match tensor.device_ref() {
            DeviceType::Cuda(idx) => *idx,
            _ => continue,
        };

        let device = get_device(device_idx)?;
        let gpu_data = device
            .default_stream()
            .clone_htod(&reduced)
            .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

        *tensor = create_cuda_tensor_f64(gpu_data, vec![len], device_idx)?;
    }

    Ok(())
}

/// Create a TensorStorage from CUDA f64 data
fn create_cuda_tensor_f64(
    data: CudaSlice<f64>,
    shape: Vec<usize>,
    device_idx: usize,
) -> Result<TensorStorage, CudaError> {
    let device = get_device(device_idx)?;
    let pool = get_pool(device_idx)?;

    let pooled = PooledSlice::new(data, pool);

    Ok(TensorStorage::new_from_cuda(
        CudaData::Float64(Arc::new(pooled)),
        device,
        shape.clone(),
        device_idx,
    ))
}

/// Get information about multi-GPU topology
pub struct MultiGpuInfo {
    pub device_count: usize,
}

impl MultiGpuInfo {
    /// Query multi-GPU information
    pub fn query() -> Result<Self, CudaError> {
        let count = DeviceManager::device_count();
        Ok(MultiGpuInfo {
            device_count: count,
        })
    }

    /// Print topology info
    pub fn print_topology(&self) {
        println!("Multi-GPU: {} devices available", self.device_count);
    }
}
