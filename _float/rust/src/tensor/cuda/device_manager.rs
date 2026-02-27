//! Global device manager for CUDA infrastructure.
//!
//! Provides cached device handles to avoid repeated device initialization overhead.
//! Note: CudaStream is not thread-safe, so streams are created on-demand per operation.

use cudarc::driver::safe::CudaContext as CudaDevice;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::memory_pool::MemoryPool;
use super::srt_memory_protocol::SRTMemoryTransferProtocol;

/// CUDA operation errors
#[derive(Debug, Clone)]
pub enum CudaError {
    DeviceNotFound(usize),
    DeviceInitFailed(String),
    StreamCreateFailed(String),
    P2PNotSupported { from: usize, to: usize },
    TransferFailed(String),
    AllocationFailed(String),
    KernelLaunchFailed(String),
    SyncFailed(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound(idx) => write!(f, "CUDA device {} not found", idx),
            CudaError::DeviceInitFailed(msg) => write!(f, "Device init failed: {}", msg),
            CudaError::StreamCreateFailed(msg) => write!(f, "Stream creation failed: {}", msg),
            CudaError::P2PNotSupported { from, to } => {
                write!(f, "P2P not supported between device {} and {}", from, to)
            }
            CudaError::TransferFailed(msg) => write!(f, "Transfer failed: {}", msg),
            CudaError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            CudaError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            CudaError::SyncFailed(msg) => write!(f, "Synchronization failed: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<CudaError> for pyo3::PyErr {
    fn from(e: CudaError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

/// Stream types for different operation categories
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum StreamKind {
    /// Default stream for synchronous operations
    Default,
    /// Dedicated stream for H2D/D2H transfers
    Transfer,
    /// Primary compute stream for kernel execution
    Compute,
    /// Secondary compute stream for overlapping work
    Auxiliary,
}

/// Global device manager (thread-safe device cache)
///
/// Note: Due to cudarc's CudaStream not being Send+Sync, we only cache
/// device handles. Streams must be created on-demand per thread.
pub struct DeviceManager {
    /// Cached device handles
    devices: RwLock<HashMap<usize, Arc<CudaDevice>>>,
    /// Memory pools per device (also not thread-safe, created on-demand)
    pools: RwLock<HashMap<usize, Arc<MemoryPool>>>,
    /// SRT Memory Transfer Protocol per device
    srt_protocols: RwLock<HashMap<usize, Arc<SRTMemoryTransferProtocol>>>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        DeviceManager {
            devices: RwLock::new(HashMap::new()),
            pools: RwLock::new(HashMap::new()),
            srt_protocols: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a cached device handle
    pub fn get_device(&self, idx: usize) -> Result<Arc<CudaDevice>, CudaError> {
        // Fast path: check if already cached
        {
            let devices = self.devices.read().unwrap();
            if let Some(device) = devices.get(&idx) {
                return Ok(device.clone());
            }
        }

        // Slow path: create and cache
        // Note: CudaDevice::new already returns Arc<CudaDevice>
        let device =
            CudaDevice::new(idx).map_err(|e| CudaError::DeviceInitFailed(e.to_string()))?;

        {
            let mut devices = self.devices.write().unwrap();
            devices.insert(idx, device.clone());
        }

        Ok(device)
    }

    /// Create a new stream for the given device
    /// Note: In cudarc 0.18.2, we use device.default_stream() instead of creating new streams
    pub fn create_stream(
        &self,
        device_idx: usize,
    ) -> Result<cudarc::driver::CudaStream, CudaError> {
        let device = self.get_device(device_idx)?;
        // In cudarc 0.18.2, streams are accessed via device.default_stream()
        // Validate device ordinal and return informative error
        Err(CudaError::StreamCreateFailed(format!(
            "Stream creation not supported in cudarc 0.18.2 for device {}. Use device.default_stream() instead.",
            device.ordinal()
        )))
    }

    /// Get or create a memory pool for the given device
    pub fn get_pool(&self, device_idx: usize) -> Result<Arc<MemoryPool>, CudaError> {
        // Fast path: check if already cached
        {
            let pools = self.pools.read().unwrap();
            if let Some(pool) = pools.get(&device_idx) {
                return Ok(pool.clone());
            }
        }

        // Slow path: create and cache
        let device = self.get_device(device_idx)?;
        let pool = Arc::new(MemoryPool::new(device_idx, device));

        {
            let mut pools = self.pools.write().unwrap();
            pools.insert(device_idx, pool.clone());
        }

        Ok(pool)
    }

    /// Get or create SRT Memory Transfer Protocol for the given device
    pub fn get_srt_protocol(
        &self,
        device_idx: usize,
    ) -> Result<Arc<SRTMemoryTransferProtocol>, CudaError> {
        // Fast path: check if already cached
        {
            let protocols = self.srt_protocols.read().unwrap();
            if let Some(protocol) = protocols.get(&device_idx) {
                return Ok(protocol.clone());
            }
        }

        // Slow path: create and cache
        let device = self.get_device(device_idx)?;
        let protocol = Arc::new(SRTMemoryTransferProtocol::default(device));

        {
            let mut protocols = self.srt_protocols.write().unwrap();
            protocols.insert(device_idx, protocol.clone());
        }

        Ok(protocol)
    }

    /// Get the number of available CUDA devices
    pub fn device_count() -> usize {
        // In cudarc 0.18.2, use the driver API
        cudarc::driver::result::device::get_count().unwrap_or(0) as usize
    }

    /// Check if CUDA is available
    pub fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    /// Synchronize a device
    pub fn sync_device(&self, device_idx: usize) -> Result<(), CudaError> {
        let device = self.get_device(device_idx)?;
        device
            .synchronize()
            .map_err(|e| CudaError::SyncFailed(e.to_string()))
    }

    /// Clear cached resources (for testing or memory pressure)
    pub fn clear_caches(&self) {
        // Clear pools
        {
            let pools = self.pools.read().unwrap();
            for pool in pools.values() {
                pool.trim();
            }
        }

        // Clear SRT protocols (they maintain their own state)
        {
            let mut protocols = self.srt_protocols.write().unwrap();
            protocols.clear();
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local device manager for non-Send types
thread_local! {
    static LOCAL_MANAGER: DeviceManager = DeviceManager::new();
}

/// Get a thread-local device manager (for non-Send types like CudaStream)
pub fn get_local_manager() -> &'static std::thread::LocalKey<DeviceManager> {
    &LOCAL_MANAGER
}

/// Convenience function to get a device
pub fn get_device(idx: usize) -> Result<Arc<CudaDevice>, CudaError> {
    LOCAL_MANAGER.with(|mgr| mgr.get_device(idx))
}

/// Convenience function to create a stream
pub fn create_stream(device_idx: usize) -> Result<cudarc::driver::CudaStream, CudaError> {
    LOCAL_MANAGER.with(|mgr| mgr.create_stream(device_idx))
}

/// Convenience function to get a memory pool
pub fn get_pool(device_idx: usize) -> Result<Arc<MemoryPool>, CudaError> {
    LOCAL_MANAGER.with(|mgr| mgr.get_pool(device_idx))
}

/// Convenience function to get SRT Memory Transfer Protocol
pub fn get_srt_protocol(device_idx: usize) -> Result<Arc<SRTMemoryTransferProtocol>, CudaError> {
    LOCAL_MANAGER.with(|mgr| mgr.get_srt_protocol(device_idx))
}

/// Convenience function to sync a device
pub fn sync_device(device_idx: usize) -> Result<(), CudaError> {
    LOCAL_MANAGER.with(|mgr| mgr.sync_device(device_idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_creation() {
        let mgr = DeviceManager::new();
        assert!(mgr.devices.read().unwrap().is_empty());
    }
}
