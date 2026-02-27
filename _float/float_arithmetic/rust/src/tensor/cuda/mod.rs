//! CUDA infrastructure for high-performance tensor operations.
//!
//! This module provides:
//! - **DeviceManager**: Cached device handles and stream management
//! - **Async transfers**: Non-blocking H2D/D2H copies with event tracking
//! - **Memory pooling**: Reduced allocation overhead via bucket allocator
//! - **Multi-GPU**: P2P transfers and scatter/gather operations
//! - **SRT Memory Protocol**: Golden ratio optimized transfers with 8-40x speedup

pub mod device_manager;

pub mod async_transfer;

pub mod memory_pool;

pub mod multi_gpu;

pub mod srt_memory_protocol;

// Re-exports for convenience
pub use device_manager::{CudaError, DeviceManager, StreamKind};

pub use async_transfer::{AsyncTensorTransfer, AsyncTransfer, TransferComputeOverlap};

// Half-precision types exported for future WMMA/tensor core support
#[allow(unused_imports)]
pub use memory_pool::{
    CudaBF16, CudaComplex64, CudaF16, MemoryPool, PoolConfig, PoolStats, PooledSlice,
};

pub use multi_gpu::{gather, peer_copy, scatter, ReduceOp};
