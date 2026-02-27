//! Memory pooling for reduced CUDA allocation overhead.
//!
//! Uses a power-of-2 bucket allocator to cache freed allocations
//! and reduce the cost of repeated alloc/free cycles.

use cudarc::driver::safe::CudaContext as CudaDevice;
use cudarc::driver::ValidAsZeroBits;
use cudarc::driver::{CudaSlice, DeviceRepr};
use num_complex::Complex64;
use std::collections::HashMap;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CudaComplex64(pub Complex64);

unsafe impl DeviceRepr for CudaComplex64 {}
unsafe impl ValidAsZeroBits for CudaComplex64 {}

/// Half-precision float wrapper for CUDA operations
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CudaF16(pub half::f16);

unsafe impl DeviceRepr for CudaF16 {}
unsafe impl ValidAsZeroBits for CudaF16 {}

impl From<half::f16> for CudaF16 {
    fn from(f: half::f16) -> Self {
        CudaF16(f)
    }
}
impl From<CudaF16> for half::f16 {
    fn from(f: CudaF16) -> Self {
        f.0
    }
}

/// Brain float 16 wrapper for CUDA operations
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CudaBF16(pub half::bf16);

unsafe impl DeviceRepr for CudaBF16 {}
unsafe impl ValidAsZeroBits for CudaBF16 {}

impl From<half::bf16> for CudaBF16 {
    fn from(f: half::bf16) -> Self {
        CudaBF16(f)
    }
}
impl From<CudaBF16> for half::bf16 {
    fn from(f: CudaBF16) -> Self {
        f.0
    }
}

impl From<Complex64> for CudaComplex64 {
    fn from(c: Complex64) -> Self {
        CudaComplex64(c)
    }
}
impl From<CudaComplex64> for Complex64 {
    fn from(c: CudaComplex64) -> Self {
        c.0
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::device_manager::CudaError;

/// Trait for types that can be pooled
pub trait Poolable: DeviceRepr + Sized {
    /// Allocate from pool
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<Self>, CudaError>;
    /// Return to pool
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<Self>);
}

/// Configuration for memory pool behavior
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Minimum block size (default: 256 bytes)
    pub min_block_size: usize,
    /// Maximum cached memory per device (default: 1GB)
    pub max_cached_bytes: usize,
    /// Round allocations to power of 2 (default: true)
    pub round_to_power_of_2: bool,
    /// Maximum single allocation size to cache (default: 256MB)
    pub max_cacheable_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_block_size: 256,
            max_cached_bytes: 1024 * 1024 * 1024, // 1GB
            round_to_power_of_2: true,
            max_cacheable_size: 256 * 1024 * 1024, // 256MB
        }
    }
}

/// Statistics about pool usage
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    /// Total bytes currently cached
    pub cached_bytes: usize,
    /// Number of allocations served from cache
    pub cache_hits: usize,
    /// Number of allocations requiring new memory
    pub cache_misses: usize,
    /// Total allocations performed
    pub total_allocations: usize,
    /// Total bytes allocated (including cache reuse)
    pub total_bytes_allocated: usize,
}

/// Per-device memory pool for reduced allocation overhead
pub struct MemoryPool {
    device_idx: usize,
    device: Arc<CudaDevice>,
    /// Free blocks organized by size bucket (power of 2)
    free_blocks_u8: RwLock<HashMap<usize, Vec<CudaSlice<u8>>>>,
    free_blocks_f32: RwLock<HashMap<usize, Vec<CudaSlice<f32>>>>,
    free_blocks_f64: RwLock<HashMap<usize, Vec<CudaSlice<f64>>>>,
    free_blocks_c128: RwLock<HashMap<usize, Vec<CudaSlice<CudaComplex64>>>>,
    free_blocks_i32: RwLock<HashMap<usize, Vec<CudaSlice<i32>>>>,
    free_blocks_i64: RwLock<HashMap<usize, Vec<CudaSlice<i64>>>>,
    free_blocks_f16: RwLock<HashMap<usize, Vec<CudaSlice<CudaF16>>>>,
    free_blocks_bf16: RwLock<HashMap<usize, Vec<CudaSlice<CudaBF16>>>>,
    /// Total cached bytes
    cached_bytes: AtomicUsize,
    /// Statistics
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    total_allocations: AtomicUsize,
    total_bytes: AtomicUsize,
    /// Configuration
    config: PoolConfig,
}

impl MemoryPool {
    /// Create a new memory pool for the given device
    pub fn new(device_idx: usize, device: Arc<CudaDevice>) -> Self {
        MemoryPool {
            device_idx,
            device,
            free_blocks_u8: RwLock::new(HashMap::new()),
            free_blocks_f32: RwLock::new(HashMap::new()),
            free_blocks_f64: RwLock::new(HashMap::new()),
            free_blocks_c128: RwLock::new(HashMap::new()),
            free_blocks_i32: RwLock::new(HashMap::new()),
            free_blocks_i64: RwLock::new(HashMap::new()),
            free_blocks_f16: RwLock::new(HashMap::new()),
            free_blocks_bf16: RwLock::new(HashMap::new()),
            cached_bytes: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            config: PoolConfig::default(),
        }
    }

    /// Create a pool with custom configuration
    pub fn with_config(device_idx: usize, device: Arc<CudaDevice>, config: PoolConfig) -> Self {
        MemoryPool {
            device_idx,
            device,
            free_blocks_u8: RwLock::new(HashMap::new()),
            free_blocks_f32: RwLock::new(HashMap::new()),
            free_blocks_f64: RwLock::new(HashMap::new()),
            free_blocks_c128: RwLock::new(HashMap::new()),
            free_blocks_i32: RwLock::new(HashMap::new()),
            free_blocks_i64: RwLock::new(HashMap::new()),
            free_blocks_f16: RwLock::new(HashMap::new()),
            free_blocks_bf16: RwLock::new(HashMap::new()),
            cached_bytes: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            config,
        }
    }

    /// Round size up to next power of 2
    fn round_size(&self, size: usize) -> usize {
        if !self.config.round_to_power_of_2 {
            return size.max(self.config.min_block_size);
        }

        let size = size.max(self.config.min_block_size);
        if size.is_power_of_two() {
            size
        } else {
            size.next_power_of_two()
        }
    }

    /// Allocate memory from pool or create new allocation (u8)
    pub fn alloc_bytes(&self, size: usize) -> Result<CudaSlice<u8>, CudaError> {
        let bucket_size = self.round_size(size);

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size, Ordering::Relaxed);

        // Try to get from cache
        {
            let mut blocks = self.free_blocks_u8.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes.fetch_sub(bucket_size, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        // Cache miss: allocate new
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<u8>(bucket_size)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (f32)
    pub fn alloc_f32(&self, count: usize) -> Result<CudaSlice<f32>, CudaError> {
        let size_bytes = count * std::mem::size_of::<f32>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<f32>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_f32.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<f32>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (f64)
    pub fn alloc_f64(&self, count: usize) -> Result<CudaSlice<f64>, CudaError> {
        let size_bytes = count * std::mem::size_of::<f64>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<f64>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_f64.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<f64>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (Complex64)
    pub fn alloc_c128(&self, count: usize) -> Result<CudaSlice<CudaComplex64>, CudaError> {
        let size_bytes = count * std::mem::size_of::<CudaComplex64>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<CudaComplex64>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_c128.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<CudaComplex64>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (i32)
    pub fn alloc_i32(&self, count: usize) -> Result<CudaSlice<i32>, CudaError> {
        let size_bytes = count * std::mem::size_of::<i32>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<i32>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_i32.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<i32>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (i64)
    pub fn alloc_i64(&self, count: usize) -> Result<CudaSlice<i64>, CudaError> {
        let size_bytes = count * std::mem::size_of::<i64>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<i64>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_i64.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<i64>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Return u8 to pool
    pub fn free_bytes(&self, slice: CudaSlice<u8>) {
        let size = slice.len();
        let bucket_size = size; // Assumes allocated with round_size or check config

        if bucket_size > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bucket_size > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bucket_size, Ordering::Relaxed);
        self.free_blocks_u8
            .write()
            .unwrap()
            .entry(bucket_size)
            .or_default()
            .push(slice);
    }

    /// Return f32 to pool
    pub fn free_f32(&self, slice: CudaSlice<f32>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<f32>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_f32
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Return f64 to pool
    pub fn free_f64(&self, slice: CudaSlice<f64>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<f64>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_f64
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Return Complex64 to pool
    pub fn free_c128(&self, slice: CudaSlice<CudaComplex64>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<CudaComplex64>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_c128
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Return i32 to pool
    pub fn free_i32(&self, slice: CudaSlice<i32>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<i32>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_i32
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Return i64 to pool
    pub fn free_i64(&self, slice: CudaSlice<i64>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<i64>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_i64
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Allocate typed memory from pool (CudaF16)
    pub fn alloc_f16(&self, count: usize) -> Result<CudaSlice<CudaF16>, CudaError> {
        let size_bytes = count * std::mem::size_of::<CudaF16>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<CudaF16>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_f16.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<CudaF16>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (CudaBF16)
    pub fn alloc_bf16(&self, count: usize) -> Result<CudaSlice<CudaBF16>, CudaError> {
        let size_bytes = count * std::mem::size_of::<CudaBF16>();
        let bucket_size_bytes = self.round_size(size_bytes);
        let bucket_size_elements = bucket_size_bytes / std::mem::size_of::<CudaBF16>();

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);

        {
            let mut blocks = self.free_blocks_bf16.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size_bytes) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes
                        .fetch_sub(bucket_size_bytes, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device
            .default_stream()
            .alloc_zeros::<CudaBF16>(bucket_size_elements)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Return f16 to pool
    pub fn free_f16(&self, slice: CudaSlice<CudaF16>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<CudaF16>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_f16
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Return bf16 to pool
    pub fn free_bf16(&self, slice: CudaSlice<CudaBF16>) {
        let elements = slice.len();
        let bytes = elements * std::mem::size_of::<CudaBF16>();

        if bytes > self.config.max_cacheable_size {
            return;
        }
        let current = self.cached_bytes.load(Ordering::Relaxed);
        if current + bytes > self.config.max_cached_bytes {
            return;
        }

        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.free_blocks_bf16
            .write()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(slice);
    }

    /// Clear all cached memory
    pub fn trim(&self) {
        self.free_blocks_u8.write().unwrap().clear();
        self.free_blocks_f32.write().unwrap().clear();
        self.free_blocks_f64.write().unwrap().clear();
        self.free_blocks_c128.write().unwrap().clear();
        self.free_blocks_i32.write().unwrap().clear();
        self.free_blocks_f16.write().unwrap().clear();
        self.free_blocks_bf16.write().unwrap().clear();
        self.free_blocks_i64.write().unwrap().clear();
        self.cached_bytes.store(0, Ordering::Relaxed);
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            cached_bytes: self.cached_bytes.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes.load(Ordering::Relaxed),
        }
    }

    /// Get the device index
    pub fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get the underlying device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

// Implement Poolable for supported types
impl Poolable for u8 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<u8>, CudaError> {
        pool.alloc_bytes(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<u8>) {
        pool.free_bytes(slice)
    }
}

impl Poolable for f32 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<f32>, CudaError> {
        pool.alloc_f32(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<f32>) {
        pool.free_f32(slice)
    }
}

impl Poolable for f64 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<f64>, CudaError> {
        pool.alloc_f64(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<f64>) {
        pool.free_f64(slice)
    }
}

impl Poolable for CudaComplex64 {
    fn alloc_from_pool(
        pool: &MemoryPool,
        size: usize,
    ) -> Result<CudaSlice<CudaComplex64>, CudaError> {
        pool.alloc_c128(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<CudaComplex64>) {
        pool.free_c128(slice)
    }
}

impl Poolable for i32 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<i32>, CudaError> {
        pool.alloc_i32(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<i32>) {
        pool.free_i32(slice)
    }
}

impl Poolable for i64 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<i64>, CudaError> {
        pool.alloc_i64(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<i64>) {
        pool.free_i64(slice)
    }
}



impl Poolable for CudaF16 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<CudaF16>, CudaError> {
        pool.alloc_f16(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<CudaF16>) {
        pool.free_f16(slice)
    }
}

impl Poolable for CudaBF16 {
    fn alloc_from_pool(pool: &MemoryPool, size: usize) -> Result<CudaSlice<CudaBF16>, CudaError> {
        pool.alloc_bf16(size)
    }
    fn free_to_pool(pool: &MemoryPool, slice: CudaSlice<CudaBF16>) {
        pool.free_bf16(slice)
    }
}

/// RAII wrapper that returns memory to pool on drop
pub struct PooledSlice<T: Poolable> {
    inner: Option<CudaSlice<T>>,
    pool: Arc<MemoryPool>,
}

impl<T: Poolable> PooledSlice<T> {
    /// Create a new pooled slice
    pub fn new(slice: CudaSlice<T>, pool: Arc<MemoryPool>) -> Self {
        PooledSlice {
            inner: Some(slice),
            pool,
        }
    }

    /// Allocate a new pooled slice using the pool (count in elements)
    pub fn alloc(pool: Arc<MemoryPool>, count: usize) -> Result<Self, CudaError> {
        let slice = T::alloc_from_pool(&pool, count)?;
        Ok(PooledSlice {
            inner: Some(slice),
            pool,
        })
    }

    /// Get the underlying slice
    pub fn as_slice(&self) -> &CudaSlice<T> {
        self.inner.as_ref().unwrap()
    }

    /// Get mutable access to the underlying slice
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        self.inner.as_mut().unwrap()
    }

    /// Take ownership of the inner slice (no return to pool)
    pub fn take(mut self) -> CudaSlice<T> {
        self.inner.take().unwrap()
    }

    /// Get the length in elements
    pub fn len(&self) -> usize {
        self.inner.as_ref().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the pool this slice belongs to
    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }
}

impl<T: Poolable> std::ops::Deref for PooledSlice<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<T: Poolable> std::ops::DerefMut for PooledSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

impl<T: Poolable> Drop for PooledSlice<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.inner.take() {
            T::free_to_pool(&self.pool, slice);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_size() {
        if !CudaDevice::new(0).is_ok() {
            return;
        }
        let device = CudaDevice::new(0).unwrap();
        let pool = MemoryPool::new(0, device);

        assert_eq!(pool.round_size(100), 256); // min size
        assert_eq!(pool.round_size(256), 256);
        assert_eq!(pool.round_size(257), 512);
        assert_eq!(pool.round_size(1000), 1024);
    }

    #[test]
    fn test_pooled_slice_take() {
        if !CudaDevice::new(0).is_ok() {
            return;
        }
        let device = CudaDevice::new(0).unwrap();
        let pool = Arc::new(MemoryPool::new(0, device));

        let slice: PooledSlice<f32> = PooledSlice::alloc(pool, 10).unwrap();
        let cuda_slice = slice.take();
        assert_eq!(cuda_slice.len(), 10);
    }
}
