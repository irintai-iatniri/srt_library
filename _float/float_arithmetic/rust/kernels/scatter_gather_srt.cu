// =============================================================================
// SRT Scatter/Gather Kernels: Theory-Aligned Index Operations
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "srt_constants.cuh"
using namespace std;

// =============================================================================
// Standard Scatter/Gather Operations (Non-SRT)
// =============================================================================

// Standard gather: out[i] = src[idx[i]]
extern "C" __global__ void gather_f64(
    double* __restrict__ out,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = src[idx[i]];
    }
}

extern "C" __global__ void gather_f32(
    float* __restrict__ out,
    const float* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = src[idx[i]];
    }
}

// Standard scatter: dst[idx[i]] = src[i]
extern "C" __global__ void scatter_f64(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[idx[i]] = src[i];
    }
}

extern "C" __global__ void scatter_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[idx[i]] = src[i];
    }
}

// Scatter-add with atomics: dst[idx[i]] += src[i]
extern "C" __global__ void scatter_add_f64(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&dst[idx[i]], src[i]);
    }
}

extern "C" __global__ void scatter_add_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&dst[idx[i]], src[i]);
    }
}

// =============================================================================
// SRT-Specific Operations
// =============================================================================

// φ-weighted gather: out[i] = φ^{-i} * src[idx[i]]
extern "C" __global__ void gather_phi_weighted_f64(
    double* __restrict__ out,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double phi_weight = pow(PHI_INV_F64, (double)i);
        out[i] = phi_weight * src[idx[i]];
    }
}

// Golden-weighted scatter: dst[idx[i]] += e^{-i²/φ} * src[i]
extern "C" __global__ void scatter_golden_f64(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double golden_weight = exp(-((double)i * (double)i) / PHI_F64);
        dst[idx[i]] = golden_weight * src[i];
    }
}

// Mersenne-stable scatter: Only scatter to Mersenne-prime indices
// c_mersenne_mask is defined in srt_constants.cu and declared extern in srt_constants.cuh

extern "C" __global__ void scatter_mersenne_stable_f64(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t target_idx = idx[i];
        if (target_idx < MAX_MERSENNE && c_mersenne_mask[target_idx]) {
            dst[target_idx] = src[i];
        }
    }
}

// Lucas shadow gather: out[i] = (1-φ)^i * src[idx[i]]
extern "C" __global__ void gather_lucas_shadow_f64(
    double* __restrict__ out,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Shadow phase: (1-φ)^i = (-φ^{-1})^i
        double shadow_weight = pow(-PHI_INV_F64, (double)i);
        out[i] = shadow_weight * src[idx[i]];
    }
}

// Pisano-hooked gather: Wrap indices to Pisano period π(p)
__constant__ int c_pisano_periods[128];  // Precomputed π(p) for primes

extern "C" __global__ void gather_pisano_hooked_f64(
    double* __restrict__ out,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n,
    int p  // Prime index for Pisano period
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int period = c_pisano_periods[p];
        int64_t hooked_idx = idx[i] % period;
        out[i] = src[hooked_idx];
    }
}

// =============================================================================
// E₈ Lattice Operations (240 roots)
// =============================================================================

#define E8_ROOTS 240
#define E8_DIM 8

__constant__ float c_e8_roots[E8_ROOTS][E8_DIM];  // E₈ root coordinates

// Gather from E₈ root positions
extern "C" __global__ void gather_e8_roots_f64(
    double* __restrict__ out,       // [240] output
    const double* __restrict__ src, // 8D source tensor
    int stride_per_dim              // Stride per dimension
) {
    int root = blockIdx.x * blockDim.x + threadIdx.x;
    if (root < E8_ROOTS) {
        double sum = 0.0;
        for (int d = 0; d < E8_DIM; d++) {
            int coord = (int)c_e8_roots[root][d];
            sum += src[coord * stride_per_dim + d];
        }
        out[root] = sum;
    }
}

// =============================================================================
// Golden Cone Operations (E₆ positive roots: 36 roots)
// =============================================================================

#define E6_POSITIVE_ROOTS 36

extern "C" __global__ void scatter_golden_cone_f64(
    double* __restrict__ dst,  // [36] output
    const double* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E6_POSITIVE_ROOTS && i < n) {
        dst[i] = src[i];  // φ-aligned positive roots only
    }
}

// =============================================================================
// Fibonacci Transcendence Gates
// =============================================================================

#define FIB_GATES_COUNT 11
__constant__ int c_fib_gates[FIB_GATES_COUNT] = {3,4,5,7,11,13,17,23,29,43,47};

extern "C" __global__ void gather_transcendence_gate_f64(
    double* __restrict__ out,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double boost = 1.0;
        int64_t target_idx = idx[i];

        // Check if index matches a transcendence gate
        for (int g = 0; g < FIB_GATES_COUNT; g++) {
            if (target_idx == c_fib_gates[g]) {
                boost = pow(PHI_F64, (double)c_fib_gates[g]);
                if (c_fib_gates[g] == 4) boost *= 0.9;  // Material anomaly adjustment
                break;
            }
        }
        out[i] = boost * src[target_idx];
    }
}

// =============================================================================
// D₄ Consciousness Threshold Operations
// =============================================================================

extern "C" __global__ void scatter_consciousness_threshold_f64(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int64_t* __restrict__ idx,
    const double* __restrict__ syntony,  // Syntony values
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (syntony[i] > D4_KISSING_F64) {  // 24.0 consciousness threshold
            dst[idx[i]] = src[i];
        }
    }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// Standard operations
void launch_gather_f64(cudaStream_t stream, double* out, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gather_f64<<<grid, block, 0, stream>>>(out, src, idx, n);
}

void launch_scatter_f64(cudaStream_t stream, double* dst, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_f64<<<grid, block, 0, stream>>>(dst, src, idx, n);
}

void launch_scatter_add_f64(cudaStream_t stream, double* dst, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_add_f64<<<grid, block, 0, stream>>>(dst, src, idx, n);
}

// SRT-specific operations
void launch_gather_phi_weighted_f64(cudaStream_t stream, double* out, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gather_phi_weighted_f64<<<grid, block, 0, stream>>>(out, src, idx, n);
}

void launch_scatter_golden_f64(cudaStream_t stream, double* dst, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_golden_f64<<<grid, block, 0, stream>>>(dst, src, idx, n);
}

void launch_scatter_mersenne_stable_f64(cudaStream_t stream, double* dst, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_mersenne_stable_f64<<<grid, block, 0, stream>>>(dst, src, idx, n);
}

void launch_gather_lucas_shadow_f64(cudaStream_t stream, double* out, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gather_lucas_shadow_f64<<<grid, block, 0, stream>>>(out, src, idx, n);
}

void launch_gather_pisano_hooked_f64(cudaStream_t stream, double* out, const double* src, const int64_t* idx, int n, int p) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gather_pisano_hooked_f64<<<grid, block, 0, stream>>>(out, src, idx, n, p);
}

void launch_gather_e8_roots_f64(cudaStream_t stream, double* out, const double* src, int stride_per_dim) {
    dim3 block(256);
    dim3 grid((E8_ROOTS + 255) / 256);
    gather_e8_roots_f64<<<grid, block, 0, stream>>>(out, src, stride_per_dim);
}

void launch_scatter_golden_cone_f64(cudaStream_t stream, double* dst, const double* src, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_golden_cone_f64<<<grid, block, 0, stream>>>(dst, src, n);
}

void launch_gather_transcendence_gate_f64(cudaStream_t stream, double* out, const double* src, const int64_t* idx, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gather_transcendence_gate_f64<<<grid, block, 0, stream>>>(out, src, idx, n);
}

void launch_scatter_consciousness_threshold_f64(cudaStream_t stream, double* dst, const double* src, const int64_t* idx, const double* syntony, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    scatter_consciousness_threshold_f64<<<grid, block, 0, stream>>>(dst, src, idx, syntony, n);
}

}