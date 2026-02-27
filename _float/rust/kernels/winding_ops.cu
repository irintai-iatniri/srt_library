/**
 * CUDA Winding Operations Kernel
 *
 * Provides GPU-accelerated winding-aware syntony computation.
 *
 * Key operations:
 * - compute_golden_weights: w(n) = exp(-|n|²/φ)
 * - compute_winding_syntony: S(Ψ) = Σ|ψ_i|²w(n_i) / Σ|ψ_i|²
 * - batch_winding_syntony: Per-sample syntony computation
 * - mobius_filter: Apply Möbius μ(n) mask
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// Golden ratio
#define PHI 1.6180339887498948482f
#define PHI_INV 0.6180339887498948482f

/**
 * Compute golden weights: w(n) = exp(-mode_norm/φ)
 *
 * @param mode_norms Input mode norm squared values |n|²
 * @param weights Output golden weights
 * @param n Number of elements
 */
extern "C" __global__
void compute_golden_weights(
    const float* __restrict__ mode_norms,
    float* __restrict__ weights,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] = expf(-mode_norms[idx] / PHI);
    }
}

/**
 * Compute energy per feature: |ψ_i|²
 *
 * @param values Input activation values
 * @param energies Output squared energies
 * @param n Number of elements
 */
extern "C" __global__
void compute_energy(
    const float* __restrict__ values,
    float* __restrict__ energies,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = values[idx];
        energies[idx] = v * v;
    }
}

/**
 * Compute weighted energy: energy[i] * weight[i]
 *
 * @param energies Input squared energies
 * @param weights Input golden weights
 * @param weighted_energies Output weighted energies
 * @param n Number of elements
 */
extern "C" __global__
void compute_weighted_energy(
    const float* __restrict__ energies,
    const float* __restrict__ weights,
    float* __restrict__ weighted_energies,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weighted_energies[idx] = energies[idx] * weights[idx];
    }
}

/**
 * Parallel reduction sum
 *
 * @param input Input array
 * @param output Output partial sums (one per block)
 * @param n Number of elements
 */
extern "C" __global__
void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load and add during load
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * Compute winding syntony for a batch (one sample per block)
 *
 * S(Ψ) = Σ|ψ_i|²w(n_i) / Σ|ψ_i|²
 *
 * @param values Input activations (batch_size x dim, row-major)
 * @param mode_norms Mode norm squared |n|² for each feature (length dim)
 * @param syntonies Output syntony per sample (length batch_size)
 * @param batch_size Number of samples
 * @param dim Feature dimension
 */
extern "C" __global__
void batch_winding_syntony_kernel(
    const float* __restrict__ values,
    const float* __restrict__ mode_norms,
    float* __restrict__ syntonies,
    int batch_size,
    int dim
) {
    extern __shared__ float sdata[];
    float* weighted_sum = sdata;
    float* total_sum = sdata + blockDim.x;
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize
    float w_sum = 0.0f;
    float t_sum = 0.0f;
    
    // Each thread processes multiple features
    for (int d = tid; d < dim; d += blockDim.x) {
        int idx = batch_idx * dim + d;
        float v = values[idx];
        float energy = v * v;
        float weight = expf(-mode_norms[d] / PHI);
        
        w_sum += energy * weight;
        t_sum += energy;
    }
    
    weighted_sum[tid] = w_sum;
    total_sum[tid] = t_sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            weighted_sum[tid] += weighted_sum[tid + s];
            total_sum[tid] += total_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        float syntony = (total_sum[0] > 1e-10f) ? 
            (weighted_sum[0] / total_sum[0]) : 0.0f;
        // Clamp to [0, 1]
        syntonies[batch_idx] = fminf(1.0f, fmaxf(0.0f, syntony));
    }
}

/**
 * Apply Möbius filter (set values to 0 where μ(n) = 0)
 *
 * @param values Input/output values
 * @param mobius_mask Precomputed Möbius mask (0 for μ(n)=0, 1 otherwise)
 * @param n Number of elements
 */
extern "C" __global__
void mobius_filter_kernel(
    float* __restrict__ values,
    const int* __restrict__ mobius_mask,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mobius_mask[idx] == 0) {
            values[idx] = 0.0f;
        }
    }
}

/**
 * Compute mode norms: |n|² = index²
 * Standard mode norm based on feature index.
 *
 * @param mode_norms Output mode norms
 * @param n Number of features
 */
extern "C" __global__
void compute_standard_mode_norms(
    float* __restrict__ mode_norms,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        mode_norms[idx] = (float)(idx * idx);
    }
}

/**
 * Golden weight decay: apply exp(-|n|²/φ) to weights
 *
 * @param weights Input/output weight values
 * @param layer_idx Layer index for φ^(-layer_idx) scaling
 * @param n Number of weights
 */
extern "C" __global__
void golden_decay_weight_kernel(
    float* __restrict__ weights,
    int layer_idx,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // φ^(-layer_idx)
        float scale = powf(PHI, -(float)layer_idx);
        weights[idx] *= scale;
    }
}
