// Syntonic CUDA Kernels - Syntony-Weighted Softmax
// Softmax with golden measure weighting w(n) = exp(-|n|²/φ)

#include "srt_constants.cuh"
#include <float.h>

// =============================================================================
// Syntonic Softmax: learned mode (exp(-mode_norms/φ) weighting)
// =============================================================================

extern "C" __global__ void cuda_syntonic_softmax_learned_f64(
    double *out,                  // Output probabilities [batch, num_classes]
    const double *logits,         // Input logits [batch, num_classes]
    const double *mode_norms,     // Mode norms |n|² [num_classes]
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Step 1: Find max for numerical stability
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_max = fmax(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Block-level max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_sum += exp(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        y[i] = exp(weighted_logit - max_val) / sum;
    }
}

extern "C" __global__ void cuda_syntonic_softmax_learned_f32(
    float *out,
    const float *logits,
    const float *mode_norms,
    float syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared_f[];
    float *s_max = (float*)shared_f;
    float *s_sum = (float*)shared_f + 256;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *x = logits + batch_idx * num_classes;
    float *y = out + batch_idx * num_classes;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_max = fmaxf(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];
    __syncthreads();

    // Step 2: Sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_sum += __expf(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        y[i] = __expf(weighted_logit - max_val) / sum;
    }
}

// =============================================================================
// Syntonic Softmax: provided mode (direct syntony weights)
// =============================================================================

extern "C" __global__ void cuda_syntonic_softmax_provided_f64(
    double *out,
    const double *logits,
    const double *syntony,       // Pre-computed syntony weights (same shape as logits)
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared_p[];
    double *s_max = shared_p;
    double *s_sum = shared_p + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    const double *w = syntony + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Find max(x + scale * log(w))
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_max = fmax(local_max, x[i] + log_weight);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_sum += exp(x[i] + log_weight - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        y[i] = exp(x[i] + log_weight - max_val) / sum;
    }
}

// =============================================================================
// Syntonic Softmax: Strided Kernels (Complex / Arbitrary Dimension)
// =============================================================================

extern "C" __global__ void cuda_syntonic_softmax_learned_strided_f64(
    double *out,
    const double *logits,
    const double *mode_norms,
    double syntony_scale,
    int outer_size,
    int dim_size,
    int inner_size
) {
    // Total number of parallel rows to process = outer_size * inner_size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;
    
    if (idx >= count) return;

    // Map linear index to (outer, inner) coordinates
    int inner = idx % inner_size;
    int outer = idx / inner_size;
    
    // Base offset for this particular softmax row
    // logits[outer, 0, inner]
    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max (numerical stability)
    double local_max = -DBL_MAX;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        // mode_norms corresponds to dimension index i
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        local_max = fmax(local_max, val + log_weight);
    }
    
    // 2. Sum exponentials
    double sum = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        sum += exp(val + log_weight - local_max);
    }
    
    // 3. Normalize and Write
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        out[offset + i * stride] = exp(val + log_weight - local_max) / sum;
    }
}

extern "C" __global__ void cuda_syntonic_softmax_provided_strided_f64(
    double *out,
    const double *logits,
    const double *syntony,
    double syntony_scale,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;

    if (idx >= count) return;

    int inner = idx % inner_size;
    int outer = idx / inner_size;

    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max
    double local_max = -DBL_MAX;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        double w = syntony[offset + i * stride];
        double log_weight = log(fmax(w, 1e-10)) * syntony_scale;
        local_max = fmax(local_max, val + log_weight);
    }

    // 2. Sum
    double sum = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        double w = syntony[offset + i * stride];
        double log_weight = log(fmax(w, 1e-10)) * syntony_scale;
        sum += exp(val + log_weight - local_max);
    }

    // 3. Normalize
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        double w = syntony[offset + i * stride];
        double log_weight = log(fmax(w, 1e-10)) * syntony_scale;
        out[offset + i * stride] = exp(val + log_weight - local_max) / sum;
    }
}

// =============================================================================
// F32 Variants: Provided and Strided Kernels
// =============================================================================

extern "C" __global__ void cuda_syntonic_softmax_provided_f32(
    float *out,
    const float *logits,
    const float *syntony,
    float syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared_pf[];
    float *s_max = (float*)shared_pf;
    float *s_sum = (float*)shared_pf + 256;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *x = logits + batch_idx * num_classes;
    const float *w = syntony + batch_idx * num_classes;
    float *y = out + batch_idx * num_classes;

    // Find max(x + scale * log(w))
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = __logf(fmaxf(w[i], 1e-10f)) * syntony_scale;
        local_max = fmaxf(local_max, x[i] + log_weight);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    float max_val = s_max[0];
    __syncthreads();

    // Sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = __logf(fmaxf(w[i], 1e-10f)) * syntony_scale;
        local_sum += __expf(x[i] + log_weight - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    float sum = s_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = __logf(fmaxf(w[i], 1e-10f)) * syntony_scale;
        y[i] = __expf(x[i] + log_weight - max_val) / sum;
    }
}

extern "C" __global__ void cuda_syntonic_softmax_learned_strided_f32(
    float *out,
    const float *logits,
    const float *mode_norms,
    float syntony_scale,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;

    if (idx >= count) return;

    int inner = idx % inner_size;
    int outer = idx / inner_size;

    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max
    float local_max = -FLT_MAX;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        local_max = fmaxf(local_max, val + log_weight);
    }

    // 2. Sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        sum += __expf(val + log_weight - local_max);
    }

    // 3. Normalize
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        out[offset + i * stride] = __expf(val + log_weight - local_max) / sum;
    }
}

extern "C" __global__ void cuda_syntonic_softmax_provided_strided_f32(
    float *out,
    const float *logits,
    const float *syntony,
    float syntony_scale,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;

    if (idx >= count) return;

    int inner = idx % inner_size;
    int outer = idx / inner_size;

    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max
    float local_max = -FLT_MAX;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float w = syntony[offset + i * stride];
        float log_weight = __logf(fmaxf(w, 1e-10f)) * syntony_scale;
        local_max = fmaxf(local_max, val + log_weight);
    }

    // 2. Sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float w = syntony[offset + i * stride];
        float log_weight = __logf(fmaxf(w, 1e-10f)) * syntony_scale;
        sum += __expf(val + log_weight - local_max);
    }

    // 3. Normalize
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        float w = syntony[offset + i * stride];
        float log_weight = __logf(fmaxf(w, 1e-10f)) * syntony_scale;
        out[offset + i * stride] = __expf(val + log_weight - local_max) / sum;
    }
}

// =============================================================================
// Identity Mode: Standard Softmax (No Golden Weighting) - GPU Accelerated
// =============================================================================

extern "C" __global__ void cuda_softmax_identity_f64(
    double *out,
    const double *logits,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + 256;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Step 1: Find max for numerical stability
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmax(local_max, x[i]);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += exp(x[i] - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        y[i] = exp(x[i] - max_val) / sum;
    }
}

extern "C" __global__ void cuda_softmax_identity_f32(
    float *out,
    const float *logits,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + 256;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *x = logits + batch_idx * num_classes;
    float *y = out + batch_idx * num_classes;

    // Step 1: Find max for numerical stability (use double precision)
    double local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmax(local_max, (double)x[i]);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) in double precision and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double val = (double)x[i];
        local_sum += exp(val - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize and store as float
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double val = (double)x[i];
        double exp_val = exp(val - max_val);
        y[i] = (float)(exp_val / sum);
    }
}

// =============================================================================
// Identity Mode: Strided Variants
// =============================================================================

extern "C" __global__ void cuda_softmax_identity_strided_f64(
    double *out,
    const double *logits,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;

    if (idx >= count) return;

    int inner = idx % inner_size;
    int outer = idx / inner_size;

    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max
    double local_max = -DBL_MAX;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        local_max = fmax(local_max, val);
    }

    // 2. Sum
    double sum = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        sum += exp(val - local_max);
    }

    // 3. Normalize
    for (int i = 0; i < dim_size; ++i) {
        double val = logits[offset + i * stride];
        out[offset + i * stride] = exp(val - local_max) / sum;
    }
}

extern "C" __global__ void cuda_softmax_identity_strided_f32(
    float *out,
    const float *logits,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outer_size * inner_size;

    if (idx >= count) return;

    int inner = idx % inner_size;
    int outer = idx / inner_size;

    int offset = outer * dim_size * inner_size + inner;
    int stride = inner_size;

    // 1. Find max
    float local_max = -FLT_MAX;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        local_max = fmaxf(local_max, val);
    }

    // 2. Sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        sum += __expf(val - local_max);
    }

    // 3. Normalize
    for (int i = 0; i < dim_size; ++i) {
        float val = logits[offset + i * stride];
        out[offset + i * stride] = __expf(val - local_max) / sum;
    }
}