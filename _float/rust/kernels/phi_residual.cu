// Syntonic CUDA Kernels - Phi-Scaled Residual Connections
// Implements golden-ratio residual connections for theory-aligned networks
//
// Theory: In SRT, the golden ratio φ provides natural dampening of
// recursive amplification, preventing exploding activations in deep networks.

#include "srt_constants.cuh"

// =============================================================================
// Phi-Residual Mode: output = identity + residual/φ
// =============================================================================

extern "C" __global__ void phi_residual_mode_phi_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // output = identity + residual * (1/φ)
        // where 1/φ = φ - 1 ≈ 0.618033988749895
        out[i] = identity[i] + residual[i] * PHI_INV_F64;
    }
}

extern "C" __global__ void phi_residual_mode_phi_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i] * PHI_INV_F32;
    }
}

// =============================================================================
// Phi-Symmetric Mode: output = (identity + residual)/φ
// =============================================================================

extern "C" __global__ void phi_residual_mode_symmetric_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // output = (identity + residual) / φ
        // Scales both paths equally by golden ratio inverse
        out[i] = (identity[i] + residual[i]) * PHI_INV_F64;
    }
}

extern "C" __global__ void phi_residual_mode_symmetric_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (identity[i] + residual[i]) * PHI_INV_F32;
    }
}

// =============================================================================
// Standard Mode: output = identity + residual (for ablation)
// =============================================================================

extern "C" __global__ void phi_residual_mode_standard_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i];
    }
}

extern "C" __global__ void phi_residual_mode_standard_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i];
    }
}

// =============================================================================
// Fused: Phi-Residual + ReLU (common pattern: skip + ReLU)
// =============================================================================

extern "C" __global__ void phi_residual_relu_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = identity[i] + residual[i] * PHI_INV_F64;
        out[i] = (val > 0.0) ? val : 0.0;
    }
}

extern "C" __global__ void phi_residual_relu_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = identity[i] + residual[i] * PHI_INV_F32;
        out[i] = (val > 0.0f) ? val : 0.0f;
    }
}

// =============================================================================
// Fused: Phi-Residual + GELU (smoother activation)
// =============================================================================
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

extern "C" __global__ void phi_residual_gelu_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = identity[i] + residual[i] * PHI_INV_F64;

        // GELU approximation
        const double sqrt_2_over_pi = 0.7978845608028654;  // sqrt(2/π)
        const double coeff = 0.044715;
        double x_cubed = x * x * x;
        double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        double gelu = 0.5 * x * (1.0 + tanh(inner));

        out[i] = gelu;
    }
}

extern "C" __global__ void phi_residual_gelu_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = identity[i] + residual[i] * PHI_INV_F32;

        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));

        out[i] = gelu;
    }
}

// =============================================================================
// Fused: Phi-Residual + LayerNorm (common in transformers)
// =============================================================================
// Combines residual connection with layer normalization in one pass

extern "C" __global__ void phi_residual_layernorm_f64(
    double *out,
    const double *identity,
    const double *residual,
    const double *gamma,          // Scale [feature_dim], NULL for 1.0
    const double *beta,           // Bias [feature_dim], NULL for 0.0
    double eps,
    int batch_size,
    int feature_dim
) {
    extern __shared__ double shared[];
    double *s_mean = shared;
    double *s_var = shared + 1;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *identity_row = identity + batch_idx * feature_dim;
    const double *residual_row = residual + batch_idx * feature_dim;
    double *out_row = out + batch_idx * feature_dim;

    // Step 1: Compute residual connection
    // temp = identity + residual/φ

    // Step 2: Compute mean
    double local_sum = 0.0;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        double temp = identity_row[i] + residual_row[i] * PHI_INV_F64;
        local_sum += temp;
    }

    local_sum = warp_reduce_sum_f64(local_sum);

    if (tid % 32 == 0) {
        atomicAdd(s_mean, local_sum);
    }
    __syncthreads();

    double mean = *s_mean / feature_dim;
    __syncthreads();

    // Step 3: Compute variance
    double local_var = 0.0;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        double temp = identity_row[i] + residual_row[i] * PHI_INV_F64;
        double diff = temp - mean;
        local_var += diff * diff;
    }

    local_var = warp_reduce_sum_f64(local_var);

    if (tid % 32 == 0) {
        atomicAdd(s_var, local_var);
    }
    __syncthreads();

    double var = *s_var / feature_dim;
    double rstd = rsqrt(var + eps);

    // Step 4: Normalize and apply affine
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        double temp = identity_row[i] + residual_row[i] * PHI_INV_F64;
        double normalized = (temp - mean) * rstd;

        if (gamma != NULL) normalized *= gamma[i];
        if (beta != NULL) normalized += beta[i];

        out_row[i] = normalized;
    }
}

extern "C" __global__ void phi_residual_layernorm_f32(
    float *out,
    const float *identity,
    const float *residual,
    const float *gamma,
    const float *beta,
    float eps,
    int batch_size,
    int feature_dim
) {
    extern __shared__ float shared_f[];
    float *s_mean = shared_f;
    float *s_var = shared_f + 1;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *identity_row = identity + batch_idx * feature_dim;
    const float *residual_row = residual + batch_idx * feature_dim;
    float *out_row = out + batch_idx * feature_dim;

    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float temp = identity_row[i] + residual_row[i] * PHI_INV_F32;
        local_sum += temp;
    }

    local_sum = warp_reduce_sum(local_sum);

    if (tid % 32 == 0) {
        atomicAdd(s_mean, local_sum);
    }
    __syncthreads();

    float mean = *s_mean / feature_dim;
    __syncthreads();

    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float temp = identity_row[i] + residual_row[i] * PHI_INV_F32;
        float diff = temp - mean;
        local_var += diff * diff;
    }

    local_var = warp_reduce_sum(local_var);

    if (tid % 32 == 0) {
        atomicAdd(s_var, local_var);
    }
    __syncthreads();

    float var = *s_var / feature_dim;
    float rstd = rsqrtf(var + eps);

    // Normalize and apply affine
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float temp = identity_row[i] + residual_row[i] * PHI_INV_F32;
        float normalized = (temp - mean) * rstd;

        if (gamma != NULL) normalized *= gamma[i];
        if (beta != NULL) normalized += beta[i];

        out_row[i] = normalized;
    }
}

// =============================================================================
// Vectorized variants (for coalesced memory access)
// =============================================================================

extern "C" __global__ void phi_residual_mode_phi_vec4_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n  // Must be multiple of 4
) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 id = *((float4*)(&identity[i]));
        float4 res = *((float4*)(&residual[i]));

        float4 result;
        result.x = id.x + res.x * PHI_INV_F32;
        result.y = id.y + res.y * PHI_INV_F32;
        result.z = id.z + res.z * PHI_INV_F32;
        result.w = id.w + res.w * PHI_INV_F32;

        *((float4*)(&out[i])) = result;
    }
}

extern "C" __global__ void phi_residual_mode_phi_vec2_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n  // Must be multiple of 2
) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        double2 id = *((double2*)(&identity[i]));
        double2 res = *((double2*)(&residual[i]));

        double2 result;
        result.x = id.x + res.x * PHI_INV_F64;
        result.y = id.y + res.y * PHI_INV_F64;

        *((double2*)(&out[i])) = result;
    }
}

// =============================================================================
// Diagnostic kernel: compute norm of residual component
// =============================================================================
// Useful for monitoring residual magnitude during training

extern "C" __global__ void phi_residual_component_norm_f64(
    double *norm_out,              // Single output value
    const double *residual,
    int n
) {
    extern __shared__ double s_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute local sum of squares
    double local_sum = 0.0;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        double val = residual[idx] * PHI_INV_F64;
        local_sum += val * val;
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction
    if (tid < 32) {
        double val = s_sum[tid];
        val = warp_reduce_sum_f64(val + ((tid + 32 < blockDim.x) ? s_sum[tid + 32] : 0.0));

        if (tid == 0) {
            // Atomic add for final accumulation
            unsigned long long int* addr = (unsigned long long int*)norm_out;
            unsigned long long int old, assumed;
            old = *addr;
            do {
                assumed = old;
                old = atomicCAS(addr, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
            } while (assumed != old);
        }
    }
}
