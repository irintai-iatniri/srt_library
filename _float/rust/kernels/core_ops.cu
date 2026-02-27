// Syntonic CUDA Kernels - Core Mathematical Operations
// Element-wise exp, reductions, layer normalization, dropout
// Part of SRT Resonant Engine (requires GPU + CPU dual-state paradigm)

#include "srt_constants.cuh"

// =============================================================================
// Element-wise Exponential
// =============================================================================

extern "C" __global__ void exp_f64(double *out, const double *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp(in[i]);
}

extern "C" __global__ void exp_f32(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(in[i]);  // Fast intrinsic
}

// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
extern "C" __global__ void exp_c128(double *out, const double *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int idx = i * 2;
    double re = in[idx];
    double im = in[idx + 1];
    
    double exp_re = exp(re);
    double cos_im, sin_im;
    sincos(im, &sin_im, &cos_im);
    
    out[idx] = exp_re * cos_im;
    out[idx + 1] = exp_re * sin_im;
}

// =============================================================================
// Golden Exponential: exp(-x/φ) — fundamental for SRT measure w(n)
// =============================================================================

extern "C" __global__ void exp_golden_f64(double *out, const double *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp(-in[i] * PHI_INV_F64);
}

extern "C" __global__ void exp_golden_f32(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(-in[i] * PHI_INV_F32);
}

// =============================================================================
// Reduction Sum (shared memory + warp reduction)
// =============================================================================

extern "C" __global__ void reduce_sum_f64(
    double *out,
    const double *in,
    int n
) {
    extern __shared__ double sdata_d[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    double val = (i < n) ? in[i] : 0.0;
    sdata_d[tid] = val;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        val = sdata_d[tid];
        val = warp_reduce_sum_f64(val + ((tid + 32 < blockDim.x) ? sdata_d[tid + 32] : 0.0));
    }
    
    // Write block result
    if (tid == 0) {
        // Atomic add for double
        unsigned long long int* addr = (unsigned long long int*)out;
        unsigned long long int old, assumed;
        old = *addr;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

extern "C" __global__ void reduce_sum_f32(
    float *out,
    const float *in,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (i < n) ? in[i] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        val = sdata[tid];
        val = warp_reduce_sum(val + ((tid + 32 < blockDim.x) ? sdata[tid + 32] : 0.0f));
    }
    
    if (tid == 0) {
        atomicAdd(out, val);
    }
}

// =============================================================================
// Layer Normalization with Golden Target Variance
// =============================================================================
// 
// Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// SRT Golden LayerNorm: target variance = 1/φ ≈ 0.618
//   y = (x - mean) / sqrt(var + eps) * sqrt(1/φ) * gamma + beta
//
// This kernel computes per-sample normalization across feature_dim

extern "C" __global__ void layer_norm_f64(
    double *out,
    const double *in,
    const double *gamma,      // Scale [feature_dim], NULL for 1.0
    const double *beta,       // Bias [feature_dim], NULL for 0.0
    double eps,
    int batch_size,
    int feature_dim,
    bool golden_target        // If true, scale to variance = 1/φ
) {
    extern __shared__ double shared[];
    double *s_mean = shared;
    double *s_var = shared + 1;
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const double *x = in + batch_idx * feature_dim;
    double *y = out + batch_idx * feature_dim;
    
    // Pass 1: Compute mean
    double local_sum = 0.0;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        local_sum += x[i];
    }
    
    // Warp reduction
    local_sum = warp_reduce_sum_f64(local_sum);
    
    if (tid % 32 == 0) {
        atomicAdd(s_mean, local_sum);
    }
    __syncthreads();
    
    double mean = *s_mean / feature_dim;
    __syncthreads();
    
    // Pass 2: Compute variance
    double local_var = 0.0;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        double diff = x[i] - mean;
        local_var += diff * diff;
    }
    
    local_var = warp_reduce_sum_f64(local_var);
    
    if (tid % 32 == 0) {
        atomicAdd(s_var, local_var);
    }
    __syncthreads();
    
    double var = *s_var / feature_dim;
    double rstd = rsqrt(var + eps);
    
    // Golden scaling: multiply by sqrt(1/φ) to target variance = 1/φ
    if (golden_target) {
        rstd *= sqrt(PHI_INV_F64);
    }
    
    // Pass 3: Normalize
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        double val = (x[i] - mean) * rstd;
        if (gamma != NULL) val *= gamma[i];
        if (beta != NULL) val += beta[i];
        y[i] = val;
    }
}

// =============================================================================
// Dropout (training mode with seeded RNG; identity at inference)
// =============================================================================

extern "C" __global__ void dropout_f64(
    double *out,
    const double *in,
    double scale,             // 1 / (1 - p) for "inverted dropout"
    unsigned long long seed,
    int n,
    bool training
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    if (!training) {
        out[i] = in[i];
        return;
    }
    
    // xorshift64 RNG (same pattern as resonant_d.cu)
    unsigned long long state = seed ^ (i * 2654435761ULL);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    double u = (double)(state * 0x2545F4914F6CDD1DULL) / (double)ULLONG_MAX;
    
    // Inverted dropout: scale active units by 1/(1-p)
    double mask = (u < (1.0 - 1.0/scale)) ? 0.0 : scale;
    out[i] = in[i] * mask;
}

extern "C" __global__ void dropout_f32(
    float *out,
    const float *in,
    float scale,
    unsigned long long seed,
    int n,
    bool training
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    if (!training) {
        out[i] = in[i];
        return;
    }
    
    unsigned long long state = seed ^ (i * 2654435761ULL);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    float u = (float)(state * 0x2545F4914F6CDD1DULL) / (float)ULLONG_MAX;
    
    float mask = (u < (1.0f - 1.0f/scale)) ? 0.0f : scale;
    out[i] = in[i] * mask;
}

// =============================================================================
// Sigmoid Activation: σ(x) = 1 / (1 + exp(-x))
// =============================================================================

extern "C" __global__ void sigmoid_f64(double *out, const double *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0 / (1.0 + exp(-in[i]));
}

extern "C" __global__ void sigmoid_f32(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + __expf(-in[i]));
}

// Tanh activation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
extern "C" __global__ void tanh_f64(double *out, const double *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanh(in[i]);
}

extern "C" __global__ void tanh_f32(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(in[i]);
}
