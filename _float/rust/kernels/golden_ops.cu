// Syntonic CUDA Kernels - Golden Ratio Operations
// Core SRT operations involving the golden ratio φ

#include "srt_constants.cuh"

// =============================================================================
// Golden Ratio Scaling Operations
// =============================================================================

// Scale by φ: out = a * φ
extern "C" __global__ void scale_phi_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * PHI_F64;
}

extern "C" __global__ void scale_phi_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * PHI_F32;
}

// Scale by φ⁻¹: out = a * φ⁻¹
extern "C" __global__ void scale_phi_inv_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * PHI_INV_F64;
}

extern "C" __global__ void scale_phi_inv_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * PHI_INV_F32;
}

// =============================================================================
// Golden Fused Multiply-Add Operations
// =============================================================================

// Fused multiply-add with φ: out = a * φ + b
extern "C" __global__ void fma_phi_kernel_f64(
    double *out, const double *a, const double *b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fma(a[i], PHI_F64, b[i]);
}

extern "C" __global__ void fma_phi_kernel_f32(
    float *out, const float *a, const float *b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmaf_rn(a[i], PHI_F32, b[i]);
}

// Fused multiply-add with φ⁻¹: out = a * φ⁻¹ + b
extern "C" __global__ void fma_phi_inv_kernel_f64(
    double *out, const double *a, const double *b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fma(a[i], PHI_INV_F64, b[i]);
}

extern "C" __global__ void fma_phi_inv_kernel_f32(
    float *out, const float *a, const float *b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmaf_rn(a[i], PHI_INV_F32, b[i]);
}

// =============================================================================
// Golden Gaussian Weight (Fundamental Measure)
// =============================================================================

// Golden Gaussian weight for single values: w = exp(-x/φ)
extern "C" __global__ void golden_gaussian_weight_scalar_f64(
    double *out, const double *x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp(-x[i] * PHI_INV_F64);
}

extern "C" __global__ void golden_gaussian_weight_scalar_f32(
    float *out, const float *x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(-x[i] * PHI_INV_F32);
}

// Golden Gaussian weight for 4D windings: w(n) = exp(-|n|²/φ)
// Input: windings as packed int4 (n₇, n₈, n₉, n₁₀)
extern "C" __global__ void golden_gaussian_weight_4d_int(
    float *weights, const int *windings, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 4;
    int n7 = windings[idx];
    int n8 = windings[idx + 1];
    int n9 = windings[idx + 2];
    int n10 = windings[idx + 3];

    float norm_sq = (float)(n7*n7 + n8*n8 + n9*n9 + n10*n10);
    weights[i] = __expf(-norm_sq * PHI_INV_F32);
}

// Golden Gaussian weight for 4D float vectors: w(x) = exp(-|x|²/φ)
extern "C" __global__ void golden_gaussian_weight_4d_f32(
    float *weights, const float *vectors, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 4;
    float x0 = vectors[idx];
    float x1 = vectors[idx + 1];
    float x2 = vectors[idx + 2];
    float x3 = vectors[idx + 3];

    float norm_sq = x0*x0 + x1*x1 + x2*x2 + x3*x3;
    weights[i] = __expf(-norm_sq * PHI_INV_F32);
}

// Golden Gaussian weight for 8D vectors (E₈): w(λ) = exp(-|λ|²/φ)
extern "C" __global__ void golden_gaussian_weight_8d_f32(
    float *weights, const float *vectors, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float v = vectors[idx + j];
        norm_sq += v * v;
    }
    weights[i] = __expf(-norm_sq * PHI_INV_F32);
}

extern "C" __global__ void golden_gaussian_weight_8d_f64(
    double *weights, const double *vectors, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    double norm_sq = 0.0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        double v = vectors[idx + j];
        norm_sq += v * v;
    }
    weights[i] = exp(-norm_sq * PHI_INV_F64);
}

// =============================================================================
// Golden Recursion Map: R(n) = ⌊φn⌋
// =============================================================================

// Golden recursion for 4D integer windings
extern "C" __global__ void golden_recursion_4d_int(
    int *out, const int *in, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 4;
    out[idx]     = __float2int_rd(PHI_F32 * in[idx]);
    out[idx + 1] = __float2int_rd(PHI_F32 * in[idx + 1]);
    out[idx + 2] = __float2int_rd(PHI_F32 * in[idx + 2]);
    out[idx + 3] = __float2int_rd(PHI_F32 * in[idx + 3]);
}

// Golden recursion for float values: R(x) = x * φ
extern "C" __global__ void golden_recursion_f32(
    float *out, const float *in, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * PHI_F32;
}

extern "C" __global__ void golden_recursion_f64(
    double *out, const double *in, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * PHI_F64;
}

// Inverse golden recursion: R⁻¹(n) = ⌊n/φ⌋ = ⌊n * φ⁻¹⌋
extern "C" __global__ void golden_recursion_inv_4d_int(
    int *out, const int *in, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 4;
    out[idx]     = __float2int_rd(PHI_INV_F32 * in[idx]);
    out[idx + 1] = __float2int_rd(PHI_INV_F32 * in[idx + 1]);
    out[idx + 2] = __float2int_rd(PHI_INV_F32 * in[idx + 2]);
    out[idx + 3] = __float2int_rd(PHI_INV_F32 * in[idx + 3]);
}

// =============================================================================
// Fibonacci/Lucas Number Generation (Related to φ)
// =============================================================================

// F(k) via Binet's formula: F(k) = (φᵏ - (-φ)⁻ᵏ) / √5
extern "C" __global__ void fibonacci_binet_f64(
    double *out, const int *k_values, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int k = k_values[i];
    double phi_k = pow(PHI_F64, (double)k);
    double neg_phi_inv_k = pow(-PHI_INV_F64, (double)k);
    out[i] = (phi_k - neg_phi_inv_k) / SQRT5_F64;
}

// Lucas numbers: L(k) = φᵏ + (-φ)⁻ᵏ
extern "C" __global__ void lucas_binet_f64(
    double *out, const int *k_values, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int k = k_values[i];
    double phi_k = pow(PHI_F64, (double)k);
    double neg_phi_inv_k = pow(-PHI_INV_F64, (double)k);
    out[i] = phi_k + neg_phi_inv_k;
}

// =============================================================================
// Generation Number Computation: g(n) = 1 + ⌊log_φ(max|nᵢ|)⌋
// =============================================================================

extern "C" __global__ void compute_generation_4d_int(
    int *generations, const int *windings, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 4;
    int max_abs = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int val = windings[idx + j];
        int abs_val = val >= 0 ? val : -val;
        if (abs_val > max_abs) max_abs = abs_val;
    }

    if (max_abs == 0) {
        generations[i] = 1;
    } else {
        // log_φ(x) = ln(x) / ln(φ)
        float log_phi = logf((float)max_abs) / logf(PHI_F32);
        generations[i] = 1 + (int)floorf(log_phi);
    }
}

// =============================================================================
// Weighted Inner Products with Golden Measure
// =============================================================================

// Weighted inner product: ⟨a, b⟩_φ = Σᵢ aᵢ * bᵢ * exp(-i²/φ)
extern "C" __global__ void weighted_inner_product_golden_f32(
    float *result, const float *a, const float *b, int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float weight = __expf(-(float)(i * i) * PHI_INV_F32);
        local_sum = a[i] * b[i] * weight;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// Golden Normalization (Project to Golden Measure)
// =============================================================================

// Normalize to golden distribution: out[i] = a[i] * exp(-i/φ) / Z
// where Z = Σⱼ |a[j]|² * exp(-2j/φ)
extern "C" __global__ void golden_normalize_f32(
    float *out, const float *a, const float *normalization, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float weight = __expf(-(float)i * PHI_INV_F32);
        float Z = *normalization;
        out[i] = a[i] * weight / sqrtf(Z + 1e-10f);
    }
}

// Compute normalization factor for golden distribution
extern "C" __global__ void golden_norm_factor_f32(
    float *result, const float *a, int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float weight = __expf(-2.0f * (float)i * PHI_INV_F32);
        local_sum = a[i] * a[i] * weight;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// Complex Golden Operations (interleaved format)
// =============================================================================

// Scale complex by φ: out = z * φ
extern "C" __global__ void scale_phi_c128(
    double *out, const double *z, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (i < n) {
        out[idx] = z[idx] * PHI_F64;
        out[idx + 1] = z[idx + 1] * PHI_F64;
    }
}

// Golden Gaussian weight for complex amplitudes: w = exp(-|z|²/φ)
extern "C" __global__ void golden_gaussian_weight_c128(
    double *weights, const double *z, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double re = z[idx];
    double im = z[idx + 1];
    double norm_sq = re * re + im * im;
    weights[i] = exp(-norm_sq * PHI_INV_F64);
}
