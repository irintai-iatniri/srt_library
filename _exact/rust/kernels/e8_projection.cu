// Syntonic CUDA Kernels - E₈ Lattice Projection Operations
// Golden projections P_φ and P_⊥ for Syntonic Resonance Theory

#include "srt_constants.cuh"

// =============================================================================
// Constant Memory for Projection Matrices
// =============================================================================

// Golden projection matrix P_φ: ℝ⁸ → ℝ⁴ (parallel subspace)
// Each row projects 8D E₈ vectors to 4D physical space
// P_φ[i][j] = (1/√(2φ+2)) × pattern where:
//   Row 0: [φ, 1, 0, 0, 0, 0, 0, 0]
//   Row 1: [0, 0, φ, 1, 0, 0, 0, 0]
//   Row 2: [0, 0, 0, 0, φ, 1, 0, 0]
//   Row 3: [0, 0, 0, 0, 0, 0, φ, 1]
__constant__ float c_P_phi_f32[4][8] = {
    {PHI_F32 * PROJ_NORM_F32, PROJ_NORM_F32, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, PHI_F32 * PROJ_NORM_F32, PROJ_NORM_F32, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, PHI_F32 * PROJ_NORM_F32, PROJ_NORM_F32, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, PHI_F32 * PROJ_NORM_F32, PROJ_NORM_F32}
};

// Perpendicular projection matrix P_⊥: ℝ⁸ → ℝ⁴ (internal subspace)
// P_⊥[i][j] = (1/√(2φ+2)) × pattern where:
//   Row 0: [1, -φ, 0, 0, 0, 0, 0, 0]
//   Row 1: [0, 0, 1, -φ, 0, 0, 0, 0]
//   Row 2: [0, 0, 0, 0, 1, -φ, 0, 0]
//   Row 3: [0, 0, 0, 0, 0, 0, 1, -φ]
__constant__ float c_P_perp_f32[4][8] = {
    {PROJ_NORM_F32, -PHI_F32 * PROJ_NORM_F32, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, PROJ_NORM_F32, -PHI_F32 * PROJ_NORM_F32, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, PROJ_NORM_F32, -PHI_F32 * PROJ_NORM_F32, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, PROJ_NORM_F32, -PHI_F32 * PROJ_NORM_F32}
};

// Double precision versions
__constant__ double c_P_phi_f64[4][8] = {
    {PHI_F64 * PROJ_NORM_F64, PROJ_NORM_F64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, PHI_F64 * PROJ_NORM_F64, PROJ_NORM_F64, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, PHI_F64 * PROJ_NORM_F64, PROJ_NORM_F64, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, PHI_F64 * PROJ_NORM_F64, PROJ_NORM_F64}
};

__constant__ double c_P_perp_f64[4][8] = {
    {PROJ_NORM_F64, -PHI_F64 * PROJ_NORM_F64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, PROJ_NORM_F64, -PHI_F64 * PROJ_NORM_F64, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, PROJ_NORM_F64, -PHI_F64 * PROJ_NORM_F64, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, PROJ_NORM_F64, -PHI_F64 * PROJ_NORM_F64}
};

// =============================================================================
// Golden Cone Null Vectors (B_a for a = 1,2,3,4)
// =============================================================================

// Null vectors defining the golden cone boundary
// B_a(λ) ≥ 0 for λ in golden cone
__constant__ float c_null_vectors_f32[4][8] = {
    {1.0f, -PHI_F32, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, -PHI_F32, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -PHI_F32, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -PHI_F32}
};

// =============================================================================
// Basic Projection Operations
// =============================================================================

// Project 8D vector to 4D parallel subspace: proj_∥ = P_φ · λ
extern "C" __global__ void project_parallel_f32(
    float *proj, const float *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        float sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            sum = __fmaf_rn(c_P_phi_f32[row][col], lambda[in_idx + col], sum);
        }
        proj[out_idx + row] = sum;
    }
}

extern "C" __global__ void project_parallel_f64(
    double *proj, const double *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        double sum = 0.0;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            sum = fma(c_P_phi_f64[row][col], lambda[in_idx + col], sum);
        }
        proj[out_idx + row] = sum;
    }
}

// Project 8D vector to 4D perpendicular subspace: proj_⊥ = P_⊥ · λ
extern "C" __global__ void project_perpendicular_f32(
    float *proj, const float *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        float sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            sum = __fmaf_rn(c_P_perp_f32[row][col], lambda[in_idx + col], sum);
        }
        proj[out_idx + row] = sum;
    }
}

extern "C" __global__ void project_perpendicular_f64(
    double *proj, const double *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        double sum = 0.0;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            sum = fma(c_P_perp_f64[row][col], lambda[in_idx + col], sum);
        }
        proj[out_idx + row] = sum;
    }
}

// =============================================================================
// Quadratic Form Q(λ) = ||P_∥λ||² - ||P_⊥λ||²
// =============================================================================

// Compute quadratic form for 8D vectors
extern "C" __global__ void quadratic_form_f32(
    float *Q, const float *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    float parallel_sq = 0.0f;
    float perp_sq = 0.0f;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        float p = 0.0f, q = 0.0f;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            p = __fmaf_rn(c_P_phi_f32[row][col], lambda[idx + col], p);
            q = __fmaf_rn(c_P_perp_f32[row][col], lambda[idx + col], q);
        }
        parallel_sq += p * p;
        perp_sq += q * q;
    }

    Q[i] = parallel_sq - perp_sq;
}

extern "C" __global__ void quadratic_form_f64(
    double *Q, const double *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    double parallel_sq = 0.0;
    double perp_sq = 0.0;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        double p = 0.0, q = 0.0;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            p = fma(c_P_phi_f64[row][col], lambda[idx + col], p);
            q = fma(c_P_perp_f64[row][col], lambda[idx + col], q);
        }
        parallel_sq += p * p;
        perp_sq += q * q;
    }

    Q[i] = parallel_sq - perp_sq;
}

// =============================================================================
// Golden Cone Test: is_in_cone = ∧_a (B_a(λ) ≥ 0)
// =============================================================================

// Test if 8D vector is in the golden cone
extern "C" __global__ void golden_cone_test_f32(
    int *in_cone, const float *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    int is_in = 1;

    #pragma unroll
    for (int a = 0; a < 4; a++) {
        float B_a = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            B_a = __fmaf_rn(c_null_vectors_f32[a][j], lambda[idx + j], B_a);
        }
        if (B_a < 0.0f) {
            is_in = 0;
            break;
        }
    }

    in_cone[i] = is_in;
}

extern "C" __global__ void golden_cone_test_f64(
    int *in_cone, const double *lambda, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    int is_in = 1;

    // Using c_null_vectors_f32 cast to double
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        double B_a = 0.0;
        B_a += (double)c_null_vectors_f32[a][2*a] * lambda[idx + 2*a];
        B_a += (double)c_null_vectors_f32[a][2*a + 1] * lambda[idx + 2*a + 1];
        if (B_a < 0.0) {
            is_in = 0;
            break;
        }
    }

    in_cone[i] = is_in;
}

// =============================================================================
// Fused Operations: Projection + Q + Cone Test (for E₈ root processing)
// =============================================================================

// Batch projection with Q and cone test for all 240 E₈ roots
// This is the hot path kernel - processes all roots efficiently
extern "C" __global__ void e8_batch_projection_f32(
    float *proj_parallel,   // Output: 240 × 4 parallel projections
    float *proj_perp,       // Output: 240 × 4 perpendicular projections
    float *Q_values,        // Output: 240 quadratic forms
    int *in_cone,           // Output: 240 cone membership flags
    const float *roots,     // Input: 240 × 8 E₈ roots
    int count               // Usually 240
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    float parallel[4], perp[4];
    float parallel_sq = 0.0f, perp_sq = 0.0f;
    int is_in = 1;

    // Compute both projections simultaneously
    #pragma unroll
    for (int row = 0; row < 4; row++) {
        float p = 0.0f, q = 0.0f;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            float val = roots[in_idx + col];
            p = __fmaf_rn(c_P_phi_f32[row][col], val, p);
            q = __fmaf_rn(c_P_perp_f32[row][col], val, q);
        }
        parallel[row] = p;
        perp[row] = q;
        parallel_sq += p * p;
        perp_sq += q * q;
    }

    // Cone test (using null vectors structure)
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        float B_a = roots[in_idx + 2*a] - PHI_F32 * roots[in_idx + 2*a + 1];
        if (B_a < 0.0f) {
            is_in = 0;
        }
    }

    // Write outputs
    #pragma unroll
    for (int row = 0; row < 4; row++) {
        proj_parallel[out_idx + row] = parallel[row];
        proj_perp[out_idx + row] = perp[row];
    }
    Q_values[i] = parallel_sq - perp_sq;
    in_cone[i] = is_in;
}

extern "C" __global__ void e8_batch_projection_f64(
    double *proj_parallel,
    double *proj_perp,
    double *Q_values,
    int *in_cone,
    const double *roots,
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int in_idx = i * 8;
    int out_idx = i * 4;

    double parallel[4], perp[4];
    double parallel_sq = 0.0, perp_sq = 0.0;
    int is_in = 1;

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        double p = 0.0, q = 0.0;
        #pragma unroll
        for (int col = 0; col < 8; col++) {
            double val = roots[in_idx + col];
            p = fma(c_P_phi_f64[row][col], val, p);
            q = fma(c_P_perp_f64[row][col], val, q);
        }
        parallel[row] = p;
        perp[row] = q;
        parallel_sq += p * p;
        perp_sq += q * q;
    }

    #pragma unroll
    for (int a = 0; a < 4; a++) {
        double B_a = roots[in_idx + 2*a] - PHI_F64 * roots[in_idx + 2*a + 1];
        if (B_a < 0.0) {
            is_in = 0;
        }
    }

    #pragma unroll
    for (int row = 0; row < 4; row++) {
        proj_parallel[out_idx + row] = parallel[row];
        proj_perp[out_idx + row] = perp[row];
    }
    Q_values[i] = parallel_sq - perp_sq;
    in_cone[i] = is_in;
}

// =============================================================================
// Filter Golden Cone Roots (Compact Stream)
// =============================================================================

// Count how many roots are in the golden cone
extern "C" __global__ void count_cone_roots(
    int *count, const int *in_cone, int n
) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int local_count = (i < n && in_cone[i]) ? 1 : 0;
    sdata[tid] = local_count;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, sdata[0]);
    }
}

// =============================================================================
// E₈ Root Shell Enumeration (for heat kernel)
// =============================================================================

// Compute squared norm of 8D vectors (for shell sorting)
extern "C" __global__ void norm_squared_8d_f32(
    float *norms_sq, const float *vectors, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float v = vectors[idx + j];
        sum = __fmaf_rn(v, v, sum);
    }
    norms_sq[i] = sum;
}

extern "C" __global__ void norm_squared_8d_f64(
    double *norms_sq, const double *vectors, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int idx = i * 8;
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        double v = vectors[idx + j];
        sum = fma(v, v, sum);
    }
    norms_sq[i] = sum;
}

// =============================================================================
// Weight E₈ Roots by Q(λ) and Cone Membership
// =============================================================================

// Compute weighted contribution: w(λ) = in_cone(λ) * exp(-π Q(λ) / t)
extern "C" __global__ void weighted_e8_contribution_f32(
    float *weights,
    const float *Q_values,
    const int *in_cone,
    float t,              // Modular parameter
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    if (in_cone[i]) {
        float Q = Q_values[i];
        weights[i] = __expf(-PI_F32 * Q / t);
    } else {
        weights[i] = 0.0f;
    }
}

extern "C" __global__ void weighted_e8_contribution_f64(
    double *weights,
    const double *Q_values,
    const int *in_cone,
    double t,
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    if (in_cone[i]) {
        double Q = Q_values[i];
        weights[i] = exp(-PI_F64 * Q / t);
    } else {
        weights[i] = 0.0;
    }
}
