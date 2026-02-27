// Syntonic CUDA Kernels - Heat Kernel and Theta Series
// Critical path for SRT lattice computations

#include "srt_constants.cuh"

// =============================================================================
// Shared Memory Cache for E₈ Root Data
// =============================================================================

// Structure for caching E₈ root data in shared memory
// Each root is 8 floats + 1 float for Q + 1 int for cone = 10 values
#define E8_ROOT_CACHE_SIZE  240

// Shared memory layout per block processing E₈ roots
struct E8RootCache {
    float roots[E8_ROOT_CACHE_SIZE][8];  // 8D root vectors
    float Q[E8_ROOT_CACHE_SIZE];          // Precomputed Q(λ)
    int in_cone[E8_ROOT_CACHE_SIZE];      // Golden cone membership
};

// =============================================================================
// Theta Series: Θ(t) = Σ_λ w(λ) exp(-π Q(λ) / t)
// =============================================================================

// Basic theta series sum over pre-filtered golden cone roots
extern "C" __global__ void theta_series_sum_f32(
    float *result,
    const float *Q_values,     // Q(λ) for each root
    const int *in_cone,        // Cone membership flags
    const float *weights,      // Optional additional weights (NULL for unit)
    float t,                   // Modular parameter
    int count                  // Number of roots
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < count && in_cone[i]) {
        float Q = Q_values[i];
        float w = (weights != NULL) ? weights[i] : 1.0f;
        local_sum = w * __expf(-PI_F32 * Q / t);
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

extern "C" __global__ void theta_series_sum_f64(
    double *result,
    const double *Q_values,
    const int *in_cone,
    const double *weights,
    double t,
    int count
) {
    extern __shared__ double sdata_d[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < count && in_cone[i]) {
        double Q = Q_values[i];
        double w = (weights != NULL) ? weights[i] : 1.0;
        local_sum = w * exp(-PI_F64 * Q / t);
    }
    sdata_d[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Use custom double atomic add
        unsigned long long int* address_as_ull = (unsigned long long int*)result;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(sdata_d[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Heat Kernel K(t) = Σ_n w(n) K(λ_n, t)
// =============================================================================

// Vignéras-type heat kernel approximation
// K(λ, t) ≈ t^(-d/2) exp(-π |P_∥λ|² / t) × correction_term
__device__ __forceinline__ float vigneras_kernel_f32(
    float parallel_norm_sq,  // |P_∥λ|²
    float t_inv              // 1/t for efficiency
) {
    // Basic heat kernel without full correction term
    float t = 1.0f / t_inv;
    float prefactor = rsqrtf(t * t);  // t^(-2/2) for 4D parallel space
    return prefactor * __expf(-PI_F32 * parallel_norm_sq * t_inv);
}

__device__ __forceinline__ double vigneras_kernel_f64(
    double parallel_norm_sq,
    double t_inv
) {
    double t = 1.0 / t_inv;
    double prefactor = rsqrt(t * t);
    return prefactor * exp(-PI_F64 * parallel_norm_sq * t_inv);
}

// Full heat kernel with parallel projection
extern "C" __global__ void heat_kernel_e8_f32(
    float *result,
    const float *proj_parallel,  // 4D parallel projections (pre-computed)
    const int *in_cone,          // Cone membership
    float t,                     // Modular parameter
    int count
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float t_inv = 1.0f / t;

    float local_sum = 0.0f;
    if (i < count && in_cone[i]) {
        int idx = i * 4;
        float p0 = proj_parallel[idx];
        float p1 = proj_parallel[idx + 1];
        float p2 = proj_parallel[idx + 2];
        float p3 = proj_parallel[idx + 3];
        float parallel_norm_sq = p0*p0 + p1*p1 + p2*p2 + p3*p3;

        local_sum = vigneras_kernel_f32(parallel_norm_sq, t_inv);
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

extern "C" __global__ void heat_kernel_e8_f64(
    double *result,
    const double *proj_parallel,
    const int *in_cone,
    double t,
    int count
) {
    extern __shared__ double sdata_d[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double t_inv = 1.0 / t;

    double local_sum = 0.0;
    if (i < count && in_cone[i]) {
        int idx = i * 4;
        double p0 = proj_parallel[idx];
        double p1 = proj_parallel[idx + 1];
        double p2 = proj_parallel[idx + 2];
        double p3 = proj_parallel[idx + 3];
        double parallel_norm_sq = p0*p0 + p1*p1 + p2*p2 + p3*p3;

        local_sum = vigneras_kernel_f64(parallel_norm_sq, t_inv);
    }
    sdata_d[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long int* address_as_ull = (unsigned long long int*)result;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(sdata_d[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Shell-Based Theta Series (Multi-Shell Summation)
// =============================================================================

// Sum over shells: Θ(t) = Σ_{shell} Σ_{λ ∈ shell} w(λ) exp(-π Q(λ) / t)
// More efficient when roots are pre-sorted by shell (norm² = 2 for first shell)
extern "C" __global__ void theta_series_shells_f32(
    float *shell_sums,           // Output: sum for each shell
    const float *Q_values,       // Q(λ) for all roots
    const int *in_cone,          // Cone membership
    const int *shell_offsets,    // Start index of each shell
    const int *shell_sizes,      // Number of roots in each shell
    float t,
    int num_shells
) {
    int shell_id = blockIdx.x;
    if (shell_id >= num_shells) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int offset = shell_offsets[shell_id];
    int size = shell_sizes[shell_id];

    float local_sum = 0.0f;
    for (int j = tid; j < size; j += blockDim.x) {
        int i = offset + j;
        if (in_cone[i]) {
            float Q = Q_values[i];
            local_sum += __expf(-PI_F32 * Q / t);
        }
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
        shell_sums[shell_id] = sdata[0];
    }
}

// =============================================================================
// Golden Measure Weighted Summation
// =============================================================================

// Sum with golden measure: Σ w(n) f(n) where w(n) = exp(-|n|²/φ)
extern "C" __global__ void golden_weighted_sum_f32(
    float *result,
    const float *values,        // f(n) values
    const float *norm_sq,       // |n|² for each point
    int count
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < count) {
        float w = __expf(-norm_sq[i] * PHI_INV_F32);
        local_sum = w * values[i];
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

extern "C" __global__ void golden_weighted_sum_f64(
    double *result,
    const double *values,
    const double *norm_sq,
    int count
) {
    extern __shared__ double sdata_d[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < count) {
        double w = exp(-norm_sq[i] * PHI_INV_F64);
        local_sum = w * values[i];
    }
    sdata_d[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long int* address_as_ull = (unsigned long long int*)result;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(sdata_d[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Modular Transform: f(t) → f(1/t)
// =============================================================================

// Apply modular inversion to theta values (used for functional equations)
extern "C" __global__ void modular_inversion_f32(
    float *out, const float *in, const float *t_values, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Under S-duality: Θ(1/t) = t^2 Θ(t) (for 4D projection)
        float t = t_values[i];
        float t_inv = 1.0f / t;
        // Use t_inv directly to compute scaling: Θ(1/t) = t^2 Θ(t)
        // t^2 = 1 / (t_inv * t_inv)
        out[i] = in[i] * (1.0f / (t_inv * t_inv));
    }
}

extern "C" __global__ void modular_inversion_f64(
    double *out, const double *in, const double *t_values, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double t = t_values[i];
        out[i] = in[i] * (t * t);
    }
}

// =============================================================================
// Spectral Zeta Function ζ_Θ(s) = Σ λ_n^(-s) (Mellin Transform)
// =============================================================================

// Compute spectral zeta values at given s
extern "C" __global__ void spectral_zeta_f64(
    double *result,
    const double *eigenvalues,  // λ_n (positive eigenvalues)
    double s,                    // Zeta argument (Re(s) > d/2 for convergence)
    int count
) {
    extern __shared__ double sdata_d[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < count && eigenvalues[i] > 0.0) {
        local_sum = pow(eigenvalues[i], -s);
    }
    sdata_d[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long int* address_as_ull = (unsigned long long int*)result;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(sdata_d[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Winding Number Lattice Heat Kernel
// =============================================================================

// Heat kernel on winding lattice: K(n, n', t) contribution
extern "C" __global__ void winding_heat_kernel_f32(
    float *result,
    const int *windings,         // 4D winding vectors (packed as 4×int)
    float t,
    int count
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < count) {
        int idx = i * 4;
        int n7 = windings[idx];
        int n8 = windings[idx + 1];
        int n9 = windings[idx + 2];
        int n10 = windings[idx + 3];

        float norm_sq = (float)(n7*n7 + n8*n8 + n9*n9 + n10*n10);
        // Golden-weighted heat kernel
        float golden_weight = __expf(-norm_sq * PHI_INV_F32);
        float heat_contrib = __expf(-PI_F32 * norm_sq / t);

        local_sum = golden_weight * heat_contrib;
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
// Knot Invariant Contribution (Alexander Polynomial Style)
// =============================================================================

// Contribution from a knot winding: Δ(n) factor
extern "C" __global__ void knot_contribution_f32(
    float *result,
    const int *windings,
    const float *knot_weights,   // Pre-computed knot polynomial weights
    int count
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < count) {
        int idx = i * 4;
        int n7 = windings[idx];
        int n8 = windings[idx + 1];
        int n9 = windings[idx + 2];
        int n10 = windings[idx + 3];

        float norm_sq = (float)(n7*n7 + n8*n8 + n9*n9 + n10*n10);
        float golden_weight = __expf(-norm_sq * PHI_INV_F32);

        // Knot weight from pre-computed table
        float knot_w = knot_weights[i];

        local_sum = golden_weight * knot_w;
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
