// Syntonic CUDA Kernels - Resonant D-Phase Operations
// GPU = D̂ (Differentiation) - chaotic flux generator for Resonant Engine
//
// The D-phase introduces stochastic perturbations to the crystallized lattice,
// generating chaotic flux that will be re-crystallized in the H-phase (CPU).
// Noise scale is modulated by syntony: lower syntony = more exploration.

#include "srt_constants.cuh"

// =============================================================================
// D-Phase: Lattice → Flux (Wake the Shadow)
// =============================================================================

// D-phase transformation: adds syntony-dependent stochastic noise
// flux[i] = lattice[i] * (1 + α(S) * √|n_i|²) + noise_scale * (1 - S) * noise[i]
//
// The differentiation operator spreads energy to higher modes while adding
// controlled stochasticity. The noise term is weighted by (1 - S) so that
// high-syntony states receive less perturbation.
extern "C" __global__ void resonant_d_phase_f64(
    double *flux,               // Output: ephemeral float values
    const double *lattice,      // Input: crystallized values
    const double *mode_norm_sq, // |n|² for each mode
    const double *noise,        // Pre-generated Gaussian noise
    double syntony,             // Current syntony S ∈ [0, 1]
    double noise_scale,         // Base noise amplitude
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double val = lattice[i];
    double norm_sq = mode_norm_sq[i];
    double S = syntony;

    // Diffusion coefficient α(S) = φ⁻² × (1 - S) ≈ 0.382 × (1 - S)
    // This increases amplitude at higher modes (spreading energy)
    double alpha = PHI_INV_SQ_F64 * (1.0 - S);
    double d_scale = 1.0 + alpha * sqrt(norm_sq);

    // Apply differentiation scaling
    double differentiated = val * d_scale;

    // Add stochastic noise, scaled by (1 - S) for exploration/exploitation balance
    // High syntony → less noise (exploitation)
    // Low syntony → more noise (exploration)
    double noise_factor = noise_scale * (1.0 - S);

    // Mode-dependent noise: high modes get less noise (stability)
    // weight = exp(-|n|²/(φ × noise_damping))
    double noise_damping = 4.0;  // Controls how fast noise decays with mode number
    double mode_weight = exp(-norm_sq / (PHI_F64 * noise_damping));

    flux[i] = differentiated + noise_factor * mode_weight * noise[i];
}

// Single-precision version
extern "C" __global__ void resonant_d_phase_f32(
    float *flux,
    const float *lattice,
    const float *mode_norm_sq,
    const float *noise,
    float syntony,
    float noise_scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float val = lattice[i];
    float norm_sq = mode_norm_sq[i];
    float S = syntony;

    float alpha = PHI_INV_SQ_F32 * (1.0f - S);
    float d_scale = 1.0f + alpha * sqrtf(norm_sq);
    float differentiated = val * d_scale;

    float noise_factor = noise_scale * (1.0f - S);
    float noise_damping = 4.0f;
    float mode_weight = __expf(-norm_sq / (PHI_F32 * noise_damping));

    flux[i] = differentiated + noise_factor * mode_weight * noise[i];
}

// =============================================================================
// D-Phase Batch: For RES Population Evolution
// =============================================================================

// Batch D-phase for multiple tensors in RES population
// Each tensor in the population gets independent noise
extern "C" __global__ void resonant_d_phase_batch_f64(
    double *flux_batch,           // Output: pop_size × n flattened
    const double *lattice_batch,  // Input: pop_size × n flattened
    const double *mode_norm_sq,   // Shared mode norms (length n)
    const double *noise_batch,    // pop_size × n noise values
    const double *syntonies,      // Syntony per individual (length pop_size)
    double noise_scale,           // Base noise amplitude
    int n,                        // Tensor size
    int pop_size                  // Population size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * pop_size;
    if (idx >= total) return;

    int individual = idx / n;
    int mode = idx % n;

    double val = lattice_batch[idx];
    double norm_sq = mode_norm_sq[mode];
    double S = syntonies[individual];

    double alpha = PHI_INV_SQ_F64 * (1.0 - S);
    double d_scale = 1.0 + alpha * sqrt(norm_sq);
    double differentiated = val * d_scale;

    double noise_factor = noise_scale * (1.0 - S);
    double noise_damping = 4.0;
    double mode_weight = exp(-norm_sq / (PHI_F64 * noise_damping));

    flux_batch[idx] = differentiated + noise_factor * mode_weight * noise_batch[idx];
}

// =============================================================================
// Syntony Computation on Flux
// =============================================================================

// Compute syntony S(ψ) = Σ |ψ_n|² exp(-|n|²/φ) / Σ |ψ_n|²
// This computes syntony on the flux values to track D-phase output quality
extern "C" __global__ void resonant_compute_syntony_f64(
    double *numerator,           // Output: Σ |ψ|² × w(n)
    double *denominator,         // Output: Σ |ψ|²
    const double *flux,          // Input: flux values
    const double *mode_norm_sq,  // |n|² for each mode
    int n
) {
    extern __shared__ double sdata_d[];
    double *s_num = sdata_d;
    double *s_den = sdata_d + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_num = 0.0;
    double local_den = 0.0;

    if (i < n) {
        double val = flux[i];
        double amp_sq = val * val;
        double weight = exp(-mode_norm_sq[i] * PHI_INV_F64);

        local_num = amp_sq * weight;
        local_den = amp_sq;
    }

    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic add for double precision
        unsigned long long int* addr_num = (unsigned long long int*)numerator;
        unsigned long long int* addr_den = (unsigned long long int*)denominator;
        unsigned long long int old, assumed;

        old = *addr_num;
        do {
            assumed = old;
            old = atomicCAS(addr_num, assumed,
                __double_as_longlong(s_num[0] + __longlong_as_double(assumed)));
        } while (assumed != old);

        old = *addr_den;
        do {
            assumed = old;
            old = atomicCAS(addr_den, assumed,
                __double_as_longlong(s_den[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// Single-precision syntony computation
extern "C" __global__ void resonant_compute_syntony_f32(
    float *numerator,
    float *denominator,
    const float *flux,
    const float *mode_norm_sq,
    int n
) {
    extern __shared__ float sdata[];
    float *s_num = sdata;
    float *s_den = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_num = 0.0f;
    float local_den = 0.0f;

    if (i < n) {
        float val = flux[i];
        float amp_sq = val * val;
        float weight = __expf(-mode_norm_sq[i] * PHI_INV_F32);

        local_num = amp_sq * weight;
        local_den = amp_sq;
    }

    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(numerator, s_num[0]);
        atomicAdd(denominator, s_den[0]);
    }
}

// =============================================================================
// Gradient Computation for Snap Direction
// =============================================================================

// Compute gradient of lattice snap residuals
// This helps identify which modes should receive more exploration
// gradient[i] = |flux[i] - nearest_lattice[i]| / mode_weight[i]
extern "C" __global__ void resonant_snap_gradient_f64(
    double *gradient,             // Output: gradient per mode
    const double *flux,           // Current flux values
    const double *lattice,        // Nearest lattice approximation
    const double *mode_norm_sq,   // Mode norms
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double residual = flux[i] - lattice[i];
    double mode_weight = exp(-mode_norm_sq[i] * PHI_INV_F64);

    // Gradient normalized by mode weight
    // High modes (small weight) contribute more to gradient
    double eps = 1e-10;
    gradient[i] = fabs(residual) / (mode_weight + eps);
}

// =============================================================================
// Resonant Selection: Pick Best from Population
// =============================================================================

// Find index of maximum syntony in batch (for RES selection)
// Returns the index and syntony value of the best individual
extern "C" __global__ void resonant_argmax_syntony_f64(
    int *best_idx,                // Output: index of best
    double *best_syntony,         // Output: best syntony value
    const double *syntonies,      // Input: syntony per individual
    int pop_size
) {
    extern __shared__ double sdata_max[];
    int *s_idx = (int*)(sdata_max + blockDim.x);

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_max = -1e30;
    int local_idx = 0;

    if (i < pop_size) {
        local_max = syntonies[i];
        local_idx = i;
    }

    sdata_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Argmax reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata_max[tid + s] > sdata_max[tid]) {
                sdata_max[tid] = sdata_max[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && blockIdx.x == 0) {
        *best_idx = s_idx[0];
        *best_syntony = sdata_max[0];
    }
}

// =============================================================================
// Noise Generation Helpers
// =============================================================================

// Apply Box-Muller transform to uniform random numbers to get Gaussian
// Input: pairs of uniform random numbers in [0, 1)
// Output: Gaussian random numbers with mean 0, std 1
extern "C" __global__ void resonant_box_muller_f64(
    double *gaussian,            // Output: Gaussian noise
    const double *uniform,       // Input: uniform random pairs (2n values)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Box-Muller transform
    double u1 = uniform[2*i];
    double u2 = uniform[2*i + 1];

    // Avoid log(0)
    double eps = 1e-10;
    u1 = fmax(eps, u1);

    double r = sqrt(-2.0 * log(u1));
    double theta = TWO_PI_F64 * u2;

    gaussian[i] = r * cos(theta);
}

// Single-precision version
extern "C" __global__ void resonant_box_muller_f32(
    float *gaussian,
    const float *uniform,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float u1 = uniform[2*i];
    float u2 = uniform[2*i + 1];

    float eps = 1e-10f;
    u1 = fmaxf(eps, u1);

    float r = sqrtf(-2.0f * __logf(u1));
    float theta = TWO_PI_F32 * u2;

    gaussian[i] = r * __cosf(theta);
}

// =============================================================================
// Modulated Noise Based on Lattice Residual
// =============================================================================

// Generate noise modulated by how far each value is from its lattice point
// High residual → more noise (encourage exploration where snap is poor)
extern "C" __global__ void resonant_residual_modulated_noise_f64(
    double *noise,                // Output: modulated noise
    const double *base_noise,     // Input: base Gaussian noise
    const double *residuals,      // Input: snap residuals
    double min_scale,             // Minimum noise scale
    double max_scale,             // Maximum noise scale
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Scale noise by residual: larger residual → more noise
    double residual = fabs(residuals[i]);

    // Map residual to noise scale with saturation
    // sigmoid-like mapping: scale = min + (max - min) * tanh(residual)
    double scale = min_scale + (max_scale - min_scale) * tanh(residual);

    noise[i] = scale * base_noise[i];
}

// =============================================================================
// Snap Gradient Computation
// =============================================================================

// Compute gradient from pre-snap to post-snap values
// gradient[i] = (lattice[i] - flux[i]) * exp(-mode_norm_sq[i] / PHI)
// Weighted by golden ratio decay to emphasize low-frequency corrections
extern "C" __global__ void resonant_weighted_snap_gradient_f64(
    double *gradient,            // Output: snap gradient
    const double *flux,          // Input: pre-snap values
    const double *lattice,       // Input: post-snap lattice values
    const double *mode_norm_sq,  // Input: |n|² for golden weighting
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Compute basic gradient: post - pre
    double basic_gradient = lattice[i] - flux[i];

    // Apply golden weighting: emphasize corrections for low-frequency modes
    // Weight = exp(-|n|² / φ) - higher weight for fundamental modes
    double weight = exp(-mode_norm_sq[i] / PHI_F64);

    gradient[i] = basic_gradient * weight;
}

// Compute ideal dwell times based on syntony
// H-phase should take φ× longer than D-phase when syntony is high
// dwell_time = base_time × φ^syntony
extern "C" __global__ void resonant_compute_dwell_f64(
    double *h_dwell,              // Output: H-phase dwell time
    double *d_dwell,              // Output: D-phase dwell time
    double base_time,             // Base time unit (microseconds)
    double syntony
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // φ-dwell: H takes φ × D when S=1, equal when S=0
        // D-phase dwell = base
        // H-phase dwell = base × φ^S
        *d_dwell = base_time;
        *h_dwell = base_time * pow(PHI_F64, syntony);
    }
}
