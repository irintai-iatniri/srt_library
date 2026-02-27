// Syntonic CUDA Kernels - DHSR Cycle with Q32.32 Fixed-Point
// Exact, deterministic DHSR operators for zero-entropy recursion
// Critical for Syntonic Generative Codec (SGC) bit-perfect reconstruction

// Forward declarations from fixed_point.cuh (to avoid multiple definition linker errors)
typedef int64_t fp64;

#define SRT_FP_ZERO 0LL
#define SRT_FP_ONE (1LL << 32)
#define FP_PHI_INV 2654435769LL

__device__ __forceinline__ fp64 fp_add(fp64 a, fp64 b) { return a + b; }
__device__ __forceinline__ fp64 fp_sub(fp64 a, fp64 b) { return a - b; }

__device__ __forceinline__ fp64 fp_mul(fp64 a, fp64 b) {
  long long high = __mul64hi(a, b);
  unsigned long long low = (unsigned long long)a * (unsigned long long)b;
  unsigned long long result = ((unsigned long long)high << 32) | (low >> 32);
  return (fp64)result;
}

__device__ __forceinline__ fp64 fp_div(fp64 a, fp64 b) {
  if (b == 0) return 0;
  #if defined(__CUDA_ARCH__)
  __int128_t num = (__int128_t)a << 32;
  __int128_t den = (__int128_t)b;
  return (fp64)(num / den);
  #else
  return 0;
  #endif
}

// =============================================================================
// Syntony Metric (Exact)
// =============================================================================

// Compute exact syntony score using Q32.32 arithmetic
// Syntony measures coherence between state and golden measure
extern "C" __global__ void compute_syntony_fp64(
    fp64 *syntony_out,
    const fp64 *state,
    const fp64 *golden_measure,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    fp64 s = state[i];
    fp64 g = golden_measure[i];

    // Syntony = 1 - |s - g| / (|s| + |g| + ε)
    fp64 diff = (s > g) ? fp_sub(s, g) : fp_sub(g, s);  // |s - g|
    fp64 s_abs = (s < 0) ? -s : s;
    fp64 g_abs = (g < 0) ? -g : g;
    fp64 denom = fp_add(fp_add(s_abs, g_abs), SRT_FP_ONE >> 10);  // Add ε = 2^-10

    fp64 ratio = fp_div(diff, denom);
    fp64 syntony = fp_sub(SRT_FP_ONE, ratio);

    syntony_out[i] = syntony;
}

// Reduction to compute mean syntony across entire state
extern "C" __global__ void mean_syntony_fp64(
    fp64 *mean_out,
    const fp64 *syntony,
    int n
) {
    extern __shared__ fp64 sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? syntony[i] : SRT_FP_ZERO;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fp_add(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        fp64 n_fp64 = ((fp64)n) << 32;  // Convert n to Q32.32
        mean_out[blockIdx.x] = fp_div(sdata[0], n_fp64);
    }
}

// =============================================================================
// Differentiation Operator (Exact)
// =============================================================================

// Apply differentiation with exact golden-ratio coupling
// D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]
extern "C" __global__ void differentiation_fp64(
    fp64 *out,
    const fp64 *state,
    const fp64 *laplacian,
    fp64 syntony,
    fp64 alpha_0,
    fp64 zeta_0,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // αᵢ(S) = α₀ × (1 - S)
    fp64 one_minus_s = fp_sub(SRT_FP_ONE, syntony);
    fp64 alpha = fp_mul(alpha_0, one_minus_s);

    // ζ(S) = ζ₀ × φ⁻¹ × (1 - S)
    fp64 zeta = fp_mul(fp_mul(zeta_0, FP_PHI_INV), one_minus_s);

    // Golden-decay weight: wᵢ = φ⁻ⁱ
    // For mode i, weight decays by golden ratio
    fp64 mode_weight = FP_PHI_INV;  // φ⁻¹ for first mode

    // Apply differentiation: Ψ + α × (mode coupling) + ζ × ∇²Ψ
    fp64 projection = fp_mul(state[i], mode_weight);
    fp64 diff_term = fp_mul(alpha, projection);
    fp64 lapl_term = fp_mul(zeta, laplacian[i]);

    out[i] = fp_add(fp_add(state[i], diff_term), lapl_term);
}

// =============================================================================
// Harmonization Operator (Exact)
// =============================================================================

// Apply harmonization toward golden measure
// Ĥ[Ψ] = (1 - β) × Ψ + β × G where G is golden measure
extern "C" __global__ void harmonization_fp64(
    fp64 *out,
    const fp64 *state,
    const fp64 *golden_measure,
    fp64 beta,
    fp64 syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // β(S) = β₀ × (1 - S)  (harmonize more when syntony is low)
    fp64 one_minus_s = fp_sub(SRT_FP_ONE, syntony);
    fp64 beta_eff = fp_mul(beta, one_minus_s);

    // Complementary weight
    fp64 one_minus_beta = fp_sub(SRT_FP_ONE, beta_eff);

    // Weighted blend toward golden measure
    fp64 state_term = fp_mul(one_minus_beta, state[i]);
    fp64 measure_term = fp_mul(beta_eff, golden_measure[i]);

    out[i] = fp_add(state_term, measure_term);
}

// =============================================================================
// Complete DHSR Cycle (Exact)
// =============================================================================

// Full D→H→S→R cycle in one kernel (optimized for throughput)
// This is the core of the Syntonic Generative Codec
extern "C" __global__ void dhsr_cycle_fp64(
    fp64 *state_out,
    fp64 *syntony_out,
    const fp64 *state_in,
    const fp64 *golden_measure,
    const fp64 *laplacian,
    fp64 alpha_0,
    fp64 beta_0,
    fp64 zeta_0,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    fp64 state = state_in[i];
    fp64 measure = golden_measure[i];

    // Step 1: Compute syntony (S)
    fp64 diff = (state > measure) ? fp_sub(state, measure) : fp_sub(measure, state);
    fp64 s_abs = (state < 0) ? -state : state;
    fp64 m_abs = (measure < 0) ? -measure : measure;
    fp64 denom = fp_add(fp_add(s_abs, m_abs), SRT_FP_ONE >> 10);
    fp64 syntony = fp_sub(SRT_FP_ONE, fp_div(diff, denom));

    // Step 2: Differentiation (D)
    fp64 one_minus_s = fp_sub(SRT_FP_ONE, syntony);
    fp64 alpha = fp_mul(alpha_0, one_minus_s);
    fp64 zeta = fp_mul(fp_mul(zeta_0, FP_PHI_INV), one_minus_s);

    fp64 projection = fp_mul(state, FP_PHI_INV);
    fp64 diff_result = fp_add(fp_add(state, fp_mul(alpha, projection)),
                               fp_mul(zeta, laplacian[i]));

    // Step 3: Harmonization (H)
    fp64 beta_eff = fp_mul(beta_0, one_minus_s);
    fp64 one_minus_beta = fp_sub(SRT_FP_ONE, beta_eff);

    fp64 harm_result = fp_add(fp_mul(one_minus_beta, diff_result),
                              fp_mul(beta_eff, measure));

    // Step 4: Write outputs
    state_out[i] = harm_result;
    syntony_out[i] = syntony;
}

// =============================================================================
// Laplacian (1D Discrete)
// =============================================================================

// Compute discrete Laplacian for 1D state: ∇²Ψ[i] = Ψ[i-1] - 2Ψ[i] + Ψ[i+1]
extern "C" __global__ void laplacian_1d_fp64(
    fp64 *laplacian_out,
    const fp64 *state,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    fp64 center = state[i];
    fp64 left = (i > 0) ? state[i - 1] : center;  // Neumann BC
    fp64 right = (i < n - 1) ? state[i + 1] : center;

    // ∇²Ψ = Ψ[i-1] - 2Ψ[i] + Ψ[i+1]
    fp64 twice_center = fp_add(center, center);
    laplacian_out[i] = fp_sub(fp_add(left, right), twice_center);
}

// 2D Laplacian (for image-like states)
extern "C" __global__ void laplacian_2d_fp64(
    fp64 *laplacian_out,
    const fp64 *state,
    int height,
    int width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    int idx = row * width + col;
    fp64 center = state[idx];

    // Neighboring pixels (with Neumann BC)
    fp64 up = (row > 0) ? state[(row - 1) * width + col] : center;
    fp64 down = (row < height - 1) ? state[(row + 1) * width + col] : center;
    fp64 left = (col > 0) ? state[row * width + (col - 1)] : center;
    fp64 right = (col < width - 1) ? state[row * width + (col + 1)] : center;

    // ∇²Ψ = Ψ_up + Ψ_down + Ψ_left + Ψ_right - 4×Ψ_center
    fp64 sum_neighbors = fp_add(fp_add(up, down), fp_add(left, right));
    fp64 four_center = fp_add(fp_add(center, center), fp_add(center, center));

    laplacian_out[idx] = fp_sub(sum_neighbors, four_center);
}
