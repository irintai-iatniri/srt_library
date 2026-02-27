// Syntonic CUDA Kernels - Correction Factor Operations
// (1 ± q/N) corrections for Standard Model calculations

#include "srt_constants.cuh"

// =============================================================================
// Structure Dimension Constants (N values)
// =============================================================================

// Index mapping for structure dimensions
#define STRUCT_E8_DIM       0   // 248 - dim(E₈)
#define STRUCT_E8_ROOTS     1   // 240 - |Φ(E₈)|
#define STRUCT_E8_POS       2   // 120 - |Φ⁺(E₈)| chiral
#define STRUCT_E6_DIM       3   // 78  - dim(E₆)
#define STRUCT_E6_CONE      4   // 36  - |Φ⁺(E₆)| Golden Cone
#define STRUCT_E6_27        5   // 27  - dim(27_E₆)
#define STRUCT_D4_KISSING   6   // 24  - K(D₄) consciousness threshold
#define STRUCT_G2_DIM       7   // 14  - dim(G₂)
#define NUM_STRUCTURES      8

// Structure dimensions in constant memory
__constant__ int c_structure_N[NUM_STRUCTURES] = {
    248,  // E₈ dimension
    240,  // E₈ roots
    120,  // E₈ positive roots (chiral)
    78,   // E₆ dimension
    36,   // E₆ golden cone
    27,   // E₆ 27-representation
    24,   // D₄ kissing number
    14    // G₂ dimension
};

// =============================================================================
// Basic Correction Factor Operations
// =============================================================================

// Correction factor: (1 + sign * q / N)
__device__ __forceinline__ float correction_factor_f32(int structure_idx, int sign) {
    float N = (float)c_structure_N[structure_idx];
    return 1.0f + sign * Q_DEFICIT_F32 / N;
}

__device__ __forceinline__ double correction_factor_f64(int structure_idx, int sign) {
    double N = (double)c_structure_N[structure_idx];
    return 1.0 + sign * Q_DEFICIT_F64 / N;
}

// =============================================================================
// Apply Corrections to Values
// =============================================================================

// Apply correction factor: out = in × (1 ± q/N)
extern "C" __global__ void apply_correction_f64(
    double *out,
    const double *in,
    int structure_idx,
    int sign,                    // +1 or -1
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor = correction_factor_f64(structure_idx, sign);
    out[i] = in[i] * factor;
}

extern "C" __global__ void apply_correction_f32(
    float *out,
    const float *in,
    int structure_idx,
    int sign,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float factor = correction_factor_f32(structure_idx, sign);
    out[i] = in[i] * factor;
}

// =============================================================================
// Batch Corrections with Per-Element Structure Assignment
// =============================================================================

// Apply different corrections to each element
extern "C" __global__ void apply_corrections_batch_f64(
    double *values,              // In/out values
    const int *structure_indices,// Structure index per value
    const int *signs,            // Sign per value (+1 or -1)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int struct_idx = structure_indices[i];
    int sign = signs[i];

    double factor = correction_factor_f64(struct_idx, sign);
    values[i] *= factor;
}

extern "C" __global__ void apply_corrections_batch_f32(
    float *values,
    const int *structure_indices,
    const int *signs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int struct_idx = structure_indices[i];
    int sign = signs[i];

    float factor = correction_factor_f32(struct_idx, sign);
    values[i] *= factor;
}

// =============================================================================
// Fine Structure Constant Corrections
// =============================================================================

// α = α₀ × (1 + q/N)(1 - q/N') type compound corrections
extern "C" __global__ void compound_correction_f64(
    double *out,
    const double *in,
    int struct_plus,             // Structure for (1 + q/N) factor
    int struct_minus,            // Structure for (1 - q/N') factor
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor_plus = correction_factor_f64(struct_plus, +1);
    double factor_minus = correction_factor_f64(struct_minus, -1);

    out[i] = in[i] * factor_plus * factor_minus;
}

extern "C" __global__ void compound_correction_f32(
    float *out,
    const float *in,
    int struct_plus,
    int struct_minus,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float factor_plus = correction_factor_f32(struct_plus, +1);
    float factor_minus = correction_factor_f32(struct_minus, -1);

    out[i] = in[i] * factor_plus * factor_minus;
}

// =============================================================================
// Mass Ratio Corrections
// =============================================================================

// Lepton mass ratios with generation corrections
// m_e/m_μ, m_μ/m_τ involve multiple structure factors
extern "C" __global__ void lepton_mass_correction_f64(
    double *out,
    const double *base_mass,
    int generation,              // 1, 2, or 3
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor = 1.0;

    // Generation-dependent corrections
    switch (generation) {
        case 1:  // Electron
            // (1 + q/248)(1 - q/120)
            factor = correction_factor_f64(STRUCT_E8_DIM, +1) *
                     correction_factor_f64(STRUCT_E8_POS, -1);
            break;
        case 2:  // Muon
            // (1 + q/78)(1 - q/36)
            factor = correction_factor_f64(STRUCT_E6_DIM, +1) *
                     correction_factor_f64(STRUCT_E6_CONE, -1);
            break;
        case 3:  // Tau
            // (1 + q/27)(1 - q/14)
            factor = correction_factor_f64(STRUCT_E6_27, +1) *
                     correction_factor_f64(STRUCT_G2_DIM, -1);
            break;
    }

    out[i] = base_mass[i] * factor;
}

// =============================================================================
// Quark Mass Corrections
// =============================================================================

// Quark mass with flavor and generation structure
extern "C" __global__ void quark_mass_correction_f64(
    double *out,
    const double *base_mass,
    int flavor,                  // 0-5 for u,d,c,s,t,b
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor = 1.0;

    // Flavor-dependent corrections based on SRT structure
    switch (flavor) {
        case 0:  // Up
            factor = correction_factor_f64(STRUCT_E8_ROOTS, +1);
            break;
        case 1:  // Down
            factor = correction_factor_f64(STRUCT_E8_ROOTS, -1);
            break;
        case 2:  // Charm
            factor = correction_factor_f64(STRUCT_E6_DIM, +1);
            break;
        case 3:  // Strange
            factor = correction_factor_f64(STRUCT_E6_DIM, -1);
            break;
        case 4:  // Top
            factor = correction_factor_f64(STRUCT_D4_KISSING, +1);
            break;
        case 5:  // Bottom
            factor = correction_factor_f64(STRUCT_D4_KISSING, -1);
            break;
    }

    out[i] = base_mass[i] * factor;
}

// =============================================================================
// Coupling Constant Corrections
// =============================================================================

// Apply SRT corrections to coupling constants
// α_em, α_weak, α_strong each have specific structure assignments
extern "C" __global__ void coupling_correction_f64(
    double *out,
    const double *base_coupling,
    int coupling_type,           // 0=em, 1=weak, 2=strong
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor = 1.0;

    switch (coupling_type) {
        case 0:  // Electromagnetic α_em
            // E₈ root structure correction
            factor = correction_factor_f64(STRUCT_E8_ROOTS, +1) *
                     correction_factor_f64(STRUCT_E6_CONE, -1);
            break;
        case 1:  // Weak α_W
            // E₆ structure correction
            factor = correction_factor_f64(STRUCT_E6_DIM, +1) *
                     correction_factor_f64(STRUCT_E6_27, -1);
            break;
        case 2:  // Strong α_s
            // Golden cone / D₄ correction
            factor = correction_factor_f64(STRUCT_E6_CONE, +1) *
                     correction_factor_f64(STRUCT_D4_KISSING, -1);
            break;
    }

    out[i] = base_coupling[i] * factor;
}

// =============================================================================
// Custom N-value Corrections
// =============================================================================

// For corrections not in the standard table, use arbitrary N
extern "C" __global__ void custom_correction_f64(
    double *out,
    const double *in,
    int N,                       // Custom structure dimension
    int sign,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double factor = 1.0 + sign * Q_DEFICIT_F64 / (double)N;
    out[i] = in[i] * factor;
}

extern "C" __global__ void custom_correction_f32(
    float *out,
    const float *in,
    int N,
    int sign,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float factor = 1.0f + sign * Q_DEFICIT_F32 / (float)N;
    out[i] = in[i] * factor;
}

// =============================================================================
// Higgs-Related Corrections
// =============================================================================

// Higgs mass involves E₆ and E₈ structures
extern "C" __global__ void higgs_mass_correction_f64(
    double *out,
    const double *base_mass,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Higgs gets corrections from E₈/E₆ symmetry breaking
    double factor = correction_factor_f64(STRUCT_E8_DIM, +1) *
                    correction_factor_f64(STRUCT_E6_DIM, -1) *
                    correction_factor_f64(STRUCT_E6_27, +1);

    out[i] = base_mass[i] * factor;
}

// =============================================================================
// CKM/PMNS Matrix Corrections
// =============================================================================

// Mixing matrix element corrections based on generation pairs
extern "C" __global__ void mixing_matrix_correction_f64(
    double *out,
    const double *base_element,
    int gen_i,                   // First generation (0-2)
    int gen_j,                   // Second generation (0-2)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Off-diagonal elements get generation-dependent corrections
    double factor = 1.0;

    if (gen_i != gen_j) {
        // Cross-generation mixing
        int min_gen = (gen_i < gen_j) ? gen_i : gen_j;
        int max_gen = (gen_i > gen_j) ? gen_i : gen_j;

        if (min_gen == 0 && max_gen == 1) {
            // 1-2 mixing: E₈/E₆
            factor = correction_factor_f64(STRUCT_E8_ROOTS, +1) *
                     correction_factor_f64(STRUCT_E6_DIM, -1);
        } else if (min_gen == 1 && max_gen == 2) {
            // 2-3 mixing: E₆/G₂
            factor = correction_factor_f64(STRUCT_E6_DIM, +1) *
                     correction_factor_f64(STRUCT_G2_DIM, -1);
        } else {
            // 1-3 mixing: E₈/G₂
            factor = correction_factor_f64(STRUCT_E8_ROOTS, +1) *
                     correction_factor_f64(STRUCT_G2_DIM, -1);
        }
    }

    out[i] = base_element[i] * factor;
}

// =============================================================================
// Correction Factor Computation Only
// =============================================================================

// Just compute correction factors (no input values)
extern "C" __global__ void compute_correction_factors_f64(
    double *factors,
    const int *structure_indices,
    const int *signs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    factors[i] = correction_factor_f64(structure_indices[i], signs[i]);
}

extern "C" __global__ void compute_correction_factors_f32(
    float *factors,
    const int *structure_indices,
    const int *signs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    factors[i] = correction_factor_f32(structure_indices[i], signs[i]);
}

// =============================================================================
// q-Deficit Value Access
// =============================================================================

// Get the q-deficit value (useful for diagnostics)
extern "C" __global__ void get_q_deficit(double *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = Q_DEFICIT_F64;
    }
}

// Get structure dimension by index
extern "C" __global__ void get_structure_dimension(int *out, int index) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && index >= 0 && index < NUM_STRUCTURES) {
        *out = c_structure_N[index];
    }
}
