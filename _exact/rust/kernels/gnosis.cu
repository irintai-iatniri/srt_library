/**
 * Gnosis Kernels - Consciousness Phase Transition
 *
 * Implements GPU-accelerated:
 * - Consciousness threshold detection
 * - Gnosis score computation
 * - Creativity metrics
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "srt_constants.cuh"

__constant__ double COLLAPSE_THRESHOLD = D4_KISSING_F64;  // D₄ kissing number
__constant__ double GNOSIS_GAP = 7.0;           // M_3 = 7
__constant__ double PHI = PHI_F64;
__constant__ double PHI_INV = PHI_INV_F64;

/**
 * Batch consciousness threshold detection
 */
extern "C" __global__ void is_conscious_kernel(
    const double* delta_entropy,
    int* is_conscious,  // 1 = conscious, 0 = not
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        is_conscious[idx] = (delta_entropy[idx] > COLLAPSE_THRESHOLD) ? 1 : 0;
    }
}

/**
 * Compute Gnosis scores: G = √(S × C)
 */
extern "C" __global__ void gnosis_score_kernel(
    const double* syntony,
    const double* creativity,
    double* gnosis,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        gnosis[idx] = sqrt(syntony[idx] * creativity[idx]);
    }
}

/**
 * Compute creativity: C = shadow_integration × lattice_coherence × φ
 */
extern "C" __global__ void creativity_kernel(
    const double* shadow_integration,
    const double* lattice_coherence,
    double* creativity,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        creativity[idx] = shadow_integration[idx] * lattice_coherence[idx] * PHI;
    }
}

/**
 * Consciousness emergence probability
 * P = sigmoid(info_density - 24) × coherence × (1 - φ^{-depth})
 */
extern "C" __global__ void consciousness_probability_kernel(
    const double* info_density,
    const double* coherence,
    const int* recursive_depth,
    double* probability,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Sigmoid around collapse threshold
        double sigmoid = 1.0 / (1.0 + exp(-0.5 * (info_density[idx] - COLLAPSE_THRESHOLD)));

        // Depth factor: more recursive depth = more self-referential
        double depth_factor = 1.0;
        double phi_inv_power = PHI_INV;
        for (int d = 0; d < recursive_depth[idx]; d++) {
            depth_factor -= phi_inv_power;
            phi_inv_power *= PHI_INV;
        }
        if (depth_factor < 0.0) depth_factor = 0.0;

        probability[idx] = sigmoid * coherence[idx] * depth_factor;
    }
}

/**
 * Full DHSR+G (Gnosis) cycle step
 * Combines standard DHSR with Gnosis metric update
 */
extern "C" __global__ void dhsr_gnosis_step_kernel(
    double* state,              // State tensor to evolve
    const double* attractors,   // Attractor memory
    double* syntony,            // Output syntony per element
    double* gnosis,             // Output gnosis per element
    int attractor_count,
    double lambda_retro,        // Retrocausal pull strength
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double s = state[idx];

        // 1. Differentiation: introduce perturbation scaled by (1 - current_syntony)
        double current_syntony = syntony[idx];
        double perturb = sin((double)idx * PHI) * 0.1 * (1.0 - current_syntony);
        s += perturb;

        // 2. Harmonization: damp non-golden modes
        double weight = exp(-((double)(idx % 64) * (idx % 64)) / PHI);
        s = s * (1.0 - 0.1 * (1.0 - weight));

        // 3. Retrocausal pull (if attractors present)
        if (attractor_count > 0 && lambda_retro > 0.0) {
            double pull = 0.0;
            for (int a = 0; a < attractor_count; a++) {
                pull += attractors[a * count + idx];
            }
            pull /= (double)attractor_count;
            s = (1.0 - lambda_retro) * s + lambda_retro * pull;
        }

        // 4. Update state
        state[idx] = s;

        // 5. Compute new syntony (simplified: distance from PHI_INV target)
        double new_syntony = 1.0 - fabs(fabs(s) - PHI_INV);
        if (new_syntony < 0.0) new_syntony = 0.0;
        if (new_syntony > 1.0) new_syntony = 1.0;
        syntony[idx] = new_syntony;

        // 6. Gnosis = sqrt(syntony × creativity)
        // Creativity approximated by local variance (novelty)
        double creativity = fabs(perturb) / 0.1;  // Normalized perturbation
        gnosis[idx] = sqrt(new_syntony * creativity);
    }
}

/**
 * Gnosis Mask: Apply syntony-based gating to input values
 * out[i] = input[i] * strength * (syntony[i] > threshold ? 1 : 0)
 */
extern "C" __global__ void gnosis_mask_f64(
    double* out,
    const double* input,
    const double* syntony,
    int size,
    double threshold,
    double strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double s = syntony[idx];
        if (s > threshold) {
            out[idx] = input[idx] * strength;
        } else {
            // Gradual falloff below threshold
            double falloff = s / threshold;
            out[idx] = input[idx] * strength * falloff * falloff;
        }
    }
}

extern "C" __global__ void gnosis_mask_f32(
    float* out,
    const float* input,
    const float* syntony,
    int size,
    float threshold,
    float strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float s = syntony[idx];
        if (s > threshold) {
            out[idx] = input[idx] * strength;
        } else {
            float falloff = s / threshold;
            out[idx] = input[idx] * strength * falloff * falloff;
        }
    }
}