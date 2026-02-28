/**
 * Phase-State Compiler CUDA Kernels
 *
 * Four kernels implementing the full Phase-State compile cycle on GPU
 * using exact int8 arithmetic in the Gaussian integer ring Z[i].
 *
 * Memory layout (Structure of Arrays for coalescing):
 *   int8_t* m4  — real component  (-1, 0, 1)
 *   int8_t* t4  — imaginary component (-1, 0, 1)
 *
 * State encoding:
 *   Differentiation: (+1, 0)    Harmonization: (-1, 0)
 *   Recursion +i:    (0, +1)    Recursion -i:  (0, -1)
 *   Syntony/Aperture: (0, 0)
 *
 * All comparisons are exact integer — no floating-point epsilon needed.
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* ========================================================================
 * Kernel 1: 2D Novelty Injection (Majority Vote)
 *
 * Fills syntonic (empty) nodes via majority vote of 4 toroidal neighbors.
 * Each thread owns exactly one index — no write collisions on m4[i]/t4[i].
 * ====================================================================== */
extern "C" __global__ void ps_inject_novelty_2d_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    int rows, int cols, int* activity_flag
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i >= n) return;

    /* Only fill the Aperture (syntonic / empty nodes) */
    if (m4[i] != 0 || t4[i] != 0) return;

    /* Toroidal 4-neighbors */
    int r = i / cols;
    int c = i % cols;

    int neighbors[4];
    neighbors[0] = ((r - 1 + rows) % rows) * cols + c;       /* Up   */
    neighbors[1] = ((r + 1) % rows) * cols + c;              /* Down */
    neighbors[2] = r * cols + ((c - 1 + cols) % cols);       /* Left */
    neighbors[3] = r * cols + ((c + 1) % cols);              /* Right */

    /* Gather majority vote from active neighbors */
    int vote = 0;
    for (int k = 0; k < 4; k++) {
        int nb = neighbors[k];
        if (m4[nb] != 0 || t4[nb] != 0) {
            vote += m4[nb];
        }
    }

    /* Inject if polarity consensus exists */
    if (vote != 0) {
        int8_t new_val = (vote > 0) ? 1 : -1;
        m4[i] = new_val;
        t4[i] = 0;
        interacted[i] = 1;
        atomicExch(activity_flag, 1);
    }
}


/* ========================================================================
 * Kernel 2: 2D Toroidal Wavefront Propagation
 *
 * Active (non-syntonic) nodes fill empty neighbors via atomicCAS locking.
 * Prevents write collisions when multiple active nodes share a neighbor.
 * ====================================================================== */
extern "C" __global__ void ps_wavefront_propagate_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    int rows, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i >= n) return;

    /* Only active (non-syntonic) nodes that haven't interacted propagate */
    int my_m4 = m4[i];
    int my_t4 = t4[i];
    if ((my_m4 == 0 && my_t4 == 0) || interacted[i] == 1) return;

    /* Toroidal 4-neighbors */
    int r = i / cols;
    int c = i % cols;

    int neighbors[4];
    neighbors[0] = ((r - 1 + rows) % rows) * cols + c;
    neighbors[1] = ((r + 1) % rows) * cols + c;
    neighbors[2] = r * cols + ((c - 1 + cols) % cols);
    neighbors[3] = r * cols + ((c + 1) % cols);

    bool propagated = false;

    for (int k = 0; k < 4; k++) {
        int nb = neighbors[k];

        /* Check if neighbor is the Aperture (syntonic) */
        if (m4[nb] == 0 && t4[nb] == 0) {
            /* Atomic lock: first thread to change 0->1 wins */
            if (atomicCAS(&interacted[nb], 0, 1) == 0) {
                m4[nb] = (int8_t)my_m4;
                t4[nb] = (int8_t)my_t4;
                propagated = true;
            }
        }
    }

    if (propagated) {
        atomicExch(&interacted[i], 1);
    }
}


/* ========================================================================
 * Kernel 3: Global Harmonization (Destructive Interference)
 *
 * O(N) per thread scan for exact Z[i] cancellation:
 *   A.re + B.re == 0  &&  A.im + B.im == 0
 * Uses atomicCAS to prevent race conditions on paired interactions.
 * ====================================================================== */
extern "C" __global__ void ps_harmonize_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    const int8_t* is_source, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* Skip syntonic, already-interacted, or source nodes */
    if ((m4[i] == 0 && t4[i] == 0) || interacted[i] == 1) return;
    if (is_source[i] == 1) return;

    for (int j = i + 1; j < n; j++) {
        if ((m4[j] == 0 && t4[j] == 0) || interacted[j] == 1) continue;
        if (is_source[j] == 1) continue;

        /* Exact destructive interference check */
        if (m4[i] + m4[j] == 0 && t4[i] + t4[j] == 0) {
            /* Atomic CAS to prevent race conditions */
            if (atomicCAS(&interacted[i], 0, 1) == 0) {
                if (atomicCAS(&interacted[j], 0, 1) == 0) {
                    /* Harmonization achieved -> collapse to Aperture */
                    m4[i] = 0; t4[i] = 0;
                    m4[j] = 0; t4[j] = 0;
                    break;
                } else {
                    /* Rollback if node j was snatched by another thread */
                    interacted[i] = 0;
                }
            }
            break;
        }
    }
}


/* ========================================================================
 * Kernel 4: Dampened Recursion (Orthogonal Phase Shift)
 *
 * Nodes that are "stuck" (no interaction for stale_threshold cycles)
 * undergo multiplication by i: (a + bi) * i = -b + ai
 * Signals gnosis when recursive_depth reaches K threshold.
 * ====================================================================== */
extern "C" __global__ void ps_dampened_recursion_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    const int8_t* is_source, int* stale_cycles, int* recursive_depth,
    int stale_threshold, int k_threshold, int* gnosis_flag,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* Source nodes never recurse or go stale */
    if (is_source[i] == 1) {
        stale_cycles[i] = 0;
        return;
    }

    /* Syntonic or interacted -> reset staleness */
    if ((m4[i] == 0 && t4[i] == 0) || interacted[i] == 1) {
        stale_cycles[i] = 0;
        return;
    }

    /* Active but did not interact -> increment staleness */
    stale_cycles[i] += 1;

    /* Trigger orthogonal phase shift if threshold reached */
    if (stale_cycles[i] >= stale_threshold) {
        /* R: Multiplication by i. (a + bi) * i = -b + ai */
        int8_t old_m4 = m4[i];
        int8_t old_t4 = t4[i];

        m4[i] = -old_t4;
        t4[i] = old_m4;

        stale_cycles[i] = 0;
        recursive_depth[i] += 1;

        /* Check for Gnosis phase transition (K=24) */
        if (recursive_depth[i] >= k_threshold) {
            atomicExch(gnosis_flag, 1);
        }
    }
}
