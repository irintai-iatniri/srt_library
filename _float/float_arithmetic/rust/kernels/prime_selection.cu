/**
 * Prime Selection Kernels for SRT Physics
 *
 * GPU-accelerated computation of:
 * - Fermat numbers and primality
 * - Mersenne numbers and Lucas-Lehmer test
 * - Lucas sequence and shadow phases
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================

__constant__ double PHI = 1.6180339887498948;
__constant__ double PHI_CONJUGATE = -0.6180339887498948;  // 1 - φ
__constant__ double Q = 0.027395146920;  // Syntony deficit

// ============================================================================
// Fermat Prime Kernels
// ============================================================================

/**
 * Compute Fermat numbers F_n = 2^(2^n) + 1 for array of n values
 */
extern "C" __global__ void fermat_numbers_kernel(
    const int* n_values,
    unsigned long long* results,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int n = n_values[idx];
        if (n <= 5) {
            unsigned long long exp = 1ULL << n;  // 2^n
            results[idx] = (1ULL << exp) + 1;    // 2^(2^n) + 1
        } else {
            results[idx] = 0;  // Overflow
        }
    }
}

/**
 * Check Fermat primality (batch operation)
 * F_0..F_4 are prime, F_5+ composite
 */
extern "C" __global__ void is_fermat_prime_kernel(
    const int* n_values,
    int* results,  // 1 = prime, 0 = composite
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = (n_values[idx] <= 4) ? 1 : 0;
    }
}

// ============================================================================
// Mersenne Prime Kernels
// ============================================================================

/**
 * Compute Mersenne numbers M_p = 2^p - 1
 */
extern "C" __global__ void mersenne_numbers_kernel(
    const int* p_values,
    unsigned long long* results,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int p = p_values[idx];
        if (p < 64) {
            results[idx] = (1ULL << p) - 1;
        } else {
            results[idx] = 0;  // Overflow
        }
    }
}

/**
 * Lucas-Lehmer primality test for Mersenne numbers
 * M_p is prime iff s_{p-2} ≡ 0 (mod M_p)
 * where s_0 = 4, s_{i+1} = s_i² - 2
 */
extern "C" __global__ void lucas_lehmer_kernel(
    const int* p_values,
    int* is_prime,  // 1 = prime, 0 = composite
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int p = p_values[idx];

        if (p == 2) {
            is_prime[idx] = 1;  // M_2 = 3 is prime
            return;
        }
        if (p < 2 || p >= 32) {  // Limit for 64-bit arithmetic
            is_prime[idx] = 0;
            return;
        }

        unsigned long long mp = (1ULL << p) - 1;
        unsigned long long s = 4;

        for (int i = 0; i < p - 2; i++) {
            // s = (s * s - 2) mod mp
            // Use 128-bit intermediate to avoid overflow
            unsigned __int128 s2 = (unsigned __int128)s * s;
            s = (unsigned long long)((s2 - 2) % mp);
        }

        is_prime[idx] = (s == 0) ? 1 : 0;
    }
}

// ============================================================================
// Lucas Sequence Kernels
// ============================================================================

/**
 * Compute Lucas numbers L_n for array of n values
 */
extern "C" __global__ void lucas_numbers_kernel(
    const int* n_values,
    unsigned long long* results,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int n = n_values[idx];

        if (n == 0) {
            results[idx] = 2;
            return;
        }
        if (n == 1) {
            results[idx] = 1;
            return;
        }

        unsigned long long a = 2, b = 1;
        for (int i = 1; i < n; i++) {
            unsigned long long temp = a + b;
            a = b;
            b = temp;
        }
        results[idx] = b;
    }
}

/**
 * Compute shadow phases (1-φ)^n for array of n values
 */
extern "C" __global__ void shadow_phase_kernel(
    const int* n_values,
    double* results,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int n = n_values[idx];
        double result = 1.0;
        double base = PHI_CONJUGATE;

        int exp = (n < 0) ? -n : n;
        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }

        results[idx] = (n < 0) ? 1.0 / result : result;
    }
}

/**
 * Compute Lucas boost ratios L_n1 / L_n2 with convergence metrics
 * Used for dark matter mass prediction and golden ratio verification
 * 
 * Outputs:
 *   boost_ratios[idx] = L_n1 / L_n2
 *   convergence[idx] = L_n1 / L_{n1-1} (approaches φ as n1 → ∞)
 */
extern "C" __global__ void lucas_boost_kernel(
    const int* n1_values,
    const int* n2_values,
    double* boost_ratios,
    double* convergence,  // Optional: can be NULL
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Compute L_{n1} and L_{n1-1}
        int n1 = n1_values[idx];
        unsigned long long l1 = 2;
        unsigned long long l1_prev = 1;  // L_{n1-1} for convergence ratio
        
        if (n1 == 0) {
            l1 = 2;
            l1_prev = 1;  // L_{-1} = 1 by convention
        } else if (n1 == 1) {
            l1 = 1;
            l1_prev = 2;  // L_0 = 2
        } else {
            unsigned long long a = 2, b = 1;
            for (int i = 1; i < n1; i++) {
                unsigned long long temp = a + b;
                a = b;
                b = temp;
            }
            l1 = b;
            l1_prev = a;  // The previous value in sequence
        }

        // Compute L_{n2}
        int n2 = n2_values[idx];
        unsigned long long l2 = 2;
        if (n2 == 0) l2 = 2;
        else if (n2 == 1) l2 = 1;
        else {
            unsigned long long a = 2, b = 1;
            for (int i = 1; i < n2; i++) {
                unsigned long long temp = a + b;
                a = b;
                b = temp;
            }
            l2 = b;
        }

        // Output boost ratio: L_n1 / L_n2
        boost_ratios[idx] = (double)l1 / (double)l2;
        
        // Output convergence to golden ratio: L_n / L_{n-1} → φ
        if (convergence != NULL && l1_prev > 0) {
            convergence[idx] = (double)l1 / (double)l1_prev;
        }
    }
}

/**
 * Extended Lucas computation with full sequence information
 * 
 * For each n, computes:
 *   - L_n (Lucas number)
 *   - L_{n-1} (previous Lucas number)
 *   - L_n / L_{n-1} (ratio approaching φ)
 *   - |L_n/L_{n-1} - φ| (error from golden ratio)
 *   - Shadow phase (1-φ)^n
 */
extern "C" __global__ void lucas_extended_kernel(
    const int* n_values,
    unsigned long long* lucas_n,       // L_n
    unsigned long long* lucas_n_prev,  // L_{n-1}
    double* ratio_to_phi,              // L_n / L_{n-1}
    double* phi_error,                 // |ratio - φ|
    double* shadow_phase,              // (1-φ)^n
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int n = n_values[idx];
        
        unsigned long long l_curr = 2;
        unsigned long long l_prev = 1;
        
        if (n == 0) {
            l_curr = 2;
            l_prev = 1;
        } else if (n == 1) {
            l_curr = 1;
            l_prev = 2;
        } else {
            unsigned long long a = 2, b = 1;
            for (int i = 1; i < n; i++) {
                unsigned long long temp = a + b;
                a = b;
                b = temp;
            }
            l_curr = b;
            l_prev = a;
        }
        
        // Store L_n and L_{n-1}
        lucas_n[idx] = l_curr;
        lucas_n_prev[idx] = l_prev;
        
        // Compute ratio L_n / L_{n-1}
        double ratio = (l_prev > 0) ? (double)l_curr / (double)l_prev : PHI;
        ratio_to_phi[idx] = ratio;
        
        // Compute error from golden ratio
        double error = (ratio > PHI) ? (ratio - PHI) : (PHI - ratio);
        phi_error[idx] = error;
        
        // Compute shadow phase (1-φ)^n using binary exponentiation
        double shadow = 1.0;
        double base = PHI_CONJUGATE;
        int exp = (n < 0) ? -n : n;
        
        while (exp > 0) {
            if (exp & 1) shadow *= base;
            base *= base;
            exp >>= 1;
        }
        
        shadow_phase[idx] = (n < 0) ? 1.0 / shadow : shadow;
    }
}

/**
 * Compute Fibonacci and Lucas numbers together (they share recurrence)
 * F_n and L_n satisfy: L_n = F_{n-1} + F_{n+1}
 */
extern "C" __global__ void fibonacci_lucas_kernel(
    const int* n_values,
    unsigned long long* fibonacci,  // F_n
    unsigned long long* lucas,      // L_n
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int n = n_values[idx];
        
        if (n == 0) {
            fibonacci[idx] = 0;
            lucas[idx] = 2;
        } else if (n == 1) {
            fibonacci[idx] = 1;
            lucas[idx] = 1;
        } else {
            // Compute both sequences in parallel
            unsigned long long f_prev = 0, f_curr = 1;
            unsigned long long l_prev = 2, l_curr = 1;
            
            for (int i = 1; i < n; i++) {
                unsigned long long f_temp = f_prev + f_curr;
                f_prev = f_curr;
                f_curr = f_temp;
                
                unsigned long long l_temp = l_prev + l_curr;
                l_prev = l_curr;
                l_curr = l_temp;
            }
            
            fibonacci[idx] = f_curr;
            lucas[idx] = l_curr;
        }
    }
}

// ============================================================================
// Correction Factor Kernels
// ============================================================================

/**
 * Apply extended hierarchy corrections (batch operation)
 * correction_type: 0=q/divisor, 1=q²/divisor, 2=qφ/divisor, etc.
 */
extern "C" __global__ void apply_hierarchy_correction_kernel(
    const double* values,
    double* results,
    int divisor,
    int correction_type,  // 0: q/d, 1: q²/d, 2: qφ/d, 3: q/φd
    int sign,             // +1 or -1
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double factor;
        switch (correction_type) {
            case 0:  // q / divisor
                factor = Q / (double)divisor;
                break;
            case 1:  // q² / divisor
                factor = Q * Q / (double)divisor;
                break;
            case 2:  // q·φ / divisor
                factor = Q * PHI / (double)divisor;
                break;
            case 3:  // q / (φ·divisor)
                factor = Q / (PHI * (double)divisor);
                break;
            default:
                factor = Q / (double)divisor;
        }

        results[idx] = values[idx] * (1.0 + sign * factor);
    }
}

/**
 * Apply multiplicative suppression factor 1/(1 + q·φ^power)
 */
extern "C" __global__ void apply_suppression_kernel(
    const double* values,
    double* results,
    int phi_power,  // -2, -1, 0, 1, 2, 3, etc.
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double phi_factor = 1.0;
        int exp = (phi_power < 0) ? -phi_power : phi_power;

        for (int i = 0; i < exp; i++) {
            phi_factor *= PHI;
        }

        if (phi_power < 0) {
            phi_factor = 1.0 / phi_factor;
        }

        results[idx] = values[idx] / (1.0 + Q * phi_factor);
    }
}

// ============================================================================
// Pisano Period Kernels (Hook Length Computation)
// ============================================================================

/**
 * Compute Pisano periods for batch of moduli n
 * π(n): period of Fibonacci sequence mod n
 * Used for determining "Hook Length" in stable winding modes
 */
extern "C" __global__ void pisano_period_kernel(
    const unsigned long long* n_values,
    unsigned long long* periods,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long n = n_values[idx];
        if (n == 0) {
            periods[idx] = 0;
            return;
        }
        if (n == 1) {
            periods[idx] = 1;
            return;
        }

        unsigned long long a = 0, b = 1;
        unsigned long long period = 0;

        // Pisano period is bounded by 6n (Wall's theorem)
        for (unsigned long long i = 0; i < 6 * n; i++) {
            unsigned long long c = (a + b) % n;
            a = b;
            b = c;
            if (a == 0 && b == 1) {
                period = i + 1;
                break;
            }
        }

        periods[idx] = period;
    }
}

// ============================================================================
// Fibonacci Transcendence Gate Kernels
// ============================================================================

/**
 * Compute Fibonacci resonance boost factors
 * Returns φ^n for transcendence gates, 1.0 otherwise
 * Special case: n=4 (Material Anomaly) gets 0.9× destabilization
 */
extern "C" __global__ void fibonacci_resonance_boost_kernel(
    const unsigned long long* n_values,
    double* boosts,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long n = n_values[idx];

        // Fibonacci Prime indices (transcendence gates)
        const unsigned long long FIB_PRIME_INDICES[] = {3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47};
        bool is_gate = false;

        for (int i = 0; i < 11; i++) {
            if (n == FIB_PRIME_INDICES[i]) {
                is_gate = true;
                break;
            }
        }

        if (!is_gate) {
            boosts[idx] = 1.0;
            return;
        }

        // Compute φ^n using binary exponentiation
        double result = 1.0;
        double base = PHI;
        unsigned long long exp = n;

        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }

        // Material Anomaly: n=4 produces prime from composite seed
        if (n == 4) {
            result *= 0.9;  // 10% destabilization factor
        }

        boosts[idx] = result;
    }
}

// ============================================================================
// Lucas Gap Kernels (Dark Energy Computation)
// ============================================================================

/**
 * Compute Lucas gap pressure (dark energy contribution)
 * In gaps: delocalized shadow energy → repulsive pressure
 * At primes: crystallized → no pressure
 */
extern "C" __global__ void lucas_gap_pressure_kernel(
    const unsigned long long* n_values,
    double* pressures,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long n = n_values[idx];

        // Lucas Prime indices (no gaps at these points)
        const unsigned long long LUCAS_PRIMES[] = {0, 2, 4, 5, 7, 8, 11, 13, 16, 17, 19, 31, 37};
        bool is_prime = false;

        for (int i = 0; i < 13; i++) {
            if (n == LUCAS_PRIMES[i]) {
                is_prime = true;
                break;
            }
        }

        if (is_prime) {
            pressures[idx] = 0.0;  // No pressure at crystallization points
            return;
        }

        // Find nearest Lucas prime index
        unsigned long long nearest = 0;
        unsigned long long min_dist = (unsigned long long)-1;

        for (int i = 0; i < 13; i++) {
            unsigned long long dist = (n > LUCAS_PRIMES[i]) ?
                (n - LUCAS_PRIMES[i]) : (LUCAS_PRIMES[i] - n);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = LUCAS_PRIMES[i];
            }
        }

        // Gap distance (signed)
        long long gap_distance = (long long)n - (long long)nearest;

        // Shadow phase: (1-φ)^n
        double shadow_phase = 1.0;
        double base = PHI_CONJUGATE;
        unsigned long long exp = (gap_distance < 0) ? -gap_distance : gap_distance;

        while (exp > 0) {
            if (exp & 1) shadow_phase *= base;
            base *= base;
            exp >>= 1;
        }

        if (gap_distance < 0) {
            shadow_phase = 1.0 / shadow_phase;
        }

        // Pressure = gap_distance × shadow_phase
        pressures[idx] = (double)gap_distance * shadow_phase;
    }
}