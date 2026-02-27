// Syntonic CUDA Kernels - Sedenion Operations
// 16-dimensional hypercomplex number operations
// WARNING: Sedenions have zero divisors - ||a*b|| != ||a||*||b|| in general
//
// Memory layout: Each sedenion is 16 consecutive f64 values (e0, e1, ..., e15)
// Batch operations process n sedenions: total array size = 16*n

// =============================================================================
// Internal Octonion Helpers (for Cayley-Dickson multiplication)
// =============================================================================

// Octonion multiplication using Fano plane structure
// a, b are 8-element arrays representing octonions
// Result is stored in out
__device__ void octonion_mul_inline(
    double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7,
    double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7,
    double *r0, double *r1, double *r2, double *r3, double *r4, double *r5, double *r6, double *r7
) {
    // e0 (real part)
    *r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;

    // e1
    *r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;

    // e2
    *r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;

    // e3
    *r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;

    // e4
    *r4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;

    // e5
    *r5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;

    // e6
    *r6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;

    // e7
    *r7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
}

// =============================================================================
// Sedenion Batch Operations
// =============================================================================

/**
 * Batch sedenion multiplication using Cayley-Dickson construction
 *
 * out[i] = a[i] * b[i] for each sedenion in the batch
 *
 * Cayley-Dickson formula for sedenions (a_lo, a_hi) * (b_lo, b_hi):
 *   result_lo = a_lo * b_lo - conj(b_hi) * a_hi
 *   result_hi = b_hi * a_lo + a_hi * conj(b_lo)
 *
 * @param out Output array: n * 16 doubles
 * @param a First operand array: n * 16 doubles
 * @param b Second operand array: n * 16 doubles
 * @param n Number of sedenions to multiply
 */
extern "C" __global__ void sedenion_mul_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;

    // Load sedenion a = (a_lo, a_hi) where each is an octonion
    double a0 = a[base + 0], a1 = a[base + 1], a2 = a[base + 2], a3 = a[base + 3];
    double a4 = a[base + 4], a5 = a[base + 5], a6 = a[base + 6], a7 = a[base + 7];
    double a8 = a[base + 8], a9 = a[base + 9], a10 = a[base + 10], a11 = a[base + 11];
    double a12 = a[base + 12], a13 = a[base + 13], a14 = a[base + 14], a15 = a[base + 15];

    // Load sedenion b = (b_lo, b_hi)
    double b0 = b[base + 0], b1 = b[base + 1], b2 = b[base + 2], b3 = b[base + 3];
    double b4 = b[base + 4], b5 = b[base + 5], b6 = b[base + 6], b7 = b[base + 7];
    double b8 = b[base + 8], b9 = b[base + 9], b10 = b[base + 10], b11 = b[base + 11];
    double b12 = b[base + 12], b13 = b[base + 13], b14 = b[base + 14], b15 = b[base + 15];

    // Cayley-Dickson: (a_lo, a_hi) * (b_lo, b_hi) = (a_lo*b_lo - conj(b_hi)*a_hi, b_hi*a_lo + a_hi*conj(b_lo))

    // Step 1: Compute a_lo * b_lo
    double t0, t1, t2, t3, t4, t5, t6, t7;
    octonion_mul_inline(a0, a1, a2, a3, a4, a5, a6, a7,
                        b0, b1, b2, b3, b4, b5, b6, b7,
                        &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

    // Step 2: Compute conj(b_hi) * a_hi
    // conj(b_hi) = (b8, -b9, -b10, -b11, -b12, -b13, -b14, -b15)
    double u0, u1, u2, u3, u4, u5, u6, u7;
    octonion_mul_inline(b8, -b9, -b10, -b11, -b12, -b13, -b14, -b15,
                        a8, a9, a10, a11, a12, a13, a14, a15,
                        &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);

    // result_lo = a_lo*b_lo - conj(b_hi)*a_hi
    out[base + 0] = t0 - u0;
    out[base + 1] = t1 - u1;
    out[base + 2] = t2 - u2;
    out[base + 3] = t3 - u3;
    out[base + 4] = t4 - u4;
    out[base + 5] = t5 - u5;
    out[base + 6] = t6 - u6;
    out[base + 7] = t7 - u7;

    // Step 3: Compute b_hi * a_lo
    double v0, v1, v2, v3, v4, v5, v6, v7;
    octonion_mul_inline(b8, b9, b10, b11, b12, b13, b14, b15,
                        a0, a1, a2, a3, a4, a5, a6, a7,
                        &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);

    // Step 4: Compute a_hi * conj(b_lo)
    // conj(b_lo) = (b0, -b1, -b2, -b3, -b4, -b5, -b6, -b7)
    double w0, w1, w2, w3, w4, w5, w6, w7;
    octonion_mul_inline(a8, a9, a10, a11, a12, a13, a14, a15,
                        b0, -b1, -b2, -b3, -b4, -b5, -b6, -b7,
                        &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

    // result_hi = b_hi*a_lo + a_hi*conj(b_lo)
    out[base + 8] = v0 + w0;
    out[base + 9] = v1 + w1;
    out[base + 10] = v2 + w2;
    out[base + 11] = v3 + w3;
    out[base + 12] = v4 + w4;
    out[base + 13] = v5 + w5;
    out[base + 14] = v6 + w6;
    out[base + 15] = v7 + w7;
}

/**
 * Batch sedenion addition
 */
extern "C" __global__ void sedenion_add_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] + b[i];
}

/**
 * Batch sedenion subtraction
 */
extern "C" __global__ void sedenion_sub_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] - b[i];
}

/**
 * Batch sedenion negation
 */
extern "C" __global__ void sedenion_neg_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = -a[i];
}

/**
 * Batch sedenion scalar multiplication
 */
extern "C" __global__ void sedenion_scale_f64(double *out, const double *a, double scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] * scalar;
}

/**
 * Batch sedenion conjugation
 * conj(e0, e1, ..., e15) = (e0, -e1, -e2, ..., -e15)
 */
extern "C" __global__ void sedenion_conjugate_f64(double *out, const double *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 16) return;

    int component = idx % 16;
    if (component == 0) {
        out[idx] = a[idx];  // Real part unchanged
    } else {
        out[idx] = -a[idx]; // Imaginary parts negated
    }
}

/**
 * Batch sedenion norm squared computation
 * Output: n doubles (one per sedenion)
 */
extern "C" __global__ void sedenion_norm_sq_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        double val = a[base + j];
        sum += val * val;
    }
    out[i] = sum;
}

/**
 * Batch sedenion norm computation
 * Output: n doubles (one per sedenion)
 */
extern "C" __global__ void sedenion_norm_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        double val = a[base + j];
        sum += val * val;
    }
    out[i] = sqrt(sum);
}

/**
 * Batch sedenion normalization (scale to unit norm)
 * Handles zero-norm case by returning zero sedenion
 */
extern "C" __global__ void sedenion_normalize_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;

    // Compute norm
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        double val = a[base + j];
        sum += val * val;
    }
    double norm = sqrt(sum);

    // Normalize (or keep as zero if norm is zero)
    if (norm > 1e-15) {
        double inv_norm = 1.0 / norm;
        for (int j = 0; j < 16; j++) {
            out[base + j] = a[base + j] * inv_norm;
        }
    } else {
        for (int j = 0; j < 16; j++) {
            out[base + j] = 0.0;
        }
    }
}

/**
 * Batch sedenion inverse computation
 * inverse(s) = conj(s) / norm_sq(s)
 * WARNING: For zero divisors, this may not behave as expected!
 */
extern "C" __global__ void sedenion_inverse_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;

    // Compute norm squared
    double norm_sq = 0.0;
    for (int j = 0; j < 16; j++) {
        double val = a[base + j];
        norm_sq += val * val;
    }

    // Compute inverse = conj / norm_sq
    if (norm_sq > 1e-30) {
        double inv_norm_sq = 1.0 / norm_sq;
        out[base + 0] = a[base + 0] * inv_norm_sq;  // Real part
        for (int j = 1; j < 16; j++) {
            out[base + j] = -a[base + j] * inv_norm_sq;  // Imaginary parts negated
        }
    } else {
        // Near-zero norm: return NaN to signal invalid inverse
        for (int j = 0; j < 16; j++) {
            out[base + j] = __longlong_as_double(0x7FF8000000000000LL);  // NaN
        }
    }
}

/**
 * Batch sedenion dot product
 * Output: n doubles (one per pair)
 * dot(a, b) = sum(a[i] * b[i])
 */
extern "C" __global__ void sedenion_dot_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        sum += a[base + j] * b[base + j];
    }
    out[i] = sum;
}

/**
 * Check for zero divisor pairs in batch
 * Output: n bools (as int: 0 or 1)
 * Returns 1 if ||a*b|| < tol * ||a|| * ||b|| (indicating zero divisor pair)
 */
extern "C" __global__ void sedenion_is_zero_divisor_pair_f64(
    int *out, const double *a, const double *b, double tol, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;

    // Compute norms
    double norm_a_sq = 0.0, norm_b_sq = 0.0;
    for (int j = 0; j < 16; j++) {
        norm_a_sq += a[base + j] * a[base + j];
        norm_b_sq += b[base + j] * b[base + j];
    }
    double norm_a = sqrt(norm_a_sq);
    double norm_b = sqrt(norm_b_sq);

    // Skip if either is zero
    if (norm_a < 1e-15 || norm_b < 1e-15) {
        out[i] = 0;
        return;
    }

    // Load sedenions for multiplication
    double a0 = a[base + 0], a1 = a[base + 1], a2 = a[base + 2], a3 = a[base + 3];
    double a4 = a[base + 4], a5 = a[base + 5], a6 = a[base + 6], a7 = a[base + 7];
    double a8 = a[base + 8], a9 = a[base + 9], a10 = a[base + 10], a11 = a[base + 11];
    double a12 = a[base + 12], a13 = a[base + 13], a14 = a[base + 14], a15 = a[base + 15];

    double b0 = b[base + 0], b1 = b[base + 1], b2 = b[base + 2], b3 = b[base + 3];
    double b4 = b[base + 4], b5 = b[base + 5], b6 = b[base + 6], b7 = b[base + 7];
    double b8 = b[base + 8], b9 = b[base + 9], b10 = b[base + 10], b11 = b[base + 11];
    double b12 = b[base + 12], b13 = b[base + 13], b14 = b[base + 14], b15 = b[base + 15];

    // Compute product using Cayley-Dickson
    double t0, t1, t2, t3, t4, t5, t6, t7;
    octonion_mul_inline(a0, a1, a2, a3, a4, a5, a6, a7,
                        b0, b1, b2, b3, b4, b5, b6, b7,
                        &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

    double u0, u1, u2, u3, u4, u5, u6, u7;
    octonion_mul_inline(b8, -b9, -b10, -b11, -b12, -b13, -b14, -b15,
                        a8, a9, a10, a11, a12, a13, a14, a15,
                        &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);

    double p0 = t0 - u0, p1 = t1 - u1, p2 = t2 - u2, p3 = t3 - u3;
    double p4 = t4 - u4, p5 = t5 - u5, p6 = t6 - u6, p7 = t7 - u7;

    double v0, v1, v2, v3, v4, v5, v6, v7;
    octonion_mul_inline(b8, b9, b10, b11, b12, b13, b14, b15,
                        a0, a1, a2, a3, a4, a5, a6, a7,
                        &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);

    double w0, w1, w2, w3, w4, w5, w6, w7;
    octonion_mul_inline(a8, a9, a10, a11, a12, a13, a14, a15,
                        b0, -b1, -b2, -b3, -b4, -b5, -b6, -b7,
                        &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

    double p8 = v0 + w0, p9 = v1 + w1, p10 = v2 + w2, p11 = v3 + w3;
    double p12 = v4 + w4, p13 = v5 + w5, p14 = v6 + w6, p15 = v7 + w7;

    // Compute product norm
    double norm_prod_sq = p0*p0 + p1*p1 + p2*p2 + p3*p3 + p4*p4 + p5*p5 + p6*p6 + p7*p7 +
                          p8*p8 + p9*p9 + p10*p10 + p11*p11 + p12*p12 + p13*p13 + p14*p14 + p15*p15;
    double norm_prod = sqrt(norm_prod_sq);

    // Check if zero divisor pair
    double threshold = tol * norm_a * norm_b;
    out[i] = (norm_prod < threshold) ? 1 : 0;
}

// =============================================================================
// Single-Precision Variants (f32)
// =============================================================================

// Single precision octonion multiplication helper
__device__ void octonion_mul_inline_f32(
    float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7,
    float b0, float b1, float b2, float b3, float b4, float b5, float b6, float b7,
    float *r0, float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7
) {
    *r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
    *r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
    *r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
    *r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
    *r4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
    *r5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
    *r6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
    *r7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
}

extern "C" __global__ void sedenion_mul_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;

    float a0 = a[base + 0], a1 = a[base + 1], a2 = a[base + 2], a3 = a[base + 3];
    float a4 = a[base + 4], a5 = a[base + 5], a6 = a[base + 6], a7 = a[base + 7];
    float a8 = a[base + 8], a9 = a[base + 9], a10 = a[base + 10], a11 = a[base + 11];
    float a12 = a[base + 12], a13 = a[base + 13], a14 = a[base + 14], a15 = a[base + 15];

    float b0 = b[base + 0], b1 = b[base + 1], b2 = b[base + 2], b3 = b[base + 3];
    float b4 = b[base + 4], b5 = b[base + 5], b6 = b[base + 6], b7 = b[base + 7];
    float b8 = b[base + 8], b9 = b[base + 9], b10 = b[base + 10], b11 = b[base + 11];
    float b12 = b[base + 12], b13 = b[base + 13], b14 = b[base + 14], b15 = b[base + 15];

    float t0, t1, t2, t3, t4, t5, t6, t7;
    octonion_mul_inline_f32(a0, a1, a2, a3, a4, a5, a6, a7,
                            b0, b1, b2, b3, b4, b5, b6, b7,
                            &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

    float u0, u1, u2, u3, u4, u5, u6, u7;
    octonion_mul_inline_f32(b8, -b9, -b10, -b11, -b12, -b13, -b14, -b15,
                            a8, a9, a10, a11, a12, a13, a14, a15,
                            &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);

    out[base + 0] = t0 - u0; out[base + 1] = t1 - u1;
    out[base + 2] = t2 - u2; out[base + 3] = t3 - u3;
    out[base + 4] = t4 - u4; out[base + 5] = t5 - u5;
    out[base + 6] = t6 - u6; out[base + 7] = t7 - u7;

    float v0, v1, v2, v3, v4, v5, v6, v7;
    octonion_mul_inline_f32(b8, b9, b10, b11, b12, b13, b14, b15,
                            a0, a1, a2, a3, a4, a5, a6, a7,
                            &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);

    float w0, w1, w2, w3, w4, w5, w6, w7;
    octonion_mul_inline_f32(a8, a9, a10, a11, a12, a13, a14, a15,
                            b0, -b1, -b2, -b3, -b4, -b5, -b6, -b7,
                            &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

    out[base + 8] = v0 + w0; out[base + 9] = v1 + w1;
    out[base + 10] = v2 + w2; out[base + 11] = v3 + w3;
    out[base + 12] = v4 + w4; out[base + 13] = v5 + w5;
    out[base + 14] = v6 + w6; out[base + 15] = v7 + w7;
}

extern "C" __global__ void sedenion_add_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] + b[i];
}

extern "C" __global__ void sedenion_sub_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] - b[i];
}

extern "C" __global__ void sedenion_neg_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = -a[i];
}

extern "C" __global__ void sedenion_scale_f32(float *out, const float *a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 16) return;
    out[i] = a[i] * scalar;
}

extern "C" __global__ void sedenion_conjugate_f32(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 16) return;

    int component = idx % 16;
    out[idx] = (component == 0) ? a[idx] : -a[idx];
}

extern "C" __global__ void sedenion_norm_sq_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        float val = a[base + j];
        sum += val * val;
    }
    out[i] = sum;
}

extern "C" __global__ void sedenion_norm_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        float val = a[base + j];
        sum += val * val;
    }
    out[i] = sqrtf(sum);
}

extern "C" __global__ void sedenion_normalize_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        float val = a[base + j];
        sum += val * val;
    }
    float norm = sqrtf(sum);

    if (norm > 1e-7f) {
        float inv_norm = 1.0f / norm;
        for (int j = 0; j < 16; j++) {
            out[base + j] = a[base + j] * inv_norm;
        }
    } else {
        for (int j = 0; j < 16; j++) {
            out[base + j] = 0.0f;
        }
    }
}

extern "C" __global__ void sedenion_dot_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * 16;
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        sum += a[base + j] * b[base + j];
    }
    out[i] = sum;
}
