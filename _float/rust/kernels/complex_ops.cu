// =============================================================================
// SRT Complex Operations: Theory-Aligned Complex64/Complex128 Kernels
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "srt_constants.cuh"
using namespace std;

// =============================================================================
// Helper Functions for Complex Operations
// =============================================================================

// Complex128 accessors (real/imag interleaved)
__device__ __forceinline__ double complex128_real(const double* z, int i) {
    return z[2*i];
}

__device__ __forceinline__ double complex128_imag(const double* z, int i) {
    return z[2*i + 1];
}

__device__ __forceinline__ void complex128_set(double* z, int i, double re, double im) {
    z[2*i] = re;
    z[2*i + 1] = im;
}

// Complex64 accessors
__device__ __forceinline__ float complex64_real(const float* z, int i) {
    return z[2*i];
}

__device__ __forceinline__ float complex64_imag(const float* z, int i) {
    return z[2*i + 1];
}

__device__ __forceinline__ void complex64_set(float* z, int i, float re, float im) {
    z[2*i] = re;
    z[2*i + 1] = im;
}

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
__device__ __forceinline__ void complex_mul(double* result, double a_re, double a_im,
                                           double b_re, double b_im) {
    double re = a_re * b_re - a_im * b_im;
    double im = a_re * b_im + a_im * b_re;
    result[0] = re;
    result[1] = im;
}

// Complex division: (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
__device__ __forceinline__ void complex_div(double* result, double a_re, double a_im,
                                           double b_re, double b_im) {
    double denom = b_re * b_re + b_im * b_im;
    double re = (a_re * b_re + a_im * b_im) / denom;
    double im = (a_im * b_re - a_re * b_im) / denom;
    result[0] = re;
    result[1] = im;
}

// =============================================================================
// Phase Operations (i≈π Aligned)
// =============================================================================

// Extract phase angle: arg(z) = atan2(im, re)
extern "C" __global__ void arg_c128(double* out, const double* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);
    out[i] = atan2(im, re);
}

extern "C" __global__ void arg_c64(float* out, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float re = complex64_real(z, i);
    float im = complex64_imag(z, i);
    out[i] = atan2f(im, re);
}

// Normalize to unit circle: z / |z|
extern "C" __global__ void normalize_phase_c128(double* out, const double* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);
    double mag = sqrt(re * re + im * im);

    if (mag > 0.0) {
        complex128_set(out, i, re / mag, im / mag);
    } else {
        complex128_set(out, i, 0.0, 0.0);
    }
}

extern "C" __global__ void normalize_phase_c64(float* out, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float re = complex64_real(z, i);
    float im = complex64_imag(z, i);
    float mag = sqrtf(re * re + im * im);

    if (mag > 0.0f) {
        complex64_set(out, i, re / mag, im / mag);
    } else {
        complex64_set(out, i, 0.0f, 0.0f);
    }
}

// Phase rotation by θ: z * e^{iθ}
extern "C" __global__ void rotate_phase_c128(double* out, const double* z, double theta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    // z * e^{iθ} = (re + i*im) * (cosθ + i*sinθ)
    double result_re = re * cos_theta - im * sin_theta;
    double result_im = re * sin_theta + im * cos_theta;

    complex128_set(out, i, result_re, result_im);
}

extern "C" __global__ void rotate_phase_c64(float* out, const float* z, float theta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float re = complex64_real(z, i);
    float im = complex64_imag(z, i);

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float result_re = re * cos_theta - im * sin_theta;
    float result_im = re * sin_theta + im * cos_theta;

    complex64_set(out, i, result_re, result_im);
}

// Phase quantization to π-multiples (i≈π postulate)
extern "C" __global__ void quantize_phase_pi_c128(double* out, const double* z, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);
    double phase = atan2(im, re);

    // Quantize to multiples of π/k (i≈π alignment)
    double pi_k = PI_F64 / (double)k;
    double quantized_phase = round(phase / pi_k) * pi_k;

    double cos_qp = cos(quantized_phase);
    double sin_qp = sin(quantized_phase);
    double mag = sqrt(re * re + im * im);

    complex128_set(out, i, mag * cos_qp, mag * sin_qp);
}

// =============================================================================
// Conjugate Operations
// =============================================================================

// Complex conjugate: z* = re - i*im
extern "C" __global__ void conj_c128(double* out, const double* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);
    complex128_set(out, i, re, -im);
}

extern "C" __global__ void conj_c64(float* out, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float re = complex64_real(z, i);
    float im = complex64_imag(z, i);
    complex64_set(out, i, re, -im);
}

// Hermitian inner product: <a|b> = Σᵢ conj(aᵢ) * bᵢ
extern "C" __global__ void hermitian_inner_c128(double* out, const double* a, const double* b, int n) {
    __shared__ double s_real[256];
    __shared__ double s_imag[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_real = 0.0;
    double local_imag = 0.0;

    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        double a_re = complex128_real(a, i);
        double a_im = complex128_imag(a, i);
        double b_re = complex128_real(b, i);
        double b_im = complex128_imag(b, i);

        // conj(a) * b = (a_re - i*a_im) * (b_re + i*b_im)
        local_real += a_re * b_re + a_im * b_im;
        local_imag += a_re * b_im - a_im * b_re;
    }

    // Warp-level reduction
    local_real = warp_reduce_sum_f64(local_real);
    local_imag = warp_reduce_sum_f64(local_imag);

    // Store to shared memory
    if (tid % 32 == 0) {
        s_real[tid / 32] = local_real;
        s_imag[tid / 32] = local_imag;
    }

    __syncthreads();

    // Final reduction
    if (tid < 32) {
        double r_val = (tid < blockDim.x / 32) ? s_real[tid] : 0.0;
        double i_val = (tid < blockDim.x / 32) ? s_imag[tid] : 0.0;
        r_val = warp_reduce_sum_f64(r_val);
        i_val = warp_reduce_sum_f64(i_val);

        if (tid == 0) {
            out[0] = r_val;
            out[1] = i_val;
        }
    }
}

// =============================================================================
// Syntonic Phase Operations (i≈π Postulate)
// =============================================================================

// Phase-syntony coupling: z * e^{iπS} where S is syntony
extern "C" __global__ void phase_syntony_c128(double* out, const double* z, const double* syntony, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);
    double s = syntony[i];

    // e^{iπS} = cos(πS) + i*sin(πS)
    double cos_pi_s = cos(PI_F64 * s);
    double sin_pi_s = sin(PI_F64 * s);

    // z * e^{iπS}
    double result_re = re * cos_pi_s - im * sin_pi_s;
    double result_im = re * sin_pi_s + im * cos_pi_s;

    complex128_set(out, i, result_re, result_im);
}

// Berry phase accumulation: geometric phase from recursion
extern "C" __global__ void berry_phase_c128(double* out, const double* psi, const double* grad_psi, int n) {
    __shared__ double s_phase[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_phase = 0.0;

    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        double psi_re = complex128_real(psi, i);
        double psi_im = complex128_imag(psi, i);

        // grad_psi is complex gradient
        double grad_re = complex128_real(grad_psi, i);
        double grad_im = complex128_imag(grad_psi, i);

        // Berry connection: Im[conj(ψ) * ∇ψ]
        double connection = psi_re * grad_im - psi_im * grad_re;
        local_phase += connection;
    }

    // Warp-level reduction
    local_phase = warp_reduce_sum_f64(local_phase);

    // Store to shared memory
    if (tid % 32 == 0) {
        s_phase[tid / 32] = local_phase;
    }

    __syncthreads();

    // Final reduction
    if (tid < 32) {
        double val = (tid < blockDim.x / 32) ? s_phase[tid] : 0.0;
        val = warp_reduce_sum_f64(val);

        if (tid == 0) {
            out[0] = -val;  // Berry phase = -Im ∮ A
        }
    }
}

// =============================================================================
// Wave Function Operations
// =============================================================================

// Probability amplitude: |ψ|² = re² + im²
extern "C" __global__ void probability_c128(double* out, const double* psi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(psi, i);
    double im = complex128_imag(psi, i);
    out[i] = re * re + im * im;
}

extern "C" __global__ void probability_c64(float* out, const float* psi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float re = complex64_real(psi, i);
    float im = complex64_imag(psi, i);
    out[i] = re * re + im * im;
}

// Wave function overlap is implemented via hermitian_inner_c128

// Normalize wave function: ψ/√(Σ|ψ|²)
extern "C" __global__ void normalize_wavefunction_c128(double* out, const double* psi, double norm_sqrt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(psi, i);
    double im = complex128_imag(psi, i);

    complex128_set(out, i, re / norm_sqrt, im / norm_sqrt);
}

// =============================================================================
// Golden Complex Operations
// =============================================================================

// Golden phase rotation: z * e^{iπ/φ}
extern "C" __global__ void golden_rotate_c128(double* out, const double* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double re = complex128_real(z, i);
    double im = complex128_imag(z, i);

    // e^{iπ/φ} = cos(π/φ) + i*sin(π/φ)
    double theta = PI_F64 / PHI_F64;
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    double result_re = re * cos_theta - im * sin_theta;
    double result_im = re * sin_theta + im * cos_theta;

    complex128_set(out, i, result_re, result_im);
}

// φ-weighted complex sum: Σᵢ zᵢ * φ^{-i}
extern "C" __global__ void phi_weighted_sum_c128(double* out, const double* z, int n) {
    __shared__ double s_real[256];
    __shared__ double s_imag[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_real = 0.0;
    double local_imag = 0.0;

    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        double re = complex128_real(z, i);
        double im = complex128_imag(z, i);
        double weight = pow(PHI_INV_F64, (double)i);

        local_real += re * weight;
        local_imag += im * weight;
    }

    // Warp-level reduction
    local_real = warp_reduce_sum_f64(local_real);
    local_imag = warp_reduce_sum_f64(local_imag);

    // Store to shared memory
    if (tid % 32 == 0) {
        s_real[tid / 32] = local_real;
        s_imag[tid / 32] = local_imag;
    }

    __syncthreads();

    // Final reduction
    if (tid < 32) {
        double r_val = (tid < blockDim.x / 32) ? s_real[tid] : 0.0;
        double i_val = (tid < blockDim.x / 32) ? s_imag[tid] : 0.0;
        r_val = warp_reduce_sum_f64(r_val);
        i_val = warp_reduce_sum_f64(i_val);

        if (tid == 0) {
            out[0] = r_val;
            out[1] = i_val;
        }
    }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// Phase operations
void launch_arg_c128(cudaStream_t stream, double* out, const double* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    arg_c128<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_normalize_phase_c128(cudaStream_t stream, double* out, const double* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    normalize_phase_c128<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_rotate_phase_c128(cudaStream_t stream, double* out, const double* z, double theta, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    rotate_phase_c128<<<grid, block, 0, stream>>>(out, z, theta, n);
}

void launch_quantize_phase_pi_c128(cudaStream_t stream, double* out, const double* z, int k, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    quantize_phase_pi_c128<<<grid, block, 0, stream>>>(out, z, k, n);
}

// Conjugate operations
void launch_conj_c128(cudaStream_t stream, double* out, const double* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    conj_c128<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_hermitian_inner_c128(cudaStream_t stream, double* out, const double* a, const double* b, int n) {
    dim3 block(256);
    dim3 grid(1);
    hermitian_inner_c128<<<grid, block, 0, stream>>>(out, a, b, n);
}

// Syntonic phase operations
void launch_phase_syntony_c128(cudaStream_t stream, double* out, const double* z, const double* syntony, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    phase_syntony_c128<<<grid, block, 0, stream>>>(out, z, syntony, n);
}

void launch_berry_phase_c128(cudaStream_t stream, double* out, const double* psi, const double* grad_psi, int n) {
    dim3 block(256);
    dim3 grid(1);
    berry_phase_c128<<<grid, block, 0, stream>>>(out, psi, grad_psi, n);
}

// Wave function operations
void launch_probability_c128(cudaStream_t stream, double* out, const double* psi, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    probability_c128<<<grid, block, 0, stream>>>(out, psi, n);
}

void launch_overlap_c128(cudaStream_t stream, double* out, const double* phi, const double* psi, int n) {
    dim3 block(256);
    dim3 grid(1);
    hermitian_inner_c128<<<grid, block, 0, stream>>>(out, phi, psi, n);
}

void launch_normalize_wavefunction_c128(cudaStream_t stream, double* out, const double* psi, double norm_sqrt, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    normalize_wavefunction_c128<<<grid, block, 0, stream>>>(out, psi, norm_sqrt, n);
}

// Golden complex operations
void launch_golden_rotate_c128(cudaStream_t stream, double* out, const double* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    golden_rotate_c128<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_phi_weighted_sum_c128(cudaStream_t stream, double* out, const double* z, int n) {
    dim3 block(256);
    dim3 grid(1);
    phi_weighted_sum_c128<<<grid, block, 0, stream>>>(out, z, n);
}

// Complex64 variants (selected key operations)
void launch_arg_c64(cudaStream_t stream, float* out, const float* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    arg_c64<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_conj_c64(cudaStream_t stream, float* out, const float* z, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    conj_c64<<<grid, block, 0, stream>>>(out, z, n);
}

void launch_probability_c64(cudaStream_t stream, float* out, const float* psi, int n) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    probability_c64<<<grid, block, 0, stream>>>(out, psi, n);
}

}
