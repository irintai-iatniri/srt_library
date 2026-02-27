// Syntonic CUDA Kernels - Matrix Operations
// Native GPU matrix multiplication with SRT theory-correct operations
// Supports golden ratio algebra, complex arithmetic, and batched operations

#include "srt_constants.cuh"

// =============================================================================
// Standard Matrix Multiplication (GEMM)
// =============================================================================

// Basic matrix multiplication: C = A × B
// Row-major layout, no transposes
extern "C" __global__ void matmul_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

extern "C" __global__ void matmul_f32(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Complex matrix multiplication: C = A × B (interleaved format)
extern "C" __global__ void matmul_c128(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int c_idx = 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        int a_idx = 2 * (row * K + k);
        int b_idx = 2 * (k * N + col);

        double a_re = A[a_idx];
        double a_im = A[a_idx + 1];
        double b_re = B[b_idx];
        double b_im = B[b_idx + 1];

        // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// =============================================================================
// Tiled Matrix Multiplication (Optimized GEMM)
// =============================================================================

// Shared memory tile size
#define TILE_SIZE 16

// Tiled matrix multiplication for better memory coalescing
extern "C" __global__ void matmul_tiled_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" __global__ void matmul_tiled_f32(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Transposed Matrix Multiplication Variants
// =============================================================================

// Matrix multiplication with A transposed: C = Aᵀ × B
extern "C" __global__ void matmul_tn_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Matrix multiplication with B transposed: C = A × Bᵀ
extern "C" __global__ void matmul_nt_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// Matrix multiplication with both transposed: C = Aᵀ × Bᵀ
extern "C" __global__ void matmul_tt_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// Matrix multiplication with A transposed: C = Aᵀ × B (f32)
extern "C" __global__ void matmul_tn_f32(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Matrix multiplication with A transposed: C = Aᵀ × B (complex128, interleaved)
extern "C" __global__ void matmul_tn_c128(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int c_idx = 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        // A is accessed transposed: A[k * M + row]
        int a_idx = 2 * (k * M + row);
        int b_idx = 2 * (k * N + col);

        double a_re = A[a_idx];
        double a_im = A[a_idx + 1];
        double b_re = B[b_idx];
        double b_im = B[b_idx + 1];

        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// Matrix multiplication with B transposed: C = A × Bᵀ (f32)
extern "C" __global__ void matmul_nt_f32(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// Matrix multiplication with both transposed: C = Aᵀ × Bᵀ (f32)
extern "C" __global__ void matmul_tt_f32(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// =============================================================================
// Hermitian Matrix Operations (Complex Conjugate Transpose)
// =============================================================================

// Hermitian-None: C = A† × B
extern "C" __global__ void matmul_hn_c128(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int c_idx = 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        int a_idx = 2 * (k * M + row);  // A† means conjugate of A[k*M + row]
        int b_idx = 2 * (k * N + col);

        double a_re = A[a_idx];
        double a_im = -A[a_idx + 1];  // Conjugate: negate imaginary part
        double b_re = B[b_idx];
        double b_im = B[b_idx + 1];

        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// None-Hermitian: C = A × B†
extern "C" __global__ void matmul_nh_c128(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int c_idx = 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        int a_idx = 2 * (row * K + k);
        int b_idx = 2 * (col * K + k);  // B† means conjugate of B[col*K + k]

        double a_re = A[a_idx];
        double a_im = A[a_idx + 1];
        double b_re = B[b_idx];
        double b_im = -B[b_idx + 1];  // Conjugate: negate imaginary part

        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// =============================================================================
// General Matrix Multiply (GEMM): C = α × op(A) × op(B) + β × C
// =============================================================================

// GEMM with no transposes: C = α × A × B + β × C
extern "C" __global__ void gemm_nn_f64(
    double *C,
    const double *A,
    const double *B,
    double alpha, double beta,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    int c_idx = row * N + col;
    C[c_idx] = alpha * sum + beta * C[c_idx];
}

// GEMM with A transposed: C = α × Aᵀ × B + β × C
extern "C" __global__ void gemm_tn_f64(
    double *C,
    const double *A,
    const double *B,
    double alpha, double beta,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[k * N + col];
    }

    int c_idx = row * N + col;
    C[c_idx] = alpha * sum + beta * C[c_idx];
}

// GEMM with B transposed: C = α × A × Bᵀ + β × C
extern "C" __global__ void gemm_nt_f64(
    double *C,
    const double *A,
    const double *B,
    double alpha, double beta,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }

    int c_idx = row * N + col;
    C[c_idx] = alpha * sum + beta * C[c_idx];
}

// GEMM with both transposed: C = α × Aᵀ × Bᵀ + β × C
extern "C" __global__ void gemm_tt_f64(
    double *C,
    const double *A,
    const double *B,
    double alpha, double beta,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[k * M + row] * B[col * K + k];
    }

    int c_idx = row * N + col;
    C[c_idx] = alpha * sum + beta * C[c_idx];
}

// =============================================================================
// Batched Matrix Multiplication
// =============================================================================

// Batched matrix multiplication: C[i] = A[i] × B[i]
extern "C" __global__ void bmm_f64(
    double *C,
    const double *A,
    const double *B,
    int batch_size, int M, int N, int K
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || row >= M || col >= N) return;

    int a_offset = batch * M * K;
    int b_offset = batch * K * N;
    int c_offset = batch * M * N;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
    }
    C[c_offset + row * N + col] = sum;
}

// Batched matrix multiplication with B transposed: C[i] = A[i] × B[i]ᵀ
extern "C" __global__ void bmm_nt_f64(
    double *C,
    const double *A,
    const double *B,
    int batch_size, int M, int N, int K
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || row >= M || col >= N) return;

    // Offsets
    int a_batch_offset = batch * M * K;
    int b_batch_offset = batch * N * K;
    int c_batch_offset = batch * M * N;
    
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        // A[row, k]
        double a_val = A[a_batch_offset + row * K + k];
        // B[col, k] (since we want B^T[k, col] which is B[col, k])
        double b_val = B[b_batch_offset + col * K + k];
        sum += a_val * b_val;
    }
    
    C[c_batch_offset + row * N + col] = sum;
}

extern "C" __global__ void bmm_c128(
    double *C,
    const double *A,
    const double *B,
    int batch_size, int M, int N, int K
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || row >= M || col >= N) return;

    int a_offset = batch * M * K * 2;
    int b_offset = batch * K * N * 2;
    int c_offset = batch * M * N * 2;

    int c_idx = c_offset + 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        int a_idx = a_offset + 2 * (row * K + k);
        int b_idx = b_offset + 2 * (k * N + col);

        double a_re = A[a_idx];
        double a_im = A[a_idx + 1];
        double b_re = B[b_idx];
        double b_im = B[b_idx + 1];

        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// =============================================================================
// SRT-Specific Matrix Operations (Golden Ratio Algebra)
// =============================================================================

// φ-scaled matrix multiplication: C = φⁿ × (A × B)
extern "C" __global__ void matmul_phi_scaled_f64(
    double *C,
    const double *A,
    const double *B,
    int n,  // Power of phi
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // Apply φⁿ scaling
    double phi_power = pow(PHI_F64, (double)n);
    C[row * N + col] = phi_power * sum;
}

// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
extern "C" __global__ void golden_commutator_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double ab = 0.0;
    double ba = 0.0;

    for (int k = 0; k < K; k++) {
        ab += A[row * K + k] * B[k * N + col];
        ba += B[row * K + k] * A[k * N + col];
    }

    C[row * N + col] = ab - PHI_INV_F64 * ba;
}

// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
extern "C" __global__ void golden_anticommutator_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double ab = 0.0;
    double ba = 0.0;

    for (int k = 0; k < K; k++) {
        ab += A[row * K + k] * B[k * N + col];
        ba += B[row * K + k] * A[k * N + col];
    }

    C[row * N + col] = ab + PHI_INV_F64 * ba;
}

// Golden-weighted matrix multiplication: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(-k²/φ)
extern "C" __global__ void matmul_golden_weighted_f64(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        double weight = exp(-(double)(k * k) * PHI_INV_F64);
        sum += A[row * K + k] * B[k * N + col] * weight;
    }
    C[row * N + col] = sum;
}

// Complex golden-weighted matrix multiplication
extern "C" __global__ void matmul_golden_weighted_c128(
    double *C,
    const double *A,
    const double *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int c_idx = 2 * (row * N + col);
    double real_sum = 0.0;
    double imag_sum = 0.0;

    for (int k = 0; k < K; k++) {
        double weight = exp(-(double)(k * k) * PHI_INV_F64);

        int a_idx = 2 * (row * K + k);
        int b_idx = 2 * (k * N + col);

        double a_re = A[a_idx] * weight;
        double a_im = A[a_idx + 1] * weight;
        double b_re = B[b_idx];
        double b_im = B[b_idx + 1];

        real_sum += a_re * b_re - a_im * b_im;
        imag_sum += a_re * b_im + a_im * b_re;
    }

    C[c_idx] = real_sum;
    C[c_idx + 1] = imag_sum;
}

// =============================================================================
// Complex Division (for SRT complex arithmetic)
// =============================================================================

// Complex division: C = A / B (element-wise)
extern "C" __global__ void complex_div_c128(
    double *C,
    const double *A,
    const double *B,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double a_re = A[idx];
    double a_im = A[idx + 1];
    double b_re = B[idx];
    double b_im = B[idx + 1];

    // Division: (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
    double denom = b_re * b_re + b_im * b_im;
    double real_part = (a_re * b_re + a_im * b_im) / denom;
    double imag_part = (a_im * b_re - a_re * b_im) / denom;

    C[idx] = real_part;
    C[idx + 1] = imag_part;
}

// Complex reciprocal: C = 1 / A
extern "C" __global__ void complex_reciprocal_c128(
    double *C,
    const double *A,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double a_re = A[idx];
    double a_im = A[idx + 1];

    // 1/(a + bi) = (a - bi) / (a² + b²)
    double denom = a_re * a_re + a_im * a_im;
    double real_part = a_re / denom;
    double imag_part = -a_im / denom;

    C[idx] = real_part;
    C[idx + 1] = imag_part;
}

// =============================================================================
// Q32.32 Fixed-Point Matrix Multiplication (Exact, Deterministic)
// =============================================================================

#include "fixed_point.cuh"

// Basic Q32.32 matrix multiplication: C = A × B
// Exact arithmetic, zero floating-point drift
extern "C" __global__ void matmul_fp64(
    int64_t *C,
    const int64_t *A,
    const int64_t *B,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    fp64 sum = SRT_FP_ZERO;
    for (int k = 0; k < K; k++) {
        fp64 a_val = A[row * K + k];
        fp64 b_val = B[k * N + col];
        sum = fp_add(sum, fp_mul(a_val, b_val));
    }
    C[row * N + col] = sum;
}

// Tiled Q32.32 matmul (shared memory optimization)
// Improves performance for larger matrices
#define TILE_SIZE 16

extern "C" __global__ void matmul_fp64_tiled(
    int64_t *C,
    const int64_t *A,
    const int64_t *B,
    int M, int N, int K
) {
    __shared__ fp64 As[TILE_SIZE][TILE_SIZE];
    __shared__ fp64 Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    fp64 sum = SRT_FP_ZERO;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = SRT_FP_ZERO;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = SRT_FP_ZERO;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fp_add(sum, fp_mul(As[threadIdx.y][k], Bs[k][threadIdx.x]));
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}