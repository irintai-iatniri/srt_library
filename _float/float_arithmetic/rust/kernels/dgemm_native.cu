// =============================================================================
// SRT Native BLAS: Double Precision (f64) Matrix Multiplication
// =============================================================================

#include "srt_constants.cuh"

// SRT-optimized tile configurations for DGEMM
#define BLOCK_M 64 // Smaller tiles for f64 (register pressure)
#define BLOCK_N 64
#define BLOCK_K 8
#define THREAD_M 4 // Smaller per-thread tiles for f64
#define THREAD_N 4

// Warp configuration
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4 // 128 threads / 32 = 4 warps

// Double buffering
#define NUM_BUFFERS 2

// =============================================================================
// Helper Functions for DGEMM
// =============================================================================

__device__ __forceinline__ void
load_tile_f64(double As[BLOCK_K][BLOCK_M + 1], double Bs[BLOCK_K][BLOCK_N + 1],
              const double *__restrict__ A, const double *__restrict__ B,
              int tx, int ty, int block_row, int block_col, int k_tile, int M,
              int N, int K) {
// Cooperative load with double2 vectorization
#pragma unroll
  for (int i = 0; i < BLOCK_M / 16; i++) {
    int row = ty + i * 16;
    int col = tx % BLOCK_K;
    int global_row = block_row + row;
    int global_col = k_tile + col;

    if (global_row < M && global_col < K) {
      // Vectorized load: 2 doubles at once
      int idx = global_row * K + global_col;
      double2 a_vec = *reinterpret_cast<const double2 *>(&A[idx]);
      As[col][row] = a_vec.x;
      if (col + 1 < BLOCK_K)
        As[col + 1][row] = a_vec.y;
    } else {
      As[col][row] = 0.0;
    }
  }

#pragma unroll
  for (int i = 0; i < BLOCK_N / 16; i++) {
    int row = ty + i * 16;
    int col = tx % BLOCK_K;
    int global_row = k_tile + col;
    int global_col = block_col + row;

    if (global_row < K && global_col < N) {
      int idx = global_row * N + global_col;
      double2 b_vec = *reinterpret_cast<const double2 *>(&B[idx]);
      Bs[col][row] = b_vec.x;
      if (row + 1 < BLOCK_N)
        Bs[col][row + 1] = b_vec.y;
    } else {
      Bs[col][row] = 0.0;
    }
  }
}

// =============================================================================
// Standard DGEMM: C = alpha * A * B + beta * C
// =============================================================================

extern "C" __global__ void dgemm_native_f64(double *__restrict__ C,
                                            const double *__restrict__ A,
                                            const double *__restrict__ B, int M,
                                            int N, int K, double alpha,
                                            double beta) {
  // Double-buffered shared memory tiles
  __shared__ double As[NUM_BUFFERS][BLOCK_K][BLOCK_M + 1];
  __shared__ double Bs[NUM_BUFFERS][BLOCK_K][BLOCK_N + 1];

  // Register file for accumulation (4×4 per thread for f64)
  double acc[THREAD_M][THREAD_N] = {0.0};

  // Register buffers
  double a_frag[THREAD_M];
  double b_frag[THREAD_N];

  int tx = threadIdx.x; // 0-7
  int ty = threadIdx.y; // 0-15 (for 8×16 block)

  // Global tile position
  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  // Prefetch first tile
  int buf = 0;
  load_tile_f64(As[buf], Bs[buf], A, B, tx, ty, block_row, block_col, 0, M, N,
                K);
  __syncthreads();

  // Loop over K with double buffering
  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      int next_buf = 1 - buf;
      load_tile_f64(As[next_buf], Bs[next_buf], A, B, tx, ty, block_row,
                    block_col, k_tile + BLOCK_K, M, N, K);
    }

// Compute
#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
        a_frag[m] = As[buf][k][ty * THREAD_M + m];
      }

#pragma unroll
      for (int n = 0; n < THREAD_N; n++) {
        b_frag[n] = Bs[buf][k][tx * THREAD_N + n];
      }

#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          acc[m][n] = fma(a_frag[m], b_frag[n], acc[m][n]);
        }
      }
    }

    __syncthreads();
    buf = 1 - buf;
  }

// Write results
#pragma unroll
  for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
    for (int n = 0; n < THREAD_N; n++) {
      int row = block_row + ty * THREAD_M + m;
      int col = block_col + tx * THREAD_N + n;
      if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * acc[m][n] + beta * C[idx];
      }
    }
  }
}

// =============================================================================
// SRT-Specific: φ-Scaled DGEMM
// =============================================================================

extern "C" __global__ void dgemm_phi_scaled_native_f64(
    double *__restrict__ C, const double *__restrict__ A,
    const double *__restrict__ B, int M, int N, int K, int phi_power) {
  // Double-buffered shared memory tiles
  __shared__ double As[NUM_BUFFERS][BLOCK_K][BLOCK_M + 1];
  __shared__ double Bs[NUM_BUFFERS][BLOCK_K][BLOCK_N + 1];

  double acc[THREAD_M][THREAD_N] = {0.0};
  double a_frag[THREAD_M];
  double b_frag[THREAD_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  // Compute phi scale: φⁿ using exact formula
  double phi_scale = (phi_power >= 0) ? pow(PHI_F64, (double)phi_power)
                                      : pow(PHI_INV_F64, (double)(-phi_power));

  int buf = 0;
  load_tile_f64(As[buf], Bs[buf], A, B, tx, ty, block_row, block_col, 0, M, N,
                K);
  __syncthreads();

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      int next_buf = 1 - buf;
      load_tile_f64(As[next_buf], Bs[next_buf], A, B, tx, ty, block_row,
                    block_col, k_tile + BLOCK_K, M, N, K);
    }

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
        a_frag[m] = As[buf][k][ty * THREAD_M + m];
      }
#pragma unroll
      for (int n = 0; n < THREAD_N; n++) {
        b_frag[n] = Bs[buf][k][tx * THREAD_N + n];
      }
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          acc[m][n] = fma(a_frag[m], b_frag[n], acc[m][n]);
        }
      }
    }
    __syncthreads();
    buf = 1 - buf;
  }

// Apply φ scaling and write results
#pragma unroll
  for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
    for (int n = 0; n < THREAD_N; n++) {
      int row = block_row + ty * THREAD_M + m;
      int col = block_col + tx * THREAD_N + n;
      if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = phi_scale * acc[m][n];
      }
    }
  }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {
void launch_dgemm_native_f64(void *stream, double *d_C, const double *d_A,
                             const double *d_B, int M, int N, int K,
                             double alpha, double beta) {
  dim3 block(8, 16); // 128 threads
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  dgemm_native_f64<<<grid, block, 0, cuda_stream>>>(d_C, d_A, d_B, M, N, K,
                                                    alpha, beta);
}
}