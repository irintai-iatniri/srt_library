// =============================================================================
// SRT Native BLAS: High-Performance Matrix Multiplication
// =============================================================================

#include "srt_constants.cuh"

// Maximum K dimension for golden weights constant memory
#define MAX_K 8192

// SRT-optimized tile configurations
#define BLOCK_M 128 // M dimension tile (register blocked)
#define BLOCK_N 128 // N dimension tile
#define BLOCK_K 8   // K dimension tile (fits in shared memory)
#define THREAD_M 8  // Elements per thread in M (8×8 sub-tile per thread)
#define THREAD_N 8  // Elements per thread in N

// Warp configuration
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8 // 256 threads / 32 = 8 warps

// Double buffering configuration
#define NUM_BUFFERS 2

// =============================================================================
// Helper Functions
// =============================================================================

// Cooperative tile loading with vectorized loads (float4)
__device__ __forceinline__ void load_tile(float As[BLOCK_K][BLOCK_M + 1],
                                          float Bs[BLOCK_K][BLOCK_N + 1],
                                          const float *__restrict__ A,
                                          const float *__restrict__ B, int tx,
                                          int ty, int block_row, int block_col,
                                          int k_tile, int M, int N, int K) {
// Cooperative load: A tile [BLOCK_K × BLOCK_M] (transposed in shared)
#pragma unroll
  for (int i = 0; i < BLOCK_M / 16; i++) {
    int row = ty + i * 16;
    int col = tx % BLOCK_K;
    int global_row = block_row + row;
    int global_col = k_tile + col;

    if (global_row < M && global_col < K) {
      // Vectorized load: 4 floats at once
      int idx = global_row * K + global_col;
      float4 a_vec = *reinterpret_cast<const float4 *>(&A[idx]);
      As[col][row] = a_vec.x; // Store first element
      if (col + 1 < BLOCK_K)
        As[col + 1][row] = a_vec.y;
      if (col + 2 < BLOCK_K)
        As[col + 2][row] = a_vec.z;
      if (col + 3 < BLOCK_K)
        As[col + 3][row] = a_vec.w;
    } else {
      As[col][row] = 0.0f;
    }
  }

// Cooperative load: B tile [BLOCK_K × BLOCK_N]
#pragma unroll
  for (int i = 0; i < BLOCK_N / 16; i++) {
    int row = ty + i * 16;
    int col = tx % BLOCK_K;
    int global_row = k_tile + col;
    int global_col = block_col + row;

    if (global_row < K && global_col < N) {
      // Vectorized load: 4 floats at once
      int idx = global_row * N + global_col;
      float4 b_vec = *reinterpret_cast<const float4 *>(&B[idx]);
      Bs[col][row] = b_vec.x;
      if (row + 1 < BLOCK_N)
        Bs[col][row + 1] = b_vec.y;
      if (row + 2 < BLOCK_N)
        Bs[col][row + 2] = b_vec.z;
      if (row + 3 < BLOCK_N)
        Bs[col][row + 3] = b_vec.w;
    } else {
      Bs[col][row] = 0.0f;
    }
  }
}

// =============================================================================
// Standard SGEMM: C = alpha * A * B + beta * C
// =============================================================================

extern "C" __global__ void sgemm_native_f32(float *__restrict__ C,
                                            const float *__restrict__ A,
                                            const float *__restrict__ B, int M,
                                            int N, int K, float alpha,
                                            float beta) {
  // Double-buffered shared memory tiles with padding to avoid bank conflicts
  __shared__ float As[NUM_BUFFERS][BLOCK_K][BLOCK_M + 1]; // +1 padding
  __shared__ float Bs[NUM_BUFFERS][BLOCK_K][BLOCK_N + 1];

  // Register file for accumulation (8×8 per thread)
  float acc[THREAD_M][THREAD_N] = {0.0f};

  // Register buffers for A and B fragments
  float a_frag[THREAD_M];
  float b_frag[THREAD_N];

  int tx = threadIdx.x; // 0-15
  int ty = threadIdx.y; // 0-15

  // Global tile position
  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  // Prefetch first tile
  int buf = 0;
  load_tile(As[buf], Bs[buf], A, B, tx, ty, block_row, block_col, 0, M, N, K);
  __syncthreads();

  // Loop over K dimension tiles with double buffering
  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    // Prefetch next tile while computing current
    if (k_tile + BLOCK_K < K) {
      int next_buf = 1 - buf;
      load_tile(As[next_buf], Bs[next_buf], A, B, tx, ty, block_row, block_col,
                k_tile + BLOCK_K, M, N, K);
    }

// Compute: Each thread computes 8×8 output
#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
// Load A fragment from shared memory
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
        a_frag[m] = As[buf][k][ty * THREAD_M + m];
      }

// Load B fragment from shared memory
#pragma unroll
      for (int n = 0; n < THREAD_N; n++) {
        b_frag[n] = Bs[buf][k][tx * THREAD_N + n];
      }

// Outer product accumulation
#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
        }
      }
    }

    __syncthreads();
    buf = 1 - buf; // Swap buffers
  }

// Write results with alpha/beta scaling
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
// SRT-Specific: Golden-Weighted GEMM
// =============================================================================

// C[i,j] = Σₖ A[i,k] × B[k,j] × e^{-k²/φ}
// Precomputed weights in constant memory
__constant__ float golden_weights[MAX_K]; // e^{-k²/φ} for k=0 to MAX_K-1

extern "C" __global__ void sgemm_golden_weighted_native_f32(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int M, int N, int K) {
  // Same double-buffered structure as sgemm_native_f32
  __shared__ float As[NUM_BUFFERS][BLOCK_K][BLOCK_M + 1];
  __shared__ float Bs[NUM_BUFFERS][BLOCK_K][BLOCK_N + 1];

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float a_frag[THREAD_M];
  float b_frag[THREAD_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  int buf = 0;
  load_tile(As[buf], Bs[buf], A, B, tx, ty, block_row, block_col, 0, M, N, K);
  __syncthreads();

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      int next_buf = 1 - buf;
      load_tile(As[next_buf], Bs[next_buf], A, B, tx, ty, block_row, block_col,
                k_tile + BLOCK_K, M, N, K);
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

      // Apply golden weight during accumulation
      float weight = golden_weights[k_tile + k];

#pragma unroll
      for (int m = 0; m < THREAD_M; m++) {
#pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
          acc[m][n] = fmaf(a_frag[m] * weight, b_frag[n], acc[m][n]);
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
        C[idx] = acc[m][n];
      }
    }
  }
}

// =============================================================================
// SRT-Specific: φ-Scaled GEMM
// =============================================================================

// C = φⁿ × (A × B) - fused scaling for SRT theory operations
extern "C" __global__ void
sgemm_phi_scaled_native_f32(float *__restrict__ C, const float *__restrict__ A,
                            const float *__restrict__ B, int M, int N, int K,
                            int phi_power // φ^phi_power scaling factor
) {
  // Double-buffered shared memory tiles
  __shared__ float As[NUM_BUFFERS][BLOCK_K][BLOCK_M + 1];
  __shared__ float Bs[NUM_BUFFERS][BLOCK_K][BLOCK_N + 1];

  float acc[THREAD_M][THREAD_N] = {0.0f};
  float a_frag[THREAD_M];
  float b_frag[THREAD_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  // Compute phi scale: φⁿ using Fibonacci formula
  float phi_scale = (phi_power >= 0) ? powf(PHI_F32, (float)phi_power)
                                     : powf(PHI_INV_F32, (float)(-phi_power));

  int buf = 0;
  load_tile(As[buf], Bs[buf], A, B, tx, ty, block_row, block_col, 0, M, N, K);
  __syncthreads();

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      int next_buf = 1 - buf;
      load_tile(As[next_buf], Bs[next_buf], A, B, tx, ty, block_row, block_col,
                k_tile + BLOCK_K, M, N, K);
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
          acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
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
void launch_sgemm_native_f32(void *stream, float *d_C, const float *d_A,
                             const float *d_B, int M, int N, int K, float alpha,
                             float beta) {
  dim3 block(16, 16); // 256 threads
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  sgemm_native_f32<<<grid, block, 0, cuda_stream>>>(d_C, d_A, d_B, M, N, K,
                                                    alpha, beta);
}

void launch_sgemm_golden_weighted_native_f32(void *stream, float *d_C,
                                             const float *d_A, const float *d_B,
                                             const float *d_golden_weights,
                                             int M, int N, int K) {
  // Upload golden weights to constant memory
  cudaMemcpyToSymbol(golden_weights, d_golden_weights, K * sizeof(float));

  dim3 block(16, 16);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  sgemm_golden_weighted_native_f32<<<grid, block, 0, cuda_stream>>>(
      d_C, d_A, d_B, M, N, K);
}

void launch_sgemm_phi_scaled_native_f32(void *stream, float *d_C,
                                        const float *d_A, const float *d_B,
                                        int M, int N, int K, int phi_power) {
  dim3 block(16, 16);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  sgemm_phi_scaled_native_f32<<<grid, block, 0, cuda_stream>>>(d_C, d_A, d_B, M,
                                                               N, K, phi_power);
}
}