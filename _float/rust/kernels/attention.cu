// =============================================================================
// SRT Flash Attention Kernels: Theory-Aligned Attention Mechanisms
// =============================================================================

#include "srt_constants.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using namespace std;

// =============================================================================
// Block Size Configuration (SRT-optimized)
// =============================================================================

#define ATTN_BLOCK_M 64 // Query block size
#define ATTN_BLOCK_N 64 // Key/Value block size
#define ATTN_BLOCK_D 64 // Head dimension block
// SRT-specific block sizes
#define ATTN_BLOCK_GOLDEN 36        // E₆ Golden Cone
#define ATTN_BLOCK_MERSENNE_127 127 // M₇ head dim
#define ATTN_BLOCK_MERSENNE_31 31   // M₅ head dim

// =============================================================================
// Online Softmax Helper Functions
// =============================================================================

// Update running statistics for online softmax
__device__ __forceinline__ void
online_softmax_update(float &running_max, float &running_sum, float new_val) {
  float old_max = running_max;
  running_max = fmaxf(running_max, new_val);
  running_sum =
      running_sum * expf(old_max - running_max) + expf(new_val - running_max);
}

// Normalize attention scores with running statistics
__device__ __forceinline__ void online_softmax_normalize(float *scores,
                                                         float running_max,
                                                         float running_sum,
                                                         int len) {
  float inv_sum = 1.0f / running_sum;
#pragma unroll
  for (int i = 0; i < len; i++) {
    scores[i] = expf(scores[i] - running_max) * inv_sum;
  }
}

// =============================================================================
// Standard Flash Attention (Memory-Efficient Fused)
// =============================================================================

// NOTE: warp_reduce_sum defined in srt_constants.cuh

// Flash Attention: O(N) memory attention with blocking
extern "C" __global__ void
flash_attention_f32(float *__restrict__ out,     // [B, H, N, D]
                    const float *__restrict__ Q, // [B, H, N, D]
                    const float *__restrict__ K, // [B, H, N, D]
                    const float *__restrict__ V, // [B, H, N, D]
                    int batch, int heads, int seq_len, int head_dim,
                    float scale // 1/√d
) {
  // Block indices
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = blockDim.x / 32;

  // Shared memory for Q, K, V tiles
  __shared__ float s_Q[ATTN_BLOCK_M][ATTN_BLOCK_D];
  __shared__ float s_K[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_V[ATTN_BLOCK_N][ATTN_BLOCK_D];

  // Block-level running statistics for online softmax
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];
  __shared__ float s_row_acc[ATTN_BLOCK_M][ATTN_BLOCK_D];

  // Global memory offsets
  int q_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;
  int kv_offset = ((batch_idx * heads + head_idx) * seq_len) * head_dim;
  int out_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;

  // Initialize shared memory statistics
  if (tid < ATTN_BLOCK_M) {
    s_row_max[tid] = -INFINITY;
    s_row_sum[tid] = 0.0f;
  }
  // Initialize s_row_acc
  for (int m_idx = tid / ATTN_BLOCK_D; m_idx < ATTN_BLOCK_M;
       m_idx += blockDim.x / ATTN_BLOCK_D) {
    int d_idx = tid % ATTN_BLOCK_D;
    if (d_idx < ATTN_BLOCK_D) {
      s_row_acc[m_idx][d_idx] = 0.0f;
    }
  }

  // Load Q tile into shared memory
  for (int m = 0; m < ATTN_BLOCK_M; m++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m < seq_len && d < head_dim) {
        s_Q[m][d] = Q[q_offset + m * head_dim + d];
      } else {
        s_Q[m][d] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Loop over K/V blocks
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K and V tiles
    for (int n = 0; n < ATTN_BLOCK_N; n++) {
      for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len && d < head_dim) {
          s_K[n][d] = K[kv_offset + global_n * head_dim + d];
          s_V[n][d] = V[kv_offset + global_n * head_dim + d];
        } else {
          s_K[n][d] = 0.0f;
          s_V[n][d] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Each warp handles one query row
    for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m >= seq_len)
        continue;

      // Compute attention scores for all K positions in this tile
      float local_scores[ATTN_BLOCK_N];
      float tile_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;

        // Compute Q·K^T dot product
        float score = 0.0f;
        for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
          score += s_Q[m][d] * s_K[n][d];
        }

        // Warp-level reduction for dot product
        score = warp_reduce_sum(score);

        // Apply scale and check bounds
        if (global_n < seq_len) {
          local_scores[n] = score * scale;
          tile_max = fmaxf(tile_max, local_scores[n]);
        } else {
          local_scores[n] = -INFINITY;
        }
      }

      // Update running max across tiles (only lane 0 does shared memory ops)
      if (lane_id == 0) {
        float old_max = s_row_max[m];
        float new_max = fmaxf(old_max, tile_max);

        // Rescale previous sum and accumulator if max changed
        if (new_max > old_max) {
          float scale_factor = expf(old_max - new_max);
          s_row_sum[m] *= scale_factor;

          // Rescale accumulated values
          for (int d = 0; d < ATTN_BLOCK_D; d++) {
            s_row_acc[m][d] *= scale_factor;
          }
        }
        s_row_max[m] = new_max;
      }

      __syncwarp();

      // Compute softmax weights and accumulate V
      float row_max_val = s_row_max[m];
      float local_sum = 0.0f;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len) {
          float weight = expf(local_scores[n] - row_max_val);
          local_sum += weight;

          // Accumulate weighted V values
          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            atomicAdd(&s_row_acc[m][d], weight * s_V[n][d]);
          }
        }
      }

      // Update running sum (only lane 0)
      if (lane_id == 0) {
        s_row_sum[m] += local_sum;
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
    int global_m = query_block * ATTN_BLOCK_M + m;
    if (global_m >= seq_len)
      continue;

    float inv_sum = 1.0f / s_row_sum[m];

    for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
      if (d < head_dim) {
        out[out_offset + m * head_dim + d] = s_row_acc[m][d] * inv_sum;
      }
    }
  }
}

// =============================================================================
// SRT Syntony-Focused Attention
// =============================================================================

// Attention with syntony density focusing: ∇(ΔS) · n̂_target
extern "C" __global__ void flash_attention_syntony_f32(
    float *__restrict__ out, const float *__restrict__ Q,
    const float *__restrict__ K, const float *__restrict__ V,
    const float *__restrict__ syntony, // Per-position syntony density [B, H, N]
    int batch, int heads, int seq_len, int head_dim, float scale) {
  // Syntony density focusing: A[i,j] *= (syntony[i] + syntony[j]) / 2
  // Implements ∇(ΔS) · n̂_target attention mechanism

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = blockDim.x / 32;

  // Shared memory for Q, K, V tiles and syntony
  __shared__ float s_Q[ATTN_BLOCK_M][ATTN_BLOCK_D];
  __shared__ float s_K[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_V[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_syntony_Q[ATTN_BLOCK_M];
  __shared__ float s_syntony_K[ATTN_BLOCK_N];

  // Block-level running statistics for online softmax
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];
  __shared__ float s_row_acc[ATTN_BLOCK_M][ATTN_BLOCK_D];

  // Global memory offsets
  int syntony_offset = (batch_idx * heads + head_idx) * seq_len;
  int q_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;
  int kv_offset = ((batch_idx * heads + head_idx) * seq_len) * head_dim;
  int out_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;

  // Initialize shared memory statistics
  if (tid < ATTN_BLOCK_M) {
    s_row_max[tid] = -INFINITY;
    s_row_sum[tid] = 0.0f;
    int global_m = query_block * ATTN_BLOCK_M + tid;
    s_syntony_Q[tid] =
        (global_m < seq_len) ? syntony[syntony_offset + global_m] : 0.0f;
  }
  // Initialize s_row_acc
  for (int m_idx = 0; m_idx < ATTN_BLOCK_M; m_idx++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      s_row_acc[m_idx][d] = 0.0f;
    }
  }

  // Load Q tile into shared memory
  for (int m = 0; m < ATTN_BLOCK_M; m++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m < seq_len && d < head_dim) {
        s_Q[m][d] = Q[q_offset + m * head_dim + d];
      } else {
        s_Q[m][d] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Loop over K/V blocks
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K, V and syntony tiles
    for (int n = 0; n < ATTN_BLOCK_N; n++) {
      int global_n = kv_block * ATTN_BLOCK_N + n;
      for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
        if (global_n < seq_len && d < head_dim) {
          s_K[n][d] = K[kv_offset + global_n * head_dim + d];
          s_V[n][d] = V[kv_offset + global_n * head_dim + d];
        } else {
          s_K[n][d] = 0.0f;
          s_V[n][d] = 0.0f;
        }
      }
      if (tid == n) {
        s_syntony_K[n] =
            (global_n < seq_len) ? syntony[syntony_offset + global_n] : 0.0f;
      }
    }

    __syncthreads();

    // Each warp handles one query row
    for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m >= seq_len)
        continue;

      float local_scores[ATTN_BLOCK_N];
      float tile_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;

        // Compute Q·K^T dot product
        float score = 0.0f;
        for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
          score += s_Q[m][d] * s_K[n][d];
        }
        score = warp_reduce_sum(score);

        // Apply scale and syntony weighting
        if (global_n < seq_len) {
          float syntony_weight = (s_syntony_Q[m] + s_syntony_K[n]) * 0.5f;
          local_scores[n] = score * scale * syntony_weight;
          tile_max = fmaxf(tile_max, local_scores[n]);
        } else {
          local_scores[n] = -INFINITY;
        }
      }

      // Update running max across tiles
      if (lane_id == 0) {
        float old_max = s_row_max[m];
        float new_max = fmaxf(old_max, tile_max);

        if (new_max > old_max) {
          float scale_factor = expf(old_max - new_max);
          s_row_sum[m] *= scale_factor;
          for (int d = 0; d < ATTN_BLOCK_D; d++) {
            s_row_acc[m][d] *= scale_factor;
          }
        }
        s_row_max[m] = new_max;
      }
      __syncwarp();

      // Compute softmax weights and accumulate V
      float row_max_val = s_row_max[m];
      float local_sum = 0.0f;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len) {
          float weight = expf(local_scores[n] - row_max_val);
          local_sum += weight;

          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            atomicAdd(&s_row_acc[m][d], weight * s_V[n][d]);
          }
        }
      }

      if (lane_id == 0) {
        s_row_sum[m] += local_sum;
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
    int global_m = query_block * ATTN_BLOCK_M + m;
    if (global_m >= seq_len)
      continue;

    float inv_sum = 1.0f / s_row_sum[m];

    for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
      if (d < head_dim) {
        out[out_offset + m * head_dim + d] = s_row_acc[m][d] * inv_sum;
      }
    }
  }
}

// =============================================================================
// Golden-Weighted Attention (E₆ Cone)
// =============================================================================

// Attention with golden ratio decay: exp(-|i-j|²/φ)
extern "C" __global__ void
flash_attention_golden_f32(float *__restrict__ out, const float *__restrict__ Q,
                           const float *__restrict__ K,
                           const float *__restrict__ V, int batch, int heads,
                           int seq_len, int head_dim, float scale) {
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;

  // Shared memory tiles
  __shared__ float s_Q[ATTN_BLOCK_M][ATTN_BLOCK_D];
  __shared__ float s_K[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_V[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];

  // Load Q tile
  for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
    for (int m = 0; m < ATTN_BLOCK_M; m += 1) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      int q_offset =
          ((batch_idx * heads + head_idx) * seq_len + global_m) * head_dim;
      s_Q[m][d] = (global_m < seq_len) ? Q[q_offset + d] : 0.0f;
    }
  }

  __syncthreads();

  // For each KV block
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K and V
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      for (int n = 0; n < ATTN_BLOCK_N; n += 1) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        int kv_offset =
            ((batch_idx * heads + head_idx) * seq_len + global_n) * head_dim;
        s_K[n][d] = (global_n < seq_len) ? K[kv_offset + d] : 0.0f;
        s_V[n][d] = (global_n < seq_len) ? V[kv_offset + d] : 0.0f;
      }
    }

    __syncthreads();

    // Compute attention with golden weighting
    for (int m = tid / 32; m < ATTN_BLOCK_M; m += blockDim.x / 32) {
      int global_m = query_block * ATTN_BLOCK_M + m;

      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n >= seq_len)
          continue;

        // Compute Q·K^T
        float score = 0.0f;
        for (int d = tid % 32; d < ATTN_BLOCK_D; d += 32) {
          score += s_Q[m][d] * s_K[n][d];
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset >>= 1) {
          score += __shfl_down_sync(0xffffffff, score, offset);
        }

        score *= scale;

        // Apply golden weighting: exp(-|i-j|²/φ)
        int pos_diff = abs(global_m - global_n);
        float golden_weight = expf(-pos_diff * pos_diff / PHI_F32);
        score *= golden_weight;

        // Online softmax per row
        if (tid % 32 == 0) {
          s_row_max[m] = -INFINITY;
          s_row_sum[m] = 0.0f;
        }
        __syncthreads();

        // Find row max
        atomicMax((int *)&s_row_max[m], __float_as_int(score));
        __syncthreads();

        // Compute exp and sum
        if (tid % 32 == n % 32) {
          float exp_score = expf(score - s_row_max[m]);
          atomicAdd(&s_row_sum[m], exp_score);
        }
        __syncthreads();

        // Normalize and accumulate
        float attn_weight = expf(score - s_row_max[m]) / s_row_sum[m];

        for (int d = tid % 32; d < ATTN_BLOCK_D; d += 32) {
          float contrib = attn_weight * s_V[n][d];
          atomicAdd(&out[((batch_idx * heads + head_idx) * seq_len + global_m) *
                             head_dim +
                         d],
                    contrib);
        }
      }
    }

    __syncthreads();
  }
}

// =============================================================================
// Mersenne-Stable Attention (M₇ = 127)
// =============================================================================

// Attention optimized for head_dim = 127 (Mersenne prime M₇)
extern "C" __global__ void flash_attention_mersenne_127_f32(
    float *__restrict__ out, const float *__restrict__ Q,
    const float *__restrict__ K, const float *__restrict__ V, int batch,
    int heads, int seq_len, float scale) {
  const int HEAD_DIM = 127; // Mersenne prime M₇

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = blockDim.x / 32;

  // Shared memory for tiles (127 is prime, special handling)
  __shared__ float s_Q[ATTN_BLOCK_M][HEAD_DIM];
  __shared__ float s_K[ATTN_BLOCK_N][HEAD_DIM];
  __shared__ float s_V[ATTN_BLOCK_N][HEAD_DIM];

  // Block-level running statistics for online softmax
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];
  __shared__ float s_row_acc[ATTN_BLOCK_M][HEAD_DIM];

  // Global memory offsets
  int q_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      HEAD_DIM;
  int kv_offset = ((batch_idx * heads + head_idx) * seq_len) * HEAD_DIM;
  int out_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      HEAD_DIM;

  // Initialize shared memory statistics
  if (tid < ATTN_BLOCK_M) {
    s_row_max[tid] = -INFINITY;
    s_row_sum[tid] = 0.0f;
  }
  // Initialize s_row_acc (127 dims)
  for (int m_idx = 0; m_idx < ATTN_BLOCK_M; m_idx++) {
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
      s_row_acc[m_idx][d] = 0.0f;
    }
  }

  // Load Q tile (127-wide loads)
  for (int m = 0; m < ATTN_BLOCK_M; m++) {
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m < seq_len) {
        s_Q[m][d] = Q[q_offset + m * HEAD_DIM + d];
      } else {
        s_Q[m][d] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Loop over K/V blocks
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K and V tiles
    for (int n = 0; n < ATTN_BLOCK_N; n++) {
      for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len) {
          s_K[n][d] = K[kv_offset + global_n * HEAD_DIM + d];
          s_V[n][d] = V[kv_offset + global_n * HEAD_DIM + d];
        } else {
          s_K[n][d] = 0.0f;
          s_V[n][d] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Each warp handles one query row
    for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m >= seq_len)
        continue;

      float local_scores[ATTN_BLOCK_N];
      float tile_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;

        // Compute Q·K^T with 127-wide reduction
        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
          score += s_Q[m][d] * s_K[n][d];
        }
        score = warp_reduce_sum(score);

        if (global_n < seq_len) {
          local_scores[n] = score * scale;
          tile_max = fmaxf(tile_max, local_scores[n]);
        } else {
          local_scores[n] = -INFINITY;
        }
      }

      // Update running max across tiles
      if (lane_id == 0) {
        float old_max = s_row_max[m];
        float new_max = fmaxf(old_max, tile_max);

        if (new_max > old_max) {
          float scale_factor = expf(old_max - new_max);
          s_row_sum[m] *= scale_factor;
          for (int d = 0; d < HEAD_DIM; d++) {
            s_row_acc[m][d] *= scale_factor;
          }
        }
        s_row_max[m] = new_max;
      }
      __syncwarp();

      // Compute softmax weights and accumulate V
      float row_max_val = s_row_max[m];
      float local_sum = 0.0f;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len) {
          float weight = expf(local_scores[n] - row_max_val);
          local_sum += weight;

          for (int d = lane_id; d < HEAD_DIM; d += 32) {
            atomicAdd(&s_row_acc[m][d], weight * s_V[n][d]);
          }
        }
      }

      if (lane_id == 0) {
        s_row_sum[m] += local_sum;
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
    int global_m = query_block * ATTN_BLOCK_M + m;
    if (global_m >= seq_len)
      continue;

    float inv_sum = 1.0f / s_row_sum[m];

    for (int d = lane_id; d < HEAD_DIM; d += 32) {
      out[out_offset + m * HEAD_DIM + d] = s_row_acc[m][d] * inv_sum;
    }
  }
}

// =============================================================================
// Causal Attention (Forward Arrow)
// =============================================================================

// Standard causal (masked) attention: forward arrow of time
extern "C" __global__ void
flash_attention_causal_f32(float *__restrict__ out, const float *__restrict__ Q,
                           const float *__restrict__ K,
                           const float *__restrict__ V, int batch, int heads,
                           int seq_len, int head_dim, float scale) {
  // Causal masking: for query position i, only attend to j <= i
  // Implements temporal precedence in attention mechanism

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = blockDim.x / 32;

  // Shared memory for Q, K, V tiles
  __shared__ float s_Q[ATTN_BLOCK_M][ATTN_BLOCK_D];
  __shared__ float s_K[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_V[ATTN_BLOCK_N][ATTN_BLOCK_D];

  // Block-level running statistics for online softmax
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];
  __shared__ float s_row_acc[ATTN_BLOCK_M][ATTN_BLOCK_D];

  // Global memory offsets
  int q_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;
  int kv_offset = ((batch_idx * heads + head_idx) * seq_len) * head_dim;
  int out_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;

  // Initialize shared memory statistics
  if (tid < ATTN_BLOCK_M) {
    s_row_max[tid] = -INFINITY;
    s_row_sum[tid] = 0.0f;
  }
  for (int m_idx = 0; m_idx < ATTN_BLOCK_M; m_idx++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      s_row_acc[m_idx][d] = 0.0f;
    }
  }

  // Load Q tile into shared memory
  for (int m = 0; m < ATTN_BLOCK_M; m++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m < seq_len && d < head_dim) {
        s_Q[m][d] = Q[q_offset + m * head_dim + d];
      } else {
        s_Q[m][d] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Loop over K/V blocks (with causal constraints)
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K and V tiles
    for (int n = 0; n < ATTN_BLOCK_N; n++) {
      for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len && d < head_dim) {
          s_K[n][d] = K[kv_offset + global_n * head_dim + d];
          s_V[n][d] = V[kv_offset + global_n * head_dim + d];
        } else {
          s_K[n][d] = 0.0f;
          s_V[n][d] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Each warp handles one query row
    for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m >= seq_len)
        continue;

      float local_scores[ATTN_BLOCK_N];
      float tile_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;

        // Compute Q·K^T with causal masking
        float score = 0.0f;
        if (global_n <= global_m && global_n < seq_len) { // Causal: j <= i
          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            score += s_Q[m][d] * s_K[n][d];
          }
          score = warp_reduce_sum(score);
          local_scores[n] = score * scale;
          tile_max = fmaxf(tile_max, local_scores[n]);
        } else {
          local_scores[n] = -INFINITY; // Mask future positions
        }
      }

      // Update running max across tiles
      if (lane_id == 0) {
        float old_max = s_row_max[m];
        float new_max = fmaxf(old_max, tile_max);

        if (new_max > old_max && new_max != -INFINITY) {
          float scale_factor = expf(old_max - new_max);
          s_row_sum[m] *= scale_factor;
          for (int d = 0; d < ATTN_BLOCK_D; d++) {
            s_row_acc[m][d] *= scale_factor;
          }
        }
        s_row_max[m] = new_max;
      }
      __syncwarp();

      // Compute softmax weights and accumulate V
      float row_max_val = s_row_max[m];
      float local_sum = 0.0f;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n <= global_m && global_n < seq_len) { // Causal
          float weight = expf(local_scores[n] - row_max_val);
          local_sum += weight;

          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            atomicAdd(&s_row_acc[m][d], weight * s_V[n][d]);
          }
        }
      }

      if (lane_id == 0) {
        s_row_sum[m] += local_sum;
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
    int global_m = query_block * ATTN_BLOCK_M + m;
    if (global_m >= seq_len)
      continue;

    float inv_sum = (s_row_sum[m] > 0.0f) ? (1.0f / s_row_sum[m]) : 0.0f;

    for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
      if (d < head_dim) {
        out[out_offset + m * head_dim + d] = s_row_acc[m][d] * inv_sum;
      }
    }
  }
}

// =============================================================================
// Retrocausal Attention (Lucas Shadow)
// =============================================================================

// Retrocausal attention with Lucas shadow weighting
extern "C" __global__ void flash_attention_retrocausal_f32(
    float *__restrict__ out, const float *__restrict__ Q,
    const float *__restrict__ K, const float *__restrict__ V, int batch,
    int heads, int seq_len, int head_dim, float scale) {
  // Retrocausal: future can influence present with Lucas shadow weighting
  // For j > i (future positions), apply weight: (1-φ)^{j-i} = PHI_INV^{j-i}
  // Implements temporal crystallization in reverse direction

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int query_block = blockIdx.x;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = blockDim.x / 32;

  // Shared memory for Q, K, V tiles
  __shared__ float s_Q[ATTN_BLOCK_M][ATTN_BLOCK_D];
  __shared__ float s_K[ATTN_BLOCK_N][ATTN_BLOCK_D];
  __shared__ float s_V[ATTN_BLOCK_N][ATTN_BLOCK_D];

  // Block-level running statistics for online softmax
  __shared__ float s_row_max[ATTN_BLOCK_M];
  __shared__ float s_row_sum[ATTN_BLOCK_M];
  __shared__ float s_row_acc[ATTN_BLOCK_M][ATTN_BLOCK_D];

  // Global memory offsets
  int q_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;
  int kv_offset = ((batch_idx * heads + head_idx) * seq_len) * head_dim;
  int out_offset =
      ((batch_idx * heads + head_idx) * seq_len + query_block * ATTN_BLOCK_M) *
      head_dim;

  // Initialize shared memory statistics
  if (tid < ATTN_BLOCK_M) {
    s_row_max[tid] = -INFINITY;
    s_row_sum[tid] = 0.0f;
  }
  for (int m_idx = 0; m_idx < ATTN_BLOCK_M; m_idx++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      s_row_acc[m_idx][d] = 0.0f;
    }
  }

  // Load Q tile into shared memory
  for (int m = 0; m < ATTN_BLOCK_M; m++) {
    for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m < seq_len && d < head_dim) {
        s_Q[m][d] = Q[q_offset + m * head_dim + d];
      } else {
        s_Q[m][d] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Loop over K/V blocks
  for (int kv_block = 0; kv_block < (seq_len + ATTN_BLOCK_N - 1) / ATTN_BLOCK_N;
       kv_block++) {
    // Load K and V tiles
    for (int n = 0; n < ATTN_BLOCK_N; n++) {
      for (int d = tid; d < ATTN_BLOCK_D; d += blockDim.x) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len && d < head_dim) {
          s_K[n][d] = K[kv_offset + global_n * head_dim + d];
          s_V[n][d] = V[kv_offset + global_n * head_dim + d];
        } else {
          s_K[n][d] = 0.0f;
          s_V[n][d] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Each warp handles one query row
    for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
      int global_m = query_block * ATTN_BLOCK_M + m;
      if (global_m >= seq_len)
        continue;

      float local_scores[ATTN_BLOCK_N];
      float tile_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;

        // Compute Q·K^T
        float score = 0.0f;
        if (global_n < seq_len) {
          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            score += s_Q[m][d] * s_K[n][d];
          }
          score = warp_reduce_sum(score);
          score *= scale;

          // Apply Lucas shadow weighting for future (retrocausal) positions
          if (global_n > global_m) {
            // Future position: apply (1-φ)^{j-i} ≈ PHI_INV^{j-i} decay (Lucas
            // shadow)
            int time_diff = global_n - global_m;
            // Note: 1-φ = -φ_hat = -(φ-1) = -PHI_INV, we use PHI_INV for decay
            float lucas_weight = powf(PHI_INV_F32, (float)time_diff);
            score *= lucas_weight;
          }
          // Past positions (j <= i) use standard weighting (weight = 1.0)

          local_scores[n] = score;
          tile_max = fmaxf(tile_max, local_scores[n]);
        } else {
          local_scores[n] = -INFINITY;
        }
      }

      // Update running max across tiles
      if (lane_id == 0) {
        float old_max = s_row_max[m];
        float new_max = fmaxf(old_max, tile_max);

        if (new_max > old_max) {
          float scale_factor = expf(old_max - new_max);
          s_row_sum[m] *= scale_factor;
          for (int d = 0; d < ATTN_BLOCK_D; d++) {
            s_row_acc[m][d] *= scale_factor;
          }
        }
        s_row_max[m] = new_max;
      }
      __syncwarp();

      // Compute softmax weights and accumulate V
      float row_max_val = s_row_max[m];
      float local_sum = 0.0f;

#pragma unroll
      for (int n = 0; n < ATTN_BLOCK_N; n++) {
        int global_n = kv_block * ATTN_BLOCK_N + n;
        if (global_n < seq_len) {
          float weight = expf(local_scores[n] - row_max_val);
          local_sum += weight;

          for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
            atomicAdd(&s_row_acc[m][d], weight * s_V[n][d]);
          }
        }
      }

      if (lane_id == 0) {
        s_row_sum[m] += local_sum;
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  for (int m = warp_id; m < ATTN_BLOCK_M; m += num_warps) {
    int global_m = query_block * ATTN_BLOCK_M + m;
    if (global_m >= seq_len)
      continue;

    float inv_sum = 1.0f / s_row_sum[m];

    for (int d = lane_id; d < ATTN_BLOCK_D; d += 32) {
      if (d < head_dim) {
        out[out_offset + m * head_dim + d] = s_row_acc[m][d] * inv_sum;
      }
    }
  }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// Standard flash attention
void launch_flash_attention_f32(cudaStream_t stream, float *out, const float *Q,
                                const float *K, const float *V, int batch,
                                int heads, int seq_len, int head_dim,
                                float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_f32<<<grid, block, 0, stream>>>(out, Q, K, V, batch, heads,
                                                  seq_len, head_dim, scale);
}

// SRT syntony-focused attention
void launch_flash_attention_syntony_f32(cudaStream_t stream, float *out,
                                        const float *Q, const float *K,
                                        const float *V, const float *syntony,
                                        int batch, int heads, int seq_len,
                                        int head_dim, float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_syntony_f32<<<grid, block, 0, stream>>>(
      out, Q, K, V, syntony, batch, heads, seq_len, head_dim, scale);
}

// Golden-weighted attention
void launch_flash_attention_golden_f32(cudaStream_t stream, float *out,
                                       const float *Q, const float *K,
                                       const float *V, int batch, int heads,
                                       int seq_len, int head_dim, float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_golden_f32<<<grid, block, 0, stream>>>(
      out, Q, K, V, batch, heads, seq_len, head_dim, scale);
}

// Causal attention
void launch_flash_attention_causal_f32(cudaStream_t stream, float *out,
                                       const float *Q, const float *K,
                                       const float *V, int batch, int heads,
                                       int seq_len, int head_dim, float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_causal_f32<<<grid, block, 0, stream>>>(
      out, Q, K, V, batch, heads, seq_len, head_dim, scale);
}

// Retrocausal attention
void launch_flash_attention_retrocausal_f32(cudaStream_t stream, float *out,
                                            const float *Q, const float *K,
                                            const float *V, int batch,
                                            int heads, int seq_len,
                                            int head_dim, float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_retrocausal_f32<<<grid, block, 0, stream>>>(
      out, Q, K, V, batch, heads, seq_len, head_dim, scale);
}

// Mersenne-stable attention (head_dim = 127)
void launch_flash_attention_mersenne_127_f32(cudaStream_t stream, float *out,
                                             const float *Q, const float *K,
                                             const float *V, int batch,
                                             int heads, int seq_len,
                                             float scale) {
  dim3 block(256);
  dim3 grid((seq_len + ATTN_BLOCK_M - 1) / ATTN_BLOCK_M, heads, batch);
  flash_attention_mersenne_127_f32<<<grid, block, 0, stream>>>(
      out, Q, K, V, batch, heads, seq_len, scale);
}
}
