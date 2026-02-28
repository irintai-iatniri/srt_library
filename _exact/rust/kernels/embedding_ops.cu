/**
 * CUDA Embedding Operations Kernels
 *
 * Provides GPU-accelerated embedding lookup and backward scatter.
 *
 * Key operations:
 * - embedding_lookup_kernel: Parallel token lookup
 * - embedding_backward_kernel: Scatter gradients back to embedding table
 */

#include <cuda_runtime.h>
#include <math.h>

#include "srt_constants.cuh"

/**
 * Embedding Lookup Kernel
 *
 * output[i, :] = table[indices[i], :]
 *
 * Each thread copies one element of one embedding vector.
 *
 * @param table Embedding table [vocab_size, embed_dim]
 * @param indices Token indices [num_indices]
 * @param output Output tensor [num_indices, embed_dim]
 * @param num_indices Number of indices to look up
 * @param embed_dim Embedding dimension
 * @param vocab_size Vocabulary size (for bounds checking)
 */
extern "C" __global__
void embedding_lookup_kernel(
    const float* __restrict__ table,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int num_indices,
    int embed_dim,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_indices * embed_dim;

    if (idx >= total) return;

    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    int token_id = indices[token_idx];

    // Bounds check: clamp to valid range
    if (token_id >= 0 && token_id < vocab_size) {
        output[idx] = table[token_id * embed_dim + dim_idx];
    } else {
        output[idx] = 0.0f;
    }
}

/**
 * Embedding Backward Kernel (Gradient Scatter)
 *
 * Scatters gradients back to the embedding table.
 * Uses atomicAdd for concurrent writes to same embedding row.
 *
 * grad_table[indices[i], :] += grad_output[i, :]
 *
 * @param grad_output Gradient of output [num_indices, embed_dim]
 * @param indices Token indices [num_indices]
 * @param grad_table Gradient of embedding table [vocab_size, embed_dim]
 * @param num_indices Number of indices
 * @param embed_dim Embedding dimension
 * @param vocab_size Vocabulary size
 */
extern "C" __global__
void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int* __restrict__ indices,
    float* __restrict__ grad_table,
    int num_indices,
    int embed_dim,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_indices * embed_dim;

    if (idx >= total) return;

    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    int token_id = indices[token_idx];

    if (token_id >= 0 && token_id < vocab_size) {
        atomicAdd(&grad_table[token_id * embed_dim + dim_idx], grad_output[idx]);
    }
}
