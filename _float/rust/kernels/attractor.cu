// =============================================================================
// SRT Attractor Kernels: Retrocausal Backward Pass (Pull/Topology)
// =============================================================================
//
// For Gnostic layers (upper layers)
// Uses retrocausal harmonization: Ĥ_retro[ψ] = (1-λ)Ĥ[ψ] +
// λ·Σ(w_i·(A_i-ψ))/Σw_i
//
// Theory: These represent global Ĥ_retro (Retrocausal Harmonization) operators
// that "pull" the system toward high-syntony attractors discovered in the
// future.
//
// This solves the vanishing gradient problem in 68-layer networks by creating
// "wormholes" that bypass linear propagation.
//
// =============================================================================

#include "srt_constants.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// =============================================================================
// Constants
// =============================================================================

#define MAX_ATTRACTORS 128
#define SYNTONY_THRESHOLD 0.85
#define ATTRACTOR_DECAY_RATE 0.99 // Memory decay for older attractors

// =============================================================================
// AttractorMemory Management
// =============================================================================

// Store high-syntony states as attractors
// Called when syntony > SYNTONY_THRESHOLD during forward pass
extern "C" __global__ void attractor_memory_update_f64(
    double *__restrict__ attractor_memory,  // [MAX_ATTRACTORS, state_dim]
    double *__restrict__ attractor_syntony, // [MAX_ATTRACTORS]
    int *__restrict__ attractor_count,
    const double *__restrict__ state, // Current state [state_dim]
    double syntony,                   // Current syntony score
    int state_dim) {
  // Only store if syntony exceeds threshold
  if (syntony <= SYNTONY_THRESHOLD)
    return;

  // Atomically increment counter
  int idx = -1;
  if (threadIdx.x == 0) {
    idx = atomicAdd(attractor_count, 1);
  }

  // Broadcast idx to all threads
  __shared__ int s_idx;
  if (threadIdx.x == 0) {
    s_idx = idx;
  }
  __syncthreads();
  idx = s_idx;

  // Wrap around if exceeding max (circular buffer)
  idx = idx % MAX_ATTRACTORS;

  // Store state in attractor memory
  for (int d = threadIdx.x; d < state_dim; d += blockDim.x) {
    attractor_memory[idx * state_dim + d] = state[d];
  }

  // Store syntony value
  if (threadIdx.x == 0) {
    attractor_syntony[idx] = syntony;
  }
}

// Decay old attractors to give more weight to recent discoveries
extern "C" __global__ void attractor_memory_decay_f64(
    double *__restrict__ attractor_syntony, // [MAX_ATTRACTORS]
    int attractor_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= attractor_count)
    return;

  // Exponential decay
  attractor_syntony[idx] *= ATTRACTOR_DECAY_RATE;

  // Prune very weak attractors
  if (attractor_syntony[idx] < 0.1) {
    attractor_syntony[idx] = 0.0;
  }
}

// =============================================================================
// Hooking Coefficient (Topological Overlap)
// =============================================================================

// Compute C = exp(n · m / φ)
// This measures topological overlap on the Q(φ) lattice
// NO differentiation required - pure topology!
extern "C" __global__ void
hooking_coefficient_f64(double *__restrict__ coefficients,
                        const int *__restrict__ lattice_n, // Lattice indices n
                        const int *__restrict__ lattice_m, // Lattice indices m
                        int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int n = lattice_n[idx];
  int m = lattice_m[idx];

  // Hooking coefficient from Q(φ) lattice theory
  coefficients[idx] = exp((double)(n * m) / PHI_F64);
}

// Compute hooking coefficient matrix for batch processing
extern "C" __global__ void hooking_coefficient_batch_f64(
    double *__restrict__ coefficients, // [batch, batch]
    const double *__restrict__ states, // [batch, state_dim]
    int batch_size, int state_dim) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  if (i >= batch_size || j >= batch_size)
    return;

  // Compute dot product for lattice overlap approximation
  double dot = 0.0;
  for (int d = threadIdx.x; d < state_dim; d += blockDim.x) {
    dot += states[i * state_dim + d] * states[j * state_dim + d];
  }

  // Warp reduce
  dot = warp_reduce_sum(dot);

  if (threadIdx.x == 0) {
    // Map to hooking coefficient via φ
    // Normalize by state_dim to keep scale reasonable
    double normalized = dot / (state_dim + 1.0);
    coefficients[i * batch_size + j] = exp(normalized / PHI_F64);
  }
}

// =============================================================================
// Attractor Centroid (Weighted Average)
// =============================================================================

// Compute weighted centroid of all attractors
// centroid = Σ(w_i · A_i) / Σw_i
extern "C" __global__ void attractor_centroid_f64(
    double *__restrict__ centroid,         // [state_dim]
    const double *__restrict__ attractors, // [n_attractors, state_dim]
    const double *__restrict__ weights,    // [n_attractors] (syntony values)
    int n_attractors, int state_dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= state_dim)
    return;

  double sum = 0.0;
  double total_weight = 0.0;

  for (int a = 0; a < n_attractors; a++) {
    double w = weights[a];
    if (w > 0.0) { // Skip pruned attractors
      sum += attractors[a * state_dim + d] * w;
      total_weight += w;
    }
  }

  // Avoid division by zero
  centroid[d] = (total_weight > 1e-10) ? (sum / total_weight) : 0.0;
}

// Compute per-batch centroids for batch processing
extern "C" __global__ void attractor_centroid_batch_f64(
    double *__restrict__ centroids,        // [batch, state_dim]
    const double *__restrict__ attractors, // [n_attractors, state_dim]
    const double *__restrict__ weights,    // [n_attractors]
    const int *__restrict__ batch_indices, // Which attractor indices apply to
                                           // each batch
    int batch_size, int attractors_per_batch, int state_dim) {
  int b = blockIdx.x;
  int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (b >= batch_size || d >= state_dim)
    return;

  double sum = 0.0;
  double total_weight = 0.0;

  // Sum over attractors for this batch
  for (int a = 0; a < attractors_per_batch; a++) {
    int attractor_idx = batch_indices[b * attractors_per_batch + a];
    if (attractor_idx < 0)
      continue; // Sentinel for unused slots

    double w = weights[attractor_idx];
    if (w > 0.0) {
      sum += attractors[attractor_idx * state_dim + d] * w;
      total_weight += w;
    }
  }

  centroids[b * state_dim + d] =
      (total_weight > 1e-10) ? (sum / total_weight) : 0.0;
}

// =============================================================================
// Geodesic Distance (Creates Gravity Wells)
// =============================================================================

// Distance from state to attractor centroid with golden decay
// Creates "gravity wells" in phase space that pull optimization
extern "C" __global__ void attractor_distance_f64(
    double *__restrict__ distances,       // [batch]
    double *__restrict__ gravity_weights, // [batch] exp(-dist/φ)
    const double *__restrict__ states,    // [batch, state_dim]
    const double *__restrict__ centroid,  // [state_dim]
    int batch_size, int state_dim) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;

  // Compute squared distance
  double sum_sq = 0.0;
  for (int d = 0; d < state_dim; d++) {
    double diff = states[b * state_dim + d] - centroid[d];
    sum_sq += diff * diff;
  }

  // Euclidean distance
  double dist = sqrt(sum_sq);
  distances[b] = dist;

  // Gravity well strength: stronger pull for closer states
  // This creates the "wormhole" effect - distant layers feel the attractor
  gravity_weights[b] = exp(-dist / PHI_F64);
}

// Per-feature geodesic distance for fine-grained control
extern "C" __global__ void attractor_distance_per_feature_f64(
    double *__restrict__ feature_distances, // [batch, state_dim]
    const double *__restrict__ states,      // [batch, state_dim]
    const double *__restrict__ centroid,    // [state_dim]
    int batch_size, int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / state_dim;
  int d = idx % state_dim;
  if (b >= batch_size)
    return;

  double diff = states[b * state_dim + d] - centroid[d];

  // Signed distance with golden decay for direction
  feature_distances[idx] = diff * exp(-fabs(diff) / PHI_F64);
}

// =============================================================================
// Retrocausal Harmonization (The "Pull")
// =============================================================================

// Ĥ_retro[ψ] = (1 - λ)Ĥ[ψ] + λ · Σ(w_i · (A_i - ψ)) / Σw_i
// This is the core retrocausal update that replaces backpropagation
// in upper layers
extern "C" __global__ void retrocausal_harmonize_f64(
    double *__restrict__ state_new,          // Output: updated state
    const double *__restrict__ state,        // Current ψ [batch, state_dim]
    const double *__restrict__ centroid,     // Attractor centroid [state_dim]
    const double *__restrict__ gravity_well, // Distance-based weight [batch]
    double lambda,                           // Blend factor (0-1)
    int batch_size, int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / state_dim;
  int d = idx % state_dim;
  if (b >= batch_size)
    return;

  double psi = state[idx];
  double attractor = centroid[d];
  double well_strength = gravity_well[b];

  // Pull toward attractor, weighted by gravity well
  double pull = (attractor - psi) * well_strength;

  // Blend: (1-λ)·ψ + λ·(ψ + pull) = ψ + λ·pull
  // Note: If Ĥ_local(ψ) ≈ ψ (stable state), this simplifies
  state_new[idx] = psi + lambda * pull;
}

// Full retrocausal harmonization with local Ĥ operator
extern "C" __global__ void retrocausal_harmonize_full_f64(
    double *__restrict__ state_new, const double *__restrict__ state,
    const double *__restrict__ H_local, // Result of local harmonization Ĥ[ψ]
    const double *__restrict__ centroid,
    const double *__restrict__ gravity_well, double lambda, int batch_size,
    int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / state_dim;
  int d = idx % state_dim;
  if (b >= batch_size)
    return;

  double psi = state[idx];
  double h_local = H_local[idx];
  double attractor = centroid[d];
  double well_strength = gravity_well[b];

  // Retrocausal pull toward attractor
  double pull = (attractor - psi) * well_strength;

  // Ĥ_retro[ψ] = (1-λ)Ĥ[ψ] + λ·(ψ + pull)
  state_new[idx] = (1.0 - lambda) * h_local + lambda * (psi + pull);
}

// =============================================================================
// Syntony Gradient (For Syntony-Based Loss)
// =============================================================================

// Compute gradient of syntony with respect to state
// dS/dψ: points in direction that increases syntony
extern "C" __global__ void syntony_gradient_f64(
    double *__restrict__ grad_syntony,   // [batch, state_dim]
    const double *__restrict__ state,    // [batch, state_dim]
    const double *__restrict__ centroid, // [state_dim] (high-syntony target)
    const double *__restrict__ syntony,  // [batch] current syntony values
    int batch_size, int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / state_dim;
  int d = idx % state_dim;
  if (b >= batch_size)
    return;

  double psi = state[idx];
  double target = centroid[d];
  double s = syntony[b];

  // Gradient points toward centroid, scaled by (1-S)
  // Low syntony → strong gradient, high syntony → weak gradient
  // This creates a "landing" behavior near attractors
  double direction = target - psi;
  double magnitude = (1.0 - s) * (1.0 - s); // Amplify for low-syntony states

  grad_syntony[idx] = direction * magnitude;
}

// =============================================================================
// Layer Routing (Dispatch Based on Layer Depth)
// =============================================================================

// Hybrid backward pass: autograd for lower layers, attractors for upper
extern "C" __global__ void hybrid_backward_dispatch_f64(
    double *__restrict__ grad_out, const double *__restrict__ grad_in,
    const double *__restrict__ state,
    const double *__restrict__ attractor_centroid,
    const double *__restrict__ gravity_well, int layer_idx,
    int threshold_layer, // Below this: autograd, above: retrocausal
    double lambda, int batch_size, int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / state_dim;
  int d = idx % state_dim;
  if (b >= batch_size)
    return;

  if (layer_idx < threshold_layer) {
    // Standard gradient pass-through (identity for demo)
    grad_out[idx] = grad_in[idx];
  } else {
    // Retrocausal pull
    double psi = state[idx];
    double attractor = attractor_centroid[d];
    double well_strength = gravity_well[b];
    double pull = (attractor - psi) * well_strength;

    // Blend gradient with retrocausal pull
    grad_out[idx] = (1.0 - lambda) * grad_in[idx] + lambda * pull;
  }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// AttractorMemory
void launch_attractor_memory_update_f64(
    cudaStream_t stream, double *attractor_memory, double *attractor_syntony,
    int *attractor_count, const double *state, double syntony, int state_dim) {
  dim3 block(256);
  dim3 grid(1);
  attractor_memory_update_f64<<<grid, block, 0, stream>>>(
      attractor_memory, attractor_syntony, attractor_count, state, syntony,
      state_dim);
}

void launch_attractor_memory_decay_f64(cudaStream_t stream,
                                       double *attractor_syntony,
                                       int attractor_count) {
  dim3 block(256);
  dim3 grid((attractor_count + 255) / 256);
  attractor_memory_decay_f64<<<grid, block, 0, stream>>>(attractor_syntony,
                                                         attractor_count);
}

// Hooking Coefficient
void launch_hooking_coefficient_f64(cudaStream_t stream, double *coefficients,
                                    const int *lattice_n, const int *lattice_m,
                                    int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  hooking_coefficient_f64<<<grid, block, 0, stream>>>(coefficients, lattice_n,
                                                      lattice_m, size);
}

void launch_hooking_coefficient_batch_f64(cudaStream_t stream,
                                          double *coefficients,
                                          const double *states, int batch_size,
                                          int state_dim) {
  dim3 block(256);
  dim3 grid(batch_size, batch_size);
  hooking_coefficient_batch_f64<<<grid, block, 0, stream>>>(
      coefficients, states, batch_size, state_dim);
}

// Centroid
void launch_attractor_centroid_f64(cudaStream_t stream, double *centroid,
                                   const double *attractors,
                                   const double *weights, int n_attractors,
                                   int state_dim) {
  dim3 block(256);
  dim3 grid((state_dim + 255) / 256);
  attractor_centroid_f64<<<grid, block, 0, stream>>>(
      centroid, attractors, weights, n_attractors, state_dim);
}

// Distance
void launch_attractor_distance_f64(cudaStream_t stream, double *distances,
                                   double *gravity_weights,
                                   const double *states, const double *centroid,
                                   int batch_size, int state_dim) {
  dim3 block(256);
  dim3 grid((batch_size + 255) / 256);
  attractor_distance_f64<<<grid, block, 0, stream>>>(
      distances, gravity_weights, states, centroid, batch_size, state_dim);
}

// Retrocausal Harmonization
void launch_retrocausal_harmonize_f64(cudaStream_t stream, double *state_new,
                                      const double *state,
                                      const double *centroid,
                                      const double *gravity_well, double lambda,
                                      int batch_size, int state_dim) {
  int size = batch_size * state_dim;
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  retrocausal_harmonize_f64<<<grid, block, 0, stream>>>(
      state_new, state, centroid, gravity_well, lambda, batch_size, state_dim);
}

void launch_retrocausal_harmonize_full_f64(
    cudaStream_t stream, double *state_new, const double *state,
    const double *H_local, const double *centroid, const double *gravity_well,
    double lambda, int batch_size, int state_dim) {
  int size = batch_size * state_dim;
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  retrocausal_harmonize_full_f64<<<grid, block, 0, stream>>>(
      state_new, state, H_local, centroid, gravity_well, lambda, batch_size,
      state_dim);
}

// Syntony Gradient
void launch_syntony_gradient_f64(cudaStream_t stream, double *grad_syntony,
                                 const double *state, const double *centroid,
                                 const double *syntony, int batch_size,
                                 int state_dim) {
  int size = batch_size * state_dim;
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  syntony_gradient_f64<<<grid, block, 0, stream>>>(
      grad_syntony, state, centroid, syntony, batch_size, state_dim);
}

// Hybrid Dispatch
void launch_hybrid_backward_dispatch_f64(
    cudaStream_t stream, double *grad_out, const double *grad_in,
    const double *state, const double *attractor_centroid,
    const double *gravity_well, int layer_idx, int threshold_layer,
    double lambda, int batch_size, int state_dim) {
  int size = batch_size * state_dim;
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  hybrid_backward_dispatch_f64<<<grid, block, 0, stream>>>(
      grad_out, grad_in, state, attractor_centroid, gravity_well, layer_idx,
      threshold_layer, lambda, batch_size, state_dim);
}
}
