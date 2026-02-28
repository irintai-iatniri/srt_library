/**
 * SRT Host Wrapper Functions
 *
 * C-callable host functions that launch CUDA kernels.
 * These are called from Rust via FFI.
 */

#include "srt_constants.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Calculate grid size for n elements with given block size
inline int grid_size(int n, int block_size) {
  return (n + block_size - 1) / block_size;
}

// =============================================================================
// Kernel Forward Declarations (defined in other .cu files)
// =============================================================================

// From dhsr.cu
extern "C" __global__ void compute_syntony_f32(float *, float *, const float *,
                                               const float *, const float *,
                                               int);
extern "C" __global__ void
compute_syntony_c128(double *, double *, const double *, const double *, int);
extern "C" __global__ void differentiation_f64(double *, const double *,
                                               const double *, double, int);
extern "C" __global__ void harmonization_f64(double *, const double *,
                                             const double *, double, int);
extern "C" __global__ void dhsr_cycle_f64(double *, const double *, double,
                                          double, int);
extern "C" __global__ void laplacian_1d_f64(double *, const double *, int);
extern "C" __global__ void fourier_project_batch_f64(double *, const double *,
                                                     const int *,
                                                     const double *, int, int);
extern "C" __global__ void damping_cascade_f64(double *, const double *, double,
                                               double, int, int, double);
extern "C" __global__ void syntony_projection_f64(double *, const double *,
                                                  const double *, double, int);
extern "C" __global__ void dhsr_step_fused_f64(double *, double *,
                                               const double *, const double *,
                                               double, double, double, int);

// From gnosis.cu
extern "C" __global__ void is_conscious_kernel(const double *, int *, int);
extern "C" __global__ void gnosis_score_kernel(const double *, const double *,
                                               double *, int);
extern "C" __global__ void gnosis_mask_f64(double *, const double *,
                                           const double *, int, double, double);

// From attractor.cu
extern "C" __global__ void attractor_memory_update_f64(double *, const double *,
                                                       const double *, double,
                                                       double, int, int);
extern "C" __global__ void retrocausal_harmonize_f64(double *, const double *,
                                                     const double *, double,
                                                     double, int);
extern "C" __global__ void attractor_centroid_f64(double *, const double *,
                                                  const double *, int, int);

// From golden_ops.cu
extern "C" __global__ void scale_phi_f64(double *, const double *, int);

// From elementwise.cu
extern "C" __global__ void sin_toroidal_f64(double *, const double *, int);
extern "C" __global__ void cos_toroidal_f64(double *, const double *, int);
extern "C" __global__ void atan2_toroidal_f64(double *, const double *,
                                              const double *, int);
extern "C" __global__ void phi_exp_f64(double *, const double *, int);
extern "C" __global__ void phi_exp_inv_f64(double *, const double *, int);
extern "C" __global__ void golden_entropy_f64(double *, const double *, int);

// From matmul.cu
extern "C" __global__ void matmul_f64(double *, const double *, const double *,
                                      int, int, int);
extern "C" __global__ void matmul_tiled_f64(double *, const double *,
                                            const double *, int, int, int);

// From scatter_gather_srt.cu
extern "C" __global__ void gather_f64(double *, const double *, const int *,
                                      int);
extern "C" __global__ void scatter_add_f32(float *, const float *, const int *,
                                           int);
extern "C" __global__ void gather_phi_weighted_f64(double *, const double *,
                                                   const int *, const double *,
                                                   int);

// From trilinear.cu
extern "C" __global__ void trilinear_f64(double *, const double *,
                                         const double *, const double *, int,
                                         int, int, int);

// From complex_ops.cu
extern "C" __global__ void arg_c128(double *, const double *, int);
extern "C" __global__ void phase_syntony_c128(double *, const double *, int);

// =============================================================================
// Host Wrapper Functions (called from Rust FFI)
// =============================================================================

extern "C" {

// --- DHSR Operations ---

int host_compute_syntony_c128(double *numerator, double *denominator,
                              const double *psi, const double *mode_norm_sq,
                              int n) {
  int block = 256;
  int grid = grid_size(n, block);
  size_t shared = 2 * block * sizeof(double);

  // Allocate device buffers and copy host inputs to device.
  double *d_numerator = NULL;
  double *d_denominator = NULL;
  double *d_psi = NULL;
  double *d_mode = NULL;

  CUDA_CHECK(cudaMalloc((void **)&d_numerator, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_denominator, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_psi, sizeof(double) * 2 * n));
  CUDA_CHECK(cudaMalloc((void **)&d_mode, sizeof(double) * n));

  // Initialize outputs on device
  CUDA_CHECK(cudaMemset(d_numerator, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_denominator, 0, sizeof(double)));

  // Copy inputs
  CUDA_CHECK(
      cudaMemcpy(d_psi, psi, sizeof(double) * 2 * n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mode, mode_norm_sq, sizeof(double) * n,
                        cudaMemcpyHostToDevice));

  // Launch kernel with device pointers
  compute_syntony_c128<<<grid, block, shared>>>(d_numerator, d_denominator,
                                                d_psi, d_mode, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back to host
  CUDA_CHECK(cudaMemcpy(numerator, d_numerator, sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(denominator, d_denominator, sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_numerator));
  CUDA_CHECK(cudaFree(d_denominator));
  CUDA_CHECK(cudaFree(d_psi));
  CUDA_CHECK(cudaFree(d_mode));

  return 0;
}

int host_laplacian_1d_f64(double *out, const double *in_data, int size) {
  int block = 256;
  int grid = grid_size(size, block);

  laplacian_1d_f64<<<grid, block>>>(out, in_data, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_fourier_project_batch_f64(double *out, const double *in_data,
                                   const int *modes, const double *weights,
                                   int num_modes, int size) {
  int block = 256;
  int grid = grid_size(size, block);

  fourier_project_batch_f64<<<grid, block>>>(out, in_data, modes, weights,
                                             num_modes, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_damping_cascade_f64(double *out, const double *in_data, double gamma_0,
                             double delta_d, int num_dampers, int size,
                             double phi_weight) {
  int block = 256;
  int grid = grid_size(size, block);

  damping_cascade_f64<<<grid, block>>>(out, in_data, gamma_0, delta_d,
                                       num_dampers, size, phi_weight);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_syntony_projection_f64(double *out, const double *in_data,
                                const double *target_syntony, double strength,
                                int size) {
  int block = 256;
  int grid = grid_size(size, block);

  syntony_projection_f64<<<grid, block>>>(out, in_data, target_syntony,
                                          strength, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_dhsr_step_fused_f64(double *state, double *syntony_out,
                             const double *mode_norms, const double *attractors,
                             double diff_strength, double harm_strength,
                             double retro_pull, int size) {
  int block = 256;
  int grid = grid_size(size, block);

  dhsr_step_fused_f64<<<grid, block>>>(state, syntony_out, mode_norms,
                                       attractors, diff_strength, harm_strength,
                                       retro_pull, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Attractor Operations ---

int host_attractor_memory_update_f64(double *attractors,
                                     const double *new_state,
                                     const double *syntony_scores,
                                     double min_syntony, double decay_rate,
                                     int state_size, int max_attractors) {
  int block = 256;
  int grid = grid_size(state_size, block);

  attractor_memory_update_f64<<<grid, block>>>(
      attractors, new_state, syntony_scores, min_syntony, decay_rate,
      state_size, max_attractors);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_retrocausal_harmonize_f64(double *out, const double *current,
                                   const double *attractor_centroid,
                                   double pull_strength,
                                   double syntony_threshold, int size) {
  int block = 256;
  int grid = grid_size(size, block);

  retrocausal_harmonize_f64<<<grid, block>>>(
      out, current, attractor_centroid, pull_strength, syntony_threshold, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_attractor_centroid_f64(double *centroid, const double *attractors,
                                const double *weights, int num_attractors,
                                int state_size) {
  int block = 256;
  int grid = grid_size(state_size, block);

  attractor_centroid_f64<<<grid, block>>>(centroid, attractors, weights,
                                          num_attractors, state_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Golden Operations ---

// =============================================================================
// DEVICE-POINTER WRAPPERS (dev_*)
// =============================================================================
// These functions REQUIRE device pointers (GPU memory addresses).
// Callers must:
//   1. Allocate GPU memory (cudaMalloc or pool allocation)
//   2. Copy input data H2D (cudaMemcpy or stream.clone_htod)
//   3. Call dev_* with device pointers
//   4. Copy output data D2H (cudaMemcpy or stream.memcpy_dtoh)
//
// The old host_* functions are deprecated - they assumed device pointers
// but callers often passed host pointers, causing illegal memory access.
// =============================================================================

extern "C" int dev_sin_toroidal_f64(double *out_dev, const double *in_dev,
                                    int n) {
  int block = 256;
  int grid = grid_size(n, block);

  sin_toroidal_f64<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_cos_toroidal_f64(double *out_dev, const double *in_dev,
                                    int n) {
  int block = 256;
  int grid = grid_size(n, block);

  cos_toroidal_f64<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_phi_exp_f64(double *out_dev, const double *in_dev, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  phi_exp_f64<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_phi_exp_inv_f64(double *out_dev, const double *in_dev,
                                   int n) {
  int block = 256;
  int grid = grid_size(n, block);

  phi_exp_inv_f64<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_atan2_toroidal_f64(double *out_dev, const double *y_dev,
                                      const double *x_dev, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  atan2_toroidal_f64<<<grid, block>>>(out_dev, y_dev, x_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_golden_entropy_f64(double *out_dev, const double *in_dev,
                                      int n) {
  int block = 256;
  int grid = grid_size(n, block);

  golden_entropy_f64<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_gather_f64(double *out_dev, const double *src_dev,
                              const int *indices_dev, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  gather_f64<<<grid, block>>>(out_dev, src_dev, indices_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_arg_c128(double *out_dev, const double *in_dev, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  arg_c128<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int dev_phase_syntony_c128(double *out_dev, const double *in_dev,
                                      int n) {
  int block = 256;
  int grid = grid_size(n, block);

  phase_syntony_c128<<<grid, block>>>(out_dev, in_dev, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// =============================================================================
// HOST WRAPPERS (host_*) - Fixed to properly handle host memory
// These functions accept HOST pointers. They:
//   1. Allocate device memory
//   2. Copy input H2D
//   3. Launch kernel with device pointers
//   4. Copy output D2H
//   5. Free device memory
// =============================================================================

int host_sin_toroidal_f64(double *out, const double *in_data, int n) {
  double *d_out = NULL, *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(double) * n));
  CUDA_CHECK(
      cudaMemcpy(d_in, in_data, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  sin_toroidal_f64<<<grid, block>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
  return 0;
}

int host_cos_toroidal_f64(double *out, const double *in_data, int n) {
  double *d_out = NULL, *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(double) * n));
  CUDA_CHECK(
      cudaMemcpy(d_in, in_data, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  cos_toroidal_f64<<<grid, block>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
  return 0;
}

int host_atan2_toroidal_f64(double *out, const double *y, const double *x,
                            int n) {
  double *d_out = NULL, *d_y = NULL, *d_x = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(double) * n));
  CUDA_CHECK(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  atan2_toroidal_f64<<<grid, block>>>(d_out, d_y, d_x, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_x));
  return 0;
}

int host_phi_exp_f64(double *out, const double *in_data, int n) {
  double *d_out = NULL, *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(double) * n));
  CUDA_CHECK(
      cudaMemcpy(d_in, in_data, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  phi_exp_f64<<<grid, block>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
  return 0;
}

int host_phi_exp_inv_f64(double *out, const double *in_data, int n) {
  double *d_out = NULL, *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(double) * n));
  CUDA_CHECK(
      cudaMemcpy(d_in, in_data, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  phi_exp_inv_f64<<<grid, block>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
  return 0;
}

int host_golden_entropy_f64(double *out, const double *in_data, int n) {
  double *d_out = NULL, *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(double) * n));
  CUDA_CHECK(
      cudaMemcpy(d_in, in_data, sizeof(double) * n, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = grid_size(n, block);
  golden_entropy_f64<<<grid, block>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(out, d_out, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
  return 0;
}

// --- Scatter/Gather Operations ---

int host_gather_f64(double *out, const double *src, const int *indices, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  gather_f64<<<grid, block>>>(out, src, indices, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_scatter_add_f32(float *out, const float *src, const int *indices,
                         int n) {
  int block = 256;
  int grid = grid_size(n, block);

  scatter_add_f32<<<grid, block>>>(out, src, indices, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_gather_phi_weighted_f64(double *out, const double *src,
                                 const int *indices, const double *weights,
                                 int n) {
  int block = 256;
  int grid = grid_size(n, block);

  gather_phi_weighted_f64<<<grid, block>>>(out, src, indices, weights, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Complex Operations ---

int host_arg_c128(double *out, const double *in_data, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  arg_c128<<<grid, block>>>(out, in_data, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int host_phase_syntony_c128(double *out, const double *in_data, int n) {
  int block = 256;
  int grid = grid_size(n, block);

  phase_syntony_c128<<<grid, block>>>(out, in_data, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Trilinear ---

int host_trilinear_f64(double *out, const double *x, const double *y,
                       const double *z, int nx, int ny, int nz, int count) {
  int block = 256;
  int grid = grid_size(count, block);

  trilinear_f64<<<grid, block>>>(out, x, y, z, nx, ny, nz, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Gnosis Operations ---

int host_gnosis_mask_f64(double *out, const double *input,
                         const double *syntony, int size, double threshold,
                         double strength) {
  int block = 256;
  int grid = grid_size(size, block);

  gnosis_mask_f64<<<grid, block>>>(out, input, syntony, size, threshold,
                                   strength);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// --- Matrix Operations ---

int host_matmul_f64(double *c, const double *a, const double *b, int m, int n,
                    int k) {
  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (m + 15) / 16);

  matmul_f64<<<grid, block>>>(c, a, b, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// =============================================================================
// Phase-State Compiler Operations (int8 exact-integer CUDA pipeline)
// =============================================================================

// Forward declarations from phase_state_ops.cu
extern "C" __global__ void ps_inject_novelty_2d_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    int rows, int cols, int* activity_flag);

extern "C" __global__ void ps_wavefront_propagate_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    int rows, int cols);

extern "C" __global__ void ps_harmonize_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    const int8_t* is_source, int n);

extern "C" __global__ void ps_dampened_recursion_kernel(
    int8_t* m4, int8_t* t4, int* interacted,
    const int8_t* is_source, int* stale_cycles, int* recursive_depth,
    int stale_threshold, int k_threshold, int* gnosis_flag,
    int n);

/**
 * host_phase_state_run_2d
 *
 * Runs the full Phase-State compile cycle loop on GPU.
 * Data stays in VRAM between cycles — only copied back at the end.
 *
 * Parameters (all host pointers):
 *   m4, t4:            int8_t arrays of size n (real/imag components)
 *   is_source:         int8_t array of size n (1 = source node)
 *   stale_cycles:      int32 array of size n (per-node staleness)
 *   recursive_depth:   int32 array of size n (per-node recursion depth)
 *   rows, cols:        grid dimensions (n = rows * cols)
 *   stale_threshold:   cycles before recursion triggers
 *   k_threshold:       K=24 kissing number for gnosis
 *   allow_novelty:     1 = enable novelty injection, 0 = disable
 *   max_cycles:        maximum number of compile cycles
 *   out_gnosis:        output: 1 if gnosis reached, 0 otherwise
 *   out_cycles_used:   output: number of cycles actually run
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int host_phase_state_run_2d(
    int8_t* m4, int8_t* t4,
    const int8_t* is_source,
    int* stale_cycles, int* recursive_depth,
    int rows, int cols,
    int stale_threshold, int k_threshold,
    int allow_novelty, int max_cycles,
    int* out_gnosis, int* out_cycles_used
) {
    int n = rows * cols;
    int block = 256;
    int grd = grid_size(n, block);

    /* Device buffers */
    int8_t *d_m4 = NULL, *d_t4 = NULL, *d_is_source = NULL;
    int *d_interacted = NULL, *d_stale = NULL, *d_depth = NULL;
    int *d_gnosis_flag = NULL, *d_activity_flag = NULL;

    /* Allocate */
    CUDA_CHECK(cudaMalloc((void**)&d_m4, sizeof(int8_t) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_t4, sizeof(int8_t) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_is_source, sizeof(int8_t) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_interacted, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_stale, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_depth, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_gnosis_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_activity_flag, sizeof(int)));

    /* Copy H2D */
    CUDA_CHECK(cudaMemcpy(d_m4, m4, sizeof(int8_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t4, t4, sizeof(int8_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_source, is_source, sizeof(int8_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stale, stale_cycles, sizeof(int) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, recursive_depth, sizeof(int) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_gnosis_flag, 0, sizeof(int)));

    int gnosis_host = 0;
    int cycle;

    for (cycle = 0; cycle < max_cycles; cycle++) {
        /* Reset per-cycle state */
        CUDA_CHECK(cudaMemset(d_interacted, 0, sizeof(int) * n));
        CUDA_CHECK(cudaMemset(d_activity_flag, 0, sizeof(int)));

        /* 1. Novelty injection (if enabled) */
        if (allow_novelty) {
            ps_inject_novelty_2d_kernel<<<grd, block>>>(
                d_m4, d_t4, d_interacted, rows, cols, d_activity_flag);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        /* 2. Wavefront propagation */
        ps_wavefront_propagate_kernel<<<grd, block>>>(
            d_m4, d_t4, d_interacted, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* 3. Global harmonization */
        ps_harmonize_kernel<<<grd, block>>>(
            d_m4, d_t4, d_interacted, d_is_source, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* 4. Dampened recursion */
        ps_dampened_recursion_kernel<<<grd, block>>>(
            d_m4, d_t4, d_interacted, d_is_source,
            d_stale, d_depth,
            stale_threshold, k_threshold, d_gnosis_flag, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Check gnosis flag */
        CUDA_CHECK(cudaMemcpy(&gnosis_host, d_gnosis_flag, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (gnosis_host == 1) {
            cycle++;
            break;
        }

        /* Quick syntony check: copy m4 back and see if all zeros */
        /* (Only check every 4 cycles to reduce PCI-e traffic) */
        if ((cycle + 1) % 4 == 0 || cycle == max_cycles - 1) {
            CUDA_CHECK(cudaMemcpy(m4, d_m4, sizeof(int8_t) * n,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(t4, d_t4, sizeof(int8_t) * n,
                                  cudaMemcpyDeviceToHost));
            int all_syntonic = 1;
            for (int i = 0; i < n; i++) {
                if (m4[i] != 0 || t4[i] != 0) {
                    all_syntonic = 0;
                    break;
                }
            }
            if (all_syntonic) {
                cycle++;
                break;
            }
        }
    }

    /* Copy final state D2H */
    CUDA_CHECK(cudaMemcpy(m4, d_m4, sizeof(int8_t) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(t4, d_t4, sizeof(int8_t) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stale_cycles, d_stale, sizeof(int) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(recursive_depth, d_depth, sizeof(int) * n, cudaMemcpyDeviceToHost));

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_m4));
    CUDA_CHECK(cudaFree(d_t4));
    CUDA_CHECK(cudaFree(d_is_source));
    CUDA_CHECK(cudaFree(d_interacted));
    CUDA_CHECK(cudaFree(d_stale));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_gnosis_flag));
    CUDA_CHECK(cudaFree(d_activity_flag));

    *out_gnosis = gnosis_host;
    *out_cycles_used = cycle;
    return 0;
}

} // extern "C"
