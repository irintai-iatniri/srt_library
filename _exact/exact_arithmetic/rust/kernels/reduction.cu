// Syntonic CUDA Kernels - Efficient Parallel Reductions
// Native GPU reductions with SRT theory-correct operations
// Supports sum, mean, max, min, norm with golden-weighted variants
//
// Design Philosophy:
// - Uses SRT-aligned block sizes from srt_constants.cuh
// - Warp-level reduction primitives for efficiency
// - Golden-weighted reductions for theory-aligned computations

#include "srt_constants.cuh"

// =============================================================================
// Block Size Configuration (SRT-Aligned)
// =============================================================================

#define REDUCTION_BLOCK_SIZE 256 // BLOCK_DEFAULT from srt_constants.cuh
#define REDUCTION_WARP_SIZE 32   // Standard warp size
#define REDUCTION_GOLDEN_BLOCK BLOCK_GOLDEN_CONE // 36 for E₆ alignment

// =============================================================================
// Warp-Level Min Reduction Primitives
// =============================================================================

__device__ __forceinline__ float warp_reduce_min(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ double warp_reduce_min_f64(double val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// =============================================================================
// Full Reduction: Sum (single output)
// =============================================================================

// Reduce entire array to single sum value
extern "C" __global__ void reduce_sum_f64(double *output, // Single output value
                                          const double *input, int n) {
  extern __shared__ double s_sum_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // Load first pass with coalesced access
  double local_sum = 0.0;
  if (i < n)
    local_sum = input[i];
  if (i + blockDim.x < n)
    local_sum += input[i + blockDim.x];

  s_sum_f64[tid] = local_sum;
  __syncthreads();

  // Block-level reduction in shared memory
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_sum_f64[tid] += s_sum_f64[tid + s];
    }
    __syncthreads();
  }

  // Warp-level reduction (no sync needed within warp)
  if (tid < 32) {
    double val = s_sum_f64[tid];
    if (blockDim.x >= 64)
      val += s_sum_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

extern "C" __global__ void reduce_sum_f32(float *output, const float *input,
                                          int n) {
  extern __shared__ float s_sum_f32[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float local_sum = 0.0f;
  if (i < n)
    local_sum = input[i];
  if (i + blockDim.x < n)
    local_sum += input[i + blockDim.x];

  s_sum_f32[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_sum_f32[tid] += s_sum_f32[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = s_sum_f32[tid];
    if (blockDim.x >= 64)
      val += s_sum_f32[tid + 32];
    val = warp_reduce_sum(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// Full Reduction: Mean
// =============================================================================

extern "C" __global__ void
reduce_mean_f64(double *output, // Single output value
                const double *input, int n) {
  extern __shared__ double s_mean_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n)
    local_sum = input[i];
  if (i + blockDim.x < n)
    local_sum += input[i + blockDim.x];

  s_mean_f64[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_mean_f64[tid] += s_mean_f64[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_mean_f64[tid];
    if (blockDim.x >= 64)
      val += s_mean_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      // Divide by n for mean
      atomicAdd(output, val / (double)n);
    }
  }
}

extern "C" __global__ void reduce_mean_f32(float *output, const float *input,
                                           int n) {
  extern __shared__ float s_mean_f32[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float local_sum = 0.0f;
  if (i < n)
    local_sum = input[i];
  if (i + blockDim.x < n)
    local_sum += input[i + blockDim.x];

  s_mean_f32[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_mean_f32[tid] += s_mean_f32[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = s_mean_f32[tid];
    if (blockDim.x >= 64)
      val += s_mean_f32[tid + 32];
    val = warp_reduce_sum(val);

    if (tid == 0) {
      atomicAdd(output, val / (float)n);
    }
  }
}

// =============================================================================
// Full Reduction: Max
// =============================================================================

extern "C" __global__ void
reduce_max_f64(double *output, // Single output value (initialized to -inf)
               const double *input, int n) {
  extern __shared__ double s_max_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_max = -INFINITY;
  if (i < n)
    local_max = input[i];
  if (i + blockDim.x < n)
    local_max = fmax(local_max, input[i + blockDim.x]);

  s_max_f64[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_max_f64[tid] = fmax(s_max_f64[tid], s_max_f64[tid + s]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_max_f64[tid];
    if (blockDim.x >= 64)
      val = fmax(val, s_max_f64[tid + 32]);
    val = warp_reduce_max_f64(val);

    if (tid == 0) {
      // Atomic max for f64 (using CAS)
      unsigned long long int *addr = (unsigned long long int *)output;
      unsigned long long int old = *addr;
      unsigned long long int assumed;
      do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(val, old_val);
        old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
      } while (assumed != old);
    }
  }
}

extern "C" __global__ void reduce_max_f32(float *output, const float *input,
                                          int n) {
  extern __shared__ float s_max_f32[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float local_max = -INFINITY;
  if (i < n)
    local_max = input[i];
  if (i + blockDim.x < n)
    local_max = fmaxf(local_max, input[i + blockDim.x]);

  s_max_f32[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_max_f32[tid] = fmaxf(s_max_f32[tid], s_max_f32[tid + s]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = s_max_f32[tid];
    if (blockDim.x >= 64)
      val = fmaxf(val, s_max_f32[tid + 32]);
    val = warp_reduce_max(val);

    if (tid == 0) {
      // Atomic max for f32 (using int comparison)
      int *addr_as_int = (int *)output;
      int old_as_int = *addr_as_int;
      int assumed;
      do {
        assumed = old_as_int;
        old_as_int =
            atomicCAS(addr_as_int, assumed,
                      __float_as_int(fmaxf(val, __int_as_float(assumed))));
      } while (assumed != old_as_int);
    }
  }
}

// =============================================================================
// Full Reduction: Min
// =============================================================================

extern "C" __global__ void
reduce_min_f64(double *output, // Single output value (initialized to +inf)
               const double *input, int n) {
  extern __shared__ double s_min_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_min = INFINITY;
  if (i < n)
    local_min = input[i];
  if (i + blockDim.x < n)
    local_min = fmin(local_min, input[i + blockDim.x]);

  s_min_f64[tid] = local_min;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_min_f64[tid] = fmin(s_min_f64[tid], s_min_f64[tid + s]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_min_f64[tid];
    if (blockDim.x >= 64)
      val = fmin(val, s_min_f64[tid + 32]);
    val = warp_reduce_min_f64(val);

    if (tid == 0) {
      // Atomic min for f64 (using CAS)
      unsigned long long int *addr = (unsigned long long int *)output;
      unsigned long long int old = *addr;
      unsigned long long int assumed;
      do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmin(val, old_val);
        old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
      } while (assumed != old);
    }
  }
}

extern "C" __global__ void reduce_min_f32(float *output, const float *input,
                                          int n) {
  extern __shared__ float s_min_f32[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float local_min = INFINITY;
  if (i < n)
    local_min = input[i];
  if (i + blockDim.x < n)
    local_min = fminf(local_min, input[i + blockDim.x]);

  s_min_f32[tid] = local_min;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_min_f32[tid] = fminf(s_min_f32[tid], s_min_f32[tid + s]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = s_min_f32[tid];
    if (blockDim.x >= 64)
      val = fminf(val, s_min_f32[tid + 32]);
    val = warp_reduce_min(val);

    if (tid == 0) {
      int *addr_as_int = (int *)output;
      int old_as_int = *addr_as_int;
      int assumed;
      do {
        assumed = old_as_int;
        old_as_int =
            atomicCAS(addr_as_int, assumed,
                      __float_as_int(fminf(val, __int_as_float(assumed))));
      } while (assumed != old_as_int);
    }
  }
}

// =============================================================================
// L2 Norm (Frobenius Norm)
// =============================================================================

extern "C" __global__ void
reduce_norm_l2_f64(double *output, // Single output (squared sum, sqrt after)
                   const double *input, int n) {
  extern __shared__ double s_norm_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n) {
    double val = input[i];
    local_sum = val * val;
  }
  if (i + blockDim.x < n) {
    double val = input[i + blockDim.x];
    local_sum += val * val;
  }

  s_norm_f64[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_norm_f64[tid] += s_norm_f64[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_norm_f64[tid];
    if (blockDim.x >= 64)
      val += s_norm_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

extern "C" __global__ void reduce_norm_l2_f32(float *output, const float *input,
                                              int n) {
  extern __shared__ float s_norm_f32[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float local_sum = 0.0f;
  if (i < n) {
    float val = input[i];
    local_sum = val * val;
  }
  if (i + blockDim.x < n) {
    float val = input[i + blockDim.x];
    local_sum += val * val;
  }

  s_norm_f32[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_norm_f32[tid] += s_norm_f32[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = s_norm_f32[tid];
    if (blockDim.x >= 64)
      val += s_norm_f32[tid + 32];
    val = warp_reduce_sum(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// SRT Theory-Aligned Reductions
// =============================================================================

// Golden-Weighted Sum: Σᵢ xᵢ × e^{-i²/φ}
// Each element weighted by golden Gaussian decay
extern "C" __global__ void
reduce_sum_golden_weighted_f64(double *output, const double *input, int n) {
  extern __shared__ double s_golden_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n) {
    double weight = exp(-(double)(i * i) * PHI_INV_F64);
    local_sum = input[i] * weight;
  }
  if (i + blockDim.x < n) {
    int idx = i + blockDim.x;
    double weight = exp(-(double)(idx * idx) * PHI_INV_F64);
    local_sum += input[idx] * weight;
  }

  s_golden_f64[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_golden_f64[tid] += s_golden_f64[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_golden_f64[tid];
    if (blockDim.x >= 64)
      val += s_golden_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// Syntony Measure: S(x) = -Σ pᵢ log(pᵢ) normalized to [0, φ]
// where pᵢ = |xᵢ|² / Σⱼ|xⱼ|²
extern "C" __global__ void
reduce_syntony_f64(double *output, // Two outputs: [0]=entropy, [1]=total_norm²
                   const double *input, int n) {
  extern __shared__ double s_syntony[];
  double *s_norm_sq = s_syntony;
  double *s_entropy = s_syntony + blockDim.x;

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // First pass: compute norm² and prepare for entropy
  double local_norm_sq = 0.0;
  if (i < n) {
    double val = input[i];
    local_norm_sq = val * val;
  }
  if (i + blockDim.x < n) {
    double val = input[i + blockDim.x];
    local_norm_sq += val * val;
  }

  s_norm_sq[tid] = local_norm_sq;
  s_entropy[tid] = 0.0; // Will compute after we know total norm
  __syncthreads();

  // Reduce norm²
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_norm_sq[tid] += s_norm_sq[tid + s];
    }
    __syncthreads();
  }

  double total_norm_sq = s_norm_sq[0];
  __syncthreads();

  // Second pass: compute entropy terms -pᵢ log(pᵢ)
  if (total_norm_sq > 1e-15) {
    double local_entropy = 0.0;
    if (i < n) {
      double val = input[i];
      double p = (val * val) / total_norm_sq;
      if (p > 1e-15) {
        local_entropy = -p * log(p);
      }
    }
    if (i + blockDim.x < n) {
      double val = input[i + blockDim.x];
      double p = (val * val) / total_norm_sq;
      if (p > 1e-15) {
        local_entropy -= p * log(p);
      }
    }
    s_entropy[tid] = local_entropy;
  }
  __syncthreads();

  // Reduce entropy
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_entropy[tid] += s_entropy[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&output[0], s_entropy[0]);
    atomicAdd(&output[1], total_norm_sq);
  }
}

// =============================================================================
// Row/Column Reductions (for matrices)
// =============================================================================

// Sum along rows: output[i] = Σⱼ input[i,j]
extern "C" __global__ void
reduce_sum_rows_f64(double *output,      // [M] output
                    const double *input, // [M, N] input
                    int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  extern __shared__ double s_row[];
  int tid = threadIdx.x;

  // Each thread sums strided elements in the row
  double local_sum = 0.0;
  for (int j = tid; j < N; j += blockDim.x) {
    local_sum += input[row * N + j];
  }

  s_row[tid] = local_sum;
  __syncthreads();

  // Block-level reduction
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_row[tid] += s_row[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_row[tid];
    if (blockDim.x >= 64)
      val += s_row[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      output[row] = val;
    }
  }
}

// Sum along columns: output[j] = Σᵢ input[i,j]
extern "C" __global__ void
reduce_sum_cols_f64(double *output,      // [N] output
                    const double *input, // [M, N] input
                    int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N)
    return;

  double sum = 0.0;
  for (int i = 0; i < M; i++) {
    sum += input[i * N + col];
  }
  output[col] = sum;
}

// =============================================================================
// φ-Scaled Reductions (SRT Theory-Correct)
// =============================================================================

// φ-Scaled Sum: Σᵢ xᵢ × φ^{-i}
// Exponential decay by golden ratio powers
extern "C" __global__ void
reduce_sum_phi_scaled_f64(double *output, const double *input, int n) {
  extern __shared__ double s_phi_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n) {
    double scale = pow(PHI_INV_F64, (double)i);
    local_sum = input[i] * scale;
  }
  if (i + blockDim.x < n) {
    int idx = i + blockDim.x;
    double scale = pow(PHI_INV_F64, (double)idx);
    local_sum += input[idx] * scale;
  }

  s_phi_f64[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_phi_f64[tid] += s_phi_f64[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_phi_f64[tid];
    if (blockDim.x >= 64)
      val += s_phi_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// Variance with Golden Target: Σᵢ (xᵢ - φ⁻¹)²
// Used for golden norm computation
extern "C" __global__ void
reduce_variance_golden_target_f64(double *output, const double *input, int n) {
  extern __shared__ double s_var_f64[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n) {
    double diff = input[i] - PHI_INV_F64;
    local_sum = diff * diff;
  }
  if (i + blockDim.x < n) {
    double diff = input[i + blockDim.x] - PHI_INV_F64;
    local_sum += diff * diff;
  }

  s_var_f64[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_var_f64[tid] += s_var_f64[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_var_f64[tid];
    if (blockDim.x >= 64)
      val += s_var_f64[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// Complex Reductions
// =============================================================================

// Complex sum (interleaved format)
extern "C" __global__ void
reduce_sum_c128(double *output,      // [real, imag] output
                const double *input, // Interleaved complex
                int n                // Number of complex elements
) {
  extern __shared__ double s_complex[];
  double *s_real = s_complex;
  double *s_imag = s_complex + blockDim.x;

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_real = 0.0;
  double local_imag = 0.0;

  if (i < n) {
    local_real = input[2 * i];
    local_imag = input[2 * i + 1];
  }
  if (i + blockDim.x < n) {
    local_real += input[2 * (i + blockDim.x)];
    local_imag += input[2 * (i + blockDim.x) + 1];
  }

  s_real[tid] = local_real;
  s_imag[tid] = local_imag;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_real[tid] += s_real[tid + s];
      s_imag[tid] += s_imag[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val_real = s_real[tid];
    double val_imag = s_imag[tid];
    if (blockDim.x >= 64) {
      val_real += s_real[tid + 32];
      val_imag += s_imag[tid + 32];
    }
    val_real = warp_reduce_sum_f64(val_real);
    val_imag = warp_reduce_sum_f64(val_imag);

    if (tid == 0) {
      atomicAdd(&output[0], val_real);
      atomicAdd(&output[1], val_imag);
    }
  }
}

// Complex norm² = Σᵢ |zᵢ|² = Σᵢ (re² + im²)
extern "C" __global__ void reduce_norm_c128(double *output, const double *input,
                                            int n) {
  extern __shared__ double s_cnorm[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double local_sum = 0.0;
  if (i < n) {
    double re = input[2 * i];
    double im = input[2 * i + 1];
    local_sum = re * re + im * im;
  }
  if (i + blockDim.x < n) {
    double re = input[2 * (i + blockDim.x)];
    double im = input[2 * (i + blockDim.x) + 1];
    local_sum += re * re + im * im;
  }

  s_cnorm[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s_cnorm[tid] += s_cnorm[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    double val = s_cnorm[tid];
    if (blockDim.x >= 64)
      val += s_cnorm[tid + 32];
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// SRT Theory-Correct Reductions (Grand Synthesis Aligned)
// =============================================================================

// =============================================================================
// Mersenne-Stable Sum: Only sum values at Mersenne-stable indices (M_p)
// =============================================================================

extern "C" __global__ void reduce_sum_mersenne_stable_f64(
    double* __restrict__ output,
    const double* __restrict__ input,
    int n
) {
  __shared__ double s_sum[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double local_sum = 0.0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    // Check if index is Mersenne-stable (from constant memory mask)
    if (i < MAX_MERSENNE && c_mersenne_mask[i]) {
      local_sum += input[i];
    }
  }

  // Warp-level reduction
  local_sum = warp_reduce_sum_f64(local_sum);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_sum[tid / REDUCTION_WARP_SIZE] = local_sum;
  }

  __syncthreads();

  // Final reduction in shared memory
  if (tid < REDUCTION_WARP_SIZE) {
    double val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_sum[tid] : 0.0;
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// Lucas Shadow Sum: Sum with Lucas shadow weighting (1-φ)^i = (-φ⁻¹)^i
// =============================================================================

extern "C" __global__ void reduce_sum_lucas_shadow_f64(
    double* __restrict__ output,
    const double* __restrict__ input,
    int n
) {
  __shared__ double s_sum[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double local_sum = 0.0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    double shadow_weight = pow(-PHI_INV_F64, (double)i);
    local_sum += input[i] * shadow_weight;
  }

  // Warp-level reduction
  local_sum = warp_reduce_sum_f64(local_sum);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_sum[tid / REDUCTION_WARP_SIZE] = local_sum;
  }

  __syncthreads();

  // Final reduction in shared memory
  if (tid < REDUCTION_WARP_SIZE) {
    double val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_sum[tid] : 0.0;
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// Syntony Deviation: |S(x) - S*| where S* = φ⁻¹ (equilibrium syntony)
// =============================================================================

extern "C" __global__ void reduce_syntony_deviation_f64(
    double* __restrict__ output,
    const double* __restrict__ input,
    int n
) {
  __shared__ double s_sum[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double local_sum = 0.0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    // Compute normalized Shannon entropy (syntony)
    double h_max = log2((double)n);
    double h = 0.0;

    // Simple approximation: assume input[i] is probability
    if (input[i] > 0.0) {
      h -= input[i] * log2(input[i]);
    }

    double syntony = 1.0 - h / h_max;
    double deviation = fabs(syntony - PHI_INV_F64);
    local_sum += deviation;
  }

  // Warp-level reduction
  local_sum = warp_reduce_sum_f64(local_sum);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_sum[tid / REDUCTION_WARP_SIZE] = local_sum;
  }

  __syncthreads();

  // Final reduction in shared memory
  if (tid < REDUCTION_WARP_SIZE) {
    double val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_sum[tid] : 0.0;
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// D₄ Consciousness Count: Count elements exceeding kissing threshold (24)
// =============================================================================

extern "C" __global__ void reduce_consciousness_count_f64(
    int* __restrict__ output,
    const double* __restrict__ input,
    int n
) {
  __shared__ int s_count[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  int local_count = 0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    if (input[i] > D4_KISSING_F64) {
      local_count++;
    }
  }

  // Warp-level reduction (sum for count)
  local_count = warp_reduce_sum(local_count);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_count[tid / REDUCTION_WARP_SIZE] = local_count;
  }

  __syncthreads();

  // Final reduction in shared memory
  if (tid < REDUCTION_WARP_SIZE) {
    int val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_count[tid] : 0;
    val = warp_reduce_sum(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// q-Corrected Sum: (1 ± q/N) × Σ xᵢ for structure correction
// =============================================================================

extern "C" __global__ void reduce_sum_q_corrected_f64(
    double* __restrict__ output,
    const double* __restrict__ input,
    int n,
    int structure_N,
    int sign  // +1 or -1 for correction direction
) {
  __shared__ double s_sum[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double local_sum = 0.0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    local_sum += input[i];
  }

  // Warp-level reduction
  local_sum = warp_reduce_sum_f64(local_sum);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_sum[tid / REDUCTION_WARP_SIZE] = local_sum;
  }

  __syncthreads();

  // Final reduction with q-correction
  if (tid < REDUCTION_WARP_SIZE) {
    double val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_sum[tid] : 0.0;
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      double correction = 1.0 + (double)sign * Q_DEFICIT_F64 / (double)structure_N;
      atomicAdd(output, val * correction);
    }
  }
}

// =============================================================================
// E₈ Root Norm: Specialized norm over 240 E₈ roots
// =============================================================================

extern "C" __global__ void reduce_e8_norm_f64(
    double* __restrict__ output,
    const double* __restrict__ input
) {
  const int N = E8_ROOTS;  // 240 roots
  __shared__ double s_sum[REDUCTION_BLOCK_SIZE / REDUCTION_WARP_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double local_sum = 0.0;
  for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
    local_sum += input[i] * input[i];  // L2 norm squared
  }

  // Warp-level reduction
  local_sum = warp_reduce_sum_f64(local_sum);

  // Store to shared memory
  if (tid % REDUCTION_WARP_SIZE == 0) {
    s_sum[tid / REDUCTION_WARP_SIZE] = local_sum;
  }

  __syncthreads();

  // Final reduction in shared memory
  if (tid < REDUCTION_WARP_SIZE) {
    double val = (tid < blockDim.x / REDUCTION_WARP_SIZE) ? s_sum[tid] : 0.0;
    val = warp_reduce_sum_f64(val);

    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// SRT theory-correct reductions
void launch_reduce_sum_mersenne_stable_f64(
    cudaStream_t stream, double* output, const double* input, int n
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_sum_mersenne_stable_f64<<<grid, block, 0, stream>>>(output, input, n);
}

void launch_reduce_sum_lucas_shadow_f64(
    cudaStream_t stream, double* output, const double* input, int n
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_sum_lucas_shadow_f64<<<grid, block, 0, stream>>>(output, input, n);
}

void launch_reduce_syntony_deviation_f64(
    cudaStream_t stream, double* output, const double* input, int n
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_syntony_deviation_f64<<<grid, block, 0, stream>>>(output, input, n);
}

void launch_reduce_consciousness_count_f64(
    cudaStream_t stream, int* output, const double* input, int n
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_consciousness_count_f64<<<grid, block, 0, stream>>>(output, input, n);
}

void launch_reduce_sum_q_corrected_f64(
    cudaStream_t stream, double* output, const double* input,
    int n, int structure_N, int sign
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_sum_q_corrected_f64<<<grid, block, 0, stream>>>(output, input, n, structure_N, sign);
}

void launch_reduce_e8_norm_f64(
    cudaStream_t stream, double* output, const double* input
) {
  dim3 block(REDUCTION_BLOCK_SIZE);
  dim3 grid((E8_ROOTS + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE);
  reduce_e8_norm_f64<<<grid, block, 0, stream>>>(output, input);
}

}
