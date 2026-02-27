// =============================================================================
// SRT Autograd Kernels: Standard Backward Pass (Push/Calculus)
// =============================================================================
//
// For layers 32-37 (Probabilistic/Deterministic)
// Uses standard gradient descent: w_new = w_old - η · ∇L
//
// Theory: These represent localized Ĥ (Harmonization) operators
// that propagate "blame" backward through the network.
//
// =============================================================================

#include "srt_constants.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// =============================================================================
// Elementwise Backward Kernels
// =============================================================================

// Forward: z = x + y → Backward: dx = dz, dy = dz
extern "C" __global__ void
backward_add_f64(double *__restrict__ grad_x, double *__restrict__ grad_y,
                 const double *__restrict__ grad_output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double dz = grad_output[idx];
  grad_x[idx] = dz;
  grad_y[idx] = dz;
}

extern "C" __global__ void
backward_add_f32(float *__restrict__ grad_x, float *__restrict__ grad_y,
                 const float *__restrict__ grad_output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float dz = grad_output[idx];
  grad_x[idx] = dz;
  grad_y[idx] = dz;
}

// Forward: z = x * y → Backward: dx = dz * y, dy = dz * x
extern "C" __global__ void
backward_mul_f64(double *__restrict__ grad_x, double *__restrict__ grad_y,
                 const double *__restrict__ grad_output,
                 const double *__restrict__ x, const double *__restrict__ y,
                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double dz = grad_output[idx];
  grad_x[idx] = dz * y[idx];
  grad_y[idx] = dz * x[idx];
}

extern "C" __global__ void
backward_mul_f32(float *__restrict__ grad_x, float *__restrict__ grad_y,
                 const float *__restrict__ grad_output,
                 const float *__restrict__ x, const float *__restrict__ y,
                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float dz = grad_output[idx];
  grad_x[idx] = dz * y[idx];
  grad_y[idx] = dz * x[idx];
}

// Forward: z = exp(x) → Backward: dx = dz * exp(x) = dz * z
extern "C" __global__ void backward_exp_f64(
    double *__restrict__ grad_input, const double *__restrict__ grad_output,
    const double *__restrict__ output, // z = exp(x) saved from forward
    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = grad_output[idx] * output[idx];
}

extern "C" __global__ void
backward_exp_f32(float *__restrict__ grad_input,
                 const float *__restrict__ grad_output,
                 const float *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = grad_output[idx] * output[idx];
}

// Forward: z = log(x) → Backward: dx = dz / x
extern "C" __global__ void
backward_log_f64(double *__restrict__ grad_input,
                 const double *__restrict__ grad_output,
                 const double *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = grad_output[idx] / (input[idx] + 1e-10);
}

extern "C" __global__ void
backward_log_f32(float *__restrict__ grad_input,
                 const float *__restrict__ grad_output,
                 const float *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = grad_output[idx] / (input[idx] + 1e-6f);
}

// Forward: z = sqrt(x) → Backward: dx = dz / (2 * sqrt(x)) = dz / (2 * z)
extern "C" __global__ void
backward_sqrt_f64(double *__restrict__ grad_input,
                  const double *__restrict__ grad_output,
                  const double *__restrict__ output, // z = sqrt(x)
                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = grad_output[idx] / (2.0 * output[idx] + 1e-10);
}

// Forward: z = 1/x → Backward: dx = -dz / x²
extern "C" __global__ void
backward_reciprocal_f64(double *__restrict__ grad_input,
                        const double *__restrict__ grad_output,
                        const double *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double x = input[idx];
  grad_input[idx] = -grad_output[idx] / (x * x + 1e-10);
}

// =============================================================================
// Activation Backward Kernels
// =============================================================================

// Forward: z = max(0, x) → Backward: dx = dz if x > 0 else 0
extern "C" __global__ void
backward_relu_f64(double *__restrict__ grad_input,
                  const double *__restrict__ grad_output,
                  const double *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = input[idx] > 0.0 ? grad_output[idx] : 0.0;
}

extern "C" __global__ void
backward_relu_f32(float *__restrict__ grad_input,
                  const float *__restrict__ grad_output,
                  const float *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
}

// Forward: z = σ(x) = 1/(1+exp(-x)) → Backward: dx = dz * z * (1-z)
extern "C" __global__ void
backward_sigmoid_f64(double *__restrict__ grad_input,
                     const double *__restrict__ grad_output,
                     const double *__restrict__ output, // z = sigmoid(x)
                     int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double z = output[idx];
  grad_input[idx] = grad_output[idx] * z * (1.0 - z);
}

// Forward: z = tanh(x) → Backward: dx = dz * (1 - z²)
extern "C" __global__ void
backward_tanh_f64(double *__restrict__ grad_input,
                  const double *__restrict__ grad_output,
                  const double *__restrict__ output, // z = tanh(x)
                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double z = output[idx];
  grad_input[idx] = grad_output[idx] * (1.0 - z * z);
}

// Forward: z = GELU(x) = x * Φ(x) where Φ is CDF of normal
// Backward: dx = dz * (Φ(x) + x * φ(x)) where φ is PDF
extern "C" __global__ void
backward_gelu_f64(double *__restrict__ grad_input,
                  const double *__restrict__ grad_output,
                  const double *__restrict__ input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double x = input[idx];
  double dz = grad_output[idx];

  // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  double sqrt_2_over_pi = 0.7978845608028654;
  double cdf_coeff = 0.044715;

  double x3 = x * x * x;
  double inner = sqrt_2_over_pi * (x + cdf_coeff * x3);
  double tanh_inner = tanh(inner);
  double sech2_inner = 1.0 - tanh_inner * tanh_inner;

  // d/dx GELU = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * sqrt(2/π) *
  // (1 + 3*0.044715*x²)
  double d_inner = sqrt_2_over_pi * (1.0 + 3.0 * cdf_coeff * x * x);
  double grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner;

  grad_input[idx] = dz * grad;
}

// =============================================================================
// Softmax Backward (Fused with Online Statistics)
// =============================================================================

// Forward: y_i = exp(x_i - max) / sum(exp(x_j - max))
// Backward: dx_i = y_i * (dz_i - sum(y_j * dz_j))
extern "C" __global__ void backward_softmax_f64(
    double *__restrict__ grad_input, const double *__restrict__ grad_output,
    const double *__restrict__ output, // softmax(x) from forward
    int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const double *y = output + row * cols;
  const double *dy = grad_output + row * cols;
  double *dx = grad_input + row * cols;

  // Compute dot product: sum(y_j * dy_j)
  __shared__ double s_dot;

  double local_dot = 0.0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    local_dot += y[c] * dy[c];
  }

  // Warp reduce
  local_dot = warp_reduce_sum(local_dot);

  if (threadIdx.x == 0) {
    s_dot = local_dot;
  }
  __syncthreads();

  double dot = s_dot;

  // Compute gradient: dx_i = y_i * (dy_i - dot)
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    dx[c] = y[c] * (dy[c] - dot);
  }
}

extern "C" __global__ void
backward_softmax_f32(float *__restrict__ grad_input,
                     const float *__restrict__ grad_output,
                     const float *__restrict__ output, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float *y = output + row * cols;
  const float *dy = grad_output + row * cols;
  float *dx = grad_input + row * cols;

  __shared__ float s_dot;

  float local_dot = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    local_dot += y[c] * dy[c];
  }

  local_dot = warp_reduce_sum(local_dot);

  if (threadIdx.x == 0) {
    s_dot = local_dot;
  }
  __syncthreads();

  float dot = s_dot;

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    dx[c] = y[c] * (dy[c] - dot);
  }
}

// =============================================================================
// LayerNorm Backward (Golden Variance Target: σ² → 1/φ)
// =============================================================================

// Forward: y = γ * (x - μ) / σ + β
// Backward: Complex chain rule involving μ, σ, and their gradients
extern "C" __global__ void backward_layernorm_f64(
    double *__restrict__ grad_input, double *__restrict__ grad_gamma,
    double *__restrict__ grad_beta, const double *__restrict__ grad_output,
    const double *__restrict__ input, const double *__restrict__ gamma,
    const double *__restrict__ mean,    // [batch]
    const double *__restrict__ inv_std, // [batch] = 1/σ
    int batch_size, int features) {
  int batch = blockIdx.x;
  if (batch >= batch_size)
    return;

  const double *x = input + batch * features;
  const double *dy = grad_output + batch * features;
  double *dx = grad_input + batch * features;

  double mu = mean[batch];
  double inv_sigma = inv_std[batch];

  // Shared memory for reductions
  __shared__ double s_sum_dy_xhat;
  __shared__ double s_sum_dy;

  double local_sum_dy_xhat = 0.0;
  double local_sum_dy = 0.0;

  for (int f = threadIdx.x; f < features; f += blockDim.x) {
    double xhat = (x[f] - mu) * inv_sigma;
    double dy_gamma = dy[f] * gamma[f];

    local_sum_dy_xhat += dy_gamma * xhat;
    local_sum_dy += dy_gamma;

    // Accumulate grad_gamma and grad_beta (atomic or separate pass)
    atomicAdd(&grad_gamma[f], dy[f] * xhat);
    atomicAdd(&grad_beta[f], dy[f]);
  }

  // Reduce within block
  local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);
  local_sum_dy = warp_reduce_sum(local_sum_dy);

  if (threadIdx.x == 0) {
    s_sum_dy_xhat = local_sum_dy_xhat;
    s_sum_dy = local_sum_dy;
  }
  __syncthreads();

  double sum_dy_xhat = s_sum_dy_xhat;
  double sum_dy = s_sum_dy;
  double n = (double)features;

  // Compute input gradient
  for (int f = threadIdx.x; f < features; f += blockDim.x) {
    double xhat = (x[f] - mu) * inv_sigma;
    double dy_gamma = dy[f] * gamma[f];

    dx[f] = inv_sigma * (dy_gamma - (sum_dy + xhat * sum_dy_xhat) / n);
  }
}

// =============================================================================
// φ-Residual Backward (Theory-Aligned Residual Connection)
// =============================================================================

// Forward: y = x + φ * layer(x)
// Backward: dx = dy + φ * d_layer, d_layer_input = dy * φ
extern "C" __global__ void
backward_phi_residual_f64(double *__restrict__ grad_input,
                          double *__restrict__ grad_layer_input,
                          const double *__restrict__ grad_output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double dy = grad_output[idx];

  // Gradient flows through residual connection
  grad_input[idx] = dy;

  // Gradient to layer input is scaled by φ
  grad_layer_input[idx] = dy * PHI_F64;
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// Elementwise
void launch_backward_add_f64(cudaStream_t stream, double *grad_x,
                             double *grad_y, const double *grad_output,
                             int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_add_f64<<<grid, block, 0, stream>>>(grad_x, grad_y, grad_output,
                                               size);
}

void launch_backward_mul_f64(cudaStream_t stream, double *grad_x,
                             double *grad_y, const double *grad_output,
                             const double *x, const double *y, int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_mul_f64<<<grid, block, 0, stream>>>(grad_x, grad_y, grad_output, x,
                                               y, size);
}

void launch_backward_exp_f64(cudaStream_t stream, double *grad_input,
                             const double *grad_output, const double *output,
                             int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_exp_f64<<<grid, block, 0, stream>>>(grad_input, grad_output, output,
                                               size);
}

void launch_backward_log_f64(cudaStream_t stream, double *grad_input,
                             const double *grad_output, const double *input,
                             int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_log_f64<<<grid, block, 0, stream>>>(grad_input, grad_output, input,
                                               size);
}

// Activations
void launch_backward_relu_f64(cudaStream_t stream, double *grad_input,
                              const double *grad_output, const double *input,
                              int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_relu_f64<<<grid, block, 0, stream>>>(grad_input, grad_output, input,
                                                size);
}

void launch_backward_sigmoid_f64(cudaStream_t stream, double *grad_input,
                                 const double *grad_output,
                                 const double *output, int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_sigmoid_f64<<<grid, block, 0, stream>>>(grad_input, grad_output,
                                                   output, size);
}

void launch_backward_tanh_f64(cudaStream_t stream, double *grad_input,
                              const double *grad_output, const double *output,
                              int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_tanh_f64<<<grid, block, 0, stream>>>(grad_input, grad_output, output,
                                                size);
}

void launch_backward_gelu_f64(cudaStream_t stream, double *grad_input,
                              const double *grad_output, const double *input,
                              int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_gelu_f64<<<grid, block, 0, stream>>>(grad_input, grad_output, input,
                                                size);
}

// Softmax
void launch_backward_softmax_f64(cudaStream_t stream, double *grad_input,
                                 const double *grad_output,
                                 const double *output, int rows, int cols) {
  dim3 block(256);
  dim3 grid(rows);
  backward_softmax_f64<<<grid, block, 0, stream>>>(grad_input, grad_output,
                                                   output, rows, cols);
}

void launch_backward_softmax_f32(cudaStream_t stream, float *grad_input,
                                 const float *grad_output, const float *output,
                                 int rows, int cols) {
  dim3 block(256);
  dim3 grid(rows);
  backward_softmax_f32<<<grid, block, 0, stream>>>(grad_input, grad_output,
                                                   output, rows, cols);
}

// LayerNorm
void launch_backward_layernorm_f64(cudaStream_t stream, double *grad_input,
                                   double *grad_gamma, double *grad_beta,
                                   const double *grad_output,
                                   const double *input, const double *gamma,
                                   const double *mean, const double *inv_std,
                                   int batch_size, int features) {
  dim3 block(256);
  dim3 grid(batch_size);
  backward_layernorm_f64<<<grid, block, 0, stream>>>(
      grad_input, grad_gamma, grad_beta, grad_output, input, gamma, mean,
      inv_std, batch_size, features);
}

// φ-Residual
void launch_backward_phi_residual_f64(cudaStream_t stream, double *grad_input,
                                      double *grad_layer_input,
                                      const double *grad_output, int size) {
  dim3 block(256);
  dim3 grid((size + 255) / 256);
  backward_phi_residual_f64<<<grid, block, 0, stream>>>(
      grad_input, grad_layer_input, grad_output, size);
}
}
