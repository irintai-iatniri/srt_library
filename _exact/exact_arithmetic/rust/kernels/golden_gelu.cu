//! GoldenGELU activation kernels
//!
//! GoldenGELU: x * sigmoid(phi * x)
//!
//! Represents winding probability of a token passing through W⁴ aperture
//! based on its energy state x.
//!
//! Mathematical Formulation:
//!   GeLUφ(x) = x * σ(φ * x)
//!
//! Where:
//!   - φ = 1.6180339887 (golden ratio)
//!   - σ(z) = 1 / (1 + e^(-z)) is sigmoid function
//!   - x is input tensor
//!
//! This represents theory-correct GeLU where the scaling factor is
//! exactly the golden ratio φ, derived from SRT geometry.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "srt_constants.cuh"
#define PHI PHI_F64  // alias for backward compat

// =============================================================================
// GOLDEN GELU KERNELS
// =============================================================================

/// GoldenGELU forward pass: x * sigmoid(phi * x)
///
/// Args:
///   input: Input tensor
///   output: Output tensor (GeLUφ(x))
///   n: Number of elements
__global__ void golden_gelu_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double x = input[idx];

    // Scale by phi: phi * x
    double scaled = PHI * x;

    // Compute sigmoid: 1 / (1 + exp(-scaled))
    double exp_neg_scaled = exp(-scaled);
    double gate = 1.0 / (1.0 + exp_neg_scaled);

    // Apply gate: x * gate
    output[idx] = x * gate;
}

__global__ void golden_gelu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    // Scale by phi
    float scaled = (float)PHI * x;

    // Compute sigmoid
    float exp_neg_scaled = expf(-scaled);
    float gate = 1.0f / (1.0f + exp_neg_scaled);

    // Apply gate
    output[idx] = x * gate;
}

// =============================================================================
// GOLDEN GELU BACKWARD KERNELS (For training)
// =============================================================================

/// GoldenGELU backward pass
///
/// Derivative: d/dx [x * σ(φx)] = σ(φx) + φ * x * σ(φx) * (1 - σ(φx))
///
/// Args:
///   input: Original input
///   grad_output: Gradient from next layer
///   grad_input: Output gradient
///   n: Number of elements
__global__ void golden_gelu_backward_f64(
    const double* __restrict__ input,
    const double* __restrict__ grad_output,
    double* __restrict__ grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double x = input[idx];

    // Scale by phi
    double scaled = PHI * x;

    // Compute sigmoid: σ(φx)
    double exp_neg_scaled = exp(-scaled);
    double gate = 1.0 / (1.0 + exp_neg_scaled);

    // Compute derivative: σ + φ * x * σ * (1 - σ)
    double gate_complement = 1.0 - gate;
    double derivative = gate + PHI * x * gate * gate_complement;

    // Chain rule: grad_output * derivative
    grad_input[idx] = grad_output[idx] * derivative;
}

__global__ void golden_gelu_backward_f32(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    // Scale by phi
    float scaled = (float)PHI * x;

    // Compute sigmoid: 1 / (1 + exp(-scaled))
    float exp_neg_scaled = expf(-scaled);
    float gate = 1.0f / (1.0f + exp_neg_scaled);

    // Compute derivative
    float gate_complement = 1.0f - gate;
    float derivative = gate + (float)PHI * x * gate * gate_complement;

    // Chain rule
    grad_input[idx] = grad_output[idx] * derivative;
}

// =============================================================================
// BATCHED GOLDEN GELU (For multiple activations in parallel)
// =============================================================================

/// Batch GoldenGELU forward pass for multiple inputs
///
/// Args:
///   inputs: Flattened input tensors
///   outputs: Flattened output tensors
///   batch_size: Number of tensors in batch
///   n_elements: Number of elements per tensor
__global__ void batched_golden_gelu_f64(
    const double* __restrict__ inputs,
    double* __restrict__ outputs,
    int batch_size,
    int n_elements
) {
    int batch_idx = blockIdx.y;
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || elem_idx >= n_elements) return;

    int linear_idx = batch_idx * n_elements + elem_idx;
    double x = inputs[linear_idx];

    // Scale by phi
    double scaled = PHI * x;

    // Compute sigmoid
    double exp_neg_scaled = exp(-scaled);
    double gate = 1.0 / (1.0 + exp_neg_scaled);

    // Apply gate
    outputs[linear_idx] = x * gate;
}

// =============================================================================
// HOST INTERFACE
// =============================================================================

extern "C" {

/// GoldenGELU forward pass
cudaError_t golden_gelu_forward(
    const double* d_input,
    double* d_output,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    golden_gelu_f64<<<blocks, threads, 0, stream>>>(d_input, d_output, n);
    return cudaGetLastError();
}

/// GoldenGELU backward pass
cudaError_t golden_gelu_backward(
    const double* d_input,
    const double* d_grad_output,
    double* d_grad_input,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    golden_gelu_backward_f64<<<blocks, threads, 0, stream>>>(
        d_input, d_grad_output, d_grad_input, n
    );
    return cudaGetLastError();
}

/// Batched GoldenGELU forward pass
cudaError_t batched_golden_gelu_forward(
    const double* d_inputs,
    double* d_outputs,
    int batch_size,
    int n_elements,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int elem_blocks = (n_elements + threads - 1) / threads;

    dim3 blocks(elem_blocks, batch_size, 1);
    dim3 threads_per_block(threads, 1, 1);

    batched_golden_gelu_f64<<<blocks, threads_per_block, 0, stream>>>(
        d_inputs, d_outputs, batch_size, n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
