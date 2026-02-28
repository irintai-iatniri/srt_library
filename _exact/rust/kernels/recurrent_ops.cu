/**
 * CUDA Recurrent Neural Network Operations Kernels
 *
 * Provides GPU-accelerated fused GRU and LSTM cell operations
 * with syntonic gating (σ_s(x) = 1/(1 + exp(-x/φ))).
 *
 * Key operations:
 * - gru_cell_kernel: Fused GRU cell with syntonic sigmoid
 * - lstm_cell_kernel: Fused LSTM cell with φ-biased forget gate
 */

#include <cuda_runtime.h>
#include <math.h>

#include "srt_constants.cuh"

// =============================================================================
// Syntonic Gate Primitives
// =============================================================================

/**
 * Syntonic sigmoid: σ_s(x) = 1/(1 + exp(-x/φ))
 * Gentler gating scaled by golden ratio.
 */
__device__ __forceinline__
float syntonic_sigmoid(float x) {
    float clamped = fminf(fmaxf(x / PHI_F32, -500.0f), 500.0f);
    return 1.0f / (1.0f + expf(-clamped));
}

/**
 * Standard sigmoid (for comparison / fallback)
 */
__device__ __forceinline__
float sigmoid_f(float x) {
    float clamped = fminf(fmaxf(x, -500.0f), 500.0f);
    return 1.0f / (1.0f + expf(-clamped));
}

// =============================================================================
// GRU Cell Kernel
// =============================================================================

/**
 * Fused GRU Cell Kernel
 *
 * Computes one timestep of GRU for all batch elements in parallel.
 * Each thread computes one hidden unit for one batch element.
 *
 * r = σ_s(W_ir @ x + W_hr @ h + b_r)     (reset gate)
 * z = σ_s(W_iz @ x + W_hz @ h + b_z)     (update gate)
 * n = tanh(W_in @ x + W_hn @ (r * h) + b_n)  (candidate)
 * h' = (1 - z) * n + z * h                (new hidden state)
 *
 * @param x Input [batch, input_size]
 * @param h Previous hidden state [batch, hidden_size]
 * @param w_ir Reset input weights [hidden_size, input_size]
 * @param w_hr Reset hidden weights [hidden_size, hidden_size]
 * @param b_r Reset bias [hidden_size]
 * @param w_iz Update input weights [hidden_size, input_size]
 * @param w_hz Update hidden weights [hidden_size, hidden_size]
 * @param b_z Update bias [hidden_size]
 * @param w_in Candidate input weights [hidden_size, input_size]
 * @param w_hn Candidate hidden weights [hidden_size, hidden_size]
 * @param b_n Candidate bias [hidden_size]
 * @param h_new Output hidden state [batch, hidden_size]
 * @param batch Batch size
 * @param input_size Input dimension
 * @param hidden_size Hidden dimension
 */
extern "C" __global__
void gru_cell_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h,
    const float* __restrict__ w_ir,
    const float* __restrict__ w_hr,
    const float* __restrict__ b_r,
    const float* __restrict__ w_iz,
    const float* __restrict__ w_hz,
    const float* __restrict__ b_z,
    const float* __restrict__ w_in,
    const float* __restrict__ w_hn,
    const float* __restrict__ b_n,
    float* __restrict__ h_new,
    int batch,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size;

    if (idx >= total) return;

    int b = idx / hidden_size;
    int j = idx % hidden_size;

    int x_off = b * input_size;
    int h_off = b * hidden_size;

    // Reset gate: r = σ_s(W_ir @ x + W_hr @ h + b_r)
    float r_val = b_r[j];
    for (int i = 0; i < input_size; i++) {
        r_val += w_ir[j * input_size + i] * x[x_off + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r_val += w_hr[j * hidden_size + i] * h[h_off + i];
    }
    float r = syntonic_sigmoid(r_val);

    // Update gate: z = σ_s(W_iz @ x + W_hz @ h + b_z)
    float z_val = b_z[j];
    for (int i = 0; i < input_size; i++) {
        z_val += w_iz[j * input_size + i] * x[x_off + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z_val += w_hz[j * hidden_size + i] * h[h_off + i];
    }
    float z = syntonic_sigmoid(z_val);

    // Candidate: n = tanh(W_in @ x + W_hn @ (r * h) + b_n)
    float n_val = b_n[j];
    for (int i = 0; i < input_size; i++) {
        n_val += w_in[j * input_size + i] * x[x_off + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n_val += w_hn[j * hidden_size + i] * (r * h[h_off + i]);
    }
    float n = tanhf(n_val);

    // New hidden: h' = (1 - z) * n + z * h
    h_new[idx] = (1.0f - z) * n + z * h[h_off + j];
}

// =============================================================================
// LSTM Cell Kernel
// =============================================================================

/**
 * Fused LSTM Cell Kernel
 *
 * Computes one timestep of LSTM for all batch elements in parallel.
 * Each thread computes one hidden unit for one batch element.
 *
 * Forget gate bias initialized to φ (golden ratio) for better gradient flow.
 * Gates use syntonic sigmoid: σ_s(x) = 1/(1 + exp(-x/φ))
 *
 * i = σ_s(W_ii @ x + W_hi @ h + b_i)     (input gate)
 * f = σ_s(W_if @ x + W_hf @ h + b_f)     (forget gate, b_f init to φ)
 * g = tanh(W_ig @ x + W_hg @ h + b_g)     (cell gate)
 * o = σ_s(W_io @ x + W_ho @ h + b_o)     (output gate)
 * c' = f * c + i * g                       (new cell state)
 * h' = o * tanh(c')                        (new hidden state)
 *
 * @param x Input [batch, input_size]
 * @param h Previous hidden state [batch, hidden_size]
 * @param c Previous cell state [batch, hidden_size]
 * @param w_ii Input-input weights [hidden_size, input_size]
 * @param w_hi Hidden-input weights [hidden_size, hidden_size]
 * @param b_i Input gate bias [hidden_size]
 * @param w_if Input-forget weights [hidden_size, input_size]
 * @param w_hf Hidden-forget weights [hidden_size, hidden_size]
 * @param b_f Forget gate bias [hidden_size] (should be initialized to φ)
 * @param w_ig Input-cell weights [hidden_size, input_size]
 * @param w_hg Hidden-cell weights [hidden_size, hidden_size]
 * @param b_g Cell gate bias [hidden_size]
 * @param w_io Input-output weights [hidden_size, input_size]
 * @param w_ho Hidden-output weights [hidden_size, hidden_size]
 * @param b_o Output gate bias [hidden_size]
 * @param h_new Output hidden state [batch, hidden_size]
 * @param c_new Output cell state [batch, hidden_size]
 * @param batch Batch size
 * @param input_size Input dimension
 * @param hidden_size Hidden dimension
 */
extern "C" __global__
void lstm_cell_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h,
    const float* __restrict__ c,
    const float* __restrict__ w_ii,
    const float* __restrict__ w_hi,
    const float* __restrict__ b_i,
    const float* __restrict__ w_if,
    const float* __restrict__ w_hf,
    const float* __restrict__ b_f,
    const float* __restrict__ w_ig,
    const float* __restrict__ w_hg,
    const float* __restrict__ b_g,
    const float* __restrict__ w_io,
    const float* __restrict__ w_ho,
    const float* __restrict__ b_o,
    float* __restrict__ h_new,
    float* __restrict__ c_new,
    int batch,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size;

    if (idx >= total) return;

    int bb = idx / hidden_size;
    int j = idx % hidden_size;

    int x_off = bb * input_size;
    int h_off = bb * hidden_size;

    // Input gate: i = σ_s(W_ii @ x + W_hi @ h + b_i)
    float i_val = b_i[j];
    for (int k = 0; k < input_size; k++) {
        i_val += w_ii[j * input_size + k] * x[x_off + k];
    }
    for (int k = 0; k < hidden_size; k++) {
        i_val += w_hi[j * hidden_size + k] * h[h_off + k];
    }
    float ig = syntonic_sigmoid(i_val);

    // Forget gate: f = σ_s(W_if @ x + W_hf @ h + b_f)
    float f_val = b_f[j];
    for (int k = 0; k < input_size; k++) {
        f_val += w_if[j * input_size + k] * x[x_off + k];
    }
    for (int k = 0; k < hidden_size; k++) {
        f_val += w_hf[j * hidden_size + k] * h[h_off + k];
    }
    float fg = syntonic_sigmoid(f_val);

    // Cell gate: g = tanh(W_ig @ x + W_hg @ h + b_g)
    float g_val = b_g[j];
    for (int k = 0; k < input_size; k++) {
        g_val += w_ig[j * input_size + k] * x[x_off + k];
    }
    for (int k = 0; k < hidden_size; k++) {
        g_val += w_hg[j * hidden_size + k] * h[h_off + k];
    }
    float gg = tanhf(g_val);

    // Output gate: o = σ_s(W_io @ x + W_ho @ h + b_o)
    float o_val = b_o[j];
    for (int k = 0; k < input_size; k++) {
        o_val += w_io[j * input_size + k] * x[x_off + k];
    }
    for (int k = 0; k < hidden_size; k++) {
        o_val += w_ho[j * hidden_size + k] * h[h_off + k];
    }
    float og = syntonic_sigmoid(o_val);

    // New cell state: c' = f * c + i * g
    float new_c = fg * c[h_off + j] + ig * gg;
    c_new[idx] = new_c;

    // New hidden: h' = o * tanh(c')
    h_new[idx] = og * tanhf(new_c);
}
