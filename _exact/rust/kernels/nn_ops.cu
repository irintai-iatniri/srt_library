// =============================================================================
// Neural Network Operations — CUDA Kernels
//
// GPU-accelerated kernels for: silu, golden_silu, where, pixel_shuffle,
// upsample (nearest + bilinear), GRU gates, LSTM gates.
//
// All constants derived from srt_constants.cuh.
// =============================================================================

#include "srt_constants.cuh"

// =============================================================================
// SiLU Activation: x * sigmoid(x)
// =============================================================================

extern "C" __global__ void silu_f64(double *out, const double *in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in_data[i];
        out[i] = x / (1.0 + exp(-x));
    }
}

extern "C" __global__ void silu_f32(float *out, const float *in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in_data[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

// =============================================================================
// Golden SiLU: x * sigmoid(φ*x)
// =============================================================================

extern "C" __global__ void golden_silu_f64(double *out, const double *in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in_data[i];
        out[i] = x / (1.0 + exp(-PHI_F64 * x));
    }
}

extern "C" __global__ void golden_silu_f32(float *out, const float *in_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in_data[i];
        out[i] = x / (1.0f + expf(-PHI_F32 * x));
    }
}

// =============================================================================
// Where (conditional select): out[i] = cond[i] > 0 ? x[i] : y[i]
// =============================================================================

extern "C" __global__ void where_f64(
    double *out, const double *cond, const double *x, const double *y, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (cond[i] > 0.0) ? x[i] : y[i];
    }
}

extern "C" __global__ void where_f32(
    float *out, const float *cond, const float *x, const float *y, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (cond[i] > 0.0f) ? x[i] : y[i];
    }
}

// =============================================================================
// PixelShuffle: (B, H, W, C*r²) → (B, H*r, W*r, C)
// =============================================================================

extern "C" __global__ void pixel_shuffle_f64(
    double *out, const double *in_data,
    int batch, int h, int w, int c, int r, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;
    if (idx >= total) return;

    // Decode output index: [b, oh, ow, oc]
    int oc = idx % c;
    int tmp = idx / c;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int b = tmp / out_h;

    // Map to input coords
    int ih = oh / r;
    int iw = ow / r;
    int rh = oh % r;
    int rw = ow % r;
    int ic = oc * r * r + rh * r + rw;
    int c_total = c * r * r;

    int in_idx = b * (h * w * c_total) + ih * (w * c_total) + iw * c_total + ic;
    out[idx] = in_data[in_idx];
}

// =============================================================================
// Nearest-neighbor Upsample
// =============================================================================

extern "C" __global__ void upsample_nearest_f64(
    double *out, const double *in_data,
    int batch, int h, int w, int c, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;
    if (idx >= total) return;

    int ch = idx % c;
    int tmp = idx / c;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int b = tmp / out_h;

    int sh = oh * h / out_h;
    int sw = ow * w / out_w;

    int in_idx = b * (h * w * c) + sh * (w * c) + sw * c + ch;
    out[idx] = in_data[in_idx];
}

// =============================================================================
// Bilinear Upsample
// =============================================================================

extern "C" __global__ void upsample_bilinear_f64(
    double *out, const double *in_data,
    int batch, int h, int w, int c, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;
    if (idx >= total) return;

    int ch = idx % c;
    int tmp = idx / c;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int b = tmp / out_h;

    double src_h = ((double)oh + 0.5) * (double)h / (double)out_h - 0.5;
    double src_w = ((double)ow + 0.5) * (double)w / (double)out_w - 0.5;

    int h0 = max(0, min((int)floor(src_h), h - 1));
    int h1 = max(0, min(h0 + 1, h - 1));
    int w0 = max(0, min((int)floor(src_w), w - 1));
    int w1 = max(0, min(w0 + 1, w - 1));

    double fh = fmax(0.0, fmin(1.0, src_h - floor(src_h)));
    double fw = fmax(0.0, fmin(1.0, src_w - floor(src_w)));

    double v00 = in_data[b * (h * w * c) + h0 * (w * c) + w0 * c + ch];
    double v01 = in_data[b * (h * w * c) + h0 * (w * c) + w1 * c + ch];
    double v10 = in_data[b * (h * w * c) + h1 * (w * c) + w0 * c + ch];
    double v11 = in_data[b * (h * w * c) + h1 * (w * c) + w1 * c + ch];

    out[idx] = v00 * (1.0 - fh) * (1.0 - fw)
             + v01 * (1.0 - fh) * fw
             + v10 * fh * (1.0 - fw)
             + v11 * fh * fw;
}

// =============================================================================
// GRU Gate Computation (fused: r, z, n gates per element)
//
// Each thread computes one (batch, hidden) element.
// x_proj = W_ir/iz/in @ x (precomputed), h_proj = W_hr/hz/hn @ h (precomputed)
// =============================================================================

extern "C" __global__ void gru_gates_f64(
    double *r_out, double *z_out, double *n_out,
    const double *x_proj_r, const double *h_proj_r,
    const double *x_proj_z, const double *h_proj_z,
    const double *x_proj_n, const double *h_proj_n,
    const double *b_r, const double *b_z, const double *b_n,
    const double *h_prev,
    int batch, int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int j = idx % hidden;

    // Reset gate: r = σ(x_proj_r + h_proj_r + b_r)
    double r = 1.0 / (1.0 + exp(-(x_proj_r[idx] + h_proj_r[idx] + b_r[j])));
    r_out[idx] = r;

    // Update gate: z = σ(x_proj_z + h_proj_z + b_z)
    double z = 1.0 / (1.0 + exp(-(x_proj_z[idx] + h_proj_z[idx] + b_z[j])));
    z_out[idx] = z;

    // Candidate: n = tanh(x_proj_n + (r * h_proj_n) + b_n)
    double n = tanh(x_proj_n[idx] + r * h_proj_n[idx] + b_n[j]);
    n_out[idx] = n;
}

// =============================================================================
// LSTM Gate Computation (fused: i, f, g, o gates per element)
// =============================================================================

extern "C" __global__ void lstm_gates_f64(
    double *i_out, double *f_out, double *g_out, double *o_out,
    const double *x_proj_i, const double *h_proj_i,
    const double *x_proj_f, const double *h_proj_f,
    const double *x_proj_g, const double *h_proj_g,
    const double *x_proj_o, const double *h_proj_o,
    const double *b_i, const double *b_f, const double *b_g, const double *b_o,
    int batch, int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int j = idx % hidden;

    // Input gate
    i_out[idx] = 1.0 / (1.0 + exp(-(x_proj_i[idx] + h_proj_i[idx] + b_i[j])));

    // Forget gate
    f_out[idx] = 1.0 / (1.0 + exp(-(x_proj_f[idx] + h_proj_f[idx] + b_f[j])));

    // Cell gate
    g_out[idx] = tanh(x_proj_g[idx] + h_proj_g[idx] + b_g[j]);

    // Output gate
    o_out[idx] = 1.0 / (1.0 + exp(-(x_proj_o[idx] + h_proj_o[idx] + b_o[j])));
}

// =============================================================================
// Linear (dense) layer: out[b,o] = sum_i input[b,i] * weight[o,i] + bias[o]
// One thread per output element.
// =============================================================================

extern "C" __global__ void linear_f64(
    double *out, const double *input, const double *weight, const double *bias,
    int batch_size, int in_features, int out_features, int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx >= total) return;

    int o = idx % out_features;
    int b = idx / out_features;

    double sum = 0.0;
    for (int i = 0; i < in_features; i++) {
        sum += input[b * in_features + i] * weight[o * in_features + i];
    }
    if (has_bias) {
        sum += bias[o];
    }
    out[idx] = sum;
}
