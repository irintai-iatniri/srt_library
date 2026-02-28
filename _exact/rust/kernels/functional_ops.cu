/**
 * CUDA Functional Operations Kernels
 *
 * Provides GPU-accelerated elementwise and spatial transform operations.
 *
 * Key operations:
 * - where_kernel: Conditional elementwise select
 * - silu_kernel: SiLU (Swish) activation
 * - pixel_shuffle_kernel: Sub-pixel convolution rearrangement
 * - upsample_nearest_kernel: Nearest neighbor upsampling
 * - upsample_bilinear_kernel: Bilinear interpolation upsampling
 */

#include <cuda_runtime.h>
#include <math.h>

#include "srt_constants.cuh"

// =============================================================================
// Elementwise Operations
// =============================================================================

/**
 * Conditional Select Kernel
 *
 * output[i] = condition[i] > 0 ? x[i] : y[i]
 *
 * @param condition Condition tensor (>0 selects x, <=0 selects y)
 * @param x True-branch values
 * @param y False-branch values
 * @param output Output tensor
 * @param n Total number of elements
 */
extern "C" __global__
void where_kernel(
    const float* __restrict__ condition,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = (condition[idx] > 0.0f) ? x[idx] : y[idx];
}

/**
 * SiLU (Swish) Activation Kernel
 *
 * output[i] = x[i] * sigmoid(x[i])
 * Self-gating is natural syntonic self-reference.
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param n Total number of elements
 */
extern "C" __global__
void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float clamped = fminf(fmaxf(x, -500.0f), 500.0f);
    output[idx] = x / (1.0f + expf(-clamped));
}

// =============================================================================
// Spatial Transform Operations
// =============================================================================

/**
 * Pixel Shuffle Kernel
 *
 * Rearranges: (B, H, W, C*r²) → (B, H*r, W*r, C)
 * NHWC layout throughout.
 *
 * Each thread computes one output element.
 *
 * @param input Input tensor [batch, h, w, c*r*r]
 * @param output Output tensor [batch, h*r, w*r, c]
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param c_total Input channels (c * r * r)
 * @param c Output channels
 * @param r Upscale factor
 * @param out_h Output height (h * r)
 * @param out_w Output width (w * r)
 */
extern "C" __global__
void pixel_shuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w,
    int c_total, int c,
    int r,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;

    if (idx >= total) return;

    // Decode output indices
    int oc = idx % c;
    int ow = (idx / c) % out_w;
    int oh = (idx / c / out_w) % out_h;
    int b = idx / c / out_w / out_h;

    // Map back to input
    int ih = oh / r;
    int iw = ow / r;
    int rh = oh % r;
    int rw = ow % r;
    int ic = oc * r * r + rh * r + rw;

    int in_idx = b * (in_h * in_w * c_total) + ih * (in_w * c_total) + iw * c_total + ic;
    output[idx] = input[in_idx];
}

/**
 * Nearest Neighbor Upsampling Kernel
 *
 * Each thread computes one output element.
 * Input/output layout: [batch, height, width, channels] (NHWC)
 *
 * @param input Input tensor [batch, in_h, in_w, channels]
 * @param output Output tensor [batch, out_h, out_w, channels]
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param channels Number of channels
 * @param scale Upscale factor
 * @param out_h Output height (in_h * scale)
 * @param out_w Output width (in_w * scale)
 */
extern "C" __global__
void upsample_nearest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int channels,
    int scale,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * channels;

    if (idx >= total) return;

    int ch = idx % channels;
    int ow = (idx / channels) % out_w;
    int oh = (idx / channels / out_w) % out_h;
    int b = idx / channels / out_w / out_h;

    // Map to source position
    int sh = oh * in_h / out_h;
    int sw = ow * in_w / out_w;

    int in_idx = b * (in_h * in_w * channels) + sh * (in_w * channels) + sw * channels + ch;
    output[idx] = input[in_idx];
}

/**
 * Bilinear Upsampling Kernel
 *
 * Each thread computes one output element using bilinear interpolation.
 * Uses align_corners=false (half-pixel offset).
 * Input/output layout: [batch, height, width, channels] (NHWC)
 *
 * @param input Input tensor [batch, in_h, in_w, channels]
 * @param output Output tensor [batch, out_h, out_w, channels]
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param channels Number of channels
 * @param out_h Output height
 * @param out_w Output width
 */
extern "C" __global__
void upsample_bilinear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int channels,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * channels;

    if (idx >= total) return;

    int ch = idx % channels;
    int ow = (idx / channels) % out_w;
    int oh = (idx / channels / out_w) % out_h;
    int b = idx / channels / out_w / out_h;

    // Compute source coordinates (half-pixel offset)
    float src_h = ((float)oh + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    float src_w = ((float)ow + 0.5f) * (float)in_w / (float)out_w - 0.5f;

    int h0 = max(0, min((int)floorf(src_h), in_h - 1));
    int h1 = min(h0 + 1, in_h - 1);
    int w0 = max(0, min((int)floorf(src_w), in_w - 1));
    int w1 = min(w0 + 1, in_w - 1);

    float fh = fminf(fmaxf(src_h - floorf(src_h), 0.0f), 1.0f);
    float fw = fminf(fmaxf(src_w - floorf(src_w), 0.0f), 1.0f);

    float v00 = input[b * (in_h * in_w * channels) + h0 * (in_w * channels) + w0 * channels + ch];
    float v01 = input[b * (in_h * in_w * channels) + h0 * (in_w * channels) + w1 * channels + ch];
    float v10 = input[b * (in_h * in_w * channels) + h1 * (in_w * channels) + w0 * channels + ch];
    float v11 = input[b * (in_h * in_w * channels) + h1 * (in_w * channels) + w1 * channels + ch];

    output[idx] = v00 * (1.0f - fh) * (1.0f - fw)
                + v01 * (1.0f - fh) * fw
                + v10 * fh * (1.0f - fw)
                + v11 * fh * fw;
}
