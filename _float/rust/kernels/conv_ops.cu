/**
 * CUDA Convolution Operations Kernels
 *
 * Provides GPU-accelerated 2D convolution and pooling operations.
 *
 * Key operations:
 * - conv2d_kernel: 2D convolution with stride/padding
 * - max_pool2d_kernel: Max pooling
 * - avg_pool2d_kernel: Average pooling
 * - global_avg_pool2d_kernel: Global average pooling
 */

#include <cuda_runtime.h>
#include <math.h>

// Golden ratio for potential syntonic weighting
#define PHI 1.6180339887498948482f

/**
 * 2D Convolution Kernel
 *
 * Input layout: [batch, height, width, in_channels] (NHWC)
 * Kernel layout: [out_channels, kernel_h, kernel_w, in_channels]
 * Output layout: [batch, out_h, out_w, out_channels] (NHWC)
 *
 * Each thread computes one output element.
 *
 * @param input Input tensor
 * @param kernel Convolution kernel weights
 * @param bias Bias (per output channel)
 * @param output Output tensor
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param in_c Input channels
 * @param out_c Output channels
 * @param k_h Kernel height
 * @param k_w Kernel width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param pad_h Vertical padding
 * @param pad_w Horizontal padding
 * @param out_h Output height
 * @param out_w Output width
 */
extern "C" __global__
void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int in_c,
    int out_c,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx >= total) return;
    
    // Decode output indices
    int oc = idx % out_c;
    int ow = (idx / out_c) % out_w;
    int oh = (idx / out_c / out_w) % out_h;
    int b = idx / out_c / out_w / out_h;
    
    float sum = 0.0f;
    
    // Convolution sum
    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            int ih = oh * stride_h + kh - pad_h;
            int iw = ow * stride_w + kw - pad_w;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                for (int ic = 0; ic < in_c; ic++) {
                    // Input index: [b, ih, iw, ic]
                    int in_idx = b * (in_h * in_w * in_c) + ih * (in_w * in_c) + iw * in_c + ic;
                    
                    // Kernel index: [oc, kh, kw, ic]
                    int k_idx = oc * (k_h * k_w * in_c) + kh * (k_w * in_c) + kw * in_c + ic;
                    
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[idx] = sum;
}

/**
 * Conv2d with ReLU fused
 */
extern "C" __global__
void conv2d_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int in_c,
    int out_c,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx >= total) return;
    
    int oc = idx % out_c;
    int ow = (idx / out_c) % out_w;
    int oh = (idx / out_c / out_w) % out_h;
    int b = idx / out_c / out_w / out_h;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            int ih = oh * stride_h + kh - pad_h;
            int iw = ow * stride_w + kw - pad_w;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                for (int ic = 0; ic < in_c; ic++) {
                    int in_idx = b * (in_h * in_w * in_c) + ih * (in_w * in_c) + iw * in_c + ic;
                    int k_idx = oc * (k_h * k_w * in_c) + kh * (k_w * in_c) + kw * in_c + ic;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    // Fused ReLU
    output[idx] = fmaxf(0.0f, sum);
}

/**
 * 2D Max Pooling Kernel
 *
 * @param input Input tensor [batch, height, width, channels]
 * @param output Output tensor
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param channels Number of channels
 * @param pool_h Pool height
 * @param pool_w Pool width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param out_h Output height
 * @param out_w Output width
 */
extern "C" __global__
void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int channels,
    int pool_h, int pool_w,
    int stride_h, int stride_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * channels;
    
    if (idx >= total) return;
    
    int c = idx % channels;
    int ow = (idx / channels) % out_w;
    int oh = (idx / channels / out_w) % out_h;
    int b = idx / channels / out_w / out_h;
    
    float max_val = -INFINITY;
    
    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int ih = oh * stride_h + ph;
            int iw = ow * stride_w + pw;
            
            if (ih < in_h && iw < in_w) {
                int in_idx = b * (in_h * in_w * channels) + ih * (in_w * channels) + iw * channels + c;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }
    
    output[idx] = max_val;
}

/**
 * 2D Average Pooling Kernel
 */
extern "C" __global__
void avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int channels,
    int pool_h, int pool_w,
    int stride_h, int stride_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * channels;
    
    if (idx >= total) return;
    
    int c = idx % channels;
    int ow = (idx / channels) % out_w;
    int oh = (idx / channels / out_w) % out_h;
    int b = idx / channels / out_w / out_h;
    
    float sum = 0.0f;
    float count = (float)(pool_h * pool_w);
    
    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int ih = oh * stride_h + ph;
            int iw = ow * stride_w + pw;
            
            if (ih < in_h && iw < in_w) {
                int in_idx = b * (in_h * in_w * channels) + ih * (in_w * channels) + iw * channels + c;
                sum += input[in_idx];
            }
        }
    }
    
    output[idx] = sum / count;
}

/**
 * Global Average Pooling (reduce spatial dimensions to 1x1)
 *
 * @param input Input tensor [batch, height, width, channels]
 * @param output Output tensor [batch, channels]
 * @param batch Batch size
 * @param height Input height
 * @param width Input width  
 * @param channels Number of channels
 */
extern "C" __global__
void global_avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int height, int width, int channels
) {
    extern __shared__ float sdata[];
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    
    if (b >= batch || c >= channels) return;
    
    // Each thread sums multiple spatial positions
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        int in_idx = b * (height * width * channels) + h * (width * channels) + w * channels + c;
        sum += input[in_idx];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[b * channels + c] = sdata[0] / (float)spatial_size;
    }
}

/**
 * Optimized 3x3 convolution with shared memory
 * Uses tiling for better memory access patterns
 */
extern "C" __global__
void conv2d_3x3_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_h, int in_w, int in_c,
    int out_c,
    int pad
) {
    // Tile dimensions
    const int TILE_W = 16;
    const int TILE_H = 16;
    const int HALO = 1; // For 3x3 kernel with pad=1
    
    extern __shared__ float tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_W;
    int by = blockIdx.y * TILE_H;
    int b = blockIdx.z / out_c;
    int oc = blockIdx.z % out_c;
    
    // Load input tile with halo
    int tile_w = TILE_W + 2 * HALO;
    const int tile_h = TILE_H + 2 * HALO;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_c; ic++) {
        // Load tile (simplified - each thread loads one element)
        int gx = bx + tx - HALO;
        int gy = by + ty - HALO;
        
        float val = 0.0f;
        if (gx >= 0 && gx < in_w && gy >= 0 && gy < in_h) {
            int in_idx = b * (in_h * in_w * in_c) + gy * (in_w * in_c) + gx * in_c + ic;
            val = input[in_idx];
        }
        tile[ty * tile_w + tx] = val;
        __syncthreads();
        
        // Compute convolution for this channel
        if (tx >= HALO && tx < tile_w - HALO && ty >= HALO && ty < tile_h - HALO) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int k_idx = oc * (9 * in_c) + kh * (3 * in_c) + kw * in_c + ic;
                    sum += tile[(ty - HALO + kh) * tile_w + (tx - HALO + kw)] * kernel[k_idx];
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    int ox = bx + tx - HALO;
    int oy = by + ty - HALO;
    
    if (tx >= HALO && tx < tile_w - HALO && ty >= HALO && ty < tile_h - HALO) {
        if (ox >= 0 && ox < in_w && oy >= 0 && oy < in_h) {
            int out_idx = b * (in_h * in_w * out_c) + oy * (in_w * out_c) + ox * out_c + oc;
            float result = sum;
            if (bias != nullptr) {
                result += bias[oc];
            }
            output[out_idx] = result;
        }
    }
}

/**
 * Im2Col transformation for efficient convolution via GEMM
 *
 * Transforms input patches into columns that can be matrix-multiplied with kernel.
 *
 * @param input Input tensor [batch, height, width, channels]
 * @param col Output column matrix
 * @param batch Batch size
 * @param in_h Input height
 * @param in_w Input width
 * @param in_c Input channels  
 * @param k_h Kernel height
 * @param k_w Kernel width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param out_h Output height
 * @param out_w Output width
 */
extern "C" __global__
void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    int batch,
    int in_h, int in_w, int in_c,
    int k_h, int k_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int b = idx / out_w / out_h;
    
    int col_size = k_h * k_w * in_c;
    float* col_ptr = col + idx * col_size;
    
    int col_idx = 0;
    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            for (int ic = 0; ic < in_c; ic++) {
                int ih = oh * stride_h + kh - pad_h;
                int iw = ow * stride_w + kw - pad_w;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = b * (in_h * in_w * in_c) + ih * (in_w * in_c) + iw * in_c + ic;
                    col_ptr[col_idx] = input[in_idx];
                } else {
                    col_ptr[col_idx] = 0.0f;
                }
                col_idx++;
            }
        }
    }
}
