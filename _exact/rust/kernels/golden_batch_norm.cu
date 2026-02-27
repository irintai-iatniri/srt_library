// Syntonic CUDA Kernels - Golden Batch Normalization
// Implements batch normalization with golden ratio target variance
//
// Theory: In SRT, natural systems exhibit golden ratio structure at equilibrium.
// Standard BatchNorm targets variance = 1.0, but theory predicts variance = 1/φ ≈ 0.618
//
// This golden target variance aligns with the syntonic equilibrium S* = φ - q ≈ 1.591

#include "srt_constants.cuh"

// =============================================================================
// Batch Norm 1D: (batch_size, num_features)
// =============================================================================

// Compute mean and variance for each feature (across batch dimension)
extern "C" __global__ void golden_bn_1d_compute_stats_f64(
    const double *input,        // (batch_size, num_features)
    double *mean,               // (num_features,)
    double *variance,           // (num_features,)
    int batch_size,
    int num_features
) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat_idx < num_features) {
        // Compute mean and variance across batch for this feature
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        double m = sum / (double)batch_size;
        double v = (sum_sq / (double)batch_size) - (m * m);

        mean[feat_idx] = m;
        variance[feat_idx] = v;
    }
}

extern "C" __global__ void golden_bn_1d_compute_stats_f32(
    const float *input,
    float *mean,
    float *variance,
    int batch_size,
    int num_features
) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat_idx < num_features) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        float m = sum / (float)batch_size;
        float v = (sum_sq / (float)batch_size) - (m * m);

        mean[feat_idx] = m;
        variance[feat_idx] = v;
    }
}

// Apply normalization using pre-computed statistics
// mode: 0 = golden (1/φ), 1 = standard (1.0), 2 = custom
extern "C" __global__ void golden_bn_1d_normalize_f64(
    double *out,                // (batch_size, num_features)
    const double *input,        // (batch_size, num_features)
    const double *mean,         // (num_features,)
    const double *variance,     // (num_features,)
    const double *gamma,        // (num_features,) or NULL
    const double *beta,         // (num_features,) or NULL
    double target_variance,     // Target variance (1/φ for golden mode)
    double eps,                 // Epsilon for numerical stability
    int batch_size,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_features;

    if (idx < total_size) {
        int batch_idx = idx / num_features;
        int feat_idx = idx % num_features;
        int full_idx = batch_idx * num_features + feat_idx;

        double val = input[full_idx];
        double m = mean[feat_idx];
        double v = variance[feat_idx];

        // Normalize to zero mean, unit variance
        double std = sqrt(v + eps);
        double normalized = (val - m) / std;

        // Scale to target variance
        double scale = sqrt(target_variance);
        double scaled = normalized * scale;

        // Apply affine transform if provided
        double g = (gamma != NULL) ? gamma[feat_idx] : 1.0;
        double b = (beta != NULL) ? beta[feat_idx] : 0.0;

        out[full_idx] = g * scaled + b;
    }
}

extern "C" __global__ void golden_bn_1d_normalize_f32(
    float *out,
    const float *input,
    const float *mean,
    const float *variance,
    const float *gamma,
    const float *beta,
    float target_variance,
    float eps,
    int batch_size,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_features;

    if (idx < total_size) {
        int batch_idx = idx / num_features;
        int feat_idx = idx % num_features;
        int full_idx = batch_idx * num_features + feat_idx;

        float val = input[full_idx];
        float m = mean[feat_idx];
        float v = variance[feat_idx];

        float std = sqrtf(v + eps);
        float normalized = (val - m) / std;

        float scale = sqrtf(target_variance);
        float scaled = normalized * scale;

        float g = (gamma != NULL) ? gamma[feat_idx] : 1.0f;
        float b = (beta != NULL) ? beta[feat_idx] : 0.0f;

        out[full_idx] = g * scaled + b;
    }
}

// =============================================================================
// Batch Norm 2D: (batch_size, channels, height, width)
// =============================================================================

// Compute mean and variance for each channel (across batch, height, width)
extern "C" __global__ void golden_bn_2d_compute_stats_f64(
    const double *input,        // (batch_size, channels, height, width)
    double *mean,               // (channels,)
    double *variance,           // (channels,)
    int batch_size,
    int channels,
    int height,
    int width
) {
    int chan_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (chan_idx < channels) {
        int spatial_size = height * width;
        int count = batch_size * spatial_size;

        double sum = 0.0;
        double sum_sq = 0.0;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((batch_idx * channels + chan_idx) * height + h) * width + w;
                    double val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        double m = sum / (double)count;
        double v = (sum_sq / (double)count) - (m * m);

        mean[chan_idx] = m;
        variance[chan_idx] = v;
    }
}

extern "C" __global__ void golden_bn_2d_compute_stats_f32(
    const float *input,
    float *mean,
    float *variance,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int chan_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (chan_idx < channels) {
        int spatial_size = height * width;
        int count = batch_size * spatial_size;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((batch_idx * channels + chan_idx) * height + h) * width + w;
                    float val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        float m = sum / (float)count;
        float v = (sum_sq / (float)count) - (m * m);

        mean[chan_idx] = m;
        variance[chan_idx] = v;
    }
}

// Apply normalization for 2D tensors
extern "C" __global__ void golden_bn_2d_normalize_f64(
    double *out,                // (batch_size, channels, height, width)
    const double *input,        // (batch_size, channels, height, width)
    const double *mean,         // (channels,)
    const double *variance,     // (channels,)
    const double *gamma,        // (channels,) or NULL
    const double *beta,         // (channels,) or NULL
    double target_variance,     // Target variance (1/φ for golden mode)
    double eps,                 // Epsilon for numerical stability
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;

    if (idx < total_size) {
        // Decompose linear index
        int spatial_size = height * width;
        int channel_spatial_size = channels * spatial_size;

        int batch_idx = idx / channel_spatial_size;
        int remainder = idx % channel_spatial_size;
        int chan_idx = remainder / spatial_size;
        int spatial_offset = remainder % spatial_size;
        int base = (batch_idx * channels + chan_idx) * spatial_size;
        int full_idx = base + spatial_offset;

        double val = input[full_idx];
        double m = mean[chan_idx];
        double v = variance[chan_idx];

        // Normalize to zero mean, unit variance
        double std = sqrt(v + eps);
        double normalized = (val - m) / std;

        // Scale to target variance
        double scale = sqrt(target_variance);
        double scaled = normalized * scale;

        // Apply affine transform if provided
        double g = (gamma != NULL) ? gamma[chan_idx] : 1.0;
        double b = (beta != NULL) ? beta[chan_idx] : 0.0;

        out[full_idx] = g * scaled + b;
    }
}

extern "C" __global__ void golden_bn_2d_normalize_f32(
    float *out,
    const float *input,
    const float *mean,
    const float *variance,
    const float *gamma,
    const float *beta,
    float target_variance,
    float eps,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;

    if (idx < total_size) {
        int spatial_size = height * width;
        int channel_spatial_size = channels * spatial_size;

        int batch_idx = idx / channel_spatial_size;
        int remainder = idx % channel_spatial_size;
        int chan_idx = remainder / spatial_size;
        int spatial_offset = remainder % spatial_size;
        int base = (batch_idx * channels + chan_idx) * spatial_size;
        int full_idx = base + spatial_offset;

        float val = input[full_idx];
        float m = mean[chan_idx];
        float v = variance[chan_idx];

        float std = sqrtf(v + eps);
        float normalized = (val - m) / std;

        float scale = sqrtf(target_variance);
        float scaled = normalized * scale;

        float g = (gamma != NULL) ? gamma[chan_idx] : 1.0f;
        float b = (beta != NULL) ? beta[chan_idx] : 0.0f;

        out[full_idx] = g * scaled + b;
    }
}

// =============================================================================
// Fused kernels: Compute stats and normalize in single pass (optimization)
// =============================================================================

// Fused 1D batch norm (golden mode)
extern "C" __global__ void golden_bn_1d_fused_f64(
    double *out,                // (batch_size, num_features)
    const double *input,        // (batch_size, num_features)
    const double *gamma,        // (num_features,) or NULL
    const double *beta,         // (num_features,) or NULL
    double eps,                 // Epsilon
    int batch_size,
    int num_features
) {
    int feat_idx = blockIdx.x;

    if (feat_idx < num_features) {
        // Compute mean and variance for this feature
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum / (double)batch_size;
        double variance = (sum_sq / (double)batch_size) - (mean * mean);

        // Normalize all elements for this feature
        double std = sqrt(variance + eps);
        double scale = sqrt(PHI_INV_F64);  // Target variance = 1/φ

        double g = (gamma != NULL) ? gamma[feat_idx] : 1.0;
        double b = (beta != NULL) ? beta[feat_idx] : 0.0;

        for (int batch_idx = threadIdx.x; batch_idx < batch_size; batch_idx += blockDim.x) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            double normalized = (val - mean) / std;
            double scaled = normalized * scale;
            out[idx] = g * scaled + b;
        }
    }
}

extern "C" __global__ void golden_bn_1d_fused_f32(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    float eps,
    int batch_size,
    int num_features
) {
    int feat_idx = blockIdx.x;

    if (feat_idx < num_features) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / (float)batch_size;
        float variance = (sum_sq / (float)batch_size) - (mean * mean);

        float std = sqrtf(variance + eps);
        float scale = sqrtf(PHI_INV_F32);

        float g = (gamma != NULL) ? gamma[feat_idx] : 1.0f;
        float b = (beta != NULL) ? beta[feat_idx] : 0.0f;

        for (int batch_idx = threadIdx.x; batch_idx < batch_size; batch_idx += blockDim.x) {
            int idx = batch_idx * num_features + feat_idx;
            float val = input[idx];
            float normalized = (val - mean) / std;
            float scaled = normalized * scale;
            out[idx] = g * scaled + b;
        }
    }
}

// =============================================================================
// Layer Norm variant: Normalize across features (not batch)
// Useful for transformers
// =============================================================================

extern "C" __global__ void golden_layer_norm_f64(
    double *out,                // (batch_size, num_features)
    const double *input,        // (batch_size, num_features)
    const double *gamma,        // (num_features,) or NULL
    const double *beta,         // (num_features,) or NULL
    double eps,
    int batch_size,
    int num_features
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        // Compute mean and variance across features for this sample
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int feat_idx = 0; feat_idx < num_features; feat_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum / (double)num_features;
        double variance = (sum_sq / (double)num_features) - (mean * mean);

        // Normalize all features for this sample
        double std = sqrt(variance + eps);
        double scale = sqrt(PHI_INV_F64);

        for (int feat_idx = 0; feat_idx < num_features; feat_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            double normalized = (val - mean) / std;
            double scaled = normalized * scale;

            double g = (gamma != NULL) ? gamma[feat_idx] : 1.0;
            double b = (beta != NULL) ? beta[feat_idx] : 0.0;

            out[idx] = g * scaled + b;
        }
    }
}

extern "C" __global__ void golden_layer_norm_f32(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    float eps,
    int batch_size,
    int num_features
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int feat_idx = 0; feat_idx < num_features; feat_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / (float)num_features;
        float variance = (sum_sq / (float)num_features) - (mean * mean);

        float std = sqrtf(variance + eps);
        float scale = sqrtf(PHI_INV_F32);

        for (int feat_idx = 0; feat_idx < num_features; feat_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            float val = input[idx];
            float normalized = (val - mean) / std;
            float scaled = normalized * scale;

            float g = (gamma != NULL) ? gamma[feat_idx] : 1.0f;
            float b = (beta != NULL) ? beta[feat_idx] : 0.0f;

            out[idx] = g * scaled + b;
        }
    }
}

// =============================================================================
// Diagnostic kernels: Compute actual output statistics for validation
// =============================================================================

extern "C" __global__ void compute_output_stats_f64(
    const double *input,
    double *mean_out,
    double *variance_out,
    int batch_size,
    int num_features
) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat_idx < num_features) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int idx = batch_idx * num_features + feat_idx;
            double val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum / (double)batch_size;
        double variance = (sum_sq / (double)batch_size) - (mean * mean);

        mean_out[feat_idx] = mean;
        variance_out[feat_idx] = variance;
    }
}
