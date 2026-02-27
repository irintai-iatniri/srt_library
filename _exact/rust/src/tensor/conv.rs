// 2D Convolution Operations for Syntonic Tensors
//
// Provides efficient convolution for image/spatial data processing.

/// Perform 2D convolution on flattened input data
///
/// # Arguments
/// * `input` - Input data (batch * height * width * in_channels)
/// * `input_shape` - [batch, height, width, in_channels]
/// * `kernel` - Kernel weights (out_channels * kernel_h * kernel_w * in_channels)
/// * `kernel_shape` - [out_channels, kernel_h, kernel_w, in_channels]
/// * `stride` - (stride_h, stride_w)
/// * `padding` - (pad_h, pad_w)
///
/// # Returns
/// * Tuple of (output_data, output_shape)
pub fn conv2d(
    input: &[f64],
    input_shape: &[usize; 4], // [batch, height, width, in_channels]
    kernel: &[f64],
    kernel_shape: &[usize; 4], // [out_channels, kernel_h, kernel_w, in_channels]
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f64>, [usize; 4]) {
    let [batch, in_h, in_w, in_c] = *input_shape;
    let [out_c, k_h, k_w, _] = *kernel_shape;

    // Calculate output dimensions
    let out_h = (in_h + 2 * padding.0 - k_h) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - k_w) / stride.1 + 1;

    let output_size = batch * out_h * out_w * out_c;
    let mut output = vec![0.0f64; output_size];

    // Perform convolution
    for b in 0..batch {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;

                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            // Input position with padding
                            let ih = (oh * stride.0 + kh) as isize - padding.0 as isize;
                            let iw = (ow * stride.1 + kw) as isize - padding.1 as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                for ic in 0..in_c {
                                    // Input index: [b, ih, iw, ic]
                                    let in_idx = b * (in_h * in_w * in_c)
                                        + ih * (in_w * in_c)
                                        + iw * in_c
                                        + ic;

                                    // Kernel index: [oc, kh, kw, ic]
                                    let k_idx = oc * (k_h * k_w * in_c)
                                        + kh * (k_w * in_c)
                                        + kw * in_c
                                        + ic;

                                    if in_idx < input.len() && k_idx < kernel.len() {
                                        sum += input[in_idx] * kernel[k_idx];
                                    }
                                }
                            }
                        }
                    }

                    // Output index: [b, oh, ow, oc]
                    let out_idx =
                        b * (out_h * out_w * out_c) + oh * (out_w * out_c) + ow * out_c + oc;
                    output[out_idx] = sum;
                }
            }
        }
    }

    (output, [batch, out_h, out_w, out_c])
}

/// Perform 2D max pooling
pub fn max_pool2d(
    input: &[f64],
    input_shape: &[usize; 4], // [batch, height, width, channels]
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> (Vec<f64>, [usize; 4]) {
    let [batch, in_h, in_w, channels] = *input_shape;

    let out_h = (in_h - pool_size.0) / stride.0 + 1;
    let out_w = (in_w - pool_size.1) / stride.1 + 1;

    let output_size = batch * out_h * out_w * channels;
    let mut output = vec![f64::NEG_INFINITY; output_size];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;

                    for ph in 0..pool_size.0 {
                        for pw in 0..pool_size.1 {
                            let ih = oh * stride.0 + ph;
                            let iw = ow * stride.1 + pw;

                            if ih < in_h && iw < in_w {
                                let in_idx = b * (in_h * in_w * channels)
                                    + ih * (in_w * channels)
                                    + iw * channels
                                    + c;
                                if in_idx < input.len() {
                                    max_val = max_val.max(input[in_idx]);
                                }
                            }
                        }
                    }

                    let out_idx = b * (out_h * out_w * channels)
                        + oh * (out_w * channels)
                        + ow * channels
                        + c;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    (output, [batch, out_h, out_w, channels])
}

/// Perform 2D average pooling
pub fn avg_pool2d(
    input: &[f64],
    input_shape: &[usize; 4],
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> (Vec<f64>, [usize; 4]) {
    let [batch, in_h, in_w, channels] = *input_shape;

    let out_h = (in_h - pool_size.0) / stride.0 + 1;
    let out_w = (in_w - pool_size.1) / stride.1 + 1;

    let output_size = batch * out_h * out_w * channels;
    let mut output = vec![0.0f64; output_size];

    let pool_area = (pool_size.0 * pool_size.1) as f64;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;

                    for ph in 0..pool_size.0 {
                        for pw in 0..pool_size.1 {
                            let ih = oh * stride.0 + ph;
                            let iw = ow * stride.1 + pw;

                            if ih < in_h && iw < in_w {
                                let in_idx = b * (in_h * in_w * channels)
                                    + ih * (in_w * channels)
                                    + iw * channels
                                    + c;
                                if in_idx < input.len() {
                                    sum += input[in_idx];
                                }
                            }
                        }
                    }

                    let out_idx = b * (out_h * out_w * channels)
                        + oh * (out_w * channels)
                        + ow * channels
                        + c;
                    output[out_idx] = sum / pool_area;
                }
            }
        }
    }

    (output, [batch, out_h, out_w, channels])
}

/// Global average pooling (spatial dimensions -> single value per channel)
pub fn global_avg_pool2d(
    input: &[f64],
    input_shape: &[usize; 4], // [batch, height, width, channels]
) -> (Vec<f64>, [usize; 2]) {
    let [batch, in_h, in_w, channels] = *input_shape;
    let spatial_size = (in_h * in_w) as f64;

    let mut output = vec![0.0f64; batch * channels];

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = 0.0;
            for h in 0..in_h {
                for w in 0..in_w {
                    let in_idx =
                        b * (in_h * in_w * channels) + h * (in_w * channels) + w * channels + c;
                    if in_idx < input.len() {
                        sum += input[in_idx];
                    }
                }
            }
            output[b * channels + c] = sum / spatial_size;
        }
    }

    (output, [batch, channels])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_basic() {
        // Simple 3x3 input, 2x2 kernel
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input_shape = [1, 3, 3, 1];

        let kernel = vec![1.0, 0.0, 0.0, 1.0];
        let kernel_shape = [1, 2, 2, 1];

        let (output, out_shape) =
            conv2d(&input, &input_shape, &kernel, &kernel_shape, (1, 1), (0, 0));

        assert_eq!(out_shape, [1, 2, 2, 1]);
        // Expected: [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14]
        assert!((output[0] - 6.0).abs() < 1e-6);
        assert!((output[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool2d() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input_shape = [1, 4, 4, 1];

        let (output, out_shape) = max_pool2d(&input, &input_shape, (2, 2), (2, 2));

        assert_eq!(out_shape, [1, 2, 2, 1]);
        // Expected: max of each 2x2 block
        assert!((output[0] - 6.0).abs() < 1e-6);
        assert!((output[1] - 8.0).abs() < 1e-6);
        assert!((output[2] - 14.0).abs() < 1e-6);
        assert!((output[3] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_avg_pool2d() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1];

        let (output, out_shape) = global_avg_pool2d(&input, &input_shape);

        assert_eq!(out_shape, [1, 1]);
        assert!((output[0] - 2.5).abs() < 1e-6);
    }
}
