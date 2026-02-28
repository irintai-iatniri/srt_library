//! Neural Network Operations for Syntonic Tensors
//!
//! Provides CPU implementations of NN operations (arange, linspace, eye, stack,
//! silu, golden_silu, where, einsum, linear, pixel_shuffle, upsample, gru_cell,
//! lstm_cell) with CUDA dispatch for compute-heavy paths.

use crate::exact::golden::GoldenExact;

/// φ = (1+√5)/2
#[inline]
fn phi() -> f64 {
    GoldenExact::phi().to_f64()
}

// =============================================================================
// Tensor Creation (CPU only)
// =============================================================================

/// Create evenly spaced values in [start, end) with given step.
pub fn arange(start: f64, end: f64, step: f64) -> Vec<f64> {
    let mut result = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < end {
            result.push(v);
            v += step;
        }
    } else if step < 0.0 {
        while v > end {
            result.push(v);
            v += step;
        }
    }
    if result.is_empty() {
        result.push(0.0);
    }
    result
}

/// Create `steps` linearly spaced values in [start, end].
pub fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![start];
    }
    let mut result = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        result.push(start + t * (end - start));
    }
    result
}

/// Create n×n identity matrix. Returns (data, [n, n]).
pub fn eye(n: usize) -> (Vec<f64>, [usize; 2]) {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    (data, [n, n])
}

/// Stack tensors along a new dimension.
///
/// All tensors must have the same shape. `dim` is where the new axis is inserted.
/// Returns (data, new_shape).
pub fn stack(tensors: &[&[f64]], shapes: &[&[usize]], dim: usize) -> (Vec<f64>, Vec<usize>) {
    let n = tensors.len();
    if n == 0 {
        return (vec![], vec![0]);
    }

    let base_shape = shapes[0];
    let ndim = base_shape.len();

    // Build output shape: insert n at position dim
    let mut new_shape = Vec::with_capacity(ndim + 1);
    for i in 0..dim {
        new_shape.push(base_shape[i]);
    }
    new_shape.push(n);
    for i in dim..ndim {
        new_shape.push(base_shape[i]);
    }

    // For dim=0 (most common), just concatenate
    if dim == 0 {
        let mut data = Vec::new();
        for t in tensors {
            data.extend_from_slice(t);
        }
        return (data, new_shape);
    }

    // General case: compute outer/inner sizes
    let mut outer_size = 1usize;
    for i in 0..dim {
        outer_size *= base_shape[i];
    }
    let mut inner_size = 1usize;
    for i in dim..ndim {
        inner_size *= base_shape[i];
    }

    let total = outer_size * n * inner_size;
    let mut data = Vec::with_capacity(total);

    for outer in 0..outer_size {
        for t_idx in 0..n {
            let start = outer * inner_size;
            let end = start + inner_size;
            data.extend_from_slice(&tensors[t_idx][start..end]);
        }
    }

    (data, new_shape)
}

// =============================================================================
// Convolution Operations
// =============================================================================

/// 1D Convolution
///
/// Input layout: [batch, length, in_channels] (NLC)
/// Kernel layout: [out_channels, kernel_len, in_channels]
/// Returns (output_data, [batch, out_len, out_channels])
pub fn conv1d(
    input: &[f64],
    input_shape: &[usize; 3], // [batch, length, in_channels]
    kernel: &[f64],
    kernel_shape: &[usize; 3], // [out_channels, kernel_len, in_channels]
    stride: usize,
    padding: usize,
) -> (Vec<f64>, [usize; 3]) {
    let [batch, in_len, in_c] = *input_shape;
    let [out_c, k_len, _] = *kernel_shape;

    let out_len = (in_len + 2 * padding - k_len) / stride + 1;
    let output_size = batch * out_len * out_c;
    let mut output = vec![0.0f64; output_size];

    for b in 0..batch {
        for oc in 0..out_c {
            for ol in 0..out_len {
                let mut sum = 0.0;
                for kl in 0..k_len {
                    let il = (ol * stride + kl) as isize - padding as isize;
                    if il >= 0 && il < in_len as isize {
                        let il = il as usize;
                        for ic in 0..in_c {
                            let in_idx = b * (in_len * in_c) + il * in_c + ic;
                            let k_idx = oc * (k_len * in_c) + kl * in_c + ic;
                            if in_idx < input.len() && k_idx < kernel.len() {
                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                }
                let out_idx = b * (out_len * out_c) + ol * out_c + oc;
                output[out_idx] = sum;
            }
        }
    }

    (output, [batch, out_len, out_c])
}

/// Transposed 2D Convolution (Deconvolution)
///
/// Input layout: [batch, height, width, in_channels] (NHWC)
/// Kernel layout: [in_channels, kernel_h, kernel_w, out_channels]
/// Returns (output_data, [batch, out_h, out_w, out_channels])
pub fn conv_transpose2d(
    input: &[f64],
    input_shape: &[usize; 4], // [batch, height, width, in_channels]
    kernel: &[f64],
    kernel_shape: &[usize; 4], // [in_channels, kernel_h, kernel_w, out_channels]
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f64>, [usize; 4]) {
    let [batch, in_h, in_w, in_c] = *input_shape;
    let [_, k_h, k_w, out_c] = *kernel_shape;

    let out_h = (in_h - 1) * stride.0 + k_h - 2 * padding.0;
    let out_w = (in_w - 1) * stride.1 + k_w - 2 * padding.1;

    let output_size = batch * out_h * out_w * out_c;
    let mut output = vec![0.0f64; output_size];

    for b in 0..batch {
        for ih in 0..in_h {
            for iw in 0..in_w {
                for ic in 0..in_c {
                    let in_idx = b * (in_h * in_w * in_c) + ih * (in_w * in_c) + iw * in_c + ic;
                    let in_val = input[in_idx];

                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let oh = ih * stride.0 + kh;
                            let ow = iw * stride.1 + kw;

                            if oh >= padding.0 && ow >= padding.1 {
                                let oh = oh - padding.0;
                                let ow = ow - padding.1;

                                if oh < out_h && ow < out_w {
                                    for oc in 0..out_c {
                                        let k_idx = ic * (k_h * k_w * out_c)
                                            + kh * (k_w * out_c)
                                            + kw * out_c
                                            + oc;
                                        let out_idx = b * (out_h * out_w * out_c)
                                            + oh * (out_w * out_c)
                                            + ow * out_c
                                            + oc;
                                        output[out_idx] += in_val * kernel[k_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (output, [batch, out_h, out_w, out_c])
}

/// Embedding lookup: output[i] = table[indices[i]]
///
/// table: [vocab_size, embed_dim], indices: [num_indices]
/// Returns (output_data, [num_indices, embed_dim])
pub fn embedding_lookup(
    table: &[f64],
    indices: &[usize],
    vocab_size: usize,
    embed_dim: usize,
) -> (Vec<f64>, [usize; 2]) {
    let n = indices.len();
    let mut output = vec![0.0f64; n * embed_dim];

    for i in 0..n {
        let token_id = indices[i];
        if token_id < vocab_size {
            let src_off = token_id * embed_dim;
            let dst_off = i * embed_dim;
            for d in 0..embed_dim {
                output[dst_off + d] = table[src_off + d];
            }
        }
    }

    (output, [n, embed_dim])
}

// =============================================================================
// Activation Functions
// =============================================================================

/// SiLU (Swish): x * sigmoid(x)
pub fn silu(input: &[f64]) -> Vec<f64> {
    input
        .iter()
        .map(|&x| {
            let clamped = x.max(-500.0).min(500.0);
            x / (1.0 + (-clamped).exp())
        })
        .collect()
}

/// Golden SiLU: x * sigmoid(φ*x)
pub fn golden_silu(input: &[f64]) -> Vec<f64> {
    let p = phi();
    input
        .iter()
        .map(|&x| {
            let clamped = (p * x).max(-500.0).min(500.0);
            x / (1.0 + (-clamped).exp())
        })
        .collect()
}

// =============================================================================
// Conditional Select
// =============================================================================

/// Elementwise conditional: out[i] = x[i] if cond[i] > 0 else y[i]
pub fn where_select(condition: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
    condition
        .iter()
        .zip(x.iter().zip(y.iter()))
        .map(|(&c, (&xv, &yv))| if c > 0.0 { xv } else { yv })
        .collect()
}

// =============================================================================
// Linear (dense) layer
// =============================================================================

/// Linear transformation: output = input @ weight^T + bias
///
/// input shape: [..., in_features]
/// weight shape: [out_features, in_features]
/// Returns (data, out_shape) where out_shape is [..., out_features]
pub fn linear(
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    in_shape: &[usize],
    w_shape: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    let in_features = *in_shape.last().unwrap();
    let out_features = w_shape[0];

    // Number of rows in the batch
    let batch_size: usize = in_shape.iter().take(in_shape.len() - 1).product();
    let batch_size = if batch_size == 0 { 1 } else { batch_size };

    let mut output = vec![0.0; batch_size * out_features];

    // output[b, o] = sum_i input[b, i] * weight[o, i]
    for b in 0..batch_size {
        for o in 0..out_features {
            let mut sum = 0.0;
            for i in 0..in_features {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            if let Some(bias) = bias {
                sum += bias[o];
            }
            output[b * out_features + o] = sum;
        }
    }

    let mut out_shape: Vec<usize> = in_shape[..in_shape.len() - 1].to_vec();
    out_shape.push(out_features);

    (output, out_shape)
}

// =============================================================================
// Pixel Shuffle
// =============================================================================

/// PixelShuffle: (B, H, W, C*r²) → (B, H*r, W*r, C)
///
/// shape: [batch, h, w, c_total] where c_total = c * r * r
pub fn pixel_shuffle(data: &[f64], shape: &[usize; 4], r: usize) -> (Vec<f64>, [usize; 4]) {
    let [batch, h, w, c_total] = *shape;
    let c = c_total / (r * r);
    let out_h = h * r;
    let out_w = w * r;

    let mut output = vec![0.0; batch * out_h * out_w * c];

    for b in 0..batch {
        for ih in 0..h {
            for iw in 0..w {
                for oc in 0..c {
                    for rh in 0..r {
                        for rw in 0..r {
                            let ic = oc * r * r + rh * r + rw;
                            let in_idx =
                                b * (h * w * c_total) + ih * (w * c_total) + iw * c_total + ic;
                            let oh = ih * r + rh;
                            let ow = iw * r + rw;
                            let out_idx =
                                b * (out_h * out_w * c) + oh * (out_w * c) + ow * c + oc;
                            output[out_idx] = data[in_idx];
                        }
                    }
                }
            }
        }
    }

    (output, [batch, out_h, out_w, c])
}

// =============================================================================
// Upsample
// =============================================================================

/// Nearest-neighbor upsampling.
///
/// shape: [batch, h, w, c], scale applied to h and w.
pub fn upsample_nearest(
    data: &[f64],
    shape: &[usize; 4],
    scale: usize,
) -> (Vec<f64>, [usize; 4]) {
    let [batch, h, w, c] = *shape;
    let out_h = h * scale;
    let out_w = w * scale;

    let mut output = Vec::with_capacity(batch * out_h * out_w * c);

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let sh = oh * h / out_h;
                let sw = ow * w / out_w;
                for ch in 0..c {
                    let idx = b * (h * w * c) + sh * (w * c) + sw * c + ch;
                    output.push(data[idx]);
                }
            }
        }
    }

    (output, [batch, out_h, out_w, c])
}

/// Bilinear upsampling.
///
/// shape: [batch, h, w, c], scale applied to h and w.
pub fn upsample_bilinear(
    data: &[f64],
    shape: &[usize; 4],
    scale: usize,
) -> (Vec<f64>, [usize; 4]) {
    let [batch, h, w, c] = *shape;
    let out_h = h * scale;
    let out_w = w * scale;

    let mut output = Vec::with_capacity(batch * out_h * out_w * c);

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let src_h = (oh as f64 + 0.5) * h as f64 / out_h as f64 - 0.5;
                let src_w = (ow as f64 + 0.5) * w as f64 / out_w as f64 - 0.5;

                let h0 = (src_h.floor() as isize).max(0).min(h as isize - 1) as usize;
                let h1 = (h0 + 1).min(h - 1);
                let w0 = (src_w.floor() as isize).max(0).min(w as isize - 1) as usize;
                let w1 = (w0 + 1).min(w - 1);

                let fh = (src_h - src_h.floor()).max(0.0).min(1.0);
                let fw = (src_w - src_w.floor()).max(0.0).min(1.0);

                for ch in 0..c {
                    let v00 = data[b * (h * w * c) + h0 * (w * c) + w0 * c + ch];
                    let v01 = data[b * (h * w * c) + h0 * (w * c) + w1 * c + ch];
                    let v10 = data[b * (h * w * c) + h1 * (w * c) + w0 * c + ch];
                    let v11 = data[b * (h * w * c) + h1 * (w * c) + w1 * c + ch];

                    let val = v00 * (1.0 - fh) * (1.0 - fw)
                        + v01 * (1.0 - fh) * fw
                        + v10 * fh * (1.0 - fw)
                        + v11 * fh * fw;
                    output.push(val);
                }
            }
        }
    }

    (output, [batch, out_h, out_w, c])
}

// =============================================================================
// Einsum
// =============================================================================

/// General einsum supporting common contraction patterns.
///
/// Fast paths for: matmul (ij,jk->ik), batch matmul (bij,bjk->bik),
/// matvec (ij,j->i), dot (i,i->), transpose (ij->ji), outer (i,j->ij),
/// batch outer (bti,btj->bij).
///
/// Falls back to general contraction loop for other patterns.
pub fn einsum(equation: &str, operands: &[(&[f64], &[usize])]) -> (Vec<f64>, Vec<usize>) {
    let eq = equation.replace(' ', "");
    let parts: Vec<&str> = eq.split("->").collect();
    let inputs: Vec<&str> = parts[0].split(',').collect();
    let output_subs = if parts.len() > 1 { parts[1] } else { "" };

    // --- Two-operand patterns ---
    if operands.len() == 2 && inputs.len() == 2 {
        let (a_data, a_shape) = operands[0];
        let (b_data, b_shape) = operands[1];
        let li = inputs[0];
        let ri = inputs[1];

        // matmul: ij,jk->ik
        if li == "ij" && ri == "jk" && output_subs == "ik" {
            return mm(a_data, a_shape, b_data, b_shape);
        }

        // batch matmul: bij,bjk->bik
        if li == "bij" && ri == "bjk" && output_subs == "bik" {
            return bmm(a_data, a_shape, b_data, b_shape);
        }

        // matvec: ij,j->i
        if li == "ij" && ri == "j" && output_subs == "i" {
            return matvec(a_data, a_shape, b_data);
        }

        // dot product: i,i->
        if li == "i" && ri == "i" && output_subs.is_empty() {
            let dot: f64 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();
            return (vec![dot], vec![1]);
        }

        // outer product: i,j->ij
        if li == "i" && ri == "j" && output_subs == "ij" {
            let m = a_shape[0];
            let n = b_shape[0];
            let mut out = Vec::with_capacity(m * n);
            for i in 0..m {
                for j in 0..n {
                    out.push(a_data[i] * b_data[j]);
                }
            }
            return (out, vec![m, n]);
        }

        // batch outer: bti,btj->bij
        if li == "bti" && ri == "btj" && output_subs == "bij" {
            let b = a_shape[0];
            let t = a_shape[1];
            let i_dim = a_shape[2];
            let j_dim = b_shape[2];
            let mut out = vec![0.0; b * i_dim * j_dim];
            for bb in 0..b {
                for tt in 0..t {
                    for ii in 0..i_dim {
                        for jj in 0..j_dim {
                            let a_idx = bb * (t * i_dim) + tt * i_dim + ii;
                            let b_idx = bb * (t * j_dim) + tt * j_dim + jj;
                            out[bb * (i_dim * j_dim) + ii * j_dim + jj] +=
                                a_data[a_idx] * b_data[b_idx];
                        }
                    }
                }
            }
            return (out, vec![b, i_dim, j_dim]);
        }
    }

    // --- Single-operand patterns ---
    if operands.len() == 1 && inputs.len() == 1 {
        let (a_data, a_shape) = operands[0];

        // transpose: ij->ji
        if inputs[0] == "ij" && output_subs == "ji" {
            let rows = a_shape[0];
            let cols = a_shape[1];
            let mut out = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = a_data[i * cols + j];
                }
            }
            return (out, vec![cols, rows]);
        }

        // trace: ii->
        if inputs[0] == "ii" && output_subs.is_empty() {
            let n = a_shape[0];
            let trace: f64 = (0..n).map(|i| a_data[i * n + i]).sum();
            return (vec![trace], vec![1]);
        }

        // diagonal: ii->i
        if inputs[0] == "ii" && output_subs == "i" {
            let n = a_shape[0];
            let diag: Vec<f64> = (0..n).map(|i| a_data[i * n + i]).collect();
            return (diag, vec![n]);
        }

        // sum over axis: ij->i (row sums) or ij->j (col sums)
        if inputs[0] == "ij" && output_subs == "i" {
            let rows = a_shape[0];
            let cols = a_shape[1];
            let mut out = vec![0.0; rows];
            for i in 0..rows {
                for j in 0..cols {
                    out[i] += a_data[i * cols + j];
                }
            }
            return (out, vec![rows]);
        }
        if inputs[0] == "ij" && output_subs == "j" {
            let rows = a_shape[0];
            let cols = a_shape[1];
            let mut out = vec![0.0; cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j] += a_data[i * cols + j];
                }
            }
            return (out, vec![cols]);
        }
    }

    // --- General fallback ---
    einsum_general(&eq, operands)
}

/// General einsum fallback: parses subscripts, builds index maps,
/// iterates over all output + contracted index combinations.
fn einsum_general(equation: &str, operands: &[(&[f64], &[usize])]) -> (Vec<f64>, Vec<usize>) {
    let parts: Vec<&str> = equation.split("->").collect();
    let inputs: Vec<&str> = parts[0].split(',').collect();
    let output_subs: Vec<char> = if parts.len() > 1 {
        parts[1].chars().collect()
    } else {
        vec![]
    };

    // Build index→size map from all operands
    let mut index_sizes: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (idx, input_sub) in inputs.iter().enumerate() {
        let (_, shape) = operands[idx];
        for (dim, ch) in input_sub.chars().enumerate() {
            if dim < shape.len() {
                index_sizes.insert(ch, shape[dim]);
            }
        }
    }

    // Determine contracted (summed) indices: appear in inputs but not output
    let mut all_input_chars: Vec<char> = Vec::new();
    for input_sub in &inputs {
        for ch in input_sub.chars() {
            if !all_input_chars.contains(&ch) {
                all_input_chars.push(ch);
            }
        }
    }
    let contracted: Vec<char> = all_input_chars
        .iter()
        .filter(|ch| !output_subs.contains(ch))
        .copied()
        .collect();

    // Compute output shape and size
    let out_shape: Vec<usize> = output_subs
        .iter()
        .map(|ch| *index_sizes.get(ch).unwrap_or(&1))
        .collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    // Compute contracted range sizes
    let contracted_sizes: Vec<usize> = contracted
        .iter()
        .map(|ch| *index_sizes.get(ch).unwrap_or(&1))
        .collect();
    let contracted_total: usize = contracted_sizes.iter().product::<usize>().max(1);

    let mut output = vec![0.0; out_size];

    // Iterate over output indices
    for out_flat in 0..out_size {
        // Decode output flat index into per-dim indices
        let mut out_indices: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        let mut remainder = out_flat;
        for (dim, ch) in output_subs.iter().enumerate().rev() {
            let size = out_shape[dim];
            out_indices.insert(*ch, remainder % size);
            remainder /= size;
        }

        let mut sum = 0.0;

        // Iterate over contracted indices
        for c_flat in 0..contracted_total {
            let mut c_indices: std::collections::HashMap<char, usize> =
                std::collections::HashMap::new();
            let mut rem = c_flat;
            for (dim, ch) in contracted.iter().enumerate().rev() {
                let size = contracted_sizes[dim];
                c_indices.insert(*ch, rem % size);
                rem /= size;
            }

            // Compute product of all operand elements at these indices
            let mut product = 1.0;
            for (op_idx, input_sub) in inputs.iter().enumerate() {
                let (op_data, op_shape) = operands[op_idx];
                let mut flat_idx = 0usize;
                let mut stride = 1usize;
                for (dim, ch) in input_sub.chars().rev().enumerate() {
                    let idx_val = out_indices
                        .get(&ch)
                        .or_else(|| c_indices.get(&ch))
                        .copied()
                        .unwrap_or(0);
                    flat_idx += idx_val * stride;
                    let dim_from_end = dim;
                    let actual_dim = input_sub.len() - 1 - dim_from_end;
                    if actual_dim < op_shape.len() {
                        stride *= op_shape[actual_dim];
                    }
                }
                if flat_idx < op_data.len() {
                    product *= op_data[flat_idx];
                }
            }
            sum += product;
        }

        output[out_flat] = sum;
    }

    (output, out_shape)
}

/// Matrix multiply helper: (m,k) × (k,n) → (m,n)
fn mm(a: &[f64], a_shape: &[usize], b: &[f64], b_shape: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    (out, vec![m, n])
}

/// Batch matrix multiply helper: (b,m,k) × (b,k,n) → (b,m,n)
fn bmm(a: &[f64], a_shape: &[usize], b: &[f64], b_shape: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];
    let mut out = vec![0.0; batch * m * n];
    for bb in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[bb * (m * k) + i * k + p] * b[bb * (k * n) + p * n + j];
                }
                out[bb * (m * n) + i * n + j] = sum;
            }
        }
    }
    (out, vec![batch, m, n])
}

/// Matrix-vector multiply helper: (m,n) × (n,) → (m,)
fn matvec(a: &[f64], a_shape: &[usize], b: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let m = a_shape[0];
    let n = a_shape[1];
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * b[j];
        }
        out.push(sum);
    }
    (out, vec![m])
}

// =============================================================================
// RNN Cell Operations
// =============================================================================

/// Sigmoid helper
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.max(-500.0).min(500.0)).exp())
}

/// GRU cell: computes one timestep of GRU.
///
/// x: [batch, input_size], h: [batch, hidden_size]
/// Weight matrices: w_ir [hidden, input], w_hr [hidden, hidden], etc.
/// Returns h_new: [batch * hidden_size]
pub fn gru_cell(
    x: &[f64],
    h: &[f64],
    w_ir: &[f64],
    w_hr: &[f64],
    b_r: &[f64],
    w_iz: &[f64],
    w_hz: &[f64],
    b_z: &[f64],
    w_in: &[f64],
    w_hn: &[f64],
    b_n: &[f64],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Vec<f64> {
    let mut h_new = vec![0.0; batch * hidden_size];

    for b in 0..batch {
        let x_off = b * input_size;
        let h_off = b * hidden_size;

        for j in 0..hidden_size {
            // Reset gate: r = σ(W_ir @ x + W_hr @ h + b_r)
            let mut r_val = b_r[j];
            for i in 0..input_size {
                r_val += w_ir[j * input_size + i] * x[x_off + i];
            }
            for i in 0..hidden_size {
                r_val += w_hr[j * hidden_size + i] * h[h_off + i];
            }
            let r = sigmoid(r_val);

            // Update gate: z = σ(W_iz @ x + W_hz @ h + b_z)
            let mut z_val = b_z[j];
            for i in 0..input_size {
                z_val += w_iz[j * input_size + i] * x[x_off + i];
            }
            for i in 0..hidden_size {
                z_val += w_hz[j * hidden_size + i] * h[h_off + i];
            }
            let z = sigmoid(z_val);

            // Candidate: n = tanh(W_in @ x + W_hn @ (r * h) + b_n)
            let mut n_val = b_n[j];
            for i in 0..input_size {
                n_val += w_in[j * input_size + i] * x[x_off + i];
            }
            for i in 0..hidden_size {
                n_val += w_hn[j * hidden_size + i] * (r * h[h_off + i]);
            }
            let n = n_val.tanh();

            // New hidden: h' = (1 - z) * n + z * h
            h_new[h_off + j] = (1.0 - z) * n + z * h[h_off + j];
        }
    }

    h_new
}

/// LSTM cell: computes one timestep of LSTM.
///
/// x: [batch, input_size], h: [batch, hidden_size], c: [batch, hidden_size]
/// Returns (h_new, c_new), each [batch * hidden_size]
pub fn lstm_cell(
    x: &[f64],
    h: &[f64],
    c: &[f64],
    w_ii: &[f64],
    w_hi: &[f64],
    b_i: &[f64],
    w_if: &[f64],
    w_hf: &[f64],
    b_f: &[f64],
    w_ig: &[f64],
    w_hg: &[f64],
    b_g: &[f64],
    w_io: &[f64],
    w_ho: &[f64],
    b_o: &[f64],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut h_new = vec![0.0; batch * hidden_size];
    let mut c_new = vec![0.0; batch * hidden_size];

    for b in 0..batch {
        let x_off = b * input_size;
        let h_off = b * hidden_size;

        for j in 0..hidden_size {
            // Input gate: i = σ(W_ii @ x + W_hi @ h + b_i)
            let mut i_val = b_i[j];
            for k in 0..input_size {
                i_val += w_ii[j * input_size + k] * x[x_off + k];
            }
            for k in 0..hidden_size {
                i_val += w_hi[j * hidden_size + k] * h[h_off + k];
            }
            let ig = sigmoid(i_val);

            // Forget gate: f = σ(W_if @ x + W_hf @ h + b_f)
            let mut f_val = b_f[j];
            for k in 0..input_size {
                f_val += w_if[j * input_size + k] * x[x_off + k];
            }
            for k in 0..hidden_size {
                f_val += w_hf[j * hidden_size + k] * h[h_off + k];
            }
            let fg = sigmoid(f_val);

            // Cell gate: g = tanh(W_ig @ x + W_hg @ h + b_g)
            let mut g_val = b_g[j];
            for k in 0..input_size {
                g_val += w_ig[j * input_size + k] * x[x_off + k];
            }
            for k in 0..hidden_size {
                g_val += w_hg[j * hidden_size + k] * h[h_off + k];
            }
            let gg = g_val.tanh();

            // Output gate: o = σ(W_io @ x + W_ho @ h + b_o)
            let mut o_val = b_o[j];
            for k in 0..input_size {
                o_val += w_io[j * input_size + k] * x[x_off + k];
            }
            for k in 0..hidden_size {
                o_val += w_ho[j * hidden_size + k] * h[h_off + k];
            }
            let og = sigmoid(o_val);

            // New cell state: c' = f * c + i * g
            c_new[h_off + j] = fg * c[h_off + j] + ig * gg;

            // New hidden: h' = o * tanh(c')
            h_new[h_off + j] = og * c_new[h_off + j].tanh();
        }
    }

    (h_new, c_new)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange() {
        let v = arange(0.0, 5.0, 1.0);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 0.0).abs() < 1e-10);
        assert!((v[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 0.0).abs() < 1e-10);
        assert!((v[4] - 1.0).abs() < 1e-10);
        assert!((v[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let (data, shape) = eye(3);
        assert_eq!(shape, [3, 3]);
        assert!((data[0] - 1.0).abs() < 1e-10); // [0,0]
        assert!((data[1] - 0.0).abs() < 1e-10); // [0,1]
        assert!((data[4] - 1.0).abs() < 1e-10); // [1,1]
        assert!((data[8] - 1.0).abs() < 1e-10); // [2,2]
    }

    #[test]
    fn test_silu() {
        let result = silu(&[0.0, 1.0, -1.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        // silu(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((result[1] - 0.7310585786).abs() < 1e-4);
    }

    #[test]
    fn test_where_select() {
        let result = where_select(&[1.0, 0.0, 1.0], &[10.0, 20.0, 30.0], &[-1.0, -2.0, -3.0]);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - (-2.0)).abs() < 1e-10);
        assert!((result[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_einsum_matmul() {
        // 2x2 identity × 2x2 identity = 2x2 identity
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let (result, shape) = einsum("ij,jk->ik", &[(&a, &[2, 2]), (&b, &[2, 2])]);
        assert_eq!(shape, vec![2, 2]);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_einsum_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (result, _shape) = einsum("i,i->", &[(&a, &[3]), (&b, &[3])]);
        assert!((result[0] - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_linear() {
        // input [1, 2], weight [2, 2] (out_features=2, in_features=2), bias [2]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bias = vec![0.5, 0.5];
        let (result, shape) = linear(&input, &weight, Some(&bias), &[1, 2], &[2, 2]);
        assert_eq!(shape, vec![1, 2]);
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_pixel_shuffle() {
        // 1×1×1×4, r=2 → 1×2×2×1
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (result, shape) = pixel_shuffle(&data, &[1, 1, 1, 4], 2);
        assert_eq!(shape, [1, 2, 2, 1]);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_upsample_nearest() {
        // 1×2×2×1, scale=2 → 1×4×4×1
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (result, shape) = upsample_nearest(&data, &[1, 2, 2, 1], 2);
        assert_eq!(shape, [1, 4, 4, 1]);
        assert_eq!(result.len(), 16);
        // Top-left 2×2 block should all be 1.0
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gru_cell() {
        // Simple: batch=1, input=2, hidden=2, all zeros except input
        let x = vec![1.0, 0.5];
        let h = vec![0.0, 0.0];
        let zeros2x2 = vec![0.0; 4];
        let zeros2 = vec![0.0; 2];

        let result = gru_cell(
            &x, &h, &zeros2x2, &zeros2x2, &zeros2, &zeros2x2, &zeros2x2, &zeros2, &zeros2x2,
            &zeros2x2, &zeros2, 1, 2, 2,
        );
        assert_eq!(result.len(), 2);
        // With all-zero weights, gates should be σ(0)=0.5
        // n = tanh(0) = 0, h' = 0.5*0 + 0.5*0 = 0
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_lstm_cell() {
        let x = vec![1.0, 0.5];
        let h = vec![0.0, 0.0];
        let c = vec![0.0, 0.0];
        let zeros2x2 = vec![0.0; 4];
        let zeros2 = vec![0.0; 2];

        let (h_new, c_new) = lstm_cell(
            &x, &h, &c, &zeros2x2, &zeros2x2, &zeros2, &zeros2x2, &zeros2x2, &zeros2,
            &zeros2x2, &zeros2x2, &zeros2, &zeros2x2, &zeros2x2, &zeros2, 1, 2, 2,
        );
        assert_eq!(h_new.len(), 2);
        assert_eq!(c_new.len(), 2);
    }
}
