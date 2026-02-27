//! Tensor reduction operations (sum, mean, max, min, argmax, argmin).

use crate::tensor::broadcast::compute_strides;

/// Reduce a tensor along a specific axis using a reduction function.
///
/// # Arguments
/// * `data` - Input data (flattened)
/// * `shape` - Input shape
/// * `axis` - Axis to reduce along
/// * `init` - Initial value for the accumulator
/// * `op` - Reduction operation (accumulator, value) -> new_accumulator
///
/// # Returns
/// (reduced_data, reduced_shape)
pub fn reduce_axis<T, R, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    init: R,
    op: F,
) -> (Vec<R>, Vec<usize>)
where
    T: Copy,
    R: Copy + Clone,
    F: Fn(R, T) -> R,
{
    let ndim = shape.len();
    assert!(axis < ndim, "Axis out of bounds");

    let mut out_shape = shape.to_vec();
    // Reduce dimension at axis involves removing it or setting it to 1?
    // Usually keepdims=False in numpy removes it.
    // For internal consistency with standard tensor ops, let's remove it for now,
    // or return the shape with that dim removed.
    out_shape.remove(axis);

    let dim_size = shape[axis];
    let out_size: usize = out_shape.iter().product();
    let mut result = vec![init; out_size];

    let strides = compute_strides(shape);
    let axis_stride = strides[axis];
    
    // We iterate over the output tensor linear indices
    // For each output index, we reconstruct the input indices (with the reduced axis missing)
    // Then we iterate over the reduced axis.
    
    // Actually, it is more efficient to iterate over input and accumulate.
    // But initializing result is easy.
    // We need to map input index to output index.
    
    // Calculate strides for the output tensor
    let out_strides = compute_strides(&out_shape);

    // Iterating over input is easier if we can compute output index cheaply.
    // Specially when reducing.
    // Given input index i, what is its corresponding output index?
    // We just need to zero out the contribution of the axis dimension from the linear index?
    // No, linear index mapping is complex.
    
    // Let's use the standard "outer, inner" loop approach if possible, but shape is dynamic.
    
    // Better: Iterate over all elements. Compute multi-index. Remove axis index. 
    // Compute linear index for output. Update.
    // This is O(N * ndim) which is slow for small N inner loops.
    // Optimization: Precomputations.
    
    // Let's iterate over output indices. For each, we construct the partial multi-index,
    // insert the axis index, and iterate.
    
    for i in 0..out_size {
        // 1. Get multi-index for this output position
        let mut indices = Vec::with_capacity(ndim);
        let mut remaining = i;
        for &s in &out_strides {
            indices.push(remaining / s);
            remaining %= s;
        }
        
        // 2. Iterate along the reduction axis
        let mut acc = init;
        
        // We need to insert the loop variable at 'axis' position in indices.
        // To calculate linear offset.
        // Base offset without the axis term:
        // We know indices maps to 'i' in output.
        // We need 'i' in input corresponding to these indices, assuming axis index is 0.
        // We can't reuse 'i' because strides are different.
        
        // Re-calcluate input base offset
        let mut base_offset = 0;
        let mut out_dim_idx = 0;
        for k in 0..ndim {
             if k == axis { continue; }
             base_offset += indices[out_dim_idx] * strides[k];
             out_dim_idx += 1;
        }
        
        for j in 0..dim_size {
            let idx = base_offset + j * axis_stride;
            acc = op(acc, data[idx]);
        }
        
        result[i] = acc;
    }

    (result, out_shape)
}


/// Reduce axis for ArgMax/ArgMin.
/// Returns indices of the extreme values.
pub fn arg_reduce_axis<F>(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    init_val: f64, // Typically -inf or +inf
    compare: F, // (current_best_val, new_val) -> bool (true if new is better)
) -> (Vec<usize>, Vec<usize>) 
where 
    F: Fn(f64, f64) -> bool
{
     let ndim = shape.len();
     assert!(axis < ndim, "Axis out of bounds");
 
     let mut out_shape = shape.to_vec();
     out_shape.remove(axis);
 
     let dim_size = shape[axis];
     let out_size: usize = out_shape.iter().product();
     
     // Store the index of the best value
     let mut result_indices = vec![0; out_size];
     // Keep track of best values during reduction
     
     let strides = compute_strides(shape);
     let axis_stride = strides[axis];
     let out_strides = compute_strides(&out_shape);
 
     for i in 0..out_size {
         let mut indices = Vec::with_capacity(ndim - 1);
         let mut remaining = i;
         for &s in &out_strides {
             indices.push(remaining / s);
             remaining %= s;
         }
         
         let mut base_offset = 0;
         let mut out_dim_idx = 0;
         for k in 0..ndim {
              if k == axis { continue; }
              base_offset += indices[out_dim_idx] * strides[k];
              out_dim_idx += 1;
         }
         
         let mut best_val = init_val;
         let mut best_idx = 0;
         
         for j in 0..dim_size {
             let idx = base_offset + j * axis_stride;
             let val = data[idx];
             if j == 0 || compare(best_val, val) {
                 best_val = val;
                 best_idx = j;
             }
         }
         
         result_indices[i] = best_idx;
     }
 
     (result_indices, out_shape)
}

pub fn sum(data: &[f64], shape: &[usize], axis: usize) -> (Vec<f64>, Vec<usize>) {
    reduce_axis(data, shape, axis, 0.0, |acc, x| acc + x)
}

pub fn mean(data: &[f64], shape: &[usize], axis: usize) -> (Vec<f64>, Vec<usize>) {
    let dim_size = shape[axis] as f64;
    let (mut sums, out_shape) = sum(data, shape, axis);
    for x in sums.iter_mut() {
        *x /= dim_size;
    }
    (sums, out_shape)
}

pub fn max(data: &[f64], shape: &[usize], axis: usize) -> (Vec<f64>, Vec<usize>) {
    reduce_axis(data, shape, axis, f64::NEG_INFINITY, |acc, x| acc.max(x))
}

pub fn min(data: &[f64], shape: &[usize], axis: usize) -> (Vec<f64>, Vec<usize>) {
    reduce_axis(data, shape, axis, f64::INFINITY, |acc, x| acc.min(x))
}

pub fn argmax(data: &[f64], shape: &[usize], axis: usize) -> (Vec<usize>, Vec<usize>) {
    // compare(best, new) -> true if new > best
    arg_reduce_axis(data, shape, axis, f64::NEG_INFINITY, |best, new| new > best)
}

pub fn argmin(data: &[f64], shape: &[usize], axis: usize) -> (Vec<usize>, Vec<usize>) {
    // compare(best, new) -> true if new < best
    arg_reduce_axis(data, shape, axis, f64::INFINITY, |best, new| new < best)
}
