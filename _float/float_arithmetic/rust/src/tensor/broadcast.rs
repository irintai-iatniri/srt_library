//! Broadcasting utilities for element-wise operations.
//!
//! Provides NumPy-style broadcasting for tensors with different shapes.
//!
//! Broadcasting rules:
//! 1. If tensors have different number of dimensions, prepend 1s to the smaller shape
//! 2. For each dimension, sizes must be equal or one of them must be 1
//! 3. The tensor with size 1 is "stretched" to match the other
//!
//! Also includes SRT-aligned Inflationary Broadcasting:
//! - `inflationary_broadcast`: Expand using golden ratio scaling
//! - `golden_inflationary_broadcast`: Pure φ-scaling expansion
//! - `consciousness_inflationary_broadcast`: Syntony-weighted expansion

use pyo3::prelude::*;

/// Compute the broadcast shape for two shapes.
///
/// Returns None if shapes are incompatible.
///
/// # Examples
/// ```
/// assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
/// assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
/// assert_eq!(broadcast_shape(&[2, 3], &[4]), None);
/// ```
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Iterate from the last dimension backwards
    for i in 0..max_ndim {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            // Incompatible dimensions
            return None;
        }
    }

    result.reverse();
    Some(result)
}

/// Check if shapes are broadcastable.
pub fn are_broadcastable(a: &[usize], b: &[usize]) -> bool {
    broadcast_shape(a, b).is_some()
}

/// Compute strides for a tensor with given shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Calculate the linear index in a tensor given multi-dimensional indices.
pub fn linear_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i * s)
        .sum()
}

/// Calculate multi-dimensional indices from a linear index.
pub fn unravel_index(linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = linear_idx;

    for &stride in &strides {
        indices.push(remaining / stride);
        remaining %= stride;
    }

    indices
}

/// Map a linear index from the output tensor to source tensor indices,
/// handling broadcasting (size-1 dimensions map to index 0).
pub fn broadcast_index(linear_idx: usize, output_shape: &[usize], source_shape: &[usize]) -> usize {
    let output_indices = unravel_index(linear_idx, output_shape);
    let output_strides = compute_strides(output_shape);
    debug_assert_eq!(output_indices.len(), output_strides.len());
    let source_strides = compute_strides(source_shape);

    let offset = output_shape.len().saturating_sub(source_shape.len());
    let mut source_idx = 0;

    for (i, &dim) in source_shape.iter().enumerate() {
        let out_idx = output_indices[offset + i];
        // If dimension is 1, broadcast (always use index 0)
        let src_idx = if dim == 1 { 0 } else { out_idx };
        source_idx += src_idx * source_strides[i];
    }

    source_idx
}

/// Apply a binary operation with broadcasting.
///
/// # Arguments
/// * `a` - First tensor data
/// * `a_shape` - First tensor shape
/// * `b` - Second tensor data
/// * `b_shape` - Second tensor shape
/// * `op` - Binary operation to apply
///
/// # Returns
/// Result tensor with broadcast shape
pub fn broadcast_binary_op<F>(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    op: F,
) -> Option<(Vec<f64>, Vec<usize>)>
where
    F: Fn(f64, f64) -> f64,
{
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_size: usize = out_shape.iter().product();

    let mut result = Vec::with_capacity(out_size);

    for i in 0..out_size {
        let a_idx = broadcast_index(i, &out_shape, a_shape);
        let b_idx = broadcast_index(i, &out_shape, b_shape);
        result.push(op(a[a_idx], b[b_idx]));
    }

    Some((result, out_shape))
}

/// Broadcast addition.
pub fn broadcast_add(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x + y)
}

/// Broadcast multiplication.
pub fn broadcast_mul(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x * y)
}

/// Broadcast subtraction.
pub fn broadcast_sub(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x - y)
}

/// Broadcast division.
pub fn broadcast_div(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x / y)
}

// ============================================================================
// Inflationary Broadcasting - Thought Expansion Operators
// ============================================================================

/// Inflationary broadcasting: Expand a seed tensor into higher-dimensional space
/// using golden ratio scaling. This implements the "Inflationary Operator" where
/// a fundamental thought/idea expands to fill a reality space.
///
/// The expansion follows the pattern: seed → φ-scaled inflation → target space
///
/// # Arguments
/// * `seed` - Source tensor data (fundamental idea/thought)
/// * `seed_shape` - Source tensor shape
/// * `target_shape` - Target expanded shape
/// * `inflation_factor` - Golden ratio scaling factor (default: φ)
///
/// # Returns
/// Inflated tensor with target shape
pub fn inflationary_broadcast(
    seed: &[f64],
    seed_shape: &[usize],
    target_shape: &[usize],
    inflation_factor: Option<f64>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    let phi = inflation_factor.unwrap_or(1.618033988749895); // Golden ratio

    // Check if expansion is possible (target dimensions must be >= seed dimensions)
    if seed_shape.len() > target_shape.len() {
        return None;
    }

    // Calculate expansion ratios for each dimension
    let mut expansion_ratios = Vec::new();
    let offset = target_shape.len() - seed_shape.len();

    for i in 0..seed_shape.len() {
        let seed_dim = seed_shape[i];
        let target_dim = target_shape[offset + i];

        if target_dim < seed_dim {
            return None; // Cannot shrink dimensions
        }

        expansion_ratios.push(target_dim as f64 / seed_dim as f64);
    }

    // For new dimensions (prepended), use pure inflation
    for _ in 0..offset {
        expansion_ratios.insert(0, phi);
    }

    // Create inflated tensor
    let target_size: usize = target_shape.iter().product();
    let mut result = Vec::with_capacity(target_size);

    for linear_idx in 0..target_size {
        let target_indices = unravel_index(linear_idx, target_shape);

        // Map target indices back to seed space
        let mut seed_indices = Vec::new();
        let mut inflation_weight = 1.0;

        for (i, &target_idx) in target_indices.iter().enumerate() {
            let ratio = expansion_ratios[i];
            let seed_dim = if i >= offset {
                seed_shape[i - offset]
            } else {
                1 // Virtual dimension for new axes
            };

            // Map target index to seed index with inflation weighting
            let seed_idx = if seed_dim == 1 {
                0
            } else {
                (target_idx as f64 / ratio) as usize
            };

            seed_indices.push(seed_idx.min(seed_dim.saturating_sub(1)));

            // Accumulate inflation weight (golden ratio scaling)
            inflation_weight *= phi.powf((target_idx as f64) / (target_shape[i] as f64));
        }

        // Get seed value and apply inflationary scaling
        let seed_linear_idx = linear_index(&seed_indices, &compute_strides(seed_shape));
        let seed_value = seed.get(seed_linear_idx).copied().unwrap_or(0.0);

        result.push(seed_value * inflation_weight);
    }

    Some((result, target_shape.to_vec()))
}

/// Golden inflationary broadcast: Expand using pure golden ratio scaling
/// This implements the consciousness growth pattern where ideas expand
/// according to φ^n scaling laws observed in biological systems.
pub fn golden_inflationary_broadcast(
    seed: &[f64],
    seed_shape: &[usize],
    target_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    inflationary_broadcast(seed, seed_shape, target_shape, Some(1.618033988749895))
}

/// Consciousness inflationary broadcast: Scale by both size and syntony
/// Higher syntony seeds expand more effectively, implementing the
/// principle that coherent thoughts propagate better than incoherent ones.
pub fn consciousness_inflationary_broadcast(
    seed: &[f64],
    syntony_values: &[f64], // Syntony scores for each seed element
    seed_shape: &[usize],
    target_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    // First do basic inflationary broadcast
    let (mut result, shape) = inflationary_broadcast(seed, seed_shape, target_shape, None)?;

    // Then apply consciousness weighting
    let phi: f64 = 1.618033988749895;
    let target_size: usize = shape.iter().product();

    for i in 0..target_size {
        let target_indices = unravel_index(i, &shape);

        // Map back to seed space to get syntony weight
        let offset = shape.len() - seed_shape.len();
        let mut seed_indices = Vec::new();

        for j in 0..seed_shape.len() {
            let ratio = shape[offset + j] as f64 / seed_shape[j] as f64;
            let seed_idx = (target_indices[offset + j] as f64 / ratio) as usize;
            seed_indices.push(seed_idx.min(seed_shape[j].saturating_sub(1)));
        }

        let seed_linear_idx = linear_index(&seed_indices, &compute_strides(seed_shape));
        let syntony_weight = syntony_values.get(seed_linear_idx).copied().unwrap_or(1.0);

        // Apply consciousness scaling: higher syntony = better propagation
        result[i] *= phi.powf(syntony_weight);
    }

    Some((result, shape))
}

// ============================================================================
// Additional In-Place Operations (for compatibility)
// ============================================================================

/// In-place clamp.
pub fn inplace_clamp(data: &mut [f64], min: f64, max: f64) {
    for x in data.iter_mut() {
        *x = x.clamp(min, max);
    }
}

/// In-place golden weight: x = exp(-|n|²/φ)
pub fn inplace_golden_weight(data: &mut [f64], phi: f64) {
    for x in data.iter_mut() {
        *x = (-*x / phi).exp();
    }
}

/// In-place scalar addition.
pub fn inplace_add_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x += scalar;
    }
}

/// In-place scalar multiplication.
pub fn inplace_mul_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x *= scalar;
    }
}

/// In-place scalar subtraction.
pub fn inplace_sub_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x -= scalar;
    }
}

/// In-place scalar division.
pub fn inplace_div_scalar(data: &mut [f64], scalar: f64) {
    let inv = 1.0 / scalar;
    for x in data.iter_mut() {
        *x *= inv;
    }
}

/// In-place negation.
pub fn inplace_negate(data: &mut [f64]) {
    for x in data.iter_mut() {
        *x = -*x;
    }
}

/// In-place absolute value.
pub fn inplace_abs(data: &mut [f64]) {
    for x in data.iter_mut() {
        *x = x.abs();
    }
}

// =============================================================================
// PyO3 Bindings for SRT Inflationary Broadcasting
// =============================================================================

/// Python-exposed inflationary broadcast with golden ratio scaling.
///
/// Expands a seed tensor to a target shape using φ-weighted inflation.
/// Implements the cosmological inflation principle in tensor space.
///
/// Args:
///     seed: Source tensor data (flattened)
///     seed_shape: Shape of source tensor
///     target_shape: Desired output shape
///     inflation_factor: Optional custom factor (default: φ ≈ 1.618)
///
/// Returns:
///     Tuple of (inflated_data, output_shape) or None if incompatible
#[pyfunction]
#[pyo3(name = "inflationary_broadcast")]
pub fn py_inflationary_broadcast(
    seed: Vec<f64>,
    seed_shape: Vec<usize>,
    target_shape: Vec<usize>,
    inflation_factor: Option<f64>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    inflationary_broadcast(&seed, &seed_shape, &target_shape, inflation_factor)
}

/// Python-exposed golden inflationary broadcast (pure φ scaling).
///
/// Expands tensor using the golden ratio, implementing the growth
/// pattern observed in biological systems and consciousness expansion.
#[pyfunction]
#[pyo3(name = "golden_inflationary_broadcast")]
pub fn py_golden_inflationary_broadcast(
    seed: Vec<f64>,
    seed_shape: Vec<usize>,
    target_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    golden_inflationary_broadcast(&seed, &seed_shape, &target_shape)
}

/// Python-exposed consciousness inflationary broadcast.
///
/// Expands tensor with syntony-weighted scaling. Higher syntony elements
/// propagate more effectively, implementing the principle that coherent
/// patterns expand better than incoherent ones.
///
/// Args:
///     seed: Source tensor data
///     syntony_values: Syntony score for each seed element
///     seed_shape: Shape of source tensor
///     target_shape: Desired output shape
#[pyfunction]
#[pyo3(name = "consciousness_inflationary_broadcast")]
pub fn py_consciousness_inflationary_broadcast(
    seed: Vec<f64>,
    syntony_values: Vec<f64>,
    seed_shape: Vec<usize>,
    target_shape: Vec<usize>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    consciousness_inflationary_broadcast(&seed, &syntony_values, &seed_shape, &target_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[5, 4], &[1]), Some(vec![5, 4]));
        assert_eq!(broadcast_shape(&[1], &[5, 4]), Some(vec![5, 4]));
        assert_eq!(broadcast_shape(&[2, 3], &[4]), None);
    }

    #[test]
    fn test_broadcast_add() {
        // [2, 3] + [3] -> [2, 3]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let (result, shape) = broadcast_add(&a, &[2, 3], &b, &[3]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_mul_scalar_like() {
        // [3, 4] * [1] -> [3, 4]
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b = vec![2.0];
        let (result, shape) = broadcast_mul(&a, &[3, 4], &b, &[1]).unwrap();
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[11], 24.0);
    }

    #[test]
    fn test_inplace_ops() {
        let mut data = vec![1.0, 2.0, 3.0];
        inplace_add_scalar(&mut data, 10.0);
        assert_eq!(data, vec![11.0, 12.0, 13.0]);

        inplace_mul_scalar(&mut data, 2.0);
        assert_eq!(data, vec![22.0, 24.0, 26.0]);

        inplace_negate(&mut data);
        assert_eq!(data, vec![-22.0, -24.0, -26.0]);

        inplace_abs(&mut data);
        assert_eq!(data, vec![22.0, 24.0, 26.0]);
    }
}
