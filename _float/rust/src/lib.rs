use pyo3::prelude::*;

mod autograd;
pub mod exact;
mod gnosis;
mod golden_gelu;
mod hierarchy;
mod hypercomplex;
mod linalg;
mod math_utils;
mod prime_selection;
mod resonant;
mod spectral;
mod tensor;
mod transcendence;
mod winding;
mod sna;

use autograd::{
    py_backward_add, py_backward_mul, py_backward_softmax,
    py_backward_layernorm, py_backward_phi_residual, py_load_autograd_kernels,
};

use hypercomplex::{Octonion, Quaternion, Sedenion};
use tensor::cuda::{AsyncTensorTransfer, TransferComputeOverlap};
use tensor::srt_kernels;
use tensor::storage::{
    cuda_device_count, cuda_is_available, srt_apply_correction, srt_compute_syntony,
    srt_dhsr_cycle, srt_e8_batch_projection, srt_golden_gaussian_weights, srt_scale_phi,
    srt_theta_series, TensorStorage,
    // Fixed-Point kernels
    srt_compute_syntony_fp64, srt_dhsr_cycle_fp64, srt_differentiation_fp64,
    srt_harmonization_fp64, srt_differentiation_full_fp64, srt_laplacian_1d_fp64,
};
use tensor::data_loading::{
    DataBatch, DataType, Endianness, GoldenExactConverter, SRTBinaryLoader,
    SRTCSVParser, SRTDataPipeline, StreamingCSVIterator,
};
use tensor::precision_policy::{PrecisionPolicy, get_srt_operation_policy};


// SRT CUDA operations (toroidal math, gnosis masks, autograd, GEMM)
// Kernel loader functions are registered directly in the module function

// SRT Inflationary Broadcasting
use tensor::broadcast::{
    py_consciousness_inflationary_broadcast, py_golden_inflationary_broadcast,
    py_inflationary_broadcast,
};

// Causal History DAG for DHSR tracking
use tensor::causal_history::{
    create_causal_tracker, d4_consciousness_threshold, PyCausalHistoryTracker,
};

// Golden Momentum optimizer
use tensor::srt_optimization::GoldenMomentum;
use tensor::storage::{
    srt_memory_resonance, srt_pool_stats, srt_reserve_memory, srt_transfer_stats,
    srt_wait_for_resonance,
};

// Winding state and enumeration
use winding::{
    count_windings, enumerate_windings, enumerate_windings_by_norm, enumerate_windings_exact_norm,
    WindingState, WindingStateIterator,
};

// Prime selection, gnosis, and transcendence
use gnosis::register_gnosis;
use math_utils::register_math_utils;
use prime_selection::register_extended_prime_selection;
use transcendence::register_transcendence;

// Spectral operations
use spectral::{
    compute_eigenvalues,
    compute_golden_weights,
    compute_knot_eigenvalues,
    compute_norm_squared,
    count_by_generation,
    filter_by_generation,
    heat_kernel_derivative,
    heat_kernel_trace,
    heat_kernel_weighted,
    // Knot Laplacian operations
    knot_eigenvalue,
    knot_heat_kernel_trace,
    knot_spectral_zeta,
    knot_spectral_zeta_complex,
    partition_function,
    spectral_zeta,
    spectral_zeta_weighted,
    theta_series_derivative,
    theta_series_evaluate,
    theta_series_weighted,
    theta_sum_combined,
};

// New exact arithmetic types
use exact::{CorrectionLevel, FundamentalConstant, GoldenExact, PySymExpr, Rational, Structure, SyntonicExact, SyntonicDual, 
    PythagoreanTriple, RationalRotator, TernarySolver, WaveLayer, BASIS_SCALE};

// Resonant Engine types
use resonant::{RESConfig, RESResult, ResonantEvolver, ResonantTensor};

// E8 Lattice and Golden Projector wrappers
use resonant::{
    py_compute_8d_weight, py_e8_generate_roots, py_e8_generate_weights, py_golden_project_parallel,
    py_golden_project_perp, py_golden_projector_phi, py_golden_projector_q, py_is_in_golden_cone,
};

// Neural Network E8 Lattice and Golden Projector wrappers
use resonant::{
    py_compute_8d_weight_nn, py_e8_generate_roots_nn, py_e8_generate_weights_nn,
    py_golden_project_parallel_nn, py_golden_project_perp_nn, py_golden_projector_phi_nn,
    py_golden_projector_q_nn, py_is_in_golden_cone_nn,
};

// Number theory and syntony wrappers
use resonant::py_wrappers::{
    py_aggregate_syntony,
    py_are_broadcastable,
    py_avg_pool2d,
    py_batch_cross_entropy_loss,
    py_batch_winding_syntony,
    py_broadcast_add,
    py_broadcast_div,
    py_broadcast_mul,
    // Broadcasting
    py_broadcast_shape,
    py_broadcast_sub,
    py_compute_snap_gradient,
    py_compute_winding_syntony,
    // Convolution
    py_conv2d,
    py_cross_entropy_loss,
    py_crystallize_with_dwell_legacy,
    py_e_star,
    py_estimate_syntony_from_probs,
    py_get_q_deficit,
    py_get_target_syntony,
    py_global_avg_pool2d,
    py_golden_decay_loss,
    py_golden_weight,
    py_golden_weights,
    py_inplace_abs,
    // In-place
    py_inplace_add_scalar,
    py_inplace_clamp,
    py_inplace_div_scalar,
    py_inplace_golden_weight,
    py_inplace_mul_scalar,
    py_inplace_negate,
    py_inplace_sub_scalar,
    py_is_square_free,
    py_linear_index,
    py_max_pool2d,
    py_mertens,
    py_mobius,
    py_mse_loss,
    py_phase_alignment_loss,
    py_snap_distance,
    py_softmax,
    py_standard_mode_norms,
    py_syntonic_loss,
    py_syntony_loss,
    py_syntony_loss_srt,
};

// Trilinear operations (CUDA)
use tensor::py_srt_cuda_ops::{
    py_arg_c128, py_atan2_toroidal, py_attractor_memory_update, py_autograd_filter,
    py_bilinear_f64, py_cos_toroidal, py_cuda_dgemm, py_cuda_sgemm, py_fractal_gnosis_mask,
    py_gather_e8_roots_f64, py_gather_lucas_shadow_f64, py_gather_pisano_hooked_f64,
    py_gather_transcendence_gate_f64, py_gnosis_mask, py_golden_entropy, py_phase_syntony_c128,
    py_phi_exp, py_phi_exp_inv, py_scatter_consciousness_threshold_f64,
    py_scatter_golden_cone_f64, py_scatter_golden_f64, py_scatter_mersenne_stable_f64,
    py_sin_toroidal, py_syntony_metric, py_temporal_gnosis_mask, py_trilinear_acausal_f64,
    py_trilinear_causal_f64, py_trilinear_f64, py_trilinear_golden_decay_f64,
    py_trilinear_phi_weighted_f64, py_trilinear_retrocausal_f64, py_trilinear_symmetric_f64,
    py_trilinear_toroidal_f64, py_adaptive_gnosis_mask,
    // Hierarchy corrections
    py_hierarchy_e8_correction, py_hierarchy_e7_correction, py_hierarchy_coxeter_kissing_correction,
    // Attractor kernels (retrocausal training)
    py_load_attractor_kernels, py_attractor_centroid, py_retrocausal_harmonize, py_attractor_distance,
    // DHSR kernels
    py_dhsr_step_fused, py_damping_cascade, py_differentiation_full, py_fourier_project_batch,
    // Static kernel wrappers (use host_* FFI functions)
    py_static_sin_toroidal, py_static_cos_toroidal, py_static_phi_exp, py_static_phi_exp_inv,
    py_static_atan2_toroidal, py_static_golden_entropy,
};

// Hierarchy Correction (SRT-Zero) - Functions imported for direct Rust usage
use hierarchy::{
    apply_chain, apply_correction, apply_correction_uniform, apply_e7_correction,
    apply_coxeter_kissing_correction, apply_collapse_threshold_correction, apply_special,
    apply_suppression, compute_e_star_n, e6_coxeter, e6_dim, e6_fundamental, e6_positive_roots,
    e6_rank, e6_roots, e7_coxeter, e7_dim, e7_fundamental, e7_positive_roots, e7_rank,
    e7_roots, e8_coxeter, e8_dim, e8_positive_roots, e8_rank, e8_roots, coxeter_kissing_720,
    d4_coxeter, d4_dim, d4_kissing, d4_rank, f4_dim, f4_rank, g2_dim, g2_rank, hierarchy_exponent,
    init_divisors,
    // New functions from Universal Syntony Correction Hierarchy
    apply_e6_correction, apply_mersenne_correction, apply_lucas_correction,
    apply_fermat_correction, apply_loop_integral, apply_phi_power_correction,
    // Prime tower accessors
    mersenne_m2, mersenne_m3, mersenne_m5, mersenne_m7,
    lucas_l4, lucas_l5, lucas_l6, lucas_l7, lucas_l11,
    fermat_f0, fermat_f1, fermat_f2, fermat_f3, fermat_f4,
};

/// Validate hierarchy correction functions are working properly
/// This function uses all imported hierarchy functions to ensure they're callable from Rust
fn validate_hierarchy_corrections() {
    // Test basic correction functions
    let test_values = vec![1.0, 2.0, 3.0];
    let divisors = vec![8.0, 4.0, 2.0];
    let signs = vec![1, -1, 1];
    let _corrected = apply_correction(test_values, divisors, signs).unwrap();
    let _corrected_uniform = apply_correction_uniform(vec![1.0, 2.0, 3.0], 8.0, 1).unwrap();
    let _chained = apply_chain(vec![1.0, 2.0], vec![4.0, 6.0], vec![1, 1], vec![2, 2], vec![0, 0]).unwrap();

    // Test E7 corrections
    let _e7_corrected = apply_e7_correction(1.0, 0);

    // Test special corrections
    let _special = apply_special(vec![1.0, 2.0], vec![0, 1]).unwrap();

    // Test suppression
    let _suppressed = apply_suppression(vec![1.0, 2.0], 0).unwrap();

    // Test coxeter kissing corrections
    let _coxeter = apply_coxeter_kissing_correction(1.0);

    // Test collapse threshold
    let _collapsed = apply_collapse_threshold_correction(1.0);

    // Test E* computation
    let _e_star_vals = compute_e_star_n(vec![1.0, 2.0], vec![4.0, 6.0], vec![1, 1], 10);

    // Test dimension constants (these are used in expressions)
    let _total_dim = e8_dim() + e7_dim() + e6_dim() + d4_dim() + f4_dim() + g2_dim();
    let _total_roots = e8_roots() + e7_roots() + e6_roots();
    let _coxeter_sum = e8_coxeter() + e7_coxeter() + e6_coxeter() + d4_coxeter();

    // Test rank and fundamental representations
    let _ranks = e8_rank() + e7_rank() + e6_rank() + d4_rank() + f4_rank() + g2_rank();
    let _fundamentals = e7_fundamental() + e6_fundamental();

    // Test positive roots
    let _pos_roots = e8_positive_roots() + e7_positive_roots() + e6_positive_roots();

    // Test kissing numbers and exponents
    let _kissing = d4_kissing() + coxeter_kissing_720();
    let _exponent = hierarchy_exponent();

    // Test divisors initialization
    let _divisors = init_divisors(vec![1.0, 2.0, 4.0]);

    // Test new E6 correction
    let _e6_corrected = apply_e6_correction(1.0, 0);

    // Test prime tower corrections
    let _mersenne = apply_mersenne_correction(1.0, 2, 1);
    let _lucas = apply_lucas_correction(1.0, 4, 1);
    let _fermat = apply_fermat_correction(1.0, 0, 1);

    // Test loop integral and phi power corrections
    let _loop = apply_loop_integral(1.0, 4, 1);
    let _phi_power = apply_phi_power_correction(1.0, 3, 1);

    // Test prime tower accessors
    let _mersenne_sum = mersenne_m2() + mersenne_m3() + mersenne_m5() + mersenne_m7();
    let _lucas_sum = lucas_l4() + lucas_l5() + lucas_l6() + lucas_l7() + lucas_l11();
    let _fermat_sum = fermat_f0() + fermat_f1() + fermat_f2() + fermat_f3() + fermat_f4();
}

// Linear algebra operations
use linalg::{
    py_bmm,
    py_mm,
    py_mm_add,
    py_mm_corrected,
    // New generalized functions
    py_mm_gemm,
    py_mm_golden_phase,
    py_mm_golden_weighted,
    py_mm_hn,
    py_mm_nh,
    py_mm_nt,
    py_mm_phi,
    py_mm_q_corrected_direct,
    py_mm_tn,
    py_mm_tt,
    py_phi_antibracket,
    py_phi_bracket,
    py_projection_sum,
    py_q_correction_scalar,
};

// =============================================================================
// SRT Constant Functions (Python-accessible)
// =============================================================================

/// Get the golden ratio φ = (1 + √5) / 2
#[pyfunction]
fn srt_phi() -> f64 {
    srt_kernels::PHI
}

/// Get the golden ratio inverse φ⁻¹ = φ - 1
#[pyfunction]
fn srt_phi_inv() -> f64 {
    srt_kernels::PHI_INV
}

/// Get the q-deficit value q = W(∞) - 1 ≈ 0.027395
#[pyfunction]
fn srt_q_deficit() -> f64 {
    srt_kernels::Q_DEFICIT
}

/// Get π (pi) constant
#[pyfunction]
fn srt_pi() -> f64 {
    std::f64::consts::PI
}

/// Get e (Euler's number) constant
#[pyfunction]
fn srt_e() -> f64 {
    std::f64::consts::E
}

/// Get structure dimension by index
/// 0: E₈ dim (248), 1: E₈ roots (240), 2: E₈ pos (120),
/// 3: E₆ dim (78), 4: E₆ cone (36), 5: E₆ 27 (27),
/// 6: D₄ kissing (24), 7: G₂ dim (14)
#[pyfunction]
fn srt_structure_dimension(index: i32) -> i32 {
    srt_kernels::get_structure_dimension(index)
}

/// Compute correction factor (1 + sign * q / N)
#[pyfunction]
fn srt_correction_factor(structure_index: i32, sign: i32) -> f64 {
    let n = srt_kernels::get_structure_dimension(structure_index);
    srt_kernels::cpu_correction_factor(n, sign)
}


// =============================================================================
// Scalar Math Functions (Python math module replacement)
// =============================================================================

/// Square root: √x
#[pyfunction]
fn srt_sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Exponential: e^x
#[pyfunction]
fn srt_exp(x: f64) -> f64 {
    x.exp()
}

/// Natural logarithm: ln(x)
#[pyfunction]
fn srt_log(x: f64) -> f64 {
    x.ln()
}

/// Logarithm base 10: log₁₀(x)
#[pyfunction]
fn srt_log10(x: f64) -> f64 {
    x.log10()
}

/// Logarithm base 2: log₂(x)
#[pyfunction]
fn srt_log2(x: f64) -> f64 {
    x.log2()
}

/// ln(1 + x) - numerically stable for small x
#[pyfunction]
fn srt_log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// e^x - 1 - numerically stable for small x
#[pyfunction]
fn srt_expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Sine: sin(x)
#[pyfunction]
fn srt_sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine: cos(x)
#[pyfunction]
fn srt_cos(x: f64) -> f64 {
    x.cos()
}

/// Tangent: tan(x)
#[pyfunction]
fn srt_tan(x: f64) -> f64 {
    x.tan()
}

/// Arc sine: arcsin(x)
#[pyfunction]
fn srt_asin(x: f64) -> f64 {
    x.asin()
}

/// Arc cosine: arccos(x)
#[pyfunction]
fn srt_acos(x: f64) -> f64 {
    x.acos()
}

/// Arc tangent: arctan(x)
#[pyfunction]
fn srt_atan(x: f64) -> f64 {
    x.atan()
}

/// Arc tangent of y/x, handling quadrants: atan2(y, x)
#[pyfunction]
fn srt_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Hyperbolic sine: sinh(x)
#[pyfunction]
fn srt_sinh(x: f64) -> f64 {
    x.sinh()
}

/// Hyperbolic cosine: cosh(x)
#[pyfunction]
fn srt_cosh(x: f64) -> f64 {
    x.cosh()
}

/// Hyperbolic tangent: tanh(x)
#[pyfunction]
fn srt_tanh(x: f64) -> f64 {
    x.tanh()
}

/// Inverse hyperbolic sine: asinh(x)
#[pyfunction]
fn srt_asinh(x: f64) -> f64 {
    x.asinh()
}

/// Inverse hyperbolic cosine: acosh(x)
#[pyfunction]
fn srt_acosh(x: f64) -> f64 {
    x.acosh()
}

/// Inverse hyperbolic tangent: atanh(x)
#[pyfunction]
fn srt_atanh(x: f64) -> f64 {
    x.atanh()
}

/// Power: x^y
#[pyfunction]
fn srt_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Power with integer exponent: x^n (faster)
#[pyfunction]
fn srt_powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// Floor: ⌊x⌋
#[pyfunction]
fn srt_floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling: ⌈x⌉
#[pyfunction]
fn srt_ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round to nearest integer
#[pyfunction]
fn srt_round(x: f64) -> f64 {
    x.round()
}

/// Truncate toward zero
#[pyfunction]
fn srt_trunc(x: f64) -> f64 {
    x.trunc()
}

/// Fractional part: x - ⌊x⌋
#[pyfunction]
fn srt_fract(x: f64) -> f64 {
    x.fract()
}

/// Absolute value: |x|
#[pyfunction]
fn srt_abs(x: f64) -> f64 {
    x.abs()
}

/// Sign of x: -1, 0, or 1
#[pyfunction]
fn srt_signum(x: f64) -> f64 {
    x.signum()
}

/// Copy sign: magnitude of x with sign of y
#[pyfunction]
fn srt_copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// Fused multiply-add: x * y + z (single rounding)
#[pyfunction]
fn srt_fma(x: f64, y: f64, z: f64) -> f64 {
    x.mul_add(y, z)
}

/// Euclidean distance: √(x² + y²) (avoiding overflow)
#[pyfunction]
fn srt_hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Cube root: ∛x
#[pyfunction]
fn srt_cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Minimum of two values
#[pyfunction]
fn srt_min(x: f64, y: f64) -> f64 {
    x.min(y)
}

/// Maximum of two values
#[pyfunction]
fn srt_max(x: f64, y: f64) -> f64 {
    x.max(y)
}

/// Clamp x to [lo, hi]
#[pyfunction]
fn srt_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}

/// Convert radians to degrees
#[pyfunction]
fn srt_degrees(x: f64) -> f64 {
    x.to_degrees()
}

/// Convert degrees to radians
#[pyfunction]
fn srt_radians(x: f64) -> f64 {
    x.to_radians()
}

/// Check if x is NaN
#[pyfunction]
fn srt_isnan(x: f64) -> bool {
    x.is_nan()
}

/// Check if x is infinite
#[pyfunction]
fn srt_isinf(x: f64) -> bool {
    x.is_infinite()
}

/// Check if x is finite
#[pyfunction]
fn srt_isfinite(x: f64) -> bool {
    x.is_finite()
}

/// Linear interpolation: a + t * (b - a)
#[pyfunction]
fn srt_lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Golden lerp: a * (1 - φ⁻¹) + b * φ⁻¹
#[pyfunction]
fn srt_golden_lerp(a: f64, b: f64) -> f64 {
    let phi_inv = srt_kernels::PHI_INV;
    a * (1.0 - phi_inv) + b * phi_inv
}

/// Golden exponential: φ^x
#[pyfunction]
fn srt_phi_power(x: f64) -> f64 {
    srt_kernels::PHI.powf(x)
}

/// Inverse golden exponential: φ^(-x)
#[pyfunction]
fn srt_phi_power_inv(x: f64) -> f64 {
    srt_kernels::PHI.powf(-x)
}

// =============================================================================
// Integer Math Functions
// =============================================================================

/// Greatest common divisor
#[pyfunction]
fn srt_gcd(a: i64, b: i64) -> i64 {
    fn gcd_inner(a: i64, b: i64) -> i64 {
        if b == 0 { a.abs() } else { gcd_inner(b, a % b) }
    }
    gcd_inner(a, b)
}

/// Least common multiple
#[pyfunction]
fn srt_lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 { 0 } else { (a / srt_gcd(a, b) * b).abs() }
}

/// Factorial: n!
#[pyfunction]
fn srt_factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

/// Binomial coefficient: C(n, k)
#[pyfunction]
fn srt_comb(n: u64, k: u64) -> u64 {
    if k > n { return 0; }
    let k = k.min(n - k);
    (0..k).fold(1u64, |acc, i| acc * (n - i) / (i + 1))
}

/// Apply Geodesic Gravity Slide to weights in-place (Physical AI update)
#[pyfunction]
fn py_apply_geodesic_slide(
    weights: &TensorStorage,
    attractor: &TensorStorage,
    mode_norms: &TensorStorage,
    gravity: f64,
    temperature: f64,
) -> PyResult<()> {
    {
        use tensor::storage::{CudaData, TensorData};
        if let (
            TensorData::Cuda {
                data: w_data,
                device: dev,
                ..
            },
            TensorData::Cuda { data: a_data, .. },
            TensorData::Cuda { data: m_data, .. },
        ) = (&weights.data, &attractor.data, &mode_norms.data)
        {
            if let (
                CudaData::Float64(w_slice),
                CudaData::Float64(a_slice),
                CudaData::Float64(m_slice),
            ) = (w_data.as_ref(), a_data.as_ref(), m_data.as_ref())
            {
                let n = w_slice.len();
                srt_kernels::apply_geodesic_gravity_f64(
                    dev,
                    w_slice,
                    a_slice,
                    m_slice,
                    gravity,
                    temperature,
                    n,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
            }
        }
    }
    Ok(())
}

// ============================================================================
// TRFT (Ternary Rational Fourier Transform) - PyO3 Wrappers
// ============================================================================

/// Create TernarySolver with resonance ladder
#[pyfunction]
fn py_create_ternary_solver(_target_count: usize) -> PyResult<usize> {
    // Returns solver handle (index in global cache)
    // For now, return 0 as placeholder - full implementation needs thread-safe cache
    Ok(0)
}

/// Decompose signal using TRFT
#[pyfunction]
fn py_ternary_decompose(
    signal: Vec<i64>,
    hints: Vec<(i64, i64, i64)>,
    max_layers: usize,
    energy_threshold_dominant: f64,
    energy_threshold_subtle: f64,
) -> PyResult<Vec<(i64, i64, i64, i64, i64, f64)>> {
    // Convert i64 to i128
    let signal_i128: Vec<i128> = signal.iter().map(|&x| x as i128).collect();
    let hints_i128: Vec<(i128, i128, i128)> = hints.iter()
        .map(|&(a, b, h)| (a as i128, b as i128, h as i128))
        .collect();

    // Create solver
    let solver = TernarySolver::new(400);

    // Decompose
    let layers = solver.decompose_with_hints(
        &signal_i128,
        &hints_i128,
        max_layers,
        energy_threshold_dominant,
        energy_threshold_subtle,
    );

    // Convert results to Python-compatible format
    let results: Vec<(i64, i64, i64, i64, i64, f64)> = layers
        .iter()
        .map(|layer| {
            (
                layer.triple.a as i64,
                layer.triple.b as i64,
                layer.triple.h as i64,
                layer.gx0 as i64,
                layer.gy0 as i64,
                layer.energy_ratio,
            )
        })
        .collect();

    Ok(results)
}

/// Synthesize composite wave from layers
#[pyfunction]
fn py_ternary_synthesize(
    layers: Vec<(i64, i64, i64, i64, i64)>,
    length: usize,
) -> PyResult<Vec<i64>> {
    // Convert to WaveLayer structs
    let wave_layers: Vec<WaveLayer> = layers
        .iter()
        .map(|&(a, b, h, gx0, gy0)| WaveLayer {
            triple: PythagoreanTriple::new(a as i128, b as i128, h as i128),
            gx0: gx0 as i128,
            gy0: gy0 as i128,
            phase_offset: 0,
            threshold: 0,
            energy_ratio: 0.0,
        })
        .collect();

    // Create solver and synthesize
    let solver = TernarySolver::new(400);
    let composite = solver.synthesize_composite(&wave_layers, length);

    // Convert to i64
    let result: Vec<i64> = composite.iter().map(|&x| x as i64).collect();

    Ok(result)
}

/// Generate resonance ladder
#[pyfunction]
fn py_generate_resonance_ladder(max_h: i64) -> PyResult<Vec<(i64, i64, i64)>> {
    use exact::pythagorean::generate_resonance_ladder;
    
    let ladder = generate_resonance_ladder(max_h as i128);
    let result: Vec<(i64, i64, i64)> = ladder
        .iter()
        .map(|triple| (triple.a as i64, triple.b as i64, triple.h as i64))
        .collect();

    Ok(result)
}

/// Core module for Syntonic
///
/// This module provides:
/// - Exact arithmetic types (Rational, GoldenExact, SymExpr)
/// - The five fundamental SRT constants (π, e, φ, E*, q)
/// - Tensor storage (legacy, uses floats - to be replaced)
/// - Hypercomplex numbers (Quaternion, Octonion)
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Validate hierarchy correction functions are callable from Rust
    validate_hierarchy_corrections();

    // === Exact Arithmetic (NEW - preferred) ===
    m.add_class::<Rational>()?;
    m.add_class::<GoldenExact>()?;
    m.add_class::<FundamentalConstant>()?;
    m.add_class::<CorrectionLevel>()?;
    m.add_class::<PySymExpr>()?;
    
    // NEW: Register the Super-Field class
    m.add_class::<SyntonicExact>()?;
    m.add_class::<SyntonicDual>()?;

    // === Resonant Engine ===
    m.add_class::<ResonantTensor>()?;
    m.add_class::<ResonantEvolver>()?;
    m.add_class::<RESConfig>()?;
    m.add_class::<RESResult>()?;

    // === Golden Momentum Optimizer ===
    m.add_class::<GoldenMomentum>()?;

    // === Phi-Residual Operations ===
    m.add_class::<resonant::PhiResidualMode>()?;
    m.add_function(wrap_pyfunction!(resonant::phi_residual, m)?)?;
    m.add_function(wrap_pyfunction!(resonant::phi_residual_relu, m)?)?;

    // === Golden Batch Normalization ===
    m.add_class::<resonant::GoldenNormMode>()?;
    m.add_function(wrap_pyfunction!(
        resonant::golden_norm::golden_batch_norm_1d_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::golden_norm::golden_batch_norm_2d_py,
        m
    )?)?;

    // === Syntonic Softmax ===
    m.add_class::<resonant::SyntonicSoftmaxMode>()?;
    m.add_class::<resonant::SyntonicSoftmaxState>()?;
    m.add_function(wrap_pyfunction!(resonant::syntonic_softmax_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::syntonic_softmax::compute_syntonic_weights_py,
        m
    )?)?;

    // === Core Tensor Operations ===
    m.add_class::<TensorStorage>()?;
    m.add_class::<AsyncTensorTransfer>()?;
    m.add_class::<TransferComputeOverlap>()?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;

    // === SRT Data Loading ===
    m.add_class::<SRTCSVParser>()?;
    m.add_class::<SRTBinaryLoader>()?;
    m.add_class::<GoldenExactConverter>()?;
    m.add_class::<SRTDataPipeline>()?;
    m.add_class::<DataBatch>()?;
    m.add_class::<StreamingCSVIterator>()?;
    m.add_class::<DataType>()?;
    m.add_class::<Endianness>()?;
    
    // Register SNA Classes
    m.add_class::<sna::resonant_oscillator::DiscreteHilbertKernel>()?;
    m.add_class::<sna::resonant_oscillator::ResonantOscillator>()?;
    m.add_class::<sna::network::SyntonicNetwork>()?;

    // === SRT Precision Policy ===
    m.add_class::<PrecisionPolicy>()?;
    m.add_function(wrap_pyfunction!(get_srt_operation_policy, m)?)?;

    // === SRT Autograd (Gradient-Based Training) ===
    m.add_function(wrap_pyfunction!(py_backward_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_backward_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_backward_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(py_backward_layernorm, m)?)?;
    m.add_function(wrap_pyfunction!(py_backward_phi_residual, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_autograd_kernels, m)?)?;

    // === Hypercomplex Numbers ===
    m.add_class::<Quaternion>()?;
    m.add_class::<Octonion>()?;
    m.add_class::<Sedenion>()?;

    // === SRT Constants ===
    m.add_function(wrap_pyfunction!(srt_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_inv, m)?)?;
    m.add_function(wrap_pyfunction!(srt_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(srt_pi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_e, m)?)?;
    m.add_function(wrap_pyfunction!(srt_structure_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(srt_correction_factor, m)?)?;
    // === Scalar Math Functions ===
    m.add_function(wrap_pyfunction!(srt_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(srt_exp, m)?)?;
    m.add_function(wrap_pyfunction!(srt_log, m)?)?;
    m.add_function(wrap_pyfunction!(srt_log10, m)?)?;
    m.add_function(wrap_pyfunction!(srt_log2, m)?)?;
    m.add_function(wrap_pyfunction!(srt_log1p, m)?)?;
    m.add_function(wrap_pyfunction!(srt_expm1, m)?)?;
    m.add_function(wrap_pyfunction!(srt_sin, m)?)?;
    m.add_function(wrap_pyfunction!(srt_cos, m)?)?;
    m.add_function(wrap_pyfunction!(srt_tan, m)?)?;
    m.add_function(wrap_pyfunction!(srt_asin, m)?)?;
    m.add_function(wrap_pyfunction!(srt_acos, m)?)?;
    m.add_function(wrap_pyfunction!(srt_atan, m)?)?;
    m.add_function(wrap_pyfunction!(srt_atan2, m)?)?;
    m.add_function(wrap_pyfunction!(srt_sinh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_cosh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_asinh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_acosh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_atanh, m)?)?;
    m.add_function(wrap_pyfunction!(srt_pow, m)?)?;
    m.add_function(wrap_pyfunction!(srt_powi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_floor, m)?)?;
    m.add_function(wrap_pyfunction!(srt_ceil, m)?)?;
    m.add_function(wrap_pyfunction!(srt_round, m)?)?;
    m.add_function(wrap_pyfunction!(srt_trunc, m)?)?;
    m.add_function(wrap_pyfunction!(srt_fract, m)?)?;
    m.add_function(wrap_pyfunction!(srt_abs, m)?)?;
    m.add_function(wrap_pyfunction!(srt_signum, m)?)?;
    m.add_function(wrap_pyfunction!(srt_copysign, m)?)?;
    m.add_function(wrap_pyfunction!(srt_fma, m)?)?;
    m.add_function(wrap_pyfunction!(srt_hypot, m)?)?;
    m.add_function(wrap_pyfunction!(srt_cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(srt_min, m)?)?;
    m.add_function(wrap_pyfunction!(srt_max, m)?)?;
    m.add_function(wrap_pyfunction!(srt_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(srt_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(srt_radians, m)?)?;
    m.add_function(wrap_pyfunction!(srt_isnan, m)?)?;
    m.add_function(wrap_pyfunction!(srt_isinf, m)?)?;
    m.add_function(wrap_pyfunction!(srt_isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(srt_lerp, m)?)?;
    m.add_function(wrap_pyfunction!(srt_golden_lerp, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_power, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_power_inv, m)?)?;
    m.add_function(wrap_pyfunction!(srt_gcd, m)?)?;
    m.add_function(wrap_pyfunction!(srt_lcm, m)?)?;
    m.add_function(wrap_pyfunction!(srt_factorial, m)?)?;
    m.add_function(wrap_pyfunction!(srt_comb, m)?)?;

    // === Hierarchy Correction (SRT-Zero) ===
    m.add_function(wrap_pyfunction!(hierarchy::apply_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_correction_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_special, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_suppression, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::compute_e_star_n, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_chain, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::init_divisors, m)?)?;
    // Extended hierarchy corrections
    m.add_function(wrap_pyfunction!(hierarchy::apply_e7_correction, m)?)?;
    m.add_function(wrap_pyfunction!(
        hierarchy::apply_collapse_threshold_correction,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hierarchy::apply_coxeter_kissing_correction,
        m
    )?)?;
    // Extended hierarchy constants
    m.add_function(wrap_pyfunction!(hierarchy::e8_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_kissing, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::g2_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::f4_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::coxeter_kissing_720, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::hierarchy_exponent, m)?)?;

    // Extended hierarchy constants
    m.add_function(wrap_pyfunction!(hierarchy::e8_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_fundamental, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_fundamental, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::f4_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::g2_rank, m)?)?;

    // Universal Syntony Correction Hierarchy - New correction functions
    m.add_function(wrap_pyfunction!(hierarchy::apply_e6_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_mersenne_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_lucas_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_fermat_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_loop_integral, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_phi_power_correction, m)?)?;

    // Prime number tower accessors - Mersenne primes
    m.add_function(wrap_pyfunction!(hierarchy::mersenne_m2, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::mersenne_m3, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::mersenne_m5, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::mersenne_m7, m)?)?;

    // Prime number tower accessors - Lucas numbers
    m.add_function(wrap_pyfunction!(hierarchy::lucas_l4, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::lucas_l5, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::lucas_l6, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::lucas_l7, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::lucas_l11, m)?)?;

    // Prime number tower accessors - Fermat primes
    m.add_function(wrap_pyfunction!(hierarchy::fermat_f0, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::fermat_f1, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::fermat_f2, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::fermat_f3, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::fermat_f4, m)?)?;

    // === GoldenGELU Activation ===
    m.add_function(wrap_pyfunction!(golden_gelu::golden_gelu_forward, m)?)?;
    m.add_function(wrap_pyfunction!(golden_gelu::golden_gelu_backward, m)?)?;
    m.add_function(wrap_pyfunction!(
        golden_gelu::batched_golden_gelu_forward,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(golden_gelu::get_golden_gelu_phi, m)?)?;

    // === SRT Tensor Operations (GPU-accelerated when on CUDA) ===
    m.add_function(wrap_pyfunction!(py_apply_geodesic_slide, m)?)?;
    m.add_function(wrap_pyfunction!(srt_scale_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_golden_gaussian_weights, m)?)?;
    m.add_function(wrap_pyfunction!(srt_apply_correction, m)?)?;
    m.add_function(wrap_pyfunction!(srt_e8_batch_projection, m)?)?;
    m.add_function(wrap_pyfunction!(srt_theta_series, m)?)?;
    m.add_function(wrap_pyfunction!(srt_compute_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(srt_dhsr_cycle, m)?)?;
    // Fixed-Point kernels
    m.add_function(wrap_pyfunction!(srt_compute_syntony_fp64, m)?)?;
    m.add_function(wrap_pyfunction!(srt_dhsr_cycle_fp64, m)?)?;
    m.add_function(wrap_pyfunction!(srt_differentiation_fp64, m)?)?;
    m.add_function(wrap_pyfunction!(srt_harmonization_fp64, m)?)?;
    m.add_function(wrap_pyfunction!(srt_differentiation_full_fp64, m)?)?;
    m.add_function(wrap_pyfunction!(srt_laplacian_1d_fp64, m)?)?;

    // === SRT Scatter/Gather Operations ===
    {
        use tensor::py_srt_cuda_ops::{
            py_gather_f32, py_gather_f64, py_reduce_consciousness_count_f64, py_reduce_e8_norm_f64,
            py_reduce_max_f32, py_reduce_max_f64, py_reduce_mean_f32, py_reduce_mean_f64,
            py_reduce_min_f32, py_reduce_min_f64, py_reduce_norm_c128, py_reduce_norm_l2_f32,
            py_reduce_norm_l2_f64, py_reduce_sum_c128, py_reduce_sum_cols_f64, py_reduce_sum_f32,
            py_reduce_sum_f64, py_reduce_sum_golden_weighted_f64, py_reduce_sum_lucas_shadow_f64,
            py_reduce_sum_mersenne_stable_f64, py_reduce_sum_phi_scaled_f64,
            py_reduce_sum_q_corrected_f64, py_reduce_sum_rows_f64,
            py_reduce_syntony_deviation_f64, py_reduce_syntony_f64,
            py_reduce_variance_golden_target_f64, py_scatter_add_f64, py_scatter_f32,
            py_scatter_f64,
        };
        m.add_function(wrap_pyfunction!(py_gather_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_add_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_mean_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_mean_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_max_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_max_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_min_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_min_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_norm_l2_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_norm_l2_f32, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_golden_weighted_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_rows_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_cols_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_phi_scaled_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_syntony_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_variance_golden_target_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_lucas_shadow_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_mersenne_stable_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_syntony_deviation_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_consciousness_count_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_q_corrected_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_e8_norm_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_norm_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_reduce_sum_f32, m)?)?;
    }

    // === SRT Inflationary Broadcasting ===
    m.add_function(wrap_pyfunction!(py_inflationary_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_inflationary_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_consciousness_inflationary_broadcast,
        m
    )?)?;

    // === Causal History DAG (DHSR Tracking) ===
    m.add_class::<PyCausalHistoryTracker>()?;
    m.add_function(wrap_pyfunction!(create_causal_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(d4_consciousness_threshold, m)?)?;

    // === SRT Memory Transfer Statistics ===
    m.add_function(wrap_pyfunction!(srt_transfer_stats, m)?)?;
    m.add_function(wrap_pyfunction!(srt_reserve_memory, m)?)?;
    m.add_function(wrap_pyfunction!(srt_wait_for_resonance, m)?)?;
    m.add_function(wrap_pyfunction!(srt_pool_stats, m)?)?;
    m.add_function(wrap_pyfunction!(srt_memory_resonance, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor::storage::_debug_stress_pool_take,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(srt_kernels::validate_kernels, m)?)?;
    // Kernel loaders (direct function calls from py_srt_cuda_ops module)
    {
        use tensor::py_srt_cuda_ops::{
            py_load_attention_kernels, py_load_complex_ops_kernels, py_load_reduction_kernels,
            py_load_scatter_gather_kernels, py_load_trilinear_kernels,
            py_load_wmma_syntonic_kernels,
        };
        m.add_function(wrap_pyfunction!(py_load_wmma_syntonic_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_scatter_gather_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_reduction_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_trilinear_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_complex_ops_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_attention_kernels, m)?)?;
    }

    // === Winding State ===
    m.add_class::<WindingState>()?;
    m.add_class::<WindingStateIterator>()?;
    m.add_function(wrap_pyfunction!(enumerate_windings, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_windings_by_norm, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_windings_exact_norm, m)?)?;
    m.add_function(wrap_pyfunction!(count_windings, m)?)?;

    // === Spectral Operations ===
    m.add_function(wrap_pyfunction!(theta_series_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(theta_series_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(theta_series_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_trace, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(compute_golden_weights, m)?)?;
    m.add_function(wrap_pyfunction!(compute_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_zeta_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(partition_function, m)?)?;
    m.add_function(wrap_pyfunction!(theta_sum_combined, m)?)?;
    m.add_function(wrap_pyfunction!(count_by_generation, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_generation, m)?)?;

    // === Knot Laplacian Operations ===
    m.add_function(wrap_pyfunction!(knot_eigenvalue, m)?)?;
    m.add_function(wrap_pyfunction!(compute_knot_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(knot_heat_kernel_trace, m)?)?;
    m.add_function(wrap_pyfunction!(knot_spectral_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(knot_spectral_zeta_complex, m)?)?;

    // === E8 Lattice and Golden Projector ===
    m.add_function(wrap_pyfunction!(py_e8_generate_weights, m)?)?;
    m.add_function(wrap_pyfunction!(py_e8_generate_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_phi, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_perp, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_in_golden_cone, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_8d_weight, m)?)?;

    // === Neural-network-friendly E8 wrappers ===
    m.add_function(wrap_pyfunction!(py_e8_generate_weights_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_e8_generate_roots_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_q_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_phi_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_parallel_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_perp_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_in_golden_cone_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_8d_weight_nn, m)?)?;

    // === Number Theory and Syntony (Rust Performance Backend) ===
    m.add_function(wrap_pyfunction!(py_mobius, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_square_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_mertens, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_weight, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_weights, m)?)?;
    m.add_function(wrap_pyfunction!(py_e_star, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_winding_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_winding_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_aggregate_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_standard_mode_norms, m)?)?;

    // === SRT/CRT Prime Theory Functions ===
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_mersenne_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_fermat_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_lucas_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(resonant::py_wrappers::py_lucas_number, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_pisano_period,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_stable_winding,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_get_stability_barrier,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_transcendence_gate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_versal_grip_strength,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_mersenne_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_fermat_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(resonant::py_wrappers::py_lucas_primes, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_lucas_dark_boost,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_predict_dark_matter_mass,
        m
    )?)?;

    // === Crystallization Functions ===
    m.add_function(wrap_pyfunction!(py_crystallize_with_dwell_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(py_snap_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_snap_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_compute_snap_gradient_cuda,
        m
    )?)?;

    // === Loss Functions (Rust Performance Backend) ===
    m.add_function(wrap_pyfunction!(py_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntony_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntony_loss_srt, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_target_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(py_phase_alignment_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntonic_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_syntony_from_probs, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_decay_loss, m)?)?;

    // === Broadcasting Operations ===
    m.add_function(wrap_pyfunction!(py_broadcast_shape, m)?)?;
    m.add_function(wrap_pyfunction!(py_are_broadcastable, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_sub, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_div, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_index, m)?)?;

    // === In-place Operations ===
    m.add_function(wrap_pyfunction!(py_inplace_add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_div_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_negate, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_abs, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_golden_weight, m)?)?;

    // === Convolution Operations ===
    m.add_function(wrap_pyfunction!(py_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_global_avg_pool2d, m)?)?;

    // === Trilinear Operations ===
    m.add_function(wrap_pyfunction!(py_trilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_toroidal_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_phi_weighted_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_golden_decay_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_causal_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_retrocausal_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_symmetric_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_trilinear_acausal_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_bilinear_f64, m)?)?;

    // === Structure enum for correction factors ===
    m.add_class::<Structure>()?;

    // === Linear Algebra Operations ===
    // Core matmul
    m.add_function(wrap_pyfunction!(py_mm, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_add, m)?)?;
    // Transpose variants
    m.add_function(wrap_pyfunction!(py_mm_tn, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_nt, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_tt, m)?)?;
    // Hermitian variants
    m.add_function(wrap_pyfunction!(py_mm_hn, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_nh, m)?)?;
    // Batched
    m.add_function(wrap_pyfunction!(py_bmm, m)?)?;
    // SRT-specific operations
    m.add_function(wrap_pyfunction!(py_mm_phi, m)?)?;
    m.add_function(wrap_pyfunction!(py_phi_bracket, m)?)?;
    m.add_function(wrap_pyfunction!(py_phi_antibracket, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_corrected, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_golden_phase, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_golden_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(py_projection_sum, m)?)?;
    // Generalized GEMM and q-correction operations
    m.add_function(wrap_pyfunction!(py_mm_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_q_corrected_direct, m)?)?;
    m.add_function(wrap_pyfunction!(py_q_correction_scalar, m)?)?;

    // === Additional SRT Operations ===
    {
        use tensor::py_srt_cuda_ops::{
            py_sin_toroidal, py_cos_toroidal, py_atan2_toroidal, py_phi_exp, py_phi_exp_inv,
            py_gnosis_mask, py_adaptive_gnosis_mask, py_fractal_gnosis_mask, py_temporal_gnosis_mask,
            py_autograd_filter, py_attractor_memory_update, py_golden_entropy, py_syntony_metric,
            py_cuda_sgemm, py_scatter_golden_f64, py_scatter_mersenne_stable_f64,
            py_gather_lucas_shadow_f64, py_gather_pisano_hooked_f64, py_gather_e8_roots_f64,
            py_scatter_golden_cone_f64, py_gather_transcendence_gate_f64,
            py_scatter_consciousness_threshold_f64, py_arg_c128, py_phase_syntony_c128,
            py_cuda_mm_with_policy,
            // Static CUDA library wrappers
            py_static_laplacian_1d, py_static_syntony_projection, py_static_gather, py_static_scatter_add,
            py_static_compute_syntony_c128, py_static_fourier_project_batch, py_static_damping_cascade,
            py_static_dhsr_step_fused, py_static_attractor_memory_update, py_static_retrocausal_harmonize,
            py_static_attractor_centroid, py_static_atan2_toroidal, py_static_golden_entropy,
            py_static_gather_phi_weighted, py_static_arg_c128, py_static_phase_syntony_c128,
            py_static_trilinear, py_static_gnosis_mask, py_static_matmul,
            // Additional CUDA kernel wrappers
            py_laplacian_1d, py_syntony_projection, py_wmma_fp16_matmul,
            py_gather_phi_weighted, py_gather, py_scatter_add, py_trilinear,
            py_phase_syntony, py_cuda_attractor_memory_update, py_fourier_project_batch_fp64,
        };
        m.add_function(wrap_pyfunction!(py_sin_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_cos_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_atan2_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_phi_exp, m)?)?;
        m.add_function(wrap_pyfunction!(py_phi_exp_inv, m)?)?;
        m.add_function(wrap_pyfunction!(py_gnosis_mask, m)?)?;
        m.add_function(wrap_pyfunction!(py_adaptive_gnosis_mask, m)?)?;
        m.add_function(wrap_pyfunction!(py_fractal_gnosis_mask, m)?)?;
        m.add_function(wrap_pyfunction!(py_temporal_gnosis_mask, m)?)?;
        m.add_function(wrap_pyfunction!(py_autograd_filter, m)?)?;
        m.add_function(wrap_pyfunction!(py_attractor_memory_update, m)?)?;
        m.add_function(wrap_pyfunction!(py_golden_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(py_syntony_metric, m)?)?;
        m.add_function(wrap_pyfunction!(py_cuda_sgemm, m)?)?;
        m.add_function(wrap_pyfunction!(py_cuda_dgemm, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_golden_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_mersenne_stable_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_lucas_shadow_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_pisano_hooked_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_e8_roots_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_golden_cone_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_transcendence_gate_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_consciousness_threshold_f64, m)?)?;
        m.add_function(wrap_pyfunction!(py_arg_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_phase_syntony_c128, m)?)?;
        // Hierarchy corrections (using crate::hierarchy module)
        m.add_function(wrap_pyfunction!(py_hierarchy_e8_correction, m)?)?;
        m.add_function(wrap_pyfunction!(py_hierarchy_e7_correction, m)?)?;
        m.add_function(wrap_pyfunction!(py_hierarchy_coxeter_kissing_correction, m)?)?;
        // Attractor kernels (retrocausal training)
        m.add_function(wrap_pyfunction!(py_load_attractor_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(py_attractor_centroid, m)?)?;
        m.add_function(wrap_pyfunction!(py_retrocausal_harmonize, m)?)?;
        m.add_function(wrap_pyfunction!(py_attractor_distance, m)?)?;
        // DHSR kernels
        m.add_function(wrap_pyfunction!(py_dhsr_step_fused, m)?)?;
        m.add_function(wrap_pyfunction!(py_damping_cascade, m)?)?;
        m.add_function(wrap_pyfunction!(py_differentiation_full, m)?)?;
        m.add_function(wrap_pyfunction!(py_fourier_project_batch, m)?)?;
        // Precision policy matrix multiply
        m.add_function(wrap_pyfunction!(py_cuda_mm_with_policy, m)?)?;
        // Static CUDA library wrappers
        m.add_function(wrap_pyfunction!(py_static_laplacian_1d, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_syntony_projection, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_gather, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_scatter_add, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_compute_syntony_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_fourier_project_batch, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_damping_cascade, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_dhsr_step_fused, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_attractor_memory_update, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_retrocausal_harmonize, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_attractor_centroid, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_atan2_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_golden_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_sin_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_cos_toroidal, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_phi_exp, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_phi_exp_inv, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_gather_phi_weighted, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_arg_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_phase_syntony_c128, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_trilinear, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_gnosis_mask, m)?)?;
        m.add_function(wrap_pyfunction!(py_static_matmul, m)?)?;
        // Additional CUDA kernel wrappers
        m.add_function(wrap_pyfunction!(py_laplacian_1d, m)?)?;
        m.add_function(wrap_pyfunction!(py_syntony_projection, m)?)?;
        m.add_function(wrap_pyfunction!(py_wmma_fp16_matmul, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather_phi_weighted, m)?)?;
        m.add_function(wrap_pyfunction!(py_gather, m)?)?;
        m.add_function(wrap_pyfunction!(py_scatter_add, m)?)?;
        m.add_function(wrap_pyfunction!(py_trilinear, m)?)?;
        m.add_function(wrap_pyfunction!(py_phase_syntony, m)?)?;
        m.add_function(wrap_pyfunction!(py_cuda_attractor_memory_update, m)?)?;
        m.add_function(wrap_pyfunction!(py_fourier_project_batch_fp64, m)?)?;
    }

    // === Prime Selection Rules ===
    register_extended_prime_selection(m)?;

    // === Gnosis/Consciousness Module ===
    register_gnosis(m)?;

    // === Transcendence Module ===
    register_transcendence(m)?;

    // === Math Utilities ===
    register_math_utils(m)?;

    // === TRFT (Ternary Rational Fourier Transform) ===
    m.add_function(wrap_pyfunction!(py_create_ternary_solver, m)?)?;
    m.add_function(wrap_pyfunction!(py_ternary_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(py_ternary_synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_resonance_ladder, m)?)?;

    // === SNA Submodule (Syntonic Neural Architecture) ===
    // Create sna submodule and inject into sys.modules for proper import paths
    let py = m.py();
    let sna_module = PyModule::new_bound(py, "sna")?;
    
    // Add SNA classes to submodule
    sna_module.add_class::<sna::resonant_oscillator::DiscreteHilbertKernel>()?;
    sna_module.add_class::<sna::resonant_oscillator::ResonantOscillator>()?;
    sna_module.add_class::<sna::network::SyntonicNetwork>()?;
    
    // Expose KERNEL_SCALE constant
    sna_module.add("KERNEL_SCALE", sna::resonant_oscillator::KERNEL_SCALE)?;
    
    // Inject into sys.modules as "srt_library.sna"
    let sys = py.import_bound("sys")?;
    let sys_modules = sys.getattr("modules")?;
    sys_modules.set_item("srt_library.sna", &sna_module)?;
    
    // Also add as submodule of main module
    m.add_submodule(&sna_module)?;

    Ok(())
}
