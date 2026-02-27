//! Hierarchy correction module for SRT-Zero
//!
//! Provides Rust interface for:
//! - Batched correction application
//! - Special corrections (q²/φ, q·φ, 4q, etc.)
//! - Suppression factors
//! - Nested correction chains
//! - E*×N batch computation
//!
//! Note: CUDA backend pending cudarc API fixes. CPU fallback always active.

use crate::tensor::srt_kernels::{PHI, Q_DEFICIT};

// Compute derived phi powers locally
const PHI_SQUARED: f64 = PHI * PHI;
const PHI_CUBED: f64 = PHI_SQUARED * PHI;
const PHI_FOURTH: f64 = PHI_SQUARED * PHI_SQUARED;
const PHI_FIFTH: f64 = PHI_FOURTH * PHI;

// ============================================================================
// Prime Number Tower Constants
// ============================================================================

// Mersenne Primes: M_p = 2^p - 1 (for prime p)
// These constrain which matter generations can exist
pub const MERSENNE_M2: i32 = 3;      // 2² - 1 = 3 (generation count)
pub const MERSENNE_M3: i32 = 7;      // 2³ - 1 = 7 (first non-trivial)
pub const MERSENNE_M5: i32 = 31;     // 2⁵ - 1 = 31 (third generation)
pub const MERSENNE_M7: i32 = 127;    // 2⁷ - 1 = 127 (heavy anchor)

// Lucas Numbers: L_n = φⁿ + φ̂ⁿ (golden recursion sequence)
// Provide shadow phase corrections for dark sector physics
pub const LUCAS_L4: i32 = 7;         // = MERSENNE_M3 (Mersenne-Lucas resonance)
pub const LUCAS_L5: i32 = 11;        // M₁₁ barrier connection
pub const LUCAS_L6: i32 = 18;        // = E7_COXETER
pub const LUCAS_L7: i32 = 29;        // Lucas prime gate
pub const LUCAS_L11: i32 = 199;      // Deep shadow structure

// Fermat Primes: F_n = 2^(2^n) + 1
// Determine the number of fundamental forces (exactly 5 primes)
pub const FERMAT_F0: i32 = 3;        // 2¹ + 1 = 3 (Strong force SU(3))
pub const FERMAT_F1: i32 = 5;        // 2² + 1 = 5 (Weak isospin SU(2))
pub const FERMAT_F2: i32 = 17;       // 2⁴ + 1 = 17 (Hypercharge U(1))
pub const FERMAT_F3: i32 = 257;      // 2⁸ + 1 = 257 (Gravity seed)
pub const FERMAT_F4: i32 = 65537;    // 2¹⁶ + 1 = 65537 (Recursion completion)

// E_STAR = e^π - π (computed at runtime for now since const f64::pow is not const-fn)
fn e_star() -> f64 {
    std::f64::consts::E.powf(std::f64::consts::PI) - std::f64::consts::PI
}

// ============================================================================
// Extended Structure Dimensions (E₈ → E₇ → E₆ → SM Chain)
// ============================================================================

// E₈ Family
pub const E8_DIM: i32 = 248;
pub const E8_ROOTS: i32 = 240;
pub const E8_POSITIVE_ROOTS: i32 = 120;
pub const E8_RANK: i32 = 8;
pub const E8_COXETER: i32 = 30;

// E₇ Family (Intermediate Unification Scale)
pub const E7_DIM: i32 = 133;
pub const E7_ROOTS: i32 = 126;
pub const E7_POSITIVE_ROOTS: i32 = 63;
pub const E7_FUNDAMENTAL: i32 = 56;
pub const E7_RANK: i32 = 7;
pub const E7_COXETER: i32 = 18;

// E₆ Family
pub const E6_DIM: i32 = 78;
pub const E6_ROOTS: i32 = 72;
pub const E6_POSITIVE_ROOTS: i32 = 36;
pub const E6_FUNDAMENTAL: i32 = 27;
pub const E6_RANK: i32 = 6;
pub const E6_COXETER: i32 = 12;

// D₄ Family (SO(8) with Triality)
pub const D4_DIM: i32 = 28;
pub const D4_KISSING: i32 = 24; // Collapse threshold!
pub const D4_RANK: i32 = 4;
pub const D4_COXETER: i32 = 6;

// G₂ (Octonion Automorphisms)
pub const G2_DIM: i32 = 14;
pub const G2_RANK: i32 = 2;

// F₄ (Jordan Algebra Structure)
pub const F4_DIM: i32 = 52;
pub const F4_RANK: i32 = 4;

// Coxeter-Kissing Products
pub const COXETER_KISSING_720: i32 = E8_COXETER * D4_KISSING; // 30 × 24 = 720
pub const HIERARCHY_EXPONENT: i32 = COXETER_KISSING_720 - 1; // 719

use pyo3::prelude::*;

// =============================================================================
// PYTHON WRAPPERS
// =============================================================================

/// Apply a single standard correction: value * (1 ± q/divisor)
///
/// Args:
///   values: List of float64 values to correct
///   divisors: List of divisors (one per value)
///   signs: List of signs (+1 or -1, one per value)
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_correction")]
pub fn apply_correction(
    values: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
) -> PyResult<Vec<f64>> {
    if values.len() != divisors.len() || values.len() != signs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values, divisors, and signs must have same length",
        ));
    }

    let mut outputs = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let value = values[i];
        let divisor = divisors[i];
        let sign = signs[i];

        let factor = if divisor != 0.0 {
            1.0 + sign as f64 * Q_DEFICIT / divisor
        } else {
            1.0
        };

        outputs.push(value * factor);
    }
    Ok(outputs)
}

/// Apply a single standard correction with uniform divisor
///
/// Args:
///   values: List of float64 values to correct
///   divisor: Single divisor to apply to all values
///   sign: +1 or -1
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_correction_uniform")]
pub fn apply_correction_uniform(values: Vec<f64>, divisor: f64, sign: i32) -> PyResult<Vec<f64>> {
    let factor = if divisor != 0.0 {
        1.0 + sign as f64 * Q_DEFICIT / divisor
    } else {
        1.0
    };
    Ok(values.iter().map(|&v| v * factor).collect())
}

/// Apply special corrections (q²/φ, q·φ, 4q, etc.)
///
/// Types (0-29 - original):
///   0: q_phi_plus, 1: q_phi_minus, 2: q_phi_squared_plus, 3: q_phi_squared_minus
///   4: q_phi_cubed_plus, 5: q_phi_cubed_minus, 6: q_phi_fourth_plus, 7: q_phi_fourth_minus
///   8: q_phi_fifth_plus, 9: q_phi_fifth_minus
///   10: q_squared_plus, 11: q_squared_minus
///   12: q_squared_phi_plus, 13: q_squared_phi_minus
///   14: q_sq_phi_sq_plus, 15: q_sq_phi_sq_minus, 16: q_sq_phi_plus
///   17: 4q_plus, 18: 4q_minus, 19: 3q_plus, 20: 3q_minus
///   21: 6q_plus, 22: 8q_plus, 23: pi_q_plus
///   24: q_cubed, 25: q_phi_div_4pi_plus, 26: 8q_inv_plus, 27: q_squared_half_plus
///   28: q_6pi_plus, 29: q_phi_inv_plus
///
/// Types (30-39 - loop integrals):
///   30: q_2pi_plus, 31: q_2pi_minus (half-loop)
///   32: q_3pi_plus, 33: q_3pi_minus (3-flavor QCD)
///   34: q_4pi_plus, 35: q_4pi_minus (one-loop radiative)
///   36: q_5pi_plus, 37: q_5pi_minus (5-flavor QCD at M_Z)
///   38: q_1000_plus, 39: q_1000_minus (fixed-point stability)
///
/// Types (40-49 - large divisors):
///   40: q_720_plus, 41: q_720_minus (Coxeter-Kissing h×K)
///   42: q_360_plus, 43: q_360_minus (cone periodicity)
///   44: q_248_plus, 45: q_248_minus (E8 dimension)
///   46: q_240_plus, 47: q_240_minus (E8 roots)
///   48: q_78_plus, 49: q_78_minus (E6 dimension)
///
/// Types (50-59 - E6/prime tower):
///   50: q_72_plus, 51: q_72_minus (E6 roots)
///   52: q_36_plus, 53: q_36_minus (Golden Cone |Φ⁺(E₆)|)
///   54: q_27_plus, 55: q_27_minus (E6 fundamental)
///   56: q_31_plus (Mersenne M5), 57: q_127_plus (Mersenne M7)
///   58: q_11_plus (Lucas L5), 59: q_29_plus (Lucas L7)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_special")]
pub fn apply_special(values: Vec<f64>, types: Vec<i32>) -> PyResult<Vec<f64>> {
    if values.len() != types.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values and types must have same length",
        ));
    }

    let q = Q_DEFICIT;
    let phi = PHI;
    let phi_sq = PHI_SQUARED;
    let phi_cubed = PHI_CUBED;
    let phi_fourth = phi_sq * phi_sq;
    let phi_fifth = phi_fourth * phi;
    let pi = std::f64::consts::PI;

    let mut outputs = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let value = values[i];
        let type_ = types[i];
        let factor = match type_ {
            0 => 1.0 + q * phi,               // q_phi_plus
            1 => 1.0 - q * phi,               // q_phi_minus
            2 => 1.0 + q * phi_sq,            // q_phi_squared_plus
            3 => 1.0 - q * phi_sq,            // q_phi_squared_minus
            4 => 1.0 + q * phi_cubed,         // q_phi_cubed_plus
            5 => 1.0 - q * phi_cubed,         // q_phi_cubed_minus
            6 => 1.0 + q * phi_fourth,        // q_phi_fourth_plus
            7 => 1.0 - q * phi_fourth,        // q_phi_fourth_minus
            8 => 1.0 + q * phi_fifth,         // q_phi_fifth_plus
            9 => 1.0 - q * phi_fifth,         // q_phi_fifth_minus
            10 => 1.0 + q * q,                // q_squared_plus
            11 => 1.0 - q * q,                // q_squared_minus
            12 => 1.0 + q * q / phi,          // q_squared_phi_plus
            13 => 1.0 - q * q / phi,          // q_squared_phi_minus
            14 => 1.0 + q * q / phi_sq,       // q_sq_phi_sq_plus
            15 => 1.0 - q * q / phi_sq,       // q_sq_phi_sq_minus
            16 => 1.0 + q * q * phi,          // q_sq_phi_plus
            17 => 1.0 + 4.0 * q,              // 4q_plus
            18 => 1.0 - 4.0 * q,              // 4q_minus
            19 => 1.0 + 3.0 * q,              // 3q_plus
            20 => 1.0 - 3.0 * q,              // 3q_minus
            21 => 1.0 + 6.0 * q,              // 6q_plus
            22 => 1.0 + 8.0 * q,              // 8q_plus
            23 => 1.0 + pi * q,               // pi_q_plus
            24 => 1.0 + q * q * q,            // q_cubed
            25 => 1.0 + q * phi / (4.0 * pi), // q_phi_div_4pi_plus
            26 => 1.0 + q / 8.0,              // 8q_inv_plus
            27 => 1.0 + q * q / 2.0,          // q_squared_half_plus
            28 => 1.0 + q / (6.0 * pi),       // q_6pi_plus
            29 => 1.0 + q / phi,              // q_phi_inv_plus

            // Loop integral factors (q/nπ)
            30 => 1.0 + q / (2.0 * pi),       // q_2pi_plus (half-loop)
            31 => 1.0 - q / (2.0 * pi),       // q_2pi_minus
            32 => 1.0 + q / (3.0 * pi),       // q_3pi_plus (3-flavor QCD)
            33 => 1.0 - q / (3.0 * pi),       // q_3pi_minus
            34 => 1.0 + q / (4.0 * pi),       // q_4pi_plus (one-loop radiative)
            35 => 1.0 - q / (4.0 * pi),       // q_4pi_minus
            36 => 1.0 + q / (5.0 * pi),       // q_5pi_plus (5-flavor QCD at M_Z)
            37 => 1.0 - q / (5.0 * pi),       // q_5pi_minus

            // Large divisor factors
            38 => 1.0 + q / 1000.0,           // q_1000_plus (fixed-point stability)
            39 => 1.0 - q / 1000.0,           // q_1000_minus
            40 => 1.0 + q / 720.0,            // q_720_plus (Coxeter-Kissing h×K)
            41 => 1.0 - q / 720.0,            // q_720_minus
            42 => 1.0 + q / 360.0,            // q_360_plus (cone periodicity)
            43 => 1.0 - q / 360.0,            // q_360_minus
            44 => 1.0 + q / 248.0,            // q_248_plus (E8 dimension)
            45 => 1.0 - q / 248.0,            // q_248_minus
            46 => 1.0 + q / 240.0,            // q_240_plus (E8 roots)
            47 => 1.0 - q / 240.0,            // q_240_minus

            // E6 factors
            48 => 1.0 + q / 78.0,             // q_78_plus (E6 dimension)
            49 => 1.0 - q / 78.0,             // q_78_minus
            50 => 1.0 + q / 72.0,             // q_72_plus (E6 roots)
            51 => 1.0 - q / 72.0,             // q_72_minus
            52 => 1.0 + q / 36.0,             // q_36_plus (Golden Cone |Φ⁺(E₆)|)
            53 => 1.0 - q / 36.0,             // q_36_minus
            54 => 1.0 + q / 27.0,             // q_27_plus (E6 fundamental)
            55 => 1.0 - q / 27.0,             // q_27_minus

            // Prime tower factors
            56 => 1.0 + q / 31.0,             // q_31_plus (Mersenne M5)
            57 => 1.0 + q / 127.0,            // q_127_plus (Mersenne M7)
            58 => 1.0 + q / 11.0,             // q_11_plus (Lucas L5)
            59 => 1.0 + q / 29.0,             // q_29_plus (Lucas L7)

            _ => 1.0,
        };
        outputs.push(value * factor);
    }
    Ok(outputs)
}

/// Apply suppression factors
///
/// These factors appear as divisors for processes involving recursion layer crossings.
///
/// Types:
///   0: winding_instability (1/(1+q/φ)) ~1.7% suppression - Inverse recursion
///   1: recursion_penalty (1/(1+q·φ)) ~4.2% suppression - Double layer crossings
///   2: double_inverse (1/(1+q/φ²)) ~1.05% suppression - Deep winding instability
///   3: fixed_point_penalty (1/(1+q·φ²)) ~6.7% suppression - Triple layer crossings
///   4: base_suppression (1/(1+q)) ~2.7% suppression - Universal vacuum penalty
///   5: deep_recursion_penalty (1/(1+q·φ³)) ~10.4% suppression - Four-layer crossings
#[pyfunction]
#[pyo3(name = "hierarchy_apply_suppression")]
pub fn apply_suppression(values: Vec<f64>, suppression_type: i32) -> PyResult<Vec<f64>> {
    let q = Q_DEFICIT;
    let phi = PHI;
    let phi_sq = PHI_SQUARED;
    let phi_cubed = PHI_CUBED;

    let factor = match suppression_type {
        0 => 1.0 / (1.0 + q / phi),      // winding_instability
        1 => 1.0 / (1.0 + q * phi),      // recursion_penalty
        2 => 1.0 / (1.0 + q / phi_sq),   // double_inverse
        3 => 1.0 / (1.0 + q * phi_sq),   // fixed_point_penalty
        4 => 1.0 / (1.0 + q),            // base_suppression
        5 => 1.0 / (1.0 + q * phi_cubed),// deep_recursion_penalty
        _ => 1.0,
    };
    Ok(values.iter().map(|&v| v * factor).collect())
}

/// Compute E*×N with corrections for a batch of values
///
/// Args:
///   N: List of N multipliers
///   divisors: Flat list of divisors for all corrections
///   signs: Flat list of signs for all corrections
///   n_corrections_per_value: Number of corrections to apply to each value
///
/// Returns:
///   Computed values: E* × N × ∏(1 ± q/divisor)
#[pyfunction]
#[pyo3(name = "hierarchy_compute_e_star_n")]
pub fn compute_e_star_n(
    n: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
    n_corrections_per_value: usize,
) -> PyResult<Vec<f64>> {
    if divisors.len() != signs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "divisors and signs must have same length",
        ));
    }
    if divisors.len() != n.len() * n_corrections_per_value {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "divisors length must equal N.len() * n_corrections_per_value",
        ));
    }

    let mut outputs = Vec::with_capacity(n.len());
    for (i, &n_val) in n.iter().enumerate() {
        let mut value = e_star() * n_val;

        for j in 0..n_corrections_per_value {
            let corr_idx = i * n_corrections_per_value + j;
            let divisor = divisors[corr_idx];
            let sign = signs[corr_idx];

            if divisor != 0.0 {
                let factor = 1.0 + sign as f64 * Q_DEFICIT / divisor;
                value *= factor;
            }
        }

        outputs.push(value);
    }
    Ok(outputs)
}

/// Apply nested correction chain (supports varying length chains per value)
///
/// Args:
///   values: List of values to correct
///   divisors: Flat list of all divisors for all chains
///   signs: Flat list of all signs for all chains
///   chain_lengths: Length of correction chain for each value
///   chain_starts: Starting index in divisors/signs for each value's chain
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_chain")]
pub fn apply_chain(
    values: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
    chain_lengths: Vec<i32>,
    chain_starts: Vec<i32>,
) -> PyResult<Vec<f64>> {
    let q = Q_DEFICIT;

    let mut outputs = Vec::with_capacity(values.len());
    for (i, &value) in values.iter().enumerate() {
        let mut val = value;
        let chain_len = chain_lengths[i] as usize;
        let chain_start = chain_starts[i] as usize;

        for j in 0..chain_len {
            let corr_idx = chain_start + j;
            if corr_idx >= divisors.len() {
                break;
            }

            let divisor = divisors[corr_idx];
            let sign = signs[corr_idx];

            if divisor != 0.0 {
                let factor = 1.0 + sign as f64 * q / divisor;
                val *= factor;
            }
        }

        outputs.push(val);
    }
    Ok(outputs)
}

// ============================================================================
// Extended Correction Factor Functions
// ============================================================================

/// Apply correction with E₇ structure
#[pyfunction]
#[pyo3(name = "hierarchy_apply_e7_correction")]
pub fn apply_e7_correction(value: f64, structure_index: i32) -> f64 {
    let divisor = match structure_index {
        0 => E7_DIM,            // 133
        1 => E7_ROOTS,          // 126
        2 => E7_POSITIVE_ROOTS, // 63
        3 => E7_FUNDAMENTAL,    // 56
        4 => E7_RANK,           // 7
        5 => E7_COXETER,        // 18
        _ => return value,      // No correction
    };

    value * (1.0 + Q_DEFICIT / (divisor as f64))
}

/// Apply D₄ collapse threshold correction
#[pyfunction]
#[pyo3(name = "hierarchy_apply_collapse_threshold_correction")]
pub fn apply_collapse_threshold_correction(value: f64) -> f64 {
    value * (1.0 + Q_DEFICIT / (D4_KISSING as f64))
}

/// Apply Coxeter-Kissing product correction (720)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_coxeter_kissing_correction")]
pub fn apply_coxeter_kissing_correction(value: f64) -> f64 {
    value * (1.0 + Q_DEFICIT / (COXETER_KISSING_720 as f64))
}

/// Apply correction with E₆ structure
///
/// The exceptional Lie algebra E₆ underlies the Standard Model gauge structure.
/// Observables coupling to different E₆ structures receive corresponding corrections.
///
/// Args:
///   value: Float64 value to correct
///   structure_index: Which E₆ structure to use:
///     0 = dim(E₆) = 78 (full E₆ gauge structure)
///     1 = |Φ(E₆)| = 72 (full E₆ root system)
///     2 = |Φ⁺(E₆)| = 36 (positive roots / Golden Cone)
///     3 = dim(E₆ fund) = 27 (fundamental representation)
///     4 = rank(E₆) = 6 (Cartan subalgebra)
///     5 = h(E₆) = 12 (Coxeter number)
///
/// Returns:
///   Corrected value: value × (1 + q/divisor)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_e6_correction")]
pub fn apply_e6_correction(value: f64, structure_index: i32) -> f64 {
    let divisor = match structure_index {
        0 => E6_DIM,            // 78
        1 => E6_ROOTS,          // 72
        2 => E6_POSITIVE_ROOTS, // 36 (Golden Cone)
        3 => E6_FUNDAMENTAL,    // 27
        4 => E6_RANK,           // 6
        5 => E6_COXETER,        // 12
        _ => return value,      // No correction
    };

    value * (1.0 + Q_DEFICIT / (divisor as f64))
}

/// Apply Mersenne prime correction (q/M_p)
///
/// The Mersenne primes M_p = 2^p - 1 constrain which matter generations can exist.
/// Only p ∈ {2, 3, 5, 7} produce primes, giving exactly 3 generations plus a heavy anchor.
///
/// Args:
///   value: Float64 value to correct
///   mersenne_index: The exponent p in M_p = 2^p - 1:
///     2 → M₂ = 3 (generation count, = N_gen)
///     3 → M₃ = 7 (first non-trivial, = L₄)
///     5 → M₅ = 31 (third generation stability)
///     7 → M₇ = 127 (heavy anchor, dark matter seed)
///   sign: +1 or -1 for correction direction
///
/// Returns:
///   Corrected value: value × (1 ± q/M_p)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_mersenne_correction")]
pub fn apply_mersenne_correction(value: f64, mersenne_index: i32, sign: i32) -> f64 {
    let divisor = match mersenne_index {
        2 => MERSENNE_M2,  // 3
        3 => MERSENNE_M3,  // 7
        5 => MERSENNE_M5,  // 31
        7 => MERSENNE_M7,  // 127
        _ => return value, // No correction for invalid index
    };
    value * (1.0 + sign as f64 * Q_DEFICIT / (divisor as f64))
}

/// Apply Lucas number correction (q/L_n)
///
/// The Lucas sequence L_n = φⁿ + φ̂ⁿ provides shadow phase corrections.
/// Lucas primes at specific indices govern dark sector physics.
///
/// Args:
///   value: Float64 value to correct
///   lucas_index: The index n in L_n:
///     4 → L₄ = 7 (= M₃, Mersenne-Lucas resonance)
///     5 → L₅ = 11 (M₁₁ barrier connection)
///     6 → L₆ = 18 (= E₇ Coxeter number)
///     7 → L₇ = 29 (Lucas prime gate)
///     11 → L₁₁ = 199 (deep shadow structure)
///   sign: +1 or -1 for correction direction
///
/// Returns:
///   Corrected value: value × (1 ± q/L_n)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_lucas_correction")]
pub fn apply_lucas_correction(value: f64, lucas_index: i32, sign: i32) -> f64 {
    let divisor = match lucas_index {
        4 => LUCAS_L4,   // 7
        5 => LUCAS_L5,   // 11
        6 => LUCAS_L6,   // 18
        7 => LUCAS_L7,   // 29
        11 => LUCAS_L11, // 199
        _ => return value, // No correction for invalid index
    };
    value * (1.0 + sign as f64 * Q_DEFICIT / (divisor as f64))
}

/// Apply Fermat prime correction (q/F_n)
///
/// The Fermat primes F_n = 2^(2^n) + 1 determine the number of fundamental forces.
/// Only F₀ through F₄ are prime, giving exactly 5 forces.
///
/// Args:
///   value: Float64 value to correct
///   fermat_index: The index n in F_n = 2^(2^n) + 1:
///     0 → F₀ = 3 (Strong force SU(3))
///     1 → F₁ = 5 (Weak isospin SU(2))
///     2 → F₂ = 17 (Hypercharge U(1))
///     3 → F₃ = 257 (Gravity seed)
///     4 → F₄ = 65537 (Recursion completion)
///   sign: +1 or -1 for correction direction
///
/// Returns:
///   Corrected value: value × (1 ± q/F_n)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_fermat_correction")]
pub fn apply_fermat_correction(value: f64, fermat_index: i32, sign: i32) -> f64 {
    let divisor = match fermat_index {
        0 => FERMAT_F0,  // 3
        1 => FERMAT_F1,  // 5
        2 => FERMAT_F2,  // 17
        3 => FERMAT_F3,  // 257
        4 => FERMAT_F4,  // 65537
        _ => return value, // No correction for invalid index
    };
    value * (1.0 + sign as f64 * Q_DEFICIT / (divisor as f64))
}

/// Apply loop integral correction (q/nπ)
///
/// Loop integral factors arise from n-flavor QCD corrections.
/// The divisor nπ comes from the one-loop beta function contribution.
///
/// Common values:
///   n=3: 3-flavor QCD (below charm threshold)
///   n=4: One-loop radiative (standard 4D loop)
///   n=5: 5-flavor QCD (at M_Z scale)
///   n=6: 6-flavor QCD (above top threshold)
///
/// Args:
///   value: Float64 value to correct
///   n_flavors: Number of active flavors (or loop factor)
///   sign: +1 or -1 for correction direction
///
/// Returns:
///   Corrected value: value × (1 ± q/(nπ))
#[pyfunction]
#[pyo3(name = "hierarchy_apply_loop_integral")]
pub fn apply_loop_integral(value: f64, n_flavors: i32, sign: i32) -> f64 {
    let pi = std::f64::consts::PI;
    let divisor = (n_flavors as f64) * pi;
    value * (1.0 + sign as f64 * Q_DEFICIT / divisor)
}

/// Apply correction with arbitrary φ power divisor
///
/// Higher powers of φ appear for deeper recursion layers.
/// φⁿ corrections apply when an observable crosses n recursion boundaries.
///
/// Args:
///   value: Float64 value to correct
///   power: The power of φ in the divisor (0-5 optimized, higher computed)
///   sign: +1 or -1 for correction direction
///
/// Returns:
///   Corrected value: value × (1 ± q/φⁿ)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_phi_power_correction")]
pub fn apply_phi_power_correction(value: f64, power: i32, sign: i32) -> f64 {
    let phi_power = match power {
        0 => 1.0,
        1 => PHI,
        2 => PHI_SQUARED,
        3 => PHI_CUBED,
        4 => PHI_FOURTH,
        5 => PHI_FIFTH,
        _ => PHI.powi(power),
    };
    value * (1.0 + sign as f64 * Q_DEFICIT / phi_power)
}

// ============================================================================
// Extended Hierarchy Constant Access Functions
// ============================================================================

/// Get E₈ dimension (248)
#[pyfunction]
#[pyo3(name = "hierarchy_e8_dim")]
pub fn e8_dim() -> i32 {
    E8_DIM
}

/// Get E₇ dimension (133)
#[pyfunction]
#[pyo3(name = "hierarchy_e7_dim")]
pub fn e7_dim() -> i32 {
    E7_DIM
}

/// Get E₆ dimension (78)
#[pyfunction]
#[pyo3(name = "hierarchy_e6_dim")]
pub fn e6_dim() -> i32 {
    E6_DIM
}

/// Get D₄ dimension (28)
#[pyfunction]
#[pyo3(name = "hierarchy_d4_dim")]
pub fn d4_dim() -> i32 {
    D4_DIM
}

/// Get D₄ kissing number (24) - consciousness threshold
#[pyfunction]
#[pyo3(name = "hierarchy_d4_kissing")]
pub fn d4_kissing() -> i32 {
    D4_KISSING
}

/// Get G₂ dimension (14)
#[pyfunction]
#[pyo3(name = "hierarchy_g2_dim")]
pub fn g2_dim() -> i32 {
    G2_DIM
}

/// Get F₄ dimension (52)
#[pyfunction]
#[pyo3(name = "hierarchy_f4_dim")]
pub fn f4_dim() -> i32 {
    F4_DIM
}

/// Get Coxeter-Kissing product (720)
#[pyfunction]
#[pyo3(name = "hierarchy_coxeter_kissing_720")]
pub fn coxeter_kissing_720() -> i32 {
    COXETER_KISSING_720
}

/// Get hierarchy exponent (719)
#[pyfunction]
#[pyo3(name = "hierarchy_exponent")]
pub fn hierarchy_exponent() -> i32 {
    HIERARCHY_EXPONENT
}

// ============================================================================
// Extended Structure Constants (Unexposed in Python)
// ============================================================================

/// Get E₈ root count (240)
///
/// Returns the total number of roots in the E₈ exceptional Lie group.
/// E₈ has 240 roots, representing the fundamental geometric structure
/// underlying the Standard Model unification in SRT theory.
///
/// Returns
/// -------
/// int
///     The number of roots in E₈ (240)
///
/// Examples
/// --------
/// >>> from syntonic.crt import hierarchy_e8_roots
/// >>> hierarchy_e8_roots()
/// 240
#[pyfunction]
#[pyo3(name = "hierarchy_e8_roots")]
pub fn e8_roots() -> i32 {
    E8_ROOTS
}

/// Get E₈ positive root count (120)
///
/// Returns the number of positive roots in the E₈ root system.
/// The 120 positive roots correspond to half of the full root system,
/// representing the "positive" directions in the 8-dimensional Cartan subalgebra.
///
/// Returns
/// -------
/// int
///     The number of positive roots in E₈ (120)
///
/// Notes
/// -----
/// Used in particle physics for counting gauge symmetry breaking patterns
/// and in neural networks for optimal embedding dimensions.
#[pyfunction]
#[pyo3(name = "hierarchy_e8_positive_roots")]
pub fn e8_positive_roots() -> i32 {
    E8_POSITIVE_ROOTS
}

/// Get E₈ rank (8)
///
/// Returns the rank (dimension of Cartan subalgebra) of the E₈ Lie group.
/// The rank represents the number of independent Casimir operators and
/// corresponds to the dimension of the maximal torus.
///
/// Returns
/// -------
/// int
///     The rank of E₈ (8)
///
/// Physical Significance
/// --------------------
/// - 8 spacetime dimensions in string theory compactifications
/// - 8 gluons in QCD (before confinement)
/// - Neural network attention head constraints
#[pyfunction]
#[pyo3(name = "hierarchy_e8_rank")]
pub fn e8_rank() -> i32 {
    E8_RANK
}

/// Get E₈ Coxeter number (30)
///
/// Returns the Coxeter number of E₈, which governs the periodicity
/// of the Weyl group and appears in level-rank duality relations.
/// The Coxeter number h = 30 for E₈.
///
/// Returns
/// -------
/// int
///     The Coxeter number of E₈ (30)
///
/// Applications
/// ------------
/// - Period of recursion cycles in SRT theory
/// - Kac-Moody algebra level-rank dualities
/// - Golden ratio recursion bounds
#[pyfunction]
#[pyo3(name = "hierarchy_e8_coxeter")]
pub fn e8_coxeter() -> i32 {
    E8_COXETER
}

/// Get E₇ root count (126)
///
/// Returns the total number of roots in the E₇ exceptional Lie group.
/// E₇ represents the intermediate unification scale between E₆ and E₈
/// in the SRT Grand Unification hierarchy.
///
/// Returns
/// -------
/// int
///     The number of roots in E₇ (126)
///
/// Theoretical Context
/// ------------------
/// E₇ appears in heterotic string theory compactifications and
/// plays a role in the intermediate mass scale predictions of SRT.
#[pyfunction]
#[pyo3(name = "hierarchy_e7_roots")]
pub fn e7_roots() -> i32 {
    E7_ROOTS
}

/// Get E₇ positive roots (63)
///
/// Returns the number of positive roots in the E₇ root system.
/// The positive roots span the Weyl chamber and determine the
/// representation theory and branching rules.
///
/// Returns
/// -------
/// int
///     The number of positive roots in E₇ (63)
///
/// Notes
/// -----
/// Half of the total 126 roots, representing the fundamental
/// geometric structure of 7-dimensional exceptional geometry.
#[pyfunction]
#[pyo3(name = "hierarchy_e7_positive_roots")]
pub fn e7_positive_roots() -> i32 {
    E7_POSITIVE_ROOTS
}

/// Get E₇ fundamental representation (56)
///
/// Returns the dimension of the fundamental representation of E₇.
/// This 56-dimensional representation is fundamental to the
/// representation theory and appears in particle physics contexts.
///
/// Returns
/// -------
/// int
///     The dimension of the E₇ fundamental representation (56)
///
/// Physical Significance
/// --------------------
/// - 56 goldstino degrees of freedom in supersymmetry
/// - 56 real components of the E₇/SO(8) coset space
/// - Jordan algebra dimensions in exceptional geometry
#[pyfunction]
#[pyo3(name = "hierarchy_e7_fundamental")]
pub fn e7_fundamental() -> i32 {
    E7_FUNDAMENTAL
}

/// Get E₇ rank (7)
///
/// Returns the rank (dimension of Cartan subalgebra) of the E₇ Lie group.
/// The rank corresponds to the number of independent quantum numbers
/// needed to label representations.
///
/// Returns
/// -------
/// int
///     The rank of E₇ (7)
///
/// Applications
/// ------------
/// - 7-brane configurations in string theory
/// - 7-dimensional compactifications
/// - Neural network layer depth constraints
#[pyfunction]
#[pyo3(name = "hierarchy_e7_rank")]
pub fn e7_rank() -> i32 {
    E7_RANK
}

/// Get E₇ Coxeter number (18)
///
/// Returns the Coxeter number of E₇, governing the Weyl group
/// periodicity and appearing in affine algebra constructions.
/// The Coxeter number h = 18 for E₇.
///
/// Returns
/// -------
/// int
///     The Coxeter number of E₇ (18)
///
/// Theoretical Uses
/// ----------------
/// - Recursion cycle periods in SRT theory
/// - Modular invariance in conformal field theory
/// - Golden ratio convergence bounds
#[pyfunction]
#[pyo3(name = "hierarchy_e7_coxeter")]
pub fn e7_coxeter() -> i32 {
    E7_COXETER
}

/// Get E₆ root count (72)
///
/// Returns the total number of roots in the E₆ exceptional Lie group.
/// E₆ is the first exceptional group in the SRT unification chain
/// and corresponds to the GUT scale in particle physics.
///
/// Returns
/// -------
/// int
///     The number of roots in E₆ (72)
///
/// Theoretical Context
/// ------------------
/// E₆ appears in Calabi-Yau compactifications and heterotic string
/// phenomenology, representing the unification of electroweak and
/// strong forces with an additional U(1) gauge group.
#[pyfunction]
#[pyo3(name = "hierarchy_e6_roots")]
pub fn e6_roots() -> i32 {
    E6_ROOTS
}

/// Get E₆ positive roots / Golden Cone (36)
///
/// Returns the number of positive roots in E₆, which equals the
/// cardinality of the Golden Cone Φ⁺(E₆). This fundamental constant
/// appears throughout SRT theory as the geometric measure of
/// transcendence and consciousness emergence.
///
/// Returns
/// -------
/// int
///     The number of positive roots in E₆ (36) - Golden Cone cardinality
///
/// Physical Significance
/// --------------------
/// - **Golden Cone Cardinality**: |Φ⁺(E₆)| = 36
/// - **Consciousness Emergence**: Critical threshold for self-reference
/// - **Neural Architecture**: Optimal layer sizes in SRT networks
/// - **Transcendence Gates**: Number of ontological phase transitions
///
/// Examples
/// --------
/// >>> from syntonic.crt import hierarchy_e6_positive_roots
/// >>> hierarchy_e6_positive_roots()  # Golden Cone size
/// 36
#[pyfunction]
#[pyo3(name = "hierarchy_e6_positive_roots")]
pub fn e6_positive_roots() -> i32 {
    E6_POSITIVE_ROOTS
}

/// Get E₆ fundamental representation (27)
///
/// Returns the dimension of the fundamental representation of E₆.
/// The 27-dimensional representation is fundamental to E₆'s role
/// in particle physics and algebraic geometry.
///
/// Returns
/// -------
/// int
///     The dimension of the E₆ fundamental representation (27)
///
/// Mathematical Context
/// -------------------
/// - 27 lines on a cubic surface in algebraic geometry
/// - 27-dimensional Jordan algebra representation
/// - 27 generations in some GUT models (though not in SRT)
#[pyfunction]
#[pyo3(name = "hierarchy_e6_fundamental")]
pub fn e6_fundamental() -> i32 {
    E6_FUNDAMENTAL
}

/// Get E₆ rank (6)
///
/// Returns the rank (dimension of Cartan subalgebra) of the E₆ Lie group.
/// The rank corresponds to the number of independent gauge couplings
/// in the associated gauge theory.
///
/// Returns
/// -------
/// int
///     The rank of E₆ (6)
///
/// Physical Applications
/// --------------------
/// - 6-dimensional Calabi-Yau manifolds
/// - 6 extra dimensions in braneworld scenarios
/// - Neural network embedding dimensions
#[pyfunction]
#[pyo3(name = "hierarchy_e6_rank")]
pub fn e6_rank() -> i32 {
    E6_RANK
}

/// Get E₆ Coxeter number (12)
///
/// Returns the Coxeter number of E₆, governing the Weyl group
/// periodicity and affine extension properties.
/// The Coxeter number h = 12 for E₆.
///
/// Returns
/// -------
/// int
///     The Coxeter number of E₆ (12)
///
/// Theoretical Significance
/// -----------------------
/// - Period of Weyl group in E₆ affine algebra
/// - Golden ratio recursion cycles: φ¹² ≈ 161.8
/// - Modular forms of weight 12 in string theory
#[pyfunction]
#[pyo3(name = "hierarchy_e6_coxeter")]
pub fn e6_coxeter() -> i32 {
    E6_COXETER
}

/// Get D₄ rank (4)
///
/// Returns the rank (dimension of Cartan subalgebra) of the D₄ Lie group.
/// D₄ is isomorphic to SO(8) with triality and plays a central role
/// in SRT theory as the consciousness emergence group.
///
/// Returns
/// -------
/// int
///     The rank of D₄ (4)
///
/// Physical Significance
/// --------------------
/// - 4 spacetime dimensions in our observable universe
/// - 4 fundamental forces (gravity, electromagnetism, weak, strong)
/// - Neural network attention mechanisms
#[pyfunction]
#[pyo3(name = "hierarchy_d4_rank")]
pub fn d4_rank() -> i32 {
    D4_RANK
}

/// Get D₄ Coxeter number (6)
///
/// Returns the Coxeter number of D₄, governing the Weyl group periodicity.
/// The Coxeter number h = 6 for D₄, which appears prominently in
/// consciousness emergence calculations.
///
/// Returns
/// -------
/// int
///     The Coxeter number of D₄ (6)
///
/// SRT Applications
/// ----------------
/// - Consciousness threshold calculations
/// - D₄ kissing number (24) = 4 × 6 product
/// - Golden ratio recursion bounds
#[pyfunction]
#[pyo3(name = "hierarchy_d4_coxeter")]
pub fn d4_coxeter() -> i32 {
    D4_COXETER
}

/// Get G₂ rank (2)
///
/// Returns the rank (dimension of Cartan subalgebra) of the G₂ Lie group.
/// G₂ is the automorphism group of the octonions and represents the
/// most exceptional of the exceptional groups.
///
/// Returns
/// -------
/// int
///     The rank of G₂ (2)
///
/// Mathematical Context
/// -------------------
/// - Automorphisms of octonion algebra
/// - 2-dimensional representations in exceptional geometry
/// - Smallest exceptional Lie group
#[pyfunction]
#[pyo3(name = "hierarchy_g2_rank")]
pub fn g2_rank() -> i32 {
    G2_RANK
}

/// Get F₄ rank (4)
///
/// Returns the rank (dimension of Cartan subalgebra) of the F₄ Lie group.
/// F₄ is related to the Jordan algebra of 3×3 hermitian octonion matrices
/// and appears in the classification of exceptional geometries.
///
/// Returns
/// -------
/// int
///     The rank of F₄ (4)
///
/// Theoretical Role
/// ----------------
/// - Exceptional Jordan algebra structure group
/// - 4-dimensional parameter space for exceptional manifolds
/// - String theory compactification constraints
#[pyfunction]
#[pyo3(name = "hierarchy_f4_rank")]
pub fn f4_rank() -> i32 {
    F4_RANK
}

// ============================================================================
// Prime Number Tower Accessor Functions
// ============================================================================

/// Get Mersenne prime M₂ = 3 (generation count)
#[pyfunction]
#[pyo3(name = "hierarchy_mersenne_m2")]
pub fn mersenne_m2() -> i32 {
    MERSENNE_M2
}

/// Get Mersenne prime M₃ = 7 (first non-trivial)
#[pyfunction]
#[pyo3(name = "hierarchy_mersenne_m3")]
pub fn mersenne_m3() -> i32 {
    MERSENNE_M3
}

/// Get Mersenne prime M₅ = 31 (third generation)
#[pyfunction]
#[pyo3(name = "hierarchy_mersenne_m5")]
pub fn mersenne_m5() -> i32 {
    MERSENNE_M5
}

/// Get Mersenne prime M₇ = 127 (heavy anchor)
#[pyfunction]
#[pyo3(name = "hierarchy_mersenne_m7")]
pub fn mersenne_m7() -> i32 {
    MERSENNE_M7
}

/// Get Lucas number L₄ = 7 (Mersenne-Lucas resonance)
#[pyfunction]
#[pyo3(name = "hierarchy_lucas_l4")]
pub fn lucas_l4() -> i32 {
    LUCAS_L4
}

/// Get Lucas number L₅ = 11 (M₁₁ barrier connection)
#[pyfunction]
#[pyo3(name = "hierarchy_lucas_l5")]
pub fn lucas_l5() -> i32 {
    LUCAS_L5
}

/// Get Lucas number L₆ = 18 (= E₇ Coxeter)
#[pyfunction]
#[pyo3(name = "hierarchy_lucas_l6")]
pub fn lucas_l6() -> i32 {
    LUCAS_L6
}

/// Get Lucas number L₇ = 29 (Lucas prime gate)
#[pyfunction]
#[pyo3(name = "hierarchy_lucas_l7")]
pub fn lucas_l7() -> i32 {
    LUCAS_L7
}

/// Get Lucas number L₁₁ = 199 (deep shadow)
#[pyfunction]
#[pyo3(name = "hierarchy_lucas_l11")]
pub fn lucas_l11() -> i32 {
    LUCAS_L11
}

/// Get Fermat prime F₀ = 3 (Strong force SU(3))
#[pyfunction]
#[pyo3(name = "hierarchy_fermat_f0")]
pub fn fermat_f0() -> i32 {
    FERMAT_F0
}

/// Get Fermat prime F₁ = 5 (Weak isospin SU(2))
#[pyfunction]
#[pyo3(name = "hierarchy_fermat_f1")]
pub fn fermat_f1() -> i32 {
    FERMAT_F1
}

/// Get Fermat prime F₂ = 17 (Hypercharge U(1))
#[pyfunction]
#[pyo3(name = "hierarchy_fermat_f2")]
pub fn fermat_f2() -> i32 {
    FERMAT_F2
}

/// Get Fermat prime F₃ = 257 (Gravity seed)
#[pyfunction]
#[pyo3(name = "hierarchy_fermat_f3")]
pub fn fermat_f3() -> i32 {
    FERMAT_F3
}

/// Get Fermat prime F₄ = 65537 (Recursion completion)
#[pyfunction]
#[pyo3(name = "hierarchy_fermat_f4")]
pub fn fermat_f4() -> i32 {
    FERMAT_F4
}

/// Initialize geometric divisors in constant memory
///
/// Args:
///   divisors: List of 84 divisors matching hierarchy.py GEOMETRIC_DIVISORS
#[pyfunction]
#[pyo3(name = "hierarchy_init_divisors")]
pub fn init_divisors(_divisors: Vec<f64>) -> PyResult<()> {
    // Placeholder for CUDA constant memory initialization
    // CPU implementation doesn't need constant memory
    Ok(())
}
