// =============================================================================
// Pisano Period Functions
// =============================================================================

use crate::tensor::srt_kernels::PHI;
use pyo3::prelude::*;

/// Calculates the Pisano Period π(n): the period of Fibonacci sequence mod n.
/// This determines the "Hook Length" for stable winding modes.
///
/// # Mathematical foundation
/// - π(p) is the minimal k such that F_k ≡ 0 (mod p) and F_{k+1} ≡ 1 (mod p)
/// - For prime p: π(p) divides p - 1 if p ≡ ±1 (mod 5), else divides 2(p + 1)
#[pyfunction]
pub fn pisano_period(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut a = 0u64;
    let mut b = 1u64;

    // The period is bounded by 6n (Wall's theorem)
    for i in 0..(6 * n) {
        let c = (a + b) % n;
        a = b;
        b = c;
        if a == 0 && b == 1 {
            return i + 1;
        }
    }
    0 // Should not happen for valid n
}

/// Batch compute Pisano periods for multiple values
#[pyfunction]
pub fn pisano_periods_batch(values: Vec<u64>) -> Vec<u64> {
    values.iter().map(|&n| pisano_period(n)).collect()
}

/// Calculates the "Versal Grip" strength.
/// A mode is "Self-Hooking" if its Pisano period is a multiple of its index.
/// Returns: (period / n) if self-hooking, 0.0 otherwise
#[pyfunction]
pub fn versal_grip_strength(p: u64) -> f64 {
    let pi = pisano_period(p);
    if pi > 0 && pi % p == 0 {
        (pi as f64) / (p as f64)
    } else {
        0.0
    }
}

// =============================================================================
// Fibonacci Prime / Transcendence Gate Functions
// =============================================================================

/// Fibonacci Prime indices - where F_n is prime
pub const FIBONACCI_PRIME_INDICES: [u64; 11] = [3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47];

/// Corresponding Fibonacci Primes
pub const FIBONACCI_PRIMES: [u64; 11] = [
    2, 3, 5, 13, 89, 233, 1597, 28657, 514229, 433494437, 2971215073,
];

/// Check if index n corresponds to a "Transcendence Gate"
/// These are Fibonacci Prime indices that mark ontological phase transitions
#[pyfunction]
pub fn is_transcendence_gate(n: u64) -> bool {
    FIBONACCI_PRIME_INDICES.contains(&n)
}

/// Get the Fibonacci number at index n
#[pyfunction]
pub fn fibonacci_number(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 2..=n {
        let c = a.saturating_add(b);
        a = b;
        b = c;
    }
    b
}

/// Check if F_n is prime (for known indices)
#[pyfunction]
pub fn is_fibonacci_prime(n: u64) -> bool {
    // Check if n is in known Fibonacci Prime indices
    FIBONACCI_PRIME_INDICES.contains(&n)
}

/// Get resonance boost factor for dimension matching Fibonacci Prime index
/// Returns φ^n for resonant dimensions, 1.0 otherwise
/// Special case: n=4 (Material Anomaly) gets 0.9× destabilization
#[pyfunction]
pub fn fibonacci_resonance_boost(n: u64) -> f64 {
    if !is_transcendence_gate(n) {
        return 1.0;
    }

    let boost = PHI.powi(n as i32);
    if n == 4 {
        // Material Anomaly: composite index producing prime
        boost * 0.9
    } else {
        boost
    }
}

// =============================================================================
// Lucas Gap / Dark Energy Functions
// =============================================================================

/// Lucas Prime indices - where L_n is prime
pub const LUCAS_PRIME_INDICES: [u64; 13] = [0, 2, 4, 5, 7, 8, 11, 13, 16, 17, 19, 31, 37];

/// Check if index n is in a Lucas Gap (no Lucas prime at this index)
#[pyfunction]
pub fn is_lucas_gap(n: u64) -> bool {
    !LUCAS_PRIME_INDICES.contains(&n)
}

/// Compute "Gap Pressure" - Dark Energy contribution at index n
/// In gaps: delocalized shadow energy → repulsive pressure
/// At primes: crystallized → no pressure
#[pyfunction]
pub fn lucas_gap_pressure(n: u64) -> f64 {
    if !is_lucas_gap(n) {
        return 0.0;
    }

    // Find nearest Lucas prime index
    let nearest = LUCAS_PRIME_INDICES
        .iter()
        .min_by_key(|&&p| (p as i64 - n as i64).abs())
        .copied()
        .unwrap_or(0);

    let gap_distance = (n as i64 - nearest as i64).abs() as f64;
    let shadow_phase = (1.0 - PHI).powi(n as i32);

    gap_distance * shadow_phase.abs()
}

// =============================================================================
// Fibonacci Prime Access Functions
// =============================================================================

/// Get the list of known Fibonacci primes
///
/// Returns the complete array of Fibonacci primes, which are prime numbers
/// that appear as Fibonacci numbers F_n. These primes serve as "transcendence
/// gates" in SRT theory, marking critical points of ontological phase transitions.
///
/// Returns
/// -------
/// List[int]
///     Array of Fibonacci primes: [2, 3, 5, 13, 89, 233, 1597, 28657, 514229, 433494437, 2971215073]
///
/// Physical Significance
/// --------------------
/// - **Transcendence Gates**: F_n where F_n is prime marks ontological boundaries
/// - **Consciousness Emergence**: Prime Fibonacci indices correspond to self-reference thresholds
/// - **Neural Networks**: Prime dimensions for stable resonance patterns
///
/// Examples
/// --------
/// >>> from syntonic._core import get_fibonacci_primes
/// >>> primes = get_fibonacci_primes()
/// >>> primes[:3]  # First three Fibonacci primes
/// [2, 3, 5]
#[pyfunction]
pub fn get_fibonacci_primes() -> Vec<u64> {
    FIBONACCI_PRIMES.to_vec()
}

/// Get Fibonacci prime at index i (0-indexed)
///
/// Returns the i-th Fibonacci prime from the known sequence.
/// Useful for accessing specific transcendence gates by index.
///
/// Parameters
/// ----------
/// i : int
///     Zero-based index into the Fibonacci primes array (0-10)
///
/// Returns
/// -------
/// int or None
///     The i-th Fibonacci prime, or None if index is out of bounds
///
/// Examples
/// --------
/// >>> from syntonic._core import fibonacci_prime
/// >>> fibonacci_prime(0)  # First Fibonacci prime
/// 2
/// >>> fibonacci_prime(3)  # Fourth Fibonacci prime
/// 13
#[pyfunction]
pub fn fibonacci_prime(i: usize) -> Option<u64> {
    FIBONACCI_PRIMES.get(i).copied()
}

// =============================================================================
// Register all new functions with PyO3
// =============================================================================

pub fn register_extended_prime_selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pisano functions
    m.add_function(wrap_pyfunction!(pisano_period, m)?)?;
    m.add_function(wrap_pyfunction!(pisano_periods_batch, m)?)?;
    m.add_function(wrap_pyfunction!(versal_grip_strength, m)?)?;

    // Fibonacci/Transcendence functions
    m.add_function(wrap_pyfunction!(is_transcendence_gate, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_number, m)?)?;
    m.add_function(wrap_pyfunction!(is_fibonacci_prime, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_resonance_boost, m)?)?;

    // Lucas Gap functions
    m.add_function(wrap_pyfunction!(is_lucas_gap, m)?)?;
    m.add_function(wrap_pyfunction!(lucas_gap_pressure, m)?)?;

    // Fibonacci Prime access functions
    m.add_function(wrap_pyfunction!(get_fibonacci_primes, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_prime, m)?)?;

    Ok(())
}
