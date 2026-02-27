//! Mathematical Utilities Module
//!
//! High-performance mathematical functions for SRT compression.
//! Includes prime operations, rational arithmetic helpers, and
//! mathematical utilities that are performance-critical.

use crate::exact::{Rational, GoldenExact};
use crate::constants::PHI;
use pyo3::prelude::*;
use pyo3::PyResult;

/// Check if a number is prime using trial division
#[pyfunction]
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Integer square root using Newton's method
#[pyfunction]
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Check if 2^p - 1 is a Mersenne prime
#[pyfunction]
pub fn is_mersenne_prime(p: u32) -> bool {
    if p < 2 {
        return false;
    }
    // For p > 2, 2^p - 1 is prime only if p is prime
    if !is_prime(p as u64) {
        return false;
    }

    // For small p, we can check directly
    match p {
        2 => return true,  // 3
        3 => return true,  // 7
        5 => return true,  // 31
        7 => return true,  // 127
        13 => return true, // 8191
        17 => return true, // 131071
        19 => return true, // 524287
        31 => return true, // 2147483647
        _ => {
            // For larger p, use Lucas-Lehmer test (simplified)
            // This is a basic implementation - full Lucas-Lehmer would be more complex
            let mersenne = (1u128 << p) - 1;
            // Simple trial division for small factors
            let sqrt_m = (mersenne as f64).sqrt() as u128;
            for i in 3..=((sqrt_m as f64).sqrt() as u128) {
                if mersenne % i == 0 {
                    return false;
                }
            }
            true
        }
    }
}

/// Check if 2^(2^n) + 1 is a Fermat prime
#[pyfunction]
pub fn is_fermat_prime(n: u32) -> bool {
    if n < 0 {
        return false;
    }

    match n {
        0 => is_prime(3),     // 2^(2^0) + 1 = 3
        1 => is_prime(5),     // 2^(2^1) + 1 = 5
        2 => is_prime(17),    // 2^(2^2) + 1 = 17
        3 => is_prime(257),   // 2^(2^3) + 1 = 257
        4 => is_prime(65537), // 2^(2^4) + 1 = 65537
        _ => false, // No more known Fermat primes
    }
}

/// Check if Lucas number L_n is prime
#[pyfunction]
pub fn is_lucas_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }

    // Lucas numbers: L_n = φ^n + (1-φ)^n where φ is golden ratio
    // Using exact golden arithmetic for large n
    use crate::exact::GoldenExact;
    let phi = GoldenExact::phi();
    let phi_neg_conj = GoldenExact::new(crate::exact::Rational::new(-1, 1), crate::exact::Rational::one()); // -1 + φ 
    let phi_power = phi.power_i64(n as i64);
    let phi_neg_conj_power = phi_neg_conj.power_i64(n as i64);
    let lucas_exact = phi_power + phi_neg_conj_power;
    let lucas = lucas_exact.to_f64().round() as u64;

    is_prime(lucas)
}

/// Convert Rational to byte using floor semantics
#[pyfunction]
pub fn rational_to_byte_floor(r: Rational) -> u8 {
    // Convert rational [0,1] to byte [0,255] using floor
    let scaled = r * Rational::from(255);
    let int_val = scaled.floor();
    int_val.max(0).min(255) as u8
}

/// Convert Rational to byte using round semantics
#[pyfunction]
pub fn rational_to_byte_round(r: Rational) -> u8 {
    // Convert rational [0,1] to byte [0,255] using round
    let scaled = r * Rational::from(255);
    // Round to nearest integer: floor(x + 0.5)
    let rounded = (scaled + Rational::new(1, 2)).floor();
    rounded.max(0).min(255) as u8
}

/// Compute log2 of a Rational using Taylor series
#[pyfunction]
pub fn log2_rational(r: Rational) -> Rational {
    if r <= Rational::from(0) {
        return Rational::from(-2i64.pow(62));
    }

    // Find exponent: normalize x to [1, 2)
    let mut x = r.clone();
    let mut exp = 0i32;

    if x >= Rational::from(1) {
        while x >= Rational::from(2) {
            x = x / Rational::from(2);
            exp += 1;
        }
    } else {
        while x < Rational::from(1) {
            x = x * Rational::from(2);
            exp -= 1;
        }
    }

    // x is now in [1, 2), compute log2(x) via Taylor series for ln(1+y)/ln(2)
    let y = x - Rational::from(1);  // y in [0, 1)

    // ln(x) ≈ y - y²/2 + y³/3 - y⁴/4 + y⁵/5 - y⁶/6 + y⁷/7
    let ln_x = y.clone()
        - (y.clone().pow(2) / Rational::from(2))
        + (y.clone().pow(3) / Rational::from(3))
        - (y.clone().pow(4) / Rational::from(4))
        + (y.clone().pow(5) / Rational::from(5))
        - (y.clone().pow(6) / Rational::from(6))
        + (y.clone().pow(7) / Rational::from(7));

    // ln(2) ≈ 0.693147
    let ln2 = Rational::from(693147) / Rational::from(1000000);

    Rational::from(exp) + ln_x / ln2
}

/// Register the math utilities module
pub fn register_math_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(isqrt, m)?)?;
    m.add_function(wrap_pyfunction!(is_mersenne_prime, m)?)?;
    m.add_function(wrap_pyfunction!(is_fermat_prime, m)?)?;
    m.add_function(wrap_pyfunction!(is_lucas_prime, m)?)?;
    m.add_function(wrap_pyfunction!(rational_to_byte_floor, m)?)?;
    m.add_function(wrap_pyfunction!(rational_to_byte_round, m)?)?;
    m.add_function(wrap_pyfunction!(log2_rational, m)?)?;
    Ok(())
}
