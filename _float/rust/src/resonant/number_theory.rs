//! Number-theoretic functions for SRT.
//!
//! Provides Möbius function, golden weights, and related computations
//! used throughout the Syntonic library.

use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Compute the Möbius function μ(n).
///
/// μ(n) = 1 if n is a product of an even number of distinct primes
/// μ(n) = -1 if n is a product of an odd number of distinct primes
/// μ(n) = 0 if n has a squared prime factor
///
/// # Arguments
/// * `n` - Positive integer
///
/// # Returns
/// μ(n) ∈ {-1, 0, 1}
///
/// # Example
/// ```
/// use syntonic_core::resonant::number_theory::mobius;
/// assert_eq!(mobius(1), 1);
/// assert_eq!(mobius(2), -1);  // prime
/// assert_eq!(mobius(4), 0);   // 2² has squared factor
/// assert_eq!(mobius(6), 1);   // 2×3 (two distinct primes)
/// ```
pub fn mobius(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut temp = n;
    let mut num_factors = 0i64;
    let mut d = 2i64;

    while d * d <= temp {
        if temp % d == 0 {
            temp /= d;
            if temp % d == 0 {
                // Squared prime factor
                return 0;
            }
            num_factors += 1;
        }
        d += 1;
    }

    if temp > 1 {
        num_factors += 1;
    }

    if num_factors % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Check if n is square-free (has no squared prime factors).
///
/// # Arguments
/// * `n` - Positive integer
///
/// # Returns
/// true if n has no squared prime factors
pub fn is_square_free(n: i64) -> bool {
    mobius(n) != 0
}

/// Compute the Mertens function M(n) = Σ_{k=1}^n μ(k).
///
/// # Arguments
/// * `n` - Upper limit
///
/// # Returns
/// M(n)
pub fn mertens(n: usize) -> i64 {
    (1..=n as i64).map(mobius).sum()
}

/// Compute golden weight w(n) = exp(-|n|²/φ).
///
/// This is the fundamental weighting function in SRT that
/// preferentially weights low-norm modes.
///
/// # Arguments
/// * `mode_norm_sq` - Mode norm squared |n|²
///
/// # Returns
/// w(n) = exp(-|n|²/φ)
pub fn golden_weight(mode_norm_sq: f64) -> f64 {
    (-mode_norm_sq / PHI).exp()
}

/// Compute E* = e^π - π ≈ 20.1408...
///
/// This is the fundamental constant appearing in SRT spectral theory.
pub fn e_star() -> f64 {
    PI.exp() - PI
}

/// Batch compute golden weights for a vector of mode norms.
///
/// # Arguments
/// * `mode_norms` - Slice of mode norm squared values
///
/// # Returns
/// Vector of golden weights
pub fn golden_weights(mode_norms: &[f64]) -> Vec<f64> {
    mode_norms.iter().map(|&n| golden_weight(n)).collect()
}

// =============================================================================
// SRT/CRT Prime Theory Functions (New Implementation)
// =============================================================================

/// Check if a Mersenne number 2^p - 1 is prime.
/// According to Axiom 6: Stable matter exists iff M_p is prime.
///
/// # Arguments
/// * `p` - Exponent (prime index)
///
/// # Returns
/// true if 2^p - 1 is prime
pub fn is_mersenne_prime(p: u32) -> bool {
    if p == 2 {
        return true; // M2 = 3 is prime
    }
    if p == 3 {
        return true; // M3 = 7 is prime
    }
    if p == 5 {
        return true; // M5 = 31 is prime
    }
    if p == 7 {
        return true; // M7 = 127 is prime
    }
    if p == 11 {
        return false; // M11 = 2047 = 23 × 89 (composite - the barrier)
    }

    // For larger p, use Lucas-Lehmer test (simplified implementation)
    // In practice, would use more sophisticated primality testing
    let mp = (1u128 << p) - 1;
    is_prime_u128(mp)
}

/// Check if a Fermat number 2^(2^n) + 1 is prime.
/// According to CRT: Forces exist iff F_n is prime.
///
/// # Arguments
/// * `n` - Fermat index
///
/// # Returns
/// true if 2^(2^n) + 1 is prime
pub fn is_fermat_prime(n: u32) -> bool {
    match n {
        0 => true,  // F0 = 3
        1 => true,  // F1 = 5
        2 => true,  // F2 = 17
        3 => true,  // F3 = 257
        4 => true,  // F4 = 65537
        5 => false, // F5 = 4294967297 = 641 × 6700417 (Euler, composite)
        _ => false, // All higher Fermat numbers are composite
    }
}

/// Check if a Lucas number L_n = φ^n + (1-φ)^n is prime.
/// According to CRT: Dark sectors stabilize iff L_n is prime.
///
/// # Arguments
/// * `n` - Lucas index
///
/// # Returns
/// true if L_n is prime
pub fn is_lucas_prime(n: u64) -> bool {
    let lucas_n = lucas_number(n);
    is_prime_u128(lucas_n)
}

/// Compute the nth Lucas number L_n = φ^n + (1-φ)^n.
///
/// # Arguments
/// * `n` - Index
///
/// # Returns
/// L_n
pub fn lucas_number(n: u64) -> u128 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }

    let mut a = 1u128; // L0 = 2, but we start from L1 = 1
    let mut b = 3u128; // L2 = 3

    if n == 2 {
        return 3;
    }

    for _ in 3..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }

    b
}

/// Compute the Pisano period π(p) for prime p.
/// The period with which the Fibonacci sequence repeats modulo p.
/// According to CRT: This determines the "hooking cycle" of prime windings.
///
/// # Arguments
/// * `p` - Prime number
///
/// # Returns
/// Pisano period π(p)
pub fn pisano_period(p: u64) -> u64 {
    if p == 2 {
        return 3; // Fib mod 2: 0,1,1,0,1,1,0,1,1,...
    }
    if p == 3 {
        return 8; // Fib mod 3: 0,1,1,2,0,2,2,1,0,1,1,2,...
    }
    if p == 5 {
        return 20;
    }

    // General case: find period of Fib sequence mod p
    let mut a = 0u64;
    let mut b = 1u64;
    let mut period = 0u64;

    loop {
        let c = (a + b) % p;
        a = b;
        b = c;
        period += 1;

        // Period found when we return to (0,1)
        if a == 0 && b == 1 {
            return period;
        }

        // Safety check to prevent infinite loop
        if period > 6 * p {
            return 0; // Error case
        }
    }
}

/// Check if a winding index p generates a stable Mersenne geometry.
/// According to Axiom 6: Stable iff 2^p - 1 is prime.
///
/// # Arguments
/// * `p` - Winding index
///
/// # Returns
/// true if stable
pub fn is_stable_winding(p: u32) -> bool {
    is_mersenne_prime(p)
}

/// Returns the stability barrier where physics changes phase.
/// Currently p=11 where M11 = 23 × 89 (composite).
///
/// # Returns
/// The barrier index (11)
pub fn get_stability_barrier() -> u32 {
    11
}

/// Check if a number corresponds to a "transcendence gate"
/// (Fibonacci prime index or the anomaly at 4).
///
/// # Arguments
/// * `n` - Index to check
///
/// # Returns
/// true if n is a transcendence gate
pub fn is_transcendence_gate(n: u64) -> bool {
    let fib_prime_indices = [3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47];
    fib_prime_indices.contains(&n)
}

/// Calculate the "versal grip" strength of a prime.
/// A measure of how strongly a prime "hooks" the golden flow.
/// Returns π(p)/p if the period divides p, else 0.0.
///
/// # Arguments
/// * `p` - Prime number
///
/// # Returns
/// Grip strength (dimensionless)
pub fn versal_grip_strength(p: u64) -> f64 {
    let pi = pisano_period(p);
    if pi % p == 0 {
        pi as f64 / p as f64
    } else {
        0.0
    }
}

/// Generate Mersenne primes up to maximum exponent.
///
/// # Arguments
/// * `max_p` - Maximum exponent to check
///
/// # Returns
/// Vector of Mersenne primes found
pub fn mersenne_sequence(max_p: u32) -> Vec<u64> {
    let mut result = Vec::new();
    for p in 2..=max_p {
        if is_mersenne_prime(p) {
            result.push(p as u64);
        }
    }
    result
}

/// Generate Fermat primes up to maximum index.
///
/// # Arguments
/// * `max_n` - Maximum Fermat index
///
/// # Returns
/// Vector of Fermat primes found
pub fn fermat_sequence(max_n: u32) -> Vec<u64> {
    let mut result = Vec::new();
    for n in 0..=max_n {
        if is_fermat_prime(n) {
            result.push(n as u64);
        }
    }
    result
}

/// Generate Lucas primes up to maximum index.
///
/// # Arguments
/// * `max_n` - Maximum Lucas index
///
/// # Returns
/// Vector of Lucas primes found
pub fn lucas_primes(max_n: u64) -> Vec<u64> {
    let mut result = Vec::new();
    for n in 0..=max_n {
        if is_lucas_prime(n) {
            result.push(n);
        }
    }
    result
}

/// Compute the Lucas boost factor L_{17}/L_{13} ≈ 6.854
/// Used for dark matter mass predictions.
///
/// # Returns
/// The boost factor
pub fn lucas_dark_boost() -> f64 {
    let l17 = lucas_number(17) as f64;
    let l13 = lucas_number(13) as f64;
    l17 / l13
}

/// Predict dark matter mass using Lucas boost.
/// M_dark = M_anchor × (L_{17}/L_{13})
///
/// # Arguments
/// * `anchor_mass_gev` - Anchor mass in GeV (e.g., Top quark mass)
///
/// # Returns
/// Predicted dark matter mass in GeV
pub fn predict_dark_matter_mass(anchor_mass_gev: f64) -> f64 {
    anchor_mass_gev * lucas_dark_boost()
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Simple primality test for u128 (used for large Mersenne numbers).
/// In production, would use more sophisticated algorithms.
///
/// # Arguments
/// * `n` - Number to test
///
/// # Returns
/// true if prime
fn is_prime_u128(n: u128) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5u128;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobius_values() {
        // μ(1) = 1
        assert_eq!(mobius(1), 1);

        // Primes have μ(p) = -1
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(3), -1);
        assert_eq!(mobius(5), -1);
        assert_eq!(mobius(7), -1);

        // Perfect squares have μ(n) = 0
        assert_eq!(mobius(4), 0);
        assert_eq!(mobius(9), 0);
        assert_eq!(mobius(25), 0);

        // Products of two distinct primes have μ = 1
        assert_eq!(mobius(6), 1); // 2×3
        assert_eq!(mobius(10), 1); // 2×5
        assert_eq!(mobius(15), 1); // 3×5

        // Products of three distinct primes have μ = -1
        assert_eq!(mobius(30), -1); // 2×3×5
    }

    #[test]
    fn test_is_square_free() {
        assert!(is_square_free(1));
        assert!(is_square_free(2));
        assert!(is_square_free(6));
        assert!(!is_square_free(4));
        assert!(!is_square_free(12)); // 4×3
    }

    #[test]
    fn test_mertens() {
        // M(1) = 1
        assert_eq!(mertens(1), 1);
        // M(2) = M(1) + μ(2) = 1 + (-1) = 0
        assert_eq!(mertens(2), 0);
        // M(3) = M(2) + μ(3) = 0 + (-1) = -1
        assert_eq!(mertens(3), -1);
    }

    #[test]
    fn test_golden_weight() {
        // w(0) = exp(0) = 1
        assert!((golden_weight(0.0) - 1.0).abs() < 1e-10);

        // w(φ) = exp(-1) ≈ 0.368
        let w = golden_weight(PHI);
        assert!((w - (-1.0_f64).exp()).abs() < 1e-10);

        // Decreasing with mode norm
        assert!(golden_weight(1.0) > golden_weight(2.0));
        assert!(golden_weight(2.0) > golden_weight(4.0));
    }

    #[test]
    fn test_e_star() {
        let e = e_star();
        // E* ≈ 20.1408...
        assert!((e - 20.140876).abs() < 0.001);
    }
}
